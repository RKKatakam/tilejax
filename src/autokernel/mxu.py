"""How `@` is computed: an MXU matmul accumulating in f32 (i32 for integer inputs).

Plain JAX rounds `bf16 @ bf16` to bf16 and wraps `int8 @ int8` in int8; the kernel keeps the MXU's
wide accumulator instead. The same helpers serve three places so they can't disagree: the generated
kernel (as source text in its prelude, so the kernel stays self-contained), the trace that finds the
dtypes of state variables, and `TpuKernel.reference`, so `verify` compares against Python code with the
kernel's accumulation semantics.
"""

from __future__ import annotations

import ast
import copy
import functools
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["HELPERS", "MATMUL_PRECISIONS", "helpers", "prelude", "rewrite_matmuls"]

# Mosaic's default f32 matmul is a single bf16 pass; HIGHEST does the multi-pass f32 matmul the Python
# code actually describes.
MATMUL_PRECISIONS = {"highest": "jax.lax.Precision.HIGHEST", "default": "None"}

HELPERS = ("_mxu_params", "_mxu_dot", "_mxu_dot_t")

_PRELUDE = """\
def _mxu_params(x, y):
    # `@` accumulates in f32 (i32 for integer inputs). bf16/int8 operands are exact on the MXU in one pass;
    # only f32 operands take a precision.
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    precision = {precision} if x.dtype == jnp.float32 or y.dtype == jnp.float32 else None
    return acc, precision


def _mxu_dot(x, y):
    # `x @ y`: leading dims are batch dims.
    acc, precision = _mxu_params(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return jnp.dot(x, y, preferred_element_type=acc, precision=precision)
    batch = tuple(range(x.ndim - 2))
    dims = (((x.ndim - 1,), (y.ndim - 2,)), (batch, batch))
    return jax.lax.dot_general(x, y, dims, preferred_element_type=acc, precision=precision)


def _mxu_dot_t(x, y):
    # `x @ y.T`: contract the last dims of both, so the MXU reads y transposed instead of us transposing it.
    if y.ndim != 2:
        return _mxu_dot(x, y.T)
    acc, precision = _mxu_params(x, y)
    return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=acc, precision=precision)
"""


def prelude(matmul_precision: str) -> str:
    """The helpers as source, for the generated kernel."""
    return _PRELUDE.format(precision=MATMUL_PRECISIONS[matmul_precision])


@functools.cache
def helpers(matmul_precision: str) -> dict[str, Any]:
    """The helpers as callables, for tracing user code and for the reference."""
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(prelude(matmul_precision), namespace)
    return {name: namespace[name] for name in HELPERS}


class _Matmuls(ast.NodeTransformer):
    def visit_BinOp(self, n: ast.BinOp) -> ast.AST:
        self.generic_visit(n)
        if not isinstance(n.op, ast.MatMult):
            return n
        right, fn = n.right, "_mxu_dot"
        if isinstance(right, ast.Attribute) and right.attr == "T":  # x @ y.T
            right, fn = right.value, "_mxu_dot_t"
        return ast.copy_location(ast.Call(ast.Name(fn, ast.Load()), [n.left, right], []), n)


def rewrite_matmuls(node: ast.AST) -> ast.AST:
    """A copy of `node` with `a @ b` as `_mxu_dot(a, b)` and `a @ b.T` as `_mxu_dot_t(a, b)`."""
    return ast.fix_missing_locations(_Matmuls().visit(copy.deepcopy(node)))
