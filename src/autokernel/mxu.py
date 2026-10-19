"""How `@` is computed: an MXU matmul accumulating in f32 (i32 for integer inputs).

Plain JAX rounds `bf16 @ bf16` to bf16 and wraps `int8 @ int8` in int8; the kernel keeps the MXU's
wide accumulator instead. `rewrite_matmuls` turns `a @ b` into a call of a helper, and there are two sets
of helpers:

- the kernel's (`prelude`, `helpers`): the shapes Mosaic lowers well (a stack of matrices times a matrix as
  one 2-D matmul, `x @ y.T` contracting both last dims so the MXU reads y transposed). The generated kernel
  carries them as source; the trace that finds the dtypes of state variables and the cost model run the
  same code;
- the reference's (`reference_helpers`): `jnp.matmul` with nothing but the accumulator type, NumPy's rules
  for every rank. `TpuKernel.reference` and `verify` use these, so a mistake in the kernel's helpers shows
  up as a mismatch instead of being shared by both sides.
"""

from __future__ import annotations

import ast
import copy
import functools
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["CALLS", "HELPERS", "MATMUL_PRECISIONS", "helpers", "prelude", "reference_helpers", "rewrite_matmuls"]

# Mosaic's default f32 matmul is a single bf16 pass; HIGHEST does the multi-pass f32 matmul the Python
# code actually describes.
MATMUL_PRECISIONS = {"highest": "jax.lax.Precision.HIGHEST", "default": "None"}

# The kernel's helpers, by the names the prelude gives them unless the caller picks others.
HELPERS = ("_mxu_params", "_mxu_dot", "_mxu_dot_t")
# What `rewrite_matmuls` calls for `a @ b` and `a @ b.T`. The prefix is reserved (as in `reference`), so a
# call can't pick up a user function; generated kernels rename the calls to their helpers' names.
CALLS = {"_mxu_dot": "_autokernel_mxu_dot", "_mxu_dot_t": "_autokernel_mxu_dot_t"}

_PRELUDE = """\
def {_mxu_params}(x, y):
    # `@` accumulates in f32 (i32 for integer inputs). Operands of different types are promoted to a common
    # one first, as NumPy does (Mosaic has no mixed-type matmul). bf16/int8 operands are exact on the MXU in
    # one pass; only f32 operands take a precision.
    if x.dtype != y.dtype:
        common = jnp.promote_types(x.dtype, y.dtype)
        x, y = x.astype(common), y.astype(common)
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    precision = {precision} if x.dtype == jnp.float32 else None
    return x, y, acc, precision


def {_mxu_dot}(x, y):
    # `x @ y` with NumPy's rules: leading dims are batch dims (broadcast when they differ).
    x, y, acc, precision = {_mxu_params}(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return jnp.dot(x, y, preferred_element_type=acc, precision=precision)
    if x.ndim > 2 and y.ndim == 2:  # a stack of matrices times one matrix: one big matmul
        out = jnp.dot(x.reshape(-1, x.shape[-1]), y, preferred_element_type=acc, precision=precision)
        return out.reshape(*x.shape[:-1], y.shape[-1])
    if x.ndim == y.ndim > 2 and x.shape[:-2] == y.shape[:-2]:
        batch = tuple(range(x.ndim - 2))
        dims = (((x.ndim - 1,), (y.ndim - 2,)), (batch, batch))
        return jax.lax.dot_general(x, y, dims, preferred_element_type=acc, precision=precision)
    return jnp.matmul(x, y, preferred_element_type=acc, precision=precision)  # vectors, broadcast batches


def {_mxu_dot_t}(x, y):
    # `x @ y.T` for a matrix y: contract x's last dim with y's, so the MXU reads y transposed instead of us
    # transposing it (a stack of matrices x as one big matmul).
    if y.ndim != 2 or x.ndim < 2:
        return {_mxu_dot}(x, y.T)
    x, y, acc, precision = {_mxu_params}(x, y)
    dims = (((1,), (1,)), ((), ()))
    out = jax.lax.dot_general(x.reshape(-1, x.shape[-1]), y, dims, preferred_element_type=acc, precision=precision)
    return out if x.ndim == 2 else out.reshape(*x.shape[:-1], y.shape[0])
"""


def prelude(matmul_precision: str, names: dict[str, str] | None = None) -> str:
    """The kernel's helpers as source, named `names[h]` (by default as in `HELPERS`)."""
    names = {h: (names or {}).get(h, h) for h in HELPERS}
    return _PRELUDE.format(precision=MATMUL_PRECISIONS[matmul_precision], **names)


@functools.cache
def helpers(matmul_precision: str) -> dict[str, Any]:
    """The kernel's helpers as callables, under the names `rewrite_matmuls` calls: for tracing user code."""
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(prelude(matmul_precision), namespace)
    return {call: namespace[name] for name, call in CALLS.items()}


def _accumulator(x: Any, y: Any) -> Any:
    return jnp.int32 if jnp.issubdtype(jnp.result_type(x, y), jnp.integer) else jnp.float32


def _reference_dot(x: Any, y: Any) -> Any:
    return jnp.matmul(x, y, preferred_element_type=_accumulator(x, y))


def _reference_dot_t(x: Any, y: Any) -> Any:
    return _reference_dot(x, y.T)


def reference_helpers() -> dict[str, Any]:
    """The reference's helpers, under the names `rewrite_matmuls` calls: `jnp.matmul` accumulating in f32
    (i32), at the precision of the enclosing `jax.default_matmul_precision` (the reference runs under
    "highest" unless `matmul_precision="default"`)."""
    return {CALLS["_mxu_dot"]: _reference_dot, CALLS["_mxu_dot_t"]: _reference_dot_t}


class _Matmuls(ast.NodeTransformer):
    def visit_BinOp(self, n: ast.BinOp) -> ast.AST:
        self.generic_visit(n)
        if not isinstance(n.op, ast.MatMult):
            return n
        right, fn = n.right, CALLS["_mxu_dot"]
        if isinstance(right, ast.Attribute) and right.attr == "T":  # x @ y.T
            right, fn = right.value, CALLS["_mxu_dot_t"]
        return ast.copy_location(ast.Call(ast.Name(fn, ast.Load()), [n.left, right], []), n)


def rewrite_matmuls(node: ast.AST) -> ast.AST:
    """A copy of `node` with `a @ b` and `a @ b.T` as calls of the helpers (see `CALLS`)."""
    return ast.fix_missing_locations(_Matmuls().visit(copy.deepcopy(node)))
