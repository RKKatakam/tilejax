"""What `@` means in kernel code: a matmul on the MXU, accumulating in f32 (i32 for integers).

Plain JAX rounds `bf16 @ bf16` to bf16; the MXU accumulates in f32 and so do kernels. The kernel, the
type analysis and the reference all compute `@` with these functions, so they agree with each other. The
generated source carries their code (`inspect.getsource`), so it runs without autokernel.
"""

from __future__ import annotations

import ast

import jax
import jax.numpy as jnp

__all__ = ["DOT", "DOT_NT", "FUNCTIONS", "rewrite_matmuls"]

DOT = "_mxu_dot"  # the names kernel code calls them by
DOT_NT = "_mxu_dot_nt"


def _mxu_operands(x, y):
    # Mixed operand types are promoted first, as `@` does (Mosaic has no mixed-type matmul). The result
    # accumulates in f32 (i32 for integers); f32 operands get full precision, not Mosaic's one bf16 pass.
    if x.dtype != y.dtype:
        common = jnp.promote_types(x.dtype, y.dtype)
        x, y = x.astype(common), y.astype(common)
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    precision = jax.lax.Precision.HIGHEST if x.dtype == jnp.float32 else None
    return x, y, acc, precision


def _mxu_dot(x, y):
    # `x @ y` on the MXU.
    x, y, acc, precision = _mxu_operands(x, y)
    return jnp.matmul(x, y, preferred_element_type=acc, precision=precision)


def _mxu_dot_nt(x, y):
    # `x @ y.T`: for two matrices, contract both last dims, so the MXU reads y transposed in place.
    if x.ndim != 2 or y.ndim != 2:
        return _mxu_dot(x, y.T)
    x, y, acc, precision = _mxu_operands(x, y)
    return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=acc, precision=precision)


# The functions by the names kernel code calls them by, in the order the generated source defines them.
FUNCTIONS = {"_mxu_operands": _mxu_operands, DOT: _mxu_dot, DOT_NT: _mxu_dot_nt}


class _MatmulRewriter(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.op, ast.MatMult):
            return node
        right = node.right
        if isinstance(right, ast.Attribute) and right.attr == "T":
            call = _call(DOT_NT, node.left, right.value)
        else:
            call = _call(DOT, node.left, right)
        return ast.copy_location(call, node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        self.generic_visit(node)
        if not isinstance(node.op, ast.MatMult):
            return node
        assert isinstance(node.target, ast.Name)  # parse.py only allows augmenting names in kernel code
        target = ast.Name(node.target.id, ast.Store())
        product = self.visit_BinOp(ast.BinOp(ast.Name(node.target.id, ast.Load()), ast.MatMult(), node.value))
        return ast.copy_location(ast.Assign([target], product), node)


def _call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(ast.Name(name, ast.Load()), list(args), [])


def rewrite_matmuls[T: ast.AST](tree: T) -> T:
    """`tree` with `a @ b` a call of `_mxu_dot`, `a @ b.T` of `_mxu_dot_nt`, and `a @= b` `a = _mxu_dot(a, b)`.

    Rewrites in place and returns the tree."""
    tree = _MatmulRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return tree
