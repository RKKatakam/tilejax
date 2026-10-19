"""`jax.vmap` of a kernel: the same loop nest with one more parallel loop, over the batch.

A Pallas kernel with hand-written DMAs has no batching rule JAX could derive, so each kernel is wrapped in
a `jax.custom_batching.custom_vmap` whose rule builds (once per batching pattern and size) a new kernel
from the loop nest's source: an outer `for b in range(B)` loop around the whole nest, every batched input
indexed `x[b, ...]`, every output stored `out.at[b, ...]` and allocated with a leading batch dim. That
is one gridless kernel over the whole batch, with the batch as its outermost tile loop, rather than B
kernel launches; unbatched inputs are read as they are by every batch element.

Inputs the transform can't batch fall back to running the kernel once per batch element in a
`jax.lax.map`: a batched runtime scalar (a 0-d integer input), or a batched input the prologue uses as
a value rather than for its shape.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Sequence
from typing import Any

import jax

from . import frontend

__all__ = ["ALLOC", "BatchingError", "batched_nest", "sequential"]

# The helper allocations of a batched nest call: `ALLOC(size, lambda x: <alloc>, x)` evaluates the original
# allocation per batch element (reading batched inputs `x` per element) and stacks the results.
ALLOC = "_autokernel_batched_alloc"

_SHAPE_ATTRS = frozenset({"shape", "ndim", "size"})


class BatchingError(Exception):
    """The nest can't be batched by rewriting it (the caller falls back to a `jax.lax.map`)."""


def batched_alloc(size: int, fn: Callable[..., Any], *batched: Any) -> Any:
    """The output allocation `fn(*per_element_inputs)` for every batch element, stacked."""
    return jax.vmap(fn, axis_size=size)(*batched)


def batched_nest(nest: frontend.LoopNest, in_batched: Sequence[bool], size: int) -> frontend.LoopNest:
    """`nest` mapped over a leading batch dim of size `size` of the inputs marked in `in_batched` (and of
    every output). Raises BatchingError when it can't be rewritten."""
    batched = {p for p, b in zip(nest.params, in_batched, strict=True) if b}
    fdef = copy.deepcopy(nest.fdef)
    taken = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)}
    taken |= {a.arg for a in ast.walk(fdef) if isinstance(a, ast.arg)}
    var = _fresh("_vmap_b", taken)
    out_vars = set(nest.out_vars)
    body, docstring = list(fdef.body), []
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        docstring, body = body[:1], body[1:]
    new_body: list[ast.stmt] = []
    for s in body:
        if isinstance(s, ast.Assign) and frontend.is_simple_assignment(s) and s.targets[0].id in out_vars:
            s.value = _batched_allocation(s.value, batched, size)
        elif isinstance(s, ast.For):
            s = _BatchIndices(batched, out_vars, var, size).visit(s)
            loop = ast.For(
                target=ast.Name(var, ast.Store()),
                iter=ast.Call(ast.Name("range", ast.Load()), [ast.Constant(size)], []),
                body=[s],
                orelse=[],
            )
            s = ast.copy_location(loop, s)
        elif not isinstance(s, ast.Return):
            s = _BatchIndices(batched, out_vars, var, size).visit(s)  # the prologue: shapes are per element
        new_body.append(s)
    fdef.body = docstring + new_body
    fdef.name = _fresh(f"{nest.fn_name}_vmap", taken)
    ast.fix_missing_locations(fdef)
    try:
        return frontend.parse_function(fdef, fdef.name, nest.first_line)
    except Exception as e:  # the rewrite broke one of the frontend's rules
        raise BatchingError(str(e)) from None


def sequential(fn: Callable[..., Any], in_batched: Sequence[bool], args: Sequence[Any]) -> Any:
    """`fn` mapped over the leading dim of the batched `args`, one call per element (`jax.lax.map`)."""
    batched_args = [a for a, b in zip(args, in_batched, strict=True) if b]

    def one(xs: Sequence[Any]) -> Any:
        it = iter(xs)
        return fn(*[next(it) if b else a for a, b in zip(args, in_batched, strict=True)])

    return jax.lax.map(one, batched_args)


def _fresh(base: str, taken: set[str]) -> str:
    name, n = base, 0
    while name in taken:
        n += 1
        name = f"{base}{n}"
    taken.add(name)
    return name


def _batched_allocation(alloc: ast.expr, batched: set[str], size: int) -> ast.expr:
    """`ALLOC(size, lambda <batched inputs it reads>: alloc, <those inputs>)`."""
    reads = sorted({n.id for n in ast.walk(alloc) if isinstance(n, ast.Name) and n.id in batched})
    args = ast.arguments(
        posonlyargs=[], args=[ast.arg(r) for r in reads], vararg=None, kwonlyargs=[], kw_defaults=[], defaults=[]
    )
    fn = ast.Lambda(args, alloc)
    call = ast.Call(
        ast.Name(ALLOC, ast.Load()), [ast.Constant(size), fn, *[ast.Name(r, ast.Load()) for r in reads]], []
    )
    return ast.copy_location(call, alloc)


class _BatchIndices(ast.NodeTransformer):
    """Index batched inputs and the outputs by the batch loop's variable; give batched inputs' shapes per
    batch element (`x.shape` -> `x.shape[1:]`)."""

    def __init__(self, batched: set[str], out_vars: set[str], var: str, size: int):
        self.batched, self.out_vars, self.var, self.size = batched, out_vars, var, size

    def _prepend(self, node: ast.Subscript) -> ast.Subscript:
        index = node.slice
        elts = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        node.slice = ast.Tuple([ast.Name(self.var, ast.Load()), *elts], ast.Load())
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        value = node.value
        if isinstance(value, ast.Name) and value.id in self.batched:
            node.slice = self.visit(node.slice)
            return self._prepend(node)
        is_store = (
            isinstance(value, ast.Attribute)
            and value.attr == "at"
            and isinstance(value.value, ast.Name)
            and value.value.id in self.out_vars
        )
        if is_store:
            node.slice = self.visit(node.slice)
            return self._prepend(node)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        value = node.value
        if not (isinstance(value, ast.Name) and value.id in self.batched):
            return self.generic_visit(node)
        if node.attr not in _SHAPE_ATTRS:  # x.dtype: the same per element
            return node
        if node.attr == "shape":  # x.shape[1:]
            return ast.Subscript(node, ast.Slice(ast.Constant(1), None, None), ast.Load())
        if node.attr == "ndim":
            return ast.BinOp(node, ast.Sub(), ast.Constant(1))
        return ast.BinOp(node, ast.FloorDiv(), ast.Constant(self.size))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        f = node.func
        if (
            isinstance(f, ast.Name)
            and f.id == "len"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.batched
        ):
            shape = ast.Attribute(node.args[0], "shape", ast.Load())
            return ast.Subscript(shape, ast.Constant(1), ast.Load())
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.batched and isinstance(node.ctx, ast.Load):
            # Any other use of a batched input's value (a helper called on the whole array, say).
            raise BatchingError(f"`{node.id}` is used as a whole array; only its tiles and shape can be batched")
        return node
