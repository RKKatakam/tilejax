"""The Python loop nest as a JAX function that can be traced: runtime masks as `jnp.where`, and tiles past
the end of an array as zeros.

A mask that reads input values (`if keep[i // T, k // T] == 0: continue`) is Python control flow on
array values, which only runs eagerly. The reference that `jax.jit`, `jax.vmap` and differentiation
trace evaluates such a mask as an array instead (the same rewrite the kernel's step table uses) and
selects with it: a masked tile's stores keep the output's previous value, and a masked reduction step's
assignments keep each variable's previous value. Masks on loop coordinates and constants stay Python
`if`s: their values are known at trace time.

A tile that runs past the end of an array (the last tile of a ragged loop, `range(0, 200, 64)`) is a whole
tile in the kernel: it reads zeros past the end of an input and drops what it writes past the end of an
output. Python slicing would give a shorter tile instead (which `acc += x[i:i + T] @ ...` couldn't add to
`acc`), so `ragged_tiles` makes the reference read and write tiles that way too: plain Python and NumPy
semantics, sharing nothing with the kernel.
"""

from __future__ import annotations

import ast
import copy
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import analysis, frontend

__all__ = ["helpers", "ragged_tiles", "read_tile", "store_tile", "traceable_masks"]

_WHERE = "_autokernel_where"
_MISSING = "_autokernel_missing"
_READ = "_autokernel_read_tile"
_STORE = "_autokernel_store_tile"


class _Missing:
    """A variable a masked step assigns that had no value before (it stays as the step computes it).

    Only a value the reduction body itself uses can be missing: reading one after its loop is rejected
    unless it was assigned before the loop too (or a static mask runs a step of every tile, which stays a
    Python `if`), so a skipped step's value is never observed."""


def _where(keep: Any, new: Any, old: Any) -> Any:
    if isinstance(old, _Missing):
        return new
    return jnp.where(keep, new, old)


def helpers() -> dict[str, Any]:
    """The names a reference built from `traceable_masks` and `ragged_tiles` needs."""
    return {_WHERE: _where, _MISSING: _Missing(), analysis.XP: jnp, _READ: read_tile, _STORE: store_tile}


def _overrun(shape: tuple[int, ...], index: tuple[Any, ...]) -> list[tuple[int, int]] | None:
    """Per dim of `shape` indexed by `index`, how many elements a slice `a:b` (ints, `0 <= a`) asks for past
    the end, as (dim, missing); None when every slice fits (or one isn't such a slice)."""
    if len(index) > len(shape):
        return None
    over = []
    for d, ix in enumerate(index):
        ints = isinstance(ix, slice) and ix.step is None and _is_int(ix.start) and _is_int(ix.stop)
        if ints and 0 <= ix.start <= ix.stop and ix.stop > shape[d]:
            over.append((d, int(ix.stop) - max(int(ix.start), shape[d])))
    return over or None


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _kept_dim(index: tuple[Any, ...], d: int) -> int:
    """Where dim `d` of an array is in `array[index]` (integer-indexed dims are dropped)."""
    return sum(1 for ix in index[:d] if isinstance(ix, slice))


def read_tile(x: Any, index: Any) -> Any:
    """`x[index]`, with zeros for the elements a slice asks for past the end of `x` (a whole tile)."""
    index = index if isinstance(index, tuple) else (index,)
    over = _overrun(tuple(x.shape), index)
    if over is None:
        return x[index]
    part = jnp.asarray(x[index])
    widths = [(0, 0)] * part.ndim
    for d, missing in over:
        widths[_kept_dim(index, d)] = (0, missing)
    return jnp.pad(part, widths)


def store_tile(out: Any, index: Any, value: Any) -> Any:
    """`out.at[index].set(value)`, dropping the part of a whole tile `value` that lies past the end of `out`."""
    index = index if isinstance(index, tuple) else (index,)
    over = _overrun(tuple(out.shape), index)
    if over is not None and getattr(value, "ndim", 0):
        value = jnp.asarray(value)
        lead = value.ndim - sum(1 for ix in index if isinstance(ix, slice)) - (len(out.shape) - len(index))
        crop = [slice(None)] * value.ndim
        for d, missing in over:
            k = lead + _kept_dim(index, d)
            if 0 <= k < value.ndim and value.shape[k] > 1:  # not a dim the value broadcasts along
                crop[k] = slice(0, value.shape[k] - missing)
        value = value[tuple(crop)]
    return out.at[index].set(value)


class _Tiles(ast.NodeTransformer):
    """`x[idx]` of an array `x` -> `read_tile(x, idx)`, `v = v.at[idx].set(e)` -> `v = store_tile(v, idx, e)`."""

    def __init__(self, arrays: set[str]):
        self.arrays = arrays

    @staticmethod
    def _index(node: ast.expr) -> ast.expr:
        """A subscript's index as a value: `a:b` as `slice(a, b)`."""

        def one(d: ast.expr) -> ast.expr:
            if not isinstance(d, ast.Slice):
                return d
            parts = [d.lower, d.upper, d.step]
            args = [p if p is not None else ast.Constant(None) for p in parts]
            return ast.Call(ast.Name("slice", ast.Load()), args, [])

        if isinstance(node, ast.Tuple):
            return ast.Tuple([one(d) for d in node.elts], ast.Load())
        return one(node)

    def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
        self.generic_visit(n)
        if isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) and n.value.id in self.arrays:
            return ast.copy_location(ast.Call(ast.Name(_READ, ast.Load()), [n.value, self._index(n.slice)], []), n)
        return n

    def visit_Call(self, n: ast.Call) -> ast.AST:
        f = n.func  # `v.at[idx].set(e)`
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "set"
            and len(n.args) == 1
            and not n.keywords
            and isinstance(f.value, ast.Subscript)
            and isinstance(f.value.value, ast.Attribute)
            and f.value.value.attr == "at"
            and isinstance(f.value.value.value, ast.Name)
        ):
            at = f.value
            target, index, value = at.value.value, self.visit(at.slice), self.visit(n.args[0])
            return ast.copy_location(ast.Call(ast.Name(_STORE, ast.Load()), [target, self._index(index), value], []), n)
        self.generic_visit(n)
        return n


def ragged_tiles(fdef: ast.FunctionDef, arrays: set[str], loops: list[ast.For] | None = None) -> ast.FunctionDef:
    """`fdef` (changed in place) with the reads and stores of `arrays` inside `loops` (tile loop nests of
    `fdef`; by default its top-level `for` statements) reading zeros and dropping writes past the end of an
    array (`read_tile`, `store_tile`)."""
    targets = None if loops is None else {id(lp) for lp in loops}
    for parent in [fdef] if loops is None else list(ast.walk(fdef)):
        for field_name in ("body", "orelse"):
            stmts = getattr(parent, field_name, None)
            if not isinstance(stmts, list):
                continue
            hit = [isinstance(s, ast.For) if targets is None else id(s) in targets for s in stmts]
            if any(hit):
                setattr(
                    parent, field_name, [_Tiles(arrays).visit(s) if h else s for s, h in zip(stmts, hit, strict=True)]
                )
    return ast.fix_missing_locations(fdef)


def traceable_masks(nest: frontend.LoopNest) -> ast.FunctionDef:
    """A copy of the nest's function with every mask that reads input values evaluated as an array."""
    memo: dict[int, Any] = {}
    fdef = copy.deepcopy(nest.fdef, memo)
    masks = [(nest.guard_node, True)] + [(r.guard_node, False) for r in nest.reductions]
    found: dict[int, tuple[int, bool]] = {}
    for k, (node, tile) in enumerate(masks):
        if node is not None and analysis.reads_inputs(nest, node.test):
            found[id(memo[id(node)])] = (k, tile)
    if not found:
        return fdef
    stores = {id(memo[id(st.node)]) for st in nest.stores}
    for parent in list(ast.walk(fdef)):
        body = getattr(parent, "body", None)
        if not isinstance(body, list) or not body or id(body[0]) not in found:
            continue
        guard = body[0]
        k, tile = found[id(guard)]
        skip = len(guard.body) == 1 and isinstance(guard.body[0], ast.Continue)
        keep = ast.UnaryOp(ast.Not(), guard.test) if skip else guard.test
        var = f"_autokernel_keep{k}"
        rest = body[1:] if skip else guard.body
        rewrite = (lambda s: _masked_store(s, var) if id(s) in stores else s) if tile else (lambda s: _masked(s, var))
        assign = ast.Assign([ast.Name(var, ast.Store())], analysis.vectorize(ast.copy_location(keep, guard.test)))
        parent.body = [ast.copy_location(assign, guard), *map(rewrite, rest)]
    return ast.fix_missing_locations(fdef)


def _call(fn: str, *args: ast.expr) -> ast.expr:
    return ast.Call(ast.Name(fn, ast.Load()), list(args), [])


def _masked_store(s: ast.stmt, keep: str) -> ast.stmt:
    """`out = out.at[idx].set(v)` -> `out = out.at[idx].set(where(keep, v, out[idx]))`."""
    call = s.value
    at = call.func.value  # out.at[idx]
    old = ast.Subscript(ast.Name(s.targets[0].id, ast.Load()), copy.deepcopy(at.slice), ast.Load())
    call.args = [_call(_WHERE, ast.Name(keep, ast.Load()), call.args[0], old)]
    return s


def _masked(s: ast.stmt, keep: str) -> ast.stmt:
    """`x = e` -> `x = where(keep, e, <x's previous value>)` (and `x op= e` likewise)."""
    name = frontend.assigned_name(s)
    value = s.value
    if isinstance(s, ast.AugAssign):
        value = ast.BinOp(ast.Name(name, ast.Load()), s.op, value)
    previous = ast.Call(
        ast.Attribute(_call("locals"), "get", ast.Load()), [ast.Constant(name), ast.Name(_MISSING, ast.Load())], []
    )
    new = ast.Assign([ast.Name(name, ast.Store())], _call(_WHERE, ast.Name(keep, ast.Load()), value, previous))
    return ast.copy_location(new, s)
