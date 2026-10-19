"""The Python loop nest as a JAX function that can be traced: runtime masks as `jnp.where`.

A mask that reads input values (`if keep[i // T, k // T] == 0: continue`) is Python control flow on
array values, which only runs eagerly. The reference that `jax.jit`, `jax.vmap` and differentiation
trace evaluates such a mask as an array instead (the same rewrite the kernel's step table uses) and
selects with it: a masked tile's stores keep the output's previous value, and a masked reduction step's
assignments keep each variable's previous value. Masks on loop coordinates and constants stay Python
`if`s: their values are known at trace time.
"""

from __future__ import annotations

import ast
import copy
from typing import Any

import jax.numpy as jnp

from . import analysis, frontend

__all__ = ["helpers", "traceable_masks"]

_WHERE = "_autokernel_where"
_MISSING = "_autokernel_missing"


class _Missing:
    """A variable a masked step assigns that had no value before (it stays as the step computes it).

    Only a value the reduction body itself uses can be missing: the frontend rejects reading one after its
    loop unless it was assigned before the loop too, so a skipped step's value is never observed."""


def _where(keep: Any, new: Any, old: Any) -> Any:
    if isinstance(old, _Missing):
        return new
    return jnp.where(keep, new, old)


def helpers() -> dict[str, Any]:
    """The names a reference built from `traceable_masks` needs."""
    return {_WHERE: _where, _MISSING: _Missing(), analysis.XP: jnp}


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
