"""Kernels on sharded inputs: run each shard's part of the kernel on its own device, under `jax.shard_map`.

A `pallas_call` only runs on whole arrays of one device, so a kernel called with inputs sharded over a
mesh (`NamedSharding`) is run under `jax.shard_map` when its work splits along the sharded dims: when a
sharded dim of an input is tiled by a parallel loop that is used for nothing but indexing (no masks, no
values computed from it), each output tile reads only the matching shard of that input. That is the case
for batch, head and row dims of most kernels. Each device then runs the kernel generated for its shard's
shapes (the loop over the sharded dim shortened to that shard), and the outputs come out sharded the same
way. Inputs the loop doesn't index are given to every device whole; an input dim sharded along a dim the
kernel can't split (a reduction dim, say) is gathered first, with a one-time warning.

Shardings are read from the arguments: a concrete `jax.Array`'s `NamedSharding` (eager calls), or, under
`jax.jit`, the sharding in a traced value's type, which JAX keeps for meshes with `Explicit` axes (the
default of `jax.make_mesh`). Inside `jax.jit` with `Auto` mesh axes the kernel can't see shardings, and
XLA runs it on gathered inputs.
"""

from __future__ import annotations

import ast
import copy
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from . import frontend
from .analysis import Specialization
from .errors import AutoKernelError

__all__ = ["ShardPlan", "find_mesh", "gather_all", "local_nest", "local_specs", "plan", "run"]

FULL = "_autokernel_full"  # jnp.full, for the per-shard output allocations


@dataclass(frozen=True)
class ShardPlan:
    """How a kernel runs on sharded inputs: a `jax.shard_map` over `mesh`.

    `loops` maps each loop split across devices to its mesh axes and shard count; with none, every
    device runs the whole kernel on gathered inputs. `gathered` lists the (input, dim) pairs that are
    sharded but have to be gathered first.
    """

    mesh: Any  # jax.sharding.Mesh or AbstractMesh
    explicit: bool  # the mesh has Explicit axes: inputs are resharded to `in_specs` before the shard_map
    in_specs: tuple[PartitionSpec, ...]
    out_specs: tuple[PartitionSpec, ...]
    loops: tuple[tuple[str, tuple[str, ...], int], ...]  # (loop variable, mesh axes, shards)
    gathered: tuple[tuple[str, int], ...]

    @property
    def key(self) -> tuple:
        return (self.mesh, self.in_specs, self.out_specs, self.loops)


def _is_trivial(spec: PartitionSpec | None) -> bool:
    return spec is None or all(p is None for p in spec)


def _sharding(x: Any) -> NamedSharding | None:
    """`x`'s NamedSharding if it is split over some mesh axis, else None."""
    s = jax.typeof(x).sharding if isinstance(x, jax.core.Tracer) else getattr(x, "sharding", None)
    if not isinstance(s, NamedSharding) or _is_trivial(s.spec):
        return None
    if any(math.prod(s.mesh.shape[a] for a in _axes(p)) > 1 for p in s.spec if p is not None):
        return s
    # Split only over axes of size 1: nothing to split, but an Explicit mesh's axes are in the value's type,
    # which a pallas_call outside a shard_map rejects.
    return s if _is_explicit(s.mesh) else None


def _is_explicit(mesh: Any) -> bool:
    return any(t == jax.sharding.AxisType.Explicit for t in getattr(mesh, "axis_types", ()))


def _axes(entry: Any) -> tuple[str, ...]:
    """The mesh axes of one PartitionSpec entry."""
    if entry is None:
        return ()
    return tuple(entry) if isinstance(entry, (tuple, list)) else (entry,)


def find_mesh(args: Sequence[Any]) -> tuple[Any, list[PartitionSpec | None]] | None:
    """(mesh, each argument's PartitionSpec or None) when some argument is sharded, else None."""
    shardings = [_sharding(a) for a in args]
    found = [s for s in shardings if s is not None]
    if not found:
        return None
    return found[0].mesh, [s.spec if s is not None else None for s in shardings]


# --------------------------------------------------------------------------- which loops can be split


def _index_vars(index: ast.expr, loops: set[str]) -> tuple[list[str | None], set[int]]:
    """Per dim of a subscript, the loop variable indexing it directly (`v` or `v:v + T`), or None; and the
    ids of those direct uses' Name nodes."""
    elts = list(index.elts) if isinstance(index, ast.Tuple) else [index]
    out: list[str | None] = []
    used: set[int] = set()
    for e in elts:
        if isinstance(e, ast.Name) and e.id in loops:
            out.append(e.id)
            used.add(id(e))
        elif isinstance(e, ast.Slice) and isinstance(e.lower, ast.Name) and e.lower.id in loops:
            var = e.lower.id
            up = e.upper
            ok = isinstance(up, ast.BinOp) and isinstance(up.op, ast.Add)
            side = [n for n in (up.left, up.right) if isinstance(n, ast.Name) and n.id == var] if ok else []
            if not side or e.step is not None:
                out.append(None)
                continue
            out.append(var)
            used |= {id(e.lower), id(side[0])}
        else:
            out.append(None)
    return out, used


def _splittable(spec: Specialization) -> tuple[dict[str, dict[str, int]], dict[str, str], dict[tuple, set[str]]]:
    """The parallel loops a shard can run part of on its own: ({loop: {input: dim it indexes}}, why each
    other parallel loop can't be split, and {(input, dim): the loops indexing it anywhere})."""
    nest = spec.nest
    loops = set(spec.loops)
    params = set(nest.params) - set(spec.scalars)
    top = nest.parallel[0].node
    subscripts: dict[str, list[list[str | None]]] = {p: [] for p in params}
    direct: set[int] = set()
    for node in ast.walk(top):
        if not isinstance(node, ast.Subscript):
            continue
        v = node.value
        if isinstance(v, ast.Name) and v.id in params:
            dims, used = _index_vars(node.slice, loops)
            subscripts[v.id].append(dims)
            direct |= used
        elif isinstance(v, ast.Attribute) and v.attr == "at" and isinstance(v.value, ast.Name):
            direct |= _index_vars(node.slice, loops)[1]
    targets = {id(n.target) for n in ast.walk(top) if isinstance(n, ast.For)}
    other_uses = {
        n.id for n in ast.walk(top) if isinstance(n, ast.Name) and id(n) not in direct and id(n) not in targets
    }
    masked = nest.guard is not None
    ok: dict[str, dict[str, int]] = {}
    why: dict[str, str] = {}
    for lb in spec.parallel:
        var = lb.var
        if var in other_uses:
            why[var] = f"loop `{var}` is used for more than indexing (a mask or a value)"
            continue
        if lb.start != 0:
            why[var] = f"loop `{var}` doesn't start at 0"
            continue
        dims: dict[str, int] = {}
        for p, subs in subscripts.items():
            found = [tuple(d for d, v in enumerate(s) if v == var) for s in subs]
            if not found or not any(found):
                continue
            if len(set(found)) > 1 or len(found[0]) != 1:
                why[var] = f"input `{p}` is read both along loop `{var}` and otherwise"
                break
            d = found[0][0]
            if spec.given_shape(p)[d] != lb.stop:
                why[var] = f"loop `{var}` covers only part of `{p}`'s dim {d}"
                break
            dims[p] = d
        else:
            for out in spec.outputs:
                if out.padded:
                    why[var] = f"`{out.var}`'s last tiles run past its end"
                    break
                if getattr(out, "partial", False) or (masked and out.fill is None):
                    why[var] = f"`{out.var}` keeps values of its allocation that a shard can't compute alone"
                    break
            else:
                ok[var] = dims
    indexers: dict[tuple, set[str]] = {}
    for p, subs in subscripts.items():
        for s in subs:
            for d, v in enumerate(s):
                if v is not None:
                    indexers.setdefault((p, d), set()).add(v)
    return ok, why, indexers


def plan(spec: Specialization, mesh: Any, arg_specs: Sequence[PartitionSpec | None]) -> ShardPlan:
    """How to run the kernel specialised as `spec` on arguments sharded as `arg_specs` over `mesh`."""
    nest = spec.nest
    splittable, why, indexers = _splittable(spec)
    explicit = _is_explicit(mesh)
    assigned: dict[str, tuple[str, ...]] = {}
    gathered: list[tuple[str, int, str]] = []
    used_axes: set[str] = set()
    for p, arg_spec in zip(nest.params, arg_specs, strict=True):
        if _is_trivial(arg_spec):
            continue
        for d, entry in enumerate(arg_spec):
            axes = _axes(entry)
            if not axes:
                continue
            var = next((v for v, dims in splittable.items() if dims.get(p) == d), None)
            if var is None:
                loops = sorted(indexers.get((p, d), ()))
                reason = "; ".join(why[v] for v in loops if v in why)
                if not reason:
                    reason = f"tiled by reduction loop `{loops[0]}`" if loops else "read whole by every tile"
                gathered.append((p, d, reason))
                continue
            lb = spec.loops[var]
            shards = math.prod(mesh.shape[a] for a in axes)
            if assigned.get(var, axes) != axes or (var not in assigned and used_axes & set(axes)):
                gathered.append((p, d, f"loop `{var}` is already split another way"))
                continue
            if lb.count % shards:
                gathered.append((p, d, f"loop `{var}`'s {lb.count} blocks don't split evenly into {shards} shards"))
                continue
            assigned[var] = axes
            used_axes |= set(axes)
    if gathered:
        _warn_gathered(nest.fn_name, gathered)
    return _make(spec, mesh, explicit, assigned, splittable, gathered)


def gather_all(spec: Specialization, shard_plan: ShardPlan, reason: str) -> ShardPlan:
    """`shard_plan` with nothing split: every device runs the whole kernel on gathered inputs."""
    gathered = [
        (p, d, reason) for p, s in zip(spec.nest.params, shard_plan.in_specs, strict=True) for d, e in enumerate(s) if e
    ]
    _warn_gathered(spec.nest.fn_name, gathered)
    return _make(spec, shard_plan.mesh, shard_plan.explicit, {}, {}, gathered)


def _make(
    spec: Specialization,
    mesh: Any,
    explicit: bool,
    assigned: dict[str, tuple[str, ...]],
    splittable: dict[str, dict[str, int]],
    gathered: list[tuple[str, int, str]],
) -> ShardPlan:
    def entry(axes: tuple[str, ...] | None) -> Any:
        return None if not axes else axes if len(axes) > 1 else axes[0]

    in_specs = []
    for p in spec.nest.params:
        entries: list[Any] = [None] * len(spec.input_shape(p))
        for var, axes in assigned.items():
            d = splittable[var].get(p)
            if d is not None:
                entries[d] = entry(axes)
        in_specs.append(PartitionSpec(*entries))
    out_specs = tuple(PartitionSpec(*[entry(assigned.get(v)) for v in out.block.loops]) for out in spec.outputs)
    loops = tuple((v, axes, math.prod(mesh.shape[a] for a in axes)) for v, axes in assigned.items())
    return ShardPlan(mesh, explicit, tuple(in_specs), out_specs, loops, tuple((p, d) for p, d, _ in gathered))


def local_specs(shard_plan: ShardPlan, args: Sequence[Any]) -> list[jax.ShapeDtypeStruct]:
    """The shapes and dtypes of one device's shards of `args`."""
    out = []
    for a, s in zip(args, shard_plan.in_specs, strict=True):
        shape = list(np.shape(a))
        for d, e in enumerate(s):
            shape[d] //= math.prod(shard_plan.mesh.shape[x] for x in _axes(e))
        dtype = a.dtype if hasattr(a, "dtype") else jnp.int32
        out.append(jax.ShapeDtypeStruct(tuple(shape), dtype))
    return out


_warned: set[tuple] = set()


def _warn_gathered(name: str, gathered: list[tuple[str, int, str]]) -> None:
    key = (name, tuple(gathered))
    if key in _warned:
        return
    _warned.add(key)
    dims = "; ".join(f"`{p}` dim {d} ({reason})" for p, d, reason in gathered)
    warnings.warn(
        f"autokernel: {name}: the kernel can't split along every sharded input dim, so these are gathered onto "
        f"each device first: {dims}",
        stacklevel=4,
    )


# --------------------------------------------------------------------------- the per-shard kernel


def local_nest(spec: Specialization, shard_plan: ShardPlan) -> tuple[frontend.LoopNest, dict[str, Any]]:
    """The nest each device runs: the split loops shortened to one shard, the outputs allocated at their
    per-shard shapes, and the inputs' shapes still the global ones (so constants the prologue computes from
    them keep their values). Returns the nest and the names its code needs besides the kernel's own."""
    nest = spec.nest
    shards = {v: n for v, _, n in shard_plan.loops}
    fdef = copy.deepcopy(nest.fdef)
    arrays = {p: tuple(spec.given_shape(p)) for p in nest.params if p not in spec.scalars}
    _GlobalShapes(arrays).visit(fdef)
    extra: dict[str, Any] = {FULL: jnp.full}
    outputs = {o.var: o for o in spec.outputs}
    for s in ast.walk(fdef):
        if isinstance(s, ast.For) and isinstance(s.target, ast.Name) and s.target.id in shards:
            lb = spec.loops[s.target.id]
            step = s.iter.args[2] if len(s.iter.args) == 3 else ast.Constant(1)
            stop = ast.Constant(lb.stop // shards[lb.var])
            s.iter = ast.Call(ast.Name("range", ast.Load()), [ast.Constant(0), stop, step], [])
    for s in fdef.body:  # the allocations (the stores inside the nest assign the same names)
        if not (isinstance(s, ast.Assign) and frontend.is_simple_assignment(s) and s.targets[0].id in outputs):
            continue
        out = outputs[s.targets[0].id]
        shape = tuple(n // shards.get(v, 1) for n, v in zip(out.out.shape, out.block.loops, strict=True))
        fill = f"_autokernel_fill{len(extra)}"
        extra[fill] = out.fill if out.fill is not None else 0  # without a tile mask, every element is stored
        dtype = ast.Constant(np.dtype(out.out.dtype).name)
        s.value = ast.Call(ast.Name(FULL, ast.Load()), [ast.Constant(shape), ast.Name(fill, ast.Load()), dtype], [])
    split = {p for p, ps in zip(nest.params, shard_plan.in_specs, strict=True) if not _is_trivial(ps)}
    for p in sorted(_whole_uses(fdef) & split):
        # `jnp.shape(x)`, say: the shard's value would differ from the global one the kernel computes with.
        raise AutoKernelError(f"input `{p}` is split, but the code uses it whole")
    name = f"{nest.fn_name}_shard"
    fdef.name = name
    ast.fix_missing_locations(fdef)
    return frontend.parse_function(fdef, name, nest.first_line), extra


def _whole_uses(fdef: ast.FunctionDef) -> set[str]:
    """Names the code reads other than through a subscript (a tile) or for their dtype or rank."""
    parts = {
        id(n.value)
        for n in ast.walk(fdef)
        if isinstance(n, ast.Subscript) or (isinstance(n, ast.Attribute) and n.attr in ("dtype", "ndim"))
    }
    loads = [n for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    return {n.id for n in loads if id(n) not in parts}


class _GlobalShapes(ast.NodeTransformer):
    """`x.shape` (and `len(x)`, `x.size`) as the global shape's literal values."""

    def __init__(self, shapes: dict[str, tuple[int, ...]]):
        self.shapes = shapes

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        v = node.value
        if isinstance(v, ast.Name) and v.id in self.shapes and node.attr in ("shape", "size"):
            shape = self.shapes[v.id]
            return ast.copy_location(ast.Constant(shape if node.attr == "shape" else math.prod(shape)), node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        f, args = node.func, node.args
        if (
            isinstance(f, ast.Name)
            and f.id == "len"
            and len(args) == 1
            and isinstance(args[0], ast.Name)
            and args[0].id in self.shapes
        ):
            return ast.copy_location(ast.Constant(self.shapes[args[0].id][0]), node)
        return self.generic_visit(node)


def run(shard_plan: ShardPlan, local: Callable[..., Any], tuple_out: bool) -> Callable[..., Any]:
    """`local` (the per-shard kernel) run under `jax.shard_map` as `shard_plan` says."""
    out_specs = shard_plan.out_specs if tuple_out else shard_plan.out_specs[0]
    mapped = jax.shard_map(
        local, mesh=shard_plan.mesh, in_specs=shard_plan.in_specs, out_specs=out_specs, check_vma=False
    )

    def call(*args: Any) -> Any:
        args = [a if hasattr(a, "shape") else jnp.asarray(a, jnp.int32) for a in args]
        if shard_plan.explicit:
            args = [
                jax.sharding.reshard(a, NamedSharding(shard_plan.mesh, s))
                for a, s in zip(args, shard_plan.in_specs, strict=True)
            ]
        return mapped(*args)

    return jax.jit(call)
