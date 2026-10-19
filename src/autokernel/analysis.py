"""Specialise a parsed loop nest to concrete input shapes.

Everything here is analysis: it evaluates the user's loop bounds, slices, masks and output allocations
at trace time and records what the kernel must do, without generating code. `schedule` turns the
result into the list of steps the kernel runs, and `codegen` into Pallas source.

A tile runs as a sequence of *units*: segment 0, reduction body 0, segment 1, ..., the last segment and
the stores. Each unit is a list of the user's statements. A value written in one unit and read in another
(an accumulator, or a per-tile constant read inside a reduction loop) has to survive between steps, so it
becomes a *state variable* in VMEM (SMEM for scalars); every other value is a plain local.
"""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal

from . import mxu
from .errors import AutoKernelError
from .frontend import Loop, LoopNest, Reduction, assigned_name

__all__ = [
    "Block",
    "LoopBounds",
    "Operand",
    "OutputSpec",
    "Phase",
    "Specialization",
    "StateVar",
    "Unit",
    "evaluate_mask",
    "phase_coords",
    "specialize",
    "tile_coords",
    "user_globals",
]

Scalar = bool | int | float

# Primitives that keep "every element is the same scalar" true, used to find constant fills.
_SHAPE_ONLY_OPS = frozenset({"broadcast_in_dim", "reshape", "transpose", "copy", "squeeze", "expand_dims"})
_ELEMENTWISE_OPS = frozenset(
    {
        "add", "sub", "mul", "div", "max", "min", "neg", "abs", "sign", "exp", "log", "sqrt", "rsqrt",
        "pow", "integer_pow", "floor", "ceil", "round", "convert_element_type",
    }
)  # fmt: skip


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class LoopBounds:
    """A tile loop's evaluated `range(start, stop, step)`."""

    var: str
    start: int
    stop: int
    step: int

    @property
    def count(self) -> int:
        return (self.stop - self.start) // self.step

    @property
    def blocks(self) -> range:
        """Absolute block indices the loop visits (start and stop are multiples of step)."""
        return range(self.start // self.step, self.stop // self.step)


@dataclass(frozen=True)
class Block:
    """Which tile of an array one access reads or writes.

    Per dim: the loop variable tiling it (`loops[d]`), or None with the constant block index in
    `const[d]`. Dims indexed by an integer (`x[b, ...]`) are `squeezed`: size 1 and dropped from the tile
    the kernel sees.
    """

    shape: tuple[int, ...]
    loops: tuple[str | None, ...]
    const: tuple[int, ...]
    squeezed: tuple[bool, ...]
    text: str  # the index as the user wrote it, for comments and errors

    @property
    def key(self) -> tuple:
        """Everything that decides which HBM window the access touches and how the kernel sees it."""
        return (self.shape, self.loops, self.const, self.squeezed)

    @property
    def kernel_shape(self) -> tuple[int, ...]:
        return self.kept(self.shape)

    def kept(self, full: tuple[int, ...]) -> tuple[int, ...]:
        """The dims of `full` the kernel's tile keeps (squeezed dims dropped)."""
        return tuple(n for n, sq in zip(full, self.squeezed, strict=True) if not sq)


@dataclass(frozen=True)
class Operand:
    """An input read through one particular tiling (an input sliced two ways is two operands).

    `phase` is the reduction whose loops tile it (its block changes every step of that reduction), or
    None for a per-tile operand (tiled by parallel loops only; it stays put for the whole tile).
    """

    param: str
    ref: str
    block: Block
    full_shape: tuple[int, ...]
    dtype: np.dtype
    phase: int | None


@dataclass(frozen=True)
class OutputSpec:
    var: str
    ref: str
    out: jax.ShapeDtypeStruct
    block: Block
    alloc: ast.expr
    fill: Scalar | None  # the allocation's value when every element is the same known constant


@dataclass(frozen=True)
class StateVar:
    """A tile value that survives between steps."""

    name: str
    ref: str
    aval: jax.ShapeDtypeStruct  # as stored (narrow float accumulators are widened to f32)
    smem: bool  # 0-d values live in SMEM, arrays in VMEM


@dataclass(frozen=True)
class Phase:
    """A reduction loop (nest): its loops, mask and step count per tile."""

    index: int
    loops: tuple[LoopBounds, ...]
    reduction: Reduction

    @property
    def steps(self) -> int:
        return math.prod(lb.count for lb in self.loops)


@dataclass(frozen=True)
class Unit:
    """One run of the user's statements within a tile: a segment, or a reduction body."""

    kind: str  # "segment" | "body"
    index: int  # segment or phase index
    statements: tuple[ast.stmt, ...]

    @property
    def phase(self) -> int | None:
        """The reduction whose loops the unit's reads may be tiled by (None: parallel loops only)."""
        return self.index if self.kind == "body" else None


@dataclass
class Specialization:
    """A loop nest with every trace-time fact resolved for one set of input shapes/dtypes."""

    nest: LoopNest
    specs: list[jax.ShapeDtypeStruct]
    filename: str
    consts: dict[str, Any]  # names the user's code can see, minus the inputs
    loops: dict[str, LoopBounds]
    parallel: list[LoopBounds]
    phases: list[Phase]
    units: list[Unit]
    names: Namer
    code: UserCode
    overrides: dict[str, Any] = field(default_factory=dict)
    # Filled in by the passes below.
    tiles: list[tuple[int, ...]] = field(default_factory=list)  # parallel block indices of every tile, in loop order
    constant_tiles: list[tuple[int, ...]] = field(default_factory=list)  # tiles a static tile mask skips
    tile_mask_runtime: bool = False  # the tile mask reads inputs: decided on device, per tile
    # Per phase: the (tile x step) mask evaluated at trace time, or None when the phase has no static mask.
    phase_masks: list[np.ndarray | None] = field(default_factory=list)
    phase_runtime: list[bool] = field(default_factory=list)  # per phase: its mask reads inputs
    operands: list[Operand] = field(default_factory=list)
    outputs: list[OutputSpec] = field(default_factory=list)
    state: dict[str, StateVar] = field(default_factory=dict)
    casts: dict[int, np.dtype] = field(default_factory=dict)  # id(statement) -> dtype stored to its state variable
    _reads: dict[tuple[str, str, int | None], Operand] = field(default_factory=dict, repr=False)

    @property
    def runtime_masks(self) -> bool:
        return self.tile_mask_runtime or any(self.phase_runtime)

    @property
    def total_tiles(self) -> int:
        """Output tiles before any tile mask."""
        return len(self.tiles) + len(self.constant_tiles)

    def loop_vars(self, phase: int | None) -> set[str]:
        """Loop variables that may tile an access in a body of `phase` (None: at tile level)."""
        allowed = {lb.var for lb in self.parallel}
        if phase is not None:
            allowed |= {lb.var for lb in self.phases[phase].loops}
        return allowed

    def namespace(self, arrays: Sequence[Any] | None = None, **extra: Any) -> dict[str, Any]:
        """What the user's code can see: constants, the inputs (as `arrays`, or their specs) and `extra`."""
        inputs = self.specs if arrays is None else arrays
        return {**self.consts, **dict(zip(self.nest.params, inputs, strict=True)), **extra}

    def input_shape(self, param: str) -> tuple[int, ...]:
        return tuple(self.specs[self.nest.params.index(param)].shape)

    def input_dtype(self, param: str) -> np.dtype:
        return np.dtype(self.specs[self.nest.params.index(param)].dtype)

    def resolve_input(self, node: ast.Subscript, phase: int | None) -> Block:
        """The block a tile slice of an input reads (in a body of `phase`, or at tile level)."""
        param = node.value.id
        return _resolve_block(self, param, node.slice, self.input_shape(param), self.loop_vars(phase))

    def operand(self, node: ast.Subscript, phase: int | None) -> Operand:
        """The operand a tile slice of an input refers to (`node` may be a copy of the collected read)."""
        key = (node.value.id, ast.dump(node.slice), phase)
        op = self._reads.get(key)
        if op is None:  # not one of the collected reads: match it by block
            block = self.resolve_input(node, phase)
            op = next((o for o in self.operands if (o.param, o.block.key) == (node.value.id, block.key)), None)
            if op is None:
                raise AssertionError(f"operand for {ast.unparse(node)} was not collected")  # analysis bug
            self._reads[key] = op
        return op


class Namer:
    """Hands out identifiers for generated code that can't collide with the user's names."""

    def __init__(self, taken: set[str]):
        self._taken = set(taken)

    def fresh(self, base: str) -> str:
        name = base
        while name in self._taken:
            name += "_"
        self._taken.add(name)
        return name


class UserCode:
    """Compiles fragments of the user's function so tracebacks point at their file and lines."""

    def __init__(self, nest: LoopNest, filename: str):
        self.nest, self.filename = nest, filename

    def compile(self, node: ast.AST | list[ast.stmt], mode: str):
        node = copy.deepcopy(node)
        wrapped = ast.Expression(node) if mode == "eval" else ast.Module(body=node, type_ignores=[])
        ast.increment_lineno(wrapped, self.nest.first_line - 1)
        ast.fix_missing_locations(wrapped)
        return compile(wrapped, self.filename, mode)

    def eval_int(self, node: ast.expr, what: str, env: dict[str, Any]) -> int:
        try:
            value = eval(self.compile(node, "eval"), dict(env))
        except Exception as e:
            raise self.nest.error(
                node, f"can't evaluate {what} `{ast.unparse(node)}` at trace time ({type(e).__name__}: {e})"
            ) from None
        if not isinstance(value, (int, np.integer)):
            raise self.nest.error(node, f"{what} `{ast.unparse(node)}` must be a Python int, got {value!r}")
        return int(value)


def user_globals(fn: Callable) -> dict[str, Any]:
    """The names `fn`'s body can see besides its parameters: module globals and closure cells."""
    env = dict(fn.__globals__)
    env.update(inspect.getclosurevars(fn).nonlocals)
    return env


# --------------------------------------------------------------------------- entry point


def specialize(
    nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct], overrides: dict[str, Any] | None = None
) -> Specialization:
    """Resolve `nest` for inputs of the given shapes and dtypes, validating it along the way.

    `overrides` replaces module or closure constants (tile sizes being tuned, for example).
    """
    specs = list(specs)
    if len(specs) != len(nest.params):
        raise AutoKernelError(f"{nest.fn_name} takes {len(nest.params)} arrays, got {len(specs)}")
    code = UserCode(nest, inspect.getsourcefile(fn) or "<autokernel>")

    # Names the user's code can see: module globals, closure cells, overrides, then the prologue.
    # Inputs are bound to ShapeDtypeStructs, so the prologue can only use their shapes and dtypes.
    env = user_globals(fn)
    env.update(overrides or {})
    env.update(zip(nest.params, specs, strict=True))
    if nest.prologue:
        exec(code.compile(nest.prologue, "exec"), env)
    consts = {k: v for k, v in env.items() if k not in nest.params}

    loops = {lp.var: _bounds(lp, code, env, nest) for lp in nest.loops}
    user_names = {n.id for s in nest.tile_statements() for n in ast.walk(s) if isinstance(n, ast.Name)}
    spec = Specialization(
        nest=nest,
        specs=specs,
        filename=code.filename,
        consts=consts,
        loops=loops,
        parallel=[loops[lp.var] for lp in nest.parallel],
        phases=[Phase(p, tuple(loops[lp.var] for lp in r.loops), r) for p, r in enumerate(nest.reductions)],
        units=_units(nest),
        names=Namer(user_names | set(nest.params) | set(consts) | set(nest.out_vars)),
        code=code,
        overrides=dict(overrides or {}),
    )
    spec.tiles, spec.constant_tiles, spec.tile_mask_runtime = _tile_mask(spec)
    spec.phase_masks, spec.phase_runtime = _phase_masks(spec)
    spec.outputs = _outputs(spec)
    spec.operands = _collect_operands(spec)
    spec.state, spec.casts = _state(spec)
    return spec


def _bounds(lp: Loop, code: UserCode, env: dict[str, Any], nest: LoopNest) -> LoopBounds:
    what = f"`range` argument of loop `{lp.var}`"
    start, stop, step = (code.eval_int(e, what, env) for e in (lp.start, lp.stop, lp.step))
    if step <= 0 or start % step or (stop - start) % step or stop <= start:
        raise nest.error(
            lp.node,
            f"loop `{lp.var}`: range({start}, {stop}, {step}) must be non-empty, positive-step and aligned "
            "(start and stop-start divisible by step)",
        )
    return LoopBounds(lp.var, start, stop, step)


def _units(nest: LoopNest) -> list[Unit]:
    units = []
    for p, seg in enumerate(nest.segments):
        units.append(Unit("segment", p, tuple(seg)))
        if p < len(nest.reductions):
            units.append(Unit("body", p, tuple(nest.reductions[p].body)))
    return units


# --------------------------------------------------------------------------- tiles and slices


def _index_text(index: ast.expr) -> str:
    """The index as the user wrote it: `b, h, i:i + T, :` rather than the unparser's `(b, h, ...)`."""
    return ", ".join(map(ast.unparse, index.elts)) if isinstance(index, ast.Tuple) else ast.unparse(index)


@dataclass(frozen=True)
class _Dim:
    """One resolved dimension of a Block."""

    size: int
    loop: str | None
    const: int
    squeezed: bool


def _resolve_block(spec: Specialization, name: str, index: ast.expr, full: tuple[int, ...], allowed: set[str]) -> Block:
    """Turn a tile index like `x[b, i:i+T, :]` into a Block, checking it against the loops."""
    text = _index_text(index)
    what = f"`{name}[{text}]`"
    dims = list(index.elts) if isinstance(index, ast.Tuple) else [index]
    if len(dims) > len(full):
        raise spec.nest.error(index, f"{what}: {len(dims)} indices for a rank-{len(full)} array")
    dims += [ast.Slice()] * (len(full) - len(dims))
    dims_and_sizes = enumerate(zip(dims, full, strict=True))
    resolved = [_resolve_dim(spec, index, what, d, sl, size, allowed) for d, (sl, size) in dims_and_sizes]
    return Block(
        tuple(r.size for r in resolved),
        tuple(r.loop for r in resolved),
        tuple(r.const for r in resolved),
        tuple(r.squeezed for r in resolved),
        text,
    )


def _resolve_dim(
    spec: Specialization, index: ast.expr, what: str, d: int, sl: ast.expr, size: int, allowed: set[str]
) -> _Dim:
    nest, env = spec.nest, spec.namespace()  # tile sizes may use x.shape
    if not isinstance(sl, ast.Slice):  # an integer index: a loop stepping by 1, or a constant
        if isinstance(sl, ast.Name) and sl.id in spec.loops:
            lb = _loop_dim(spec, index, what, d, sl.id, size, allowed)
            if lb.step != 1:
                raise nest.error(
                    index,
                    f"{what}: dim {d} is indexed by loop `{sl.id}`, which steps by {lb.step}; only loops "
                    f"stepping by 1 can index a dim (use a slice `{sl.id}:{sl.id} + {lb.step}`)",
                )
            return _Dim(1, sl.id, 0, True)
        if {n.id for n in ast.walk(sl) if isinstance(n, ast.Name)} & set(spec.loops):
            raise nest.error(
                index, f"{what}: an integer index must be a loop variable or a constant, not `{ast.unparse(sl)}`"
            )
        c = spec.code.eval_int(sl, "index", env)
        if not 0 <= c < size:
            raise nest.error(index, f"{what}: index {c} in dim {d} is outside the array (size {size})")
        return _Dim(1, None, c, True)
    if sl.step is not None:
        raise nest.error(index, f"{what}: dim {d} must be a tile slice, not a strided one")
    if sl.lower is None and sl.upper is None:
        return _Dim(size, None, 0, False)
    if isinstance(sl.lower, ast.Name) and sl.lower.id in spec.loops:  # `v:v + T`
        var = sl.lower.id
        lb = _loop_dim(spec, index, what, d, var, size, allowed)
        tile = _tile_size(sl.upper, var)
        if tile is None:
            raise nest.error(index, f"{what}: dim {d} must look like `{var}:{var} + T`")
        t = spec.code.eval_int(tile, "tile size", env)
        if t != lb.step:
            raise nest.error(index, f"{what}: tile size {t} in dim {d} must equal the step ({lb.step}) of loop `{var}`")
        return _Dim(t, var, 0, False)
    lo = 0 if sl.lower is None else spec.code.eval_int(sl.lower, "slice bound", env)
    hi = size if sl.upper is None else spec.code.eval_int(sl.upper, "slice bound", env)
    t = hi - lo
    if t <= 0 or lo % t or hi > size:
        raise nest.error(
            index, f"{what}: constant slice {lo}:{hi} in dim {d} must be a whole aligned block inside the array"
        )
    return _Dim(t, None, lo // t, False)


def _loop_dim(
    spec: Specialization, index: ast.expr, what: str, d: int, var: str, size: int, allowed: set[str]
) -> LoopBounds:
    """Check that loop `var` may tile dim `d` here and fits in it."""
    if var not in allowed:
        raise spec.nest.error(index, f"{what}: can't be tiled by loop `{var}` here")
    lb = spec.loops[var]
    if lb.stop > size:
        raise spec.nest.error(index, f"{what}: loop `{var}` runs to {lb.stop} but dim {d} only has size {size}")
    return lb


def _tile_size(upper: ast.expr | None, var: str) -> ast.expr | None:
    """T in an upper bound written `var + T` or `T + var`."""
    if isinstance(upper, ast.BinOp) and isinstance(upper.op, ast.Add):
        if isinstance(upper.left, ast.Name) and upper.left.id == var:
            return upper.right
        if isinstance(upper.right, ast.Name) and upper.right.id == var:
            return upper.left
    return None


def _input_reads(nest: LoopNest, nodes: Sequence[ast.AST]) -> Iterator[ast.Subscript]:
    """Every tile slice of an input in `nodes`, in source order. Rejects other reads of inputs/outputs."""
    found: list[ast.Subscript] = []

    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in nest.params:
                found.append(node)
                self.visit(node.slice)
            else:
                self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in nest.params:
                raise nest.error(
                    node,
                    f"input `{node.id}` may only be read through a tile slice like `{node.id}[i:i+T, ...]` "
                    "inside the loop nest",
                )
            if node.id in nest.out_vars:
                raise nest.error(node, f"the output `{node.id}` can't be read inside the loop nest")

    visitor = Visitor()
    for node in nodes:
        visitor.visit(node)
    return iter(found)


def _collect_operands(spec: Specialization) -> list[Operand]:
    nest = spec.nest
    reads = [(node, unit.phase) for unit in spec.units for node in _input_reads(nest, unit.statements)]
    reads += [(node, None) for node in _input_reads(nest, [st.value for st in nest.stores])]
    operands: list[Operand] = []
    for node, phase in reads:
        param, block = node.value.id, spec.resolve_input(node, phase)
        op = next((o for o in operands if (o.param, o.block.key) == (param, block.key)), None)
        if op is None:
            tiled_by_phase = phase is not None and any(lb.var in block.loops for lb in spec.phases[phase].loops)
            existing = sum(o.param == param for o in operands)
            ref = spec.names.fresh(f"{param}_ref" if existing == 0 else f"{param}_ref{existing}")
            op = Operand(
                param, ref, block, spec.input_shape(param), spec.input_dtype(param), phase if tiled_by_phase else None
            )
            operands.append(op)
        spec._reads[(param, ast.dump(node.slice), phase)] = op
    return operands


# --------------------------------------------------------------------------- outputs


def _outputs(spec: Specialization) -> list[OutputSpec]:
    nest = spec.nest
    stores = {st.out: st for st in nest.stores}
    par_vars = {lb.var for lb in spec.parallel}
    outputs = []
    for output in nest.outputs:
        alloc = spec.code.compile(output.alloc, "eval")
        try:
            out = jax.eval_shape(lambda *xs, alloc=alloc: eval(alloc, spec.namespace(xs)), *spec.specs)
        except Exception as e:
            raise nest.error(
                output.alloc, f"couldn't evaluate the output allocation ({type(e).__name__}: {e})"
            ) from None
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise nest.error(output.alloc, "an output allocation must produce a single array")
        store = stores[output.var]
        block = _resolve_block(spec, f"{output.var}.at", store.index, tuple(out.shape), par_vars)
        tiled_by = [v for v in block.loops if v]
        for var in sorted(par_vars - set(tiled_by)):
            raise nest.error(
                store.node,
                f"parallel loop `{var}` must tile one of `{output.var}`'s dims, otherwise tiles race on the same "
                "output block",
            )
        for var in sorted({v for v in tiled_by if tiled_by.count(v) > 1}):
            raise nest.error(
                store.node,
                f"parallel loop `{var}` tiles two dims of `{output.var}`, so the tiles only cover its block "
                "diagonal; every output element must be produced by the kernel",
            )
        for d, (var, n, t) in enumerate(zip(block.loops, out.shape, block.shape, strict=True)):
            covered = (spec.loops[var].start, spec.loops[var].stop) == (0, n) if var else t == n
            if not covered:
                raise nest.error(
                    store.node,
                    f"the tiles don't write all of `{output.var}`'s dim {d} (size {n}); every output element must "
                    "be produced by the kernel",
                )
        ref = spec.names.fresh(f"{output.var}_ref")
        outputs.append(OutputSpec(output.var, ref, out, block, output.alloc, _constant_fill(spec, alloc)))
    return outputs


def _constant_fill(spec: Specialization, alloc: Any) -> Scalar | None:
    """The scalar every element of a (compiled) output allocation holds, or None if it isn't uniform.

    Traces the allocation and follows values that are the same everywhere: a broadcast scalar,
    elementwise ops on such values, reshapes of them. So `jnp.zeros`, `jnp.full(s, -inf)` and
    `jnp.zeros(s) + 1` all count, however the code spells them.
    """
    try:
        closed = jax.make_jaxpr(lambda *xs: eval(alloc, spec.namespace(xs)))(*spec.specs)
    except Exception:
        return None
    jaxpr = closed.jaxpr
    uniform = {
        v: np.asarray(c).reshape(()) for v, c in zip(jaxpr.constvars, closed.consts, strict=True) if np.size(c) == 1
    }

    def read(atom):
        if isinstance(atom, Literal):
            return np.asarray(atom.val)
        return uniform[atom]  # KeyError: depends on something that isn't uniform

    try:
        for eqn in jaxpr.eqns:
            name, ins = eqn.primitive.name, eqn.invars
            if name == "empty":  # unspecified contents: any constant is a valid choice
                val = np.zeros((), eqn.outvars[0].aval.dtype)
            elif name in _SHAPE_ONLY_OPS:
                val = read(ins[0])
            elif name in _ELEMENTWISE_OPS:
                val = np.asarray(eqn.primitive.bind(*[jnp.asarray(read(a)) for a in ins], **eqn.params))
            else:
                return None
            uniform[eqn.outvars[0]] = val.reshape(())
        return read(jaxpr.outvars[0]).item()
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- masks


def _reads_inputs(nest: LoopNest, cond: ast.expr) -> bool:
    return any(isinstance(n, ast.Name) and n.id in nest.params for n in ast.walk(cond))


def _vectorize(cond: ast.expr) -> ast.expr:
    """Rewrite a scalar mask condition so it evaluates elementwise on arrays of loop coordinates.

    `and`/`or`/`not` and chained comparisons become `xp.logical_and`/`xp.logical_or`/`xp.logical_not`, and
    `a if c else b` becomes `xp.where`; `xp` is numpy at trace time or jax.numpy on device.
    """

    def call(fn: str, *args: ast.expr) -> ast.expr:
        return ast.Call(ast.Attribute(ast.Name("xp", ast.Load()), fn, ast.Load()), list(args), [])

    class Vectorizer(ast.NodeTransformer):
        def visit_BoolOp(self, n: ast.BoolOp) -> ast.AST:
            values = [self.visit(v) for v in n.values]
            fn = "logical_and" if isinstance(n.op, ast.And) else "logical_or"
            out = values[0]
            for v in values[1:]:
                out = call(fn, out, v)
            return out

        def visit_UnaryOp(self, n: ast.UnaryOp) -> ast.AST:
            if isinstance(n.op, ast.Not):
                return call("logical_not", self.visit(n.operand))
            return self.generic_visit(n)

        def visit_Compare(self, n: ast.Compare) -> ast.AST:
            left, parts = self.visit(n.left), []
            for op, right in zip(n.ops, n.comparators, strict=True):
                right = self.visit(right)
                parts.append(ast.Compare(left, [op], [right]))
                left = right
            out = parts[0]
            for p in parts[1:]:
                out = call("logical_and", out, p)
            return out

        def visit_IfExp(self, n: ast.IfExp) -> ast.AST:
            return call("where", self.visit(n.test), self.visit(n.body), self.visit(n.orelse))

    out = Vectorizer().visit(copy.deepcopy(cond))
    return ast.fix_missing_locations(ast.copy_location(out, cond))


def evaluate_mask(
    spec: Specialization, cond: ast.expr, coords: dict[str, Any], xp: Any, inputs: dict[str, Any] | None = None
) -> Any:
    """Evaluate a mask for arrays of loop coordinates (numpy at trace time, jax.numpy on device)."""
    env = {**spec.consts, **(inputs or {}), **coords, "xp": xp}
    return eval(spec.code.compile(_vectorize(cond), "eval"), env)


def tile_coords(spec: Specialization, xp: Any, column: bool = False) -> dict[str, Any]:
    """Parallel-loop values for each candidate tile: arrays of shape (tiles,), or (tiles, 1) with `column`."""
    coords = {}
    for n, lb in enumerate(spec.parallel):
        values = np.array([t[n] for t in spec.tiles], np.int64).reshape((-1, 1) if column else (-1,)) * lb.step
        coords[lb.var] = xp.asarray(values)
    return coords


def phase_coords(phase: Phase) -> dict[str, np.ndarray]:
    """Reduction-loop values for each step of a phase, in order (row-major over its loops): shape (1, steps)."""
    grids = np.meshgrid(*[np.arange(lb.count) for lb in phase.loops], indexing="ij")
    return {lb.var: (lb.start + g.reshape(1, -1) * lb.step) for lb, g in zip(phase.loops, grids, strict=True)}


def _tile_mask(spec: Specialization) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], bool]:
    """(tiles the kernel computes, tiles a static mask skips, whether the mask is decided on device)."""
    nest = spec.nest
    tiles = list(itertools.product(*(lb.blocks for lb in spec.parallel)))
    if nest.guard is None:
        return tiles, [], False
    reduction_vars = {lp.var for r in nest.reductions for lp in r.loops}
    for n in ast.walk(nest.guard):
        if isinstance(n, ast.Name) and n.id in reduction_vars:
            raise nest.error(nest.guard_node, f"the tile mask can't depend on the reduction loop `{n.id}`")
    if _reads_inputs(nest, nest.guard):
        return tiles, [], True  # every tile is a candidate; the device decides
    spec.tiles = tiles  # tile_coords reads them
    keep = _eval_static(spec, nest.guard, tile_coords(spec, np), nest.guard_node, "tile mask")
    keep = np.broadcast_to(np.asarray(keep, bool), (len(tiles),))
    kept = [t for t, k in zip(tiles, keep, strict=True) if k]
    if not kept:
        raise nest.error(nest.guard_node, "the tile mask skips every tile")
    return kept, [t for t, k in zip(tiles, keep, strict=True) if not k], False


def _phase_masks(spec: Specialization) -> tuple[list[np.ndarray | None], list[bool]]:
    """Per phase: its static (tile x step) mask, or None; and whether its mask is decided on device."""
    nest = spec.nest
    masks: list[np.ndarray | None] = []
    runtime: list[bool] = []
    for phase in spec.phases:
        r = phase.reduction
        if r.guard is None:
            masks.append(None)
            runtime.append(False)
        elif _reads_inputs(nest, r.guard):
            masks.append(None)
            runtime.append(True)
        else:
            coords = {**tile_coords(spec, np, column=True), **phase_coords(phase)}
            keep = _eval_static(spec, r.guard, coords, r.guard_node, "reduction mask")
            masks.append(np.broadcast_to(np.asarray(keep, bool), (len(spec.tiles), phase.steps)).copy())
            runtime.append(False)
    return masks, runtime


def _eval_static(spec: Specialization, cond: ast.expr, coords: dict[str, Any], node: ast.stmt | None, what: str) -> Any:
    try:
        return evaluate_mask(spec, cond, coords, np)
    except Exception as e:
        raise spec.nest.error(
            node, f"can't evaluate the {what} `{ast.unparse(cond)}` at trace time ({type(e).__name__}: {e})"
        ) from None


# --------------------------------------------------------------------------- state


def _reads(s: ast.stmt) -> set[str]:
    names = {n.id for n in ast.walk(s.value) if isinstance(n, ast.Name)}
    if isinstance(s, ast.AugAssign):
        names.add(s.target.id)
    return names


def _needs_state(spec: Specialization) -> set[str]:
    """Tile values read in a unit other than the one that last wrote them."""
    nest = spec.nest
    tile_names = {assigned_name(s) for s in nest.tile_statements()}
    needs: set[str] = set()
    for unit in spec.units:
        written: set[str] = set()
        for s in unit.statements:
            needs |= (_reads(s) & tile_names) - written  # read before written here: comes from elsewhere
            written.add(assigned_name(s))
    final_reads = {n.id for st in nest.stores for n in ast.walk(st.value) if isinstance(n, ast.Name)}
    return needs | (final_reads & tile_names) - {assigned_name(s) for s in spec.units[-1].statements}


def _trace_tile(spec: Specialization) -> list[tuple[ast.stmt, jax.ShapeDtypeStruct]]:
    """One pass through a tile (one iteration of every reduction): each assignment and the type it produces.

    `@` is traced as the kernel computes it (an MXU dot accumulating in f32 / i32), so the types match the
    generated code's.
    """
    nest = spec.nest
    statements = [s for u in spec.units for s in u.statements]
    compiled = [spec.code.compile([mxu.rewrite_matmuls(s)], "exec") for s in statements]
    helpers = mxu.helpers("highest")

    def run(*arrays):
        ns = spec.namespace(arrays, **helpers)
        ns.update({v: lb.start for v, lb in spec.loops.items()})
        values = []
        for s, code in zip(statements, compiled, strict=True):
            exec(code, ns)
            values.append(jnp.asarray(ns[assigned_name(s)]))
        return values

    try:
        traced = jax.eval_shape(run, *spec.specs)
    except Exception as e:
        raise nest.error(nest.parallel[0].node, f"couldn't trace the tile body ({type(e).__name__}: {e})") from None
    return list(zip(statements, traced, strict=True))


def _state(spec: Specialization) -> tuple[dict[str, StateVar], dict[int, np.dtype]]:
    """The state variables, the stored type of each, and the assignments whose values must be cast to it."""
    nest = spec.nest
    needs = _needs_state(spec)
    if not needs:
        return {}, {}
    body_stmts = {id(s) for u in spec.units if u.kind == "body" for s in u.statements}
    by_name: dict[str, list[tuple[ast.stmt, jax.ShapeDtypeStruct]]] = {}
    for s, aval in _trace_tile(spec):
        by_name.setdefault(assigned_name(s), []).append((s, aval))
    state: dict[str, StateVar] = {}
    casts: dict[int, np.dtype] = {}
    for name in sorted(needs):
        entries = by_name[name]
        shapes = list(dict.fromkeys(tuple(a.shape) for _, a in entries))
        if len(shapes) != 1:
            raise nest.error(entries[0][0], f"`{name}` changes shape ({' -> '.join(map(str, shapes))}) between steps")
        dtype = np.dtype(jnp.result_type(*[a.dtype for _, a in entries]))
        accumulates = any(id(s) in body_stmts for s, _ in entries)
        if accumulates and jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < 4:
            dtype = np.dtype(jnp.float32)  # narrow float accumulators live in f32: the MXU produces f32 anyway
        shape = shapes[0]
        state[name] = StateVar(name, spec.names.fresh(f"{name}_ref"), jax.ShapeDtypeStruct(shape, dtype), shape == ())
        casts.update({id(s): dtype for s, aval in entries if np.dtype(aval.dtype) != dtype})
    return state, casts
