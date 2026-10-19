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
from jax._src.lax.lax import _convert_element_type  # the only way to set a weak type
from jax.experimental import checkify
from jax.extend.core import Literal

from . import hardware, mxu
from .errors import AutoKernelError
from .frontend import Loop, LoopNest, Reduction, assigned_name, reads
from .views import ViewError, ViewMap, Window

__all__ = [
    "Accumulator",
    "Block",
    "Environment",
    "LoopBounds",
    "Operand",
    "OutputSpec",
    "Phase",
    "Specialization",
    "StateVar",
    "Unit",
    "check_indices",
    "environment",
    "evaluate_mask",
    "is_scalar_spec",
    "specialize",
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
    the kernel sees. A dim indexed by a *runtime* integer (an expression of the kernel's scalar inputs,
    `w[layer, ...]`) has that expression's source in `dynamic[d]`; its index is read at run time.
    """

    shape: tuple[int, ...]
    loops: tuple[str | None, ...]
    const: tuple[int, ...]
    squeezed: tuple[bool, ...]
    text: str  # the index as the user wrote it, for comments and errors
    dynamic: tuple[str | None, ...] = ()  # per dim: a runtime index expression, or None (empty: none at all)

    @property
    def key(self) -> tuple:
        """Everything that decides which HBM window the access touches and how the kernel sees it."""
        return (self.shape, self.loops, self.const, self.squeezed, self.dynamic)

    def dynamic_index(self, d: int) -> str | None:
        return self.dynamic[d] if self.dynamic else None

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
    # When the input is a view of another array (`views`): the window of that base array the tile is, which
    # the kernel copies instead (it is given the base array).
    window: Window | None = None


@dataclass(frozen=True)
class OutputSpec:
    var: str
    ref: str
    out: jax.ShapeDtypeStruct
    block: Block
    alloc: ast.expr
    fill: Scalar | None  # the allocation's value when every element is the same known constant
    partial: bool = False  # the tiles don't cover it: the rest keeps the allocation's values (an input)


@dataclass(frozen=True)
class StateVar:
    """A tile value that survives between steps."""

    name: str
    ref: str
    aval: jax.ShapeDtypeStruct  # its type: what every value assigned to it promotes to (weakly typed if they all are)
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


@dataclass(frozen=True)
class Accumulator:
    """`var = <zeros>` in the segment before a reduction, then `var += expr` as the reduction body's first use.

    In a tile of one step the kernel computes `var = expr` instead of adding `expr` to zeros: Mosaic doesn't
    fold an add into a matmul's accumulator. Only there: splitting a multi-step reduction's first step off
    (store `expr`, add it on later steps) measured slower on a v5e, since Mosaic then materialises the
    matmul's result before the branch, where `acc + a @ b` streams the add.
    """

    var: str
    phase: int
    init: ast.stmt  # `var = <zeros>`
    update: ast.stmt  # `var += expr`
    expr: ast.expr


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
    # Inputs that are 0-d numbers: runtime scalars (read from SMEM), usable as values; integers also as indices.
    scalars: list[str] = field(default_factory=list)
    views: dict[str, ViewMap] = field(default_factory=dict)  # inputs that are views of a base array
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
    broadcast_stores: set[str] = field(default_factory=set)  # outputs whose stored value is broadcast to the tile
    # Every step is the first and last of its tile (no reduction, or one unmasked reduction step): the whole tile
    # runs in one step, so nothing has to survive between steps.
    single_step: bool = False
    accumulators: list[Accumulator] = field(default_factory=list)
    avals: dict[int, jax.ShapeDtypeStruct] = field(default_factory=dict)  # id(statement) -> the type it assigns
    # Values one unit of a single-step tile passes to the next as plain locals (otherwise state), and their types.
    carried: dict[str, jax.ShapeDtypeStruct] = field(default_factory=dict)
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

    def finished_loops(self, before: int) -> dict[str, int]:
        """The variables of the reduction loops before phase `before`, which have finished by then: as in
        Python, each keeps its last value (whatever the masks skipped)."""
        return {lb.var: lb.stop - lb.step for phase in self.phases[:before] for lb in phase.loops}

    def unit_code(self, node: ast.AST, unit: Unit | None) -> ast.AST:
        """A copy of `node` (from `unit`, or None for the output stores) with `@` as an MXU matmul and the
        variables of finished reduction loops as their final values."""
        done = self.finished_loops(len(self.phases) if unit is None else unit.index)

        class Finished(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in done and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(ast.Constant(done[n.id]), n)
                return n

        return ast.fix_missing_locations(Finished().visit(mxu.rewrite_matmuls(node)))

    def namespace(self, arrays: Sequence[Any] | None = None, **extra: Any) -> dict[str, Any]:
        """What the user's code can see: constants, the inputs (as `arrays`, or their specs) and `extra`."""
        inputs = self.specs if arrays is None else arrays
        return {**self.consts, **dict(zip(self.nest.params, inputs, strict=True)), **extra}

    @property
    def arrays(self) -> list[str]:
        """The inputs that are arrays in HBM (every input but the scalars)."""
        return [p for p in self.nest.params if p not in self.scalars]

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

    def copy(self) -> Namer:
        """A namer that has taken the same names, and hands out the same ones from here on."""
        return Namer(self._taken)


class UserCode:
    """Compiles fragments of the user's function so tracebacks point at their file and lines."""

    def __init__(self, nest: LoopNest, filename: str):
        self.nest, self.filename = nest, filename
        # Names the loop nest binds: at trace time any global of the same name is a different value.
        self.rebound = {lp.var for lp in nest.loops} | {assigned_name(s) for s in nest.tile_statements()}

    def compile(self, node: ast.AST | list[ast.stmt], mode: str):
        node = copy.deepcopy(node)
        wrapped = ast.Expression(node) if mode == "eval" else ast.Module(body=node, type_ignores=[])
        ast.increment_lineno(wrapped, self.nest.first_line - 1)
        ast.fix_missing_locations(wrapped)
        return compile(wrapped, self.filename, mode)

    def run_prologue(self, env: dict[str, Any]) -> None:
        """Run the prologue in `env`, one statement at a time so a failure names its line."""
        for s in self.nest.prologue:
            try:
                exec(self.compile([s], "exec"), env)
            except Exception as e:
                raise self.nest.error(s, f"the prologue failed ({type(e).__name__}: {e})") from None

    def eval_int(self, node: ast.expr, what: str, env: dict[str, Any]) -> int:
        """Evaluate `node`, one of the nest's own expressions, to an int."""
        key = ("eval_int", id(node), self.filename)
        if key not in self.nest.cache:
            self.nest.cache[key] = (sorted(reads(node) & self.rebound), self.compile(node, "eval"))
        rebound, code = self.nest.cache[key]
        if rebound:
            hint = ""
            if rebound[0] in {lp.var for lp in self.nest.loops}:
                hint = "; to skip part of a loop depending on another loop, use a mask (`if k > i: continue`)"
            raise self.nest.error(
                node,
                f"{what} `{ast.unparse(node)}` is evaluated once, at trace time, so it can't use `{rebound[0]}`, "
                f"which the loop nest assigns{hint}",
            )
        try:
            value = eval(code, dict(env))
        except Exception as e:
            raise self.nest.error(
                node, f"can't evaluate {what} `{ast.unparse(node)}` at trace time ({type(e).__name__}: {e})"
            ) from None
        if not isinstance(value, (int, np.integer)):
            raise self.nest.error(node, f"{what} `{ast.unparse(node)}` must be a Python int, got {value!r}")
        return int(value)


@dataclass(frozen=True, eq=False)
class Environment:
    """What a loop nest's code can see besides its inputs, for a nest that isn't a whole Python function
    (one nest of an `autokernel` program): names mapped to values, and the file its source came from."""

    names: dict[str, Any]
    filename: str
    # Inputs that are views of another array (`views.ViewMap`): the kernel is given that base array and
    # copies the windows of it that the tiles of the view are.
    views: dict[str, ViewMap] = field(default_factory=dict)


def environment(fn: Callable | Environment) -> Environment:
    """The names a kernel's code can see and its source file, from its function or as given."""
    if isinstance(fn, Environment):
        return fn
    return Environment(user_globals(fn), inspect.getsourcefile(fn) or "<autokernel>")


def user_globals(fn: Callable | Environment) -> dict[str, Any]:
    """The names `fn`'s body can see besides its parameters: module globals and closure cells."""
    if isinstance(fn, Environment):
        return dict(fn.names)
    env = dict(fn.__globals__)
    env.update(inspect.getclosurevars(fn).nonlocals)
    return env


# --------------------------------------------------------------------------- entry point


def specialize(
    nest: LoopNest,
    fn: Callable | Environment,
    specs: Sequence[jax.ShapeDtypeStruct],
    overrides: dict[str, Any] | None = None,
) -> Specialization:
    """Resolve `nest` for inputs of the given shapes and dtypes, validating it along the way.

    `fn` is the function the nest was parsed from, or the `Environment` its code runs in. `overrides`
    replaces module or closure constants (tile sizes being tuned, for example).
    """
    specs = list(specs)
    if len(specs) != len(nest.params):
        raise AutoKernelError(f"{nest.fn_name} takes {len(nest.params)} arrays, got {len(specs)}")
    # 0-d numeric inputs are runtime scalars, read from SMEM as 32-bit words. Integers are traced weakly
    # typed, as the Python ints they usually stand for, and the kernel reads them that way too.
    scalars = [p for p, s in zip(nest.params, specs, strict=True) if is_scalar_spec(s)]
    specs = [scalar_aval(s) if p in scalars else s for p, s in zip(nest.params, specs, strict=True)]
    source = environment(fn)
    code = UserCode(nest, source.filename)

    # Names the user's code can see: module globals, closure cells, overrides, then the prologue.
    # Inputs are bound to ShapeDtypeStructs, so the prologue can only use their shapes and dtypes.
    env = dict(source.names)
    env.update(overrides or {})
    env.update(zip(nest.params, specs, strict=True))
    code.run_prologue(env)
    consts = {k: v for k, v in env.items() if k not in nest.params}

    loops = {lp.var: _bounds(lp, code, env, nest) for lp in nest.loops}
    # Every name in the function, wherever it appears (a loop variable read only by a store, say).
    user_names = {n.id for n in ast.walk(nest.fdef) if isinstance(n, ast.Name)}
    user_names |= {a.arg for a in ast.walk(nest.fdef) if isinstance(a, ast.arg)}
    spec = Specialization(
        nest=nest,
        specs=specs,
        filename=code.filename,
        consts=consts,
        loops=loops,
        parallel=[loops[lp.var] for lp in nest.parallel],
        phases=[Phase(p, tuple(loops[lp.var] for lp in r.loops), r) for p, r in enumerate(nest.reductions)],
        units=_units(nest),
        names=Namer(user_names | set(consts)),
        code=code,
        overrides=dict(overrides or {}),
        scalars=scalars,
        views={p: v for p, v in source.views.items() if p in nest.params and p not in scalars},
    )
    spec.tiles, spec.constant_tiles, spec.tile_mask_runtime = _tile_mask(spec)
    spec.phase_masks, spec.phase_runtime = _phase_masks(spec)
    spec.outputs = _outputs(spec)
    spec.single_step = _single_step(spec)
    spec.operands = _collect_operands(spec)
    spec.state, spec.casts, store_avals, spec.avals = _state(spec)
    if spec.single_step:
        last = {assigned_name(s): spec.avals[id(s)] for u in spec.units for s in u.statements}
        spec.carried = {name: last[name] for name in sorted(_crossing(spec))}
    spec.broadcast_stores = _check_stores(spec, store_avals)
    spec.accumulators = _accumulators(spec)
    return spec


def _single_step(spec: Specialization) -> bool:
    """Does every tile run in one step: no reduction, or one reduction of one step that no mask can skip?"""
    if spec.runtime_masks or any(m is not None for m in spec.phase_masks):
        return False
    return not spec.phases or (len(spec.phases) == 1 and spec.phases[0].steps == 1)


def _accumulators(spec: Specialization) -> list[Accumulator]:
    """The reductions' zero-initialised accumulators (see `Accumulator`), where every step of the reduction runs."""
    found = []
    if not spec.single_step:  # see `Accumulator`
        return found
    segments = [u for u in spec.units if u.kind == "segment"]
    bodies = [u for u in spec.units if u.kind == "body"]
    for p in range(len(spec.phases)):
        if spec.phase_masks[p] is not None or spec.phase_runtime[p]:
            continue  # a skipped first step (or an empty phase) would leave the zeros unwritten
        seg, body = segments[p].statements, bodies[p].statements
        for n, init in enumerate(seg):
            var = assigned_name(init)
            later = seg[n + 1 :]
            if not isinstance(init, ast.Assign) or any(var in reads(s) or assigned_name(s) == var for s in later):
                continue  # not the segment's last word on `var`
            update = next((s for s in body if var in reads(s) or assigned_name(s) == var), None)
            expr = _added_to(update, var)
            if expr is None or var in reads(expr) or not _is_zeros(spec, init.value):
                continue
            found.append(Accumulator(var, p, init, update, expr))
    return found


def _added_to(s: ast.stmt | None, var: str) -> ast.expr | None:
    """`e` when `s` is `var += e`, `var = var + e` or `var = e + var`."""
    if isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add) and s.target.id == var:
        return s.value
    if (
        isinstance(s, ast.Assign)
        and assigned_name(s) == var
        and isinstance(s.value, ast.BinOp)
        and isinstance(s.value.op, ast.Add)
    ):
        left, right = s.value.left, s.value.right
        if isinstance(left, ast.Name) and left.id == var:
            return right
        if isinstance(right, ast.Name) and right.id == var:
            return left
    return None


def _is_zeros(spec: Specialization, value: ast.expr) -> bool:
    """Is `value` (a tile statement's expression) an array of zeros, whatever the code evaluating it reads?"""
    if any(name in spec.loops or name in spec.nest.params for name in reads(value)):
        return False
    try:
        fill = _constant_fill(spec, spec.code.compile(value, "eval"))
    except Exception:
        return False
    return fill is not None and not isinstance(fill, bool) and fill == 0


def is_scalar_spec(s: Any) -> bool:
    """Is an input of this shape and dtype a runtime scalar (a 0-d integer or float)?"""
    return tuple(s.shape) == () and (jnp.issubdtype(s.dtype, jnp.integer) or jnp.issubdtype(s.dtype, jnp.floating))


def scalar_aval(s: Any) -> jax.ShapeDtypeStruct:
    """How the kernel reads a runtime scalar of this type: a weakly typed int32 for any integer (as a Python
    int); an f32 for an f32, weakly typed if it is; and a weakly typed f32 for narrower floats (a bf16 is read
    as f32, then mixes with tiles like a Python float, which is also what a bf16 does with float tiles)."""
    if jnp.issubdtype(s.dtype, jnp.integer):
        return jax.ShapeDtypeStruct((), jnp.int32, weak_type=True)
    weak = bool(getattr(s, "weak_type", False)) or np.dtype(s.dtype) != np.float32
    return jax.ShapeDtypeStruct((), jnp.float32, weak_type=weak)


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
    dynamic: str | None = None


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
    dynamic = tuple(r.dynamic for r in resolved)
    return Block(
        tuple(r.size for r in resolved),
        tuple(r.loop for r in resolved),
        tuple(r.const for r in resolved),
        tuple(r.squeezed for r in resolved),
        text,
        dynamic if any(dynamic) else (),
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
        names = {n.id for n in ast.walk(sl) if isinstance(n, ast.Name)}
        if names & set(spec.loops):
            raise nest.error(
                index, f"{what}: an integer index must be a loop variable or a constant, not `{ast.unparse(sl)}`"
            )
        if names & set(spec.scalars):
            _check_runtime_index(spec, index, what, sl)
            return _Dim(1, None, 0, True, ast.unparse(sl))
        c = spec.code.eval_int(sl, "index", env)
        if not -size <= c < size:
            raise nest.error(index, f"{what}: index {c} in dim {d} is outside the array (size {size})")
        return _Dim(1, None, c % size, True)  # a negative index counts from the end, as in Python
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
    lo = None if sl.lower is None else spec.code.eval_int(sl.lower, "slice bound", env)
    hi = None if sl.upper is None else spec.code.eval_int(sl.upper, "slice bound", env)
    lo, hi, _ = slice(lo, hi).indices(size)  # Python's meaning: negative bounds count from the end, and clip
    t = hi - lo
    if t <= 0 or lo % t:
        raise nest.error(
            index, f"{what}: constant slice {lo}:{hi} in dim {d} must be a whole aligned block inside the array"
        )
    return _Dim(t, None, lo // t, False)


def _check_runtime_index(spec: Specialization, index: ast.expr, what: str, sl: ast.expr) -> None:
    """A runtime index (an expression of the scalar inputs) must be one integer."""
    code = spec.code.compile(sl, "eval")
    try:
        out = jax.eval_shape(lambda *xs: eval(code, spec.namespace(xs)), *spec.specs)
    except Exception as e:
        raise spec.nest.error(
            index, f"{what}: can't evaluate the runtime index `{ast.unparse(sl)}` ({type(e).__name__}: {e})"
        ) from None
    if not (hasattr(out, "shape") and out.shape == () and jnp.issubdtype(out.dtype, jnp.integer)):
        raise spec.nest.error(index, f"{what}: the runtime index `{ast.unparse(sl)}` must be one integer, got {out}")


def check_indices(spec: Specialization, values: dict[str, Any]) -> None:
    """Check the runtime indices of every input tile against the arrays: Python's rules, so an index may be
    negative (counting from the end) but must lie inside the array.

    A known value out of range raises AutoKernelError. A traced one can't be checked now: the kernel clamps
    it into the array (it never reads outside it), and the check is staged as a `jax.experimental.checkify`
    check, an error when the program runs under `checkify.checkify` (and nothing otherwise)."""
    env = {**spec.consts, **values}
    for op in spec.operands:
        for d, expr in enumerate(op.block.dynamic):
            if expr is None or not set(values) >= _names_in(expr) & set(spec.scalars):
                continue
            value = eval(expr, dict(env))
            size = op.full_shape[d]
            what = f"{spec.nest.fn_name}: `{op.param}[{op.block.text}]`: index {{}} (`{expr}`) in dim {d} is outside"
            if isinstance(value, jax.core.Tracer):
                msg = what.replace("{", "{{").replace("}", "}}").replace("{{}}", "{}")
                msg += f" the array (size {size}); the kernel clamped it into the array"
                value = jnp.asarray(value, jnp.int32)
                checkify.check((value >= -size) & (value < size), msg, value, debug=True)
            elif not -size <= int(value) < size:
                raise AutoKernelError(what.format(int(value)) + f" the array (size {size})")


def _names_in(expr: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)}


def _loop_dim(
    spec: Specialization, index: ast.expr, what: str, d: int, var: str, size: int, allowed: set[str]
) -> LoopBounds:
    """Check that loop `var` may tile dim `d` here and fits in it."""
    if var not in allowed:
        raise spec.nest.error(index, f"{what}: can't be tiled by loop `{var}` here")
    lb = spec.loops[var]
    if lb.start < 0:  # Python would count a negative index from the end of the dim
        raise spec.nest.error(
            index, f"{what}: loop `{var}` starts at {lb.start}; a loop indexing a dim can't go negative"
        )
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


def _input_reads(nest: LoopNest, nodes: Sequence[ast.AST], scalars: Sequence[str] = ()) -> Iterator[ast.Subscript]:
    """Every tile slice of an input in `nodes`, in source order. Rejects other reads of inputs/outputs.

    `scalars` (runtime scalar inputs) are read as plain values."""
    found: list[ast.Subscript] = []

    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in scalars:
                raise nest.error(node, f"`{node.value.id}` is a scalar input (0-d); it can't be indexed")
            if isinstance(node.value, ast.Name) and node.value.id in nest.params:
                found.append(node)
                self.visit(node.slice)
            else:
                self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in scalars:
                return
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


def _view_window(spec: Specialization, param: str, block: Block) -> Window | None:
    """The window of its base array that a tile of view input `param` is (None if `param` isn't a view)."""
    vmap = spec.views.get(param)
    if vmap is None:
        return None
    nest = spec.nest
    # A mask or an allocation reading the view's values runs outside the kernel, on the view itself.
    guards = [nest.guard, *(r.guard for r in nest.reductions)]
    if any(g is not None and param in reads(g) and _reads_inputs(nest, g) for g in guards):
        raise ViewError(param, f"{nest.fn_name}: a mask reads `{param}`, a view: it needs the view's values")
    if any(param in reads(o.alloc) for o in nest.outputs):
        raise ViewError(param, f"{nest.fn_name}: an output allocation reads `{param}`, a view")
    lanes = 128

    def check(what: str, shape: tuple[int, ...], full: tuple[int, ...]) -> list[str]:
        return hardware.check_block(what, shape, full, vmap.dtype, lanes)

    return vmap.window(param, block, block.text, lanes, check)


def _collect_operands(spec: Specialization) -> list[Operand]:
    nest = spec.nest
    scalars = spec.scalars
    reads = [(node, unit.phase) for unit in spec.units for node in _input_reads(nest, unit.statements, scalars)]
    reads += [(node, None) for node in _input_reads(nest, [st.value for st in nest.stores], scalars)]
    operands: list[Operand] = []
    for node, phase in reads:
        param, block = node.value.id, spec.resolve_input(node, phase)
        op = next((o for o in operands if (o.param, o.block.key) == (param, block.key)), None)
        if op is None:
            # A one-step reduction's tiles are per-tile inputs: fetched a tile ahead into two slots rather than
            # streamed through the reduction's ring (on a v5e, 2% faster for a single-step matmul).
            tiled_by_phase = (
                not spec.single_step
                and phase is not None
                and any(lb.var in block.loops for lb in spec.phases[phase].loops)
            )
            existing = sum(o.param == param for o in operands)
            ref = spec.names.fresh(f"{param}_ref" if existing == 0 else f"{param}_ref{existing}")
            op = Operand(
                param,
                ref,
                block,
                spec.input_shape(param),
                spec.input_dtype(param),
                phase if tiled_by_phase else None,
                _view_window(spec, param, block),
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
        if block.dynamic:
            raise nest.error(store.node, f"`{output.var}`'s tiles can't be placed by a runtime index")
        tiled_by = [v for v in block.loops if v]
        for var in sorted(par_vars - set(tiled_by)):
            raise nest.error(
                store.node,
                f"parallel loop `{var}` must tile one of `{output.var}`'s dims, otherwise tiles race on the same "
                "output block",
            )
        partial = False
        for var in sorted({v for v in tiled_by if tiled_by.count(v) > 1}):
            if output.keeps:  # an input's values: the elements off the block diagonal keep them
                partial = True
                break
            raise nest.error(
                store.node,
                f"parallel loop `{var}` tiles two dims of `{output.var}`, so the tiles only cover its block "
                "diagonal; every output element must be produced by the kernel",
            )
        for d, (var, n, t) in enumerate(zip(block.loops, out.shape, block.shape, strict=True)):
            covered = (spec.loops[var].start, spec.loops[var].stop) == (0, n) if var else t == n
            if not covered and output.keeps:
                partial = True
            elif not covered:
                raise nest.error(
                    store.node,
                    f"the tiles don't write all of `{output.var}`'s dim {d} (size {n}); every output element must "
                    "be produced by the kernel",
                )
        ref = spec.names.fresh(f"{output.var}_ref")
        fill = _constant_fill(spec, alloc)
        outputs.append(OutputSpec(output.var, ref, out, block, output.alloc, fill, partial))
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


_STATIC_ATTRS = frozenset({"shape", "dtype", "ndim", "size"})  # what an input's spec tells without its values
_ELEMENTWISE_BUILTINS = {"max": "maximum", "min": "minimum", "abs": "abs"}


def _reads_inputs(nest: LoopNest, cond: ast.expr) -> bool:
    """Does a mask read input values (not just an input's `.shape`, `.dtype`, ...)?"""
    static = {id(n.value) for n in ast.walk(cond) if isinstance(n, ast.Attribute) and n.attr in _STATIC_ATTRS}
    return any(isinstance(n, ast.Name) and n.id in nest.params and id(n) not in static for n in ast.walk(cond))


def _vectorize(cond: ast.expr) -> ast.expr:
    """Rewrite a scalar mask condition so it evaluates elementwise on arrays of loop coordinates.

    `xp` is numpy at trace time or jax.numpy on device. Each rewrite keeps Python's value, not just its
    truth: `a and b` / `a or b` become `xp.where(a, b, a)` / `xp.where(a, a, b)`, `not` becomes
    `xp.logical_not`, `~` inverts integers (`~True` is -2 in Python, not False), chained comparisons are
    joined with `xp.logical_and`, `a if c else b` becomes `xp.where`, and the builtins `max`, `min` and
    `abs` their elementwise forms, `bool(a)` a test of a one-element array, and `a in (b, c)` a test
    against each element of a literal tuple, list or set.
    """

    def call(fn: str, *args: ast.expr) -> ast.expr:
        return ast.Call(ast.Attribute(ast.Name("xp", ast.Load()), fn, ast.Load()), list(args), [])

    class Vectorizer(ast.NodeTransformer):
        def visit_BoolOp(self, n: ast.BoolOp) -> ast.AST:
            values = [self.visit(v) for v in n.values]
            out = values[0]
            for v in values[1:]:  # Python's value: the first falsy (`and`) or truthy (`or`) operand
                held = copy.deepcopy(out)
                out = call("where", out, v, held) if isinstance(n.op, ast.And) else call("where", out, held, v)
            return out

        def visit_UnaryOp(self, n: ast.UnaryOp) -> ast.AST:
            if isinstance(n.op, ast.Not):
                return call("logical_not", self.visit(n.operand))
            if isinstance(n.op, ast.Invert):  # on integers, as Python does with bools (`+ 0` makes them ints)
                as_int = ast.BinOp(call("asarray", self.visit(n.operand)), ast.Add(), ast.Constant(0))
                return ast.UnaryOp(ast.Invert(), as_int)
            return self.generic_visit(n)

        def visit_Call(self, n: ast.Call) -> ast.AST:
            self.generic_visit(n)
            fn = n.func.id if isinstance(n.func, ast.Name) else None
            if (
                fn not in {*_ELEMENTWISE_BUILTINS, "bool"}
                or n.keywords
                or not n.args
                or any(isinstance(a, ast.Starred) for a in n.args)
            ):
                return n
            if fn == "bool":  # the truth of a one-element array
                one = call("reshape", call("asarray", *n.args), ast.Tuple([], ast.Load()))
                return ast.Compare(one, [ast.NotEq()], [ast.Constant(0)]) if len(n.args) == 1 else n
            if fn == "abs":
                return call("abs", *n.args) if len(n.args) == 1 else n
            if len(n.args) < 2:  # max(iterable)
                return n
            out = n.args[0]
            for a in n.args[1:]:
                out = call(_ELEMENTWISE_BUILTINS[fn], out, a)
            return out

        def visit_Compare(self, n: ast.Compare) -> ast.AST:
            left, parts = self.visit(n.left), []
            for op, right in zip(n.ops, n.comparators, strict=True):
                right = self.visit(right)
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                    # `a in (b, c)`: `a == b or a == c`
                    part = ast.Constant(False)
                    for e in right.elts:
                        part = call("logical_or", part, ast.Compare(left, [ast.Eq()], [e]))
                    parts.append(call("logical_not", part) if isinstance(op, ast.NotIn) else part)
                else:
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


def _mask(spec: Specialization, phase: int | None) -> tuple[ast.expr, ast.stmt, str]:
    """The tile mask (`phase` None) or a phase's mask: (condition, its `if` statement, what to call it)."""
    if phase is None:
        return spec.nest.guard, spec.nest.guard_node, "tile mask"
    r = spec.phases[phase].reduction
    return r.guard, r.guard_node, "reduction mask"


def evaluate_mask(spec: Specialization, phase: int | None, inputs: dict[str, Any] | None = None) -> Any:
    """Whether the tile mask (`phase` None) keeps each candidate tile, shape (tiles,), or whether phase
    `phase`'s mask keeps each step of each tile, shape (tiles, steps).

    The mask is evaluated once per combination of the values of the loops it reads, then spread over the
    tiles and steps. Without `inputs` (a static mask), each evaluation is the user's Python on Python ints,
    so it means exactly what it says. A mask reading `inputs` runs on device: jax.numpy, vectorised over the
    combinations with `jax.vmap`, with its Python operators rewritten into their array forms (`_vectorize`).
    """
    cond, _, _ = _mask(spec, phase)
    loops = list(spec.parallel) + (list(spec.phases[phase].loops) if phase is not None else [])
    used = reads(cond)
    read = [lb for lb in loops if lb.var in used]
    env = {**spec.namespace(), **(spec.finished_loops(phase) if phase is not None else {}), **(inputs or {})}
    grids = [range(lb.start, lb.stop, lb.step) for lb in read]
    if inputs is None:
        code = spec.code.compile(cond, "eval")
        values = np.empty([len(g) for g in grids], bool)
        for at in itertools.product(*(range(len(g)) for g in grids)):
            env.update((lb.var, g[n]) for lb, g, n in zip(read, grids, at, strict=True))
            values[at] = bool(eval(code, env))
        xp = np
    else:
        code = spec.code.compile(_vectorize(cond), "eval")
        env["xp"] = jnp

        def one(*coords):
            return jnp.asarray(eval(code, {**env, **dict(zip([lb.var for lb in read], coords, strict=True))}))

        mesh = np.meshgrid(*[np.asarray(g, np.int32) for g in grids], indexing="ij")
        values = jax.vmap(one)(*[jnp.asarray(m.reshape(-1)) for m in mesh]) if read else one()
        if values.shape != ((math.prod(len(g) for g in grids),) if read else ()):
            raise ValueError(
                f"the condition is an array of shape {values.shape[1:] if read else values.shape}, not one value"
            )
        values = values.astype(bool).reshape([len(g) for g in grids])
        xp = jnp
    # Spread over the tiles (and steps): the position of each tile's (and step's) value in each read loop.
    tiles, positions = len(spec.tiles), []
    steps = spec.phases[phase].steps if phase is not None else None
    counts = [lb.count for lb in spec.phases[phase].loops] if phase is not None else []
    for lb in read:
        if lb.var in {p.var for p in spec.parallel}:
            n = [p.var for p in spec.parallel].index(lb.var)
            pos = np.array([t[n] for t in spec.tiles], np.int64) - lb.start // lb.step
            positions.append(pos if phase is None else pos[:, None])
        else:
            k = [b.var for b in spec.phases[phase].loops].index(lb.var)
            positions.append(np.unravel_index(np.arange(steps), counts)[k][None, :])
    shape = (tiles,) if phase is None else (tiles, steps)
    return xp.broadcast_to(values[tuple(positions)] if read else values, shape)


def _check_mask(spec: Specialization, phase: int | None, runtime: bool) -> Any:
    """Evaluate a static mask, or check a runtime one on the inputs' specs; an AutoKernelError naming the line
    if it can't be evaluated."""
    cond, node, what = _mask(spec, phase)
    try:
        if not runtime:
            return evaluate_mask(spec, phase)
        params = spec.nest.params
        jax.eval_shape(lambda *xs: evaluate_mask(spec, phase, dict(zip(params, xs, strict=True))), *spec.specs)
        return None
    except _CONTROL_FLOW_ERRORS:
        raise spec.nest.error(
            node,
            f"the {what} `{ast.unparse(cond)}` reads inputs, so it runs on device for every step at once: Python "
            "control flow on the values it reads (`if`, `bool(...)`, `in` a non-literal container, ...) can't "
            "work there; use jnp functions",
        ) from None
    except Exception as e:
        when = "on device" if runtime else "at trace time"
        raise spec.nest.error(
            node, f"can't evaluate the {what} `{ast.unparse(cond)}` {when} ({type(e).__name__}: {e})"
        ) from None


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
    spec.tiles = tiles  # evaluate_mask reads them
    if _reads_inputs(nest, nest.guard):
        _check_mask(spec, None, runtime=True)
        return tiles, [], True  # every tile is a candidate; the device decides
    keep = _check_mask(spec, None, runtime=False)
    kept = [t for t, k in zip(tiles, keep, strict=True) if k]
    if not kept:
        raise nest.error(nest.guard_node, "the tile mask skips every tile")
    return kept, [t for t, k in zip(tiles, keep, strict=True) if not k], False


def _phase_masks(spec: Specialization) -> tuple[list[np.ndarray | None], list[bool]]:
    """Per phase: its static (tile x step) mask, or None; and whether its mask is decided on device."""
    masks: list[np.ndarray | None] = []
    runtime: list[bool] = []
    for phase in spec.phases:
        guard = phase.reduction.guard
        on_device = guard is not None and _reads_inputs(spec.nest, guard)
        keep = None if guard is None else _check_mask(spec, phase.index, on_device)
        masks.append(None if keep is None else np.array(keep, bool))
        runtime.append(on_device)
    return masks, runtime


# --------------------------------------------------------------------------- state


def _needs_state(spec: Specialization) -> set[str]:
    """The values that must survive between steps: none when a tile is one step, else `_crossing`."""
    return set() if spec.single_step else _crossing(spec)


def _crossing(spec: Specialization) -> set[str]:
    """Tile values read in a unit other than the one that last wrote them."""
    nest = spec.nest
    tile_names = {assigned_name(s) for s in nest.tile_statements()}
    needs: set[str] = set()
    for unit in spec.units:
        written: set[str] = set()
        for s in unit.statements:
            needs |= (reads(s) & tile_names) - written  # read before written here: comes from elsewhere
            written.add(assigned_name(s))
    final_reads = set().union(*(reads(st.value) for st in nest.stores))
    return needs | (final_reads & tile_names) - {assigned_name(s) for s in spec.units[-1].statements}


# Python control flow on a traced value: the loop variables and tile values are only known in the kernel.
_CONTROL_FLOW_ERRORS = (
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
    jax.errors.ConcretizationTypeError,
)


def _as_stored(value: Any, aval: jax.ShapeDtypeStruct) -> Any:
    """`value` as a state variable of type `aval` holds it (weak types included)."""
    return _convert_element_type(value, np.dtype(aval.dtype), weak_type=aval.weak_type)


def _trace_tile(
    spec: Specialization, state: dict[str, jax.ShapeDtypeStruct]
) -> tuple[list[tuple[ast.stmt, jax.ShapeDtypeStruct]], list[jax.ShapeDtypeStruct]]:
    """One pass through a tile (one step of every reduction): the type of each assignment and store value.

    The types are the kernel's: loop variables are weakly typed int32 scalars (Python ints, in the loops),
    a state variable reads back as it is stored (`state`, for those known so far), and `@` is an MXU dot
    accumulating in f32 / i32.
    """
    nest = spec.nest

    def tiled(node: ast.AST, unit: Unit | None) -> ast.AST:
        """`unit_code`, with each input tile read as a value of the tile's shape, as in the kernel."""

        class Tiles(ast.NodeTransformer):
            def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
                if isinstance(n.value, ast.Name) and n.value.id in nest.params:
                    return ast.copy_location(ast.Name(spec.operand(n, unit and unit.phase).ref, ast.Load()), n)
                return self.generic_visit(n)

        return Tiles().visit(spec.unit_code(node, unit))

    statements = [(u, s) for u in spec.units for s in u.statements]
    reads_ = tuple((key, op.ref) for key, op in spec._reads.items())
    key = ("trace", spec.code.filename, tuple(spec.finished_loops(len(spec.phases)).items()), reads_)
    if key not in nest.cache:
        nest.cache[key] = (
            [spec.code.compile([tiled(s, u)], "exec") for u, s in statements],
            [spec.code.compile(tiled(st.value, None), "eval") for st in nest.stores],
        )
    compiled, stores = nest.cache[key]
    helpers = mxu.helpers("highest")
    loop_vars = list(spec.loops)
    at: list[ast.AST | None] = [None]  # the statement being traced, for errors

    def run(tiles, loop_values, scalar_values):
        ns = spec.namespace(None, **helpers)
        ns.update(zip([op.ref for op in spec.operands], tiles, strict=True))
        ns.update(zip(loop_vars, loop_values, strict=True))
        ns.update(zip(spec.scalars, scalar_values, strict=True))  # runtime scalars: traced, as in the kernel
        values = []
        for (_, s), code in zip(statements, compiled, strict=True):
            at[0] = s
            exec(code, ns)
            name = assigned_name(s)
            values.append(jnp.asarray(ns[name]))
            if name in state:
                ns[name] = _as_stored(values[-1], state[name])
        outs = []
        for st, code in zip(nest.stores, stores, strict=True):
            at[0] = st.node
            outs.append(jnp.asarray(eval(code, ns)))
        return values, outs

    loop_avals = [jax.ShapeDtypeStruct((), jnp.int32, weak_type=True)] * len(loop_vars)
    scalar_avals = [spec.specs[nest.params.index(p)] for p in spec.scalars]
    try:
        tiles = [jax.ShapeDtypeStruct(op.block.kernel_shape, op.dtype) for op in spec.operands]
        traced, store_avals = jax.eval_shape(run, tiles, loop_avals, scalar_avals)
    except _CONTROL_FLOW_ERRORS:
        raise nest.error(
            at[0],
            "Python control flow (`if`, `and`/`or`, `int(...)`, shapes) can't depend on loop variables or tile "
            "values, which are only known in the kernel; use jnp.where, or a mask (`if cond: continue`)",
        ) from None
    except Exception as e:
        raise nest.error(at[0], f"couldn't trace the tile body ({type(e).__name__}: {e})") from None
    return [(s, aval) for (_, s), aval in zip(statements, traced, strict=True)], store_avals


def _join(avals: Sequence[jax.ShapeDtypeStruct]) -> tuple[np.dtype, bool]:
    """The type that holds all of `avals`, by JAX's promotion rules (weak types included)."""
    examples = [_convert_element_type(jnp.zeros((), a.dtype), np.dtype(a.dtype), weak_type=a.weak_type) for a in avals]
    dtype, weak = jax.dtypes.result_type(*examples, return_weak_type_flag=True)
    return np.dtype(dtype), bool(weak)


def _aval_key(a: jax.ShapeDtypeStruct) -> tuple:
    return tuple(a.shape), np.dtype(a.dtype), bool(a.weak_type)


def _state(
    spec: Specialization,
) -> tuple[dict[str, StateVar], dict[int, np.dtype], list[jax.ShapeDtypeStruct], dict[int, jax.ShapeDtypeStruct]]:
    """The state variables, the assignments whose values must be cast to their type, the store values' types and
    the type each assignment produces.

    A state variable's type holds every value assigned to it. Reading it back as that type can change the
    types computed from it (`acc = acc + prev` with an int8 `acc` and an int32 `prev`), so the tile is traced
    again until the types settle.
    """
    nest = spec.nest
    needs = _needs_state(spec)
    types: dict[str, jax.ShapeDtypeStruct] = {}
    for _ in range(16):  # the types only ever widen, a few times at most
        traced, store_avals = _trace_tile(spec, types)
        by_name: dict[str, list[tuple[ast.stmt, jax.ShapeDtypeStruct]]] = {}
        for s, aval in traced:
            by_name.setdefault(assigned_name(s), []).append((s, aval))
        joined = {}
        for name in sorted(needs):
            entries = by_name[name]
            shapes = list(dict.fromkeys(tuple(a.shape) for _, a in entries))
            if len(shapes) != 1:
                raise nest.error(
                    entries[0][0], f"`{name}` changes shape ({' -> '.join(map(str, shapes))}) between steps"
                )
            dtype, weak = _join([a for _, a in entries] + ([types[name]] if name in types else []))
            joined[name] = jax.ShapeDtypeStruct(shapes[0], dtype, weak_type=weak)
        # Settled when reading the state back as `joined` gives the types just traced: the trace read it as
        # `joined` already, or (first pass) every value it read already had that type.
        uniform = all(_aval_key(a) == _aval_key(joined[name]) for name in needs for _, a in by_name[name])
        if (not types and uniform) or {n: _aval_key(a) for n, a in joined.items()} == {
            n: _aval_key(a) for n, a in types.items()
        }:
            break
        types = joined
    else:
        raise nest.error(nest.parallel[0].node, f"the types of {sorted(needs)} don't settle between steps")
    state: dict[str, StateVar] = {}
    casts: dict[int, np.dtype] = {}
    for name, aval in joined.items():
        shape, dtype = tuple(aval.shape), np.dtype(aval.dtype)
        state[name] = StateVar(name, spec.names.fresh(f"{name}_ref"), aval, shape == ())
        casts.update({id(s): dtype for s, a in by_name[name] if np.dtype(a.dtype) != dtype})
    return state, casts, store_avals, {id(s): a for s, a in traced}


def _check_stores(spec: Specialization, store_avals: list[jax.ShapeDtypeStruct]) -> set[str]:
    """Outputs whose stored value has to be broadcast to the tile, as `.at[...].set(...)` does."""
    broadcast = set()
    for st, aval in zip(spec.nest.stores, store_avals, strict=True):
        tile = next(o for o in spec.outputs if o.var == st.out).block.kernel_shape
        if tuple(aval.shape) == tile:
            continue
        try:
            ok = np.broadcast_shapes(tuple(aval.shape), tile) == tile
        except ValueError:
            ok = False
        if not ok:
            raise spec.nest.error(
                st.node,
                f"the value stored into `{st.out}` has shape {tuple(aval.shape)}, which doesn't fit its tile {tile}",
            )
        broadcast.add(st.out)
    return broadcast
