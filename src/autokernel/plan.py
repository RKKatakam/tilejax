"""Resolve a parsed loop nest for concrete input shapes: what the kernel copies, keeps and computes.

`make_plan` runs the prologue on the inputs' shapes, evaluates the loop ranges, turns every tile read and
store into a window of an array, finds the values a tile carries from one step to the next, and traces the
tile code once with abstract values to learn every type. Nothing here depends on how the kernel is
pipelined (pipeline.py) or the order its steps run in (schedule.py).

A tile's reduction loops become its phases: the steps of the first, then of the second, ... A reduction
loop of one iteration (and no mask) takes no step of its own: its body runs inline, in the code around it,
with its variable bound to its one value. So a tunable block size that covers a whole dim turns a loop
into straight-line code, and the windows it moved into ones fetched once.

A repeat loop's phases take their steps once per iteration, one iteration after another; windows the
repeat loop moves (a layer's weights, `w[l, :, f : f + bf]`) stream like any other, so the next
iteration's are fetched while this one's run.

Inputs read element by element (`lengths[b]`) are index arrays, kept whole in SMEM: their elements are
values in tile code, conditions in reduction masks (which the kernel then evaluates when it runs: runtime
masks), bounds of reduction loops (`for p in range(pages[b])`: runtime ranges, whose steps the kernel finds
as it runs), and integer indices of tile reads (`w[expert[e], :, :]`: runtime windows) and of stores into an
output updated in place.

An output allocated as an input (`new = pool`) is that input updated in place: the kernel writes the tiles
it stores into the input's buffer, and everything else keeps the input's values. Only such an output may be
stored at windows index arrays choose (`new = new.at[slot[b], :, :].set(...)`, a scatter): what no tile
stores is the input's.

An input gathered across the chips of a mesh axis (`xs = autokernel.all_gather(x, "x")`) is planned as
an input of its gathered shape, `x`'s rows times the axis's chips; its rows travel a ring of the chips while
the kernel runs (`Ring`), so its tiles must take whole rows of one chip's: the outermost parallel loop's.
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import itertools
import traceback
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src import source_info_util

from . import collective, mxu, packing, ragged, v5e
from .errors import KernelError
from .parse import Loop, Nest
from .tilecode import as_store, assigned, element_reads, first_reads, flatten, reads

__all__ = [
    "Carried",
    "Index",
    "Kind",
    "Operand",
    "Output",
    "Phase",
    "Plan",
    "Range",
    "Repeat",
    "Ring",
    "Tile",
    "Window",
    "make_plan",
]


@dataclass(frozen=True)
class Range:
    """A loop's iterations: `start + block * step` for block in `range(count)`. A runtime range (its bounds
    read index arrays, `range(pages[b])`) has `bounds`, its start and stop, read when the kernel runs: its
    value at block k is start + k * step, and `count` (2) only says that it moves what it indexes."""

    var: str
    start: int
    step: int
    count: int
    mask: ast.expr | None  # `if mask: continue` opening the loop's body
    runtime: bool = False  # does the mask read index arrays (so it is evaluated when the kernel runs)?
    bounds: tuple[ast.expr, ast.expr] | None = None  # a runtime range's start and stop

    def value(self, block: int) -> int:
        return self.start + block * self.step


@dataclass(frozen=True)
class Window:
    """How a tile indexes one array dim: from element `offset` (plus the value of `loop`, if any), `size`
    elements; `size` None is an integer index, which drops the dim from the tile. An integer index may be an
    element of an index array (`index`, `expert[e]`), read when the kernel runs: `index_loops` move it. With
    `index` a slice of an index array (`pages[b, p : p + n]`, `size` n) the window gathers those n
    elements of the dim; with `rows` (`autokernel.rows(start, stop, size)`), it is `size` elements from
    start, of which those before stop are copied."""

    loop: str | None
    offset: int
    size: int | None
    index: ast.Subscript | None = None
    index_loops: frozenset[str] = frozenset()
    rows: tuple[ast.expr, ast.expr] | None = None


@dataclass(frozen=True)
class Tile:
    """A window of an array that the kernel copies between HBM and VMEM."""

    array: str
    windows: tuple[Window, ...]
    dtype: np.dtype
    text: str  # as written, e.g. "x[i:i + TM, k:k + TK]"

    @property
    def shape(self) -> tuple[int, ...]:
        """The tile's shape in VMEM: the sliced dims."""
        return tuple(w.size for w in self.windows if w.size is not None)

    @property
    def loops(self) -> frozenset[str]:
        """The loop variables that move the window."""
        return frozenset(w.loop for w in self.windows if w.loop).union(*(w.index_loops for w in self.windows))


class Kind(Enum):
    """When an input tile changes, and so how the kernel buffers it."""

    STREAM = "stream"  # every step (of its phase): a ring of slots, fetched ahead
    TILE = "tile"  # every output tile: two slots, the next tile's fetched at a tile's first step
    # every iteration of a repeat loop (and tile): a ring of slots, fetched ahead for its phase's first step
    ITERATION = "iteration"
    ONCE = "once"  # never: fetched once, before the first step


@dataclass(frozen=True)
class Operand:
    """An input tile the kernel reads."""

    name: str  # unique among operands: "x", or "x2" for a second window of x
    param: int  # which input
    tile: Tile
    kind: Kind
    phase: int | None = None  # a streamed operand of a phase: the phase (None: it streams every step)
    # An ITERATION operand only its phase's body reads: an iteration whose steps are all masked out (one
    # empty step) doesn't fetch it.
    in_body: bool = False


@dataclass(frozen=True)
class Index:
    """An index array: an input read element by element, kept whole in SMEM."""

    name: str
    param: int  # which input
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class Computed:
    """A computed index array: a value the tile code computes (int32, 1-D or 2-D) whose elements index the
    windows of later phases, or bound their loops (`w[layer, chosen[c], :, :]`, `range(n[0] * F)`). Its
    statement runs at the last step of phase `phase` (in an iteration of repeat loop `repeat`, if any); the
    kernel copies the value into SMEM there, and the phases `consumers` read it when the kernel runs."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    phase: int
    repeat: int | None
    consumers: tuple[int, ...]


@dataclass(frozen=True)
class Phase:
    """A reduction loop that takes steps: the tile's steps for it, and its body. In a repeat loop it takes
    them once per iteration; the repeat loop's first phase runs the loop's own code before it (`before`) at
    its first step, every iteration."""

    range: Range
    body: tuple[ast.stmt, ...]
    before: tuple[ast.stmt, ...] = ()
    repeat: int | None = None  # the repeat loop it runs in, an index into `Plan.repeats`


@dataclass(frozen=True)
class Repeat:
    """A repeat loop that takes steps: its phases, `first` to `last` (exclusive), run `range.count` times
    over. `after` is the tile's code after the loop (up to the next phase), run as its last iteration ends."""

    range: Range
    first: int
    last: int
    after: tuple[ast.stmt, ...]


@dataclass(frozen=True)
class Output:
    """An output: its array, the fill of tiles a mask skips, and the tile each step stores."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fill: Any  # the allocation's value (0 for jnp.zeros); None for an input updated in place
    tile: Tile
    value: jax.ShapeDtypeStruct  # the type of the value the tile code stores
    phase: int | None = None  # the phase whose steps each store a tile of it (None: a tile per output tile)
    aliases: int | None = None  # the input it updates in place (its buffer is the output's), if any


@dataclass(frozen=True)
class Carried:
    """A value a tile carries from one reduction step to the next: VMEM scratch in the kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class Ring:
    """An input all-gathered across the `size` chips of a mesh axis (`xs = autokernel.all_gather(x, axis)`),
    its rows passed around a ring of them as the kernel runs. The outermost parallel loop, `loop`, walks the
    rows a chip's shard at a time (`shard_blocks` of its blocks each), this chip's own first: the tiles of ring
    step r read the shard of chip (me - r) mod size, which arrived while ring step r - 1 ran."""

    name: str  # the gathered array: a param, whose HBM buffer the kernel fills
    source: str  # the input it gathers: this chip's shard
    axis: str
    size: int  # the axis's chips
    rows: int  # rows of one chip's shard
    loop: str
    shard_blocks: int


@dataclass(frozen=True)
class Plan:
    """A loop nest resolved for one input signature."""

    nest: Nest
    inputs: tuple[jax.ShapeDtypeStruct, ...]
    tunable: Any  # the tunable's values it was planned with (None: the kernel has none)
    parallel: tuple[Range, ...]  # outermost first
    reductions: tuple[Range, ...]  # every reduction loop, in order (the inlined ones too)
    phases: tuple[Phase, ...]  # the reduction loops that take steps
    # The code at a tile's first step (before the phases), then after each phase, at its last step (in a
    # repeat loop: every iteration).
    segments: tuple[tuple[ast.stmt, ...], ...]
    repeats: tuple[Repeat, ...]  # the repeat loops that take steps
    repeat_loops: tuple[Range, ...]  # every repeat loop (one of one iteration runs inline)
    operands: tuple[Operand, ...]
    indices: tuple[Index, ...]  # the index arrays, in parameter order
    computed: tuple[Computed, ...]  # the computed index arrays
    reads: Mapping[str, Operand]  # `read_key` of a tile read -> its operand
    # Reads of windows only a repeat loop moves, outside phase bodies: (line, column) -> the phase whose
    # steps run them (they stream with that phase).
    read_phases: Mapping[tuple[int, int], int]
    outputs: tuple[Output, ...]
    carried: Mapping[str, Carried]
    assigned_types: Mapping[int, Mapping[str, jax.ShapeDtypeStruct]]  # id(stmt) -> the types it assigns
    constants: Mapping[str, Any]  # names the tile code reads from the module, closure and prologue
    scopes: tuple[str, ...]  # the names of the tile code's `jax.named_scope`s
    # Tile reads taken as words (`autokernel.pairs(x[...])`): the call's (line, column) -> the operand read;
    # names the tile code binds once to such words (`words = pairs(x[...])`) -> the operand.
    packed: Mapping[tuple[int, int], str]
    packed_names: Mapping[str, str]
    unpacked: frozenset[str]  # outputs stored from words (`out.at[...].set(autokernel.unpairs(words, dtype))`)
    tile_ops: int  # operations in a tile's code traced as plain JAX (each phase body twice): its size
    env: dict[str, Any]  # the function's module and closure values, with the prologue's names over them
    rings: tuple[Ring, ...] = ()  # the inputs gathered across chips

    @property
    def single_step(self) -> bool:
        """Does every tile run in one step (no phases)? Then nothing is carried between steps."""
        return not self.phases

    @property
    def loops(self) -> tuple[Range, ...]:
        return self.parallel + self.reductions + self.repeat_loops

    @property
    def instances(self) -> tuple[tuple[int, int], ...]:
        """A tile's phases in the order its steps run them: (phase, iteration of its repeat loop, or 0)."""
        out: list[tuple[int, int]] = []
        p = 0
        while p < len(self.phases):
            r = self.phases[p].repeat
            if r is None:
                out.append((p, 0))
                p += 1
                continue
            rep = self.repeats[r]
            out += [(q, it) for it in range(rep.range.count) for q in range(rep.first, rep.last)]
            p = rep.last
        return tuple(out)

    def read_key(self, node: ast.Subscript) -> str:
        """Identifies a tile read `x[...]` (the same window of the same input is one operand)."""
        return _read_key(node, self.read_phases)

    @property
    def runtime(self) -> bool:
        """Do masks read index arrays, so the kernel's steps are found when it runs?"""
        return any(phase.range.runtime for phase in self.phases)

    @property
    def cursor(self) -> bool:
        """Does a loop have runtime bounds (or a window read computed indices), so the kernel finds each step
        (and tile) as it runs?"""
        ranges = (*self.parallel, *(phase.range for phase in self.phases))
        return any(r.bounds is not None for r in ranges) or bool(self.computed)

    @property
    def either(self) -> bool:
        """May the kernel find its steps either way: from tables the device makes of its runtime masks before
        it runs, or with a cursor as it runs (its outputs are stored per tile)? Tuning tries both."""
        return self.runtime and not self.cursor and all(out.phase is None for out in self.outputs)

    def eval(self, node: ast.expr, values: Mapping[str, Any] | None = None) -> Any:
        """Evaluate an expression of the function on the prologue's names (and `values`, e.g. loop values)."""
        return evaluate(self.nest, self.env, node, values)


def evaluate(nest: Nest, env: dict[str, Any], node: ast.expr, values: Mapping[str, Any] | None = None) -> Any:
    """Evaluate an expression of `nest`'s function with `env` as its globals and `values` (loop values) as
    its locals; KernelError if it fails."""
    try:
        code = compile(ast.Expression(node), nest.filename, "eval")
        return eval(code, env, dict(values or {}))
    except Exception as e:
        raise nest.error(f"can't evaluate `{ast.unparse(node)}` when planning: {e!r}", node) from e


def _read_key(node: ast.Subscript, read_phases: Mapping[tuple[int, int], int]) -> str:
    """A tile read's operand key: its window, and the phase that streams it if only a repeat loop moves it."""
    phase = read_phases.get((node.lineno, node.col_offset))
    return ast.dump(node) if phase is None else f"{ast.dump(node)}@{phase}"


def make_plan(nest: Nest, fn: Callable, inputs: Sequence[jax.ShapeDtypeStruct], tunable: Any = None) -> Plan:
    """Resolve `nest` (parsed from `fn`) for inputs of these shapes and dtypes, and its tunable's values."""
    return _Planner(nest, fn, tuple(inputs), tunable).plan()


class _Planner:
    def __init__(self, nest: Nest, fn: Callable, inputs: tuple[jax.ShapeDtypeStruct, ...], tunable: Any) -> None:
        self.nest, self.fn, self.inputs = nest, fn, inputs
        closure = inspect.getclosurevars(fn).nonlocals
        # One namespace (code in the prologue, masks and bounds sees its names from lambdas and generator
        # expressions too): the module's values, the closure's over them, the prologue's over those.
        self.env: dict[str, Any] = {"__builtins__": builtins, **fn.__globals__, **closure}
        self.env.update(zip(nest.params, inputs, strict=True))
        if nest.tunable:
            self.env[nest.tunable] = tunable
        self.ranges: dict[str, Range] = {}
        self.indices: dict[str, Index] = {}
        self.read_phases: dict[tuple[int, int], int] = {}

    def error(self, message: str, node: ast.AST | None = None) -> KernelError:
        return self.nest.error(message, node)

    def eval(self, node: ast.expr) -> Any:
        return evaluate(self.nest, self.env, node)

    def plan(self) -> Plan:
        outputs = self._prologue()
        self.indices = {index.name: index for index in self._index_arrays()}
        self.computed = self._computed_names()
        aliased = self._aliases()
        parallel = tuple(self._range(loop) for loop in self.nest.parallel)
        reductions = tuple(self._range(loop) for loop in self.nest.reductions)
        repeat_loops = tuple(self._range(rep.loop) for rep in self.nest.repeats)
        self.ranges = {r.var: r for r in (*parallel, *reductions, *repeat_loops)}
        self._check_element_bounds()
        phases, segments, repeats = self._phases(reductions, repeat_loops)
        operands, reads_ = self._operands(phases, segments, repeats)
        rings = self._rings(parallel, operands, aliased)
        packed = self._packed(reads_)
        packed_names = self._packed_names(packed)
        stored_by = {name: self._storing_phase(name, phases) for name in self.nest.outputs}
        if any(r.bounds is not None for r in (*parallel, *(phase.range for phase in phases))) or self.computed:
            self._check_cursor(stored_by, outputs)
        output_tiles = {name: self._output_tile(name, *outputs[name], parallel) for name in self.nest.outputs}
        unpacked = self._unpacked(output_tiles)
        self._check_reads_before_assignment()
        sequence = _Sequence(phases, segments, repeats)
        # Values the tile code updates part of (`a = a.at[k].set(v)`) live in VMEM too: Mosaic can't update
        # part of a value in registers (no scatter, no dynamic_update_slice), only store into a buffer.
        updated = {store.output for stmt in flatten(self.nest.tile) if (store := as_store(stmt))}
        carried_names = (sequence.carried() | updated) - set(self.nest.outputs)
        scopes = self._scopes()
        types = _TypeTrace(self, operands, reads_, carried_names, sequence).run()
        for name in self.nest.outputs:
            self._check_store(output_tiles[name], types.stores[name])
        carried = {name: Carried(name, aval.shape, aval.dtype) for name, aval in types.carried.items()}
        computed = self._computed(types.assigned, phases, segments, repeats, operands)
        return Plan(
            nest=self.nest,
            inputs=self.inputs,
            tunable=self.env.get(self.nest.tunable) if self.nest.tunable else None,
            parallel=parallel,
            reductions=reductions,
            phases=phases,
            segments=segments,
            repeats=repeats,
            repeat_loops=repeat_loops,
            operands=operands,
            indices=tuple(self.indices.values()),
            computed=computed,
            reads=reads_,
            read_phases=self.read_phases,
            outputs=tuple(
                Output(name, *outputs[name], output_tiles[name], types.stores[name], stored_by[name], aliased[name])
                for name in self.nest.outputs
            ),
            carried=carried,
            assigned_types=types.assigned,
            constants=self._constants(),
            scopes=scopes,
            packed=packed,
            packed_names=packed_names,
            unpacked=unpacked,
            tile_ops=types.ops,
            env=self.env,
            rings=rings,
        )

    # ------------------------------------------------------------------ the prologue and outputs

    def _prologue(self) -> dict[str, tuple[tuple[int, ...], np.dtype, Any]]:
        """Run the prologue on the inputs' shapes; each output's (shape, dtype, fill)."""
        allocation_of = {id(stmt): name for name, stmt in self.nest.allocations.items()}
        gathered = {id(g.node) for g in self.nest.all_gathers}
        outputs = {}
        for stmt in self.nest.prologue:
            if id(stmt) in gathered:  # (the gathered array is an input of its gathered shape already)
                func = stmt.value.func  # type: ignore[attr-defined]
                if self.eval(func) is not collective.all_gather:
                    raise self.error("gather an input across chips with autokernel.all_gather", func)
                continue
            name = allocation_of.get(id(stmt))
            if name is not None:
                outputs[name] = self._allocation(name, stmt.value)  # type: ignore[attr-defined]
                self.env[name] = jax.ShapeDtypeStruct(*outputs[name][:2])
                continue
            try:
                exec(compile(ast.Module([stmt], []), self.nest.filename, "exec"), self.env)
            except Exception as e:
                raise self.error(f"the prologue runs on the inputs' shapes and dtypes, and failed: {e!r}", stmt) from e
        return outputs

    def _allocation(self, name: str, call: ast.expr) -> tuple[tuple[int, ...], np.dtype, Any]:
        usage = (
            f"allocate output `{name}` with jnp.zeros(shape, dtype) or jnp.full(shape, value, dtype), or as an "
            f"input updated in place (`{name} = x`)"
        )
        if isinstance(call, ast.Name) and call.id in self.nest.params:  # an input updated in place
            spec = self.inputs[self.nest.params.index(call.id)]
            return tuple(spec.shape), np.dtype(spec.dtype), None
        if not isinstance(call, ast.Call):
            raise self.error(usage, call)
        func = self.eval(call.func)
        if func is not jnp.zeros and func is not jnp.full:
            raise self.error(usage, call)
        args = [self.eval(a) for a in call.args]
        kwargs = {k.arg: self.eval(k.value) for k in call.keywords if k.arg}
        try:
            aval = jax.eval_shape(lambda: func(*args, **kwargs))
        except Exception as e:
            raise self.error(f"{usage}: {e!r}", call) from e
        fill = 0 if func is jnp.zeros else (args[1] if len(args) > 1 else kwargs.get("fill_value"))
        if not np.isscalar(fill):
            raise self.error(f"output `{name}` is filled with a scalar constant, not {type(fill).__name__}", call)
        return aval.shape, aval.dtype, fill

    def _aliases(self) -> dict[str, int | None]:
        """For each output, the input it updates in place (`new = pool`), if any: an input array (not an index
        array, which is in SMEM) that no other output updates."""
        aliased: dict[str, int | None] = {}
        for name in self.nest.outputs:
            value = self.nest.allocations[name].value
            is_input = isinstance(value, ast.Name) and value.id in self.nest.params
            aliased[name] = self.nest.params.index(value.id) if is_input else None  # type: ignore[attr-defined]
            if not is_input:
                continue
            if value.id not in self.nest.inputs:  # type: ignore[attr-defined]
                raise self.error(f"`{value.id}` is gathered across chips: an output can't update it in place", value)  # type: ignore[attr-defined]
            if value.id in self.indices:  # type: ignore[attr-defined]
                raise self.error(
                    f"output `{name}` updates `{value.id}` in place, but `{value.id}` is an index array (in SMEM)",  # type: ignore[attr-defined]
                    value,
                )
            if list(aliased.values()).count(aliased[name]) > 1:
                raise self.error(f"two outputs update `{value.id}` in place: one may", value)  # type: ignore[attr-defined]
        return aliased

    # ------------------------------------------------------------------ loops

    def _range(self, loop: Loop) -> Range:
        if loop.mask is not None and element_reads(loop.mask, assigned(self.nest.tile)):
            raise self.error(
                "a mask can't read values the tile code computes (it is evaluated before the kernel runs): bound "
                "the loop with them instead",
                loop.mask,
            )
        if any(element_reads(b, {*self.indices, *self.computed}) for b in loop.bounds):
            return self._runtime_range(loop)
        bounds = [self.eval(b) for b in loop.bounds]
        if not all(isinstance(b, int | np.integer) for b in bounds):
            raise self.error(f"loop bounds are ints when planning, got {bounds}", loop.node)
        bounds = [int(b) for b in bounds]
        if len(bounds) == 3 and bounds[2] <= 0:
            raise self.error(f"`{loop.var}` steps by {bounds[2]}: loops count up", loop.node)
        r = range(*bounds)
        if not r:
            raise self.error(f"`{loop.var}` loops over {r}, which is empty", loop.node)
        if (r.stop - r.start) % r.step:
            raise self.error(
                f"`{loop.var}` loops over {r}: the range must be a whole number of steps of {r.step}", loop.node
            )
        if r.start % r.step:
            raise self.error(f"`{loop.var}` loops over {r}: it must start at a multiple of its step", loop.node)
        runtime = loop.mask is not None and bool(element_reads(loop.mask, self.indices))
        return Range(loop.var, r.start, r.step, len(r), loop.mask, runtime)

    def _runtime_range(self, loop: Loop) -> Range:
        """A loop whose bounds read index arrays: its start and stop are read when the kernel runs (its step is
        an int when planning). A reduction loop, or the one parallel loop (the kernel's tiles, counted as it
        runs)."""
        single = self.nest.parallel == (loop,)
        if loop not in self.nest.reductions and not single:
            raise self.error(
                "only reduction loops' bounds, or a kernel's one parallel loop's, may read index arrays", loop.node
            )
        if loop.mask is not None:
            raise self.error("a loop with runtime bounds can't be masked: put the condition in its bounds", loop.mask)
        start, stop, step = (ast.Constant(0), loop.bounds[0], 1) if len(loop.bounds) == 1 else (*loop.bounds[:2], None)
        if step is None:
            step = self.eval(loop.bounds[2]) if len(loop.bounds) == 3 else 1
        if not isinstance(step, int | np.integer) or step <= 0:
            raise self.error(f"a loop with runtime bounds steps by a positive int, not {step!r}", loop.node)
        return Range(loop.var, 0, int(step), 2, None, bounds=(start, stop))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ tiles

    def _window(self, index: ast.expr, dim: int, text: str) -> Window:
        """How the index expression `index` selects from a dim of size `dim`."""
        usage = (
            f"{text}: each index is `e:e + size` (e an expression of loop variables) or `v` for a loop variable v, "
            "`:`, or constants (`start:stop`, an int)"
        )
        loop_vars = reads(index) & self.ranges.keys()
        computed = reads(index) & assigned(self.nest.tile) - self.computed
        if computed and (self._is_rows(index) or isinstance(index, ast.Subscript)):
            raise self.error(
                f"{text}: a window's index can't read `{sorted(computed)[0]}`, which the tile code computes: the "
                "kernel copies the window before the tile code runs",
                index,
            )
        elements = {*self.indices, *self.computed}
        if isinstance(index, ast.Subscript) and isinstance(index.value, ast.Name) and index.value.id in elements:
            sliced = [i for i in _dims(index) if isinstance(i, ast.Slice)]
            if not sliced:
                return Window(None, 0, None, index, frozenset(loop_vars))  # bounds: clamped when the kernel runs
            return Window(None, 0, self._gathered(sliced, text), index, frozenset(loop_vars))
        if self._is_rows(index):
            start, stop, size = index.args  # type: ignore[attr-defined]
            if reads(size) & self.ranges.keys():
                raise self.error(f"{text}: autokernel.rows's size is an int when planning", size)
            return Window(None, 0, self._int(size, text), None, frozenset(loop_vars), rows=(start, stop))
        if isinstance(index, ast.Slice):
            lower, upper = index.lower, index.upper
            if index.step is not None:
                raise self.error(usage, index)
            if isinstance(lower, ast.Name) and lower.id in self.ranges:  # v:v + size
                if not (
                    isinstance(upper, ast.BinOp)
                    and isinstance(upper.op, ast.Add)
                    and ast.dump(upper.left) == ast.dump(lower)
                    and not reads(upper.right) & self.ranges.keys()
                ):
                    raise self.error(usage, index)
                return self._checked(Window(lower.id, 0, self._int(upper.right, text)), dim, text, index)
            if (
                loop_vars
                and isinstance(upper, ast.BinOp)
                and isinstance(upper.op, ast.Add)
                and ast.dump(upper.left) == ast.dump(lower)
                and not reads(upper.right) & self.ranges.keys()
                and not computed
            ):  # `e : e + size`, e computed from loop variables: the rows of `rows(e, e + size, size)`
                size = self._int(upper.right, text)
                return Window(None, 0, size, None, frozenset(loop_vars), rows=(lower, upper))  # type: ignore[arg-type]
            if loop_vars:
                raise self.error(usage, index)
            start = 0 if lower is None else self._int(lower, text)
            stop = dim if upper is None else self._int(upper, text)
            return self._checked(Window(None, start, stop - start), dim, text, index)
        if isinstance(index, ast.Name) and index.id in self.ranges:
            return self._checked(Window(index.id, 0, None), dim, text, index)
        computed = reads(index) & assigned(self.nest.tile) - self.computed
        if loop_vars or computed:
            more = f": `{sorted(computed)[0]}` is computed by the tile code" if computed else ""
            raise self.error(f"{usage}, or an element of an index array written in place (`w[ids[e], :]`){more}", index)
        return self._checked(Window(None, self._int(index, text), None), dim, text, index)

    def _gathered(self, sliced: list[ast.Slice], text: str) -> int:
        """How many elements `idx[..., a : a + n]` gathers: n (one dim sliced so, n an int when planning)."""
        usage = f"{text}: a window gathered through an index array slices one of its dims, `idx[a : a + n]`"
        index = sliced[0]
        if (
            len(sliced) != 1
            or index.step is not None
            or index.lower is None
            or not (isinstance(index.upper, ast.BinOp) and isinstance(index.upper.op, ast.Add))
            or ast.dump(index.upper.left) != ast.dump(index.lower)
            or reads(index.upper.right) & self.ranges.keys()
        ):
            raise self.error(usage, index)
        return self._int(index.upper.right, text)

    def _is_rows(self, index: ast.expr) -> bool:
        """Is `index` a call of `autokernel.rows`?"""
        if not (isinstance(index, ast.Call) and isinstance(index.func, ast.Name | ast.Attribute)):
            return False
        try:
            func = self.eval(index.func)
        except KernelError:
            return False
        if func is ragged.rows and (len(index.args) != 3 or index.keywords):
            raise self.error("autokernel.rows takes (start, stop, size)", index)
        return func is ragged.rows

    def _int(self, node: ast.expr, text: str) -> int:
        value = self.eval(node)
        if not isinstance(value, int | np.integer):
            raise self.error(f"{text}: tile bounds are ints when planning, not {value!r}", node)
        return int(value)

    def _checked(self, window: Window, dim: int, text: str, node: ast.AST) -> Window:
        """`window`, once it is known to stay inside the dim."""
        if window.size is not None and window.size <= 0:
            raise self.error(f"{text}: a slice takes at least one element", node)
        first = window.offset
        last = window.offset + (window.size or 1)
        if window.loop and self.ranges[window.loop].bounds is not None:
            return window  # moved by a runtime range: clamped to the dim when the kernel runs
        if window.loop:
            r = self.ranges[window.loop]
            first += r.start
            last += r.value(r.count - 1)
        if first < 0 or last > dim:
            raise self.error(f"{text}: reads elements {first} to {last - 1} of a dim of {dim}", node)
        return window

    def _tile(
        self, array: str, shape: tuple[int, ...], dtype: np.dtype, index: Sequence[ast.expr], node: ast.AST
    ) -> Tile:
        text = f"{array}[{', '.join(ast.unparse(i) for i in index)}]"
        if len(index) != len(shape):
            raise self.error(f"{text}: index every one of {array}'s {len(shape)} dims", node)
        windows = tuple(self._window(i, d, text) for i, d in zip(index, shape, strict=True))
        sliced = [w.size is not None for w in windows]
        if not any(sliced):
            raise self.error(f"{text}: a tile keeps at least one dim (slice it, `v:v + 1`)", node)
        if not all(sliced[sliced.index(True) :]):
            raise self.error(f"{text}: integer indices go before every sliced dim", node)
        # (NumPy indexing zips index arrays together, and puts their dims first when a slice separates them
        # from integer indices: the loop nest's tile wouldn't be the kernel's.)
        ragged = [k for k, i in enumerate(index) if self._is_rows(i)]
        if len(ragged) > 1:
            raise self.error(f"{text}: a tile reads through one `autokernel.rows` at most", node)
        if ragged and 0 < sliced.index(True) < ragged[0]:
            raise self.error(
                f"{text}: `autokernel.rows` goes right after the integer indices (NumPy moves its dim to the front "
                "otherwise): index the dims between them with slices of one, `v : v + 1`",
                node,
            )
        tile = Tile(array, windows, np.dtype(dtype), text)
        dims = [d for d, w in zip(shape, windows, strict=True) if w.size is not None]
        problems = v5e.layout_problems(tile.shape, dims, dtype)
        if problems:
            raise self.error(f"{text}: a {tile.shape} tile breaks the TPU layout: {'; '.join(problems)}", node)
        return tile

    def _phases(
        self, reductions: tuple[Range, ...], repeat_loops: tuple[Range, ...]
    ) -> tuple[tuple[Phase, ...], tuple[tuple[ast.stmt, ...], ...], tuple[Repeat, ...]]:
        """The reduction loops that take steps, the code around them, and the repeat loops around them. A
        loop of one iteration (and no mask) joins the code around it, its variable bound to its value first."""
        nest = self.nest
        # The tile as written: code, reduction loops, and repeat loops of code and reduction loops.
        items: list[tuple[Any, ...]] = [("code", nest.segments[0])]
        starts = {rep.first: (rep, r) for rep, r in zip(nest.repeats, repeat_loops, strict=True)}
        i = 0
        while i < len(reductions):
            if i not in starts:
                items += [("loop", reductions[i], nest.bodies[i]), ("code", nest.segments[i + 1])]
                i += 1
                continue
            rep, r = starts[i]
            inner: list[tuple[Any, ...]] = [("code", rep.before)]
            for j in range(rep.first, rep.last):
                inner += [("code", nest.segments[j])] if j > rep.first else []
                inner.append(("loop", reductions[j], nest.bodies[j]))
            items += [("repeat", r, [*inner, ("code", rep.after)]), ("code", nest.segments[rep.last])]
            i = rep.last

        phases: list[Phase] = []
        segments: list[list[ast.stmt]] = [[]]
        repeats: list[tuple[Range, int, int]] = []
        afters: list[list[ast.stmt]] = []  # each repeat loop's `after`
        code = segments[-1]  # where the tile's code goes next
        for item in self._inline(items):
            if item[0] == "code":
                code += item[1]
            elif item[0] == "loop":
                phases.append(Phase(item[1], item[2]))
                segments.append([])
                code = segments[-1]
            else:
                first, before = len(phases), []
                for sub in item[2]:
                    if sub[0] == "code":
                        (before if len(phases) == first else segments[-1]).extend(sub[1])
                    else:
                        own = tuple(before) if len(phases) == first else ()
                        phases.append(Phase(sub[1], sub[2], own, len(repeats)))
                        segments.append([])
                repeats.append((item[1], first, len(phases)))
                afters.append([])
                code = afters[-1]
        loops = tuple(Repeat(r, a, b, tuple(after)) for (r, a, b), after in zip(repeats, afters, strict=True))
        return tuple(phases), tuple(tuple(s) for s in segments), loops

    def _inline(self, items: list[tuple[Any, ...]], keep_first: bool = False) -> list[tuple[Any, ...]]:
        """`items` with the loops of one iteration (and no mask) as code, a reduction loop's binding its
        variable first; with `keep_first`, the first reduction loop stays a loop (a phase of one step)."""
        out: list[tuple[Any, ...]] = []
        for item in items:
            if item[0] == "code":
                out.append(item)
            elif item[0] == "loop":
                r, body = item[1], item[2]
                stores = any(self.nest.store(s) for s in flatten(body))  # (a step stores a tile: it stays a phase)
                if r.count == 1 and r.mask is None and not keep_first and not stores:
                    out.append(("code", (_bind(r, body[0]), *body)))
                else:
                    out.append(item)
                keep_first = False
            else:
                r, inner = item[1], self._inline(item[2])
                if r.count == 1:  # (its variable, read across phases, is bound at every step: codegen.py)
                    out += inner
                    continue
                if not any(sub[0] == "loop" for sub in inner):  # every iteration takes a step at least
                    inner = self._inline(item[2], keep_first=True)
                out.append(("repeat", r, inner))
        return out

    def _operands(
        self, phases: tuple[Phase, ...], segments: tuple[tuple[ast.stmt, ...], ...], repeats: tuple[Repeat, ...]
    ) -> tuple[tuple[Operand, ...], dict[str, Operand]]:
        operands: list[Operand] = []
        by_key: dict[str, Operand] = {}
        params = {name: i for i, name in enumerate(self.nest.params)}
        # The phase whose steps run each tile read (for windows only a repeat loop moves).
        regions = [(0, segments[0])] + [(rep.last - 1, rep.after) for rep in repeats]
        for p, phase in enumerate(phases):
            regions += [(p, phase.before), (p, phase.body), (p, segments[p + 1])]
        runs_in = {
            (n.lineno, n.col_offset): p
            for p, stmts in regions
            for stmt in stmts
            for n in ast.walk(stmt)
            if isinstance(n, ast.Subscript)
        }
        in_bodies = {
            (n.lineno, n.col_offset)
            for phase in phases
            for stmt in phase.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Subscript)
        }
        outside: set[str] = set()  # the keys of reads outside every phase's body
        repeat_vars = {rep.range.var for rep in repeats}
        subscripts = [n for stmt in self.nest.tile for n in ast.walk(stmt) if isinstance(n, ast.Subscript)]
        for node in sorted(subscripts, key=lambda n: (n.lineno, n.col_offset)):  # in source order
            is_index = isinstance(node.value, ast.Name) and node.value.id in self.indices
            param = params.get(node.value.id) if isinstance(node.value, ast.Name) and not is_index else None
            if param is None:
                continue
            spec = self.inputs[param]
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            tile = self._tile(node.value.id, spec.shape, spec.dtype, index, node)
            moving = {v for v in tile.loops if self.ranges[v].count > 1}  # loops of one block don't move it
            phase = next((p for p, ph in enumerate(phases) if ph.range.var in moving), None)
            if phase is None and moving & repeat_vars:  # streamed with the phase whose steps read it
                phase = self.read_phases[(node.lineno, node.col_offset)] = runs_in[(node.lineno, node.col_offset)]
            key = _read_key(node, self.read_phases)
            if (node.lineno, node.col_offset) not in in_bodies:
                outside.add(key)
            if key in by_key:
                continue
            if not moving:
                kind = Kind.ONCE
            elif (node.lineno, node.col_offset) in self.read_phases:  # only repeat (and parallel) loops move it
                kind = Kind.ITERATION
            elif phase is not None or not phases:
                kind = Kind.STREAM
            else:
                kind = Kind.TILE
            taken = {*self.nest.params, *(op.name for op in operands)}
            name = tile.array if tile.array not in {op.tile.array for op in operands} else _fresh(tile.array, taken)
            operand = Operand(name, param, tile, kind, phase)
            operands.append(operand)
            by_key[key] = operand
        for key, operand in by_key.items():
            if operand.kind is Kind.ITERATION and key not in outside:
                by_key[key] = operands[operands.index(operand)] = replace(operand, in_body=True)
        return tuple(operands), by_key

    def _rings(
        self, parallel: tuple[Range, ...], operands: tuple[Operand, ...], aliased: Mapping[str, int | None]
    ) -> tuple[Ring, ...]:
        """The inputs gathered across chips, once their tiles are known to take whole rows of one chip's shard
        from the outermost parallel loop, which walks every tile (no mask skips one)."""
        nest = self.nest
        rings = []
        for gather in nest.all_gathers:
            node, name, source = gather.node, gather.name, gather.source
            shape = self.inputs[nest.params.index(name)].shape
            if source in self.indices:
                raise self.error(f"`{source}` is an index array (in SMEM): gather an input read as tiles", node)
            if nest.params.index(source) in aliased.values():
                raise self.error(f"`{source}` is gathered across chips: it can't be updated in place too", node)
            reads_ = [op for op in operands if op.tile.array == name]
            if not reads_:
                raise self.error(f"`{name}` is gathered across chips, but no tile reads it", node)
            masked = next((loop.mask for loop in nest.parallel if loop.mask is not None), None)
            if masked is not None:
                raise self.error(
                    f"a kernel gathering `{name}` across chips computes every tile: its parallel loops can't be masked",
                    masked,
                )
            r, rows = parallel[0], self.inputs[nest.params.index(source)].shape[0]
            for op in reads_:
                window = op.tile.windows[0]
                rows_ = (window.loop, window.offset, window.size, window.index, window.rows)
                if rows_ != (r.var, 0, r.step, None, None):
                    raise self.error(
                        f"{op.tile.text}: tiles of `{name}`, gathered across chips, take its rows "
                        f"`{r.var} : {r.var} + {r.step}` of the outermost parallel loop (they arrive a shard at once)",
                        node,
                    )
            if r.bounds is not None or r.start != 0 or r.count * r.step != shape[0]:
                raise self.error(
                    f"the outermost parallel loop, `{r.var}`, walks all of `{name}`'s {shape[0]} rows from 0, gathered "
                    "across chips",
                    nest.parallel[0].node,
                )
            if rows % r.step:
                raise self.error(
                    f"`{r.var}` steps by {r.step}, which must divide one chip's {rows} rows of `{name}`",
                    nest.parallel[0].node,
                )
            size = shape[0] // rows
            rings.append(Ring(name, source, gather.axis, size, rows, r.var, rows // r.step))
        return tuple(rings)

    def _packed(self, reads_: Mapping[str, Operand]) -> dict[tuple[int, int], str]:
        """The calls of `autokernel.pairs` in tile code, each on a tile read of a 16-bit (..., 2, d) window."""
        found = {}
        for stmt in self.nest.tile:
            for node in ast.walk(stmt):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)):
                    continue
                try:
                    func = self.eval(node.func)
                except KernelError:
                    continue  # a method of a value, a local function: not `pairs`
                if func is not packing.pairs:
                    continue
                arg = node.args[0] if len(node.args) == 1 and not node.keywords else None
                operand = reads_.get(_read_key(arg, self.read_phases)) if isinstance(arg, ast.Subscript) else None
                if operand is None:
                    raise self.error("autokernel.pairs takes a tile read of an input, `pairs(x[...])`", node)
                tile = operand.tile
                if tile.dtype.itemsize != 2 or len(tile.shape) < 2 or tile.shape[-2] != 2:
                    raise self.error(
                        f"autokernel.pairs takes a 16-bit tile of pairs of rows (..., 2, d), not {tile.dtype.name}"
                        f"{list(tile.shape)}",
                        node,
                    )
                found[(node.lineno, node.col_offset)] = operand.name
        return found

    def _unpacked(self, output_tiles: Mapping[str, Tile]) -> frozenset[str]:
        """The outputs stored from words (`unpairs(words, dtype)`, the stored value), into 16-bit tiles of
        pairs of rows; `unpairs` anywhere else is an error."""
        stored = {id(store.value): name for name, store in self.nest.stores.items()}
        found = set()
        for stmt in self.nest.tile:
            for node in ast.walk(stmt):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)):
                    continue
                try:
                    func = self.eval(node.func)
                except KernelError:
                    continue  # a method of a value, a local function: not `unpairs`
                if func is not packing.unpairs:
                    continue
                name = stored.get(id(node))
                if name is None or len(node.args) != 2 or node.keywords:
                    raise self.error(
                        "autokernel.unpairs(words, dtype) is a stored value, `out.at[...].set(unpairs(w, out.dtype))`",
                        node,
                    )
                tile = output_tiles[name]
                if tile.dtype.itemsize != 2 or len(tile.shape) < 2 or tile.shape[-2] != 2:
                    raise self.error(
                        f"autokernel.unpairs stores a 16-bit tile of pairs of rows (..., 2, d), not {tile.dtype.name}"
                        f"{list(tile.shape)}",
                        node,
                    )
                found.add(name)
        return frozenset(found)

    def _packed_names(self, packed: Mapping[tuple[int, int], str]) -> dict[str, str]:
        """The names the tile code assigns once, to words (`words = autokernel.pairs(x[...])`)."""
        counts: dict[str, int] = {}
        for stmt in flatten(self.nest.tile):
            for name in assigned([stmt]):
                counts[name] = counts.get(name, 0) + 1
        found = {}
        for stmt in flatten(self.nest.tile):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                value, name = stmt.value, stmt.targets[0].id
                if isinstance(value, ast.Call) and (value.lineno, value.col_offset) in packed and counts[name] == 1:
                    found[name] = packed[(value.lineno, value.col_offset)]
        return found

    def _index_arrays(self) -> list[Index]:
        """The inputs read element by element (`lengths[b]`), in parameter order: int32, read only that way,
        indexed by loop variables and constants."""
        nest, params = self.nest, set(self.nest.params)
        trees: list[ast.AST] = [*nest.tile, *(loop.mask for loop in nest.reductions if loop.mask is not None)]
        trees += [bound for loop in nest.loops for bound in loop.bounds]
        gathers = [n for tree in trees for n in _gathers(tree, params)]
        elements = [n for tree in trees for n in element_reads(tree, params)] + gathers
        found = []
        for p, name in enumerate(nest.params):
            mine = [n for n in elements if n.value.id == name]  # type: ignore[attr-defined]
            if not mine:
                continue
            if p >= len(nest.inputs):
                raise self.error(f"`{name}` is gathered across chips: kernel code reads tiles of it", mine[0])
            ids = {id(n) for n in mine}
            tiles = [
                n
                for tree in trees
                for n in ast.walk(tree)
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == name
                if id(n) not in ids
            ]
            if tiles:
                raise self.error(
                    f"`{name}` is read element by element (an index array, kept in SMEM) and as tiles: read it one "
                    "way or the other",
                    tiles[0],
                )
            spec = self.inputs[p]
            if np.dtype(spec.dtype) != np.int32:
                raise self.error(
                    f"`{name}` is read element by element, an index array (kept in SMEM): it must be int32, not "
                    f"{np.dtype(spec.dtype).name}",
                    mine[0],
                )
            for node in mine:
                index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                text = ast.unparse(node)
                if len(index) != len(spec.shape):
                    raise self.error(f"{text}: index every one of {name}'s {len(spec.shape)} dims", node)
                index = [i for i in index if not isinstance(i, ast.Slice)] + [  # (a gather's slice's bounds)
                    b for i in index if isinstance(i, ast.Slice) for b in (i.lower, i.upper) if b is not None
                ]
                nested = [n for i in index for n in ast.walk(i) if isinstance(n, ast.Subscript)]
                if any(not (isinstance(n.value, ast.Name) and n.value.id in params) for n in nested):
                    raise self.error(
                        f"{text}: an index array's elements are indexed by loop variables, constants and other "
                        "elements",
                        node,
                    )
            found.append(Index(name, p, tuple(spec.shape), np.dtype(spec.dtype)))
        return found

    def _computed_names(self) -> set[str]:
        """The values the tile code computes whose elements index tile reads or bound loops: computed index
        arrays."""
        made = assigned(self.nest.tile)
        arrays = set(self.nest.params)
        trees: list[ast.AST] = [bound for loop in self.nest.loops for bound in loop.bounds]
        for stmt in flatten(self.nest.tile):
            trees += [
                n.slice
                for n in ast.walk(stmt)
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in arrays
            ]
        return {n.value.id for tree in trees for n in element_reads(tree, made)}  # type: ignore[attr-defined]

    def _computed(
        self,
        types: Mapping[int, Mapping[str, jax.ShapeDtypeStruct]],
        phases: tuple[Phase, ...],
        segments: tuple[tuple[ast.stmt, ...], ...],
        repeats: tuple[Repeat, ...],
        operands: tuple[Operand, ...],
    ) -> tuple[Computed, ...]:
        """The computed index arrays: each assigned once, int32 of one or two dims, by code after a phase
        (which runs at its last step), and read by the windows a later phase streams, or its loop's bounds,
        in the same iteration of the repeat loop the code is in (if any)."""
        found = []
        for name in sorted(self.computed):
            stmts = [stmt for stmt in flatten(self.nest.tile) if name in assigned([stmt])]
            if len(stmts) != 1:
                raise self.error(f"`{name}`'s elements index windows or bound loops: assign it once", stmts[1])
            stmt = stmts[0]
            k = next((k for k, seg in enumerate(segments) if any(x is stmt for x in flatten(seg))), 0)
            after = next((r for r, rep in enumerate(repeats) if any(x is stmt for x in flatten(rep.after))), None)
            if not k and after is None:
                raise self.error(
                    f"`{name}`'s elements index windows or bound loops, so it is computed after a reduction loop "
                    "(the kernel learns it at the loop's last step): put one before its statement (a loop of one "
                    "iteration runs inline, as code)",
                    stmt,
                )
            phase = repeats[after].last - 1 if after is not None else k - 1
            repeat = None if after is not None else phases[phase].repeat
            aval = types[id(stmt)][name]
            if np.dtype(aval.dtype) != np.int32 or len(aval.shape) not in (1, 2):
                raise self.error(
                    f"`{name}`'s elements index windows or bound loops: it must be int32 of 1 or 2 dims, not "
                    f"{np.dtype(aval.dtype).name}{list(aval.shape)}",
                    stmt,
                )
            consumers = set()
            for op in operands:
                if not any(element_reads(w.index, {name}) for w in op.tile.windows if w.index is not None):
                    continue
                if op.kind is not Kind.STREAM:
                    raise self.error(
                        f"{op.tile.text}: only windows a reduction loop moves may be placed by `{name}`, which the "
                        "tile code computes",
                        stmt,
                    )
                consumers.add(op.phase)
            consumers |= {
                p for p, ph in enumerate(phases) if ph.range.bounds and element_reads(ph.range.bounds[1], {name})
            }
            for p in sorted(consumers):  # type: ignore[type-var]
                if p <= phase or (repeat is not None and phases[p].repeat != repeat):  # type: ignore[operator]
                    where = "in the same iteration of its repeat loop" if repeat is not None else "after it"
                    raise self.error(
                        f"`{name}` is computed at the last step of reduction loop "
                        f"`{phases[phase].range.var}`: only later loops {where} may read it",
                        stmt,
                    )
            shape, dtype = tuple(aval.shape), np.dtype(aval.dtype)
            found.append(Computed(name, shape, dtype, phase, repeat, tuple(sorted(consumers))))  # type: ignore[arg-type]
        return tuple(found)

    def _check_element_bounds(self) -> None:
        """Elements of index arrays read at loop values outside the array are errors, found when planning by
        evaluating each index at every value of the loops it reads (index arrays are in SMEM, unchecked)."""
        nest = self.nest
        trees: list[ast.AST] = [*nest.tile, *(loop.mask for loop in nest.reductions if loop.mask is not None)]
        trees += [bound for loop in nest.loops for bound in loop.bounds]
        for node in (n for tree in trees for n in element_reads(tree, self.indices)):
            index = self.indices[node.value.id]  # type: ignore[attr-defined]
            dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for expr, dim in zip(dims, index.shape, strict=True):
                loops = [self.ranges[v] for v in sorted(reads(expr) & self.ranges.keys())]
                if any(r.bounds is not None for r in loops) or element_reads(expr, self.indices):
                    continue  # known only when the kernel runs (a runtime range's values, an element): clamped
                for values in itertools.product(*(range(r.count) for r in loops)):
                    at = {r.var: r.value(b) for r, b in zip(loops, values, strict=True)}
                    value = evaluate(nest, self.env, expr, at)
                    if not 0 <= value < dim:
                        where = ", ".join(f"{k}={v}" for k, v in at.items())
                        raise self.error(
                            f"{ast.unparse(node)}: reads element {value} of a dim of {dim}"
                            + (f" at {where}" if where else ""),
                            node,
                        )

    def _check_cursor(self, stored_by: dict[str, int | None], outputs: dict[str, Any]) -> None:
        """A kernel with runtime ranges (or computed index arrays) finds its steps as it runs, from the bounds
        (and masks): outputs are stored per tile; with a runtime count of tiles, nothing masks tiles and every
        output is an input updated in place (no tile stores the rest)."""
        bounded = [loop.node for loop in self.nest.loops if self.ranges[loop.var].bounds is not None]
        node = bounded[0] if bounded else self.nest.parallel[0].node
        if self.ranges[self.nest.parallel[0].var].bounds is not None:
            plain = [name for name, (_, _, fill) in outputs.items() if fill is not None]
            if plain:
                raise self.error(
                    f"a kernel with a runtime count of tiles stores only outputs updated in place (`{plain[0]} = x`)",
                    node,
                )
        if any(phase is not None for phase in stored_by.values()):
            raise self.error("a kernel with runtime loop bounds stores its outputs per tile", node)

    def _storing_phase(self, name: str, phases: tuple[Phase, ...]) -> int | None:
        """The phase whose steps store a tile of output `name` each, if a reduction loop's body stores it."""
        r = self.nest.stored_in[name]
        if r is None:
            return None
        var = self.nest.reductions[r].var
        p = next(p for p, phase in enumerate(phases) if phase.range.var == var)
        node = self.nest.stores[name].node
        if phases[p].range.mask is not None or any(q.mask is not None for q in self.ranges.values()):
            raise self.error(
                f"output `{name}` is stored at every step of `{var}`'s loop, so no mask may skip tiles or steps "
                "(the kernel couldn't fill what they skip)",
                node,
            )
        return p

    def _output_tile(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        fill: Any,
        parallel: tuple[Range, ...],
    ) -> Tile:
        store = self.nest.stores[name]
        tile = self._tile(name, shape, dtype, store.index, store.node)
        uses = [w.loop for w in tile.windows if w.loop] + [v for w in tile.windows for v in sorted(w.index_loops)]
        if any(w.index is not None and w.size is not None for w in tile.windows):
            raise self.error(f"{tile.text}: an output tile isn't gathered through an index array", store.node)
        scatter = next((w.index for w in tile.windows if w.index is not None), None)
        scatter = scatter or next((w.rows[0] for w in tile.windows if w.rows is not None), None)
        if scatter is not None and fill is not None:
            raise self.error(
                f"{tile.text}: an index array places this tile, so no tile might store the rest of `{name}`: "
                f"allocate it as an input updated in place (`{name} = x`), whose values the rest keeps",
                scatter,
            )
        if any(w.rows is not None for w in tile.windows[-2:]):  # (a TPU copies whole (8, 128) tiles there)
            raise self.error(
                f"{tile.text}: ragged rows of an output go in a dim before the last two (along those a TPU copies "
                "whole tiles, which would store past the stop)",
                store.node,
            )
        r = self.nest.stored_in[name]
        placing = [*parallel, *([self.ranges[self.nest.reductions[r].var]] if r is not None else [])]
        for r in placing:
            # (Index arrays place a scattered tile: any of its dims may read the loop, the arrays' values
            # keep its tiles apart.)
            if uses.count(r.var) != 1 and not (scatter is not None and r.var in uses):
                what = "every parallel loop places" if r in parallel else f"`{r.var}`'s steps each store"
                raise self.error(
                    f"{tile.text}: {what} each output tile, so `{r.var}` indexes exactly one dim of the output",
                    store.node,
                )
        for window, dim in zip(tile.windows, shape, strict=True):
            r = self.ranges.get(window.loop) if window.loop else None
            if window.index is not None or window.rows is not None:
                continue  # a window the kernel chooses when it runs: it needn't cover the dim
            if r is None:
                if window.offset != 0 or window.size != dim:
                    raise self.error(f"{tile.text}: dims no loop places are stored whole, with `:`", store.node)
            elif (window.size or 1) != r.step or (fill is not None and (r.start != 0 or r.count * r.step != dim)):
                # (An input updated in place may be stored in part: the rest keeps the input's values.)
                raise self.error(
                    f"{tile.text}: loop `{r.var}` must cover the dim of {dim} exactly, one step per tile", store.node
                )
        return tile

    def _check_store(self, tile: Tile, value: jax.ShapeDtypeStruct | None) -> None:
        """A store's value must broadcast to its tile, as `.at[...].set` requires."""
        node = self.nest.stores[tile.array].node
        if value is None:
            raise self.error(f"{tile.text}: the stored value must be an array or a number", node)
        try:
            ok = np.broadcast_shapes(value.shape, tile.shape) == tile.shape
        except ValueError:
            ok = False
        if not ok:
            raise self.error(f"{tile.text}: can't store a value of shape {value.shape} in a {tile.shape} tile", node)

    # ------------------------------------------------------------------ names

    def _check_reads_before_assignment(self) -> None:
        """Python raises UnboundLocalError for a local read before it is assigned: in a tile, that is a
        value expected to carry over from the previous tile, which nothing does."""
        nest = self.nest
        early = (first_reads(nest.tile) & assigned(nest.tile)) - set(nest.outputs)
        for stmt in flatten(nest.tile):
            names = sorted(reads(stmt) & early)
            if names:
                raise self.error(
                    f"`{names[0]}` is read before the tile assigns it: nothing carries over from one tile to the next",
                    stmt,
                )

    def _constants(self) -> dict[str, Any]:
        """The names tile code reads that aren't its own: the module's, the closure's and the prologue's."""
        nest = self.nest
        own = assigned(nest.tile) | {r.var for r in nest.loops} | set(nest.params) | set(nest.outputs)
        masks = [r.mask for r in self.ranges.values() if r.runtime]  # evaluated in the generated source
        masks += [b for r in self.ranges.values() if r.bounds for b in r.bounds]  # (and runtime bounds)
        names = set().union(*(reads(stmt) for stmt in flatten(nest.tile)), *(reads(m) for m in masks)) - own
        # (`autokernel.pairs` becomes a read of the VMEM buffer: the kernel needs neither it nor autokernel.)
        names = {n for n in names if n in self.env and not _is_autokernel(self.env[n])}
        return {name: self.env[name] for name in sorted(names)}

    def _scopes(self) -> tuple[str, ...]:
        scopes = []
        for stmt in flatten(self.nest.tile):
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    call: ast.Call = item.context_expr  # type: ignore[assignment]  # parse.py: calls
                    name = self.eval(call.args[0]) if len(call.args) == 1 and not call.keywords else None
                    if self.eval(call.func) is not jax.named_scope or not isinstance(name, str):
                        raise self.error("tile code opens only `with jax.named_scope(name):` blocks", stmt)
                    scopes.append(name)
        return tuple(scopes)


@dataclass(frozen=True)
class _Types:
    ops: int
    carried: dict[str, jax.ShapeDtypeStruct]
    assigned: dict[int, dict[str, jax.ShapeDtypeStruct]]
    stores: dict[str, jax.ShapeDtypeStruct]


_LAST = "last"  # the code that ends a tile


class _Sequence:
    """A tile's code in the order it runs, as regions: the code at its first step; for each phase, the
    repeat loop's code before it, its body (twice: what a step leaves, the next step starts from) and the
    code after it; after a repeat loop's last phase in its last iteration, the code after the loop. A
    repeat loop runs twice over: what an iteration leaves, the next starts from."""

    def __init__(
        self, phases: tuple[Phase, ...], segments: tuple[tuple[ast.stmt, ...], ...], repeats: tuple[Repeat, ...]
    ) -> None:
        self.phases, self.segments, self.repeats = phases, segments, repeats

    def unrolled(self) -> Iterator[tuple[int, int, tuple[ast.stmt, ...]]]:
        """(phase, iteration, the code after it) in the order a tile runs them, repeat loops twice over."""
        p = 0
        while p < len(self.phases):
            r = self.phases[p].repeat
            if r is None:
                yield p, 0, self.segments[p + 1]
                p += 1
                continue
            rep = self.repeats[r]
            for it in (0, 1):
                for q in range(rep.first, rep.last):
                    end = rep.after if it == 1 and q == rep.last - 1 else ()
                    yield q, it, self.segments[q + 1] + end
            p = rep.last

    def regions(self) -> list[tuple[str, tuple[ast.stmt, ...]]]:
        """The regions, labelled: "seg..." for code (the last: `_LAST`), "body<phase>.<iteration>" for a
        phase's body at its first step, "+" after it for its next."""
        regions = [("seg0", self.segments[0])]
        for n, (p, it, end) in enumerate(self.unrolled()):
            phase = self.phases[p]
            if phase.before:
                regions.append((f"seg-before{n}", phase.before))
            regions += [(f"body{p}.{it}", phase.body), (f"body{p}.{it}+", phase.body), (f"seg-after{n}", end)]
        regions[-1] = (_LAST, regions[-1][1])
        return regions

    def carried(self) -> set[str]:
        """Values a tile carries from one step to another (kept in VMEM): read by code that runs in a scope of
        its own (the kernel's functions for the code at a tile's first step, a phase's code before it, its
        body; and at its last step, its body with the code after it) before assigning them, after an earlier
        scope assigned them. (Where a mask may leave a phase's tile one empty step, whose body doesn't run,
        the code after it is a scope apart from the body.)"""
        scopes = [self.segments[0]]
        for p, _, end in self.unrolled():
            phase = self.phases[p]
            scopes += [phase.before, phase.body]
            scopes += [phase.body, end] if phase.range.mask is not None else [phase.body + end]
        carried: set[str] = set()
        made: set[str] = set()
        for scope in scopes:
            carried |= first_reads(scope) & made
            made |= assigned(scope)
        return carried


class _TypeTrace:
    """Runs the tile code once with abstract values (`jax.make_jaxpr`): its regions in the order they run
    (`_Sequence`). Records the type of every value assigned and stored."""

    def __init__(
        self,
        planner: _Planner,
        operands: tuple[Operand, ...],
        reads_: Mapping[str, Operand],
        carried: set[str],
        sequence: _Sequence,
    ) -> None:
        self.planner, self.nest = planner, planner.nest
        self.operands, self.reads, self.carried = operands, reads_, carried
        self.regions = sequence.regions() if sequence.phases else [(_LAST, sequence.segments[0])]
        self.notes: dict[tuple[str, int], dict[str, Any]] = {}  # (region, id(stmt)) -> name -> type
        self.stores: dict[str, jax.ShapeDtypeStruct] = {}
        self.stmts: dict[int, ast.stmt] = {}

    def run(self) -> _Types:
        fn = self._function()
        specs = [jax.ShapeDtypeStruct(op.tile.shape, op.tile.dtype) for op in self.operands]
        loop_vars = [jax.ShapeDtypeStruct((), jnp.int32, weak_type=True) for _ in self.planner.ranges]
        indices = [jax.ShapeDtypeStruct(index.shape, index.dtype) for index in self.planner.indices.values()]
        try:
            traced = jax.make_jaxpr(fn)(*specs, *loop_vars, *indices)
        except Exception as e:
            frame = next(
                (f for f in reversed(traceback.extract_tb(e.__traceback__)) if f.filename == self.nest.filename),
                None,
            )
            node = ast.Pass(lineno=frame.lineno) if frame else None
            raise self.nest.error(f"the tile code fails for these input shapes: {e!r}", node) from e
        self._check_matmuls(traced.jaxpr)
        return _Types(_count_ops(traced.jaxpr), self._carried_types(), self._assigned(), self.stores)

    def _check_matmuls(self, jaxpr: Any) -> None:
        """The MXU accumulates in 32 bits: a matmul whose result is narrower (plain JAX's `bf16 @ bf16`, in
        a helper function, say) can't compile for the TPU."""
        for eqn in jaxpr.eqns:
            for inner in jax_core.jaxprs_in_params(eqn.params):
                self._check_matmuls(inner)
            if eqn.primitive.name == "dot_general" and np.dtype(eqn.outvars[0].aval.dtype).itemsize < 4:
                raise self.nest.error(
                    f"the matmul at {source_info_util.summarize(eqn.source_info)} accumulates in "
                    f"{eqn.outvars[0].aval.dtype}, but the TPU's MXU accumulates in 32 bits: use `@` in kernel "
                    "code, or `preferred_element_type=jnp.float32`"
                )

    def _function(self) -> Callable:
        """The tile code as one function of its operands and loop variables, with type hooks."""
        body: list[ast.stmt] = []
        for label, stmts in self.regions:
            body += [s for stmt in stmts for s in self._instrument(stmt, label)]
        args = [f"__operand{i}" for i in range(len(self.operands))] + [*self.planner.ranges, *self.planner.indices]
        fdef = ast.FunctionDef(
            name="__tile__",
            args=ast.arguments([], [ast.arg(a) for a in args], None, [], [], None, []),
            body=body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        module = ast.fix_missing_locations(ast.Module([ast.copy_location(fdef, self.nest.function)], []))
        namespace = dict(self.planner.env)
        namespace |= mxu.FUNCTIONS | {"__note__": self._note, "__store__": self._store}
        exec(compile(module, self.nest.filename, "exec"), namespace)
        return namespace["__tile__"]

    def _instrument(self, stmt: ast.stmt, segment: str) -> list[ast.stmt]:
        """`stmt` reading operand arguments, calling `__note__` after it assigns, `__store__` for a store."""
        self.stmts[id(stmt)] = stmt
        store = self.nest.store(stmt)
        if store is not None:
            value = self._rewrite(copy.deepcopy(store.value))
            call = ast.Call(ast.Name("__store__", ast.Load()), [ast.Constant(store.output), value], [])
            return [ast.copy_location(ast.Expr(call), stmt)]
        if isinstance(stmt, ast.With):
            new = copy.copy(stmt)
            new.body = [s for inner in stmt.body for s in self._instrument(inner, segment)]
            return [new]
        new = self._rewrite(copy.deepcopy(stmt))
        names = sorted(assigned([stmt]))
        values = ast.Dict([ast.Constant(n) for n in names], [ast.Name(n, ast.Load()) for n in names])
        key = ast.Tuple([ast.Constant(segment), ast.Constant(id(stmt))], ast.Load())
        note = ast.Expr(ast.Call(ast.Name("__note__", ast.Load()), [key, values], []))
        return [new, ast.copy_location(note, stmt)]

    def _rewrite[T: ast.AST](self, tree: T) -> T:
        index = {op.name: i for i, op in enumerate(self.operands)}
        reads_, read_phases = self.reads, self.planner.read_phases

        class Reads(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                operand = reads_.get(_read_key(node, read_phases))
                if operand is None:
                    return self.generic_visit(node)  # type: ignore[return-value]
                return ast.copy_location(ast.Name(f"__operand{index[operand.name]}", ast.Load()), node)

        return mxu.rewrite_matmuls(Reads().visit(tree))

    def _note(self, key: tuple[str, int], values: dict[str, Any]) -> None:
        self.notes[key] = {name: _type(value) for name, value in values.items()}

    def _store(self, output: str, value: Any) -> None:
        self.stores[output] = _type(value)

    def _carried_types(self) -> dict[str, jax.ShapeDtypeStruct]:
        """The one type each carried value keeps: what the first phase body that assigns it makes it (else
        the code that assigns it last).

        The kernel keeps a carried value in a VMEM buffer of that type, so the value each region (the code
        around the phases, a phase's body) leaves it with must have the type; values on their way (assigned
        earlier in the region) are locals and may have any. The code around the phases may also start it with
        a number (`m = -1e30`), which the kernel broadcasts to the carried type. After the last phase,
        assignments make new values (codegen renames them), so they may have any type."""
        types = {}
        for name in sorted(self.carried):
            bodies = [label for label, _ in self.regions if label.startswith("body") and not label.endswith("+")]
            segments = [label for label, _ in self.regions if label.startswith("seg")]
            kept = next(
                (t for t in (self._last(b, name) for b in bodies) if t is not None),
                next((t for t in (self._last(s, name) for s in [*reversed(segments), _LAST]) if t is not None), None),
            )
            node = self._assignment(name)
            if kept is None or not kept.shape:
                raise self.nest.error(
                    f"`{name}` is carried between reduction steps, so it must be an array (kept in VMEM)", node
                )
            # What each region leaves behind (its last assignment) is stored: that must have the type.
            left: dict[str, tuple[int, Any]] = {}
            for (label, stmt), values in self.notes.items():
                if name in values:
                    left[label] = (stmt, values[name])
            for label, (stmt, value) in left.items():
                code = label.startswith("seg")
                if label == _LAST or _show(value) == _show(kept) or (code and _starts(value, kept)):
                    continue
                raise self.nest.error(
                    f"`{name}` is carried between reduction steps as {_show(kept)}, but this makes it "
                    f"{_show(value)}: a carried value keeps one type (start it with that type)",
                    self.stmts[stmt],
                )
            types[name] = jax.ShapeDtypeStruct(kept.shape, kept.dtype)
        return types

    def _last(self, segment: str, name: str) -> jax.ShapeDtypeStruct | None:
        """The type of `name` after the segment's last assignment of it, if it assigns it."""
        found = None
        for (seg, _), values in self.notes.items():
            if seg == segment and name in values:
                found = values[name]
        return found

    def _assignment(self, name: str) -> ast.stmt | None:
        return next((self.stmts[i] for (_, i), values in self.notes.items() if name in values), None)

    def _assigned(self) -> dict[int, dict[str, jax.ShapeDtypeStruct]]:
        assigned_: dict[int, dict[str, Any]] = {}
        for (_, stmt), values in self.notes.items():
            assigned_.setdefault(stmt, {}).update(values)  # a body statement: its type in the second step
        return assigned_


def _is_autokernel(value: Any) -> bool:
    """Is `value` autokernel's package or one of its kernel-code functions (which only the planner reads:
    the kernel reads VMEM, or copies rows, in their place)?"""
    return (
        value is packing.pairs
        or value is packing.unpairs
        or value is ragged.rows
        or (isinstance(value, types.ModuleType) and value.__name__ == __package__)
    )


def _dims(node: ast.Subscript) -> list[ast.expr]:
    """A subscript's indices, one per dim."""
    return list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]


def _gathers(tree: ast.AST, arrays: Iterable[str]) -> list[ast.Subscript]:
    """Slices of `arrays` that are an index of another subscript: windows gathered through an index array,
    `x[idx[a : a + n], :]`."""
    names = set(arrays)
    return [
        i
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        for i in _dims(n)
        if isinstance(i, ast.Subscript) and isinstance(i.value, ast.Name) and i.value.id in names
        if any(isinstance(d, ast.Slice) for d in _dims(i))
    ]


def _count_ops(jaxpr: Any) -> int:
    """Equations in a jaxpr, those of the jaxprs inside them (calls of jitted functions) included."""
    return sum(1 + sum(_count_ops(inner) for inner in jax_core.jaxprs_in_params(eqn.params)) for eqn in jaxpr.eqns)


def _bind(r: Range, at: ast.AST) -> ast.stmt:
    """`var = start`: a loop of one iteration's variable, bound where its body runs inline."""
    bind = ast.Assign([ast.Name(r.var, ast.Store())], ast.Constant(r.start))
    return ast.fix_missing_locations(ast.copy_location(bind, at))


def _fresh(base: str, taken: set[str]) -> str:
    """`base2`, `base3`, ...: the first not in `taken`."""
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def _type(value: Any) -> jax.ShapeDtypeStruct | None:
    """The type of a value tile code computed, or None if it isn't an array or scalar."""
    try:
        aval = jax.typeof(value)
    except TypeError:
        return None
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)


def _starts(value: jax.ShapeDtypeStruct | None, kept: jax.ShapeDtypeStruct) -> bool:
    """Can `value` start a carried value of type `kept`? A number (a weakly typed scalar), or a scalar of
    its dtype: the kernel broadcasts it, which elementwise code at the first step does too."""
    return value is not None and not value.shape and (value.weak_type or value.dtype == kept.dtype)


def _show(t: jax.ShapeDtypeStruct | None) -> str:
    if t is None:
        return "not an array"
    return f"{t.dtype}{list(t.shape)}"
