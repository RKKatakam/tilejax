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
masks), and integer indices of tile reads (`w[expert[e], :, :]`: runtime windows).
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import itertools
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src import source_info_util

from . import mxu, v5e
from .errors import KernelError
from .parse import Loop, Nest
from .tilecode import assigned, element_reads, first_reads, flatten, reads

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
    "Tile",
    "Window",
    "make_plan",
]


@dataclass(frozen=True)
class Range:
    """A loop's iterations: `start + block * step` for block in `range(count)`."""

    var: str
    start: int
    step: int
    count: int
    mask: ast.expr | None  # `if mask: continue` opening the loop's body
    runtime: bool = False  # does the mask read index arrays (so it is evaluated when the kernel runs)?

    def value(self, block: int) -> int:
        return self.start + block * self.step


@dataclass(frozen=True)
class Window:
    """How a tile indexes one array dim: from element `offset` (plus the value of `loop`, if any), `size`
    elements; `size` None is an integer index, which drops the dim from the tile. An integer index may be an
    element of an index array (`index`, `expert[e]`), read when the kernel runs: `index_loops` move it."""

    loop: str | None
    offset: int
    size: int | None
    index: ast.Subscript | None = None
    index_loops: frozenset[str] = frozenset()


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


@dataclass(frozen=True)
class Index:
    """An index array: an input read element by element, kept whole in SMEM."""

    name: str
    param: int  # which input
    shape: tuple[int, ...]
    dtype: np.dtype


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
    fill: Any  # the allocation's value (0 for jnp.zeros)
    tile: Tile
    value: jax.ShapeDtypeStruct  # the type of the value the tile code stores
    phase: int | None = None  # the phase whose steps each store a tile of it (None: a tile per output tile)


@dataclass(frozen=True)
class Carried:
    """A value a tile carries from one reduction step to the next: VMEM scratch in the kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


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
    reads: Mapping[str, Operand]  # `read_key` of a tile read -> its operand
    # Reads of windows only a repeat loop moves, outside phase bodies: (line, column) -> the phase whose
    # steps run them (they stream with that phase).
    read_phases: Mapping[tuple[int, int], int]
    outputs: tuple[Output, ...]
    carried: Mapping[str, Carried]
    assigned_types: Mapping[int, Mapping[str, jax.ShapeDtypeStruct]]  # id(stmt) -> the types it assigns
    constants: Mapping[str, Any]  # names the tile code reads from the module, closure and prologue
    scopes: tuple[str, ...]  # the names of the tile code's `jax.named_scope`s
    tile_ops: int  # operations in a tile's code traced as plain JAX (each phase body twice): its size
    env: dict[str, Any]  # the function's module and closure values, with the prologue's names over them

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
        parallel = tuple(self._range(loop) for loop in self.nest.parallel)
        reductions = tuple(self._range(loop) for loop in self.nest.reductions)
        repeat_loops = tuple(self._range(rep.loop) for rep in self.nest.repeats)
        self.ranges = {r.var: r for r in (*parallel, *reductions, *repeat_loops)}
        self._check_element_bounds()
        phases, segments, repeats = self._phases(reductions, repeat_loops)
        operands, reads_ = self._operands(phases, segments, repeats)
        stored_by = {name: self._storing_phase(name, phases) for name in self.nest.outputs}
        output_tiles = {name: self._output_tile(name, *outputs[name], parallel) for name in self.nest.outputs}
        self._check_reads_before_assignment()
        sequence = _Sequence(phases, segments, repeats)
        carried_names = sequence.carried() - set(self.nest.outputs)
        scopes = self._scopes()
        types = _TypeTrace(self, operands, reads_, carried_names, sequence).run()
        for name in self.nest.outputs:
            self._check_store(output_tiles[name], types.stores[name])
        carried = {name: Carried(name, aval.shape, aval.dtype) for name, aval in types.carried.items()}
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
            reads=reads_,
            read_phases=self.read_phases,
            outputs=tuple(
                Output(name, *outputs[name], output_tiles[name], types.stores[name], stored_by[name])
                for name in self.nest.outputs
            ),
            carried=carried,
            assigned_types=types.assigned,
            constants=self._constants(),
            scopes=scopes,
            tile_ops=types.ops,
            env=self.env,
        )

    # ------------------------------------------------------------------ the prologue and outputs

    def _prologue(self) -> dict[str, tuple[tuple[int, ...], np.dtype, Any]]:
        """Run the prologue on the inputs' shapes; each output's (shape, dtype, fill)."""
        allocation_of = {id(stmt): name for name, stmt in self.nest.allocations.items()}
        outputs = {}
        for stmt in self.nest.prologue:
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
        usage = f"allocate output `{name}` with jnp.zeros(shape, dtype) or jnp.full(shape, value, dtype)"
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

    # ------------------------------------------------------------------ loops

    def _range(self, loop: Loop) -> Range:
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

    # ------------------------------------------------------------------ tiles

    def _window(self, index: ast.expr, dim: int, text: str) -> Window:
        """How the index expression `index` selects from a dim of size `dim`."""
        usage = (
            f"{text}: each index is `v:v + size` or `v` for a loop variable v, `:`, or constants (`start:stop`, an int)"
        )
        loop_vars = reads(index) & self.ranges.keys()
        if isinstance(index, ast.Subscript) and isinstance(index.value, ast.Name) and index.value.id in self.indices:
            return Window(None, 0, None, index, frozenset(loop_vars))  # bounds: clamped when the kernel runs
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
            if loop_vars:
                raise self.error(usage, index)
            start = 0 if lower is None else self._int(lower, text)
            stop = dim if upper is None else self._int(upper, text)
            return self._checked(Window(None, start, stop - start), dim, text, index)
        if isinstance(index, ast.Name) and index.id in self.ranges:
            return self._checked(Window(index.id, 0, None), dim, text, index)
        computed = reads(index) & assigned(self.nest.tile)
        if loop_vars or computed:
            more = f": `{sorted(computed)[0]}` is computed by the tile code" if computed else ""
            raise self.error(f"{usage}, or an element of an index array written in place (`w[ids[e], :]`){more}", index)
        return self._checked(Window(None, self._int(index, text), None), dim, text, index)

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
        return tuple(operands), by_key

    def _index_arrays(self) -> list[Index]:
        """The inputs read element by element (`lengths[b]`), in parameter order: int32, read only that way,
        indexed by loop variables and constants."""
        nest, params = self.nest, set(self.nest.params)
        trees: list[ast.AST] = [*nest.tile, *(loop.mask for loop in nest.reductions if loop.mask is not None)]
        elements = [n for tree in trees for n in element_reads(tree, params)]
        found = []
        for p, name in enumerate(nest.params):
            mine = [n for n in elements if n.value.id == name]  # type: ignore[attr-defined]
            if not mine:
                continue
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
                if any(isinstance(n, ast.Subscript) for i in index for n in ast.walk(i)):
                    raise self.error(
                        f"{text}: an index array's elements are indexed by loop variables and constants", node
                    )
            found.append(Index(name, p, tuple(spec.shape), np.dtype(spec.dtype)))
        return found

    def _check_element_bounds(self) -> None:
        """Elements of index arrays read at loop values outside the array are errors, found when planning by
        evaluating each index at every value of the loops it reads (index arrays are in SMEM, unchecked)."""
        nest = self.nest
        trees: list[ast.AST] = [*nest.tile, *(loop.mask for loop in nest.reductions if loop.mask is not None)]
        for node in (n for tree in trees for n in element_reads(tree, self.indices)):
            index = self.indices[node.value.id]  # type: ignore[attr-defined]
            dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for expr, dim in zip(dims, index.shape, strict=True):
                loops = [self.ranges[v] for v in sorted(reads(expr) & self.ranges.keys())]
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
        uses = [w.loop for w in tile.windows if w.loop]
        r = self.nest.stored_in[name]
        placing = [*parallel, *([self.ranges[self.nest.reductions[r].var]] if r is not None else [])]
        for r in placing:
            if uses.count(r.var) != 1:
                what = "every parallel loop places" if r in parallel else f"`{r.var}`'s steps each store"
                raise self.error(
                    f"{tile.text}: {what} each output tile, so `{r.var}` indexes exactly one dim of the output",
                    store.node,
                )
        for window, dim in zip(tile.windows, shape, strict=True):
            r = self.ranges.get(window.loop) if window.loop else None
            if r is None:
                if window.offset != 0 or window.size != dim:
                    raise self.error(f"{tile.text}: dims no loop places are stored whole, with `:`", store.node)
            elif r.start != 0 or r.count * r.step != dim or (window.size or 1) != r.step:
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
        names = set().union(*(reads(stmt) for stmt in flatten(nest.tile)), *(reads(m) for m in masks)) - own
        return {name: self.env[name] for name in sorted(names) if name in self.env}

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
                next((t for t in (self._last(s, name) for s in reversed(segments)) if t is not None), None),
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
