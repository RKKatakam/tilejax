"""Generate a kernel's Pallas source: one gridless kernel whose copies are pipelined by hand.

The kernel is a single `pl.loop` over its steps (schedule.py). Each step

1. decodes its tile and reduction block (arithmetic, or SMEM tables when masks skip work);
2. starts fetching the inputs of the step `input_buffers - 1` ahead, into a ring of VMEM slots;
3. waits for its own inputs; at a tile's first step it first starts fetching the next tile's per-tile
   inputs, then waits for this tile's (a per-tile window the next tile shares isn't fetched again);
4. at a tile's last step, waits until the output slot it is about to write is free;
5. runs the tile code: the code before the reduction loop at the tile's first step, the reduction body,
   the code after it at the last step, whose stores fill output slots and start their copies to HBM.

Values carried between steps live in VMEM scratch. Tiles a mask skips are written with the output's fill
value by DMAs from a constant VMEM tile, a few at every tile, so they overlap the compute (an output that
is an input updated in place keeps the input's values there: its buffer is the input's,
`input_output_aliases`). An empty step (a
reduction loop whose mask skips all of a tile's blocks) fetches nothing and runs no body.

With a pipeline of bursts (single-step tiles only, pipeline.py), steps 2 and 4 move to the bursts' ends: a
burst's first step waits for the previous burst's output copies, then starts fetching the next burst's
inputs; its last step waits for those, then starts copying its own outputs out. (The first burst's steps
wait for their own inputs, and the last burst's copy their outputs out as they go: nothing reads or
writes alongside them.)

With runtime ranges (a reduction loop's bounds read index arrays) there are no step tables: two cursors in
SMEM hold this step's (tile, phase, block) and the step whose inputs are fetched next, each moved on a step
at a time (`advance`, from the tile's bounds), and the step loop runs until the cursor has passed the last
tile.

Index arrays (inputs read element by element) are SMEM inputs. When masks read them, the entry function
first finds the steps that run on the device (`<name>_steps`, plain JAX), and the kernel loops over those.

An input gathered across the chips of a mesh axis (plan.Ring) is an extra HBM output the kernel fills: at
the start it copies this chip's rows in and sends them to the next chip on the ring; the tiles walk the
rows a chip's shard at a time, this chip's first; and before a copy reads a shard, the kernel waits for it
to arrive (one DMA semaphore per ring step: a semaphore only counts bytes) and forwards it to the next chip.

The source is plain Pallas, standalone: it defines the constants the tile code reads, the step tables and
the helpers it calls. With `trace=True`, steps 3, 4 and 5 are `jax.named_scope` regions, which libtpu
records in a profile (profiling.py).
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import inspect
import itertools
import keyword
import math
import textwrap
import types
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import mxu, v5e
from .errors import KernelError
from .parse import Nest
from .pipeline import TILE_SLOTS, Pipeline, rings
from .plan import Kind, Operand, Output, Plan, Range, Tile
from .schedule import CODE_SHIFT, FILL_TABLE, STEP_CODE, STEP_COUNT, STEP_TILE, TILE_TABLE, Schedule
from .tilecode import as_store, assigned, element_reads, flatten, reads
from .tunable import choices, is_tunable

__all__ = ["COMPUTE", "FILLS_IN_FLIGHT", "OUT_WAIT", "REGIONS", "WAIT", "generate"]

# Trace regions, named `<kernel>/<region>`.
WAIT, OUT_WAIT, COMPUTE = "wait", "out_wait", "compute"
REGIONS = (WAIT, OUT_WAIT, COMPUTE)
# Fill copies (masked-out tiles) in flight at once: each tile starts its share, waiting for older ones.
FILLS_IN_FLIGHT = 4
# What the generated code imports, by the names it uses.
MODULES = {"jax": jax, "jnp": jnp, "np": np, "pl": pl, "pltpu": pltpu}
_WEAK = "_weak"
_PHASE_STEPS = "_phase_steps"
_PAIRS = "_pairs"
_UNPAIRS = "_store_pairs"
_COPIES = "_Copies"
_STRIDED = "_strided"


def generate(plan: Plan, schedule: Schedule, pipeline: Pipeline, *, trace: bool = False) -> tuple[str, str]:
    """The Pallas source of `plan`'s kernel, and the name of the function it defines to run it:
    `<name>(*inputs, interpret=False)` (renamed if the kernel's code uses that name itself)."""
    generator = _Generator(plan, schedule, pipeline, trace)
    return generator.source(), generator.entry_name


class _Namer:
    """Fresh names that don't collide with any taken ones."""

    def __init__(self, taken: Iterable[str]) -> None:
        self.taken = set(taken)

    def __call__(self, base: str) -> str:
        name, n = base, 2
        while name in self.taken or keyword.iskeyword(name) or hasattr(builtins, name):
            name, n = f"{base}{n}", n + 1
        self.taken.add(name)
        return name


class _Writer:
    """Lines of source, indented by the blocks they are in."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    def __call__(self, *lines: str) -> None:
        for line in lines:
            self.lines.append("    " * self.depth + line if line else "")

    @contextmanager
    def block(self, header: str) -> Iterator[None]:
        previous = self.lines[-1].strip() if self.lines else ""
        if header.startswith("def ") and previous and not previous.startswith("@") and not previous.endswith(":"):
            self("")  # a blank line between a kernel's helper functions
        self(header)
        self.depth += 1
        yield
        self.depth -= 1

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _int_expr(x: str, op: str, n: int) -> str:
    """`x // n` or `x % n` for a non-negative traced int: a shift or mask when n is a power of two, else one
    integer division without the sign fix-ups of Python's floor division."""
    if n == 1:
        return x if op == "//" else "0"
    if n & (n - 1) == 0:
        x = f"({x})" if " " in x else x
        return f"({x} >> {n.bit_length() - 1})" if op == "//" else f"({x} & {n - 1})"
    return f"jax.lax.div({x}, {n})" if op == "//" else f"jax.lax.rem({x}, {n})"


def _tuple(names: list[str]) -> str:
    """The elements of a tuple of names, in source: `a, b`, or `a,` for one."""
    return ", ".join(names) + ("," if len(names) == 1 else "")


def _arange(r: Range) -> str:
    """The loop's values as a NumPy array, in source."""
    if r.start == 0 and r.step == 1:
        return f"np.arange({r.count})"
    return f"np.arange({r.start}, {r.start + r.count * r.step}, {r.step})"


def _scaled(block: str, r: Range) -> str:
    """The loop's value at a block index: `start + block * step` (a constant for a loop of one block: Mosaic
    copies constant windows faster than traced ones)."""
    if r.count == 1:
        return str(r.start)
    term = block if r.step == 1 else f"{block} * {r.step}"
    return f"{term} + {r.start}" if r.start else term


class _Generator:
    def __init__(self, plan: Plan, schedule: Schedule, pipeline: Pipeline, trace: bool) -> None:
        self.plan, self.schedule, self.pipeline, self.trace = plan, schedule, pipeline, trace
        self.nest = nest = plan.nest
        self.single = plan.single_step
        self.burst = pipeline.burst
        if self.burst and not (self.single and not schedule.runtime_tiles):
            raise KernelError(
                f"{nest.name}: copies move in bursts only in kernels whose tiles take one step, counted when built"
            )
        self.value_reads = _value_reads(plan)
        self.uses_weak = any(r.var in self.value_reads and r.count > 1 for r in plan.loops)
        bound = assigned(nest.tile) | {r.var for r in plan.loops}
        for name in sorted(bound & {*MODULES, *mxu.FUNCTIONS, _WEAK}):
            node = next(n for n in ast.walk(nest.parallel[0].node) if isinstance(n, ast.Name) and n.id == name)
            raise nest.error(f"the generated kernel uses the name `{name}` itself; rename your `{name}`", node)
        tile_names = {n for stmt in flatten(nest.tile) for n in reads(stmt)} | bound | set(plan.constants)

        self.indices = [index.name for index in plan.indices]
        self.arrays = [p for p in nest.inputs if p not in self.indices]  # the inputs in HBM
        self.definitions = _Definitions(plan.constants, nest)
        # (The steps function reads the module's names with the loop variables and index arrays as locals.)
        self.module = _Namer(
            {
                *self.definitions.defined,
                *MODULES,
                *mxu.FUNCTIONS,
                _WEAK,
                _PHASE_STEPS,
                _PAIRS,
                _UNPAIRS,
                _COPIES,
                _STRIDED,
                "functools",
                *self.indices,
            }
            | {r.var for r in plan.loops}
        )
        self.kernel_name = self.module(f"{nest.name}_kernel")
        self.entry_name = self.module(nest.name)
        self.tables = {name: self.module(name.upper()) for name in schedule.tables}
        self.runtime = [STEP_TILE, STEP_CODE, STEP_COUNT] if schedule.runtime else []  # made on the device
        if schedule.runtime:
            self.steps_name = self.module(f"{nest.name}_steps")
            self.tile_values = {var: self.module(f"{var.upper()}_OF_TILE") for var in schedule.tile_values}
            self.static_runs = {p: self.module(f"PHASE{p}_RUNS") for p in schedule.static_runs}
        # The entry function's parameters mustn't hide the module's names it uses.
        # (Index arrays keep their names: the module defines no name of theirs.)
        entry = self.entry_namer = _Namer({*self.module.taken, "interpret"} - set(self.indices))
        self.entry_params = [entry(p) for p in nest.inputs]

        n = self.local = _Namer(tile_names | self.module.taken)
        self.table_params = {name: n(name) for name in [*schedule.tables, *self.runtime]}
        self.smem = {name: n(f"{name}_smem") for name in self.indices}
        self.hbm = {name: n(f"{name}_hbm") for name in (*self.arrays, *nest.outputs, *(r.name for r in plan.rings))}
        # Inputs gathered across chips (plan.Ring): this chip's position on the ring, the semaphores each ring
        # step's rows are received and sent by, the copy of this chip's own rows, and the last step arrived.
        self.rings = {r.name: r for r in plan.rings}
        self.n_me, self.n_barrier = n("me"), n("barrier")
        self.ring_recv = {r.name: n(f"{r.name}_recv_sems") for r in plan.rings}
        self.ring_send = {r.name: n(f"{r.name}_send_sems") for r in plan.rings}
        self.ring_own = {r.name: n(f"{r.name}_own_sem") for r in plan.rings}
        self.ring_arrived = {r.name: n(f"{r.name}_arrived") for r in plan.rings}
        self.ring_fns = {
            r.name: {fn: n(f"{r.name}_{fn}") for fn in ("rows", "send_own", "send", "receive", "arrive")}
            for r in plan.rings
        }
        self.bufs = {
            op.name: n(f"{op.name}_buf" if op.kind is Kind.ONCE else f"{op.name}_bufs") for op in plan.operands
        }
        self.sems = {
            op.name: n(f"{op.name}_sem" if op.kind is Kind.ONCE else f"{op.name}_sems") for op in plan.operands
        }
        # Streamed windows of different phases with one tile shape share a ring (pipeline.py's `rings`).
        self.owner = rings(plan)
        for op in plan.operands:
            self.bufs[op.name], self.sems[op.name] = self.bufs[self.owner[op.name]], self.sems[self.owner[op.name]]
        self.bufs |= {out.name: n(f"{out.name}_bufs") for out in plan.outputs}
        self.sems |= {out.name: n(f"{out.name}_sems") for out in plan.outputs}
        self.refs = {c: n(f"{c}_ref") for c in plan.carried}
        # The outputs whose tiles a mask skips are filled (an input updated in place keeps its values there).
        self.filled = [out for out in plan.outputs if out.fill is not None] if schedule.fills else []
        self.fill_bufs = {out.name: n(f"{out.name}_fill") for out in self.filled}
        self.fill_sems = {out.name: n(f"{out.name}_fill_sem") for out in self.filled}
        self.fill_copies = {out.name: n(f"{out.name}_fill_copy") for out in self.filled}
        self.copy_fns = {name: n(f"{name}_copy") for name in [op.name for op in plan.operands] + list(nest.outputs)}
        self.views = {op.name: n(f"{op.name}_ref") for op in plan.operands}
        # Operands read as words (`autokernel.pairs`): their VMEM slots seen as uint32, (rows, d).
        self.words = {name: n(f"{name}_words") for name in sorted(set(plan.packed.values()))}
        self.packing = bool(self.words or plan.unpacked)  # (the kernel takes `interpret`: the words differ)
        self.n_interpret = n("interpret")  # (the interpreter can't see a VMEM buffer as another dtype)
        # Per-tile windows consecutive tiles may share: copied only when they move; SMEM holds each one's slot.
        self.shared = [op for op in plan.operands if op.kind is Kind.TILE and _may_stay(op, plan)]
        self.tile_slots = n("tile_slots")
        self.moved_fns = {op.name: n(f"{op.name}_moves") for op in self.shared}
        self.prev_blk = {r.var: n(f"{r.var}_prev") for r in plan.parallel}
        self.n_prev = n("prev")
        self.blk = {r.var: n(f"{r.var}_blk") for r in plan.loops}
        # The kernel's own locals and helper functions.
        self.n_step, self.n_blocks, self.n_fill_blocks = n("step"), n("blocks"), n("fill_blocks")
        self.n_s, self.n_tile, self.n_kb, self.n_first, self.n_last = n("s"), n("tile"), n("kb"), n("first"), n("last")
        self.n_ph, self.n_empty, self.n_steps, self.n_rb = n("ph"), n("empty"), n("n_steps"), n("rb")
        self.repeated = bool(plan.repeats)  # steps decode to a repeat loop's iteration (`rb`) too
        self.cursor = schedule.cursor
        self.n_cursor, self.n_counts, self.n_advance = n("cursor"), n("counts"), n("advance")
        # Computed index arrays (plan.Computed): a VMEM buffer the tile code's value is staged in, and its SMEM
        # copy, which windows and bounds read; the counters (the steps whose inputs started, and how many times
        # each computed array has been made) in SMEM.
        # Masked reduction loops in a kernel that finds its steps as it runs: each cursor also holds the next
        # block its mask runs and two flags (its phase's first step; an empty step), the blocks between
        # skipped by a scalar loop as it moves (`seek`).
        self.cursor_masked = [p for p, phase in enumerate(plan.phases) if phase.range.mask is not None]
        self.cursor_fields = (4 if self.repeated else 3) + (2 if self.cursor_masked else 0)
        self.n_seek, self.n_enter, self.n_skips = n("seek"), n("enter"), n("skips")
        self.stage = {c.name: n(f"{c.name}_stage") for c in plan.computed}
        self.computed_smem = {c.name: n(f"{c.name}_smem") for c in plan.computed}
        self.n_stage_sem, self.n_counters = n("stage_sem"), n("counters")
        # How many tiles run: a number, or (a runtime count of tiles) counted when the kernel starts.
        self.n_tiles = n("n_tiles") if schedule.runtime_tiles else str(schedule.tiles)
        self.n_at, self.n_rep, self.n_code, self.n_c, self.n_slot = n("at"), n("rep"), n("code"), n("c"), n("slot")

    # ------------------------------------------------------------------ the module

    def source(self) -> str:
        kernel = self._kernel()
        entry = self._entry()
        w = _Writer()
        inputs = ", ".join(
            f"{name}: {np.dtype(spec.dtype).name}{list(spec.shape)}"
            for name, spec in zip(self.nest.params, self.plan.inputs, strict=True)
        )
        w(f"# Generated by autokernel from `{self.nest.name}` ({self.nest.filename}:{self.nest.function.lineno})")
        tunable = f", {self.plan.tunable}" if self.plan.tunable is not None else ""
        w(f"# for {inputs}{tunable}; {self.pipeline}{'; with trace regions' if self.trace else ''}.")
        for ring in self.plan.rings:
            chips = f"the {ring.size} chips of mesh axis {ring.axis!r}"
            w(f"# `{ring.name}` is `{ring.source}` gathered over {chips}, passed around a ring of them.")
        w("")
        w(
            *(["import functools"] if self.packing else []),
            "import jax",
            "import jax.numpy as jnp",
            "import numpy as np",
        )
        w("from jax.experimental import pallas as pl", "from jax.experimental.pallas import tpu as pltpu")
        definitions = self.definitions
        if definitions.imports:
            w(*definitions.imports)
        if definitions.constants:
            w("", "# The values the tile code reads, as they were when the kernel was built.")
            w(*definitions.constants)
        if self.tables:
            w("", "# Step tables (SMEM): the tiles and steps that run, as the masks say.")
            for name, table in self.schedule.tables.items():
                w(f"{self.tables[name]} = np.array({table.tolist()}, np.int32)")
        if self.runtime:
            self._steps_definitions(w)
        for fn_source in definitions.functions:
            w("", "", *fn_source.splitlines())
        if self._uses_mxu(kernel):
            for fn in mxu.FUNCTIONS.values():
                w("", "", *inspect.getsource(fn).splitlines())
        if any(_gathered(op.tile) for op in self.plan.operands):
            w("", "")
            with w.block(f"class {_COPIES}:"):
                w('"""Copies started and waited together: a window gathered from several places of an array."""')
                w("")
                with w.block("def __init__(self, *copies):"):
                    w("self.copies = copies")
                w("")
                with w.block("def start(self):"), w.block("for copy in self.copies:"):
                    w("copy.start()")
                w("")
                with w.block("def wait(self):"), w.block("for copy in self.copies:"):
                    w("copy.wait()")
        if self.words:
            w("", "")
            with w.block(f"def {_PAIRS}(x):"):
                w("# autokernel.pairs: a 16-bit tile's pairs of rows (..., 2, d) as uint32 words (..., d), for the")
                w("# interpreter (on a TPU the kernel reads its VMEM buffer as words).")
                w("return jax.lax.bitcast_convert_type(jnp.swapaxes(x, -1, -2), jnp.uint32)")
            w("", "")
            with w.block(f"def {_STRIDED}(words, start, n, stride, interpret):"):
                w("# Rows start, start + stride, ...: n of them. On a TPU, a strided load from the VMEM buffer (the")
                w("# interpreter has the words as a value).")
                with w.block("if interpret:"):
                    w("return words.reshape(n, stride, words.shape[-1])[:, start, :]")
                w("return words[pl.ds(start, n, stride=stride)]")
        if self.plan.unpacked:
            w("", "")
            with w.block(f"def {_UNPAIRS}(bufs, slot, words, interpret):"):
                w("# autokernel.unpairs: uint32 words (..., d) stored into slot `slot` of a 16-bit buffer of pairs of")
                w("# rows (..., 2, d). On a TPU, into the buffer seen as words (the interpreter can't bitcast it).")
                with w.block("if interpret:"):
                    w("bufs[slot] = jnp.swapaxes(jax.lax.bitcast_convert_type(words, bufs.dtype), -1, -2)")
                    w("return")
                w("rows, d = int(np.prod(bufs.shape[1:-2])), bufs.shape[-1]")
                w("bufs.bitcast(jnp.uint32).at[slot].reshape(rows, d)[...] = words.reshape(rows, d)")
        if self.uses_weak:
            w("", "")
            with w.block(f"def {_WEAK}(x):"):
                w("# A loop variable is a Python int in the loop nest: give it an int's weak type, so `bf16_tile * i`")
                w("# stays bf16. (JAX has no public API for weak types; this is the one internal the kernel uses.)")
                w("return jax._src.lax.lax._convert_element_type(x, weak_type=True)")
        w("", "")
        return w.text() + kernel + "\n\n" + entry

    def _steps_definitions(self, w: _Writer) -> None:
        """The function that finds the steps the kernel runs when masks read index arrays, before it runs."""
        plan, sch = self.plan, self.schedule
        w("", "# For each tile the step tables cover: its loops' values, and the blocks that run of the loops")
        w("# whose masks read no index array.")
        for var, values in sch.tile_values.items():
            w(f"{self.tile_values[var]} = np.array({values.tolist()}, np.int32)")
        for p, runs in sch.static_runs.items():
            w(f"{self.static_runs[p]} = np.array({runs.tolist()})")
        w("", "")
        with w.block(f"def {_PHASE_STEPS}(run, phase, phases, iteration, iterations):"):
            w('"""A reduction loop\'s steps for each tile (and iteration of its repeat loop): the blocks that run, in')
            w("order, or else one empty step (at its first block); and each step's code,")
            w('((block * phases + phase) * iterations + iteration) << shift | empty << 2 | first << 1 | last."""')
            w("blocks = np.arange(run.shape[-1])")
            w("empty = jnp.logical_not(run.any(axis=-1, keepdims=True)) & (blocks == 0)")
            w("keep = run | empty")
            w("first = blocks == jnp.argmax(keep, axis=-1, keepdims=True)")
            w("last = blocks == run.shape[-1] - 1 - jnp.argmax(keep[..., ::-1], axis=-1, keepdims=True)")
            w(f"code = ((blocks * phases + phase) * iterations + iteration) << {CODE_SHIFT}")
            w("code = code | empty.astype(int) << 2 | first.astype(int) << 1 | last")
            w("return keep, code.astype(jnp.int32)")
        w("", "")
        n = _Namer({*self.module.taken, *plan.constants, *self.indices} | {r.var for r in plan.loops})
        with w.block(f"def {self.steps_name}({', '.join(self.indices)}):"):
            w('"""The steps the kernel runs, from its masks evaluated on the index arrays: each tile\'s steps of')
            w("each reduction loop (and repeat loop iteration), in order, packed into tables (the steps' tiles and")
            w('codes), and how many there are."""')
            tiles = sch.pattern_tiles
            for var in sch.tile_values:
                w(f"{var} = {self.tile_values[var]}[:, None]")
            parts = []  # each a (tiles, steps) array pair: a phase's, or a repeat loop's, steps of each tile
            p = 0
            while p < len(plan.phases):
                r = plan.phases[p].repeat
                if r is None:
                    parts.append(self._phase_runs(w, n, p, ()))
                    p += 1
                    continue
                rep = plan.repeats[r]
                count = rep.range.count
                w(f"{rep.range.var} = {_arange(rep.range)}[:, None, None]")
                runs = [self._phase_runs(w, n, q, (count,)) for q in range(rep.first, rep.last)]
                keep, code = n(f"keep_r{r}"), n(f"code_r{r}")
                for name, i in ((keep, 0), (code, 1)):  # (iterations, tiles, steps) -> (tiles, iterations * steps)
                    joined = f"jnp.concatenate([{', '.join(run[i] for run in runs)}], axis=-1)"
                    w(f"{name} = {joined}.swapaxes(0, 1).reshape({tiles}, -1)")
                parts.append((keep, code))
                p = rep.last
            keep, code, order, tile = n("keep"), n("code"), n("order"), n("tile")
            w(f"{keep} = jnp.concatenate([{', '.join(k for k, _ in parts)}], axis=1).reshape(-1)")
            w(f"{code} = jnp.concatenate([{', '.join(c for _, c in parts)}], axis=1).reshape(-1)")
            w(f"{tile} = jnp.repeat(jnp.arange({tiles}, dtype=jnp.int32), {sch.pattern_steps // tiles})")
            # (A stable sort brings the steps that run to the front, in order: a TPU scatters far slower.)
            w(f"{order} = jnp.argsort(jnp.logical_not({keep}), stable=True)")
            w(f"return {tile}[{order}], {code}[{order}], jnp.sum({keep}, dtype=jnp.int32).reshape(1)")

    def _phase_runs(self, w: _Writer, n: _Namer, p: int, iterations: tuple[int, ...]) -> tuple[str, str]:
        """Emit which of phase p's blocks run for each tile (each iteration of its repeat loop first, if
        `iterations`), and its steps; their names."""
        plan, sch = self.plan, self.schedule
        r = plan.phases[p].range
        shape = (*iterations, sch.pattern_tiles, r.count)
        run, keep, code = n(f"run{p}"), n(f"keep{p}"), n(f"code{p}")
        if r.runtime:
            w(f"{r.var} = {_arange(r)}[None, :]")
            mask = ast.unparse(_Vectorize().visit(copy.deepcopy(r.mask)))
            w(f"{run} = jnp.broadcast_to(jnp.logical_not({mask}), {shape})")
        elif p in sch.static_runs:
            w(f"{run} = {self.static_runs[p]}")
        else:
            w(f"{run} = np.ones({shape}, bool)")
        rep = plan.repeats[plan.phases[p].repeat].range if iterations else None  # type: ignore[index]
        iteration = f"np.arange({rep.count})[:, None, None]" if rep else "0"
        w(f"{keep}, {code} = {_PHASE_STEPS}({run}, {p}, {sch.phase_radix}, {iteration}, {sch.iterations})")
        return keep, code

    def _uses_mxu(self, kernel: str) -> bool:
        return mxu.DOT in kernel or mxu.DOT_NT in kernel

    def _entry(self) -> str:
        plan, schedule = self.plan, self.schedule
        w = _Writer()
        params = ", ".join(self.entry_params)
        with w.block(f"def {self.entry_name}({params}, *, interpret=False):"):
            tiles = f"{schedule.tiles} of {schedule.total_tiles}" if schedule.fills else str(schedule.tiles)
            steps = f"at most {schedule.steps}" if schedule.runtime else str(schedule.steps)
            w(f'"""`{self.nest.name}` as one gridless kernel: a loop over {steps} steps of {tiles} tiles."""')
            param = dict(zip(self.nest.inputs, self.entry_params, strict=True))
            runtime = [self.entry_namer(name) for name in self.runtime]
            if runtime:
                w(f"{', '.join(runtime)} = {self.steps_name}({', '.join(param[i] for i in self.indices)})")
            with w.block("return pl.pallas_call("):
                w(
                    f"functools.partial({self.kernel_name}, {self.n_interpret}=bool(interpret)),"
                    if self.packing
                    else f"{self.kernel_name},"
                )
                shapes = [f"jax.ShapeDtypeStruct({out.shape}, jnp.{out.dtype.name})" for out in plan.outputs]
                for ring in plan.rings:  # (its HBM buffer: an output the caller doesn't see)
                    spec = plan.inputs[self.nest.params.index(ring.name)]
                    shapes.append(f"jax.ShapeDtypeStruct({tuple(spec.shape)}, jnp.{np.dtype(spec.dtype).name})")
                w(f"out_shape={shapes[0] if len(shapes) == 1 else '(' + ', '.join(shapes) + ',)'},")
                with w.block("in_specs=["):
                    smem = len(self.tables) + len(runtime) + len(self.indices)
                    w(*["pl.BlockSpec(memory_space=pltpu.SMEM),"] * smem)
                    w(*["pl.BlockSpec(memory_space=pl.ANY),"] * len(self.arrays))
                w("],")
                specs = ["pl.BlockSpec(memory_space=pl.ANY)"] * (len(plan.outputs) + len(plan.rings))
                w(f"out_specs={specs[0] if len(specs) == 1 else '(' + ', '.join(specs) + ',)'},")
                with w.block("scratch_shapes=["):
                    w(*self._scratch_shapes())
                w("],")
                aliases = self._aliases()
                if aliases:
                    w(f"input_output_aliases={aliases},  # outputs updated in place")
                # (A ring's chips meet at a barrier before they write into each other's buffers.)
                collective = ", collective_id=0" if plan.rings else ""
                w(f"compiler_params=pltpu.CompilerParams(vmem_limit_bytes={v5e.VMEM_LIMIT}{collective}),")
                w("interpret=interpret,")
                w(f"name={self.nest.name!r},")
            inputs = [*self.tables.values(), *runtime, *(param[i] for i in self.indices)]
            inputs += [param[a] for a in self.arrays]
            n = len(plan.outputs)
            mine = ("[0]" if n == 1 else f"[:{n}]") if plan.rings else ""  # (not the gathered arrays)
            w(f")({', '.join(inputs)}){mine}")
        return w.text()

    def _aliases(self) -> dict[int, int]:
        """The pallas_call's input_output_aliases: each output updated in place by its input's position."""
        before = len(self.tables) + len(self.runtime) + len(self.indices)  # (the SMEM inputs come first)
        return {
            before + self.arrays.index(self.nest.params[out.aliases]): k
            for k, out in enumerate(self.plan.outputs)
            if out.aliases is not None
        }

    def _scratch_shapes(self) -> list[str]:
        lines = []
        for op in self.plan.operands:
            if self.owner[op.name] != op.name:
                continue
            slots = self.pipeline.slots(op.kind)
            shape = op.tile.shape if op.kind is Kind.ONCE else (slots, *op.tile.shape)
            sems = "()" if op.kind is Kind.ONCE else f"({slots},)"
            texts = [o.tile.text for o in self.plan.operands if self.owner[o.name] == op.name]
            lines += [
                f"pltpu.VMEM({shape}, jnp.{op.tile.dtype.name}),  # {self.bufs[op.name]}: {', '.join(texts)}",
                f"pltpu.SemaphoreType.DMA({sems}),",
            ]
        slots = self.pipeline.output_buffers
        for out in self.plan.outputs:
            note = f"{self.bufs[out.name]}: {out.tile.text}"
            lines += [
                f"pltpu.VMEM({(slots, *out.tile.shape)}, jnp.{out.dtype.name}),  # {note}",
                f"pltpu.SemaphoreType.DMA(({slots},)),",
            ]
        for c in self.plan.carried.values():
            lines.append(f"pltpu.VMEM({c.shape}, jnp.{c.dtype.name}),  # {self.refs[c.name]}: `{c.name}`")
        for out in self.plan.outputs:
            if out.name in self.fill_bufs:
                lines += [
                    f"pltpu.VMEM({out.tile.shape}, jnp.{out.dtype.name}),  # {self.fill_bufs[out.name]}",
                    "pltpu.SemaphoreType.DMA(()),",
                ]
        if self.shared:
            names = ", ".join(op.name for op in self.shared)
            lines.append(f"pltpu.SMEM(({len(self.shared)},), jnp.int32),  # {self.tile_slots}: the slots of {names}")
        if self.cursor:
            names = ["tile", "phase", "block", *(["iteration"] if self.repeated else [])]
            names += ["next block", "flags"] if self.cursor_masked else []
            lines.append(
                f"pltpu.SMEM((2, {self.cursor_fields}), jnp.int32),  # {self.n_cursor}: ({', '.join(names)}) now "
                "and fetched next"
            )
            made = "".join(f", times `{c.name}` was made" for c in self.plan.computed)
            lines.append(
                f"pltpu.SMEM(({1 + len(self.plan.computed)},), jnp.int32),  # {self.n_counters}: steps fetched{made}"
            )
        for c in self.plan.computed:
            shape = _staged(c.shape)
            lines += [
                f"pltpu.VMEM({shape}, jnp.int32),  # {self.stage[c.name]}: `{c.name}`, staged",
                f"pltpu.SMEM({shape}, jnp.int32),  # {self.computed_smem[c.name]}: `{c.name}`, its elements",
            ]
        if self.plan.computed:
            lines.append("pltpu.SemaphoreType.DMA(()),")
        for ring in self.plan.rings:
            lines += [
                f"pltpu.SemaphoreType.DMA(({ring.size},)),  # {self.ring_recv[ring.name]}: ring step k's rows received",
                f"pltpu.SemaphoreType.DMA(({ring.size},)),  # {self.ring_send[ring.name]}: ring step k's rows sent on",
                f"pltpu.SemaphoreType.DMA(()),  # {self.ring_own[ring.name]}: this chip's rows into `{ring.name}`",
                f"pltpu.SMEM((1,), jnp.int32),  # {self.ring_arrived[ring.name]}: the last ring step arrived",
            ]
        return lines

    # ------------------------------------------------------------------ the kernel

    def _kernel(self) -> str:
        plan, nest = self.plan, self.nest
        w = _Writer()
        params = [
            *self.table_params.values(),
            *self.smem.values(),
            *(self.hbm[p] for p in self.arrays),
            *(self.hbm[o] for o in nest.outputs),
            *(self.hbm[r.name] for r in plan.rings),
        ]
        for op in plan.operands:
            if self.owner[op.name] == op.name:
                params += [self.bufs[op.name], self.sems[op.name]]
        for out in plan.outputs:
            params += [self.bufs[out.name], self.sems[out.name]]
        params += list(self.refs.values())
        for out in self.fill_bufs:
            params += [self.fill_bufs[out], self.fill_sems[out]]
        params += [self.tile_slots] if self.shared else []
        params += [self.n_cursor, self.n_counters] if self.cursor else []
        for c in plan.computed:
            params += [self.stage[c.name], self.computed_smem[c.name]]
        params += [self.n_stage_sem] if plan.computed else []
        for ring in plan.rings:
            names = (self.ring_recv, self.ring_send, self.ring_own, self.ring_arrived)
            params += [names_[ring.name] for names_ in names]
        if self.packing:
            params.append(f"*, {self.n_interpret}")
        with w.block(f"def {self.kernel_name}({', '.join(params)}):"):
            for ring in plan.rings:
                w(f"{self.n_me} = jax.lax.axis_index({ring.axis!r})  # this chip's position on the ring")
                self._ring_fns(w, ring)
            self._step_fn(w)
            self._blocks_fn(w)
            for op in plan.operands:
                self._operand_copy_fn(w, op)
            for op in self.shared:
                self._moves_fn(w, op)
            for out in plan.outputs:
                self._output_copy_fn(w, out)
            if self.filled:
                self._fill_copy_fns(w)
            self._setup(w)
            self._loop(w)
            self._drain(w)
        return w.text()

    def _step_fn(self, w: _Writer) -> None:
        sch, s, tile, ph, kb = self.schedule, self.n_s, self.n_tile, self.n_ph, self.n_kb
        if self.single:
            return  # a step is a tile
        if self.cursor:
            self._cursor_fns(w)
            return
        phases = len(sch.phase_counts)
        radix = sch.phase_radix
        with w.block(f"def {self.n_step}({s}):"):
            what = "its phase (reduction loop), " + ("its repeat loop's iteration, " if self.repeated else "")
            w(f'"""Step s: its tile, {what}the loop\'s block, and whether that is the phase\'s first and last')
            w("step" + (', and empty."""' if sch.empty else '."""'))
            if not sch.step_tables and phases == 1:
                r = sch.steps_per_tile
                w(f"{tile}, {kb} = {_int_expr(s, '//', r)}, {_int_expr(s, '%', r)}")
                w(f"return {tile}, 0, {kb}, {kb} == 0, {kb} == {r - 1}")
            elif not sch.step_tables:
                within = self.local("within")
                ends = list(itertools.accumulate(sch.phase_counts))
                past = [f"({within} >= {end}).astype(jnp.int32)" for end in ends[:-1]]  # phases already done
                r = sch.steps_per_tile
                w(f"{tile}, {within} = {_int_expr(s, '//', r)}, {_int_expr(s, '%', r)}")
                w(f"{ph} = {' + '.join(past)}")
                offset = " + ".join(f"{p} * {c}" for p, c in zip(past, sch.phase_counts, strict=False))
                w(f"{kb} = {within} - ({offset})")
                last = " | ".join(f"({within} == {end - 1})" for end in ends)
                w(f"return {tile}, {ph}, {kb}, {kb} == 0, {last}")
            else:
                step_tile, step_code = self.table_params[STEP_TILE], self.table_params[STEP_CODE]
                r = s
                if sch.repeats > 1:  # the tables hold one combination of the outer loops' steps
                    if sch.runtime:  # as many steps as the device found
                        per = f"{self.table_params[STEP_COUNT]}[0]"
                        w(f"{self.n_rep} = jax.lax.div({s}, {per})")
                    else:
                        per = str(sch.pattern_steps)
                        w(f"{self.n_rep} = {_int_expr(s, '//', sch.pattern_steps)}")
                    r = self.local("r")
                    w(f"{r} = {s} - {self.n_rep} * {per}")
                    w(f"{tile} = {self.n_rep} * {sch.pattern_tiles} + {step_tile}[{r}]")
                else:
                    w(f"{tile} = {step_tile}[{r}]")
                code = self.n_code
                w(f"{code} = {step_code}[{r}]")
                block = f"({code} >> {CODE_SHIFT})"
                iteration = ""
                if self.repeated:
                    packed = self.local("packed")
                    w(f"{packed} = {block}")
                    iteration = f"{_int_expr(packed, '%', sch.iterations)}, "
                    block = _int_expr(packed, "//", sch.iterations)
                empty = f", ({code} & 4) != 0" if sch.empty else ""
                w(
                    f"return {tile}, {_int_expr(block, '%', radix)}, {iteration}{_int_expr(block, '//', radix)}, "
                    f"({code} & 2) != 0, ({code} & 1) != 0{empty}"
                )

    def _cursor_fns(self, w: _Writer) -> None:
        """`counts(tile[, rb])`, `step(row)` and `advance(row)`: the steps of runtime ranges, found as the kernel
        runs (in iteration rb of a repeat loop, for its phases)."""
        plan, tile, ph, kb, rb, cursor = self.plan, self.n_tile, self.n_ph, self.n_kb, self.n_rb, self.n_cursor
        phases = len(plan.phases)
        block_of = {r.var: self.blk[r.var] for r in plan.parallel}
        args = f"{tile}, {rb}" if self.repeated else tile
        with w.block(f"def {self.n_counts}({args}):"):
            w('"""The tile-th tile\'s blocks of each reduction loop, from its bounds (0: it takes one empty step)."""')
            self._unpack(w, f"{self.n_blocks}({tile})")
            counts = []
            for phase in plan.phases:
                r = phase.range
                if r.bounds is None:
                    counts.append(str(r.count))
                    continue
                values = dict(block_of)
                if phase.repeat is not None:  # (bounds may read the repeat loop's variable)
                    values[plan.repeats[phase.repeat].range.var] = rb
                start, stop = (self._render(b, values) for b in r.bounds)
                counts.append(f"jnp.maximum(jax.lax.div(({stop}) - ({start}) + {r.step - 1}, {r.step}), 0)")
            w(f"return ({_tuple(counts or ['1'])})")  # (a tile without reduction loops: one step)

        if self.cursor_masked:
            self._masked_cursor_fns(w, args)
            return
        n = self.local("n")
        fields = [tile, ph, kb, *([rb] if self.repeated else [])]

        def read(w: _Writer) -> None:
            w(f"{_tuple(fields)} = {', '.join(f'{cursor}[row, {i}]' for i in range(len(fields)))}")
            w(f"{n} = {self.n_counts}({args})[0]")
            for p in range(1, phases):
                w(f"{n} = jnp.where({ph} == {p}, {self.n_counts}({args})[{p}], {n})")

        with w.block(f"def {self.n_step}(row):"):
            w(
                '"""The step cursor `row` is at (0: this step, 1: the step fetched next): its tile, phase, '
                + ("repeat loop iteration, " if self.repeated else "")
                + "block,"
            )
            w('whether it is the phase\'s first and last step, and empty."""')
            read(w)
            w(f"return {self._targets(tile, ph, rb, kb, f'{kb} == 0', f'{kb} >= {n} - 1', f'{n} == 0')}")
        with w.block(f"def {self.n_advance}(row):"):
            w('"""Move cursor `row` to the next step: the next block, else the next phase (the first of its repeat')
            w('loop again, in the next iteration), else the next tile."""')
            read(w)
            w(f"more = {kb} + 1 < {n}")
            w(f"{cursor}[row, 2] = jnp.where(more, {kb} + 1, 0)")
            if phases == 1:
                w(f"{cursor}[row, 0] = jnp.where(more, {tile}, {tile} + 1)")
                return
            ends = {rep.last - 1: rep for rep in plan.repeats}  # a repeat loop's last phase -> the loop
            again = " | ".join(f"(({ph} == {p}) & ({rb} + 1 < {rep.range.count}))" for p, rep in ends.items())
            back = f"{ph} + 1"
            for p, rep in ends.items():
                back = f"jnp.where({ph} == {p}, {rep.first}, {back})"
            later = f"{ph} + 1 < {phases}"
            if again:
                w(f"again = {again}")
                following = f"jnp.where({later}, {ph} + 1, 0)"
                w(f"{cursor}[row, 1] = jnp.where(more, {ph}, jnp.where(again, {back}, {following}))")
                # (Within an iteration the next phase keeps it; a new repeat loop starts at iteration 0.)
                keep = [p for p, phase in enumerate(plan.phases[:-1]) if phase.repeat is not None and p not in ends]
                kept = " | ".join(f"({ph} == {p})" for p in keep) or "False"
                w(f"{cursor}[row, 3] = jnp.where(more, {rb}, jnp.where(again, {rb} + 1, jnp.where({kept}, {rb}, 0)))")
                w(f"{cursor}[row, 0] = jnp.where(more | again | ({later}), {tile}, {tile} + 1)")
            else:
                w(f"later = {later}")
                w(f"{cursor}[row, 1] = jnp.where(more, {ph}, jnp.where(later, {ph} + 1, 0))")
                w(f"{cursor}[row, 0] = jnp.where(more | later, {tile}, {tile} + 1)")

    def _masked_cursor_fns(self, w: _Writer, args: str) -> None:
        """The cursor functions when masks skip blocks: `skips`, `seek`, `enter`, `step` and `advance`."""
        plan, tile, ph, kb, rb, cursor = self.plan, self.n_tile, self.n_ph, self.n_kb, self.n_rb, self.n_cursor
        phases = len(plan.phases)
        nxt, flags = (4 if self.repeated else 3), (5 if self.repeated else 4)
        n, k = self.local("n"), self.local("k")

        def count(w: _Writer, phase: str) -> None:
            w(f"{n} = {self.n_counts}({args})[0]")
            for p in range(1, phases):
                w(f"{n} = jnp.where({phase} == {p}, {self.n_counts}({args})[{p}], {n})")

        with w.block(f"def {self.n_skips}({tile}, {rb}, {ph}, {k}):"):
            w('"""Does phase ph\'s mask skip its block k (of the tile-th tile, in iteration rb)?"""')
            self._unpack(w, f"{self.n_blocks}({tile})")
            skip = "False"
            for p in self.cursor_masked:  # (jnp.where reads every mask: each at a block of its own loop)
                phase = plan.phases[p]
                own = f"jnp.minimum({k}, {phase.range.count - 1})"
                values = {r.var: self.blk[r.var] for r in plan.parallel} | {phase.range.var: own}
                if phase.repeat is not None:
                    values[plan.repeats[phase.repeat].range.var] = rb
                mask = self._render(phase.range.mask, values)  # type: ignore[arg-type]
                skip = f"jnp.where({ph} == {p}, {mask}, {skip})"
            w(f"return {skip}")
        with w.block(f"def {self.n_seek}({tile}, {rb}, {ph}, {k}, {n}):"):
            w('"""The first block from k its mask runs (n if none)."""')
            # (Both sides are evaluated: the mask at a block inside the loop, so its index array reads stay in.)
            inside = f"jnp.clip(b, 0, jnp.maximum({n} - 1, 0))"
            cond = f"lambda b: (b < {n}) & {self.n_skips}({tile}, {rb}, {ph}, {inside})"
            w(f"return jax.lax.while_loop({cond}, lambda b: b + 1, {k})")
        with w.block(f"def {self.n_enter}(row, {tile}, {ph}, {rb}):"):
            w('"""Put cursor `row` at the first step of phase ph of the tile-th tile (iteration rb): its first block')
            w('that runs, or one empty step."""')
            count(w, ph)
            first = self.local("first_block")
            w(f"{first} = {self.n_seek}({tile}, {rb}, {ph}, 0, {n})")
            w(f"{cursor}[row, 0], {cursor}[row, 1] = {tile}, {ph}")
            w(f"{cursor}[row, 2] = jnp.where({first} < {n}, {first}, 0)")
            if self.repeated:
                w(f"{cursor}[row, 3] = {rb}")
            after = f"{self.n_seek}({tile}, {rb}, {ph}, {first} + 1, {n})"
            w(f"{cursor}[row, {nxt}] = jnp.where({first} < {n}, {after}, {n})")
            w(f"{cursor}[row, {flags}] = jnp.where({first} < {n}, 1, 3)")

        fields = [tile, ph, kb, *([rb] if self.repeated else [])]

        def read(w: _Writer) -> None:
            w(f"{_tuple(fields)} = {', '.join(f'{cursor}[row, {i}]' for i in range(len(fields)))}")
            count(w, ph)

        with w.block(f"def {self.n_step}(row):"):
            w(
                '"""The step cursor `row` is at (0: this step, 1: the step fetched next): its tile, phase, '
                + ("repeat loop iteration, " if self.repeated else "")
                + "block,"
            )
            w('whether it is the phase\'s first and last step, and empty."""')
            read(w)
            first, last, empty = (f"({cursor}[row, {flags}] & 1) != 0", f"{cursor}[row, {nxt}] >= {n}",
                                  f"({cursor}[row, {flags}] & 2) != 0")  # fmt: skip
            w(f"return {self._targets(tile, ph, rb, kb, first, last, empty)}")
        with w.block(f"def {self.n_advance}(row):"):
            w('"""Move cursor `row` to the next step: the next block its mask runs, else the next phase (the first of')
            w('its repeat loop again, in the next iteration), else the next tile."""')
            read(w)
            w(f"more = {cursor}[row, {nxt}] < {n}")
            iteration = rb if self.repeated else "0"
            with self._when(w, "more", "_next_block"):
                w(f"{cursor}[row, 2] = {cursor}[row, {nxt}]")
                w(f"{cursor}[row, {nxt}] = {self.n_seek}({tile}, {iteration}, {ph}, {cursor}[row, {nxt}] + 1, {n})")
                w(f"{cursor}[row, {flags}] = 0")
            with self._when(w, "jnp.logical_not(more)", "_next_phase"):
                ends = {rep.last - 1: rep for rep in plan.repeats}
                again = " | ".join(f"(({ph} == {p}) & ({rb} + 1 < {rep.range.count}))" for p, rep in ends.items())
                w(f"again = {again or 'jnp.bool_(False)'}")
                back = f"{ph} + 1"
                for p, rep in ends.items():
                    back = f"jnp.where({ph} == {p}, {rep.first}, {back})"
                later = f"{ph} + 1 < {phases}"
                keep = [p for p, phase in enumerate(plan.phases[:-1]) if phase.repeat is not None and p not in ends]
                kept = " | ".join(f"({ph} == {p})" for p in keep) or "False"
                w(f"to_ph = jnp.where(again, {back}, jnp.where({later}, {ph} + 1, 0))")
                w(f"to_rb = jnp.where(again, {iteration} + 1, jnp.where({kept}, {iteration}, 0))")
                w(f"to_tile = jnp.where(again | ({later}), {tile}, {tile} + 1)")
                with self._when(w, f"to_tile < {self.n_tiles}", "_enter"):
                    w(f"{self.n_enter}(row, to_tile, to_ph, to_rb)")
                with self._when(w, f"to_tile >= {self.n_tiles}", "_past"):
                    w(f"{cursor}[row, 0] = to_tile")

    def _fetch_once(self, w: _Writer, limit: str, name: str) -> None:
        """Start the inputs of the step cursor 1 is at, if fewer than `limit` steps' have started and it may
        start now (its windows and bounds read computed index arrays made for its iteration), and move it on."""
        plan, cursor, counters = self.plan, self.n_cursor, self.n_counters
        depth = self.pipeline.input_buffers
        conditions = [f"{counters}[0] < {limit}", f"{cursor}[1, 0] < {self.n_tiles}"]
        for i, c in enumerate(plan.computed):
            made = f"{cursor}[1, 0] + 1"
            if c.repeat is not None:
                made = f"{cursor}[1, 0] * {plan.repeats[c.repeat].range.count} + {cursor}[1, 3] + 1"
            reads = " | ".join(f"({cursor}[1, 1] == {p})" for p in c.consumers)
            conditions.append(f"(jnp.logical_not({reads}) | ({counters}[{1 + i}] >= {made}))")
        streamed = [op for op in plan.operands if op.kind is Kind.STREAM]
        iterated = [op for op in plan.operands if op.kind is Kind.ITERATION]
        with self._when(w, " & ".join(f"({c})" for c in conditions), name):
            consumers = sorted({p for c in plan.computed for p in c.consumers})
            if self.cursor_masked and consumers:
                # The cursor entered this phase before what bounds it was made: enter it again, now that it is.
                flags = 5 if self.repeated else 4
                at = " | ".join(f"({cursor}[1, 1] == {p})" for p in consumers)
                with self._when(w, f"({at}) & (({cursor}[1, {flags}] & 1) != 0)", "_enter_made"):
                    rb_now = f"{cursor}[1, 3]" if self.repeated else "0"
                    w(f"{self.n_enter}(1, {cursor}[1, 0], {cursor}[1, 1], {rb_now})")
            tile, ph, kb = self.local("ahead_tile"), self.local("ahead_ph"), self.local("ahead_kb")
            rb, first, empty = self.local("ahead_rb"), self.local("ahead_first"), self.local("ahead_empty")
            w(f"{self._targets(tile, ph, rb, kb, first if iterated else '_', '_', empty)} = {self.n_step}(1)")
            slot = self.local("ahead_slot")
            w(f"{slot} = {_int_expr(f'{counters}[0]', '%', depth)}")
            slots = self._iteration_slots(w, iterated, tile, rb, "ahead_iteration_slot")
            self._fetch_iterated(w, iterated, tile, ph, rb, first, empty, slots)
            for phase, ops in self._by_phase(streamed).items():
                with self._when(w, self._runs(ph, phase, empty), f"_fetch_phase{phase}"):
                    for op in ops:
                        w(f"{self.copy_fns[op.name]}({self.n_blocks}({tile}), {self._rb(rb)}{kb}, {slot}).start()")
            w(f"{self.n_advance}(1)", f"{counters}[0] = {counters}[0] + 1")

    def _produce(self, stmt: ast.stmt, code: _TileCode) -> list[ast.stmt]:
        """After a statement that makes computed index arrays: copy each into SMEM (staged in VMEM), count it
        made, and start the inputs of the steps that waited for it (as many as fit ahead)."""
        made = [(i, c) for i, c in enumerate(self.plan.computed) if c.name in assigned([stmt])]
        if not made:
            return []
        w = _Writer()
        for i, c in made:
            value = ast.unparse(code._expr(ast.Name(c.name, ast.Load())))
            stage, smem = self.stage[c.name], self.computed_smem[c.name]
            w(f"{stage}[...] = jnp.reshape({value}, {_staged(c.shape)}).astype(jnp.int32)")
            copy_ = self.local("staged")
            w(f"{copy_} = pltpu.make_async_copy({stage}, {smem}, {self.n_stage_sem})", f"{copy_}.start()")
            w(f"{copy_}.wait()", f"{self.n_counters}[{1 + i}] = {self.n_counters}[{1 + i}] + 1")
        for k in range(self.pipeline.input_buffers):
            self._fetch_once(w, f"{self.n_s} + {self.pipeline.input_buffers}", f"_catch_up{k}")
        return ast.parse(w.text()).body

    def _render(self, expr: ast.expr, block_of: dict[str, str]) -> str:
        """`expr` (of loop variables, index arrays' elements and constants) as kernel source: loop values from
        their blocks (`block_of`), elements read from SMEM, Python's `and`/`or`/`if` as jnp's."""
        gen = self

        runtime = {r.var for r in gen.plan.loops if r.bounds is not None}
        smem = {**gen.smem, **gen.computed_smem}
        shapes = {index.name: index.shape for index in gen.plan.indices} | {c.name: c.shape for c in gen.plan.computed}
        flat = {gen.computed_smem[c.name] for c in gen.plan.computed if len(c.shape) == 1}  # (staged as (1, n))

        class Values(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id in block_of:
                    return _parse_expr(f"({gen._val(gen._range(node.id), block_of[node.id], block_of)})")
                return ast.Name(smem[node.id], ast.Load()) if node.id in smem else node

            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                # An element at a position known only as the kernel runs (from a runtime range, or another
                # element) is clamped into its array, as JAX clamps: SMEM has no bounds checks.
                dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                late = any(reads(d) & runtime or element_reads(d, shapes) for d in dims)
                node = self.generic_visit(node)  # type: ignore[assignment]
                if not (isinstance(node.value, ast.Name) and node.value.id in smem.values()):
                    return node
                name = next(k for k, v in smem.items() if v == node.value.id)
                dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if late:
                    bounds = zip(dims, shapes[name], strict=True)
                    dims = [_parse_expr(f"jnp.clip(X, 0, {n - 1})", X=d) for d, n in bounds]
                if node.value.id in flat:
                    dims = [ast.Constant(0), *dims]
                node.slice = ast.Tuple(dims, ast.Load()) if len(dims) > 1 else dims[0]
                return node

        return ast.unparse(_Vectorize().visit(Values().visit(copy.deepcopy(expr))))

    def _val(self, r: Range, block: str, block_of: dict[str, str]) -> str:
        """The loop's value at a block index (a runtime range's from its start, read now)."""
        if r.bounds is None:
            return _scaled(block, r)
        start = self._render(r.bounds[0], block_of)
        term = block if r.step == 1 else f"{block} * {r.step}"
        return term if start == "0" else f"{start} + {term}"

    def _radix(self, w: _Writer, index: str, ranges: tuple[Range, ...]) -> list[str]:
        """Emit the blocks of `ranges` at row-major position `index`; their names."""
        names = []
        for i, r in enumerate(ranges):
            inner = math.prod(q.count for q in ranges[i + 1 :])
            value = "0" if r.count == 1 else _int_expr(index, "//", inner)
            if i and r.count > 1:
                value = _int_expr(value, "%", r.count)
            names.append(self.blk[r.var])
            w(f"{self.blk[r.var]} = {value}")
        return names

    def _blocks_fn(self, w: _Writer) -> None:
        sch, tile = self.schedule, self.n_tile
        with w.block(f"def {self.n_blocks}({tile}):"):
            w('"""The parallel loops\' blocks of the tile-th tile that runs."""')
            if not sch.tile_tables:
                names = self._radix(w, tile, sch.outer + sch.inner)
                for ring in self.plan.rings:  # (no mask skips a tile here: plan.py)
                    blk = self.blk[ring.loop]
                    w(f"# Ring order: ring step r walks the rows of chip (me - r) mod {ring.size}, this chip's first.")
                    within = _int_expr(blk, "%", ring.shard_blocks)
                    w(f"{blk} = {self._ring_step(ring, blk)} * {ring.shard_blocks} + {within}")
            else:
                t = self._pattern_index(w, tile, sch.pattern_tiles)
                names = self._radix(w, self.n_rep, sch.outer)
                for r in sch.inner:
                    w(f"{self.blk[r.var]} = {self.table_params[TILE_TABLE % r.var]}[{t}]")
                    names.append(self.blk[r.var])
            w(f"return ({_tuple(names)})")

    def _pattern_index(self, w: _Writer, index: str, per_pattern: int) -> str:
        """Emit `rep, t = divmod(index, per_pattern)` (rep is 0 when the pattern doesn't repeat); t's name."""
        if self.schedule.repeats == 1:  # the outer loops have one block each: no `rep` to compute
            return index
        t = self.local("t")
        w(f"{self.n_rep} = {_int_expr(index, '//', per_pattern)}")
        w(f"{t} = {index} - {self.n_rep} * {per_pattern}")
        return t

    def _window_index(self, tile: Tile, block_of: dict[str, str], element: int | None = None) -> str:
        """The index of `tile`'s window into its HBM array, given each loop's block expression (of a gathered
        window, its `element`-th element's)."""
        dims = []
        for k, (window, dim) in enumerate(zip(tile.windows, self._array_shape(tile.array), strict=True)):
            r = self._range(window.loop) if window.loop else None
            if window.index is not None and window.size is not None:  # gathered: one element of the slice
                dims.append(self._runtime_index(_element(window.index, element or 0), block_of, dim))
            elif window.index is not None:
                dims.append(self._runtime_index(window.index, block_of, dim))
            elif window.rows is not None:
                start, count = self._rows(tile, k, block_of, dim)
                dims.append(f"pl.ds({start}, {count})")
            elif r is None:
                if window.size is None:
                    dims.append(str(window.offset))
                elif window.offset == 0 and window.size == dim:
                    dims.append(":")
                else:
                    dims.append(f"pl.ds({window.offset}, {window.size})")
            elif r.bounds is not None:  # a runtime range moves it: clamped to the dim, as JAX clamps
                start = self._val(r, block_of[r.var], block_of)
                last = dim - (window.size or 1)
                dims.append(
                    f"jnp.clip({start}, 0, {last})"
                    if window.size is None
                    else f"pl.ds(jnp.clip({start}, 0, {last}), {window.size})"
                )
            else:
                start = _scaled(block_of[r.var], r)
                dims.append(start if window.size is None else f"pl.ds({start}, {window.size})")
        return ", ".join(dims)

    def _rows(self, tile: Tile, k: int, block_of: dict[str, str], dim: int) -> tuple[str, str]:
        """A ragged window's (dim `k` of `tile`'s) first row and how many rows it copies: those before its
        stop, inside the dim; along a tiled dim (one of the last two), rounded up to whole tiles, which is
        all a TPU copies there (an input's rows past the stop are unspecified)."""
        window = tile.windows[k]
        start, stop = (self._render(b, block_of) for b in window.rows)  # type: ignore[union-attr]
        first = f"jnp.clip({start}, 0, {dim})"
        count = f"jnp.clip(jnp.minimum({stop}, {dim}) - {first}, 0, {window.size})"
        last = len(tile.windows) - 1
        t = v5e.LANES if k == last else v5e.sublanes(tile.dtype) if k == last - 1 else 1
        if t > 1 and dim % t == 0:  # (Mosaic must see the start and size are whole tiles)
            count = f"pl.multiple_of(jnp.minimum(({count} + {t - 1}) // {t} * {t}, {dim} - {first}), {t})"
            first = f"pl.multiple_of({first}, {t})"
        return first, count

    def _slot_index(self, tile: Tile, element: int | None, block_of: dict[str, str]) -> str:
        """The part of a VMEM slot a copy of `tile`'s window fills ("": all of it): a gathered element's row,
        or a ragged window's first rows."""
        dims, part = [], False
        for k, (window, dim) in enumerate(zip(tile.windows, self._array_shape(tile.array), strict=True)):
            if window.size is None:
                continue
            if window.index is not None:
                dims.append(str(element))
                part = True
            elif window.rows is not None:
                dims.append(f"pl.ds(0, {self._rows(tile, k, block_of, dim)[1]})")
                part = True
            else:
                dims.append(":")
        return ", ".join(dims) if part else ""

    def _runtime_index(self, element: ast.Subscript, block_of: dict[str, str], dim: int) -> str:
        """An index read from an index array when the kernel runs, at the loops' values for these blocks,
        clamped to the dim (as JAX clamps indices)."""
        return f"jnp.clip({self._render(element, block_of)}, 0, {dim - 1})"

    def _array_shape(self, array: str) -> tuple[int, ...]:
        if array in self.nest.params:
            return self.plan.inputs[self.nest.params.index(array)].shape
        return next(out.shape for out in self.plan.outputs if out.name == array)

    def _range(self, var: str) -> Range:
        return next(r for r in self.plan.loops if r.var == var)

    def _unpack(self, w: _Writer, at: str) -> None:
        names = [self.blk[r.var] for r in self.plan.parallel]
        w(f"{_tuple(names)} = {at}")

    def _operand_copy_fn(self, w: _Writer, op: Operand) -> None:
        buf, sem, at, kb, slot = self.bufs[op.name], self.sems[op.name], self.n_at, self.n_kb, self.n_slot
        rb = f"{self.n_rb}, " if self.repeated else ""
        args = {
            Kind.STREAM: f"{at}, {rb}{kb}, {slot}",
            Kind.ITERATION: f"{at}, {rb}{slot}",
            Kind.TILE: f"{at}, {slot}",
            Kind.ONCE: "",
        }[op.kind]
        with w.block(f"def {self.copy_fns[op.name]}({args}):"):
            w(f"# {op.tile.text}")
            block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
            if op.kind is not Kind.ONCE:
                self._unpack(w, at)
            ring = self.rings.get(op.tile.array)
            if ring is not None and ring.size > 1:  # (its rows must have arrived; waited once)
                w(f"{self.ring_fns[ring.name]['arrive']}({self._ring_step(ring, self.blk[ring.loop])})")
            for r in self.plan.reductions:  # only the operand's own phase's loop moves it
                block_of[r.var] = kb if op.phase is not None and self.plan.phases[op.phase].range is r else "0"
            for r in self.plan.repeat_loops:  # (repeat loops move only streamed operands)
                block_of[r.var] = self.n_rb if op.kind in (Kind.STREAM, Kind.ITERATION) and self.repeated else "0"
            sem_ = sem if op.kind is Kind.ONCE else f"{sem}.at[{slot}]"
            copies = []
            for element in range(_gathered(op.tile)) if _gathered(op.tile) else [None]:
                src = f"{self.hbm[op.tile.array]}.at[{self._window_index(op.tile, block_of, element)}]"
                part = self._slot_index(op.tile, element, block_of)
                dst = buf if op.kind is Kind.ONCE else f"{buf}.at[{slot}]"
                dst = f"{dst}.at[{part}]" if part else dst
                copies.append(f"pltpu.make_async_copy({src}, {dst}, {sem_})")
            w(f"return {copies[0]}" if len(copies) == 1 else f"return {_COPIES}({', '.join(copies)})")

    def _moves_fn(self, w: _Writer, op: Operand) -> None:
        """`<op>_moves(at, prev)`: does the per-tile window move from tile `prev` to tile `at`?"""
        at, prev = self.n_at, self.n_prev
        with w.block(f"def {self.moved_fns[op.name]}({at}, {prev}):"):
            w(f"# {op.tile.text}")
            self._unpack(w, at)
            w(f"{_tuple([self.prev_blk[r.var] for r in self.plan.parallel])} = {prev}")
            now = self._window_starts(op.tile, {r.var: self.blk[r.var] for r in self.plan.parallel})
            before = self._window_starts(op.tile, dict(self.prev_blk))
            w(f"return {' | '.join(f'({a} != {b})' for a, b in zip(now, before, strict=True))}")

    def _window_starts(self, tile: Tile, block_of: dict[str, str]) -> list[str]:
        """Where `tile`'s window starts in each dim a loop (or index array) moves, given the loops' blocks."""
        starts = []
        for k, (window, dim) in enumerate(zip(tile.windows, self._array_shape(tile.array), strict=True)):
            if window.index is not None and window.size is not None:
                starts.append(self._runtime_index(_element(window.index, 0), block_of, dim))
            elif window.index is not None:
                starts.append(self._runtime_index(window.index, block_of, dim))
            elif window.rows is not None:
                starts += list(self._rows(tile, k, block_of, dim))
            elif window.loop is not None and self._range(window.loop).count > 1:
                starts.append(_scaled(block_of[window.loop], self._range(window.loop)))
        return starts

    def _output_copy_fn(self, w: _Writer, out: Output) -> None:
        at, slot, name, tile = self.n_at, self.n_slot, out.name, out.tile
        kb = "" if out.phase is None else f"{self.n_kb}, "  # a tile per step: placed by the phase's block too
        with w.block(f"def {self.copy_fns[name]}({at}, {kb}{slot}):"):
            w(f"# {tile.text}")
            self._unpack(w, at)
            block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
            if out.phase is not None:
                block_of[self.plan.phases[out.phase].range.var] = self.n_kb
            dst = f"{self.hbm[name]}.at[{self._window_index(tile, block_of)}]"
            part = self._slot_index(tile, None, block_of)
            src = f"{self.bufs[name]}.at[{slot}]" + (f".at[{part}]" if part else "")
            w(f"return pltpu.make_async_copy({src}, {dst}, {self.sems[name]}.at[{slot}])")

    def _fill_copy_fns(self, w: _Writer) -> None:
        sch, c = self.schedule, self.n_c
        with w.block(f"def {self.n_fill_blocks}({c}):"):
            w('"""The parallel loops\' blocks of the c-th tile the masks skip."""')
            f = self._pattern_index(w, c, sch.pattern_fills)
            names = self._radix(w, self.n_rep, sch.outer)
            for r in sch.inner:
                w(f"{self.blk[r.var]} = {self.table_params[FILL_TABLE % r.var]}[{f}]")
                names.append(self.blk[r.var])
            w(f"return ({_tuple(names)})")
        for out in self.filled:
            with w.block(f"def {self.fill_copies[out.name]}({self.n_at}):"):
                self._unpack(w, self.n_at)
                block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
                dst = f"{self.hbm[out.name]}.at[{self._window_index(out.tile, block_of)}]"
                w(f"return pltpu.make_async_copy({self.fill_bufs[out.name]}, {dst}, {self.fill_sems[out.name]})")

    # ------------------------------------------------------------------ before, during and after the loop

    def _at(self, tile: int) -> Any:
        """The blocks of the tile-th tile, in source: a tuple of ints, or `blocks(tile)` where they are known only
        as the kernel runs (a ring's order is this chip's)."""
        return f"{self.n_blocks}({tile})" if self.plan.rings else self.schedule.blocks_at(tile)

    def _ring_step(self, ring: Any, blk: str) -> str:
        """The ring step that reads (or the chip whose rows are) block `blk` of the ring's loop: the map between
        them is its own inverse, (me - x) mod size."""
        return f"jax.lax.rem({self.n_me} + {ring.size} - {_int_expr(blk, '//', ring.shard_blocks)}, {ring.size})"

    def _ring_fns(self, w: _Writer, ring: Any) -> None:
        """A ring's helpers: the rows of a chip's shard, the copies sending a ring step's rows to the next chip
        (this chip's own straight from its input) and receiving them, and `arrive`."""
        fns, hbm, size, axis = self.ring_fns[ring.name], self.hbm[ring.name], ring.size, ring.axis
        recv, send, arrived = self.ring_recv[ring.name], self.ring_send[ring.name], self.ring_arrived[ring.name]
        right = f"{{{axis!r}: jax.lax.rem({self.n_me} + 1, {size})}}"
        with w.block(f"def {fns['rows']}(k):"):
            w(f'"""The rows of `{ring.name}` that ring step k reads: those of chip (me - k) mod {size}."""')
            chip = f"jax.lax.rem({self.n_me} + {size} - k, {size})"
            w(f"return pl.ds(pl.multiple_of({chip} * {ring.rows}, {ring.rows}), {ring.rows})")
        if size == 1:
            return
        mesh = "device_id_type=pl.DeviceIdType.MESH"
        with w.block(f"def {fns['send_own']}():"):
            w(f'"""The rows of this chip, from `{ring.source}`, into `{ring.name}` on the next chip: step 1."""')
            dst = f"{hbm}.at[{fns['rows']}(0)]"
            w(f"return pltpu.make_async_remote_copy({self.hbm[ring.source]}, {dst}, {send}.at[0], {recv}.at[1],")
            w(f"                                     device_id={right}, {mesh})")
        with w.block(f"def {fns['send']}(k):"):
            w('"""The rows of ring step k, which arrived here, into the next chip: its ring step k + 1."""')
            rows = f"{hbm}.at[{fns['rows']}(k)]"
            w(f"return pltpu.make_async_remote_copy({rows}, {rows}, {send}.at[k], {recv}.at[k + 1],")
            w(f"                                     device_id={right}, {mesh})")
        with w.block(f"def {fns['receive']}(k):"):
            w('"""The rows of ring step k arriving from the previous chip (a semaphore of their own counts them)."""')
            rows = f"{hbm}.at[{fns['rows']}(k)]"
            w(f"return pltpu.make_async_remote_copy({rows}, {rows}, {send}.at[k - 1], {recv}.at[k],")
            w(f"                                     device_id={right}, {mesh})")
        with w.block(f"def {fns['arrive']}(r):"):
            w('"""Wait until the rows of ring step r (and of every step before it) have arrived, forwarding each')
            w('to the next chip as it comes."""')
            with w.block("def receive(k, carry):"):
                w(f"{fns['receive']}(k).wait_recv()")
                with self._when(w, f"k + 1 < {size}", "_forward"):
                    w(f"{fns['send']}(k).start()")
                w("return carry")
            w(f"jax.lax.fori_loop({arrived}[0] + 1, r + 1, receive, 0)")
            w(f"{arrived}[0] = jnp.maximum({arrived}[0], r)")

    def _ring_start(self, w: _Writer) -> None:
        """Before anything reads a gathered array: meet both neighbours (they write into this chip's buffer from
        now on), copy this chip's own rows in and send them on."""
        for ring in self.plan.rings:
            fns, size, axis = self.ring_fns[ring.name], ring.size, ring.axis
            w(f"# `{ring.name}`: this chip's rows first, sent on to the next chip on the ring.")
            if size > 1:
                w(f"{self.n_barrier} = pltpu.get_barrier_semaphore()")
                for side in (f"{self.n_me} + {size} - 1", f"{self.n_me} + 1"):
                    chip = f"{{{axis!r}: jax.lax.rem({side}, {size})}}"
                    w(f"pl.semaphore_signal({self.n_barrier}, device_id={chip}, device_id_type=pl.DeviceIdType.MESH)")
                w(f"pl.semaphore_wait({self.n_barrier}, 2)")
            own = self.local(f"{ring.name}_own")
            dst = f"{self.hbm[ring.name]}.at[{fns['rows']}(0)]"
            w(f"{own} = pltpu.make_async_copy({self.hbm[ring.source]}, {dst}, {self.ring_own[ring.name]})")
            w(f"{own}.start()")
            if size > 1:
                w(f"{fns['send_own']}().start()")
            w(f"{own}.wait()", f"{self.ring_arrived[ring.name]}[0] = 0")

    def _ring_end(self, w: _Writer) -> None:
        """Every ring step's rows arrived and went on (whatever the tiles read), and this chip's sends are done."""
        for ring in self.plan.rings:
            if ring.size == 1:
                continue
            fns = self.ring_fns[ring.name]
            w(f"# `{ring.name}`: every ring step's rows arrived and sent on.")
            w(f"{fns['arrive']}({ring.size - 1})")
            w(f"{fns['send_own']}().wait_send()")
            w(*[f"{fns['send']}({k}).wait_send()" for k in range(1, ring.size - 1)])

    def _setup(self, w: _Writer) -> None:
        plan, sch = self.plan, self.schedule
        w("")
        self._ring_start(w)
        if sch.runtime_tiles:
            r = plan.parallel[0]
            start, stop = (self._render(b, {}) for b in r.bounds)  # type: ignore[union-attr]
            w("# The tiles that run, from the parallel loop's bounds.")
            w(f"{self.n_tiles} = jnp.maximum(jax.lax.div(({stop}) - ({start}) + {r.step - 1}, {r.step}), 0)")
        if sch.runtime:
            count = f"{self.table_params[STEP_COUNT]}[0]"
            w(f"{self.n_steps} = {count}" if sch.repeats == 1 else f"{self.n_steps} = {sch.repeats} * {count}")
        if self.filled:
            w("# The fill value of the tiles the masks skip.")
            for out in self.filled:
                buf = self.fill_bufs[out.name]
                w(f"{buf}[...] = jnp.full({buf}.shape, {_literal(out.fill)}, {buf}.dtype)")
        once = [op for op in plan.operands if op.kind is Kind.ONCE]
        if once:
            w("# Inputs no loop moves: fetched once.")
            w(*[f"{self.copy_fns[op.name]}().start()" for op in once])
            w(*[f"{self.copy_fns[op.name]}().wait()" for op in once])
        per_tile = [op for op in plan.operands if op.kind is Kind.TILE]
        if per_tile:
            w("# The first tile's per-tile inputs.")
            at0 = self._at(0)
            with self._when(w, f"{self.n_tiles} > 0" if sch.runtime_tiles else None, "_first_tile_inputs"):
                w(*[f"{self.copy_fns[op.name]}({at0}, 0).start()" for op in per_tile])
            # (A shared window's slot flips when it moves: tile 0's moves into slot 0.)
            w(*[f"{self.tile_slots}[{k}] = 1" for k in range(len(self.shared))])
        streamed = [op for op in plan.operands if op.kind is Kind.STREAM]
        iterated = [op for op in plan.operands if op.kind is Kind.ITERATION]
        if self.burst:  # (every input a single-step kernel moves streams)
            first = min(self.burst, sch.steps)
            w(f"# The first burst's inputs ({first} step(s)).")
            for step in range(first):
                w(*[f"{self.copy_fns[op.name]}({self._at(step)}, 0, {step}).start()" for op in streamed])
            return
        if self.cursor:
            self._cursor_setup(w, streamed)
            return
        ahead = min(self.pipeline.input_buffers - 1, sch.steps)
        if sch.runtime_tiles:  # (a step per tile, counted only now)
            ahead = self.pipeline.input_buffers - 1
        if not ((streamed or iterated) and ahead):
            return
        w(f"# The first {ahead} step(s)' inputs.")
        if sch.runtime_tiles:
            for step in range(ahead):
                with self._when(w, f"{step} < {self.n_tiles}", f"_first_inputs{step}"):
                    w(*[f"{self.copy_fns[op.name]}({self.n_blocks}({step}), 0, {step}).start()" for op in streamed])
            return
        if sch.runtime:  # the steps are known only now
            names = ("first_tile", "first_ph", "first_rb", "first_kb", "first_first", "first_empty")
            tile, ph, rb, kb, first, empty = (self.local(name) for name in names)
            for step in range(ahead):
                with self._when(w, f"{step} < {self.n_steps}", f"_first_inputs{step}"):
                    w(f"{self._targets(tile, ph, rb, kb, first, '_', empty)} = {self.n_step}({step})")
                    for phase, ops in self._by_phase(streamed).items():
                        with self._when(w, self._runs(ph, phase, empty), f"_first_phase{phase}"):
                            for op in ops:
                                at = f"{self.n_blocks}({tile})"
                                w(f"{self.copy_fns[op.name]}({at}, {self._rb(rb)}{kb}, {step}).start()")
                    self._fetch_iterated(w, iterated, tile, ph, rb, first, empty)
            return
        for step in range(ahead):
            tile, phase, kb = sch.step_at(step)
            at, rb = self._at(tile), str(sch.iteration_at(step))
            for op in iterated:
                if op.phase == phase and sch.first_at(step) and not (op.in_body and sch.empty_at(step)):
                    w(f"{self.copy_fns[op.name]}({at}, {rb}, {self._iteration_slot(op, str(tile), rb)}).start()")
            if sch.empty_at(step):
                continue
            for op in streamed:
                if op.phase in (None, phase):
                    w(f"{self.copy_fns[op.name]}({at}, {self._rb(rb)}{kb}, {step}).start()")

    def _cursor_setup(self, w: _Writer, streamed: list[Operand]) -> None:
        """Both cursors at the first step, the counters at zero; the first `input_buffers - 1` steps' inputs
        started (those that may start), the second cursor moved past them."""
        cursor, counters = self.n_cursor, self.n_counters
        w("# The cursors: this step, and the next step whose inputs are fetched; the counters.")
        if self.cursor_masked:  # (at the first block the masks run)
            w(*[f"{self.n_enter}({row}, 0, 0, 0)" for row in (0, 1)])
        else:
            w(*[f"{cursor}[{row}, {i}] = 0" for row in (0, 1) for i in range(self.cursor_fields)])
        w(*[f"{counters}[{i}] = 0" for i in range(1 + len(self.plan.computed))])
        if not (streamed or any(op.kind is Kind.ITERATION for op in self.plan.operands)):
            return
        w(f"# The first {self.pipeline.input_buffers - 1} step(s)' inputs.")
        for step in range(self.pipeline.input_buffers - 1):
            self._fetch_once(w, str(self.pipeline.input_buffers - 1), f"_first_inputs{step}")

    def _iteration_slot(self, op: Operand, tile: str, rb: str) -> str:
        """The slot of an operand fetched once per repeat loop iteration: its iterations, counted over tiles,
        take the ring's slots in turn."""
        count = self.plan.repeats[self.plan.phases[op.phase].repeat].range.count  # type: ignore[index]
        return _int_expr(f"{tile} * {count} + {rb}", "%", self.pipeline.input_buffers)

    def _fetch_iterated(
        self,
        w: _Writer,
        ops: list[Operand],
        tile: str,
        ph: str,
        rb: str,
        first: str,
        empty: str,
        slots: dict[str, str] | None = None,
    ) -> None:
        """Start the copies of the operands fetched once per iteration for a step that is its phase's first
        (into `slots`, by operand, if given: names the step computed once)."""
        for (phase, in_body), group in self._by_iteration(ops).items():
            cond = self._iteration_first(phase, in_body, ph, first, empty)
            with self._when(w, cond, f"_iteration_inputs{phase}" + ("_body" if in_body else "")):
                for op in group:
                    slot = slots[op.name] if slots else self._iteration_slot(op, tile, rb)
                    w(f"{self.copy_fns[op.name]}({self.n_blocks}({tile}), {rb}, {slot}).start()")

    def _iteration_slots(self, w: _Writer, ops: list[Operand], tile: str, rb: str, base: str) -> dict[str, str]:
        """Emit each once-per-iteration operand's slot (one name per distinct slot expression); by operand."""
        names: dict[str, str] = {}
        slots: dict[str, str] = {}
        for op in ops:
            expr = self._iteration_slot(op, tile, rb)
            if expr not in names:
                names[expr] = self.local(base)
                w(f"{names[expr]} = {expr}")
            slots[op.name] = names[expr]
        return slots

    @staticmethod
    def _by_iteration(ops: list[Operand]) -> dict[tuple[int, bool], list[Operand]]:
        """Operands fetched once per iteration, grouped by phase and by whether only the phase's body reads
        them."""
        groups: dict[tuple[int, bool], list[Operand]] = {}
        for op in ops:
            groups.setdefault((op.phase, op.in_body), []).append(op)  # type: ignore[arg-type]
        return groups

    def _iteration_first(self, phase: int, in_body: bool, ph: str, first: str, empty: str) -> str:
        """The condition that a step fetches (waits for) phase `phase`'s once-per-iteration operands: the
        phase's first step of an iteration, unless only the phase's body reads them and the step is empty."""
        cond = f"({ph} == {phase}) & {first}" if len(self.plan.phases) > 1 else first
        if in_body and self._may_be_empty(phase):
            cond = f"{cond} & jnp.logical_not({empty})"
        return cond

    def _region(self, w: _Writer, region: str) -> Any:
        if not self.trace:
            return nullcontext()
        return w.block(f"with jax.named_scope({f'{self.nest.name}/{region}'!r}):")

    def _when(self, w: _Writer, cond: str | None, name: str) -> Any:
        """A `pl.when(cond)` block; no block for a condition of None (always)."""
        if cond is None:
            return nullcontext()
        w(f"@pl.when({cond})")
        return w.block(f"def {self.local(name)}():")

    def _by_phase(self, ops: list[Operand]) -> dict[int | None, list[Operand]]:
        """Streamed operands by the phase they stream in (None: every step)."""
        groups: dict[int | None, list[Operand]] = {}
        for op in ops:
            groups.setdefault(op.phase, []).append(op)
        return groups

    def _in_phase(self, ph: str, phase: int | None) -> str | None:
        """The condition that a step of phase `ph` is in `phase` (None: always)."""
        return None if phase is None or len(self.plan.phases) == 1 else f"{ph} == {phase}"

    def _runs(self, ph: str, phase: int | None, empty: str) -> str | None:
        """The condition that a step of phase `ph` (and emptiness `empty`) runs `phase`'s body (None: always)."""
        conditions = [c for c in (self._in_phase(ph, phase),) if c]
        if self._may_be_empty(phase):
            conditions.append(f"jnp.logical_not({empty})")
        return " & ".join(f"({c})" if len(conditions) > 1 else c for c in conditions) or None

    def _may_be_empty(self, phase: int | None) -> bool:
        """Can a step of `phase` be empty? Only a masked phase's, in a schedule that has empty steps."""
        r = self.plan.phases[phase].range if phase is not None else None
        return self.schedule.empty and r is not None and (r.mask is not None or r.bounds is not None)

    def _targets(self, tile: str, ph: str, rb: str, kb: str, first: str, last: str, empty: str) -> str:
        """The names to unpack a decoded step into (the iteration and emptiness only where steps have them)."""
        names = [tile, ph, *([rb] if self.repeated else []), kb, first, last, *([empty] if self.schedule.empty else [])]
        return ", ".join(names)

    def _rb(self, rb: str) -> str:
        """A streamed operand's copy's argument for the iteration of its repeat loop (if the kernel has any)."""
        return f"{rb}, " if self.repeated else ""

    def _ends(self, phase: int, ph: str, flag: str, last: bool) -> str:
        """The condition that a step of `ph` is `phase`'s `flag` (first or last) step, in the first (or last)
        iteration of its repeat loop, if it is in one: a tile's first (last) step."""
        conditions = [f"({ph} == {phase})"] if len(self.plan.phases) > 1 else []
        conditions.append(flag)
        r = self.plan.phases[phase].repeat
        if r is not None:
            conditions.append(f"({self.n_rb} == {self.plan.repeats[r].range.count - 1 if last else 0})")
        return " & ".join(conditions)

    def _loop(self, w: _Writer) -> None:
        plan, sch, pipe = self.plan, self.schedule, self.pipeline
        s, tile, ph, kb, at = self.n_s, self.n_tile, self.n_ph, self.n_kb, self.n_at
        first, last, empty = self.n_first, self.n_last, self.n_empty
        phases = len(plan.phases)
        steps = self.n_steps if sch.runtime else self.n_tiles if sch.runtime_tiles else str(sch.steps)
        if self.single:  # a step is a tile: it is every phase's first and last step
            tile_first = tile_last = None
        else:
            tile_first = self._ends(0, ph, first, last=False)
            tile_last = self._ends(phases - 1, ph, last, last=True)
        w("")
        body = self.local("_steps")
        if not self.cursor:
            w(f"@pl.loop(0, {steps})")
        with w.block(f"def {body}({s}):"):
            if self.single:
                w(f"{tile}, {kb} = {s}, 0")
            else:
                at_step = "0" if self.cursor else s
                w(f"{self._targets(tile, ph, self.n_rb, kb, first, last, empty)} = {self.n_step}({at_step})")
            w(f"{at} = {self.n_blocks}({tile})")

            streamed = [op for op in plan.operands if op.kind is Kind.STREAM]
            iterated = [op for op in plan.operands if op.kind is Kind.ITERATION]
            depth = pipe.input_buffers
            if self.burst:
                self._burst_start(w, streamed)
            elif self.cursor and (streamed or iterated):  # this step's, or later ones' if steps waited
                self._fetch_once(w, f"{s} + {depth}", "_fetch_ahead")
            elif streamed or iterated:
                with self._when(w, f"{s} + {depth - 1} < {steps}", "_fetch_ahead"):
                    ahead_tile, ahead_ph = self.local("ahead_tile"), self.local("ahead_ph")
                    ahead_kb, ahead_empty = self.local("ahead_kb"), self.local("ahead_empty")
                    ahead_rb, ahead_first = self.local("ahead_rb"), self.local("ahead_first")
                    if self.single:
                        w(f"{ahead_tile}, {ahead_kb} = {s} + {depth - 1}, 0")
                    else:
                        first_ = ahead_first if iterated else "_"
                        targets = self._targets(ahead_tile, ahead_ph, ahead_rb, ahead_kb, first_, "_", ahead_empty)
                        w(f"{targets} = {self.n_step}({s} + {depth - 1})")
                    ahead_slots = self._iteration_slots(w, iterated, ahead_tile, ahead_rb, "ahead_iteration_slot")
                    self._fetch_iterated(
                        w, iterated, ahead_tile, ahead_ph, ahead_rb, ahead_first, ahead_empty, ahead_slots
                    )
                    ahead_slot = _int_expr(f"{s} + {depth - 1}", "%", depth)
                    if streamed and depth & (depth - 1):  # (one division per step, not one per copy)
                        name = self.local("ahead_slot")
                        w(f"{name} = {ahead_slot}")
                        ahead_slot = name
                    for phase, ops in self._by_phase(streamed).items():
                        with self._when(w, self._runs(ahead_ph, phase, ahead_empty), f"_fetch_phase{phase}"):
                            for op in ops:
                                blocks = f"{self.n_blocks}({ahead_tile})"
                                rb = self._rb(ahead_rb)
                                w(f"{self.copy_fns[op.name]}({blocks}, {rb}{ahead_kb}, {ahead_slot}).start()")

            per_tile = [op for op in plan.operands if op.kind is Kind.TILE]
            slots = {Kind.STREAM: _int_expr(s, "%", depth), Kind.TILE: _int_expr(tile, "%", TILE_SLOTS)}
            if streamed and depth & (depth - 1) and not self.burst:  # (computed once per step)
                name = self.local("slot")
                w(f"{name} = {slots[Kind.STREAM]}")
                slots[Kind.STREAM] = name
            iteration_slots = self._iteration_slots(w, iterated, tile, self.n_rb, "iteration_slot")
            # (A kernel whose tiles take steps waits for a phase's inputs, and sees their slots, in the phase's
            # own code: below, in `_tile_code`.)
            phased = [] if self.single else [*streamed, *iterated]
            with self._region(w, WAIT) if per_tile or (self.single and streamed) else nullcontext():
                if self.burst and streamed:  # the first burst's steps wait for their inputs (later ones: at its end)
                    with self._when(w, f"{s} < {self.burst}", "_first_burst"):
                        w(*[f"{self.copy_fns[op.name]}({at}, 0, {slots[Kind.STREAM]}).wait()" for op in streamed])
                if per_tile:
                    with self._when(w, tile_first, "_tile_inputs"):
                        self._tile_inputs(w, per_tile, slots[Kind.TILE])
                if self.single and not self.burst:
                    w(*[f"{self.copy_fns[op.name]}({at}, {kb}, {slots[Kind.STREAM]}).wait()" for op in streamed])
            for op in plan.operands:
                if op not in phased:
                    self._view(w, op, slots, iteration_slots)
            self._loop_vars(w)

            out_slots = pipe.output_buffers
            per_tile_outs = [out for out in plan.outputs if out.phase is None and not self.burst]
            with self._region(w, OUT_WAIT) if per_tile_outs else nullcontext():
                cond = f"{tile} >= {out_slots}" if tile_last is None else f"{tile_last} & ({tile} >= {out_slots})"
                with self._when(w, cond, "_free_output_slot") if per_tile_outs else nullcontext():
                    zero = (0,) * len(plan.parallel)
                    for out in per_tile_outs:  # (a ragged copy's size is its tile's: the copy of tile - slots)
                        at = f"{self.n_blocks}({tile} - {out_slots})" if _ragged(out.tile) else zero
                        w(f"{self.copy_fns[out.name]}({at}, {_int_expr(tile, '%', out_slots)}).wait()")

            if self.filled:
                with self._when(w, tile_first, "_fill_skipped_tiles"):
                    self._start_fills(w)

            self._tile_code(w, tile_first, slots, iteration_slots)
            if self.cursor:
                w(f"{self.n_advance}(0)", f"return {s} + 1")
            if self.burst:
                last = (sch.steps - 1) // self.burst * self.burst  # the last burst's first step
                with self._when(w, f"{s} >= {last}", "_last_burst"):  # (no reads left: its outputs go now)
                    slot = _int_expr(tile, "%", self.burst)
                    w(*[f"{self.copy_fns[out.name]}({at}, {slot}).start()" for out in plan.outputs])
                self._burst_end(w, streamed)
        if self.cursor:  # until the cursor has passed the last tile
            w(f"jax.lax.while_loop(lambda _: {self.n_cursor}[0, 0] < {self.n_tiles}, {body}, jnp.int32(0))")

    def _burst_start(self, w: _Writer, streamed: list[Operand]) -> None:
        """At a burst's first step: wait until the previous burst's outputs are out, then start fetching the
        next burst's inputs (so the copies in flight all read, or all write)."""
        s, g, steps, zero = self.n_s, self.burst, self.schedule.steps, (0,) * len(self.plan.parallel)
        with self._when(w, f"{_int_expr(s, '%', g)} == 0", "_next_burst"):
            with self._region(w, OUT_WAIT), self._when(w, f"{s} > 0", "_outputs_out"):
                for out in self.plan.outputs:
                    w(*[f"{self.copy_fns[out.name]}({zero}, {j}).wait()" for j in range(g)])
            for j in range(g if streamed else 0):
                ahead = f"{s} + {g + j}"
                with self._when(w, f"{ahead} < {steps}", f"_fetch{j}"):
                    slot = _int_expr(ahead, "%", 2 * g)
                    w(*[f"{self.copy_fns[op.name]}({self.n_blocks}({ahead}), 0, {slot}).start()" for op in streamed])

    def _burst_end(self, w: _Writer, streamed: list[Operand]) -> None:
        """At the last step of a burst before the last: wait for the next burst's inputs, then start copying
        this burst's outputs out (its steps stored them in their slots)."""
        s, g, steps = self.n_s, self.burst, self.schedule.steps
        with self._when(w, f"({_int_expr(s, '%', g)} == {g - 1}) & ({s} + 1 < {steps})", "_end_burst"):
            with self._region(w, WAIT) if streamed else nullcontext():
                for j in range(g if streamed else 0):
                    after = f"{s} + {j + 1}"
                    with self._when(w, f"{after} < {steps}", f"_wait{j}"):
                        slot = _int_expr(after, "%", 2 * g)
                        w(*[f"{self.copy_fns[op.name]}({self.n_blocks}({after}), 0, {slot}).wait()" for op in streamed])
            for j in range(g):
                at = f"{self.n_blocks}({s} - {g - 1 - j})" if j < g - 1 else self.n_at
                w(*[f"{self.copy_fns[out.name]}({at}, {j}).start()" for out in self.plan.outputs])

    def _tile_inputs(self, w: _Writer, per_tile: list[Operand], slot: str) -> None:
        """At a tile's first step: start the next tile's per-tile inputs, and wait for this tile's. A window the
        previous tile had is in its slot already; one the next tile shares isn't fetched again."""
        tile, at, blocks, slots = self.n_tile, self.n_at, self.n_blocks, self.tile_slots
        moves = {}
        for k, op in enumerate(self.shared):
            moves[op.name] = self.local(f"{op.name}_moved")
            prev = f"{blocks}(jnp.maximum({tile} - 1, 0))"
            w(f"{moves[op.name]} = ({tile} == 0) | {self.moved_fns[op.name]}({at}, {prev})")
            w(f"{slots}[{k}] = jnp.where({moves[op.name]}, 1 - {slots}[{k}], {slots}[{k}])")
        with self._when(w, f"{tile} + 1 < {self.n_tiles}", "_next_tile_inputs"):
            next_slot = _int_expr(f"{tile} + 1", "%", TILE_SLOTS)
            for op in per_tile:
                if op in self.shared:
                    k = self.shared.index(op)
                    with self._when(w, f"{self.moved_fns[op.name]}({blocks}({tile} + 1), {at})", f"_next_{op.name}"):
                        w(f"{self.copy_fns[op.name]}({blocks}({tile} + 1), 1 - {slots}[{k}]).start()")
                else:
                    w(f"{self.copy_fns[op.name]}({blocks}({tile} + 1), {next_slot}).start()")
        for op in per_tile:
            if op in self.shared:
                with self._when(w, moves[op.name], f"_wait_{op.name}"):
                    w(f"{self.copy_fns[op.name]}({at}, {slots}[{self.shared.index(op)}]).wait()")
            else:
                w(f"{self.copy_fns[op.name]}({at}, {slot}).wait()")

    def _view(
        self, w: _Writer, op: Operand, slots: dict[Kind, str], iteration_slots: dict[str, str], words: bool = True
    ) -> None:
        """Bind the name the tile code reads `op` through: its VMEM slot this step (and its words, which the
        interpreter loads when they're bound: once the slot's copy is waited for)."""
        buf = self.bufs[op.name]
        if op in self.shared:
            slot = f"{self.tile_slots}[{self.shared.index(op)}]"
        elif op.kind is Kind.ITERATION:
            slot = iteration_slots[op.name]
        else:
            slot = slots.get(op.kind)  # type: ignore[assignment]
        view = buf if op.kind is Kind.ONCE else f"{buf}.at[{slot}]"
        w(f"{self.views[op.name]} = {view}")
        if words and op.name in self.words:  # (a 16-bit (..., 2, d) slot holds (..., d) words)
            words = f"{buf}.bitcast(jnp.uint32)" + ("" if op.kind is Kind.ONCE else f".at[{slot}]")
            shape = f"{math.prod(op.tile.shape[:-2])}, {op.tile.shape[-1]}"
            w(f"{self.words[op.name]} = {words}.reshape({shape})")
            with w.block(f"if {self.n_interpret}:"):
                w(f"{self.words[op.name]} = {_PAIRS}({self.views[op.name]}[...]).reshape({shape})")

    def _phase_inputs(
        self, w: _Writer, p: int, slots: dict[Kind, str], iteration_slots: dict[str, str], before: bool
    ) -> None:
        """In phase p's code: wait for its once-per-iteration inputs at its first step (`before` its code
        before the phase), or else for its streamed inputs at a step that runs its body; and see their slots."""
        at, rb, kb = self.n_at, self.n_rb, self.n_kb
        empty = f"jnp.logical_not({self.n_empty})" if self._may_be_empty(p) else None
        if before:  # (at the phase's first step: the caller's branch)
            iterated = [op for op in self.plan.operands if op.kind is Kind.ITERATION and op.phase == p]
            for in_body, ops in itertools.groupby(sorted(iterated, key=lambda op: op.in_body), lambda op: op.in_body):
                with self._when(w, empty if in_body else None, f"_wait_iteration{p}_body"), self._region(w, WAIT):
                    w(*[f"{self.copy_fns[op.name]}({at}, {rb}, {iteration_slots[op.name]}).wait()" for op in ops])
            return
        streamed = [op for op in self.plan.operands if op.kind is Kind.STREAM and op.phase == p]
        if streamed:
            with self._when(w, empty, f"_wait_phase{p}"), self._region(w, WAIT):
                for op in streamed:
                    w(f"{self.copy_fns[op.name]}({at}, {self._rb(rb)}{kb}, {slots[Kind.STREAM]}).wait()")
        for op in streamed:  # (after the waits: the interpreter loads words from a view when it's bound)
            self._view(w, op, slots, iteration_slots)

    def _tile_code(
        self, w: _Writer, tile_first: str | None, slots: dict[Kind, str], iteration_slots: dict[str, str]
    ) -> None:
        """The segments and the phase bodies: the code before the phases at a tile's first step; each
        phase's body at its steps (not at an empty one), with the code after it at its last step (where the
        body's values are still in scope). In a phase's code, its inputs are waited for where they are first
        needed: once-per-iteration ones before the code before the phase, streamed ones after it (so that code
        runs while they arrive)."""
        plan, code = self.plan, _TileCode(self)
        if self.single:
            with self._region(w, COMPUTE):
                w(*code.emit(plan.segments[0], "start", set()))
            return
        if plan.segments[0]:
            with self._when(w, tile_first, "_start_tile"), self._region(w, COMPUTE):
                w(*code.emit(plan.segments[0], "start", set()))
        carried = set(plan.carried)
        for p, phase in enumerate(plan.phases):
            with self._when(w, self._in_phase(self.n_ph, p), f"_phase{p}"):
                iterated = [op for op in plan.operands if op.kind is Kind.ITERATION and op.phase == p]
                for op in iterated:
                    self._view(w, op, slots, iteration_slots, words=False)
                if iterated or phase.before:
                    with self._when(w, self.n_first, f"_first_of_phase{p}"):
                        self._phase_inputs(w, p, slots, iteration_slots, before=True)
                        if phase.before:  # a repeat loop's code before its first phase, every iteration
                            with self._region(w, COMPUTE):
                                w(*code.emit(phase.before, "between", set()))
                for op in iterated:  # (their words: once they're in)
                    if op.name in self.words:
                        self._view(w, op, slots, iteration_slots)
                self._phase_inputs(w, p, slots, iteration_slots, before=False)
                empty = f"jnp.logical_not({self.n_empty})" if self._may_be_empty(p) else None
                with self._when(w, empty, f"_body{p}"), self._region(w, COMPUTE):
                    w(*code.emit(phase.body, "body", set()))
                after = plan.segments[p + 1]
                repeat = next((rep for rep in plan.repeats if rep.last - 1 == p), None)
                ends_repeat = repeat is not None and bool(repeat.after)
                if not after and not ends_repeat:
                    continue
                final = p == len(plan.phases) - 1
                # At the end, carried values are just values; before it, they're stored for later steps.
                outside = assigned(phase.body) | carried if final and not repeat else assigned(phase.body) - carried
                name = "_end_tile" if final and not repeat else f"_after_phase{p}"
                with self._when(w, self.n_last, name), self._region(w, COMPUTE):
                    w(*code.emit(after, "between" if repeat or not final else "end", outside))
                    if ends_repeat:  # the tile's code after the repeat loop, as its last iteration ends
                        cond = f"{self.n_rb} == {repeat.range.count - 1}"  # type: ignore[union-attr]
                        made = assigned(phase.body) | assigned(after)
                        with self._when(w, cond, "_end_tile" if final else f"_end_repeat{p}"):
                            more = made | carried if final else made - carried
                            w(*code.emit(repeat.after, "end" if final else "between", more, inherit=True))  # type: ignore[union-attr]

    def _loop_vars(self, w: _Writer) -> None:
        """Bind the loop variables the tile code reads, as weakly typed ints (Python ints in the loop nest).
        A reduction loop of one iteration binds its variable in the tile code itself (plan.py)."""
        plan = self.plan
        used = self.value_reads
        names = [self.blk[r.var] for r in plan.parallel]
        runtime = any(phase.range.bounds is not None and phase.range.var in used for phase in plan.phases)
        if runtime or any(r.var in used for r in plan.parallel):  # (a runtime range starts where its tile says)
            w(f"{_tuple(names)} = {self.n_at}")
        steps = [phase.range for phase in plan.phases]
        for r in plan.repeat_loops:  # a repeat loop of one iteration: its variable is a constant
            if r.count == 1 and r.var in used:
                w(f"{r.var} = {r.start}")
        for r in (*plan.parallel, *steps, *(rep.range for rep in plan.repeats)):
            if r.var not in used:
                continue
            block = self.blk[r.var] if r in plan.parallel else self.n_kb if r in steps else self.n_rb
            value = self._val(r, block, {q.var: self.blk[q.var] for q in plan.parallel})
            if r.count > 1:
                value = f"{_WEAK}({value})"
            w(f"{r.var} = {value}")

    def _start_fills(self, w: _Writer) -> None:
        sch, tile, c = self.schedule, self.n_tile, self.n_c
        begin = f"jax.lax.div({tile} * {sch.fills}, {sch.tiles})"
        end = f"jax.lax.div(({tile} + 1) * {sch.fills}, {sch.tiles})"
        w(f"# This tile's share of the {sch.fills} skipped tiles, at most {FILLS_IN_FLIGHT} copies in flight.")
        w(f"@pl.loop({begin}, {end})")
        with w.block(f"def {self.local('_fills')}({c}):"):
            zero = (0,) * len(self.plan.parallel)
            with self._when(w, f"{c} >= {FILLS_IN_FLIGHT}", "_wait_for_a_fill"):
                w(*[f"{self.fill_copies[out.name]}({zero}).wait()" for out in self.filled])
            w(f"{self.n_at} = {self.n_fill_blocks}({c})")
            w(*[f"{self.fill_copies[out.name]}({self.n_at}).start()" for out in self.filled])

    def _drain(self, w: _Writer) -> None:
        sch, pipe = self.schedule, self.pipeline
        w("")
        w("# Wait for the copies still in flight.")
        zero = (0,) * len(self.plan.parallel)
        ob = pipe.output_buffers
        for out in self.plan.outputs:
            if self.burst:  # the last burst's (single-step tiles: a step per tile)
                w(
                    *[
                        f"{self.copy_fns[out.name]}({zero}, {j}).wait()"
                        for j in range(sch.steps - (sch.steps - 1) // ob * ob)
                    ]
                )
                continue
            if sch.runtime_tiles:  # the last tiles' copies, of the tiles counted when the kernel started
                for i in range(ob):
                    last = f"{self.n_tiles} - {i + 1}"
                    with self._when(w, f"{self.n_tiles} > {i}", f"_last_out{i}"):
                        slot = _int_expr(f"({last})", "%", ob)
                        w(f"{self.copy_fns[out.name]}({self.n_blocks}({last}), {slot}).wait()")
                continue
            # The last copies of each output: a tile per tile, or per step of its phase.
            stores = sch.tiles * (1 if out.phase is None else self.plan.phases[out.phase].range.count)
            kb = "" if out.phase is None else "0, "
            for n in range(max(0, stores - ob), stores):
                at = self._at(n) if _ragged(out.tile) else zero  # (a ragged copy's size is its tile's)
                w(f"{self.copy_fns[out.name]}({at}, {kb}{n % ob}).wait()")
        for _ in range(min(sch.fills, FILLS_IN_FLIGHT) if self.filled else 0):
            w(*[f"{self.fill_copies[out.name]}({zero}).wait()" for out in self.filled])
        self._ring_end(w)


def _staged(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The 2-D shape a computed index array is staged and kept in (a 1-D one as one row)."""
    return shape if len(shape) == 2 else (1, shape[0])


def _ragged(tile: Tile) -> bool:
    """Does `tile`'s window copy as many rows as its bounds say (`autokernel.rows`)?"""
    return any(w.rows is not None for w in tile.windows)


def _gathered(tile: Tile) -> int:
    """How many elements `tile`'s window gathers through an index array (0: it gathers none)."""
    return next((w.size for w in tile.windows if w.index is not None and w.size is not None), 0)  # type: ignore[return-value]


def _element(gather: ast.Subscript, element: int) -> ast.Subscript:
    """The index array element a gathered window's `element`-th row reads: `idx[a : a + n]` -> `idx[a + element]`."""
    dims = list(gather.slice.elts) if isinstance(gather.slice, ast.Tuple) else [gather.slice]
    dims = [_parse_expr(f"({ast.unparse(d.lower)}) + {element}") if isinstance(d, ast.Slice) else d for d in dims]  # type: ignore[arg-type]
    index = ast.Tuple(dims, ast.Load()) if len(dims) > 1 else dims[0]
    return ast.Subscript(gather.value, index, ast.Load())


def _may_stay(op: Operand, plan: Plan) -> bool:
    """May a per-tile window be the same for consecutive tiles? If an index array places it (or its rows), or
    a parallel loop of more than one block doesn't move it."""
    moved_by_all = all(r.var in op.tile.loops for r in plan.parallel if r.count > 1)
    return any(w.index is not None or w.rows is not None for w in op.tile.windows) or not moved_by_all


def _value_reads(plan: Plan) -> set[str]:
    """The names the tile code reads as values: not in the indices of its tile reads and stores, which the
    kernel's copies compute from block indices instead."""

    class DropTiles(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            return ast.Constant(None) if plan.read_key(node) in plan.reads else self.generic_visit(node)  # type: ignore[return-value]

    names: set[str] = set()
    for stmt in flatten(plan.nest.tile):
        store = plan.nest.store(stmt)
        part = copy.deepcopy(store.value if store else stmt)
        names |= reads(DropTiles().visit(part))
    return names


class _TileCode:
    """The tile code as kernel code: input reads from VMEM slots, carried values in VMEM scratch, stores into
    output slots followed by their copies to HBM, `@` on the MXU."""

    def __init__(self, gen: _Generator) -> None:
        self.gen, self.plan = gen, gen.plan
        self.segment = ""
        self.outside: set[str] = set()
        self.renamed: dict[str, str] = {}

    def emit(self, stmts: tuple[ast.stmt, ...], segment: str, outside: set[str], inherit: bool = False) -> list[str]:
        """Source lines for `stmts`. The code after a phase runs in a function of its own, inside the phase's:
        an assignment there to a name in `outside` (the phase body's, and at the end, carried values') gives
        it a fresh name from then on. Before the end, a carried value is read from its VMEM buffer until the
        code assigns it, is a local after that, and its last assignment stores it. With `inherit`, the code
        continues the code emitted last (in a function inside its function), keeping its fresh names."""
        self.segment, self.outside = segment, outside
        if not inherit:
            self.renamed: dict[str, str] = {}
        self.last: dict[str, int] = {}  # carried name -> id of its last assignment in `stmts`
        if segment != "end":
            for stmt in flatten(stmts):
                for name in assigned([stmt]) & set(self.plan.carried):
                    self.last[name] = id(stmt)
        lines = []
        for stmt in stmts:
            for new in self._statement(stmt):
                lines += ast.unparse(ast.fix_missing_locations(new)).splitlines()
        return lines

    def _statement(self, stmt: ast.stmt) -> list[ast.stmt]:
        """`stmt` as kernel code (after a statement making computed index arrays: their copies into SMEM)."""
        return self._kernel_statement(stmt) + (self.gen._produce(stmt, self) if self.plan.computed else [])

    def _kernel_statement(self, stmt: ast.stmt) -> list[ast.stmt]:
        gen, plan = self.gen, self.plan
        store = plan.nest.store(stmt)
        if store is not None:
            out = next(o for o in plan.outputs if o.name == store.output)
            ob, buf = gen.pipeline.output_buffers, gen.bufs[out.name]
            value = self._cast(self._expr(store.value), out.value, out.tile.shape, out.dtype, f"{buf}.dtype")

            def put(slot: str) -> ast.stmt:
                if out.name in plan.unpacked:  # (words: into the buffer seen as words)
                    words = self._expr(store.value.args[0])  # type: ignore[attr-defined]
                    return _parse_stmt(f"{_UNPAIRS}({buf}, {slot}, WORDS, {gen.n_interpret})", WORDS=words)
                return _parse_stmt(f"{buf}[{slot}] = VALUE", VALUE=value)

            if out.phase is None:
                slot = _int_expr(gen.n_tile, "%", ob)
                store_ = put(slot)
                if gen.burst:  # copied out at the burst's end
                    return [store_]
                return [store_, _parse_stmt(f"{gen.copy_fns[out.name]}({gen.n_at}, {slot}).start()")]
            # A tile per step: the n-th waits for the copy of the (n - slots)-th to free its slot.
            n = f"{gen.n_tile} * {plan.phases[out.phase].range.count} + {gen.n_kb}"
            slot, zero = _int_expr(n, "%", ob), (0,) * len(plan.parallel)
            free = gen.local(f"_free_{out.name}_slot")
            wait = f"@pl.when({n} >= {ob})\ndef {free}():\n    {gen.copy_fns[out.name]}({zero}, 0, {slot}).wait()"
            return [
                ast.parse(wait).body[0],
                put(slot),
                _parse_stmt(f"{gen.copy_fns[out.name]}({gen.n_at}, {gen.n_kb}, {slot}).start()"),
            ]
        if isinstance(stmt, ast.With):
            new = copy.copy(stmt)
            new.body = [s for inner in stmt.body for s in self._statement(inner)]
            return [new]
        update = as_store(stmt)
        scalars = {r.var for r in plan.loops} | set(plan.constants)
        if (
            update is not None
            and update.output in plan.carried
            and set().union(*(reads(i) for i in update.index)) <= scalars
        ):  # `a = a.at[index].set(v)` of a carried value: store just that part of its VMEM buffer
            ref, index = gen.refs[update.output], ", ".join(ast.unparse(self._expr(i)) for i in update.index)
            value = _parse_expr(
                f"jnp.broadcast_to(VALUE, {ref}.at[{index}].shape).astype({ref}.dtype)", VALUE=self._expr(update.value)
            )
            stores = []
            if update.output in self.renamed:  # a value on its way, in a local: into the buffer first
                local = self.renamed.pop(update.output)
                stores.append(_parse_stmt(f"{ref}[...] = jnp.broadcast_to({local}, {ref}.shape).astype({ref}.dtype)"))
            return [*stores, _parse_stmt(f"{ref}[{index}] = VALUE", VALUE=value)]
        types_ = plan.assigned_types.get(id(stmt), {})
        original = stmt
        if isinstance(stmt, ast.AugAssign):  # `a op= b` is `a = a op b` (arrays are immutable)
            name = stmt.target.id  # type: ignore[union-attr]
            value = ast.BinOp(ast.Name(name, ast.Load()), stmt.op, stmt.value)
            stmt = ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), stmt)
        assert isinstance(stmt, ast.Assign)
        value = self._expr(stmt.value)
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and self.last.get(target.id) == id(original):  # the value to keep
            c = plan.carried[target.id]
            ref = gen.refs[c.name]
            self.renamed.pop(c.name, None)  # read from the buffer from here on
            value = self._cast(value, types_[c.name], c.shape, c.dtype, f"{ref}.dtype")
            return [_parse_stmt(f"{ref}[...] = VALUE", VALUE=value)]
        names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
        carried = [n for n in names if n in self.last]
        for name in carried:  # values on their way: locals (`a, b = ...` unpacks into them)
            self.renamed[name] = name
        new = [ast.Assign([self._rename_targets(copy.deepcopy(target))], value)]
        for name in carried:
            if self.last[name] == id(original):  # the value to keep: store it
                c, ref = plan.carried[name], gen.refs[name]
                kept = self._cast(ast.Name(name, ast.Load()), types_[name], c.shape, c.dtype, f"{ref}.dtype")
                new.append(_parse_stmt(f"{ref}[...] = VALUE", VALUE=kept))
                self.renamed.pop(name)
        return new

    def _rename_targets(self, target: ast.expr) -> ast.expr:
        """A name in `outside` gets a fresh name from its first assignment here on."""
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and node.id in self.outside:
                if node.id not in self.renamed:
                    self.renamed[node.id] = self.gen.local(node.id)
                node.id = self.renamed[node.id]
        return target

    def _expr(self, expr: ast.expr) -> ast.expr:
        gen, plan, renamed, code = self.gen, self.plan, self.renamed, self

        scalars = {r.var for r in plan.loops} | set(plan.constants)

        class Rewrite(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.expr:
                operand = plan.packed.get((node.lineno, node.col_offset))
                if operand is None:
                    return self.generic_visit(node)  # type: ignore[return-value]
                shape = (*next(op for op in plan.operands if op.name == operand).tile.shape[:-2], -1)
                return _parse_expr(f"{gen.words[operand]}[...].reshape({shape})")

            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                packed = plan.packed.get((node.value.lineno, node.value.col_offset))
                if isinstance(node.value, ast.Name) and node.value.id not in renamed:
                    packed = plan.packed_names.get(node.value.id)  # (words bound to a name: still strided)
                if isinstance(node.value, ast.Call | ast.Name) and packed is not None:
                    rows = code._packed_rows(packed, node)
                    if rows is not None:  # one row of each pair-row group: a strided load
                        return rows
                operand = plan.reads.get(plan.read_key(node))
                if operand is not None:
                    return _parse_expr(f"{gen.views[operand.name]}[...]")
                base = node.value
                if isinstance(base, ast.Name) and base.id in gen.smem:  # an element of an index array
                    return ast.Subscript(ast.Name(gen.smem[base.id], ast.Load()), self.visit(node.slice), ast.Load())
                if (
                    isinstance(base, ast.Name)
                    and base.id in plan.carried
                    and base.id not in renamed
                    and reads(node.slice) <= scalars
                ):  # index the carried value's VMEM buffer itself: load only the part read
                    return ast.Subscript(ast.Name(gen.refs[base.id], ast.Load()), self.visit(node.slice), ast.Load())
                return self.generic_visit(node)  # type: ignore[return-value]

            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id in renamed:
                    return ast.Name(renamed[node.id], ast.Load())
                if node.id in plan.carried:
                    return _parse_expr(f"{gen.refs[node.id]}[...]")
                return node

        return mxu.rewrite_matmuls(Rewrite().visit(copy.deepcopy(expr)))

    def _packed_rows(self, operand: str, node: ast.Subscript) -> ast.expr | None:
        """`pairs(x[...])[..., g, :]` (every other index `:`): row g of each group of the words' rows, as a
        strided load from the VMEM buffer; None for other indexing (of the loaded words)."""
        gen = self.gen
        shape = next(op for op in self.plan.operands if op.name == operand).tile.shape
        lead, group, d = shape[:-3], shape[-3], shape[-1]  # words (..., group, d)
        dims = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        full = [isinstance(i, ast.Slice) and i.lower is None and i.upper is None and i.step is None for i in dims]
        if len(dims) != len(shape) - 1 or full[-2] or not all(full[:-2] + full[-1:]):
            return None
        n = math.prod(lead) if lead else 1
        g = ast.unparse(self._expr(dims[-2]))
        return _parse_expr(
            f"{_STRIDED}({gen.words[operand]}, {g}, {n}, {group}, {gen.n_interpret}).reshape({(*lead, d)})"
        )

    @staticmethod
    def _cast(
        value: ast.expr, have: jax.ShapeDtypeStruct, shape: tuple[int, ...], dtype: np.dtype, dtype_expr: str
    ) -> ast.expr:
        """`value`, of type `have`, converted to `shape` and `dtype` (`dtype_expr` in the kernel) as `.at[].set`
        converts what it stores and a carried value converts what each step assigns it."""
        if have.shape != shape:
            value = _parse_expr(f"jnp.broadcast_to(VALUE, {shape})", VALUE=value)
        if have.dtype != dtype:
            value = _parse_expr(f"VALUE.astype({dtype_expr})", VALUE=value)
        return value


class _Vectorize(ast.NodeTransformer):
    """A mask as a condition on arrays of loop values (evaluated for every step at once): Python's `and`,
    `or`, `not`, chained comparisons and `x if c else y` as their elementwise forms."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        fn = "jnp.logical_and" if isinstance(node.op, ast.And) else "jnp.logical_or"
        out = node.values[0]
        for value in node.values[1:]:
            out = _parse_expr(f"{fn}(A, B)", A=out, B=value)
        return out

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        return _parse_expr("jnp.logical_not(A)", A=node.operand) if isinstance(node.op, ast.Not) else node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        left, out = node.left, None
        for op, right in zip(node.ops, node.comparators, strict=True):
            pair = ast.Compare(left, [op], [right])
            out = pair if out is None else _parse_expr("jnp.logical_and(A, B)", A=out, B=pair)
            left = right
        return out  # type: ignore[return-value]

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        self.generic_visit(node)
        return _parse_expr("jnp.where(C, A, B)", C=node.test, A=node.body, B=node.orelse)


def _parse_expr(source: str, **holes: ast.expr) -> ast.expr:
    """An expression from source, with each Name in `holes` replaced by that expression."""
    tree = ast.parse(source, mode="eval").body
    return _fill(tree, holes)


def _parse_stmt(source: str, **holes: ast.expr) -> ast.stmt:
    tree = ast.parse(source).body[0]
    return _fill(tree, holes)


def _fill[T: ast.AST](tree: T, holes: dict[str, ast.expr]) -> T:
    class Fill(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return holes.get(node.id, node)

    return Fill().visit(tree) if holes else tree


class _Definitions:
    """How the generated source defines the values the tile code reads: literals, imports of modules and
    importable functions, or the source of functions defined where they can't be imported from (with the
    values they read, recursively)."""

    def __init__(self, values: dict[str, Any], nest: Nest) -> None:
        self.nest = nest
        self.imports: list[str] = []
        self.constants: list[str] = []
        self.functions: list[str] = []
        self.defined: dict[str, Any] = {}
        for name, value in values.items():
            self._define(name, value, name)

    def _error(self, message: str, via: str) -> KernelError:
        """An error about the value of `via`, a name the tile code reads, at the line that reads it."""
        node = next(n for stmt in self.nest.tile for n in ast.walk(stmt) if isinstance(n, ast.Name) and n.id == via)
        return self.nest.error(message, node)

    def _define(self, name: str, value: Any, via: str) -> None:
        if name in self.defined:
            if self.defined[name] is not value and self.defined[name] != value:
                raise self._error(f"kernel code reads two different values named `{name}`", via)
            return
        self.defined[name] = value
        if name in MODULES:
            if value is not MODULES[name]:
                raise self._error(f"the generated kernel uses `{name}` for {MODULES[name].__name__}; rename yours", via)
            return
        if name in mxu.FUNCTIONS or name == _WEAK:
            raise self._error(f"`{name}` is a name the generated kernel defines itself; rename yours", via)
        literal = _literal(value)
        if literal is not None:
            self.constants.append(f"{name} = {literal}")
        elif is_tunable(value):  # the schedule this kernel was built with: its fields, by name
            fields = ", ".join(f"{k}={_literal(getattr(value, k))}" for k in choices(value))
            self.imports.append("import types")
            self.constants.append(f"{name} = types.SimpleNamespace({fields})  # {type(value).__name__}")
        elif isinstance(value, types.ModuleType):
            module = value.__name__
            self.imports.append(f"import {module}" if module == name else f"import {module} as {name}")
        elif callable(value):
            self._define_function(name, value, via)
        else:
            what = "an array: pass it to the kernel as an input" if hasattr(value, "shape") else type(value).__name__
            raise self._error(
                f"kernel code reads `{name}`, which is {what}; kernel code can read numbers, strings, dtypes, "
                "tuples of them, modules and functions",
                via,
            )

    def _define_function(self, name: str, fn: Any, via: str) -> None:
        module, qualname = getattr(fn, "__module__", None), getattr(fn, "__qualname__", "")
        if module and module != "__main__" and "<" not in qualname and "." not in qualname:
            try:
                found = getattr(importlib.import_module(module), qualname, None)
            except ImportError:
                found = None
            if found is fn:
                self.imports.append(f"from {module} import {qualname}" + (f" as {name}" if name != qualname else ""))
                return
        if not isinstance(fn, types.FunctionType) or fn.__name__ == "<lambda>":
            raise self._error(f"kernel code calls `{name}`, which can't be imported or written into the kernel", via)
        source = textwrap.dedent(inspect.getsource(fn))
        fdef: ast.FunctionDef = ast.parse(source).body[0]  # type: ignore[assignment]
        if fdef.decorator_list:
            raise self._error(
                f"kernel code calls `{name}`, a decorated function: it can't be copied into the kernel", via
            )
        closure = inspect.getclosurevars(fn)
        read = {**closure.globals, **closure.nonlocals}
        # (getclosurevars misses the globals of the code inside: generator expressions, lambdas.)
        read |= {n: fn.__globals__[n] for n in _inner_names(fn.__code__) if n in fn.__globals__}
        for default in [*fdef.args.defaults, *(d for d in fdef.args.kw_defaults if d is not None)]:
            read |= {n: fn.__globals__[n] for n in reads(default) if n in fn.__globals__}  # evaluated at `def`
        for inner, value in read.items():
            self._define(inner, value, via)
        alias = f"\n\n\n{name} = {fn.__name__}" if fn.__name__ != name else ""
        self.functions.append(source.rstrip() + alias)


def _inner_names(code: types.CodeType) -> set[str]:
    """The global names the code objects nested in `code` (generator expressions, lambdas) read."""
    names: set[str] = set()
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= set(const.co_names) | _inner_names(const)
    return names


def _literal(value: Any) -> str | None:
    """Source for a value that has a literal form: numbers, strings, None, dtypes, scalar types and the
    functions of jnp, NumPy and jax.lax by their own names, tuples and lists of those; None for others."""
    if isinstance(value, np.generic):  # before Python's types: np.float64 is a float, but strongly typed
        return f"np.dtype({value.dtype.name!r}).type({_literal(value.item())})"
    if value is None or isinstance(value, bool | int | str | bytes | complex):
        return repr(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else f"float({str(value)!r})"
    if isinstance(value, np.dtype):
        return f"jnp.dtype({value.name!r})"
    name = getattr(value, "__name__", None)
    for module, alias in ((jnp, "jnp"), (np, "np"), (jax.lax, "jax.lax")):
        if isinstance(name, str) and getattr(module, name, None) is value:
            return f"{alias}.{name}"
    if isinstance(value, tuple | list):
        items = [_literal(v) for v in value]
        if None in items:
            return None
        body = ", ".join(items) + ("," if isinstance(value, tuple) and len(value) == 1 else "")  # type: ignore[arg-type]
        return f"({body})" if isinstance(value, tuple) else f"[{body}]"
    return None
