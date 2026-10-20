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
value by DMAs from a constant VMEM tile, a few at every tile, so they overlap the compute. An empty step (a
reduction loop whose mask skips all of a tile's blocks) fetches nothing and runs no body.

Index arrays (inputs read element by element) are SMEM inputs. When masks read them, the entry function
first finds the steps that run on the device (`<name>_steps`, plain JAX), and the kernel loops over those.

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
from .pipeline import TILE_SLOTS, Pipeline
from .plan import Kind, Operand, Plan, Range, Tile
from .schedule import CODE_SHIFT, FILL_TABLE, STEP_CODE, STEP_COUNT, STEP_TILE, TILE_TABLE, Schedule
from .tilecode import as_store, assigned, flatten, reads
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
        self.value_reads = _value_reads(plan)
        self.uses_weak = any(r.var in self.value_reads and r.count > 1 for r in plan.loops)
        bound = assigned(nest.tile) | {r.var for r in plan.loops}
        for name in sorted(bound & {*MODULES, *mxu.FUNCTIONS, _WEAK}):
            node = next(n for n in ast.walk(nest.parallel[0].node) if isinstance(n, ast.Name) and n.id == name)
            raise nest.error(f"the generated kernel uses the name `{name}` itself; rename your `{name}`", node)
        tile_names = {n for stmt in flatten(nest.tile) for n in reads(stmt)} | bound | set(plan.constants)

        self.indices = [index.name for index in plan.indices]
        self.arrays = [p for p in nest.params if p not in self.indices]  # the inputs in HBM
        self.definitions = _Definitions(plan.constants, nest)
        # (The steps function reads the module's names with the loop variables and index arrays as locals.)
        self.module = _Namer(
            {*self.definitions.defined, *MODULES, *mxu.FUNCTIONS, _WEAK, _PHASE_STEPS, *self.indices}
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
        self.entry_params = [entry(p) for p in nest.params]

        n = self.local = _Namer(tile_names | self.module.taken)
        self.table_params = {name: n(name) for name in [*schedule.tables, *self.runtime]}
        self.smem = {name: n(f"{name}_smem") for name in self.indices}
        self.hbm = {name: n(f"{name}_hbm") for name in (*self.arrays, *nest.outputs)}
        self.bufs = {
            op.name: n(f"{op.name}_buf" if op.kind is Kind.ONCE else f"{op.name}_bufs") for op in plan.operands
        }
        self.sems = {
            op.name: n(f"{op.name}_sem" if op.kind is Kind.ONCE else f"{op.name}_sems") for op in plan.operands
        }
        self.bufs |= {out.name: n(f"{out.name}_bufs") for out in plan.outputs}
        self.sems |= {out.name: n(f"{out.name}_sems") for out in plan.outputs}
        self.refs = {c: n(f"{c}_ref") for c in plan.carried}
        fills = plan.outputs if schedule.fills else ()
        self.fill_bufs = {out.name: n(f"{out.name}_fill") for out in fills}
        self.fill_sems = {out.name: n(f"{out.name}_fill_sem") for out in fills}
        self.fill_copies = {out.name: n(f"{out.name}_fill_copy") for out in fills}
        self.copy_fns = {name: n(f"{name}_copy") for name in [op.name for op in plan.operands] + list(nest.outputs)}
        self.views = {op.name: n(f"{op.name}_ref") for op in plan.operands}
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
        w("")
        w("import jax", "import jax.numpy as jnp", "import numpy as np")
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
            keep, code, at, tile = n("keep"), n("code"), n("at"), n("tile")
            w(f"{keep} = jnp.concatenate([{', '.join(k for k, _ in parts)}], axis=1).reshape(-1)")
            w(f"{code} = jnp.concatenate([{', '.join(c for _, c in parts)}], axis=1).reshape(-1)")
            w(f"{at} = jnp.where({keep}, jnp.cumsum({keep}, dtype=jnp.int32) - 1, {sch.pattern_steps})")
            w(f"{tile} = np.repeat(np.arange({tiles}, dtype=np.int32), {sch.pattern_steps // tiles})")
            with w.block("return ("):
                w(f"jnp.zeros({sch.pattern_steps}, jnp.int32).at[{at}].set({tile}, mode='drop'),")
                w(f"jnp.zeros({sch.pattern_steps}, jnp.int32).at[{at}].set({code}, mode='drop'),")
                w(f"jnp.sum({keep}, dtype=jnp.int32).reshape(1),")
            w(")")

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
        w(f"{keep}, {code} = {_PHASE_STEPS}({run}, {p}, {len(plan.phases)}, {iteration}, {sch.iterations})")
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
            param = dict(zip(self.nest.params, self.entry_params, strict=True))
            runtime = [self.entry_namer(name) for name in self.runtime]
            if runtime:
                w(f"{', '.join(runtime)} = {self.steps_name}({', '.join(param[i] for i in self.indices)})")
            with w.block("return pl.pallas_call("):
                w(f"{self.kernel_name},")
                shapes = [f"jax.ShapeDtypeStruct({out.shape}, jnp.{out.dtype.name})" for out in plan.outputs]
                w(f"out_shape={shapes[0] if len(shapes) == 1 else '(' + ', '.join(shapes) + ',)'},")
                with w.block("in_specs=["):
                    smem = len(self.tables) + len(runtime) + len(self.indices)
                    w(*["pl.BlockSpec(memory_space=pltpu.SMEM),"] * smem)
                    w(*["pl.BlockSpec(memory_space=pl.ANY),"] * len(self.arrays))
                w("],")
                specs = ["pl.BlockSpec(memory_space=pl.ANY)"] * len(plan.outputs)
                w(f"out_specs={specs[0] if len(specs) == 1 else '(' + ', '.join(specs) + ',)'},")
                with w.block("scratch_shapes=["):
                    w(*self._scratch_shapes())
                w("],")
                w(f"compiler_params=pltpu.CompilerParams(vmem_limit_bytes={v5e.VMEM_LIMIT}),")
                w("interpret=interpret,")
                w(f"name={self.nest.name!r},")
            inputs = [*self.tables.values(), *runtime, *(param[i] for i in self.indices)]
            inputs += [param[a] for a in self.arrays]
            w(f")({', '.join(inputs)})")
        return w.text()

    def _scratch_shapes(self) -> list[str]:
        lines = []
        for op in self.plan.operands:
            slots = self.pipeline.slots(op.kind)
            shape = op.tile.shape if op.kind is Kind.ONCE else (slots, *op.tile.shape)
            sems = "()" if op.kind is Kind.ONCE else f"({slots},)"
            lines += [
                f"pltpu.VMEM({shape}, jnp.{op.tile.dtype.name}),  # {self.bufs[op.name]}: {op.tile.text}",
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
        ]
        for op in plan.operands:
            params += [self.bufs[op.name], self.sems[op.name]]
        for out in plan.outputs:
            params += [self.bufs[out.name], self.sems[out.name]]
        params += list(self.refs.values())
        for out in self.fill_bufs:
            params += [self.fill_bufs[out], self.fill_sems[out]]
        params += [self.tile_slots] if self.shared else []
        with w.block(f"def {self.kernel_name}({', '.join(params)}):"):
            self._step_fn(w)
            self._blocks_fn(w)
            for op in plan.operands:
                self._operand_copy_fn(w, op)
            for op in self.shared:
                self._moves_fn(w, op)
            for out in plan.outputs:
                self._output_copy_fn(w, out.name, out.tile)
            if self.schedule.fills:
                self._fill_copy_fns(w)
            self._setup(w)
            self._loop(w)
            self._drain(w)
        return w.text()

    def _step_fn(self, w: _Writer) -> None:
        sch, s, tile, ph, kb = self.schedule, self.n_s, self.n_tile, self.n_ph, self.n_kb
        if self.single:
            return  # a step is a tile
        phases = len(sch.phase_counts)
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
                    f"return {tile}, {_int_expr(block, '%', phases)}, {iteration}{_int_expr(block, '//', phases)}, "
                    f"({code} & 2) != 0, ({code} & 1) != 0{empty}"
                )

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

    def _window_index(self, tile: Tile, block_of: dict[str, str]) -> str:
        """The index of `tile`'s window into its HBM array, given each loop's block expression."""
        dims = []
        for window, dim in zip(tile.windows, self._array_shape(tile.array), strict=True):
            r = self._range(window.loop) if window.loop else None
            if window.index is not None:
                dims.append(self._runtime_index(window.index, block_of, dim))
            elif r is None:
                if window.size is None:
                    dims.append(str(window.offset))
                elif window.offset == 0 and window.size == dim:
                    dims.append(":")
                else:
                    dims.append(f"pl.ds({window.offset}, {window.size})")
            else:
                start = _scaled(block_of[r.var], r)
                dims.append(start if window.size is None else f"pl.ds({start}, {window.size})")
        return ", ".join(dims)

    def _runtime_index(self, element: ast.Subscript, block_of: dict[str, str], dim: int) -> str:
        """An index read from an index array when the kernel runs, at the loops' values for these blocks,
        clamped to the dim (as JAX clamps indices)."""
        gen = self

        class Values(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id in block_of:
                    return _parse_expr(f"({_scaled(block_of[node.id], gen._range(node.id))})")
                return ast.Name(gen.smem[node.id], ast.Load()) if node.id in gen.smem else node

        return f"jnp.clip({ast.unparse(Values().visit(copy.deepcopy(element)))}, 0, {dim - 1})"

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
        args = {Kind.STREAM: f"{at}, {rb}{kb}, {slot}", Kind.TILE: f"{at}, {slot}", Kind.ONCE: ""}[op.kind]
        with w.block(f"def {self.copy_fns[op.name]}({args}):"):
            w(f"# {op.tile.text}")
            block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
            if op.kind is not Kind.ONCE:
                self._unpack(w, at)
            for r in self.plan.reductions:  # only the operand's own phase's loop moves it
                block_of[r.var] = kb if op.phase is not None and self.plan.phases[op.phase].range is r else "0"
            for r in self.plan.repeat_loops:  # (repeat loops move only streamed operands)
                block_of[r.var] = self.n_rb if op.kind is Kind.STREAM and self.repeated else "0"
            src = f"{self.hbm[op.tile.array]}.at[{self._window_index(op.tile, block_of)}]"
            dst = (buf, sem) if op.kind is Kind.ONCE else (f"{buf}.at[{slot}]", f"{sem}.at[{slot}]")
            w(f"return pltpu.make_async_copy({src}, {dst[0]}, {dst[1]})")

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
        for window, dim in zip(tile.windows, self._array_shape(tile.array), strict=True):
            if window.index is not None:
                starts.append(self._runtime_index(window.index, block_of, dim))
            elif window.loop is not None and self._range(window.loop).count > 1:
                starts.append(_scaled(block_of[window.loop], self._range(window.loop)))
        return starts

    def _output_copy_fn(self, w: _Writer, name: str, tile: Tile) -> None:
        at, slot = self.n_at, self.n_slot
        with w.block(f"def {self.copy_fns[name]}({at}, {slot}):"):
            w(f"# {tile.text}")
            self._unpack(w, at)
            block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
            dst = f"{self.hbm[name]}.at[{self._window_index(tile, block_of)}]"
            w(f"return pltpu.make_async_copy({self.bufs[name]}.at[{slot}], {dst}, {self.sems[name]}.at[{slot}])")

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
        for out in self.plan.outputs:
            with w.block(f"def {self.fill_copies[out.name]}({self.n_at}):"):
                self._unpack(w, self.n_at)
                block_of = {r.var: self.blk[r.var] for r in self.plan.parallel}
                dst = f"{self.hbm[out.name]}.at[{self._window_index(out.tile, block_of)}]"
                w(f"return pltpu.make_async_copy({self.fill_bufs[out.name]}, {dst}, {self.fill_sems[out.name]})")

    # ------------------------------------------------------------------ before, during and after the loop

    def _setup(self, w: _Writer) -> None:
        plan, sch = self.plan, self.schedule
        w("")
        if sch.runtime:
            count = f"{self.table_params[STEP_COUNT]}[0]"
            w(f"{self.n_steps} = {count}" if sch.repeats == 1 else f"{self.n_steps} = {sch.repeats} * {count}")
        if sch.fills:
            w("# The fill value of the tiles the masks skip.")
            for out in plan.outputs:
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
            at0 = sch.blocks_at(0)
            w(*[f"{self.copy_fns[op.name]}({at0}, 0).start()" for op in per_tile])
            # (A shared window's slot flips when it moves: tile 0's moves into slot 0.)
            w(*[f"{self.tile_slots}[{k}] = 1" for k in range(len(self.shared))])
        streamed = [op for op in plan.operands if op.kind is Kind.STREAM]
        ahead = min(self.pipeline.input_buffers - 1, sch.steps)
        if not (streamed and ahead):
            return
        w(f"# The first {ahead} step(s)' inputs.")
        if sch.runtime:  # the steps are known only now
            names = ("first_tile", "first_ph", "first_rb", "first_kb", "first_empty")
            tile, ph, rb, kb, empty = (self.local(name) for name in names)
            for step in range(ahead):
                with self._when(w, f"{step} < {self.n_steps}", f"_first_inputs{step}"):
                    w(f"{self._targets(tile, ph, rb, kb, '_', '_', empty)} = {self.n_step}({step})")
                    for phase, ops in self._by_phase(streamed).items():
                        with self._when(w, self._runs(ph, phase, empty), f"_first_phase{phase}"):
                            for op in ops:
                                at = f"{self.n_blocks}({tile})"
                                w(f"{self.copy_fns[op.name]}({at}, {self._rb(rb)}{kb}, {step}).start()")
            return
        for step in range(ahead):
            tile, phase, kb = sch.step_at(step)
            if sch.empty_at(step):
                continue
            at = sch.blocks_at(tile)
            for op in streamed:
                if op.phase in (None, phase):
                    w(f"{self.copy_fns[op.name]}({at}, {self._rb(str(sch.iteration_at(step)))}{kb}, {step}).start()")

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
        return self.schedule.empty and phase is not None and self.plan.phases[phase].range.mask is not None

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
        steps = self.n_steps if sch.runtime else str(sch.steps)
        if self.single:  # a step is a tile: it is every phase's first and last step
            tile_first = tile_last = None
        else:
            tile_first = self._ends(0, ph, first, last=False)
            tile_last = self._ends(phases - 1, ph, last, last=True)
        w("")
        w(f"@pl.loop(0, {steps})")
        with w.block(f"def {self.local('_steps')}({s}):"):
            if self.single:
                w(f"{tile}, {kb} = {s}, 0")
            else:
                w(f"{self._targets(tile, ph, self.n_rb, kb, first, last, empty)} = {self.n_step}({s})")
            w(f"{at} = {self.n_blocks}({tile})")

            streamed = [op for op in plan.operands if op.kind is Kind.STREAM]
            depth = pipe.input_buffers
            if streamed:
                with self._when(w, f"{s} + {depth - 1} < {steps}", "_fetch_ahead"):
                    ahead_tile, ahead_ph = self.local("ahead_tile"), self.local("ahead_ph")
                    ahead_kb, ahead_empty = self.local("ahead_kb"), self.local("ahead_empty")
                    ahead_rb = self.local("ahead_rb")
                    if self.single:
                        w(f"{ahead_tile}, {ahead_kb} = {s} + {depth - 1}, 0")
                    else:
                        targets = self._targets(ahead_tile, ahead_ph, ahead_rb, ahead_kb, "_", "_", ahead_empty)
                        w(f"{targets} = {self.n_step}({s} + {depth - 1})")
                    ahead_slot = _int_expr(f"{s} + {depth - 1}", "%", depth)
                    for phase, ops in self._by_phase(streamed).items():
                        with self._when(w, self._runs(ahead_ph, phase, ahead_empty), f"_fetch_phase{phase}"):
                            for op in ops:
                                blocks = f"{self.n_blocks}({ahead_tile})"
                                rb = self._rb(ahead_rb)
                                w(f"{self.copy_fns[op.name]}({blocks}, {rb}{ahead_kb}, {ahead_slot}).start()")

            per_tile = [op for op in plan.operands if op.kind is Kind.TILE]
            slots = {Kind.STREAM: _int_expr(s, "%", depth), Kind.TILE: _int_expr(tile, "%", TILE_SLOTS)}
            with self._region(w, WAIT) if per_tile or streamed else nullcontext():  # (no region to leave empty)
                if per_tile:
                    with self._when(w, tile_first, "_tile_inputs"):
                        self._tile_inputs(w, per_tile, slots[Kind.TILE])
                for phase, ops in self._by_phase(streamed).items():
                    with self._when(w, self._runs(ph, phase, empty), f"_wait_phase{phase}"):
                        for op in ops:
                            w(f"{self.copy_fns[op.name]}({at}, {self._rb(self.n_rb)}{kb}, {slots[Kind.STREAM]}).wait()")
            for op in plan.operands:
                buf = self.bufs[op.name]
                slot = f"{self.tile_slots}[{self.shared.index(op)}]" if op in self.shared else slots.get(op.kind)
                view = buf if op.kind is Kind.ONCE else f"{buf}.at[{slot}]"
                w(f"{self.views[op.name]} = {view}")
            self._loop_vars(w)

            out_slots = pipe.output_buffers
            with self._region(w, OUT_WAIT):
                cond = f"{tile} >= {out_slots}" if tile_last is None else f"{tile_last} & ({tile} >= {out_slots})"
                with self._when(w, cond, "_free_output_slot"):
                    zero = (0,) * len(plan.parallel)
                    for out in plan.outputs:
                        w(f"{self.copy_fns[out.name]}({zero}, {_int_expr(tile, '%', out_slots)}).wait()")

            if sch.fills:
                with self._when(w, tile_first, "_fill_skipped_tiles"):
                    self._start_fills(w)

            with self._region(w, COMPUTE):
                self._tile_code(w, tile_first)

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
        with self._when(w, f"{tile} + 1 < {self.schedule.tiles}", "_next_tile_inputs"):
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

    def _tile_code(self, w: _Writer, tile_first: str | None) -> None:
        """The segments and the phase bodies: the code before the phases at a tile's first step; each
        phase's body at its steps (not at an empty one), with the code after it at its last step (where the
        body's values are still in scope)."""
        plan, code = self.plan, _TileCode(self)
        if self.single:
            w(*code.emit(plan.segments[0], "start", set()))
            return
        if plan.segments[0]:
            with self._when(w, tile_first, "_start_tile"):
                w(*code.emit(plan.segments[0], "start", set()))
        carried = set(plan.carried)
        for p, phase in enumerate(plan.phases):
            with self._when(w, self._in_phase(self.n_ph, p), f"_phase{p}"):
                if phase.before:  # a repeat loop's code before its first phase, every iteration
                    with self._when(w, self.n_first, f"_before_phase{p}"):
                        w(*code.emit(phase.before, "between", set()))
                empty = f"jnp.logical_not({self.n_empty})" if self._may_be_empty(p) else None
                with self._when(w, empty, f"_body{p}"):
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
                with self._when(w, self.n_last, name):
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
        if any(r.var in used for r in plan.parallel):
            w(f"{_tuple(names)} = {self.n_at}")
        steps = [phase.range for phase in plan.phases]
        for r in plan.repeat_loops:  # a repeat loop of one iteration: its variable is a constant
            if r.count == 1 and r.var in used:
                w(f"{r.var} = {r.start}")
        for r in (*plan.parallel, *steps, *(rep.range for rep in plan.repeats)):
            if r.var not in used:
                continue
            block = self.blk[r.var] if r in plan.parallel else self.n_kb if r in steps else self.n_rb
            value = _scaled(block, r)
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
                w(*[f"{self.fill_copies[out.name]}({zero}).wait()" for out in self.plan.outputs])
            w(f"{self.n_at} = {self.n_fill_blocks}({c})")
            w(*[f"{self.fill_copies[out.name]}({self.n_at}).start()" for out in self.plan.outputs])

    def _drain(self, w: _Writer) -> None:
        sch, pipe = self.schedule, self.pipeline
        w("")
        w("# Wait for the copies still in flight.")
        zero = (0,) * len(self.plan.parallel)
        for tile in range(max(0, sch.tiles - pipe.output_buffers), sch.tiles):
            for out in self.plan.outputs:
                w(f"{self.copy_fns[out.name]}({zero}, {tile % pipe.output_buffers}).wait()")
        for _ in range(min(sch.fills, FILLS_IN_FLIGHT)):
            w(*[f"{self.fill_copies[out.name]}({zero}).wait()" for out in self.plan.outputs])


def _may_stay(op: Operand, plan: Plan) -> bool:
    """May a per-tile window be the same for consecutive tiles? If an index array places it, or a parallel
    loop of more than one block doesn't move it."""
    moved_by_all = all(r.var in op.tile.loops for r in plan.parallel if r.count > 1)
    return any(w.index is not None for w in op.tile.windows) or not moved_by_all


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
        gen, plan = self.gen, self.plan
        store = plan.nest.store(stmt)
        if store is not None:
            out = next(o for o in plan.outputs if o.name == store.output)
            slot = _int_expr(gen.n_tile, "%", gen.pipeline.output_buffers)
            buf = gen.bufs[out.name]
            value = self._cast(self._expr(store.value), out.value, out.tile.shape, out.dtype, f"{buf}.dtype")
            return [
                _parse_stmt(f"{buf}[{slot}] = VALUE", VALUE=value),
                _parse_stmt(f"{gen.copy_fns[out.name]}({gen.n_at}, {slot}).start()"),
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
            and update.output not in self.renamed
            and set().union(*(reads(i) for i in update.index)) <= scalars
        ):  # `a = a.at[index].set(v)` of a carried value in its VMEM buffer: store just that part
            ref, index = gen.refs[update.output], ", ".join(ast.unparse(self._expr(i)) for i in update.index)
            value = _parse_expr(
                f"jnp.broadcast_to(VALUE, {ref}.at[{index}].shape).astype({ref}.dtype)", VALUE=self._expr(update.value)
            )
            return [_parse_stmt(f"{ref}[{index}] = VALUE", VALUE=value)]
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
        gen, plan, renamed = self.gen, self.plan, self.renamed

        scalars = {r.var for r in plan.loops} | set(plan.constants)

        class Rewrite(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
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
        for default in [*fdef.args.defaults, *(d for d in fdef.args.kw_defaults if d is not None)]:
            read |= {n: fn.__globals__[n] for n in reads(default) if n in fn.__globals__}  # evaluated at `def`
        for inner, value in read.items():
            self._define(inner, value, via)
        alias = f"\n\n\n{name} = {fn.__name__}" if fn.__name__ != name else ""
        self.functions.append(source.rstrip() + alias)


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
