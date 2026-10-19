"""Generate a gridless Pallas TPU kernel from a `Specialization` and its `Schedule`, then compile it.

The kernel runs every step of its share of the work in one `pl.loop` and moves data with hand-written
async copies. One step:

1. decode the step: tile ordinal, phase, reduction position, first/last/empty flags, and from those the
   block index of every tile loop (computed once, shared by every copy of the step);
2. start copying the inputs of the step `buffers - 1` ahead into a ring of VMEM slots;
3. at a tile's first step: start fetching the next tile's per-tile inputs, wait for this tile's;
4. wait for this step's inputs;
5. run the user's units: a phase's leading segment on its first step, its reduction body on non-empty
   steps, and the final segment plus output stores on the tile's last step;
6. stores write each output tile into one of two VMEM slots and copy it to HBM asynchronously.

Kernels with one unmasked reduction (or none) have no phase or empty-step tests at all: every step runs
the body. On chips with two TensorCores per device the kernel gets a one-axis "parallel" grid, one
point per TensorCore, and each core runs a contiguous run of tiles.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import hardware, mxu
from . import schedule as schedule_lib
from .analysis import Block, LoopBounds, Operand, Specialization, StateVar, Unit
from .errors import AutoKernelError
from .frontend import assigned_name
from .schedule import COLUMN_SHIFT, FIRST, LAST, RUNTIME_TABLES, Schedule

__all__ = [
    "FILL_DMA_DEPTH",
    "Emitter",
    "Generated",
    "check_layout",
    "generate",
    "layout_error",
    "plan_fills",
    "prelude",
]

# Constant (masked-out) tiles are written with at most this many fill DMAs in flight.
FILL_DMA_DEPTH = 4

_PRELUDE = """\
def _div(a, b):
    # a // b for the non-negative step and tile indices: one integer division, no floor-sign fix-ups.
    return jax.lax.div(a, b)


def _rem(a, b):
    return jax.lax.rem(a, b)
"""

_SCALAR_PRELUDE = """\
def _index(i, n):
    # A runtime index into a dim of size n: Python's rules, so a negative index counts from the end. One
    # outside the dim is clamped into it, so a copy never reads outside the array (that would halt the TPU
    # core); the caller reports it (`analysis.check_indices`).
    return jnp.minimum(jnp.maximum(jnp.where(i < 0, i + n, i), 0), n - 1)
"""


_WEAK = """\
def {name}(x):
    # A Python scalar's weak type, so it takes the other operand's type: `bf16_tile * i` stays bf16.
    return jax._src.lax.lax._convert_element_type(x, weak_type=True)
"""


@dataclass(frozen=True)
class Generated:
    """Generated source and the callable its wrapper defines."""

    source: str
    entry: Callable[..., Any]  # entry(*inputs, interpret=False) -> output(s)


def generate(
    spec: Specialization,
    sched: Schedule,
    *,
    buffers: int,
    matmul_precision: str,
    fills: Sequence[str],
    aliased: Sequence[str],
    vmem_limit_bytes: int,
    cost: pl.CostEstimate | None,
) -> Generated:
    """`fills` and `aliased` are the outputs `plan_fills` assigned to fill DMAs and to an aliased input."""
    return Emitter(spec, sched, buffers, matmul_precision, fills, aliased, vmem_limit_bytes, cost).emit()


def plan_fills(spec: Specialization, background_fill: bool) -> tuple[list[str], list[str]]:
    """How skipped tiles of each output get their value: (outputs written by fill DMAs, outputs aliased).

    An output with skipped tiles whose allocation is a known constant can have those tiles written by
    DMAs from the kernel (overlapping the compute); otherwise the allocation is passed in and aliased to
    the output so the skipped tiles keep its values.
    """
    skipped = bool(spec.constant_tiles) or spec.tile_mask_runtime
    if not skipped:  # outputs the tiles don't cover keep their allocation's values
        return [], [o.var for o in spec.outputs if o.partial]
    fill_ok = background_fill and not spec.tile_mask_runtime
    fills = [o.var for o in spec.outputs if fill_ok and o.fill is not None]
    return fills, [o.var for o in spec.outputs if o.var not in fills]


def check_layout(spec: Specialization, lanes: int) -> list[str]:
    """TPU tiling problems of every input and output tile."""
    problems = []
    for op in spec.operands:
        shape = op.block.kernel_shape
        full = op.block.kept(op.full_shape)
        problems += hardware.check_block(
            f"input `{op.param}[{op.block.text}]` tile {shape}", shape, full, op.dtype, lanes
        )
    for out in spec.outputs:
        shape = out.block.kernel_shape
        full = out.block.kept(tuple(out.out.shape))
        what = f"output `{out.var}[{out.block.text}]` tile {shape}"
        problems += hardware.check_block(what, shape, full, out.out.dtype, lanes)
    return problems


def layout_error(spec: Specialization, problems: list[str]) -> AutoKernelError:
    return AutoKernelError(
        f"{spec.nest.fn_name}: tiles don't satisfy TPU (Mosaic) layout rules:\n  - "
        + "\n  - ".join(problems)
        + "\nPick tile sizes that fit, or pass check_tiling=False to try anyway."
    )


# --------------------------------------------------------------------------- source helpers


def _dtype_src(dtype: Any) -> str:
    return f"jnp.{np.dtype(dtype).name}"


def _stored_dtype(sv: StateVar) -> np.dtype:
    """The dtype a state variable's scratch holds: its own, except that SMEM scalars take a 32-bit word."""
    dtype = np.dtype(sv.aval.dtype)
    if not sv.smem or dtype.itemsize >= 4:
        return dtype
    return np.dtype(np.float32 if jnp.issubdtype(dtype, jnp.floating) else np.int32)


def _scalar_src(value: Any) -> str:
    return repr(value) if np.isfinite(value) else f"float('{value}')"


def _indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line if line else line for line in lines]


def _fn(name: str, args: str, body: list[str], when: str | None = None, loop: str | None = None) -> list[str]:
    """A nested function, optionally under `@pl.when(when)` or `@pl.loop(loop)`."""
    decorator = [f"@pl.when({when})"] if when else [f"@pl.loop({loop})"] if loop else []
    return [*decorator, f"def {name}({args}):", *_indent(body or ["pass"])]


def _block(name: str, body: list[str], when: str | None) -> list[str]:
    """`body` as a function under `@pl.when(when)`, or inline when there's nothing to test."""
    return _fn(name, "", body, when=when) if when else list(body)


def _by_phase(values: Sequence[Any]) -> str:
    """Pick `values[ph]` with a chain of selects (Mosaic can't lower jnp.select's argmax over booleans)."""
    if len(set(values)) == 1:
        return str(values[0])
    out = str(values[-1])
    for p in range(len(values) - 2, -1, -1):
        out = f"jnp.where(ph == {p}, {values[p]}, {out})"
    return out


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _div_src(x: str, n: int) -> str:
    """`x // n` for a non-negative x: a shift for powers of two, one integer division otherwise."""
    if n == 1:
        return x
    return f"({x} >> {n.bit_length() - 1})" if _is_pow2(n) else f"_div({x}, {n})"


def _rem_src(x: str, n: int) -> str:
    """`x % n` for a non-negative x: a mask for powers of two, one integer remainder otherwise."""
    return f"({x} & {n - 1})" if _is_pow2(n) else f"_rem({x}, {n})"


def _split_src(name: str, x: str, quotient: str, n: int) -> str:
    """`name = x % n` given `quotient = x // n`: a mask for powers of two, else one multiply and subtract."""
    if n == 1:
        return f"{name} = 0"
    return f"{name} = {_rem_src(x, n)}" if _is_pow2(n) else f"{name} = {x} - {quotient} * {n}"


def _radix(index: str, loops: Sequence[LoopBounds], k: int) -> str:
    """Absolute block index of `loops[k]` when `index` counts the loops' blocks row-major."""
    stride = int(np.prod([lb.count for lb in loops[k + 1 :]], dtype=np.int64))
    idx = _div_src(index, stride)
    if k:
        idx = _rem_src(idx, loops[k].count)
    base = loops[k].start // loops[k].step
    return f"({idx} + {base})" if base else idx


class Emitter:
    """Generates one nest's kernel: as a whole `pallas_call` (`emit`), or as one part of a fused kernel
    (`part` and `prepare`, used by `fusion`), where `dependent` names the inputs an earlier part writes."""

    def __init__(self, spec, sched, buffers, precision, fills, aliased, vmem_limit, cost, dependent=frozenset()):
        self.spec, self.sched, self.nest = spec, sched, spec.nest
        self.buffers, self.precision, self.vmem_limit, self.cost = buffers, precision, vmem_limit, cost
        self.dependent = frozenset(dependent)
        self.names = spec.names.copy()  # emitting twice gives the same names
        self.P = len(spec.phases)
        self.cores = sched.cores
        # One unmasked reduction (or none): every step belongs to phase 0 and none is empty.
        self.trivial = sched.mode == "uniform" and self.P <= 1
        self.used_names: set[str] = set()  # names the rewritten user code reads
        self.weak: str | None = None  # the weak-type helper's name, once something needs it
        self._param_names: dict[str, str] | None = None
        self._tables_local: str | None = None
        self._names(fills, aliased)
        self._tables()

    def _names(self, fills: Sequence[str], aliased: Sequence[str]) -> None:
        """Identifiers the generated code binds where the user's code can see them."""
        spec, names = self.spec, self.names
        self.fills = {var: names.fresh(f"{var}_fill") for var in fills}
        self.aliased = {var: names.fresh(f"{var}_init") for var in aliased}
        self.v = {
            k: names.fresh(k)
            for k in (
                "s",
                "o",
                "ph",
                "d",
                "first",
                "last",
                "empty",
                "blocks",
                "core",
                "s_begin",
                "s_end",
                "o_begin",
                "o_end",
            )
        }
        self.blk = {var: names.fresh(f"{var}_blk") for var in spec.loops}
        self.streamed = [op for op in spec.operands if op.phase is not None]
        self.per_tile = [op for op in spec.operands if op.phase is None]
        self.hbm: dict[str, str] = {}
        for op in spec.operands:
            self.hbm.setdefault(op.param, names.fresh(f"{op.param}_hbm"))
        self.slots = {op.ref: names.fresh(f"{op.ref}_slots") for op in spec.operands}
        self.sems = {op.ref: names.fresh(f"{op.ref}_sems") for op in spec.operands}
        self.out_hbm = {o.var: names.fresh(f"{o.var}_hbm") for o in spec.outputs}
        self.out_slots = {o.var: names.fresh(f"{o.ref}_slots") for o in spec.outputs}
        self.out_sems = {o.var: names.fresh(f"{o.ref}_sems") for o in spec.outputs}
        # Runtime scalars live in SMEM: integers in one int32 array, floats in one f32 array.
        avals = dict(zip(spec.nest.params, spec.specs, strict=True))
        self.int_scalars = [p for p in spec.scalars if jnp.issubdtype(avals[p].dtype, jnp.integer)]
        self.float_scalars = [(p, bool(avals[p].weak_type)) for p in spec.scalars if p not in self.int_scalars]
        self.scalars_ref = names.fresh("scalars") if self.int_scalars else None
        self.fscalars_ref = names.fresh("fscalars") if self.float_scalars else None
        self.indices: dict[tuple[str, int], str] = {}  # (runtime index expression, dim size) -> its variable
        for op in spec.operands:
            for d, expr in enumerate(op.block.dynamic):
                if expr is not None and (expr, op.full_shape[d]) not in self.indices:
                    self.indices[(expr, op.full_shape[d])] = names.fresh(f"{op.param}_index{d}")
        # A fused kernel's part: its pieces, run in turn by the fused kernel.
        self.pieces = {
            k: names.fresh(f"_{k}") for k in ("early", "with_scratch", "late", "run", "drain", "hook", "hook_at")
        }

    def _tables(self) -> None:
        """Static SMEM tables (embedded as constants of the wrapper) and the runtime tables' names."""
        spec, sched, names = self.spec, self.sched, self.names
        self.tables: dict[str, np.ndarray] = {}  # kernel parameter -> contents
        self.table_consts: dict[str, str] = {}  # kernel parameter -> the wrapper's constant

        def table(base: str, value: np.ndarray) -> str:
            name = names.fresh(base)
            self.tables[name] = np.asarray(value, np.int32)
            self.table_consts[name] = names.fresh(name.upper())
            return name

        self.tile_tables: dict[str, str] = {}
        if spec.nest.guard is not None and not spec.tile_mask_runtime:  # a static tile mask: list the tiles
            for n, lb in enumerate(spec.parallel):
                self.tile_tables[lb.var] = table(f"{lb.var}_blocks", [t[n] for t in spec.tiles])
        if self.fills:
            self.fill_tables = {
                lb.var: table(f"{lb.var}_fill_blocks", [t[n] for t in spec.constant_tiles])
                for n, lb in enumerate(spec.parallel)
            }
            n_tiles, n_fill = len(spec.tiles), len(spec.constant_tiles)
            # Tile ordinal o writes the fills [fill_start[o], fill_start[o + 1]).
            self.fill_start = table("fill_start", [t * n_fill // n_tiles for t in range(n_tiles + 1)])
        self.step_tables: dict[str, str] = {}
        if sched.mode == "table":
            self.step_tables = {key: table(key, value) for key, value in sched.tables.items()}
        elif sched.mode == "runtime":
            self.step_tables = {k: names.fresh(k) for k in RUNTIME_TABLES}
        self.bounds_table = None
        if sched.mode != "runtime" and self.cores > 1:  # [step bound per core..., tile bound per core...]
            steps = [b for b, _ in sched.core_steps] + [sched.core_steps[-1][1]]
            tiles = [b for b, _ in sched.core_tiles] + [sched.core_tiles[-1][1]]
            self.bounds_table = table("core_bounds", steps + tiles)

    # ------------------------------------------------------------------ coordinates

    def _tile_block(self, var: str, o: str) -> str:
        """Absolute block index of parallel loop `var` for tile ordinal `o`."""
        if var in self.tile_tables:
            return f"{self.tile_tables[var]}[{o}]"
        t = f"{self.step_tables['tile_ids']}[{o}]" if self.spec.tile_mask_runtime else o
        return _radix(t, self.spec.parallel, [lb.var for lb in self.spec.parallel].index(var))

    def _reduction_block(self, var: str, d: str) -> str:
        """Absolute block index of reduction loop `var` at flattened phase position `d`."""
        phase = next(p for p in self.spec.phases if any(lb.var == var for lb in p.loops))
        return _radix(d, phase.loops, [lb.var for lb in phase.loops].index(var))

    def _blocks(self) -> list[str]:
        """`_blocks(o, d)`: the block index of every tile loop, for the copies and the loop values."""
        body = []
        for lb in self.spec.parallel:
            body.append(f"{self.blk[lb.var]} = {self._tile_block(lb.var, 'o')}")
        for phase in self.spec.phases:
            for lb in phase.loops:
                body.append(f"{self.blk[lb.var]} = {self._reduction_block(lb.var, 'd')}")
        body.append(f"return {self._blocks_tuple(self.blk)}")
        return _fn("_blocks", "o, d", body)

    def _blocks_tuple(self, names: dict[str, str]) -> str:
        items = [names.get(var, "_") for var in self.spec.loops]
        return f"({items[0]},)" if len(items) == 1 else ", ".join(items)

    def _slices(self, block: Block, blocks: dict[str, str], fill: str | None = None, full: tuple[int, ...] = ()) -> str:
        """The HBM window of a block: `pl.ds` per tiled dim, a plain index per squeezed dim."""
        parts = []
        for d, (var, size, c, sq) in enumerate(zip(block.loops, block.shape, block.const, block.squeezed, strict=True)):
            expr = block.dynamic_index(d)
            if expr is not None:  # a runtime index, read from SMEM at the start of the kernel
                start = self.indices[(expr, full[d])]
            elif var is None:
                start = str(c * size)
            elif self.spec.loops[var].count == 1:
                # A loop of one block: a constant offset, so Mosaic sees e.g. a window spanning whole rows and
                # copies it in one go (a traced 0 offset cost a single-step v5e matmul 2%).
                lb = self.spec.loops[var]
                start = str(lb.start // lb.step * size)
            else:
                idx = f"{self.fill_tables[var]}[{fill}]" if fill is not None else blocks[var]
                start = idx if size == 1 else f"{idx} * {size}"
            parts.append(start if sq else f"pl.ds({start}, {size})")
        return ", ".join(parts)

    def _unpack_blocks(self, block: Block) -> list[str]:
        """Bind the block indices a copy needs from the `blocks` tuple."""
        wanted = {var: self.blk[var] for var in block.loops if var}
        return [f"{self._blocks_tuple(wanted)} = blocks"] if wanted else []

    def _loop_values(self) -> list[str]:
        """The user's loop variables, for code that reads them as values."""
        lines = []
        for var in sorted(self.used_names & set(self.spec.loops)):
            step = self.spec.loops[var].step
            value = f"{self.blk[var]} * {step}" if step != 1 else self.blk[var]
            lines.append(f"{var} = {self._weak()}({value})")
        return lines

    def _weak(self) -> str:
        """The prelude helper giving a value a Python scalar's weak type (`bf16_tile * i` stays bf16)."""
        if self.weak is None:
            self.weak = self.names.fresh("_weak")
        return self.weak

    # ------------------------------------------------------------------ step decoding

    def _decode(self) -> list[str]:
        """`_decode(s)` -> (tile ordinal, phase, reduction position, first, last, empty)."""
        sched = self.sched
        if sched.mode == "uniform":
            per = sched.steps_per_tile
            body = [f"o = {_div_src('s', per)}", _split_src("d", "s", "o", per)]
            if self.P <= 1:
                body += ["first = d == 0", f"last = d == {per - 1}", "return o, 0, d, first, last, False"]
            else:
                offsets = np.cumsum([0, *sched.phase_steps]).tolist()
                body.append("ph = " + " + ".join(f"jnp.asarray(d >= {off}, jnp.int32)" for off in offsets[1:-1]))
                body += [f"d = d - {_by_phase(offsets[:-1])}", "first = d == 0"]
                body.append(f"last = d == {_by_phase([k - 1 for k in sched.phase_steps])}")
                body.append("return o, ph, d, first, last, False")
            return _fn("_decode", "s", body)
        t = self.step_tables
        if sched.mode == "table":
            ps, pt = sched.pattern_steps, sched.pattern_tiles
            if sched.steps != ps:
                body = [f"rep = {_div_src('s', ps)}", _split_src("r", "s", "rep", ps)]
                body.append(f"o = rep * {pt} + {t['step_tile']}[r]")
            else:
                body = ["r = s", f"o = {t['step_tile']}[r]"]
            body.append(f"code = {t['step_code']}[r]")
        else:
            body = [f"o = {t['step_tile']}[s]", f"code = {t['step_code']}[s]"]
        body += [f"c = code >> {COLUMN_SHIFT}", f"first = (code & {FIRST}) != 0", f"last = (code & {LAST}) != 0"]
        if self.P <= 1:
            k = sched.phase_steps[0] if self.P else 0
            body += ["d = c", f"empty = d == {k}" if self.P else "empty = False", "return o, 0, d, first, last, empty"]
        else:
            offsets = sched.column_offsets()
            body.append("ph = " + " + ".join(f"jnp.asarray(c >= {off}, jnp.int32)" for off in offsets[1:]))
            body.append(f"d = c - {_by_phase(offsets)}")
            body.append(f"empty = d == {_by_phase(sched.phase_steps)}")
            body.append("return o, ph, d, first, last, empty")
        return _fn("_decode", "s", body)

    def _bounds(self) -> list[str]:
        """This core's [begin, end) steps and tile ordinals."""
        v, sched = self.v, self.sched
        table = self.step_tables["bounds"] if sched.mode == "runtime" else self.bounds_table
        if table is None:
            return [
                f"{v['s_begin']}, {v['s_end']} = 0, {sched.steps}",
                f"{v['o_begin']}, {v['o_end']} = 0, {sched.tiles}",
            ]
        n = self.cores
        return [
            f"{v['core']} = {'pl.program_id(0)' if n > 1 else '0'}",
            f"{v['s_begin']}, {v['s_end']} = {table}[{v['core']}], {table}[{v['core']} + 1]",
            f"{v['o_begin']}, {v['o_end']} = {table}[{n + 1} + {v['core']}], {table}[{n + 2} + {v['core']}]",
        ]

    def _in_phase(self, p: int, ph: str) -> str | None:
        """The test for 'this step belongs to phase p', or None when there is at most one phase."""
        return None if self.P <= 1 else f"{ph} == {p}"

    def _phase_test(self, p: int, ph: str, empty: str) -> str | None:
        """The test for 'this step is a non-empty step of phase p', or None when every step is."""
        if self.trivial:
            return None
        in_phase = self._in_phase(p, ph)
        if self.sched.mode == "uniform":  # no masks: no empty steps
            return in_phase
        return f"~{empty}" if in_phase is None else f"({in_phase}) & ~{empty}"

    def _flag_test(self, p: int, ph: str, flag: str) -> str:
        """The test for 'this step is phase p's step with `flag` (first / last) set'."""
        in_phase = self._in_phase(p, ph)
        return flag if in_phase is None else f"({in_phase}) & {flag}"

    # ------------------------------------------------------------------ copies

    def _copy_fn(self, op: Operand) -> str:
        return f"_{op.ref}_copy"

    def _window(self, op: Operand) -> str:
        """The window of a view's base array that the view's tile `op` is (see `views`)."""
        starts = []
        for d, (var, size, c) in enumerate(zip(op.block.loops, op.block.shape, op.block.const, strict=True)):
            expr = op.block.dynamic_index(d)
            if expr is not None:
                starts.append((self.indices[(expr, op.full_shape[d])], 1, 0))
            elif var is None:
                starts.append((None, 0, c * size))
            else:
                starts.append((self.blk[var], size, 0))
        return ", ".join(op.window.render(starts))

    def _copies(self, ops: Sequence[Operand] | None = None, outputs: bool = True) -> list[str]:
        """The copy functions of `ops` (default: every input) and, with `outputs`, of the outputs and fills."""
        lines = []
        for op in self.spec.operands if ops is None else ops:
            if op.window is not None:  # a view: copy the window of its base array
                src = f"{self.hbm[op.param]}.at[{self._window(op)}]"
            else:
                src = f"{self.hbm[op.param]}.at[{self._slices(op.block, self.blk, full=op.full_shape)}]"
            dst = f"{self.slots[op.ref]}.at[slot], {self.sems[op.ref]}.at[slot]"
            body = [
                *self._unpack_blocks(op.block),
                f"src = {src}  # {op.param}[{op.block.text}]",
                f"return pltpu.make_async_copy(src, {dst})",
            ]
            lines += [*_fn(self._copy_fn(op), "blocks, slot", body), ""]
        for out in self.spec.outputs if outputs else []:
            dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, self.blk)}]"
            src = f"{self.out_slots[out.var]}.at[slot], dst, {self.out_sems[out.var]}.at[slot]"
            body = [
                *self._unpack_blocks(out.block),
                f"dst = {dst}  # {out.var}[{out.block.text}]",
                f"return pltpu.make_async_copy({src})",
            ]
            lines += [*_fn(f"_{out.ref}_copy", "blocks, slot", body), ""]
            if out.var in self.fills:
                fill = self.fills[out.var]
                dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, {}, fill='c')}]"
                body = [f"dst = {dst}", f"return pltpu.make_async_copy({fill}, dst, {fill}_sem)"]
                lines += [*_fn(f"_{out.var}_fill_copy", "c", body), ""]
        return lines

    # ------------------------------------------------------------------ user code -> kernel code

    def _rewrite(self, node: ast.AST, unit: Unit | None) -> ast.AST:
        """User code of `unit` (None: the stores) as kernel code: inputs become their tile refs, state variables
        their scratch refs, `@` an MXU matmul, and finished reduction loops' variables their last values."""
        spec, params, state = self.spec, self.nest.params, self.spec.state
        phase = unit.phase if unit is not None else None
        weak = self._weak

        class Rewriter(ast.NodeTransformer):
            def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
                if isinstance(n.value, ast.Name) and n.value.id in params:
                    ref = spec.operand(n, phase).ref
                    return ast.copy_location(ast.parse(f"{ref}[...]", mode="eval").body, n)
                return self.generic_visit(n)

            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id not in state:
                    return n
                sv = state[n.id]
                index = ast.Constant(0) if sv.smem else ast.Constant(...)
                ref = ast.copy_location(ast.Subscript(ast.Name(sv.ref, ast.Load()), index, n.ctx), n)
                if not isinstance(n.ctx, ast.Load):
                    return ref
                if _stored_dtype(sv) != sv.aval.dtype:  # back from its 32-bit SMEM word
                    ref = ast.parse(f"{ast.unparse(ref)}.astype({_dtype_src(sv.aval.dtype)})", mode="eval").body
                if sv.aval.weak_type:  # it holds Python scalars: read it as one
                    ref = ast.Call(ast.Name(weak(), ast.Load()), [ref], [])
                return ast.copy_location(ref, n)

        out = Rewriter().visit(spec.unit_code(node, unit))
        self.used_names |= {n.id for n in ast.walk(out) if isinstance(n, ast.Name)}
        return out

    def _statements(self, unit: Unit) -> list[str]:
        lines = []
        accumulators = {id(a.update): a for a in self.spec.accumulators}
        skipped = {id(a.init) for a in self.spec.accumulators}
        for original in unit.statements:
            if id(original) in skipped:  # an accumulator's zeros: its first step stores instead of adding
                continue
            acc = accumulators.get(id(original))
            if acc is not None:
                lines += self._accumulate(acc, original, unit)
                continue
            lines.append(self._statement(original, original, unit))
        return lines

    def _statement(self, s: ast.stmt, original: ast.stmt, unit: Unit) -> str:
        """`s` (the user's statement `original`, or one standing in for it) as kernel code."""
        target = assigned_name(s)
        if isinstance(s, ast.AugAssign) and target in self.spec.state:  # x += v  ->  x = x + v, on the ref
            value = ast.BinOp(ast.Name(target, ast.Load()), s.op, s.value)
            s = ast.copy_location(ast.Assign([ast.Name(target, ast.Store())], value), s)
        out = self._rewrite(s, unit)
        dtypes = [self.spec.casts.get(id(original))]  # to the state variable's type
        sv = self.spec.state.get(target)
        if sv is not None and _stored_dtype(sv) != sv.aval.dtype:
            dtypes.append(_stored_dtype(sv))  # then into its 32-bit SMEM word
        for dtype in dtypes:
            if dtype is not None and isinstance(out, ast.Assign):
                value = f"jnp.asarray({ast.unparse(out.value)}, {_dtype_src(dtype)})"
                out.value = ast.parse(value, mode="eval").body
        return ast.unparse(out)

    def _accumulate(self, acc: Any, original: ast.stmt, unit: Unit) -> list[str]:
        """`var += expr` on a zero-initialised accumulator in a tile of one step: just `var = expr`."""
        aval = self.spec.avals[id(original)]
        # `zeros + expr` has the sum's type and shape (`expr` may broadcast, like a (rows, 1) column).
        value = f"jnp.asarray({ast.unparse(self._rewrite(acc.expr, unit))}, {_dtype_src(aval.dtype)})"
        return [f"{acc.var} = jnp.broadcast_to({value}, {tuple(aval.shape)})"]

    def _stores(self) -> list[str]:
        v, lines = self.v, []
        for out in self.spec.outputs:
            store = next(st for st in self.nest.stores if st.out == out.var)
            value = f"jnp.asarray({ast.unparse(self._rewrite(store.value, None))}).astype({out.ref}.dtype)"
            if out.var in self.spec.broadcast_stores:  # `.at[...].set(v)` broadcasts v to the tile
                value = f"jnp.broadcast_to({value}, {out.ref}.shape)"
            copy_fn = f"_{out.ref}_copy"
            lines += [
                f"{out.ref} = {self.out_slots[out.var]}.at[{v['o']} & 1]",
                f"{out.ref}[...] = {value}",
                f"{copy_fn}({v['blocks']}, {v['o']} & 1).start()",
            ]
        return lines

    def _free_output_slots(self) -> list[str]:
        """Wait for the copies that last used this tile's output slots (two tiles ago) to leave.

        Done before the tile's last step runs any of the user's code: a wait is a barrier Mosaic schedules
        nothing across, so a value computed before it and stored after it (the tile's matmul) would be parked
        in VMEM and read back (on a v5e, 2% of a matmul).
        """
        v = self.v
        waits = [f"_{out.ref}_copy(_blocks({v['o']} - 2, 0), {v['o']} & 1).wait()" for out in self.spec.outputs]
        return _fn("_free_output_slots", "", waits, when=f"{v['o']} - 2 >= {v['o_begin']}")

    def _units(self) -> list[str]:
        """The user's code for one step, guarded by the step's phase and flags."""
        v, P, units = self.v, self.P, self.spec.units
        segments = [u for u in units if u.kind == "segment"]
        bodies = [u for u in units if u.kind == "body"]
        fill_issue = self._fill_issue()
        final = self._statements(segments[-1]) + self._stores()
        free = self._free_output_slots()
        if P == 0:
            return free + fill_issue + final
        if self.spec.single_step:  # the tile's one step: its code in order, values as plain locals
            return free + fill_issue + self._statements(segments[0]) + self._statements(bodies[0]) + final
        lines = _block("_last_step", free, self._flag_test(P - 1, v["ph"], v["last"]))
        for p in range(P):
            lead = (fill_issue if p == 0 else []) + self._statements(segments[p])
            if lead:
                lines += _fn(f"_segment_{p}", "", lead, when=self._flag_test(p, v["ph"], v["first"]))
            body = self._statements(bodies[p])
            lines += _block(f"_body_{p}", body, self._phase_test(p, v["ph"], v["empty"]))
        return lines + _fn("_finish_tile", "", final, when=self._flag_test(P - 1, v["ph"], v["last"]))

    # ------------------------------------------------------------------ masked-out tiles

    def _fill_issue(self) -> list[str]:
        v, lines = self.v, []
        for var in self.fills:
            limit = _fn(
                "_limit_in_flight",
                "",
                [f"_{var}_fill_copy(0).wait()"],
                when=f"c - {self.fill_start}[{v['o_begin']}] >= {FILL_DMA_DEPTH}",
            )
            lines += _fn(
                f"_start_{var}_fills",
                "c",
                [*limit, f"_{var}_fill_copy(c).start()"],
                loop=f"{self.fill_start}[{v['o']}], {self.fill_start}[{v['o']} + 1]",
            )
        return lines

    def _fill_drain(self) -> list[str]:
        v, lines = self.v, []
        for var in self.fills:
            issued = f"{self.fill_start}[{v['o_end']}] - {self.fill_start}[{v['o_begin']}]"
            lines += _fn(
                f"_drain_{var}_fills",
                "_",
                [f"_{var}_fill_copy(0).wait()"],
                loop=f"0, jnp.minimum({issued}, {FILL_DMA_DEPTH})",
            )
        return lines

    # ------------------------------------------------------------------ kernel

    def _by_phase_ops(self, ops: Sequence[Operand]) -> list[tuple[int, list[Operand]]]:
        return [(p, [op for op in ops if op.phase == p]) for p in range(self.P) if any(op.phase == p for op in ops)]

    def _fetch_ahead(self, step: str, name: str, ops: Sequence[Operand] | None = None) -> list[str]:
        """Start copying step `step`'s streamed inputs (or just `ops`) into their ring slots."""
        slot = _rem_src(f"({step})", self.buffers)
        body = [f"o2, ph2, d2, _, _, empty2 = _decode({step})", "blocks2 = _blocks(o2, d2)"]
        for p, phase_ops in self._by_phase_ops(self.streamed if ops is None else ops):
            starts = [f"{self._copy_fn(op)}(blocks2, {slot}).start()" for op in phase_ops]
            body += _block(f"_fetch_phase_{p}", starts, self._phase_test(p, "ph2", "empty2"))
        return _fn(name, "", body, when=f"{step} < {self.v['s_end']}")

    def _step(self, hook: bool = False) -> list[str]:
        v, N = self.v, self.buffers
        slot = _rem_src(v["s"], N)
        lines = [
            f"{v['o']}, {v['ph']}, {v['d']}, {v['first']}, {v['last']}, {v['empty']} = _decode({v['s']})",
            f"{v['blocks']} = _blocks({v['o']}, {v['d']})",
        ]
        if hook:  # once this nest's own copies are all issued: start the next nest's first copies
            hook_at, call = self.pieces["hook_at"], f"{self.pieces['hook']}()"
            lines += _fn("_start_next_nest", "", [call], when=f"{v['s']} == {hook_at}")
        if self.streamed:  # 1. prefetch the step N - 1 ahead
            lines += self._fetch_ahead(f"{v['s']} + {N - 1}", "_prefetch")
        if self.per_tile:  # 2. per-tile inputs, at a tile's first step
            next_tile = [
                f"{self._copy_fn(op)}(_blocks({v['o']} + 1, 0), ({v['o']} + 1) & 1).start()" for op in self.per_tile
            ]
            body = [
                *_fn("_next_tile_inputs", "", next_tile, when=f"{v['o']} + 1 < {v['o_end']}"),
                *[f"{self._copy_fn(op)}({v['blocks']}, {v['o']} & 1).wait()" for op in self.per_tile],
            ]
            # A tile of one step is always at its first step: no test (Mosaic schedules unguarded waits better).
            guard = self._flag_test(0, v["ph"], v["first"]) if self.P and not self.spec.single_step else None
            lines += _block("_tile_inputs", body, guard)
        for p, ops in self._by_phase_ops(self.streamed):  # 3. this step's inputs
            waits = [f"{self._copy_fn(op)}({v['blocks']}, {slot}).wait()" for op in ops]
            lines += _block(f"_wait_phase_{p}", waits, self._phase_test(p, v["ph"], v["empty"]))
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{slot}]" for op in self.streamed]
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{v['o']} & 1]" for op in self.per_tile]
        units = self._units()  # 4. the user's code
        values = self._loop_values()
        if values:
            lines.append(f"{self._blocks_tuple(self.blk)} = {v['blocks']}")
        return lines + values + units

    def _arg_groups(self) -> dict[str, list[str]]:
        """The kernel's parameters by kind, in order: SMEM inputs, HBM inputs, aliased inputs, outputs, scratch."""
        runtime = [self.step_tables[k] for k in RUNTIME_TABLES] if self.sched.mode == "runtime" else []
        return {
            "smem": [*self.tables, *runtime, *(r for r in (self.scalars_ref, self.fscalars_ref) if r)],
            "hbm": list(self.hbm.values()),
            "aliased": list(self.aliased.values()),
            "outputs": list(self.out_hbm.values()),
            "scratch": [name for names, _, _ in self._scratch_buffers() for name in names],
        }

    def _kernel_args(self) -> list[str]:
        return [a for group in self._arg_groups().values() for a in group]

    def _scalar_reads(self) -> list[str]:
        """Bind the runtime scalars (as the user's names) and the runtime indices computed from them."""
        lines = []
        for n, p in enumerate(self.int_scalars):  # a weak int32, or a narrower integer as itself (`scalar_aval`)
            dtype, value = np.dtype(self.spec.specs[self.nest.params.index(p)].dtype), f"{self.scalars_ref}[{n}]"
            read = f"{self._weak()}({value})" if dtype == np.int32 else f"{value}.astype(jnp.{dtype})"
            lines.append(f"{p} = {read}")
        for n, (p, weak) in enumerate(self.float_scalars):
            value = f"{self.fscalars_ref}[{n}]"
            lines.append(f"{p} = {self._weak()}({value})" if weak else f"{p} = {value}")
        lines += [f"{name} = _index({expr}, {size})" for (expr, size), name in self.indices.items()]
        return lines

    def _setup(self) -> list[str]:
        """The kernel's definitions: this core's bounds, the runtime scalars, step decoding and copies."""
        reads = self._scalar_reads()
        lines = [*self._bounds(), *(["", *reads] if reads else []), ""]
        return [*lines, *self._decode(), "", *self._blocks(), "", *self._copies()]

    def _fill_init(self) -> list[str]:
        lines = []
        for out in self.spec.outputs:
            if out.var in self.fills:
                fill = self.fills[out.var]
                lines.append(f"{fill}[...] = jnp.full({fill}.shape, {_scalar_src(out.fill)}, {fill}.dtype)")
        return lines

    def _first_fetches(self, dependent: bool | None = None) -> list[str]:
        """Start the first steps' copies: of every input (None), or only those an earlier fused nest writes
        (True) or doesn't (False)."""
        v, N, lines = self.v, self.buffers, []

        def wanted(op: Operand) -> bool:
            return dependent is None or (op.param in self.dependent) == dependent

        per_tile = [op for op in self.per_tile if wanted(op)]
        streamed = [op for op in self.streamed if wanted(op)]
        if per_tile:
            first = [f"{self._copy_fn(op)}(_blocks({v['o_begin']}, 0), {v['o_begin']} & 1).start()" for op in per_tile]
            lines += _fn("_first_tile_inputs", "", first, when=f"{v['o_begin']} < {v['o_end']}")
        if streamed:
            for k in range(N - 1):
                lines += self._fetch_ahead(f"{v['s_begin']} + {k}", f"_prefetch_{k}", streamed)
        return lines

    def _run(self, hook: bool = False) -> list[str]:
        """The step loop. With `hook`, it calls the next fused nest's prologue once its own copies are issued."""
        v, N = self.v, self.buffers
        lines = []
        # The step bounds are read from SMEM (not Python ints) with runtime tables or several TensorCores.
        traced = self.sched.mode == "runtime" or self.cores > 1
        if hook:
            at = f"{v['s_end']} - {N - 1}, {v['s_begin']}"
            at = f"jnp.maximum({at})" if traced else f"max({at})"
            lines.append(f"{self.pieces['hook_at']} = {at}")
        lines += _fn("_steps", v["s"], self._step(hook), loop=f"{v['s_begin']}, {v['s_end']}")
        if hook and traced:  # no steps at all (on this core): the loop never started the next nest
            lines += _fn("_no_steps", "", [f"{self.pieces['hook']}()"], when=f"{v['s_end']} <= {v['s_begin']}")
        return lines

    def _drain(self) -> list[str]:
        v, lines = self.v, []
        for out in self.spec.outputs:
            for back in (2, 1):
                wait = [f"_{out.ref}_copy(_blocks({v['o_end']} - {back}, 0), ({v['o_end']} - {back}) & 1).wait()"]
                lines += _fn(f"_drain_{out.ref}_{back}", "", wait, when=f"{v['o_end']} - {back} >= {v['o_begin']}")
        return lines + self._fill_drain()

    def _kernel(self, name: str) -> list[str]:
        body = [*self._setup(), *self._fill_init(), "", "# Fetch the first steps' inputs.", *self._first_fetches()]
        body += ["", *self._run(), "", "# Wait for the copies still in flight.", *self._drain()]
        return [f"def {name}({', '.join(self._kernel_args())}):", *_indent(body)]

    def part(self, name: str, hook: bool) -> list[str]:
        """This nest as one part of a fused kernel. `name(<its refs>, <its early scratch>)` defines its copies
        and returns two functions, which the fused kernel runs in turn:

        - early(): start the first copies of the inputs no earlier part writes (run while the previous part
          finishes), into the part's *early* scratch (`scratch_groups`), allocated before it;
        - with_scratch(<its other scratch>) -> (late, run, drain), once the rest of its scratch exists:
          - late(): start the first copies of the inputs an earlier part wrote (after it has drained);
          - run(hook): the step loop; `hook` (the next part's `early`) is called once this part's own copies
            have all been issued;
          - drain(): wait for this part's output copies.
        """
        p = self.pieces
        early_ops = [op for op in self.spec.operands if op.param not in self.dependent]
        late_ops = [op for op in self.spec.operands if op.param in self.dependent]
        groups = self._arg_groups()
        early_scratch, late_scratch = (names for names, _ in self.scratch_groups())
        args = [*groups["smem"], *groups["hbm"], *groups["aliased"], *groups["outputs"], *early_scratch]
        reads = self._scalar_reads()
        body = [*self._bounds(), *(["", *reads] if reads else []), "", *self._decode(), "", *self._blocks(), ""]
        body += self._copies(early_ops, outputs=False)
        body += _fn(p["early"], "", [*self._fill_init(), *self._first_fetches(False)])
        inner = [*self._copies(late_ops, outputs=True)]
        inner += _fn(p["late"], "", self._first_fetches(True))
        inner += _fn(p["run"], p["hook"] if hook else "", self._run(hook)) + _fn(p["drain"], "", self._drain())
        inner.append(f"return {p['late']}, {p['run']}, {p['drain']}")
        body += _fn(p["with_scratch"], ", ".join(late_scratch), inner)
        body.append(f"return {p['early']}, {p['with_scratch']}")
        return [f"def {name}({', '.join(args)}):", *_indent(body)]

    def scratch_groups(self) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
        """A fused part's scratch as ((names, shapes) early, (names, shapes) late): the early scratch is what
        its `early` copies write (the rings of inputs no earlier part writes, and the skipped tiles' fill
        values); the rest (the rings of inputs earlier parts write, output slots, state) is only needed once
        the part runs."""
        early: tuple[list[str], list[str]] = ([], [])
        late: tuple[list[str], list[str]] = ([], [])
        for names, shapes, is_early in self._scratch_buffers():
            group = early if is_early else late
            group[0].extend(names)
            group[1].extend(shapes)
        return early, late

    # ------------------------------------------------------------------ wrapper

    def _scratch_buffers(self) -> list[tuple[list[str], list[str], bool]]:
        """Each scratch buffer, in kernel-argument order: (its names, its shapes, written by a fused part's
        `early` copies)."""
        spec, N, out = self.spec, self.buffers, []
        for op in spec.operands:
            depth = N if op.phase is not None else 2
            shape = (depth, *op.block.kernel_shape)
            lines = [
                f"pltpu.VMEM({shape}, {_dtype_src(op.dtype)}),  # {op.param}[{op.block.text}]",
                f"pltpu.SemaphoreType.DMA(({depth},)),",
            ]
            out.append(([self.slots[op.ref], self.sems[op.ref]], lines, op.param not in self.dependent))
        for o in spec.outputs:
            shape = (2, *o.block.kernel_shape)
            lines = [
                f"pltpu.VMEM({shape}, {_dtype_src(o.out.dtype)}),  # {o.var}[{o.block.text}]",
                "pltpu.SemaphoreType.DMA((2,)),",
            ]
            out.append(([self.out_slots[o.var], self.out_sems[o.var]], lines, False))
        for sv in spec.state.values():
            dtype = _dtype_src(_stored_dtype(sv))
            mem = f"pltpu.SMEM((1,), {dtype})" if sv.smem else f"pltpu.VMEM({tuple(sv.aval.shape)}, {dtype})"
            out.append(([sv.ref], [f"{mem},  # {sv.name}"], False))
        for o in spec.outputs:
            if o.var in self.fills:
                fill = self.fills[o.var]
                lines = [
                    f"pltpu.VMEM({o.block.kernel_shape}, {_dtype_src(o.out.dtype)}),  # {o.var}'s skipped tiles",
                    "pltpu.SemaphoreType.DMA(()),",
                ]
                out.append(([fill, f"{fill}_sem"], lines, True))
        return out

    def _scratch(self) -> list[str]:
        return [line for _, lines, _ in self._scratch_buffers() for line in lines]

    def _params(self) -> dict[str, str]:
        """The wrapper's parameters: the kernel's, renamed where they would hide a name the wrapper uses."""
        if self._param_names is None:
            reserved = {"interpret", "jax", "jnp", "np", "pl", "pltpu", "_COST", "_runtime_tables", "_index"}
            self._param_names = {p: self.names.fresh(f"{p}_arg") if p in reserved else p for p in self.nest.params}
        return self._param_names

    def _wrapper_inputs(self) -> tuple[list[str], list[str], list[str]]:
        """What the wrapper computes before the call: (its lines, the SMEM inputs, the aliased initial values)."""
        spec, param = self.spec, self._params()
        lines, smem, inits = [], list(self.table_consts.values()), []
        if self.sched.mode == "runtime":
            if self._tables_local is None:
                self._tables_local = self.names.fresh("tables")
            tables = self._tables_local
            lines.append("# A mask reads inputs, so the steps are listed on device.")
            lines.append(f"{tables} = _runtime_tables({', '.join(f'{p}={a}' for p, a in param.items())})")
            smem += [f"{tables}[{k!r}]" for k in RUNTIME_TABLES]
        if self.scalars_ref:
            values = ", ".join(f"jnp.asarray({param[p]}, jnp.int32)" for p in self.int_scalars)
            lines.append(f"{self.scalars_ref} = jnp.stack([{values}])  # runtime scalars, read from SMEM")
            smem.append(self.scalars_ref)
        if self.fscalars_ref:
            values = ", ".join(f"jnp.asarray({param[p]}, jnp.float32)" for p, _ in self.float_scalars)
            lines.append(f"{self.fscalars_ref} = jnp.stack([{values}])  # float runtime scalars, read from SMEM")
            smem.append(self.fscalars_ref)

        class Renamed(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                return ast.copy_location(ast.Name(param.get(n.id, n.id), n.ctx), n)

        for var, init in self.aliased.items():
            alloc = Renamed().visit(copy.deepcopy(next(o.alloc for o in spec.outputs if o.var == var)))
            lines.append(f"{init} = {ast.unparse(alloc)}  # skipped tiles keep these values")
            inits.append(init)
        return lines, smem, inits

    def _wrapper(self, kernel: str) -> list[str]:
        spec, nest = self.spec, self.nest
        smem, anyspace = "pl.BlockSpec(memory_space=pltpu.SMEM)", "pl.BlockSpec(memory_space=pl.ANY)"
        prep, smem_args, inits = self._wrapper_inputs()
        in_specs = [smem] * len(smem_args) + [anyspace] * (len(self.hbm) + len(inits))
        outs = [f"jax.ShapeDtypeStruct({tuple(o.out.shape)}, {_dtype_src(o.out.dtype)})" for o in spec.outputs]
        out_shape = f"({', '.join(outs)},)" if nest.returns_tuple else outs[0]
        out_specs = f"({', '.join([anyspace] * len(outs))},)" if nest.returns_tuple else anyspace

        param = self._params()
        lines = [f"def {nest.fn_name}_pallas({', '.join(param.values())}, *, interpret=False):"]
        lines.append(f"    # No grid: one loop over {self._describe()}; inputs {self.buffers}-buffered by hand.")
        lines += _indent(prep)
        args = [*smem_args, *[param[p] for p in self.hbm], *inits]
        aliases = {len(args) - len(inits) + n: nest.out_vars.index(var) for n, var in enumerate(self.aliased)}
        params = [f"vmem_limit_bytes={self.vmem_limit}"] + (
            ['dimension_semantics=("parallel",)'] if self.cores > 1 else []
        )
        lines += [
            "    return pl.pallas_call(",
            f"        {kernel},",
            f"        out_shape={out_shape},",
            *([f"        grid=({self.cores},),"] if self.cores > 1 else []),
            f"        in_specs=[{', '.join(in_specs)}],",
            f"        out_specs={out_specs},",
            "        scratch_shapes=[",
            *_indent(self._scratch(), 3),
            "        ],",
            *([f"        input_output_aliases={aliases},"] if aliases else []),
            f"        compiler_params=pltpu.CompilerParams({', '.join(params)}),",
            *(["        cost_estimate=_COST,"] if self.cost is not None else []),
            "        interpret=interpret,",
            f"        name={nest.fn_name!r},",
            f"    )({', '.join(args)})",
        ]
        return lines

    def prepare(self, name: str) -> list[str]:
        """For a fused kernel: `name(*inputs)` -> (SMEM inputs, aliased initial values) of this part.

        Inputs an earlier part writes are passed as None: the fused kernel only fuses nests whose masks and
        aliased allocations don't read them.
        """
        prep, smem, inits = self._wrapper_inputs()
        body = [*prep, f"return [{', '.join(smem)}], [{', '.join(inits)}]"]
        return _fn(name, ", ".join(self._params().values()), body)

    def scratch(self) -> list[str]:
        return self._scratch()

    def arg_groups(self) -> dict[str, list[str]]:
        return self._arg_groups()

    def describe(self) -> str:
        return self._describe()

    def _describe(self) -> str:
        sched, spec = self.sched, self.spec
        total = spec.total_tiles
        tiles = f"{sched.tiles} tiles" + (f" (of {total})" if sched.tiles != total else "")
        if sched.mode == "runtime":
            return f"the steps of up to {tiles} that the masks keep at run time"
        cores = f", split over {self.cores} TensorCores" if self.cores > 1 else ""
        return f"{sched.steps} steps of {tiles}{cores}"

    # ------------------------------------------------------------------ assembly

    def header(self) -> str:
        spec, nest = self.spec, self.nest
        inputs = ", ".join(
            f"{p}: {_dtype_src(s.dtype)}{tuple(s.shape)}" for p, s in zip(nest.params, spec.specs, strict=True)
        )
        header = f"# Generated by autokernel from `{nest.fn_name}` ({spec.filename}:{nest.first_line}) for {inputs}"
        if spec.overrides:
            header += "\n# Tuned: " + ", ".join(f"{k}={v}" for k, v in sorted(spec.overrides.items()))
        return header

    def weak_helper(self) -> str:
        """The weak-type helper's source, when the generated code uses it (after it is generated)."""
        return _WEAK.format(name=self.weak) if self.weak is not None else ""

    def namespace(self) -> dict[str, Any]:
        """What the generated code runs with: the user's constants, the prelude's imports and the tables."""
        spec = self.spec
        namespace: dict[str, Any] = dict(spec.consts)
        namespace.update(jax=jax, jnp=jnp, np=np, pl=pl, pltpu=pltpu, _COST=self.cost)
        namespace.update({self.table_consts[name]: value for name, value in self.tables.items()})
        if self.sched.mode == "runtime":
            namespace["_runtime_tables"] = lambda **inputs: schedule_lib.runtime_tables(spec, self.sched, inputs)
        return namespace

    def emit(self) -> Generated:
        nest = self.nest
        kernel_name = self.names.fresh(f"{nest.fn_name}_kernel")
        kernel = self._kernel(kernel_name)
        wrapper = self._wrapper(kernel_name)
        helpers = [prelude(self.precision, bool(self.spec.scalars)), self.weak_helper()]
        source = "\n".join([self.header(), "", "\n\n".join(filter(None, helpers)), "", *kernel, "", "", *wrapper, ""])
        namespace = self.namespace()
        exec(compile(source, f"<autokernel:{nest.fn_name}>", "exec"), namespace)
        return Generated(source, namespace[f"{nest.fn_name}_pallas"])


def prelude(precision: str, scalars: bool) -> str:
    """The helpers generated kernels use: index arithmetic, runtime scalars (when there are any), `@`."""
    return "\n\n".join([_PRELUDE, *([_SCALAR_PRELUDE] if scalars else []), mxu.prelude(precision)])
