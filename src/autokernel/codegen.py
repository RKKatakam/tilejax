"""Generate a gridless Pallas TPU kernel from a `Specialization` and its `Schedule`, then compile it.

The kernel runs every step of its share of the work in one `pl.loop` and moves data with hand-written
async copies. One step:

1. decode the step: tile ordinal, phase, reduction position, first/last/empty flags;
2. start copying the inputs of the step `buffers - 1` ahead into a ring of VMEM slots;
3. at a tile's first step: start fetching the next tile's per-tile inputs, wait for this tile's;
4. wait for this step's inputs;
5. run the user's units: a phase's leading segment on its first step, its reduction body on non-empty
   steps, and the final segment plus output stores on the tile's last step;
6. stores write each output tile into one of two VMEM slots and copy it to HBM asynchronously.

On chips with two TensorCores per device the kernel gets a one-axis "parallel" grid, one point per
TensorCore, and each core runs a contiguous run of tiles. `Generated.source` holds the generated code.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import hardware
from . import schedule as schedule_lib
from .analysis import Block, Operand, Specialization
from .errors import AutoKernelError
from .frontend import assigned_name
from .schedule import COLUMN_SHIFT, FIRST, LAST, Schedule

__all__ = ["FILL_DMA_DEPTH", "MATMUL_PRECISIONS", "Generated", "check_layout", "generate", "layout_error"]

# Mosaic's default f32 matmul is a single bf16 pass; HIGHEST does the multi-pass f32 matmul the
# Python code actually describes.
MATMUL_PRECISIONS = {"highest": "jax.lax.Precision.HIGHEST", "default": "None"}

# Constant (masked-out) tiles are written with at most this many fill DMAs in flight.
FILL_DMA_DEPTH = 4

# Tables a runtime schedule builds on device, in kernel-argument order.
_RUNTIME_TABLES = ("step_tile", "step_code", "tile_ids", "bounds")

_PRELUDE = """\
def _div(a, b):
    # a // b for the non-negative step and tile indices: one integer division, no floor-sign fix-ups.
    return jax.lax.div(a, b)


def _rem(a, b):
    return jax.lax.rem(a, b)


def _mxu_dot(x, y):
    # `@`: an MXU matmul accumulating in f32 (i32 for integer inputs); leading dims are batch dims.
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    # bf16/int8 operands are exact on the MXU in one pass; only f32 operands take a precision.
    precision = {precision} if x.dtype == jnp.float32 or y.dtype == jnp.float32 else None
    if x.ndim == 2 and y.ndim == 2:
        return jnp.dot(x, y, preferred_element_type=acc, precision=precision)
    batch = tuple(range(x.ndim - 2))
    dims = (((x.ndim - 1,), (y.ndim - 2,)), (batch, batch))
    return jax.lax.dot_general(x, y, dims, preferred_element_type=acc, precision=precision)


def _mxu_dot_t(x, y):
    # `x @ y.T`: contract the last dims of both, so the MXU reads y transposed instead of us transposing it.
    if y.ndim != 2:
        return _mxu_dot(x, y.T)
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    precision = {precision} if x.dtype == jnp.float32 or y.dtype == jnp.float32 else None
    return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=acc, precision=precision)
"""


@dataclass
class Generated:
    """Generated source and the callable its wrapper defines."""

    source: str
    entry: Callable[..., Any]  # entry(*inputs, interpret=False) -> output(s)
    tables: dict[str, np.ndarray] = field(default_factory=dict)  # static SMEM tables


def generate(
    spec: Specialization,
    sched: Schedule,
    *,
    buffers: int,
    matmul_precision: str,
    background_fill: bool,
    vmem_limit_bytes: int,
    cost: pl.CostEstimate | None,
) -> Generated:
    return _Emitter(spec, sched, buffers, matmul_precision, background_fill, vmem_limit_bytes, cost).emit()


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


def _dtype_src(dtype: Any) -> str:
    return f"jnp.{np.dtype(dtype).name}"


def _scalar_src(value: Any) -> str:
    return repr(value) if np.isfinite(value) else f"float('{value}')"


def _indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line if line else line for line in lines]


def _fn(name: str, args: str, body: list[str], when: str | None = None, loop: str | None = None) -> list[str]:
    """A nested function, optionally under `@pl.when(when)` or `@pl.loop(loop)`."""
    decorator = [f"@pl.when({when})"] if when else [f"@pl.loop({loop})"] if loop else []
    return [*decorator, f"def {name}({args}):", *_indent(body or ["pass"])]


def _by_phase(values: Sequence[Any]) -> str:
    """Pick `values[ph]` with a chain of selects (Mosaic can't lower jnp.select's argmax over booleans)."""
    out = str(values[-1])
    for p in range(len(values) - 2, -1, -1):
        out = f"jnp.where(ph == {p}, {values[p]}, {out})"
    return out


def _slot(step: str, n: int) -> str:
    """Ring slot of a step: a bitmask for power-of-two rings, a remainder otherwise."""
    return f"({step}) & {n - 1}" if n & (n - 1) == 0 else f"_rem({step}, {n})"


class _Emitter:
    def __init__(self, spec, sched, buffers, precision, background_fill, vmem_limit, cost):
        self.spec, self.sched, self.nest = spec, sched, spec.nest
        self.buffers, self.precision, self.vmem_limit, self.cost = buffers, precision, vmem_limit, cost
        names = spec.names
        self.P = len(spec.phases)
        self.cores = sched.cores
        self.v = {
            k: names.fresh(k)
            for k in ("s", "o", "ph", "d", "first", "last", "empty", "core", "s_begin", "s_end", "o_begin", "o_end")
        }
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

        # Outputs with skipped tiles get those tiles from fill DMAs (a known constant, static mask) or
        # keep their initial values through an aliased input.
        masked = bool(spec.constant_tiles) or spec.tile_mask_runtime
        fill_ok = background_fill and not spec.tile_mask_runtime
        self.fills = {
            o.var: names.fresh(f"{o.var}_fill") for o in spec.outputs if masked and fill_ok and o.fill is not None
        }
        self.aliased = {o.var: names.fresh(f"{o.var}_init") for o in spec.outputs if masked and o.var not in self.fills}

        # Static SMEM tables, embedded as constants.
        self.tables: dict[str, np.ndarray] = {}
        self.tile_tables: dict[str, str] = {}
        if spec.nest.guard is not None and not spec.tile_mask_runtime:  # a static tile mask: list the tiles
            for n, lb in enumerate(spec.parallel):
                self.tile_tables[lb.var] = names.fresh(f"{lb.var}_blocks")
                self.tables[self.tile_tables[lb.var]] = np.array([t[n] for t in spec.tiles], np.int32)
        if self.fills:
            self.fill_tables = {lb.var: names.fresh(f"{lb.var}_fill_blocks") for lb in spec.parallel}
            for n, lb in enumerate(spec.parallel):
                self.tables[self.fill_tables[lb.var]] = np.array([t[n] for t in spec.constant_tiles], np.int32)
            n_tiles, n_fill = len(spec.tiles), len(spec.constant_tiles)
            self.fill_start = names.fresh("fill_start")  # tile ordinal o writes fills [start[o], start[o + 1])
            self.tables[self.fill_start] = np.array([t * n_fill // n_tiles for t in range(n_tiles + 1)], np.int32)
        self.step_tables: dict[str, str] = {}
        if sched.mode == "table":
            for key, value in sched.tables.items():
                self.step_tables[key] = names.fresh(key)
                self.tables[self.step_tables[key]] = value
        elif sched.mode == "runtime":
            self.step_tables = {k: names.fresh(k) for k in _RUNTIME_TABLES}
        self.bounds_table = None
        if sched.mode != "runtime" and self.cores > 1:
            self.bounds_table = names.fresh("core_bounds")  # [step bound per core..., tile bound per core...]
            steps = [b for b, _ in sched.core_steps] + [sched.core_steps[-1][1]]
            tiles = [b for b, _ in sched.core_tiles] + [sched.core_tiles[-1][1]]
            self.tables[self.bounds_table] = np.array(steps + tiles, np.int32)

    # ------------------------------------------------------------------ coordinates

    def _tile_block(self, var: str, o: str) -> str:
        """Absolute block index of parallel loop `var` for tile ordinal `o`."""
        if var in self.tile_tables:
            return f"{self.tile_tables[var]}[{o}]"
        t = f"{self.step_tables['tile_ids']}[{o}]" if self.spec.tile_mask_runtime else o
        stride = 1
        for lb in reversed(self.spec.parallel):
            if lb.var == var:
                idx = f"_div({t}, {stride})" if stride != 1 else t
                if lb is not self.spec.parallel[0]:
                    idx = f"_rem({idx}, {lb.count})"
                base = lb.start // lb.step
                return f"({idx} + {base})" if base else idx
            stride *= lb.count
        raise AssertionError(var)

    def _reduction_block(self, var: str, d: str) -> str:
        """Absolute block index of reduction loop `var` at flattened phase position `d`."""
        for phase in self.spec.phases:
            names = [lb.var for lb in phase.loops]
            if var in names:
                k = names.index(var)
                stride = int(np.prod([lb.count for lb in phase.loops[k + 1 :]], dtype=np.int64))
                lb = phase.loops[k]
                idx = f"_div({d}, {stride})" if stride != 1 else d
                if k:
                    idx = f"_rem({idx}, {lb.count})"
                base = lb.start // lb.step
                return f"({idx} + {base})" if base else idx
        raise AssertionError(var)

    def _block_index(self, var: str, o: str, d: str) -> str:
        if var in {lb.var for lb in self.spec.parallel}:
            return self._tile_block(var, o)
        return self._reduction_block(var, d)

    def _slices(self, block: Block, o: str, d: str, fill: str | None = None) -> str:
        """The HBM window of a block: `pl.ds` per tiled dim, a plain index per squeezed dim."""
        parts = []
        for var, size, c, sq in zip(block.loops, block.shape, block.const, block.squeezed, strict=True):
            if var is None:
                start = str(c * size)
            else:
                idx = f"{self.fill_tables[var]}[{fill}]" if fill is not None else self._block_index(var, o, d)
                start = idx if size == 1 else f"{idx} * {size}"
            parts.append(start if sq else f"pl.ds({start}, {size})")
        return ", ".join(parts)

    def _loop_values(self, used: set[str]) -> list[str]:
        v, lines = self.v, []
        for var in sorted(used & set(self.spec.loops)):
            lb = self.spec.loops[var]
            idx = self._block_index(var, v["o"], v["d"])
            lines.append(f"{var} = {idx} * {lb.step}" if lb.step != 1 else f"{var} = {idx}")
        return lines

    # ------------------------------------------------------------------ step decoding

    def _decode(self) -> list[str]:
        """`_decode(s)` -> (tile ordinal, phase, reduction position, first, last, empty)."""
        sched = self.sched
        if sched.mode == "uniform":
            per = sched.steps_per_tile
            body = [f"o = _div(s, {per})" if per != 1 else "o = s", f"r = _rem(s, {per})" if per != 1 else "r = s * 0"]
            if self.P <= 1:
                k = sched.phase_steps[0] if self.P else 1
                body += ["ph = r * 0", "d = r", "first = d == 0", f"last = d == {k - 1}"]
            else:
                offsets = np.cumsum([0, *sched.phase_steps]).tolist()
                body.append("ph = " + " + ".join(f"(r >= {off}).astype(jnp.int32)" for off in offsets[1:-1]))
                body += [f"d = r - {_by_phase(offsets[:-1])}", "first = d == 0"]
                body.append(f"last = d == {_by_phase([k - 1 for k in sched.phase_steps])}")
            body.append("return o, ph, d, first, last, jnp.bool_(False)")
            return _fn("_decode", "s", body)
        t = self.step_tables
        if sched.mode == "table":
            ps, pt = sched.pattern_steps, sched.pattern_tiles
            if sched.steps != ps:
                body = [f"rep, r = _div(s, {ps}), _rem(s, {ps})", f"o = rep * {pt} + {t['step_tile']}[r]"]
            else:
                body = ["r = s", f"o = {t['step_tile']}[r]"]
            body.append(f"code = {t['step_code']}[r]")
        else:
            body = [f"o = {t['step_tile']}[s]", f"code = {t['step_code']}[s]"]
        body += [f"c = code >> {COLUMN_SHIFT}", f"first = (code & {FIRST}) != 0", f"last = (code & {LAST}) != 0"]
        if self.P <= 1:
            k = sched.phase_steps[0] if self.P else 0
            body += ["ph = c * 0", "d = c", f"empty = d == {k}" if self.P else "empty = jnp.bool_(False)"]
        else:
            offsets = sched.column_offsets()
            body.append("ph = " + " + ".join(f"(c >= {off}).astype(jnp.int32)" for off in offsets[1:]))
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

    # ------------------------------------------------------------------ copies

    def _copy_fn(self, op: Operand) -> str:
        return f"_{op.ref}_copy"

    def _copies(self) -> list[str]:
        lines = []
        for op in self.spec.operands:
            src = f"{self.hbm[op.param]}.at[{self._slices(op.block, 'o', 'd')}]"
            dst = f"{self.slots[op.ref]}.at[slot], {self.sems[op.ref]}.at[slot]"
            body = [f"src = {src}  # {op.param}[{op.block.text}]", f"return pltpu.make_async_copy(src, {dst})"]
            lines += [*_fn(self._copy_fn(op), "o, d, slot", body), ""]
        for out in self.spec.outputs:
            dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, 'o', '0')}]"
            src = f"{self.out_slots[out.var]}.at[slot], dst, {self.out_sems[out.var]}.at[slot]"
            body = [f"dst = {dst}  # {out.var}[{out.block.text}]", f"return pltpu.make_async_copy({src})"]
            lines += [*_fn(f"_{out.ref}_copy", "o, slot", body), ""]
            if out.var in self.fills:
                fill = self.fills[out.var]
                dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, '0', '0', fill='c')}]"
                body = [f"dst = {dst}", f"return pltpu.make_async_copy({fill}, dst, {fill}_sem)"]
                lines += [*_fn(f"_{out.var}_fill_copy", "c", body), ""]
        return lines

    # ------------------------------------------------------------------ user code -> kernel code

    def _rewrite(self, node: ast.AST, phase: int | None) -> ast.AST:
        """Inputs become their tile refs, state variables their scratch refs, `@` an MXU matmul."""
        spec, params, state = self.spec, self.nest.params, self.spec.state

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
                return ast.copy_location(ast.Subscript(ast.Name(sv.ref, ast.Load()), index, n.ctx), n)

            def visit_BinOp(self, n: ast.BinOp) -> ast.AST:
                self.generic_visit(n)
                if isinstance(n.op, ast.MatMult):
                    right, fn = n.right, "_mxu_dot"
                    if isinstance(right, ast.Attribute) and right.attr == "T":  # x @ y.T
                        right, fn = right.value, "_mxu_dot_t"
                    return ast.copy_location(ast.Call(ast.Name(fn, ast.Load()), [n.left, right], []), n)
                return n

        return Rewriter().visit(copy.deepcopy(node))

    def _statements(self, stmts: Sequence[ast.stmt], phase: int | None) -> list[str]:
        lines = []
        for original in stmts:
            s, target = original, assigned_name(original)
            if isinstance(s, ast.AugAssign) and target in self.spec.state:  # x += v  ->  x = x + v, on the ref
                value = ast.BinOp(ast.Name(target, ast.Load()), s.op, s.value)
                s = ast.copy_location(ast.Assign([ast.Name(target, ast.Store())], value), s)
            out = self._rewrite(s, phase)
            cast = self.spec.casts.get(id(original))
            if cast is not None and isinstance(out, ast.Assign):
                out.value = ast.parse(f"({ast.unparse(out.value)}).astype({_dtype_src(cast)})", mode="eval").body
            lines.append(ast.unparse(out))
        return lines

    def _stores(self) -> list[str]:
        v, lines = self.v, []
        for out in self.spec.outputs:
            store = next(st for st in self.nest.stores if st.out == out.var)
            value = ast.unparse(self._rewrite(store.value, None))
            copy_fn = f"_{out.ref}_copy"
            lines += _fn(
                f"_reuse_{out.ref}_slot",  # the copy that last used this slot must have left
                "",
                [f"{copy_fn}({v['o']} - 2, {v['o']} & 1).wait()"],
                when=f"{v['o']} - 2 >= {v['o_begin']}",
            )
            lines += [
                f"{out.ref} = {self.out_slots[out.var]}.at[{v['o']} & 1]",
                f"{out.ref}[...] = jnp.asarray({value}).astype({out.ref}.dtype)",
                f"{copy_fn}({v['o']}, {v['o']} & 1).start()",
            ]
        return lines

    def _units(self) -> list[str]:
        """The user's code for one step, guarded by the step's phase and flags."""
        v, P, units = self.v, self.P, self.spec.units
        segments = [u for u in units if u.kind == "segment"]
        bodies = [u for u in units if u.kind == "body"]
        fill_issue = self._fill_issue()
        final = self._statements(segments[-1].statements, None) + self._stores()
        if P == 0:
            return fill_issue + final
        lines = []
        for p in range(P):
            lead = (fill_issue if p == 0 else []) + self._statements(segments[p].statements, None)
            if lead:
                lines += _fn(f"_segment_{p}", "", lead, when=f"({v['ph']} == {p}) & {v['first']}")
            body = self._statements(bodies[p].statements, p)
            lines += _fn(f"_body_{p}", "", body, when=f"({v['ph']} == {p}) & ~{v['empty']}")
        return lines + _fn("_finish_tile", "", final, when=f"({v['ph']} == {P - 1}) & {v['last']}")

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

    def _fetch_ahead(self, step: str, name: str) -> list[str]:
        """Start copying step `step`'s streamed inputs into their ring slots."""
        N = self.buffers
        fetches = [
            line
            for op in self.streamed
            for line in _fn(
                f"_fetch_{op.ref}",
                "",
                [f"{self._copy_fn(op)}(o2, d2, {_slot(step, N)}).start()"],
                when=f"(ph2 == {op.phase}) & ~empty2",
            )
        ]
        body = [f"o2, ph2, d2, _, _, empty2 = _decode({step})", *fetches]
        return _fn(name, "", body, when=f"{step} < {self.v['s_end']}")

    def _step(self) -> list[str]:
        v, N = self.v, self.buffers
        lines = [f"{v['o']}, {v['ph']}, {v['d']}, {v['first']}, {v['last']}, {v['empty']} = _decode({v['s']})"]
        if self.streamed:  # 1. prefetch the step N - 1 ahead
            lines += self._fetch_ahead(f"{v['s']} + {N - 1}", "_prefetch")
        if self.per_tile:  # 2. per-tile inputs, at a tile's first step
            starts_tile = f"{v['first']} & ({v['ph']} == 0)" if self.P else "True"
            next_tile = [f"{self._copy_fn(op)}({v['o']} + 1, 0, ({v['o']} + 1) & 1).start()" for op in self.per_tile]
            body = [
                *_fn("_next_tile_inputs", "", next_tile, when=f"{v['o']} + 1 < {v['o_end']}"),
                *[f"{self._copy_fn(op)}({v['o']}, 0, {v['o']} & 1).wait()" for op in self.per_tile],
            ]
            lines += _fn("_tile_inputs", "", body, when=starts_tile)
        for op in self.streamed:  # 3. this step's inputs
            wait = [f"{self._copy_fn(op)}({v['o']}, {v['d']}, {_slot(v['s'], N)}).wait()"]
            lines += _fn(f"_wait_{op.ref}", "", wait, when=f"({v['ph']} == {op.phase}) & ~{v['empty']}")
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{_slot(v['s'], N)}]" for op in self.streamed]
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{v['o']} & 1]" for op in self.per_tile]
        units = self._units()  # 4. the user's code
        used = {n.id for n in ast.walk(ast.parse("\n".join(units))) if isinstance(n, ast.Name)}
        return lines + self._loop_values(used) + units

    def _kernel(self, name: str) -> list[str]:
        spec, v, N = self.spec, self.v, self.buffers
        runtime = [self.step_tables[k] for k in _RUNTIME_TABLES] if self.sched.mode == "runtime" else []
        args = [
            *self.tables,
            *runtime,
            *self.hbm.values(),
            *self.aliased.values(),
            *self.out_hbm.values(),
            *[x for op in spec.operands for x in (self.slots[op.ref], self.sems[op.ref])],
            *[x for o in spec.outputs for x in (self.out_slots[o.var], self.out_sems[o.var])],
            *[sv.ref for sv in spec.state.values()],
            *[x for fill in self.fills.values() for x in (fill, f"{fill}_sem")],
        ]
        body = [*self._bounds(), "", *self._decode(), "", *self._copies()]
        for out in spec.outputs:
            if out.var in self.fills:
                fill = self.fills[out.var]
                body.append(f"{fill}[...] = jnp.full({fill}.shape, {_scalar_src(out.fill)}, {fill}.dtype)")
        body += ["", "# Fetch the first steps' inputs."]
        if self.per_tile:
            first = [f"{self._copy_fn(op)}({v['o_begin']}, 0, {v['o_begin']} & 1).start()" for op in self.per_tile]
            body += _fn("_first_tile_inputs", "", first, when=f"{v['o_begin']} < {v['o_end']}")
        if self.streamed:
            for k in range(N - 1):
                body += self._fetch_ahead(f"{v['s_begin']} + {k}", f"_prefetch_{k}")
        body += ["", *_fn("_steps", v["s"], self._step(), loop=f"{v['s_begin']}, {v['s_end']}")]
        body += ["", "# Wait for the copies still in flight."]
        for out in spec.outputs:
            for back in (2, 1):
                wait = [f"_{out.ref}_copy({v['o_end']} - {back}, ({v['o_end']} - {back}) & 1).wait()"]
                body += _fn(f"_drain_{out.ref}_{back}", "", wait, when=f"{v['o_end']} - {back} >= {v['o_begin']}")
        body += self._fill_drain()
        return [f"def {name}({', '.join(args)}):", *_indent(body)]

    # ------------------------------------------------------------------ wrapper

    def _scratch(self) -> list[str]:
        spec, N, scratch = self.spec, self.buffers, []
        for op in spec.operands:
            depth = N if op.phase is not None else 2
            shape = (depth, *op.block.kernel_shape)
            scratch += [
                f"pltpu.VMEM({shape}, {_dtype_src(op.dtype)}),  # {op.param}[{op.block.text}]",
                f"pltpu.SemaphoreType.DMA(({depth},)),",
            ]
        for out in spec.outputs:
            shape = (2, *out.block.kernel_shape)
            scratch += [
                f"pltpu.VMEM({shape}, {_dtype_src(out.out.dtype)}),  # {out.var}[{out.block.text}]",
                "pltpu.SemaphoreType.DMA((2,)),",
            ]
        for sv in spec.state.values():
            dtype = _dtype_src(sv.aval.dtype)
            mem = f"pltpu.SMEM((1,), {dtype})" if sv.smem else f"pltpu.VMEM({tuple(sv.aval.shape)}, {dtype})"
            scratch.append(f"{mem},  # {sv.name}")
        for out in spec.outputs:
            if out.var in self.fills:
                shape = out.block.kernel_shape
                scratch += [f"pltpu.VMEM({shape}, {_dtype_src(out.out.dtype)}),", "pltpu.SemaphoreType.DMA(()),"]
        return scratch

    def _wrapper(self, kernel: str) -> list[str]:
        spec, nest, sched = self.spec, self.nest, self.sched
        smem, anyspace = "pl.BlockSpec(memory_space=pltpu.SMEM)", "pl.BlockSpec(memory_space=pl.ANY)"
        n_smem = len(self.tables) + (len(_RUNTIME_TABLES) if sched.mode == "runtime" else 0)
        in_specs = [smem] * n_smem + [anyspace] * (len(self.hbm) + len(self.aliased))
        outs = [f"jax.ShapeDtypeStruct({tuple(o.out.shape)}, {_dtype_src(o.out.dtype)})" for o in spec.outputs]
        out_shape = f"({', '.join(outs)},)" if nest.returns_tuple else outs[0]
        out_specs = f"({', '.join([anyspace] * len(outs))},)" if nest.returns_tuple else anyspace

        lines = [f"def {nest.fn_name}_pallas({', '.join(nest.params)}, *, interpret=False):"]
        lines.append(f"    # No grid: one loop over {self._describe()}; inputs {self.buffers}-buffered by hand.")
        args = [t.upper() for t in self.tables]
        if sched.mode == "runtime":
            lines.append("    # A mask reads inputs, so the steps are listed on device.")
            lines.append(f"    tables = _runtime_tables({', '.join(f'{p}={p}' for p in nest.params)})")
            args += [f"tables[{k!r}]" for k in _RUNTIME_TABLES]
        args += list(self.hbm)
        for var, init in self.aliased.items():
            alloc = next(o.alloc for o in spec.outputs if o.var == var)
            lines.append(f"    {init} = {ast.unparse(alloc)}  # skipped tiles keep these values")
            args.append(init)
        aliases = {len(args) - len(self.aliased) + n: nest.out_vars.index(var) for n, var in enumerate(self.aliased)}
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

    def _describe(self) -> str:
        sched, spec = self.sched, self.spec
        total = int(np.prod([lb.count for lb in spec.parallel], dtype=np.int64))
        tiles = f"{sched.tiles} tiles" + (f" (of {total})" if sched.tiles != total else "")
        if sched.mode == "runtime":
            return f"the steps of up to {tiles} that the masks keep at run time"
        cores = f", split over {self.cores} TensorCores" if self.cores > 1 else ""
        return f"{sched.steps} steps of {tiles}{cores}"

    # ------------------------------------------------------------------ assembly

    def emit(self) -> Generated:
        spec, nest = self.spec, self.nest
        kernel_name = spec.names.fresh(f"{nest.fn_name}_kernel")
        kernel = self._kernel(kernel_name)
        wrapper = self._wrapper(kernel_name)
        inputs = ", ".join(
            f"{p}: {_dtype_src(s.dtype)}{tuple(s.shape)}" for p, s in zip(nest.params, spec.specs, strict=True)
        )
        header = f"# Generated by autokernel from `{nest.fn_name}` ({spec.filename}:{nest.first_line}) for {inputs}"
        if spec.overrides:
            header += "\n# Tuned: " + ", ".join(f"{k}={v}" for k, v in sorted(spec.overrides.items()))
        prelude = _PRELUDE.format(precision=MATMUL_PRECISIONS[self.precision])
        source = "\n".join([header, "", prelude, "", *kernel, "", "", *wrapper, ""])

        namespace: dict[str, Any] = dict(spec.consts)
        namespace.update(jax=jax, jnp=jnp, pl=pl, pltpu=pltpu, _COST=self.cost)
        namespace.update({name.upper(): value for name, value in self.tables.items()})
        if self.sched.mode == "runtime":
            namespace["_runtime_tables"] = lambda **inputs: schedule_lib.runtime_tables(spec, self.sched, inputs)
        exec(compile(source, f"<autokernel:{nest.fn_name}>", "exec"), namespace)
        return Generated(source, namespace[f"{nest.fn_name}_pallas"], dict(self.tables))
