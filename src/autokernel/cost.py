"""A cost model for generated kernels: per-step roofline timing from the traced tile code and the TPU's specs.

Each unit of the user's code (a segment, a reduction body, the final stores) is traced on tile-shaped
inputs. Its jaxpr gives the matmul FLOPs per dtype, which run on the MXU, and the vector work, counted in
whole-vreg operations (a vreg holds 8 x 128 32-bit values, or 16 x 128 packed 16-bit ones on chips whose
VPU computes on them): elementwise ops on the VALU slots, transcendentals on the EUP, reductions with
their cross-lane steps. Mosaic schedules the MXU, VALU and EUP side by side within a step, so a unit's
compute time is the slowest of them plus a fraction of the rest (dependency stalls), and the DMAs of
streamed tiles overlap the compute of earlier steps:

    step      = max(fixed + compute, input DMA, (latency + input DMA) / (buffers - 1))   (3+ buffers)
              = fixed + max(compute, input DMA + latency)                             (2 buffers)
    body      = max(sum of steps + per-tile compute, all HBM traffic / bandwidth) / TensorCores
    total     = body + pipeline fill (the first step's DMAs) + drain (the last tile's write-back) + launch

`fixed` grows with the copies a step starts and waits for. HBM traffic counts every fetch: a tiled input
is fetched again for every tile that reads it (a matmul's activations once per column tile, its weights
once per row tile), so blocks that re-stream big weights for many small row tiles lose to the HBM bound
even when their steps are compute-bound. `launch` is a kernel's own fixed cost inside a jitted program;
`Estimate.seconds` adds a standalone call's (what `autotune="measure"` times), `in_program_s` doesn't.

The constants are calibrated against v5e timings (tests/test_cost.py keeps them): dense and causal 8192^3
matmuls in bf16 and f32 within 5%, smaller ones within 10%, causal flash attention at four block sizes
within 16%, in the measured order; the transformer benchmark's QKV, output, up and down projections at
19 block sizes within 6% (the measured best of each sweep predicted best); and decode-like
weight-streaming kernels inside a `fori_loop` within 9%.

The model also rejects what the chip can't compile that the interpreter would run (vector arithmetic on
integers narrower than the VPU computes on), ranks block sizes for the tuner (`Estimate.score` adds a
mild penalty for VMEM use near the budget), sizes the kernel's VMEM limit (every buffer padded to Mosaic's
(sublanes, 128) tiles, plus the peak live temporaries of the largest unit) and feeds XLA a
`pl.CostEstimate` for the kernel.
"""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import source_info_util
from jax.experimental import pallas as pl

from . import mxu
from .analysis import Specialization, Unit, kernel_scalar_aval
from .frontend import assigned_name
from .hardware import Hardware, nbytes, sublanes, vmem_nbytes
from .schedule import COLUMN_SHIFT, Schedule

__all__ = [
    "CALL_S",
    "LAUNCH_S",
    "STEP_OVERHEAD_S",
    "Estimate",
    "estimate",
    "overhead_bound",
    "slots_nbytes",
    "step_seconds",
]

# Fixed cost of one step of the generated loop: decoding it and the loop itself, plus DMA_ISSUE_S for
# each copy it starts and waits for. Measured on a v5e: 8192^3 bf16 matmuls (two copies a step) with 128^3
# tiles run at 0.23 us per step, with 512^3 tiles at max(DMA, MXU) + 0.2 us.
STEP_BASE_S = 0.15e-6
DMA_ISSUE_S = 0.025e-6
STEP_OVERHEAD_S = STEP_BASE_S + 2 * DMA_ISSUE_S  # a typical step: two streamed copies
# Fixed cost of a kernel inside a jitted program (launch, prologue, the first tables), on top of pipeline
# fill and drain: decode-like weight-streaming kernels in a `fori_loop` on a v5e run within a few us of
# their HBM traffic.
LAUNCH_S = 2e-6
# What a standalone call (`jax.jit(kernel)(...)`, as `autotune="measure"` and the benchmarks time it) adds:
# the executable's own launch. Fitted to the README's v5e timings and the transformer's projections timed
# one kernel at a time.
CALL_S = 12e-6
# Fraction of the nominal HBM bandwidth tile DMAs reach. HBM arrays are stored in (8, 128) tiles, so a
# tile DMA moves whole contiguous 4 KiB tiles whatever its row length; on a v5e, weight-streaming kernels
# in a fori_loop moved all their traffic (activation tiles fetched again included) at 0.92-1.04 of the
# nominal 819 GB/s, and a QKV projection re-streaming its weights for every row tile at 0.95.
HBM_EFFICIENCY = 1.0
# Issue-to-completion latency of a DMA beyond its transfer time. A step's copies are started `buffers - 1`
# steps ahead, so the latency shows when that is less than one step of compute (512^3 bf16 tiles with two
# buffers: 1.80 us per step against 1.54 us with three) or when steps are DMA-bound and short.
DMA_LATENCY_S = 0.3e-6
# Fraction of the MXU's peak rate long matmul steps reach: on a v5e, 8192^3 bf16 runs at 192 of 197
# TFLOP/s, and the transformer's projections (M=8192) within 3% of the same.
MXU_EFFICIENCY = 0.975
# Within a step the MXU, VALU and EUP run in parallel, but dependencies (a softmax waits for its scores)
# leave part of the shorter work exposed: compute = max(mxu, vector) + OVERLAP_LOSS * min(mxu, vector).
OVERLAP_LOSS = 0.25
# Ranking penalty for VMEM use near the budget (Mosaic needs headroom for its own scratch, and huge tiles
# lengthen the pipeline's fill and drain): score = seconds * (1 + VMEM_PENALTY * excess**2), where excess
# grows from 0 at half the budget to 1 at the budget.
VMEM_PENALTY = 0.05


@dataclass(frozen=True)
class Estimate:
    seconds: float  # predicted run time of a standalone call (what `autotune="measure"` times)
    flops: float
    transcendentals: float
    hbm_bytes: float  # every byte the kernel's DMAs move, tiles fetched again for later tiles included
    vmem_bytes: int  # VMEM the kernel uses: buffers padded to (sublanes, 128) tiles, plus peak temporaries
    steps: int = 0  # loop steps the kernel runs (one per tile when there's no reduction), over all cores
    step_work_s: float = 0.0  # mean time of a step's own work (compute or DMA), without the fixed cost
    score: float = 0.0  # what the tuner ranks by: `seconds`, mildly penalised near the VMEM budget
    loop_s: float = 0.0  # the steps' time (fixed costs, compute, DMA waits), over all cores
    hbm_s: float = 0.0  # `hbm_bytes` at the achievable bandwidth: the step loop can't run faster
    fill_s: float = 0.0  # the pipeline's fill: the first copies, with nothing to hide behind
    drain_s: float = 0.0  # the last output copies, after the last step

    @property
    def in_program_s(self) -> float:
        """The run time inside a jitted program (a layer loop, other kernels around it): no call of its own."""
        return self.seconds - CALL_S

    def cost_estimate(self) -> pl.CostEstimate:
        return pl.CostEstimate(
            flops=int(self.flops), transcendentals=int(self.transcendentals), bytes_accessed=int(self.hbm_bytes)
        )


def overhead_bound(est: Estimate, factor: float = 2.0, min_steps: int = 16) -> bool:
    """Do the kernel's steps do too little work to hide their fixed cost?

    True when a step's MXU, vector or DMA time is under `factor` times `STEP_OVERHEAD_S` (and the kernel
    runs at least `min_steps` steps): bigger blocks would run it faster, so an untuned kernel like this is
    worth `tune="auto"`.
    """
    return est.steps >= min_steps and est.step_work_s < factor * STEP_OVERHEAD_S


@dataclass(frozen=True)
class _UnitCost:
    dots: tuple[tuple[str, float], ...]  # MXU FLOPs per operand dtype
    valu_ops: float  # whole-vreg vector-ALU ops
    eup_ops: float  # whole-vreg transcendental-unit ops
    elementwise: float  # element-wise FLOPs, for the cost estimate given to XLA
    transcendentals: float  # transcendental elements, likewise
    peak_bytes: int  # most VMEM the unit's temporaries hold at once
    # Vector arithmetic on integers the VPU can't compute on: (statement or None, primitive, dtype).
    unsupported: tuple[ast.AST | None, str, np.dtype, bool] | None = None  # (where, op, dtype, on vectors?)

    @property
    def mxu_flops(self) -> float:
        return sum(f for _, f in self.dots)


_NO_COST = _UnitCost((), 0.0, 0.0, 0.0, 0.0, 0)


def estimate(
    spec: Specialization,
    sched: Schedule,
    hw: Hardware,
    buffers: int,
    highest: bool,
    fills: Sequence[str] = (),
    cache: dict | None = None,
) -> Estimate:
    """Predicted run time, work and VMEM use; `fills` are the outputs whose skipped tiles get fill DMAs.

    `cache` memoises unit costs across estimates of the same nest: a unit is traced again only when the
    tile shapes or constants it uses change (a tuner varying TK re-traces the reduction body, not the
    stores).
    """
    vpu = (hw.vpu_bf16, hw.vpu_int_bits)
    segments = [u for u in spec.units if u.kind == "segment"]
    bodies = [u for u in spec.units if u.kind == "body"]
    body_costs = [_unit_cost(spec, u, vpu, cache) for u in bodies]
    tile_costs = [_unit_cost(spec, u, vpu, cache) for u in segments] + [_unit_cost(spec, None, vpu, cache)]
    for c in [*body_costs, *tile_costs]:
        if c.unsupported:
            node, name, dtype, vector = c.unsupported
            what = f"`{ast.unparse(node)}`" if node is not None else "the tile code"
            kind, unit = ("vectors", "vector unit") if vector else ("scalars", "scalar unit")
            raise spec.nest.error(
                node,
                f"{what} does `{name}` on {dtype.name} {kind}, but a {hw.name}'s {unit} has no "
                f"{dtype.itemsize * 8}-bit integer arithmetic (Mosaic rejects it when compiling for the chip; the "
                "interpreter doesn't); widen the values first, e.g. `x.astype(jnp.int32)` (8-bit `@` matmuls "
                "are fine: they run on the MXU)",
            )
    bandwidth = hw.hbm_bytes_per_s * HBM_EFFICIENCY

    def compute_s(c: _UnitCost) -> float:
        mxu_s = sum(f / hw.matmul_flops_per_s(np.dtype(d), highest) for d, f in c.dots) / MXU_EFFICIENCY
        vector_s = max(c.valu_ops / hw.valu_ops_per_s, c.eup_ops / hw.eup_ops_per_s)
        return max(mxu_s, vector_s) + OVERLAP_LOSS * min(mxu_s, vector_s)

    flops = transcendentals = hbm = 0.0
    loop_s = work_s = 0.0  # all steps' time, and their work without the fixed cost
    steps = 0
    first_dma = 0.0
    for p, (cost, n) in enumerate(zip(body_costs, _body_steps(spec, sched), strict=True)):
        ops = [op for op in spec.operands if op.phase == p]
        streamed = sum(nbytes(op.block.shape, op.dtype) for op in ops)
        compute, dma = compute_s(cost), streamed / bandwidth
        loop_s += n * step_seconds(compute, dma, len(ops), buffers)
        work_s += n * max(compute, dma)
        steps += n
        if p == 0:
            first_dma = dma
        flops += n * (cost.mxu_flops + cost.elementwise)
        transcendentals += n * cost.transcendentals
        hbm += n * streamed  # a tile fetched again for each later tile that reads it is traffic again
    tiles = sched.tiles
    out_bytes = sum(nbytes(o.block.shape, o.out.dtype) for o in spec.outputs)
    per_tile = [op for op in spec.operands if op.phase is None]
    in_bytes = sum(nbytes(op.block.shape, op.dtype) for op in per_tile)
    tile_compute = sum(compute_s(c) for c in tile_costs)
    if spec.phases:  # per-tile DMAs overlap the tile's steps; the HBM bound below covers them
        loop_s += tiles * tile_compute
    else:  # every tile is one step, its input and output in two slots each
        dma = (in_bytes + out_bytes) / bandwidth
        loop_s += tiles * step_seconds(tile_compute, dma, len(per_tile) + len(spec.outputs), 2)
        work_s += tiles * max(tile_compute, dma)
        steps += tiles
    flops += tiles * sum(c.mxu_flops + c.elementwise for c in tile_costs)
    transcendentals += tiles * sum(c.transcendentals for c in tile_costs)
    hbm += tiles * (in_bytes + out_bytes) + len(spec.constant_tiles) * out_bytes
    cores = max(1, sched.cores)
    hbm_s = hbm / bandwidth
    body_s = max(loop_s, hbm_s) / cores
    # The pipeline's fill (the first tile's inputs and first step's DMAs wait for nothing to hide behind)
    # and drain (the last tile's write-back follows the last step).
    fill_s = DMA_LATENCY_S + in_bytes / bandwidth + first_dma
    drain_s = DMA_LATENCY_S + out_bytes / bandwidth if tiles else 0.0
    seconds = body_s + fill_s + drain_s + LAUNCH_S + CALL_S

    # The kernel's scratch, as codegen lays it out: input rings, two slots per output, a fill tile for
    # outputs written by fill DMAs, array state (scalars live in SMEM), plus the largest unit's peak
    # temporaries; Mosaic pads each buffer to whole (sublanes, 128) tiles.
    vmem = sum(
        slots_nbytes(buffers if op.phase is not None else 2, op.block.kernel_shape, op.dtype) for op in spec.operands
    )
    vmem += sum(slots_nbytes(2, o.block.kernel_shape, o.out.dtype) for o in spec.outputs)
    vmem += sum(vmem_nbytes(o.block.kernel_shape, o.out.dtype) for o in spec.outputs if o.var in fills)
    vmem += sum(vmem_nbytes(sv.aval.shape, sv.aval.dtype) for sv in spec.state.values() if not sv.smem)
    vmem += max(c.peak_bytes for c in [*tile_costs, *body_costs])
    excess = max(0.0, vmem / hw.vmem_budget - 0.5) / 0.5
    score = seconds * (1 + VMEM_PENALTY * excess**2)
    return Estimate(
        seconds,
        flops,
        transcendentals,
        hbm,
        int(vmem),
        steps=steps,
        step_work_s=work_s / steps if steps else 0.0,
        score=score,
        loop_s=loop_s / cores,
        hbm_s=hbm_s / cores,
        fill_s=fill_s,
        drain_s=drain_s,
    )


def slots_nbytes(depth: int, tile: Sequence[int], dtype: Any) -> int:
    """Bytes of a ring of `depth` VMEM slots for `tile`, one `(depth, *tile)` buffer as codegen allocates it:
    for a 1-D tile the slots are the buffer's rows, padded to whole sublane tiles (8 rows of 32-bit values,
    more for packed types), so two slots of an f32 row take eight rows."""
    return vmem_nbytes((depth, *tile), dtype)


def step_seconds(compute: float, dma: float, copies: int, buffers: int) -> float:
    """One step of the loop: its fixed cost and compute, or its DMAs if they take longer.

    With two buffers, a step's copies start one step ahead, once the step before has freed their slots,
    so a copy's latency and transfer must fit in that one step's compute. With more, they start
    `buffers - 1` steps ahead and keep the DMA engines busy through the fixed cost of every step: a step
    then takes its fixed cost and compute, or its share of the copies in flight, or its own transfer.
    """
    fixed = STEP_BASE_S + DMA_ISSUE_S * copies
    if buffers <= 2:
        return fixed + max(compute, dma + DMA_LATENCY_S)
    return max(fixed + compute, dma, (DMA_LATENCY_S + dma) / (buffers - 1))


def _body_steps(spec: Specialization, sched: Schedule) -> list[int]:
    """Non-empty steps of each phase, over the whole kernel (an upper bound for runtime masks)."""
    if sched.mode == "table":
        reps = sched.steps // sched.pattern_steps
        counts = [0] * len(spec.phases)
        for code in sched.tables["step_code"]:
            col = sched.columns[int(code) >> COLUMN_SHIFT]
            if not col.empty:
                counts[col.phase] += 1
        return [n * reps for n in counts]
    return [sched.tiles * k for k in sched.phase_steps]


# --------------------------------------------------------------------------- per-unit costs


def _unit_cost(
    spec: Specialization, unit: Unit | None, vpu: tuple[bool, tuple[int, ...]], cache: dict | None = None
) -> _UnitCost:
    """Trace a unit (None: the output stores) on tile-shaped values and count its MXU and vector work.

    `vpu`: whether the VPU computes on packed 16-bit vregs (else 16-bit values take a 32-bit vreg's 8
    rows), and the integer widths it does arithmetic on (`Hardware.vpu_int_bits`).
    """
    nest = spec.nest
    stmts = list(unit.statements) if unit is not None else []
    exprs = [st.value for st in nest.stores] if unit is None else []
    if not stmts and not exprs:
        return _NO_COST
    phase = unit.phase if unit is not None else None
    operands = {op.ref: op for op in spec.operands}
    state = spec.state
    loop_vars = sorted(spec.loops)
    if cache is not None:
        key = (_unit_key(spec, stmts, exprs, phase), vpu)
        if key not in cache:
            cache[key] = _unit_cost(spec, unit, vpu)
        return cache[key]

    class Placeholders(ast.NodeTransformer):
        def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
            if isinstance(n.value, ast.Name) and n.value.id in nest.params:
                return ast.copy_location(ast.Name(spec.operand(n, phase).ref, ast.Load()), n)
            return self.generic_visit(n)

    body = [Placeholders().visit(mxu.rewrite_matmuls(s)) for s in stmts]
    results = [ast.Name(assigned_name(s), ast.Load()) for s in stmts]
    results += [Placeholders().visit(mxu.rewrite_matmuls(copy.deepcopy(e))) for e in exprs]
    carried = {n: a for n, a in spec.carried.items() if n not in state}  # a single-step tile's earlier units' values
    args = [*operands, *state, *carried, *loop_vars, *spec.scalars]
    fn_def = ast.FunctionDef(
        name="_unit",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(a) for a in args], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*body, ast.Return(ast.Tuple(results, ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn_def], type_ignores=[]))
    namespace = {**spec.consts, **mxu.helpers("highest")}
    exec(compile(module, f"<autokernel cost:{nest.fn_name}>", "exec"), namespace)
    fn = namespace["_unit"]
    avals = [jax.ShapeDtypeStruct(op.block.kernel_shape, op.dtype) for op in operands.values()]
    avals += [sv.aval for sv in state.values()]
    avals += list(carried.values())
    avals += [jax.ShapeDtypeStruct((), jnp.int32, weak_type=True)] * len(loop_vars)  # as Python ints trace
    avals += [kernel_scalar_aval(spec.specs[nest.params.index(p)]) for p in spec.scalars]  # as the kernel holds them
    try:
        jaxpr = jax.make_jaxpr(fn)(*avals).jaxpr
    except Exception:  # can't trace on tile shapes: count nothing rather than fail planning
        return _NO_COST
    packed, int_bits = vpu
    counts = _Counts(packed, int_bits)
    counts.add(jaxpr)
    dots = {}
    for dtype, f in counts.dots:
        dots[dtype] = dots.get(dtype, 0.0) + f
    unsupported = None
    if counts.narrow:
        name, dtype, line, vector = counts.narrow[0]
        nodes = [*stmts, *(st.node for st in nest.stores if unit is None)]
        node = next((n for n in nodes if line is not None and n.lineno <= line <= (n.end_lineno or n.lineno)), None)
        unsupported = (node, name, np.dtype(dtype), vector)
    return _UnitCost(
        tuple(sorted(dots.items())),
        counts.valu,
        counts.eup,
        counts.elementwise,
        counts.transcendentals,
        _peak_live_bytes(jaxpr),
        unsupported,
    )


# --------------------------------------------------------------------------- counting vector work

_TRANSCENDENTAL = frozenset(
    {"exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "sin", "cos", "tan", "rsqrt", "sqrt", "cbrt",
     "pow", "erf", "erfc", "erf_inv", "atan2", "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh",
     "lgamma", "digamma"}
)  # fmt: skip
# Extra VALU ops per vreg around a transcendental's EUP op (range reduction, polynomial tails), and for
# ops the VPU builds from several (division: an EUP reciprocal and Newton steps).
_EXTRA_VALU = {
    "exp": 1, "exp2": 1, "log": 2, "logistic": 2, "tanh": 1, "pow": 3, "div": 2, "rem": 4, "erf": 8, "erfc": 8,
    "erf_inv": 8, "sin": 8, "cos": 8, "tan": 10, "atan2": 10, "lgamma": 20, "digamma": 20,
}  # fmt: skip
_EUP_OPS = {"pow": 2, "div": 1}  # EUP ops per vreg besides the transcendentals' one
# Ops that only move or relabel vregs: tiles are reused, not copied, and scalars splat into their users.
_FREE = frozenset(
    {"reshape", "squeeze", "expand_dims", "tile", "concatenate", "slice", "copy", "copy_p", "pjit_identity",
     "stop_gradient", "optimization_barrier", "convert_element_type_scalar"}
)  # fmt: skip
_REDUCE = frozenset({"reduce_max", "reduce_min", "reduce_sum", "reduce_prod", "reduce_and", "reduce_or",
                     "reduce_xor", "argmax", "argmin"})  # fmt: skip
# Ops the VPU computes (and Mosaic only allows on `Hardware.vpu_int_bits`-wide integers): arithmetic,
# comparisons, bitwise ops, reductions. Converts, selects, loads, stores, relayouts and dots aren't.
_INT_ARITHMETIC = frozenset(
    {"add", "sub", "mul", "div", "rem", "neg", "abs", "sign", "max", "min", "clamp", "integer_pow", "square",
     "pow", "and", "or", "xor", "not", "shift_left", "shift_right_arithmetic", "shift_right_logical",
     "population_count", "clz", "eq", "ne", "lt", "le", "gt", "ge", "add_any", "cumsum", "cumprod", "cummax",
     "cummin", *_REDUCE}
)  # fmt: skip
# Steps of a reduction across a vreg's 128 lanes (log2 of the lanes: rotate and combine) or 8 sublanes.
_LANE_REDUCE_OPS = 7
_SUBLANE_REDUCE_OPS = 3


class _Counts:
    """Accumulates the MXU and vector work of a jaxpr, recursing into nested jaxprs (jit, custom_jvp, ...)."""

    def __init__(self, packed: bool, int_bits: tuple[int, ...] = (8, 16, 32)) -> None:
        self.packed, self.int_bits = packed, int_bits
        self.dots: list[tuple[str, float]] = []
        self.valu = self.eup = self.elementwise = self.transcendentals = 0.0
        self.narrow: list[tuple[str, Any, int | None, bool]] = []  # ops on integers the chip can't compute on

    def check_ints(self, eqn: Any) -> None:
        """Record `eqn` if it's vector arithmetic on integers narrower than the VPU computes on."""
        if eqn.primitive.name not in _INT_ARITHMETIC:
            return
        avals = [a for v in [*eqn.invars, *eqn.outvars] if hasattr(a := getattr(v, "aval", None), "dtype")]
        # An op on vectors runs on the VPU, at its widths (a 0-d operand is broadcast); one on scalars only runs
        # on the scalar unit, which computes in 32 bits. The kernel holds narrow integer runtime scalars as their
        # 32-bit values (analysis checks that's exact), so a narrow scalar here is one the tile code makes, like
        # `jnp.int16(3)`; a 64-bit one is a Python int under x64, which the kernel's trace has too.
        vector = any(getattr(a, "shape", ()) for a in avals)
        for aval in avals:
            dtype = np.dtype(aval.dtype)
            narrow = dtype.itemsize * 8 not in self.int_bits if vector else dtype.itemsize < 4
            if np.issubdtype(dtype, np.integer) and narrow:
                frames = list(source_info_util.user_frames(eqn.source_info.traceback))
                lines = [f.start_line for f in frames if f.file_name.startswith("<autokernel cost:")]
                self.narrow.append((eqn.primitive.name, dtype, lines[0] if lines else None, vector))
                return

    def vregs(self, aval: Any) -> float:
        shape = getattr(aval, "shape", ())
        if not shape:
            return 0.0  # scalars run on the scalar unit
        dtype = np.dtype(aval.dtype)
        if dtype == np.bool_ or dtype.itemsize >= 4 or (dtype.itemsize == 2 and not self.packed):
            rows = 8  # 32-bit vregs (masks come from 32-bit compares; old VPUs unpack 16-bit values)
        else:
            rows = sublanes(dtype)
        shape = (1, *shape) if len(shape) == 1 else shape
        return float(math.prod(shape[:-2]) * -(-shape[-2] // rows) * -(-shape[-1] // 128))

    def add(self, jaxpr: Any, times: float = 1.0) -> None:
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            if subs := _sub_jaxprs(eqn):
                reps = float(eqn.params.get("length", 1)) if name == "scan" else 1.0
                for sub in subs:
                    self.add(sub, times * reps)
                continue
            self.check_ints(eqn)
            ins = [v.aval for v in eqn.invars]
            outs = [v.aval for v in eqn.outvars]
            out_vregs = sum(self.vregs(a) for a in outs)
            in_vregs = sum(self.vregs(a) for a in ins)
            elements = float(sum(math.prod(getattr(a, "shape", ())) for a in outs))
            if name == "dot_general":
                (contracting, _), _ = eqn.params["dimension_numbers"]
                x, y = ins
                contract = math.prod(x.shape[d] for d in contracting)
                dtype = np.dtype(jnp.result_type(x.dtype, y.dtype))
                self.dots.append((dtype.str, times * 2.0 * elements * contract))
                continue
            if name in _REDUCE:
                shape = getattr(ins[0], "shape", ())
                if not shape:
                    continue
                axes = set(eqn.params.get("axes", ()))
                ops = in_vregs  # combine vregs elementwise, then within each vreg:
                if len(shape) - 1 in axes:  # across the lanes of each row group
                    ops += in_vregs / -(-shape[-1] // 128) * _LANE_REDUCE_OPS
                elif len(shape) >= 2 and len(shape) - 2 in axes:  # across the sublanes
                    ops += in_vregs / -(-shape[-2] // 8) * _SUBLANE_REDUCE_OPS
                self.valu += times * ops
                self.elementwise += times * float(math.prod(shape))
                continue
            if not out_vregs or name in _FREE:
                continue
            if name == "broadcast_in_dim" and not getattr(ins[0], "shape", ()):
                continue  # a scalar splat: folded into its users
            ops = max(out_vregs, in_vregs) if name == "convert_element_type" else out_vregs  # (un)packing
            if name in _TRANSCENDENTAL:
                self.eup += times * out_vregs * _EUP_OPS.get(name, 1)
                self.transcendentals += times * elements
                ops = out_vregs * _EXTRA_VALU.get(name, 1)
            else:
                self.eup += times * out_vregs * _EUP_OPS.get(name, 0)
                self.elementwise += times * elements
                ops += out_vregs * _EXTRA_VALU.get(name, 0)
            self.valu += times * ops


def _sub_jaxprs(eqn: Any) -> list[Any]:
    """The jaxprs nested in an equation's parameters (jit, custom_jvp/vjp, cond branches, loop bodies)."""
    found = []
    for value in eqn.params.values():
        for v in value if isinstance(value, (tuple, list)) else (value,):
            if isinstance(v, jax.extend.core.ClosedJaxpr):
                found.append(v.jaxpr)
            elif isinstance(v, jax.extend.core.Jaxpr):
                found.append(v)
    return found


def _peak_live_bytes(jaxpr: Any) -> int:
    """Most VMEM the unit's temporaries hold at once: a value lives from the equation that makes it to its
    last use. Inputs (tiles and state) live in their own buffers, and a matmul whose only use is an add
    accumulates in place (Mosaic fuses `acc + x @ y` into the matmul)."""
    eqns = jaxpr.eqns
    last_use: dict[Any, int] = {}
    uses: dict[Any, list[int]] = {}
    for i, eqn in enumerate(eqns):
        for v in eqn.invars:
            if isinstance(v, jax.extend.core.Var):
                last_use[v] = i
                uses.setdefault(v, []).append(i)
    fused = set()
    for eqn in eqns:
        if eqn.primitive.name == "dot_general":
            out = eqn.outvars[0]
            users = uses.get(out, [])
            if len(users) == 1 and eqns[users[0]].primitive.name == "add":
                fused.add(out)
    ends: dict[int, int] = {}  # equation index -> bytes freed after it
    live = peak = 0
    for i, eqn in enumerate(eqns):
        for v in eqn.outvars:
            aval = v.aval
            if v in fused or not getattr(aval, "shape", ()):
                continue
            size = vmem_nbytes(aval.shape, aval.dtype)
            live += size
            end = last_use.get(v, i)
            ends[end] = ends.get(end, 0) + size
        peak = max(peak, live)
        live -= ends.pop(i, 0)
    return peak


def _unit_key(spec: Specialization, stmts: list[ast.stmt], exprs: list[ast.expr], phase: int | None) -> tuple:
    """What a unit's cost depends on: its statements, and the types of the tiles and state and the values of
    the constants they use."""
    nest = spec.nest
    names: set[str] = set()
    tiles = []
    for tree in [*stmts, *exprs]:
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in nest.params:
                op = spec.operand(n, phase)
                tiles.append((op.ref, op.block.kernel_shape, op.dtype.str))
    state = [(n, sv.aval.shape, np.dtype(sv.aval.dtype).str) for n, sv in spec.state.items() if n in names]
    state += [(n, a.shape, np.dtype(a.dtype).str) for n, a in spec.carried.items() if n in names]
    consts = [
        (n, _freeze(spec.consts[n])) for n in sorted(names - set(spec.state) - set(spec.loops)) if n in spec.consts
    ]
    return (tuple(map(id, stmts or exprs)), tuple(sorted(set(tiles))), tuple(state), tuple(consts))


def _freeze(value: Any) -> Any:
    """A hashable stand-in for a constant in a cache key: its value if it's a plain scalar or tuple of them."""
    if isinstance(value, (bool, int, float, complex, str, bytes, type(None), np.generic)):
        return (type(value), value)
    if isinstance(value, tuple):
        return tuple(map(_freeze, value))
    return ("id", id(value))  # modules, functions, arrays: the same object for the whole search
