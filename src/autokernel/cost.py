"""A roofline cost model for generated kernels, built from JAX's own cost analysis and the TPU's specs.

Each unit of the user's code (a segment, a reduction body, the final stores) is traced on tile-shaped
inputs. Its jaxpr gives the matmul FLOPs that run on the MXU, the other elementwise work and the
transcendentals; where XLA's own `cost_analysis()` is available before compilation (a CPU-only JAX),
its FLOP and transcendental counts replace the jaxpr's. With the hardware rates from `hardware.target()`:

    step time  = max(compute, input DMA) + per-step overhead
    tile time  = max(segment and store compute, per-tile input and output DMA)
    total      = (sum over steps + sum over tiles) / TensorCores

The model ranks block sizes for the tuner; it also sizes the kernel's VMEM limit and feeds XLA a
`pl.CostEstimate` for the kernel.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from .analysis import Specialization, Unit
from .frontend import assigned_name
from .hardware import Hardware, nbytes
from .schedule import Schedule

__all__ = ["STEP_OVERHEAD_S", "Estimate", "estimate"]

# Fixed cost of one step of the generated loop (decode, DMA issue and waits), measured on a v5e.
STEP_OVERHEAD_S = 0.25e-6


@dataclass(frozen=True)
class Estimate:
    seconds: float
    flops: float
    transcendentals: float
    hbm_bytes: float
    vmem_bytes: int

    def cost_estimate(self) -> pl.CostEstimate:
        return pl.CostEstimate(
            flops=int(self.flops), transcendentals=int(self.transcendentals), bytes_accessed=int(self.hbm_bytes)
        )


@dataclass(frozen=True)
class _UnitCost:
    mxu_flops: float
    other_flops: float  # elementwise and reductions, on the VPU
    transcendentals: float
    matmul_dtype: Any
    intermediate_bytes: int


def estimate(spec: Specialization, sched: Schedule, hw: Hardware, buffers: int, highest: bool) -> Estimate:
    segments = [u for u in spec.units if u.kind == "segment"]
    bodies = [u for u in spec.units if u.kind == "body"]
    seg_costs = [_unit_cost(spec, u, None) for u in segments]
    body_costs = [_unit_cost(spec, u, u.index) for u in bodies]
    store_cost = _unit_cost(spec, None, None)

    tiles = sched.tiles
    body_steps = _body_steps(spec, sched)

    def compute_s(c: _UnitCost) -> float:
        mxu = c.mxu_flops / hw.matmul_flops_per_s(c.matmul_dtype, highest) if c.mxu_flops else 0.0
        return mxu + (c.other_flops + c.transcendentals) / hw.vpu_ops_per_s

    seconds = 0.0
    flops = transcendentals = hbm = 0.0
    for p, (cost, steps) in enumerate(zip(body_costs, body_steps, strict=True)):
        streamed = sum(nbytes(op.block.shape, op.dtype) for op in spec.operands if op.phase == p)
        seconds += steps * (max(compute_s(cost), streamed / hw.hbm_bytes_per_s) + STEP_OVERHEAD_S)
        flops += steps * (cost.mxu_flops + cost.other_flops)
        transcendentals += steps * cost.transcendentals
        hbm += steps * streamed
    tile_bytes = sum(nbytes(op.block.shape, op.dtype) for op in spec.operands if op.phase is None)
    tile_bytes += sum(nbytes(o.block.shape, o.out.dtype) for o in spec.outputs)
    tile_compute = sum(compute_s(c) for c in [*seg_costs, store_cost])
    seconds += tiles * (max(tile_compute, tile_bytes / hw.hbm_bytes_per_s) + (0 if spec.phases else STEP_OVERHEAD_S))
    flops += tiles * sum(c.mxu_flops + c.other_flops for c in [*seg_costs, store_cost])
    transcendentals += tiles * sum(c.transcendentals for c in [*seg_costs, store_cost])
    hbm += tiles * tile_bytes
    hbm += len(spec.constant_tiles) * sum(nbytes(o.block.shape, o.out.dtype) for o in spec.outputs)
    seconds /= sched.cores

    vmem = sum(
        (buffers if op.phase is not None else 2) * nbytes(op.block.kernel_shape, op.dtype) for op in spec.operands
    )
    vmem += sum(3 * nbytes(o.block.kernel_shape, o.out.dtype) for o in spec.outputs)  # two slots and a fill tile
    vmem += sum(nbytes(sv.aval.shape, sv.aval.dtype) for sv in spec.state.values())
    vmem += max(c.intermediate_bytes for c in [*seg_costs, *body_costs, store_cost])
    return Estimate(seconds, flops, transcendentals, hbm, int(vmem))


def _body_steps(spec: Specialization, sched: Schedule) -> list[int]:
    """Non-empty steps of each phase, over the whole kernel (an upper bound for runtime masks)."""
    if sched.mode == "table":
        reps = sched.steps // sched.pattern_steps
        codes = sched.tables["step_code"] >> 2
        columns = sched.columns
        counts = [0] * len(spec.phases)
        for c in codes:
            col = columns[int(c)]
            if not col.empty:
                counts[col.phase] += 1
        return [n * reps for n in counts]
    return [sched.tiles * k for k in sched.phase_steps]


# --------------------------------------------------------------------------- per-unit costs


def _unit_cost(spec: Specialization, unit: Unit | None, phase: int | None) -> _UnitCost:
    """Trace a unit (None: the output stores) on tile-shaped values and ask JAX what it costs."""
    nest = spec.nest
    stmts = list(unit.statements) if unit is not None else []
    exprs = [st.value for st in nest.stores] if unit is None else []
    if not stmts and not exprs:
        return _UnitCost(0, 0, 0, jnp.bfloat16, 0)

    operands = {op.ref: op for op in spec.operands}
    state = spec.state

    class Placeholders(ast.NodeTransformer):
        def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
            if isinstance(n.value, ast.Name) and n.value.id in nest.params:
                return ast.copy_location(ast.Name(spec.operand(n, phase).ref, ast.Load()), n)
            return self.generic_visit(n)

    body = [Placeholders().visit(copy.deepcopy(s)) for s in stmts]
    results = [ast.Name(assigned_name(s), ast.Load()) for s in stmts]
    results += [Placeholders().visit(copy.deepcopy(e)) for e in exprs]
    loop_vars = sorted(spec.loops)
    args = [*operands, *state, *loop_vars]
    fn_def = ast.FunctionDef(
        name="_unit",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(a) for a in args], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*body, ast.Return(ast.Tuple(results, ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn_def], type_ignores=[]))
    namespace = dict(spec.consts)
    exec(compile(module, f"<autokernel cost:{nest.fn_name}>", "exec"), namespace)
    fn = namespace["_unit"]
    avals = [jax.ShapeDtypeStruct(op.block.kernel_shape, op.dtype) for op in operands.values()]
    avals += [sv.aval for sv in state.values()]
    avals += [jax.ShapeDtypeStruct((), jnp.int32)] * len(loop_vars)
    try:
        jaxpr = jax.make_jaxpr(fn)(*avals)
    except Exception:  # can't trace on tile shapes: count nothing rather than fail planning
        return _UnitCost(0, 0, 0, jnp.bfloat16, 0)
    mxu, dtype, inter, elementwise, transcendental = 0.0, jnp.bfloat16, 0, 0.0, 0.0
    for eqn in jaxpr.jaxpr.eqns:
        size = 0
        for out in eqn.outvars:
            aval = out.aval
            if hasattr(aval, "shape"):
                size += int(np.prod(aval.shape, dtype=np.int64))
                if len(aval.shape) >= 1:
                    inter += nbytes(aval.shape, aval.dtype)
        if eqn.primitive.name == "dot_general":
            (contracting, _), _ = eqn.params["dimension_numbers"]
            x, y = (a.aval for a in eqn.invars)
            contract = np.prod([x.shape[d] for d in contracting], dtype=np.int64)
            mxu += 2.0 * np.prod(eqn.outvars[0].aval.shape, dtype=np.int64) * contract
            dtype = jnp.result_type(x.dtype, y.dtype)
        elif eqn.primitive.name in _TRANSCENDENTAL:
            transcendental += size
        else:
            elementwise += size
    analysis = _cpu_cost_analysis(fn, avals)
    if analysis is not None:  # JAX's own count, when the CPU backend can give one
        elementwise = max(0.0, float(analysis.get("flops", 0.0)) - mxu)
        transcendental = float(analysis.get("transcendentals", transcendental))
    return _UnitCost(mxu, elementwise, transcendental, dtype, inter)


_TRANSCENDENTAL = frozenset(
    {"exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "sin", "cos", "rsqrt", "sqrt", "pow", "erf", "atan2"}
)


def _cpu_cost_analysis(fn: Any, avals: list[jax.ShapeDtypeStruct]) -> dict[str, float] | None:
    """XLA's HLO cost analysis of `fn` before compilation, or None where JAX can't give it.

    A CPU-only JAX computes it on lowering. With a TPU backend attached it's only available after
    compiling, which planning can't afford, so the jaxpr counts are used instead.
    """
    try:
        analysis = jax.jit(fn).trace(*avals).lower(lowering_platforms=("cpu",)).cost_analysis()
    except Exception:
        return None
    analysis = analysis[0] if isinstance(analysis, list) else analysis
    return analysis if isinstance(analysis, dict) else None
