"""Lower a parsed loop nest, for concrete input shapes, into a compiled Pallas kernel.

`lower()` runs the pipeline: specialise (analysis) -> check TPU layout rules -> schedule the steps ->
estimate cost and VMEM -> generate and compile the Pallas source. `plan()` stops before code generation,
which is all the tuner needs to rank block sizes; `build()` finishes a plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from . import codegen, hardware, mxu
from . import cost as cost_lib
from . import schedule as schedule_lib
from .analysis import Operand, OutputSpec, Specialization, StateVar, check_indices, specialize
from .errors import AutoKernelError
from .frontend import LoopNest

__all__ = [
    "AUTO",
    "AUTOTUNE",
    "DEFAULT_BUFFERS",
    "Lowered",
    "Options",
    "Plan",
    "VmemError",
    "build",
    "lower",
    "plan",
]

AUTOTUNE = ("model", "measure")
# Slots per streamed input when none is asked for: three where they fit in VMEM (on a v5e, three
# buffers ran a 512^3-tiled bf16 matmul 14% faster than two), else two.
DEFAULT_BUFFERS = (3, 2)
# `tune` value asking for candidates to be generated, and the name standing for every block size the
# kernel's loops use (`tune="auto"`).
AUTO = "auto"
ALL_NAMES = "*"

TuneValues = tuple[Any, ...] | str  # candidate values, or AUTO


class VmemError(AutoKernelError):
    """The kernel's buffers don't fit in the chip's VMEM (bigger tiles never would: the tuner prunes them)."""


@dataclass(frozen=True)
class Options:
    """How kernels are generated; see `autokernel.tpu` for what each option means."""

    check_tiling: bool = True
    vmem_limit_bytes: int | None = None
    matmul_precision: str = "highest"
    background_fill: bool = True
    buffers: int | None = None
    target: str | None = None
    cores: int | None = None
    tune: tuple[tuple[str, TuneValues], ...] = ()
    autotune: str = "model"
    measure_top: int = 8

    def __post_init__(self) -> None:
        if self.matmul_precision not in mxu.MATMUL_PRECISIONS:
            raise AutoKernelError(
                f"matmul_precision must be one of {sorted(mxu.MATMUL_PRECISIONS)}, got {self.matmul_precision!r}"
            )
        if self.buffers is not None and self.buffers < 2:
            raise AutoKernelError(f"buffers must be at least 2, got {self.buffers}")
        if self.cores is not None and self.cores < 1:
            raise AutoKernelError(f"cores must be at least 1, got {self.cores}")
        if self.autotune not in AUTOTUNE:
            raise AutoKernelError(f"autotune must be one of {list(AUTOTUNE)}, got {self.autotune!r}")
        for name, values in self.tune:
            if isinstance(values, str) and values != AUTO:
                raise AutoKernelError(f"tune[{name!r}] must be {AUTO!r} or a sequence of values, got {values!r}")
            if not values:
                raise AutoKernelError(f"tune[{name!r}] has no candidate values")
        if self.target is not None:
            hardware.target(self.target)  # unknown targets fail here, at decoration time

    @staticmethod
    def make_tune(tune: Mapping[str, Sequence[Any] | str] | str | None) -> tuple[tuple[str, TuneValues], ...]:
        """`tune` as given to `autokernel.tpu`: `"auto"`, or a mapping from names to values or `"auto"`."""
        if isinstance(tune, str):
            if tune != AUTO:
                raise AutoKernelError(f"tune must be {AUTO!r} or a mapping of names to candidate values, got {tune!r}")
            return ((ALL_NAMES, AUTO),)
        if tune is not None and not isinstance(tune, Mapping):
            raise AutoKernelError(f"tune must be {AUTO!r} or a mapping of names to candidate values, got {tune!r}")
        return tuple((k, v if isinstance(v, str) else tuple(v)) for k, v in sorted((tune or {}).items()))


@dataclass(frozen=True)
class Lowered:
    """A kernel specialised to one set of input shapes/dtypes."""

    source: str  # the generated Pallas code
    entry: Callable[..., Any]  # entry(*inputs, interpret=...) -> output(s)
    schedule: str  # how steps are decoded: "uniform", "table" or "runtime"
    steps: int  # steps the kernel runs (an upper bound with runtime masks)
    tiles: int  # output tiles computed (candidates with a runtime tile mask)
    total_tiles: int  # output tiles without any mask
    cores: int  # TensorCores the work is split across
    core_steps: list[tuple[int, int]]  # each core's [begin, end) steps (empty with runtime masks)
    buffers: int  # VMEM slots per streamed input
    operands: list[Operand]
    outputs: list[OutputSpec]
    state: dict[str, StateVar]
    estimate: cost_lib.Estimate
    vmem_limit_bytes: int
    fill: str | None  # skipped tiles' values: "background" (fill DMAs), "prefill" (aliased input) or None
    overrides: dict[str, Any] = field(default_factory=dict)  # tuned constants this kernel was built with
    scalars: list[str] = field(default_factory=list)  # runtime scalar inputs (0-d integers, read from SMEM)
    plan: Plan | None = field(default=None, repr=False, compare=False)  # what it was generated from
    user_vmem_limit: int | None = None  # the `vmem_limit_bytes` the user set (None: the chip's limit)

    def check_indices(self, values: Mapping[str, Any]) -> None:
        """Raise AutoKernelError if known values of the runtime scalars index outside an input (traced ones are
        clamped by the kernel, and checked under `jax.experimental.checkify`)."""
        if self.plan is not None:  # only integers index: a float scalar (inf, NaN, ...) is just a value
            spec = self.plan.spec
            ints = {p for p in spec.scalars if jnp.issubdtype(spec.input_dtype(p), jnp.integer)}
            known = {k: v if isinstance(v, jax.core.Tracer) else int(v) for k, v in values.items() if k in ints}
            check_indices(spec, known)  # traced values: a checkify check


@dataclass(frozen=True)
class Plan:
    """A kernel specialised, checked, scheduled and costed, but not yet generated."""

    spec: Specialization
    sched: schedule_lib.Schedule
    hw: hardware.Hardware
    cores: int
    buffers: int
    fills: list[str]
    aliased: list[str]
    estimate: cost_lib.Estimate
    overrides: dict[str, Any]


def lower(
    nest: LoopNest,
    fn: Callable,
    specs: Sequence[jax.ShapeDtypeStruct],
    options: Options,
    overrides: Mapping[str, Any] | None = None,
    buffers: int | None = None,
) -> Lowered:
    return build(plan(nest, fn, specs, options, overrides, buffers), options)


def plan(
    nest: LoopNest,
    fn: Callable,
    specs: Sequence[jax.ShapeDtypeStruct],
    options: Options,
    overrides: Mapping[str, Any] | None = None,
    buffers: int | None = None,
    cost_cache: dict | None = None,
) -> Plan:
    """Everything up to code generation; raises AutoKernelError if the kernel can't run on the target.

    `cost_cache` (any dict, reused across calls) memoises the per-unit cost analysis between plans of
    the same nest that differ only in some constants, which is what makes tuning fast.
    """
    spec = specialize(nest, fn, specs, dict(overrides or {}))
    hw = hardware.target(options.target)
    if options.check_tiling and (problems := codegen.check_layout(spec, hw.lanes)):
        raise codegen.layout_error(spec, problems)
    cores = options.cores or hw.cores
    sched = schedule_lib.plan(spec, cores, hw.smem_budget)
    fills, aliased = codegen.plan_fills(spec, options.background_fill)
    if fills:  # the fill tables live in SMEM too
        schedule_lib.check_smem(spec, sched, True, hw.smem_budget)
    highest = options.matmul_precision == "highest"
    # What the kernel may plan to use: the chip's budget, or the VMEM limit the user set (Mosaic can't give
    # the kernel more than its limit).
    budget = min(options.vmem_limit_bytes, hw.vmem_bytes) if options.vmem_limit_bytes else hw.vmem_budget
    if buffers or options.buffers:
        buffers = buffers or options.buffers
        est = cost_lib.estimate(spec, sched, hw, buffers, highest, fills, cost_cache)
    else:
        for buffers in DEFAULT_BUFFERS:  # the deepest default that fits
            est = cost_lib.estimate(spec, sched, hw, buffers, highest, fills, cost_cache)
            if est.vmem_bytes <= budget:
                break
    if est.vmem_bytes > budget:
        if options.vmem_limit_bytes:
            room = f"its vmem_limit_bytes ({budget / 2**20:.1f} MiB)"
        else:
            room = f"the {budget / 2**20:.0f} MiB a {hw.name} kernel can use"
        raise VmemError(
            f"{nest.fn_name}: needs about {est.vmem_bytes / 2**20:.0f} MiB of VMEM, more than {room}; use smaller "
            "tiles or fewer buffers"
        )
    return Plan(spec, sched, hw, cores, buffers, fills, aliased, est, dict(overrides or {}))


def build(p: Plan, options: Options) -> Lowered:
    """Generate and compile the Pallas kernel for a plan."""
    est = p.estimate
    limit = options.vmem_limit_bytes or p.hw.vmem_limit
    generated = codegen.generate(
        p.spec,
        p.sched,
        buffers=p.buffers,
        matmul_precision=options.matmul_precision,
        fills=p.fills,
        aliased=p.aliased,
        vmem_limit_bytes=limit,
        cost=est.cost_estimate(),
    )
    return Lowered(
        source=generated.source,
        entry=generated.entry,
        schedule=p.sched.mode,
        steps=p.sched.steps,
        tiles=p.sched.tiles,
        total_tiles=p.spec.total_tiles,
        cores=p.cores,
        core_steps=list(p.sched.core_steps),
        buffers=p.buffers,
        operands=p.spec.operands,
        outputs=p.spec.outputs,
        state=p.spec.state,
        estimate=est,
        vmem_limit_bytes=limit,
        fill="prefill" if p.aliased else "background" if p.fills else None,
        overrides=dict(p.overrides),
        scalars=list(p.spec.scalars),
        plan=p,
        user_vmem_limit=options.vmem_limit_bytes,
    )
