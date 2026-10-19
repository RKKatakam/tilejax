"""Lower a parsed loop nest, for concrete input shapes, into a compiled Pallas kernel.

`lower()` runs the pipeline: specialise (analysis) -> check TPU layout rules -> schedule the steps ->
estimate cost and VMEM -> generate and compile the Pallas source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax

from . import codegen, hardware, mxu
from . import cost as cost_lib
from . import schedule as schedule_lib
from .analysis import Operand, OutputSpec, StateVar, specialize
from .errors import AutoKernelError
from .frontend import LoopNest

__all__ = ["AUTOTUNE", "DEFAULT_BUFFERS", "Lowered", "Options", "lower"]

AUTOTUNE = ("model", "measure")
# Slots per streamed input when none is asked for: three where they fit in VMEM (on a v5e, three
# buffers ran a 512^3-tiled bf16 matmul 14% faster than two), else two.
DEFAULT_BUFFERS = (3, 2)


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
    tune: tuple[tuple[str, tuple[Any, ...]], ...] = ()
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
            if not values:
                raise AutoKernelError(f"tune[{name!r}] has no candidate values")
        if self.target is not None:
            hardware.target(self.target)  # unknown targets fail here, at decoration time

    @staticmethod
    def make_tune(tune: Mapping[str, Sequence[Any]] | None) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple((k, tuple(v)) for k, v in sorted((tune or {}).items()))


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


def lower(
    nest: LoopNest,
    fn: Callable,
    specs: Sequence[jax.ShapeDtypeStruct],
    options: Options,
    overrides: Mapping[str, Any] | None = None,
    buffers: int | None = None,
) -> Lowered:
    spec = specialize(nest, fn, specs, dict(overrides or {}))
    hw = hardware.target(options.target)
    if options.check_tiling and (problems := codegen.check_layout(spec, hw.lanes)):
        raise codegen.layout_error(spec, problems)
    cores = options.cores or hw.cores
    sched = schedule_lib.plan(spec, cores, hw.smem_budget)
    fills, aliased = codegen.plan_fills(spec, options.background_fill)
    highest = options.matmul_precision == "highest"
    if buffers or options.buffers:
        buffers = buffers or options.buffers
        est = cost_lib.estimate(spec, sched, hw, buffers, highest, fills)
    else:
        for buffers in DEFAULT_BUFFERS:  # the deepest default that fits
            est = cost_lib.estimate(spec, sched, hw, buffers, highest, fills)
            if est.vmem_bytes <= hw.vmem_budget:
                break
    if est.vmem_bytes > hw.vmem_budget:
        raise AutoKernelError(
            f"{nest.fn_name}: needs about {est.vmem_bytes / 2**20:.0f} MiB of VMEM, more than the "
            f"{hw.vmem_budget / 2**20:.0f} MiB a {hw.name} kernel can use; use smaller tiles or fewer buffers"
        )
    limit = options.vmem_limit_bytes or min(
        int(hw.vmem_bytes * 0.9), max(32 * 2**20, int(est.vmem_bytes * 1.5) + 8 * 2**20)
    )
    generated = codegen.generate(
        spec,
        sched,
        buffers=buffers,
        matmul_precision=options.matmul_precision,
        fills=fills,
        aliased=aliased,
        vmem_limit_bytes=limit,
        cost=est.cost_estimate(),
    )
    return Lowered(
        source=generated.source,
        entry=generated.entry,
        schedule=sched.mode,
        steps=sched.steps,
        tiles=sched.tiles,
        total_tiles=spec.total_tiles,
        cores=cores,
        core_steps=list(sched.core_steps),
        buffers=buffers,
        operands=spec.operands,
        outputs=spec.outputs,
        state=spec.state,
        estimate=est,
        vmem_limit_bytes=limit,
        fill="prefill" if aliased else "background" if fills else None,
        overrides=dict(overrides or {}),
    )
