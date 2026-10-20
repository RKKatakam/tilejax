"""Profile kernels on a TPU v5e through `jax.named_scope` trace regions, and choose buffering from them.

    autokernel.enable_profiling()          # first thing: before JAX starts its TPU backend
    report = autokernel.profile(matmul, x, y)
    print(report)                          # device time, input waits, compute, output waits, your scopes
    result = autokernel.optimize(matmul, x, y)   # the kernel keeps the buffering the profiles chose

For profiling, a kernel is generated with trace regions: in every step its input waits
(`<kernel>/wait`), its output-slot waits (`<kernel>/out_wait`) and its tile code (`<kernel>/compute`)
are `jax.named_scope`s. Scopes the tile code opens itself (`with jax.named_scope("softmax"):`) are
reported too. Pallas lowers named scopes in a kernel to Mosaic trace regions, which libtpu records only
with `--xla_enable_custom_call_region_trace=true` in `LIBTPU_INIT_ARGS` (what `enable_profiling` adds);
without it a profile has only the kernel's device time.

A wait region's length is copy time the pipeline did not hide behind compute. The first step's wait is
the pipeline filling; every later wait is exposed DMA, which more input slots can hide (fetches start
further ahead). Output-slot waits are copies to HBM still in flight when their slot comes round again, which
another output slot can hide. `optimize` adds slots while the profile shows exposed waits and the kernel
gets faster.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import jax

from . import v5e
from .codegen import COMPUTE, OUT_WAIT, REGIONS, WAIT
from .errors import KernelError
from .pipeline import MAX_INPUT_BUFFERS, MAX_OUTPUT_BUFFERS, Pipeline, fits

if TYPE_CHECKING:
    from .api import Kernel

__all__ = [
    "REGION_FLAG",
    "Event",
    "KernelProfile",
    "Optimization",
    "candidates",
    "enable_profiling",
    "optimize",
    "profile",
    "regions_enabled",
    "summarize",
]

REGION_FLAG = "--xla_enable_custom_call_region_trace=true"
OPS_LINE = "XLA Ops"  # the trace line XLA ops (kernel calls among them) are on
# Shares of the device time above which waits count as exposed, and steps as overhead-bound.
EXPOSED_SHARE = 0.03
DMA_BOUND_SHARE = 0.2
OVERHEAD_BOUND_SHARE = 0.3
# A pipeline must be this much faster to replace the one before it.
MIN_GAIN = 0.01


def regions_enabled() -> bool:
    """Will libtpu record kernels' trace regions (is `REGION_FLAG` in `LIBTPU_INIT_ARGS`)?"""
    return REGION_FLAG in os.environ.get("LIBTPU_INIT_ARGS", "").split()


def _backend_started() -> bool:
    from jax._src import xla_bridge

    return bool(xla_bridge._backends)


def enable_profiling() -> None:
    """Make libtpu record kernels' trace regions. Call it before anything starts JAX's TPU backend (the
    first array, `jax.devices()`, ...): libtpu reads its flags once."""
    if regions_enabled():
        return
    if _backend_started():
        raise KernelError(
            "autokernel.enable_profiling() must run before JAX starts its backend (before the first array or "
            f"jax.devices()), or start the program with LIBTPU_INIT_ARGS={REGION_FLAG}"
        )
    os.environ["LIBTPU_INIT_ARGS"] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} {REGION_FLAG}".strip()


class Event(NamedTuple):
    """One event of a device trace, times in seconds."""

    line: str
    name: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class KernelProfile:
    """A kernel's time per call on the device, averaged over the calls profiled (seconds)."""

    name: str
    pipeline: Pipeline
    calls: int
    device_s: float
    regions: bool = False  # were trace regions recorded (else only `device_s` is known)
    steps: float = 0.0
    wait_s: float = 0.0  # waiting for inputs
    fill_s: float = 0.0  # of which the first step's wait: the pipeline filling
    out_wait_s: float = 0.0  # waiting for a free output slot
    compute_s: float = 0.0  # the tile code
    scopes: dict[str, float] = field(default_factory=dict)  # the tile code's own named scopes
    logdir: str = ""  # the trace, for TensorBoard or XProf

    @property
    def exposed_wait_s(self) -> float:
        """Input copy time the pipeline did not hide, its filling excluded."""
        return max(0.0, self.wait_s - self.fill_s)

    @property
    def other_s(self) -> float:
        """Time outside every region: decoding steps, starting copies, draining the last copies."""
        return max(0.0, self.device_s - self.wait_s - self.out_wait_s - self.compute_s)

    @property
    def bottleneck(self) -> str | None:
        """ "dma" (copies the pipeline doesn't hide), "overhead" (fixed per-step costs: bigger tiles help),
        "compute", or None without regions."""
        if not self.regions:
            return None
        if (self.exposed_wait_s + self.out_wait_s) > DMA_BOUND_SHARE * self.device_s:
            return "dma"
        if self.other_s > OVERHEAD_BOUND_SHARE * self.device_s:
            return "overhead"
        return "compute"

    def __str__(self) -> str:
        us = lambda s: f"{s * 1e6:10.1f} us"  # noqa: E731
        lines = [f"{self.name} ({self.pipeline}): {us(self.device_s).strip()} per call over {self.calls} calls"]
        if not self.regions:
            lines.append(
                f"  no trace regions: call autokernel.enable_profiling() first (LIBTPU_INIT_ARGS={REGION_FLAG})"
            )
        else:
            per_step = self.device_s / self.steps if self.steps else 0.0
            lines += [
                f"  steps          {self.steps:10.0f}     ({per_step * 1e6:.2f} us each)",
                f"  compute        {us(self.compute_s)}",
                f"  input waits    {us(self.exposed_wait_s)}  (+{us(self.fill_s).strip()} filling the pipeline)",
                f"  output waits   {us(self.out_wait_s)}",
                f"  other          {us(self.other_s)}  (decoding steps, starting copies, draining)",
                *(f"  scope {name!r:9}{us(s)}" for name, s in self.scopes.items()),
                f"  bottleneck: {self.bottleneck}",
            ]
        if self.logdir:
            lines.append(f"  trace: {self.logdir}")
        return "\n".join(lines)


@dataclass
class Optimization:
    """What `optimize` measured: the profile of each pipeline it tried, and the one the kernel now uses."""

    profile: KernelProfile
    tried: list[KernelProfile]

    @property
    def pipeline(self) -> Pipeline:
        return self.profile.pipeline

    def __str__(self) -> str:
        rows = [f"  {p.pipeline}: {p.device_s * 1e6:.1f} us, bottleneck {p.bottleneck}" for p in self.tried]
        return "\n".join([f"{self.profile.name}: chose {self.pipeline}", *rows])


# --------------------------------------------------------------------------- reading a trace


def device_events(logdir: str, plane: str = "/device:TPU") -> list[Event]:
    """Every event the trace under `logdir` has on planes whose name starts with `plane`."""
    from jax.profiler import ProfileData

    files = sorted(glob.glob(os.path.join(logdir, "**", "*.xplane.pb"), recursive=True))
    if not files:
        raise KernelError(f"the profiler wrote no trace under {logdir}")
    events = []
    for p in ProfileData.from_file(files[-1]).planes:
        if p.name.startswith(plane):
            for line in p.lines:
                for e in line.events:
                    start = e.start_ns * 1e-9
                    events.append(Event(line.name, e.name, start, start + e.duration_ns * 1e-9))
    return events


def _named(event_name: str, name: str) -> bool:
    """Is a trace event named `name`, possibly under enclosing scopes (`.../name`)?"""
    return event_name == name or event_name.endswith("/" + name)


def summarize(events: Sequence[Event], kernel: str, pipeline: Pipeline, scopes: Sequence[str] = ()) -> KernelProfile:
    """The profile of the pallas_call named `kernel` from a device trace's events."""
    op = re.compile(rf"^%?{re.escape(kernel)}(\.\S*)?(\s|$)")  # `name.3`, `name.1.out_hbm`, `%name.3 = ...`
    calls = [e for e in events if e.line == OPS_LINE and op.match(e.name)]
    if not calls:
        raise KernelError(f"the trace has no call of kernel {kernel!r}")
    regions = [e for e in events if e.line != OPS_LINE]
    totals = dict.fromkeys(REGIONS, 0.0)
    scope_totals = dict.fromkeys(scopes, 0.0)
    steps = fill = 0.0
    for call in calls:
        first_wait = True
        for e in sorted((r for r in regions if call.start <= r.start and r.end <= call.end), key=lambda r: r.start):
            region = next((r for r in REGIONS if _named(e.name, f"{kernel}/{r}")), None)
            if region is None:
                scope = next((s for s in scopes if _named(e.name, s)), None)
                if scope is not None:
                    scope_totals[scope] += e.seconds
                continue
            totals[region] += e.seconds
            if region == WAIT and first_wait:
                fill += e.seconds
            if region == COMPUTE:
                steps += 1
            first_wait = first_wait and region != COMPUTE
    n = len(calls)
    return KernelProfile(
        name=kernel,
        pipeline=pipeline,
        calls=n,
        device_s=sum(c.seconds for c in calls) / n,
        regions=steps > 0,
        steps=steps / n,
        wait_s=totals[WAIT] / n,
        fill_s=fill / n,
        out_wait_s=totals[OUT_WAIT] / n,
        compute_s=totals[COMPUTE] / n,
        scopes={s: t / n for s, t in scope_totals.items()},
    )


# --------------------------------------------------------------------------- choosing buffering


def candidates(profile: KernelProfile, fits: Callable[[Pipeline], bool]) -> list[Pipeline]:
    """The buffering worth trying after `profile`'s, most promising first: one more input slot if inputs
    waited, one more output slot if outputs waited (each if it fits in VMEM)."""
    if not profile.regions:
        return []
    pipeline, exposed = profile.pipeline, EXPOSED_SHARE * profile.device_s
    found = []
    if profile.exposed_wait_s > exposed and pipeline.input_buffers < MAX_INPUT_BUFFERS:
        found.append(dataclasses.replace(pipeline, input_buffers=pipeline.input_buffers + 1))
    if profile.out_wait_s > exposed and pipeline.output_buffers < MAX_OUTPUT_BUFFERS:
        found.append(dataclasses.replace(pipeline, output_buffers=pipeline.output_buffers + 1))
    return [p for p in found if fits(p)]


# --------------------------------------------------------------------------- running on the TPU


def _require_tpu() -> None:
    if not v5e.on_tpu():
        raise KernelError("profiling needs a TPU v5e: it reads the device's own trace")


def _measure(kernel: Kernel, inputs: Sequence[Any], pipeline: Pipeline, runs: int, trace: bool) -> KernelProfile:
    """Run the kernel `runs` times under the JAX profiler (after a warm-up call) and summarize the trace."""
    lowered = kernel.lowered(kernel.signature(*inputs), pipeline, trace=trace)
    jax.block_until_ready(lowered.run(*inputs))
    logdir = tempfile.mkdtemp(prefix=f"autokernel-{kernel.nest.name}-")
    jax.profiler.start_trace(logdir)
    try:
        for _ in range(runs):
            jax.block_until_ready(lowered.run(*inputs))
    finally:
        jax.profiler.stop_trace()
    result = summarize(device_events(logdir), kernel.nest.name, pipeline, lowered.plan.scopes)
    result.logdir = logdir
    return result


def profile(kernel: Kernel, inputs: Sequence[Any], runs: int = 5) -> KernelProfile:
    """Where `kernel`'s time goes on the TPU for these inputs (see the module docstring). Without trace
    regions enabled, only its device time."""
    _require_tpu()
    return _measure(kernel, inputs, kernel.pipeline(kernel.signature(*inputs)), runs, trace=regions_enabled())


def optimize(kernel: Kernel, inputs: Sequence[Any], runs: int = 5) -> Optimization:
    """Profile `kernel` on these inputs and add buffer slots where its copies are exposed, keeping each
    change that makes it faster; the kernel uses the result for inputs like these from then on."""
    _require_tpu()
    if not regions_enabled():
        raise KernelError(
            "autokernel.optimize reads trace regions: call autokernel.enable_profiling() before JAX starts "
            f"(or start with LIBTPU_INIT_ARGS={REGION_FLAG})"
        )
    signature = kernel.signature(*inputs)
    lowered = kernel.lowered(signature, kernel.pipeline(signature))
    best = _measure(kernel, inputs, lowered.pipeline, runs, trace=True)
    tried = [best]
    improved = True
    while improved:  # from the best so far, try what its profile suggests until something is faster
        improved = False
        for candidate in candidates(best, lambda p: fits(lowered.plan, lowered.schedule, p)):
            result = _measure(kernel, inputs, candidate, runs, trace=True)
            tried.append(result)
            if result.device_s < best.device_s * (1 - MIN_GAIN):
                best, improved = result, True
                break
    kernel.choose(signature, best.pipeline)
    return Optimization(best, tried)
