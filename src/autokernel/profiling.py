"""Profile kernels on the TPU: when each step waits for its data, when it computes, and how well the two
overlap, read from a JAX profiler trace.

    autokernel.enable_profiling()  # before JAX starts its TPU backend (first thing in the program)
    ...
    report = autokernel.profile(matmul, x, y)  # a kernel, or a whole program
    print(report)  # per kernel: device time, exposed DMA waits, compute, fill/drain, the bottleneck
    # report.logdir holds the trace: `tensorboard --logdir <it>` (or XProf) shows the timeline

The kernels are built again with *trace regions* (`codegen.generate(trace=True)`): in every step, the waits
for its inputs (`<kernel>/wait`), the user's code (`<kernel>/compute`) and, inside that, the waits for an
output slot (`<kernel>/out_wait`) are marked with `jax.named_scope`, which Pallas lowers to Mosaic trace
regions. libtpu only records those with `--xla_enable_custom_call_region_trace=true` in `LIBTPU_INIT_ARGS`,
read when JAX starts its TPU backend: `enable_profiling` sets it. Without it the report still has each
kernel's device time, just no breakdown.

A wait region's length is the DMA time the pipeline did *not* hide behind compute: zero when the copy had
already landed, the copy's whole latency on the pipeline's first step (the fill). What isn't in a region
(step decoding, issuing copies, the last output copies) is `other`.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import tempfile
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax

from . import codegen, hardware
from .errors import AutoKernelError

__all__ = [
    "REGION_FLAG",
    "KernelProfile",
    "Profile",
    "enable_profiling",
    "profile",
    "profile_lowered",
    "read_trace",
    "regions_enabled",
]

REGION_FLAG = "--xla_enable_custom_call_region_trace=true"
# A kernel spending more than this share of its time waiting for copies is DMA-bound; one spending more
# than OVERHEAD_SHARE outside its regions (decoding steps, issuing copies) is overhead-bound.
DMA_SHARE = 0.2
XLA_OPS_SHOWN = 5  # other XLA ops a report lists by name
OVERHEAD_SHARE = 0.3


def regions_enabled() -> bool:
    """Will libtpu record kernels' trace regions (is `REGION_FLAG` in `LIBTPU_INIT_ARGS`)?"""
    return REGION_FLAG in os.environ.get("LIBTPU_INIT_ARGS", "").split()


def _backend_started() -> bool:
    from jax._src import xla_bridge

    return bool(getattr(xla_bridge, "_backends", None))


def enable_profiling() -> None:
    """Make libtpu record kernels' trace regions: adds `REGION_FLAG` to `LIBTPU_INIT_ARGS`. Call it before
    anything starts JAX's TPU backend (creating an array, `jax.devices()`, ...): libtpu reads its flags once."""
    if regions_enabled():
        return
    if _backend_started():
        raise AutoKernelError(
            "autokernel.enable_profiling() must run before JAX starts its backend (before the first array or "
            f"jax.devices()); or start the program with LIBTPU_INIT_ARGS={REGION_FLAG}"
        )
    os.environ["LIBTPU_INIT_ARGS"] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} {REGION_FLAG}".strip()


# --------------------------------------------------------------------------- results


@dataclass
class KernelProfile:
    """One kernel's time per call, averaged over the calls profiled (seconds)."""

    name: str
    calls: int
    device_s: float  # the kernel's whole run on the device
    predicted_s: float | None = None  # the cost model's estimate
    steps: float = 0.0  # steps per call (compute regions)
    wait_s: float = 0.0  # waiting for inputs: DMA time the pipeline didn't hide
    fill_s: float = 0.0  # of which the first step's wait (the pipeline filling)
    out_wait_s: float = 0.0  # waiting for an output slot (its previous copy to HBM)
    compute_s: float = 0.0  # the user's code (output waits excluded)
    parts: dict[str, dict[str, float]] = field(default_factory=dict)  # a fused kernel's, by part: region -> s
    regions: bool = False  # were trace regions recorded (else only `device_s` is known)

    @property
    def other_s(self) -> float:
        """Time outside every region: decoding steps, issuing copies, the last output copies."""
        return max(0.0, self.device_s - self.wait_s - self.out_wait_s - self.compute_s)

    @property
    def exposed_dma_s(self) -> float:
        """DMA time not hidden behind compute, the pipeline's fill excluded."""
        return max(0.0, self.wait_s - self.fill_s) + self.out_wait_s

    @property
    def overlap(self) -> float | None:
        """The share of the kernel's time spent not waiting for copies: 1.0 when every copy landed before it
        was needed (all DMA hidden behind compute). None without regions."""
        if not self.regions or not self.device_s:
            return None
        return 1.0 - (self.wait_s + self.out_wait_s) / self.device_s

    @property
    def bottleneck(self) -> str | None:
        """ "dma" (copies exposed), "overhead" (fixed per-step costs: bigger blocks help), "compute", or None
        without regions."""
        if not self.regions or not self.device_s:
            return None
        if (self.exposed_dma_s + self.fill_s) / self.device_s > DMA_SHARE:
            return "dma"
        if self.other_s / self.device_s > OVERHEAD_SHARE:
            return "overhead"
        return "compute"

    def row(self) -> str:
        us = lambda s: f"{s * 1e6:9.1f}"  # noqa: E731
        pred = us(self.predicted_s) if self.predicted_s is not None else "        -"
        if not self.regions:
            return f"{self.name[:32]:32} {us(self.device_s)} {pred}" + " " * 50 + "(no regions)"
        return (
            f"{self.name[:32]:32} {us(self.device_s)} {pred} {self.steps:7.0f} {us(self.compute_s)} "
            f"{us(self.exposed_dma_s)} {us(self.fill_s)} {us(self.other_s)}  {self.bottleneck}"
        )


_HEADER = (
    f"{'kernel':32} {'device us':>9} {'model us':>9} {'steps':>7} {'compute':>9} {'dma wait':>9} {'fill':>9} "
    f"{'other':>9}  bottleneck"
)


@dataclass
class Profile:
    """What `profile` measured: each kernel's time per call, the rest of the device time (XLA ops) per run,
    and the trace directory (for TensorBoard / XProf)."""

    kernels: list[KernelProfile]
    xla_s: float  # device time per run in everything that isn't one of the kernels
    runs: int
    logdir: str
    xla_ops: list[tuple[str, float]] = field(default_factory=list)  # the biggest of those, per run: (op, s)

    def __getitem__(self, name: str) -> KernelProfile:
        for k in self.kernels:
            if k.name == name:
                return k
        raise KeyError(f"no kernel named {name!r}; profiled: {[k.name for k in self.kernels]}")

    def __str__(self) -> str:
        lines = [_HEADER, *(k.row() for k in self.kernels)]
        lines.append(f"{'other XLA ops (per run)':32} {self.xla_s * 1e6:9.1f}")
        lines += [f"  {name[:60]:60} {s * 1e6:9.1f}" for name, s in self.xla_ops]
        if self.kernels and not any(k.regions for k in self.kernels):
            lines.append(f"(no trace regions: start the program with LIBTPU_INIT_ARGS={REGION_FLAG})")
        lines.append(f"trace: {self.logdir}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- reading a trace


def _op_pattern(name: str) -> re.Pattern:
    """An XLA op event of the pallas_call named `name`: `name.3`, `name.1.h_hbm` (XLA may append the output's
    name), or without regions `%name.3 = ...`."""
    return re.compile(rf"^%?{re.escape(name)}(\.\S*)?(\s|$)")


def _events(logdir: str) -> list[tuple[str, str, float, float]]:
    """Every device event in the trace under `logdir`: (line, name, start, end) in seconds."""
    from jax.profiler import ProfileData

    files = sorted(glob.glob(os.path.join(logdir, "**", "*.xplane.pb"), recursive=True))
    if not files:
        raise AutoKernelError(f"the profiler wrote no trace under {logdir}")
    out = []
    for plane in ProfileData.from_file(files[-1]).planes:
        if not plane.name.startswith("/device:TPU"):
            continue
        for line in plane.lines:
            for ev in line.events:
                start = ev.start_ns * 1e-9
                out.append((line.name, ev.name, start, start + ev.duration_ns * 1e-9))
    return out


def read_trace(
    logdir: str, kernels: Sequence[tuple[str, Sequence[str], float | None]], runs: int
) -> tuple[list[KernelProfile], float, list[tuple[str, float]]]:
    """Each kernel's profile from the trace under `logdir`, the other XLA ops' device time per run, and the
    biggest of those ops (`XLA_OPS_SHOWN`, by time per run).

    `kernels` is (pallas_call name, the names of its parts' regions, predicted seconds) per kernel."""
    events = _events(logdir)
    ops = [e for e in events if e[0] == "XLA Ops"]
    regions = [e for e in events if e[0] != "XLA Ops" and "/" in e[1]]
    profiles, claimed = [], set()
    for name, parts, predicted in kernels:
        pattern = _op_pattern(name)
        calls = [e for e in ops if pattern.match(e[1])]
        claimed.update(id(e) for e in calls)
        p = KernelProfile(name, len(calls), 0.0, predicted)
        if calls:
            p.device_s = sum(e[3] - e[2] for e in calls) / len(calls)
        per_part = {part: {codegen.WAIT: 0.0, codegen.COMPUTE: 0.0, codegen.OUT_WAIT: 0.0} for part in parts}
        steps = fill = 0.0
        for _, _, start, end in calls:
            inside = sorted((r for r in regions if start <= r[2] and r[3] <= end), key=lambda r: r[2])
            first_wait = True
            for _, rname, rs, re_ in inside:
                part, _, what = rname.rpartition("/")
                if part not in per_part or what not in per_part[part]:
                    continue
                per_part[part][what] += re_ - rs
                p.regions = True
                if what == codegen.COMPUTE:
                    steps += 1
                    first_wait = False
                elif what == codegen.WAIT and first_wait:
                    fill += re_ - rs
                    first_wait = False
        n = max(1, len(calls))
        totals = {what: sum(d[what] for d in per_part.values()) / n for what in codegen.REGIONS}
        p.wait_s, p.out_wait_s = totals[codegen.WAIT], totals[codegen.OUT_WAIT]
        p.compute_s = max(0.0, totals[codegen.COMPUTE] - p.out_wait_s)  # output waits are inside compute
        p.steps, p.fill_s = steps / n, fill / n
        if len(parts) > 1:
            p.parts = {part: {w: t / n for w, t in d.items()} for part, d in per_part.items()}
        profiles.append(p)
    others: dict[str, float] = {}
    for e in ops:
        if id(e) not in claimed:
            name = e[1].split(" = ")[0].lstrip("%")
            others[name] = others.get(name, 0.0) + (e[3] - e[2]) / max(1, runs)
    top = sorted(others.items(), key=lambda kv: -kv[1])[:XLA_OPS_SHOWN]
    return profiles, sum(others.values()), top


# --------------------------------------------------------------------------- running


def _kernel_names(lowered: Any) -> tuple[str, list[str], float | None]:
    """(pallas_call name, its parts' region names, predicted seconds) of a generated kernel."""
    parts = getattr(lowered, "parts", None)
    if parts is not None:  # fused
        return lowered.name, [lw.plan.spec.nest.fn_name for _, lw in parts], lowered.estimate.seconds
    name = lowered.plan.spec.nest.fn_name
    return name, [name], lowered.estimate.seconds


def _traced(fn: Any) -> tuple[Callable[..., Any], Callable[..., list]]:
    """`fn` (a TpuKernel or TpuProgram) built again with trace regions: (the function to run, a function of
    the arguments listing (pallas name, region names, predicted seconds) for each of its kernels)."""
    import dataclasses

    from .api import TpuKernel
    from .lowering import build
    from .program import TpuProgram

    if isinstance(fn, TpuKernel):
        traced_options = dataclasses.replace(fn.options, trace=True)

        built: dict[int, tuple[Any, Callable]] = {}  # id(lowered) -> (lowered, its traced twin, jitted)

        def run(*args: Any) -> Any:
            ctx, inputs = fn._bind(args, {})
            lowered = fn._lower(ctx, inputs)
            if id(lowered) not in built or built[id(lowered)][0] is not lowered:
                traced = build(dataclasses.replace(lowered.plan, trace=True), traced_options)
                built[id(lowered)] = (lowered, jax.jit(functools.partial(traced.entry, interpret=False)))
            return built[id(lowered)][1](*inputs)

        return run, lambda *args: [_kernel_names(fn.lower(*args))]
    if isinstance(fn, TpuProgram):
        twin = TpuProgram(
            fn.fn,
            interpret=fn.interpret,
            options=dataclasses.replace(fn.options, trace=True),
            fuse=fn.max_fused if fn.fuse else False,
            unroll=fn.unroll,
            backward=fn.backward,
        )
        jitted = jax.jit(twin)
        return jitted, lambda *args: [_kernel_names(lw) for _, lw in twin.lower(*args)]
    raise AutoKernelError(f"autokernel.profile takes a kernel or program made by autokernel.tpu, got {fn!r}")


def profile_lowered(
    lowered: Any, args: Sequence[Any], runs: int = 10, trace: bool = False, options: Any = None
) -> KernelProfile:
    """One generated kernel's profile on the attached TPU: its device time over `runs` calls after a warm-up
    (from the profiler, so without host overhead), and with `trace` its breakdown (built again with trace
    regions, `options` its build options)."""
    import dataclasses
    import shutil

    from .lowering import Options, build

    if trace:
        lowered = build(dataclasses.replace(lowered.plan, trace=True), options or Options())
    fn = jax.jit(functools.partial(lowered.entry, interpret=False))
    jax.block_until_ready(fn(*args))
    logdir = tempfile.mkdtemp(prefix="autokernel-tune-")
    try:
        jax.profiler.start_trace(logdir)
        try:
            for _ in range(runs):
                jax.block_until_ready(fn(*args))
        finally:
            jax.profiler.stop_trace()
        (kernel,), _, _ = read_trace(logdir, [_kernel_names(lowered)], runs)
    finally:
        shutil.rmtree(logdir, ignore_errors=True)
    if not kernel.calls:
        raise AutoKernelError(f"the trace has no call of kernel {kernel.name!r}")
    return kernel


def profile(fn: Any, *args: Any, runs: int = 5, logdir: str | None = None) -> Profile:
    """Run `fn` (an `autokernel.tpu` kernel or program) on `args` under the JAX profiler, its kernels built
    with trace regions, and report each kernel's device time and where it goes (see the module docstring).

    One warm-up call compiles, then `runs` calls are traced. `logdir` (default: a new temporary directory)
    keeps the trace for TensorBoard or XProf."""
    if not hardware.on_tpu():
        raise AutoKernelError("autokernel.profile needs a TPU: it reads the device's own trace")
    if not regions_enabled():
        warnings.warn(
            f"autokernel: trace regions are off (start with LIBTPU_INIT_ARGS={REGION_FLAG}, or call "
            "autokernel.enable_profiling() first): profiling only kernels' device times",
            stacklevel=2,
        )
    run, names = _traced(fn)
    jax.block_until_ready(run(*args))
    logdir = logdir or tempfile.mkdtemp(prefix="autokernel-profile-")
    jax.profiler.start_trace(logdir)
    try:
        for _ in range(runs):
            jax.block_until_ready(run(*args))
    finally:
        jax.profiler.stop_trace()
    kernels, xla, xla_ops = read_trace(logdir, names(*args), runs)
    return Profile(kernels, xla, runs, logdir, xla_ops)
