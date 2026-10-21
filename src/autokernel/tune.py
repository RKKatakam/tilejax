"""The autotuner: find the fastest schedule for a kernel by running it, from first principles.

A schedule (`Config`) is the kernel's tunable (its block sizes, or whatever its fields mean) and its
buffering (`Pipeline`). Tuning runs on the TPU, on the inputs of the call that asked for it (random inputs of
their shapes if they are being traced, under `jax.jit`): a kernel whose masks read index arrays runs the
steps those inputs need. (Random index arrays that place stores into an output updated in place hold
distinct values: two tiles storing one window would race.)

1. **Test.** Every candidate's output is compared with the loop nest run as plain JAX; a schedule that
   computes something else (or doesn't build, or doesn't compile) is out.
2. **Trace.** Each candidate runs under the JAX profiler. Its device time ranks it, and its trace
   regions (profiling.py) say what limits it.
3. **Change the schedule** the way the bottleneck says, and keep what's faster:
   - copies exposed (`dma`): more input buffers (in a kernel whose tiles take one step, copies in bursts
     too: pipeline.py), then the neighbouring schedules that move fewer bytes (or as many in fewer steps);
   - output copies exposed: more output buffers;
   - per-step overhead (`overhead`): the neighbouring schedules with fewer, bigger steps;
   - compute-bound, or no regions: every neighbouring schedule, fewest steps first.
   A neighbour moves one tunable field one step along its choices. The search climbs from the
   tunable's defaults until no proposal is faster, or it has run `budget` candidates.

Off a TPU nothing can be timed, so the defaults are used as they are.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from . import profiling, v5e
from .errors import KernelError
from .pipeline import MAX_BURST, MAX_INPUT_BUFFERS, MAX_OUTPUT_BUFFERS, Pipeline, default_pipeline, fits, vmem_bytes
from .plan import Kind, Operand, Plan
from .profiling import KernelProfile
from .schedule import Schedule
from .tunable import choices, neighbours

if TYPE_CHECKING:
    from .api import Kernel, Signature

__all__ = ["BUDGET", "Config", "Trial", "Tuning", "proposals", "search", "tune"]

BUDGET = 12  # candidates run per tuning, the defaults included
RUNS = 5  # profiled calls per candidate
EXPOSED_SHARE = 0.03  # waits above this share of the device time are worth more buffering
MIN_GAIN = 0.01  # a schedule must be this much faster to replace the best so far


@dataclass(frozen=True)
class Config:
    """A schedule: the kernel's tunable (None if it has none) and its buffering."""

    tunable: Any
    pipeline: Pipeline

    def __str__(self) -> str:
        if self.tunable is None:
            return str(self.pipeline)
        fields = ", ".join(f"{k}={getattr(self.tunable, k)!r}" for k in choices(self.tunable))
        return f"{type(self.tunable).__name__}({fields}), {self.pipeline}"


@dataclass
class Trial:
    """One candidate schedule run on the TPU: its profile, or why it was ruled out."""

    config: Config
    profile: KernelProfile | None = None
    error: str | None = None

    @property
    def seconds(self) -> float:
        return self.profile.device_s if self.profile else math.inf


@dataclass
class Tuning:
    """What tuning found: the fastest trial, and every trial in the order they ran."""

    best: Trial
    trials: list[Trial] = field(default_factory=list)

    @property
    def config(self) -> Config:
        return self.best.config

    def __str__(self) -> str:
        lines = [f"chose {self.config}" + ("" if self.trials else " (the defaults: tuning needs a TPU)")]
        for t in self.trials:
            mark = "*" if t is self.best else " "
            result = f"{t.seconds * 1e6:10.1f} us  {t.profile.bottleneck}" if t.profile else f"  {t.error}"
            lines.append(f" {mark} {t.config!s:72} {result}")
        return "\n".join(lines)


def search(
    start: Config, measure: Callable[[Config], Trial], propose: Callable[[Trial], Sequence[Config]], budget: int
) -> Tuning:
    """Climb from `start`: measure what the best trial proposes, in order, moving to the first proposal that
    is faster; stop when none is, or after `budget` trials. A trial that fails is infinitely slow, so if the
    start fails (Mosaic can't compile it, say), the first proposal that runs replaces it."""
    best = measure(start)
    trials = {start: best}
    improved = True
    while improved:
        improved = False
        for config in propose(best):
            if config in trials:
                continue
            if len(trials) >= budget:
                return Tuning(best, list(trials.values()))
            trial = trials[config] = measure(config)
            if trial.seconds < best.seconds * (1 - MIN_GAIN):
                best, improved = trial, True
                break
    if best.profile is None:
        errors = "; ".join(f"{t.config}: {t.error}" for t in trials.values())
        raise KernelError(f"no schedule tried ran: {errors}")
    return Tuning(best, list(trials.values()))


@dataclass(frozen=True)
class _Stats:
    """What a schedule costs without running it: its steps, and the bytes its copies move."""

    steps: int
    dma_bytes: int


def _stats(plan: Plan, schedule: Schedule) -> _Stats:
    def copies(op: Operand) -> int:
        if op.kind is Kind.ITERATION:
            return schedule.tiles * sum(1 for p, _ in plan.instances if p == op.phase)
        if op.kind is Kind.STREAM:
            return schedule.steps if op.phase is None else schedule.phase_steps[op.phase]
        return schedule.tiles if op.kind is Kind.TILE else 1

    moved = sum(copies(op) * _bytes(op.tile.shape, op.tile.dtype) for op in plan.operands)
    for out in plan.outputs:
        fills = schedule.fills if out.fill is not None else 0  # (an input updated in place isn't filled)
        stores = schedule.tiles + fills if out.phase is None else schedule.phase_steps[out.phase]
        moved += stores * _bytes(out.tile.shape, out.dtype)
    return _Stats(schedule.steps, moved)


def _loop_steps(plan: Plan) -> tuple[tuple[str, int, int], ...]:
    """What a tunable changes in a kernel: its loops' ranges."""
    return tuple((r.var, r.step, r.count) for r in plan.loops)


def _bytes(shape: tuple[int, ...], dtype: Any) -> int:
    return math.prod(shape) * np.dtype(dtype).itemsize


def proposals(kernel: Kernel, signature: Signature, trial: Trial) -> list[Config]:
    """The schedules to try after `trial`, most promising first, as its profile's bottleneck says (see the
    module docstring). Schedules that don't build or don't fit in VMEM aren't proposed."""
    config, profile = trial.config, trial.profile
    if profile is None:  # it didn't run: the neighbours that need the least VMEM first, then less buffering
        return _smaller(kernel, signature, config)
    plan, schedule = kernel.build(signature, config.tunable)
    here, pipe, loops = _stats(plan, schedule), config.pipeline, _loop_steps(plan)
    exposed = EXPOSED_SHARE * profile.device_s
    buffering = []
    if profile.regions and profile.exposed_wait_s > exposed and pipe.input_buffers < MAX_INPUT_BUFFERS:
        buffering.append(dataclasses.replace(pipe, input_buffers=pipe.input_buffers + 1))
    if profile.regions and profile.out_wait_s > exposed and pipe.output_buffers < MAX_OUTPUT_BUFFERS:
        buffering.append(dataclasses.replace(pipe, output_buffers=pipe.output_buffers + 1))
    if pipe.burst:
        buffering = []  # (a burst's slots follow from its steps)
    if plan.single_step and not schedule.runtime_tiles and profile.bottleneck == "dma":
        buffering += [Pipeline.bursts(g) for g in _bursts(pipe.burst)]
    moves = [Config(config.tunable, p) for p in buffering if fits(plan, schedule, p)]

    near: list[tuple[_Stats, Config]] = []
    for tunable in neighbours(config.tunable) if config.tunable is not None else ():
        try:
            plan, schedule = kernel.build(signature, tunable)
            pipeline = _keep(plan, schedule, pipe, here)
        except KernelError:
            continue  # breaks a layout rule, doesn't divide a loop, doesn't fit in VMEM, ...
        if _loop_steps(plan) == loops:
            continue  # the same kernel (the code clamps the field to the array, say)
        near.append((_stats(plan, schedule), Config(tunable, pipeline)))
    if profile.bottleneck == "dma":  # (fewer bytes; or as many in fewer steps, which never costs more)
        near = [n for n in near if (n[0].dma_bytes, n[0].steps) < (here.dma_bytes, here.steps)]
        near = sorted(near, key=lambda n: (n[0].dma_bytes, n[0].steps))
    elif profile.bottleneck == "overhead":
        near = sorted((n for n in near if n[0].steps < here.steps), key=lambda n: n[0].steps)
    else:
        near = sorted(near, key=lambda n: n[0].steps)
    return moves + [c for _, c in near]


def _keep(plan: Plan, schedule: Schedule, pipe: Pipeline, before: _Stats) -> Pipeline:
    """The buffering for new tiles: `pipe` if it fits with them, else the most that does. Bursts keep the
    bytes they move (what a burst gains depends on), as near as a power of two of steps does, as long as
    that fits; none if the new tiles take more than one step."""
    if not pipe.burst:
        return pipe if fits(plan, schedule, pipe) else default_pipeline(plan, schedule)
    if not plan.single_step:
        return default_pipeline(plan, schedule)
    after = _stats(plan, schedule)
    burst = pipe.burst * (before.dma_bytes / before.steps) / (after.dma_bytes / after.steps)
    burst = min(MAX_BURST, 2 ** max(0, round(math.log2(burst))))
    while burst > 1 and not fits(plan, schedule, Pipeline.bursts(burst)):
        burst //= 2
    return Pipeline.bursts(burst)


def _bursts(burst: int) -> list[int]:
    """The bursts (in steps) to try after copies in bursts of `burst` steps (0: no bursts): the largest
    first from none (on a v5e, bursts of 8-16 MiB moved the most), else twice and half as many steps."""
    if not burst:
        return [g for g in (16, 8, 4) if g <= MAX_BURST]
    return [g for g in (2 * burst, burst // 2) if 2 <= g <= MAX_BURST]


def _smaller(kernel: Kernel, signature: Signature, config: Config) -> list[Config]:
    """After `config` failed to run: two buffers each, then the neighbouring tunables, least VMEM first."""
    least = Pipeline(2, 2)
    found = [] if config.pipeline == least else [Config(config.tunable, least)]
    near = []
    for tunable in neighbours(config.tunable) if config.tunable is not None else ():
        try:
            plan, schedule = kernel.build(signature, tunable)
        except KernelError:
            continue
        if fits(plan, schedule, least):
            near.append((vmem_bytes(plan, schedule, least), Config(tunable, least)))
    return found + [c for _, c in sorted(near, key=lambda n: n[0])]


def random_inputs(
    signature: Signature, bounds: dict[int, int] | None = None, seed: int = 0, distinct: frozenset[int] = frozenset()
) -> list[jax.Array]:
    """Inputs of these shapes and dtypes to tune on: normal floats, small integers, random booleans; the
    inputs `bounds` names (index arrays) from 0 to their bound, distinct values for those in `distinct` (as
    far as the bound allows)."""
    keys = jax.random.split(jax.random.key(seed), len(signature))
    inputs = []
    for i, (key, (shape, dtype)) in enumerate(zip(keys, signature, strict=True)):
        if bounds and i in distinct and math.prod(shape) <= bounds[i] + 1:
            values = jax.random.permutation(key, bounds[i] + 1)[: math.prod(shape)]
            inputs.append(values.reshape(shape).astype(dtype))
        elif bounds and i in bounds:
            inputs.append(jax.random.randint(key, shape, 0, bounds[i] + 1).astype(dtype))
        elif jnp.issubdtype(dtype, jnp.floating):
            inputs.append(jax.random.normal(key, shape, jnp.float32).astype(dtype))
        elif dtype == np.bool_:
            inputs.append(jax.random.bernoulli(key, 0.5, shape))
        else:
            inputs.append(jax.random.randint(key, shape, -8, 8).astype(dtype))
    return inputs


def _index_bounds(kernel: Kernel, signature: Signature, start: Config) -> tuple[dict[int, int], frozenset[int]]:
    """For each index array, the bound of the values to tune on: the furthest any loop goes (sequence
    lengths up to a cache's size, expert offsets up to the rows, ...), or for one that places stores, the
    last element of the dim it indexes; and those that place stores (their values must be distinct)."""
    plan, _ = kernel.build(signature, start.tunable)
    furthest = max(r.start + r.count * r.step for r in plan.loops)
    bounds = {index.param: furthest for index in plan.indices}
    params = {index.name: index.param for index in plan.indices}
    distinct = set()
    for out in plan.outputs:
        for window, dim in zip(out.tile.windows, out.shape, strict=True):
            if window.index is not None:
                param = params[window.index.value.id]  # type: ignore[attr-defined]
                bounds[param] = min(bounds[param], dim - 1)
                distinct.add(param)
    return bounds, frozenset(distinct)


def _reference_config(kernel: Kernel, signature: Signature, start: Config) -> Any:
    """The tunable to compute the plain-JAX reference with: of the defaults and the largest choices, the one
    whose loops run the fewest iterations (the reference unrolls them)."""
    if start.tunable is None:
        return None
    largest = dataclasses.replace(start.tunable, **{k: v[-1] for k, v in choices(start.tunable).items()})
    best, fewest = start.tunable, math.inf
    for tunable in (start.tunable, largest):
        try:
            plan, _ = kernel.build(signature, tunable)
        except KernelError:
            continue
        iterations = math.prod(r.count for r in plan.loops)
        if iterations < fewest:
            best, fewest = tunable, iterations
    return best


def _measure(kernel: Kernel, signature: Signature, config: Config, inputs: Sequence[jax.Array], want: Any) -> Trial:
    """Build, test and profile one schedule."""
    try:
        lowered = kernel.lowered(signature, config, trace=profiling.regions_enabled())
        got = jax.block_until_ready(lowered.run(*inputs))
    except Exception as e:  # doesn't build, or Mosaic can't compile it (out of VMEM, ...)
        return Trial(config, error=f"{type(e).__name__}: {str(e).splitlines()[0]}")
    from .api import compare

    try:
        for name, g, w in zip(kernel.nest.outputs, _outputs(got), _outputs(want), strict=True):
            compare(f"output {name!r}", g, w, inputs=tuple(dtype for _, dtype in signature))
    except AssertionError as e:
        return Trial(config, error=f"wrong result: {e}")
    return Trial(config, profiling.measure(lowered, inputs, RUNS))


def _outputs(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _can_time() -> bool:
    """Can schedules be timed here? Only on the TPU."""
    return v5e.on_tpu()


def _concrete(arrays: Sequence[Any] | None) -> bool:
    """Are these arrays with values (not tracers, not shapes)?"""
    return arrays is not None and all(
        isinstance(a, np.ndarray | jax.Array) and not isinstance(a, jax.core.Tracer) for a in arrays
    )


def tune(kernel: Kernel, signature: Signature, arrays: Sequence[Any] | None = None, budget: int = BUDGET) -> Tuning:
    """Tune `kernel` for inputs of this signature, on `arrays` if they have values (see the module
    docstring)."""
    start = kernel.default_config(signature)
    kernel.lowered(signature, start)  # the default schedule builds, before anything runs
    if not _can_time():
        return Tuning(Trial(start))

    def run() -> Tuning:
        if _concrete(arrays):
            inputs = list(arrays)  # type: ignore[arg-type]
        else:
            bounds, distinct = _index_bounds(kernel, signature, start)
            inputs = random_inputs(signature, bounds, distinct=distinct)
        args = [_reference_config(kernel, signature, start)] if start.tunable is not None else []
        want = kernel.reference(*inputs, *args)
        return search(
            start,
            lambda config: _measure(kernel, signature, config, inputs, want),
            lambda trial: proposals(kernel, signature, trial),
            budget,
        )

    # Tuning runs and times kernels now, also when the first call is being traced (under jax.jit): JAX's
    # trace state is per thread, so a thread of its own runs them outside that trace.
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(run).result()
