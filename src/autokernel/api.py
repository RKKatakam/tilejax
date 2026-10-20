"""`@autokernel.kernel`: a Python loop nest, compiled to a gridless Pallas kernel for a TPU v5e."""

from __future__ import annotations

import functools
import itertools
import linecache
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from . import codegen, v5e
from .errors import KernelError
from .parse import Nest, parse
from .pipeline import Pipeline, default_pipeline, vmem_bytes
from .plan import Plan, make_plan
from .reference import make_reference
from .schedule import Schedule, make_schedule
from .tunable import is_tunable, neighbours
from .tune import Config, Tuning, tune

__all__ = ["Kernel", "Lowered", "kernel"]

Signature = tuple[tuple[tuple[int, ...], np.dtype], ...]

_generated = itertools.count()  # numbers the generated sources' file names, which linecache keys by
JIT_OPS = 20_000  # a reference unrolling into more operations than this runs op by op, not jitted


@dataclass(frozen=True)
class Lowered:
    """A kernel generated for one input signature and schedule: its plan, schedule and Pallas source."""

    plan: Plan
    schedule: Schedule
    config: Config
    source: str
    trace: bool  # does it mark trace regions (for profiling)
    entry: Callable[..., Any] = field(repr=False)  # entry(*inputs, interpret=False): the pallas_call

    @property
    def pipeline(self) -> Pipeline:
        return self.config.pipeline

    @property
    def steps(self) -> int:
        return self.schedule.steps

    @property
    def tiles(self) -> int:
        """Output tiles the kernel computes (masked-out ones excluded)."""
        return self.schedule.tiles

    @property
    def total_tiles(self) -> int:
        return self.schedule.total_tiles

    @property
    def vmem_bytes(self) -> int:
        """VMEM taken by the kernel's own buffers."""
        return vmem_bytes(self.plan, self.schedule, self.pipeline)

    @functools.cached_property
    def run(self) -> Callable[..., Any]:
        """The kernel, jitted: on the TPU, or in the Pallas TPU interpreter elsewhere."""
        return jax.jit(functools.partial(self.entry, interpret=interpret_mode()))


def interpret_mode() -> Any:
    """How kernels run here: natively on a TPU v5e, in the Pallas TPU interpreter without a TPU."""
    if not v5e.on_tpu():
        return pltpu.InterpretParams()
    kind = jax.devices()[0].device_kind
    if kind != v5e.DEVICE_KIND:
        raise KernelError(f"autokernel targets TPU v5e ({v5e.DEVICE_KIND!r}); this TPU is {kind!r}")
    return False


class Kernel:
    """A loop nest compiled to a gridless Pallas kernel. Call it like the function it was made from.

    Given its tunable (`matmul(x, y, MatmulTiles(tm=256))`), the kernel uses it. Without it, the kernel is
    tuned for the inputs' shapes when first called with them (tune.py), and remembers the result. A kernel
    is built when it is first called with a new shape and schedule, reading the constants its code uses at
    that moment (as `jax.jit` does).
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.nest: Nest = parse(fn)
        functools.update_wrapper(self, fn)
        self.default_tunable = fn.__defaults__[-1] if self.nest.tunable else None
        if self.nest.tunable and not is_tunable(self.default_tunable):
            raise self.nest.error(
                f"`{self.nest.tunable}`'s default must be an @autokernel.tunable, not {self.default_tunable!r}",
                self.nest.function.args,
            )
        self._builds: dict[tuple[Signature, Any], tuple[Plan, Schedule]] = {}
        self._references: dict[Signature, Callable[..., Any]] = {}
        self._jitted: dict[Signature, Callable[..., Any]] = {}
        self._lowered: dict[tuple[Signature, Config, bool], Lowered] = {}
        self._tunings: dict[Signature, Tuning] = {}

    def __repr__(self) -> str:
        return f"<autokernel.Kernel {self.nest.name}>"

    def __call__(self, *args: Any) -> Any:
        arrays, _ = self._split(args)
        return self.lower(*args).run(*arrays)

    def lower(self, *args: Any) -> Lowered:
        """The kernel for inputs like these (arrays or `jax.ShapeDtypeStruct`s), and the tunable if given."""
        arrays, tunable = self._split(args)
        signature = self.signature(*arrays)
        if tunable is None:
            return self.lowered(signature, self.tuning(*arrays).config)
        plan, schedule = self.build(signature, tunable)
        return self.lowered(signature, Config(tunable, default_pipeline(plan, schedule)))

    def source(self, *args: Any) -> str:
        """The generated Pallas source for inputs like these."""
        return self.lower(*args).source

    def reference(self, *args: Any) -> Any:
        """What the kernel computes: the loop nest run as plain JAX (with `@` on the MXU)."""
        arrays, tunable = self._split(args)
        signature = self.signature(*arrays)
        self.build(signature, tunable if tunable is not None else self.default_tunable)  # reads the constants
        if self.nest.tunable:
            tunable = tunable if tunable is not None else self.lower(*arrays).config.tunable
        plan, _ = self.build(signature, tunable)
        reference = self._references[signature]
        # The loops unroll: XLA would take minutes to compile a reference of hundreds of thousands of
        # operations, which run op by op in seconds. (Masks reading index arrays need values: op by op too.)
        if not self.nest.runtime_masks and plan.tile_ops * math.prod(r.count for r in plan.loops) <= JIT_OPS:
            reference = self._jitted.setdefault(signature, jax.jit(reference))
        return reference(*arrays, tunable) if self.nest.tunable else reference(*arrays)

    def verify(self, *args: Any, rtol: float | None = None, atol: float | None = None) -> float:
        """Run the kernel and its reference; raise AssertionError unless they agree (within tolerances for the
        dtypes, or `rtol` and `atol`). Returns the largest absolute difference."""
        arrays, tunable = self._split(args)
        if self.nest.tunable:
            arrays = (*arrays, tunable if tunable is not None else self.lower(*arrays).config.tunable)
        got, want = self(*arrays), self.reference(*arrays)
        got, want = (got, want) if isinstance(got, tuple) else ((got,), (want,))
        dtypes = tuple(dtype for _, dtype in self.signature(*arrays[: len(self.nest.params)]))
        errors = [
            compare(f"{self.nest.name} output {name!r}", g, w, rtol, atol, dtypes)
            for name, g, w in zip(self.nest.outputs, got, want, strict=True)
        ]
        return max(errors)

    def tuning(self, *arrays: Any) -> Tuning:
        """The schedule for inputs like these: tuned on the TPU when first asked (see tune.py), the
        defaults elsewhere."""
        signature = self.signature(*arrays)
        if signature not in self._tunings:
            self._tunings[signature] = tune(self, signature, arrays)
        return self._tunings[signature]

    # ------------------------------------------------------------------ building (used by tune.py too)

    def signature(self, *arrays: Any) -> Signature:
        """What the kernel is built for: the inputs' shapes and dtypes."""
        signature = []
        for name, x in zip(self.nest.params, arrays, strict=True):
            if not (hasattr(x, "shape") and hasattr(x, "dtype")) or not tuple(x.shape):
                raise KernelError(f"{self.nest.name}: input `{name}` must be an array, got {type(x).__name__}")
            signature.append((tuple(x.shape), np.dtype(x.dtype)))
        return tuple(signature)

    def default_config(self, signature: Signature) -> Config:
        """The schedule to start from: the tunable's defaults with their default buffering, or, if they don't
        build for these inputs (their tiles don't fit in VMEM, say), the nearest tunable that does."""
        tunable = self.default_tunable
        try:
            return Config(tunable, default_pipeline(*self.build(signature, tunable)))
        except KernelError:
            if tunable is None:
                raise
            found = _nearest(tunable, lambda t: Config(t, default_pipeline(*self.build(signature, t))))
            if found is None:
                raise
            return found

    def build(self, signature: Signature, tunable: Any) -> tuple[Plan, Schedule]:
        """The plan and schedule for this signature and tunable."""
        key = (signature, tunable)
        if signature not in self._references:  # read the constants once per signature, with the plan
            self._references[signature] = make_reference(self.nest, self.fn)
        if key not in self._builds:
            specs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in signature]
            plan = make_plan(self.nest, self.fn, specs, tunable)
            self._builds[key] = (plan, make_schedule(plan))
        return self._builds[key]

    def lowered(self, signature: Signature, config: Config, trace: bool = False) -> Lowered:
        """The kernel for this signature and schedule (with trace regions, for profiling)."""
        key = (signature, config, trace)
        if key not in self._lowered:
            plan, schedule = self.build(signature, config.tunable)
            source, entry_name = codegen.generate(plan, schedule, config.pipeline, trace=trace)
            entry = _load(source, f"<autokernel {self.nest.name} #{next(_generated)}>", entry_name)
            self._lowered[key] = Lowered(plan, schedule, config, source, trace, entry)
        return self._lowered[key]

    def _split(self, args: tuple[Any, ...]) -> tuple[tuple[Any, ...], Any]:
        """The input arrays, and the tunable if one was passed."""
        n = len(self.nest.params)
        if self.nest.tunable and len(args) == n + 1:
            tunable = args[-1]
            if type(tunable) is not type(self.default_tunable):
                raise KernelError(
                    f"{self.nest.name}: `{self.nest.tunable}` must be a {type(self.default_tunable).__name__}, "
                    f"not {type(tunable).__name__}"
                )
            return args[:-1], tunable
        if len(args) != n:
            more = f", then optionally `{self.nest.tunable}`" if self.nest.tunable else ""
            raise KernelError(
                f"{self.nest.name} takes {n} input arrays ({', '.join(self.nest.params)}){more}; got {len(args)}"
            )
        return args, None


def _nearest(tunable: Any, make: Callable[[Any], Config]) -> Config | None:
    """The first tunable, breadth-first over neighbours from `tunable`, that `make` turns into a Config
    without a KernelError."""
    seen, frontier = {tunable}, [tunable]
    while frontier:
        later = []
        for t in frontier:
            for n in neighbours(t):
                if n in seen:
                    continue
                seen.add(n)
                try:
                    return make(n)
                except KernelError:
                    later.append(n)
        frontier = later
    return None


def _load(source: str, filename: str, entry: str) -> Callable[..., Any]:
    """The function `entry` of generated source, compiled under `filename` (registered with linecache, so
    tracebacks through the kernel show its lines)."""
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {"__name__": "autokernel.generated"}
    exec(compile(source, filename, "exec"), namespace)
    return namespace[entry]


# Elements `compare` checks at a time: float64 copies of a whole output of several GB (an MoE layer's rows at
# 16384 tokens) and their temporaries ran a TPU VM's host out of memory.
COMPARE_CHUNK = 1 << 22


def compare(
    what: str,
    got: Any,
    want: Any,
    rtol: float | None = None,
    atol: float | None = None,
    inputs: tuple[Any, ...] = (),
) -> float:
    """The largest absolute difference between `got` and `want`; AssertionError if they disagree beyond
    `rtol` and `atol` (default: exact for integers, 1e-4 for f32, 2e-2 of the largest value for 16-bit, also
    for an f32 result computed from 16-bit inputs, the dtypes `inputs`: Mosaic and XLA may round its 16-bit
    intermediates one ulp apart)."""
    dtype = np.dtype(want.dtype)
    narrow = any(jnp.issubdtype(d, jnp.floating) and np.dtype(d).itemsize <= 2 for d in inputs)
    if not jnp.issubdtype(dtype, jnp.inexact):
        default = 0.0
    elif dtype.itemsize <= 2 or narrow:
        default = 2e-2
    else:
        default = 1e-4
    rtol = default if rtol is None else rtol
    got_all, want_all = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
    chunks = range(0, want_all.size, COMPARE_CHUNK)
    if atol is None:
        # 16-bit results round at every step: their error grows with the output's magnitude.
        scale = 1.0
        if default >= 2e-2:
            for start in chunks:
                w = want_all[start : start + COMPARE_CHUNK].astype(np.float64)
                scale = max(scale, float(np.max(np.abs(w), where=np.isfinite(w), initial=0.0)))
        atol = default * scale
    error, differ = 0.0, 0
    for start in chunks:
        g = got_all[start : start + COMPARE_CHUNK].astype(np.float64)
        w = want_all[start : start + COMPARE_CHUNK].astype(np.float64)
        same = (g == w) | (np.isnan(g) & np.isnan(w))
        with np.errstate(invalid="ignore"):
            error = max(error, float(np.max(np.abs(np.where(same, 0.0, g - w)), initial=0.0)))
        differ += int(np.sum(~np.isclose(g, w, rtol=rtol, atol=atol, equal_nan=True)))
    if differ:
        raise AssertionError(
            f"{what}: the kernel and its reference differ in {differ} of {want_all.size} elements "
            f"(largest difference {error:.3g}; rtol={rtol}, atol={atol:.3g})"
        )
    return error


def kernel(fn: Callable[..., Any]) -> Kernel:
    """Compile a Python loop nest over tiles into a gridless Pallas kernel for a TPU v5e.

        @autokernel.tunable
        class Tiles:
            tm: int = autokernel.choice(128, 256, 512, 1024, default=512)
            tn: int = autokernel.choice(128, 256, 512, 1024, default=512)
            tk: int = autokernel.choice(128, 256, 512, 1024, default=512)

        @autokernel.kernel
        def matmul(x, y, tiles=Tiles()):
            M, K = x.shape
            out = jnp.zeros((M, y.shape[1]), jnp.float32)
            for i in range(0, M, tiles.tm):
                for j in range(0, y.shape[1], tiles.tn):
                    acc = jnp.zeros((tiles.tm, tiles.tn), jnp.float32)
                    for k in range(0, K, tiles.tk):
                        acc += x[i : i + tiles.tm, k : k + tiles.tk] @ y[k : k + tiles.tk, j : j + tiles.tn]
                    out = out.at[i : i + tiles.tm, j : j + tiles.tn].set(acc)
            return out

    Called without its tunable, a kernel tunes it on the TPU (see tune.py). Raises KernelError, naming the
    line, for code it can't compile.
    """
    return Kernel(fn)
