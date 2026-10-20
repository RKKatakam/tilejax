"""`@autokernel.kernel`: a Python loop nest, compiled to a gridless Pallas kernel for a TPU v5e."""

from __future__ import annotations

import functools
import itertools
import linecache
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
from .pipeline import Pipeline, default_pipeline, fits, vmem_bytes
from .plan import Plan, make_plan
from .reference import make_reference
from .schedule import Schedule, make_schedule

__all__ = ["Kernel", "Lowered", "kernel"]

Signature = tuple[tuple[tuple[int, ...], np.dtype], ...]

_generated = itertools.count()  # numbers the generated sources' file names, which linecache keys by


@dataclass(frozen=True)
class Lowered:
    """A kernel generated for one input signature: its plan, schedule, pipeline and Pallas source."""

    plan: Plan
    schedule: Schedule
    pipeline: Pipeline
    source: str
    trace: bool  # does it mark trace regions (for profiling)
    entry: Callable[..., Any] = field(repr=False)  # entry(*inputs, interpret=False): the pallas_call

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


@dataclass(frozen=True)
class _Build:
    """What one input signature fixes: the plan, its schedule, and the reference (constants read now)."""

    plan: Plan
    schedule: Schedule
    reference: Callable[..., Any]


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

    The kernel is built when it is first called with inputs of a new shape and dtype, reading the
    constants its code uses at that moment (as `jax.jit` does). It runs natively on a TPU v5e and in the
    Pallas TPU interpreter elsewhere.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.nest: Nest = parse(fn)
        functools.update_wrapper(self, fn)
        self._builds: dict[Signature, _Build] = {}
        self._lowered: dict[tuple[Signature, Pipeline, bool], Lowered] = {}
        self._pipelines: dict[Signature, Pipeline] = {}  # what `optimize` chose

    def __repr__(self) -> str:
        return f"<autokernel.Kernel {self.nest.name}>"

    def __call__(self, *inputs: Any) -> Any:
        return self.lower(*inputs).run(*inputs)

    def lower(self, *inputs: Any) -> Lowered:
        """The kernel for inputs like these (arrays or `jax.ShapeDtypeStruct`s)."""
        signature = self.signature(*inputs)
        return self.lowered(signature, self.pipeline(signature))

    def source(self, *inputs: Any) -> str:
        """The generated Pallas source for inputs like these."""
        return self.lower(*inputs).source

    def reference(self, *inputs: Any) -> Any:
        """What the kernel computes, by running the loop nest as plain JAX (with `@` on the MXU)."""
        return self._build(self.signature(*inputs)).reference(*inputs)

    def verify(self, *inputs: Any, rtol: float | None = None, atol: float | None = None) -> float:
        """Run the kernel and its reference on `inputs`; raise AssertionError unless they agree (within
        tolerances for the dtypes, or `rtol` and `atol`). Returns the largest absolute difference."""
        got, want = self(*inputs), self.reference(*inputs)
        got, want = (got, want) if isinstance(got, tuple) else ((got,), (want,))
        errors = [
            compare(f"{self.nest.name} output {name!r}", g, w, rtol, atol)
            for name, g, w in zip(self.nest.outputs, got, want, strict=True)
        ]
        return max(errors)

    # ------------------------------------------------------------------ building (used by profiling too)

    def signature(self, *inputs: Any) -> Signature:
        """What the kernel is built for: the inputs' shapes and dtypes."""
        if len(inputs) != len(self.nest.params):
            raise KernelError(
                f"{self.nest.name} takes {len(self.nest.params)} inputs ({', '.join(self.nest.params)}), "
                f"got {len(inputs)}"
            )
        signature = []
        for name, x in zip(self.nest.params, inputs, strict=True):
            if not (hasattr(x, "shape") and hasattr(x, "dtype")) or not tuple(x.shape):
                raise KernelError(f"{self.nest.name}: input `{name}` must be an array, got {type(x).__name__}")
            signature.append((tuple(x.shape), np.dtype(x.dtype)))
        return tuple(signature)

    def pipeline(self, signature: Signature) -> Pipeline:
        """The buffering for inputs of this signature: what `optimize` chose, else the default."""
        if signature not in self._pipelines:
            build = self._build(signature)
            self._pipelines[signature] = default_pipeline(build.plan, build.schedule)
        return self._pipelines[signature]

    def choose(self, signature: Signature, pipeline: Pipeline) -> None:
        """Use `pipeline` for inputs of this signature from now on; raises KernelError if it doesn't fit."""
        build = self._build(signature)
        if not fits(build.plan, build.schedule, pipeline):
            raise KernelError(f"{self.nest.name}: {pipeline} don't fit in VMEM for these inputs")
        self._pipelines[signature] = pipeline

    def lowered(self, signature: Signature, pipeline: Pipeline, trace: bool = False) -> Lowered:
        """The kernel for this signature with this buffering (and trace regions, for profiling)."""
        key = (signature, pipeline, trace)
        if key not in self._lowered:
            build = self._build(signature)
            source, entry_name = codegen.generate(build.plan, build.schedule, pipeline, trace=trace)
            entry = _load(source, f"<autokernel {self.nest.name} #{next(_generated)}>", entry_name)
            self._lowered[key] = Lowered(build.plan, build.schedule, pipeline, source, trace, entry)
        return self._lowered[key]

    def _build(self, signature: Signature) -> _Build:
        if signature not in self._builds:
            specs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in signature]
            plan = make_plan(self.nest, self.fn, specs)
            self._builds[signature] = _Build(plan, make_schedule(plan), make_reference(self.nest, self.fn))
        return self._builds[signature]


def _load(source: str, filename: str, entry: str) -> Callable[..., Any]:
    """The function `entry` of generated source, compiled under `filename` (registered with linecache, so
    tracebacks through the kernel show its lines)."""
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {"__name__": "autokernel.generated"}
    exec(compile(source, filename, "exec"), namespace)
    return namespace[entry]


def compare(what: str, got: Any, want: Any, rtol: float | None = None, atol: float | None = None) -> float:
    """The largest absolute difference between `got` and `want`; AssertionError if they disagree beyond
    `rtol` and `atol` (default: exact for integers, 1e-4 for f32, 2e-2 of the largest value for 16-bit)."""
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    dtype = np.dtype(want.dtype)
    if not jnp.issubdtype(dtype, jnp.inexact):
        default = 0.0
    elif dtype.itemsize <= 2:
        default = 2e-2
    else:
        default = 1e-4
    rtol = default if rtol is None else rtol
    if atol is None:
        # 16-bit results round at every step: their error grows with the output's magnitude.
        finite = np.abs(w[np.isfinite(w)])
        scale = float(finite.max()) if default >= 2e-2 and finite.size else 1.0
        atol = default * max(scale, 1.0)
    same = (g == w) | (np.isnan(g) & np.isnan(w))
    with np.errstate(invalid="ignore"):
        error = float(np.max(np.abs(np.where(same, 0.0, g - w)), initial=0.0))
    close = np.isclose(g, w, rtol=rtol, atol=atol, equal_nan=True)
    if not close.all():
        raise AssertionError(
            f"{what}: the kernel and its reference differ in {int((~close).sum())} of {g.size} elements "
            f"(largest difference {error:.3g}; rtol={rtol}, atol={atol:.3g})"
        )
    return error


def kernel(fn: Callable[..., Any]) -> Kernel:
    """Compile a Python loop nest over tiles into a gridless Pallas kernel for a TPU v5e.

        @autokernel.kernel
        def matmul(x, y):
            out = jnp.zeros((M, N), x.dtype)
            for i in range(0, M, TM):
                for j in range(0, N, TN):
                    acc = jnp.zeros((TM, TN))
                    for k in range(0, K, TK):
                        acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
                    out = out.at[i : i + TM, j : j + TN].set(acc)
            return out

    Raises KernelError, naming the line, for code it can't compile.
    """
    return Kernel(fn)
