"""The `@autokernel.tpu` decorator and the `TpuKernel` it returns."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from . import frontend, hardware
from . import tune as tune_lib
from .errors import AutoKernelError
from .lowering import Lowered, Options, lower

__all__ = ["TpuKernel", "tpu"]

_Signature = tuple[tuple[tuple[int, ...], np.dtype], ...]


def _spec(x: Any) -> jax.ShapeDtypeStruct:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    if not hasattr(x, "shape") or not hasattr(x, "dtype"):
        raise AutoKernelError(f"kernel arguments must be arrays, got {type(x).__name__}")
    return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)


def _signature(args: tuple[Any, ...]) -> _Signature:
    return tuple((tuple(s.shape), np.dtype(s.dtype)) for s in map(_spec, args))


def _concrete(args: tuple[Any, ...]) -> bool:
    return all(isinstance(a, jax.Array) and not isinstance(a, jax.core.Tracer) for a in args)


def _default_tolerance(dtype: Any) -> float:
    return 2e-2 if np.dtype(dtype).itemsize <= 2 else 1e-4


@functools.cache
def _warn_interpreting() -> None:
    warnings.warn(
        f"autokernel: no TPU found (backend={jax.default_backend()!r}); running kernels in the Pallas TPU interpreter",
        stacklevel=5,
    )


class TpuKernel:
    """A Python tile loop nest compiled to a Pallas TPU kernel.

    Calling it runs the generated kernel. Other entry points:

    - `lower(*args)`: the `Lowered` kernel (schedule, tiles, cost estimate, generated `.source`)
    - `source(*args)`: the generated Pallas source
    - `reference(*args)`: run the original Python loop nest, the semantic ground truth
    - `verify(*args)`: run both and raise if they disagree; returns the max abs error
    - `tuning(*args)`: every tuning candidate tried, with predicted (and measured) times

    `args` for `lower`, `source` and `tuning` may be arrays or `jax.ShapeDtypeStruct`s. Kernels are
    generated, and tuned, once per input signature (shapes and dtypes).
    """

    def __init__(self, fn: Callable, *, interpret: Any = None, verify: bool = False, options: Options | None = None):
        self.fn = fn
        self.nest = frontend.parse(fn)  # fail at decoration time on unsupported code
        self.interpret = interpret
        self.verify_on_first_call = verify
        self.options = options or Options()
        self._lowered: dict[_Signature, Lowered] = {}
        self._tuning: dict[_Signature, list[tune_lib.Candidate]] = {}
        self._compiled: dict[_Signature, Callable[..., Any]] = {}
        functools.update_wrapper(self, fn)

    def _interpret_mode(self) -> Any:
        if self.interpret is None:
            if hardware.on_tpu():
                return False
            _warn_interpreting()
            return pltpu.InterpretParams()
        if self.interpret is True:
            return pltpu.InterpretParams()
        if self.interpret is False and not hardware.on_tpu():
            raise AutoKernelError(
                f"autokernel only targets TPU, but the default backend is {jax.default_backend()!r}; "
                "pass interpret=True to simulate it"
            )
        return self.interpret

    def lower(self, *args: Any) -> Lowered:
        key = _signature(args)
        if key not in self._lowered:
            specs = [_spec(a) for a in args]
            if self.options.tune:
                self._lowered[key] = self._tune(key, specs, args)
            else:
                self._lowered[key] = lower(self.nest, self.fn, specs, self.options)
        return self._lowered[key]

    def _tune(self, key: _Signature, specs: list[jax.ShapeDtypeStruct], args: tuple[Any, ...]) -> Lowered:
        candidates = tune_lib.search(self.nest, self.fn, specs, self.options)
        self._tuning[key] = candidates
        valid = [c for c in candidates if c.lowered is not None]
        if self.options.autotune == "measure" and hardware.on_tpu() and _concrete(args):
            for c in valid[: self.options.measure_top]:
                try:
                    c.measured_s = tune_lib.time_kernel(c.lowered, args)
                except Exception as e:  # e.g. Mosaic rejects it: drop the candidate
                    c.error = f"failed to run: {type(e).__name__}"
            timed = [c for c in valid if c.measured_s is not None]
            if timed:
                return min(timed, key=lambda c: c.measured_s).lowered
        return valid[0].lowered

    def tuning(self, *args: Any) -> list[tune_lib.Candidate]:
        self.lower(*args)
        return self._tuning.get(_signature(args), [])

    def source(self, *args: Any) -> str:
        return self.lower(*args).source

    def reference(self, *args: Any) -> Any:
        if self.options.matmul_precision == "default":
            return self.fn(*args)
        with jax.default_matmul_precision("highest"):
            return self.fn(*args)

    def __call__(self, *args: Any) -> Any:
        lowered = self.lower(*args)
        key = _signature(args)
        first = key not in self._compiled
        if first:
            self._compiled[key] = jax.jit(functools.partial(lowered.entry, interpret=self._interpret_mode()))
        out = self._compiled[key](*args)
        if first and self.verify_on_first_call and _concrete(args):
            self._check(out, self.reference(*args))
        return out

    def verify(self, *args: Any, rtol: float | None = None, atol: float | None = None) -> float:
        return self._check(self(*args), self.reference(*args), rtol, atol)

    def _check(self, got: Any, want: Any, rtol: float | None = None, atol: float | None = None) -> float:
        gots = got if isinstance(got, tuple) else (got,)
        wants = want if isinstance(want, tuple) else (want,)
        return max(self._check_one(g, w, rtol, atol) for g, w in zip(gots, wants, strict=True))

    def _check_one(self, got: jax.Array, want: jax.Array, rtol: float | None, atol: float | None) -> float:
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(
                f"{self.__name__}: kernel returned {got.dtype}{got.shape}, reference returned {want.dtype}{want.shape}"
            )
        g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
        tol = _default_tolerance(want.dtype)
        if self.options.matmul_precision == "default" and hardware.on_tpu():
            tol = max(tol, 2e-2)  # single-pass bf16 MXU matmuls
        if atol is None:
            # Low-precision references round at every step (the kernel accumulates in f32), so their
            # error grows with the output's magnitude: compare against that instead of per element.
            low = np.dtype(want.dtype).itemsize <= 2 or tol > _default_tolerance(want.dtype)
            scale = float(np.max(np.abs(w))) if w.size and low else 1.0
            atol = tol * max(scale, 1.0)
        rtol = rtol if rtol is not None else tol
        with np.errstate(invalid="ignore"):  # inf - inf; equal values (infinities included) are exact
            err = float(np.max(np.abs(np.where(g == w, 0.0, g - w)))) if g.size else 0.0
        if not np.allclose(g, w, rtol=rtol, atol=atol):
            bad = int(np.sum(~np.isclose(g, w, rtol=rtol, atol=atol)))
            raise AssertionError(
                f"{self.__name__}: Pallas kernel disagrees with the Python reference in {bad}/{g.size} elements "
                f"(max abs err {err:.3g}, rtol={rtol}, atol={atol})"
            )
        return err


def tpu(
    fn: Callable | None = None,
    *,
    interpret: Any = None,
    verify: bool = False,
    tune: Mapping[str, Sequence[Any]] | None = None,
    autotune: str = "model",
    buffers: int | None = None,
    target: str | None = None,
    cores: int | None = None,
    check_tiling: bool = True,
    vmem_limit_bytes: int | None = None,
    matmul_precision: str = "highest",
    background_fill: bool = True,
) -> Any:
    """Compile a Python tile loop nest into a Pallas TPU kernel.

    Use it bare or with options::

        @autokernel.tpu
        def matmul(a, b): ...

        @autokernel.tpu(tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (256, 512)})
        def matmul(a, b): ...  # TM, TN, TK are module constants the loops use

    Args:
        interpret: None (default) runs natively on TPU and in the Pallas TPU interpreter elsewhere;
            False requires a real TPU; True or a `pltpu.InterpretParams` always interprets.
        verify: on the first call per input signature, compare against the original Python code.
        tune: constants to tune (tile sizes, usually) mapped to candidate values. Every combination is
            tried; invalid ones (misaligned tiles, too much VMEM) are dropped and the rest ranked by a cost
            model built from the TPU's specs and JAX's cost analysis.
        autotune: "model" picks the best predicted candidate; "measure" also times the best few on the
            TPU on the first call with concrete arrays, and searches buffer depths 2 and 3.
        buffers: VMEM slots per streamed input (2 = double buffering). Default 2, or tuned.
        target: the TPU to plan for when not running on one, e.g. "v5e" (default) or "v5p".
        cores: TensorCores to split the work across (default: all of the target's, 2 on v4 and v5p).
        check_tiling: reject tiles that break TPU (8, 128) layout rules before Mosaic does.
        vmem_limit_bytes: Mosaic's scoped-VMEM limit; by default sized from the kernel's estimated use.
        matmul_precision: "highest" (default) makes f32 `@` a full-precision multi-pass MXU matmul;
            "default" uses Mosaic's faster single bf16 pass (verify then uses looser tolerances).
        background_fill: with a static tile mask, write the skipped tiles' constant value with async DMAs
            from the kernel instead of filling the whole output before the kernel runs.
    """
    options = Options(
        check_tiling=check_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        matmul_precision=matmul_precision,
        background_fill=background_fill,
        buffers=buffers,
        target=target,
        cores=cores,
        tune=Options.make_tune(tune),
        autotune=autotune,
    )

    def wrap(f: Callable) -> TpuKernel:
        return TpuKernel(f, interpret=interpret, verify=verify, options=options)

    return wrap(fn) if fn is not None else wrap
