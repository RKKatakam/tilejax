"""Block-size tuning: try every combination of the constants being tuned, rank by the cost model, and
optionally time the best few on the TPU.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from .errors import AutoKernelError
from .frontend import LoopNest
from .lowering import Lowered, Options, lower

__all__ = ["Candidate", "search", "time_kernel"]


@dataclass
class Candidate:
    overrides: dict[str, Any]
    buffers: int
    lowered: Lowered | None
    error: str | None = None  # why it can't run (invalid tiling, too much VMEM, ...)
    measured_s: float | None = None

    @property
    def predicted_s(self) -> float:
        return self.lowered.estimate.seconds if self.lowered else float("inf")


def search(nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct], options: Options) -> list[Candidate]:
    """Every candidate, valid ones first, each group fastest-predicted first."""
    names = [name for name, _ in options.tune]
    grids = [values for _, values in options.tune]
    depths = [options.buffers] if options.buffers else [2, 3] if options.autotune == "measure" else [2]
    candidates = []
    for values in itertools.product(*grids):
        overrides = dict(zip(names, values, strict=True))
        for buffers in depths:
            try:
                candidates.append(Candidate(overrides, buffers, lower(nest, fn, specs, options, overrides, buffers)))
            except AutoKernelError as e:
                candidates.append(Candidate(overrides, buffers, None, " ".join(str(e).split())))
    candidates.sort(key=lambda c: (c.lowered is None, c.predicted_s))
    if not candidates or candidates[0].lowered is None:
        reasons = "\n  ".join(f"{c.overrides} buffers={c.buffers}: {c.error}" for c in candidates[:8])
        raise AutoKernelError(f"{nest.fn_name}: no tuning candidate is valid:\n  {reasons}")
    return candidates


def time_kernel(lowered: Lowered, args: Sequence[Any], runs: int = 10, repeats: int = 3) -> float:
    """Median seconds per call on the attached TPU, after a compiling warm-up."""
    fn = jax.jit(lambda *xs: lowered.entry(*xs, interpret=False))
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs)
    return float(np.median(samples))
