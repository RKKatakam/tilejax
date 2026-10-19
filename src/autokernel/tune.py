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

__all__ = ["Candidate", "measure", "search", "time_kernel"]


@dataclass
class Candidate:
    overrides: dict[str, Any]
    buffers: int | None  # None: the default depth, for a candidate that couldn't be lowered
    lowered: Lowered | None
    error: str | None = None  # why it can't run (invalid tiling, too much VMEM, ...)
    measured_s: float | None = None

    @property
    def predicted_s(self) -> float:
        return self.lowered.estimate.seconds if self.lowered else float("inf")


def search(nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct], options: Options) -> list[Candidate]:
    """Every candidate, valid ones first, each group fastest-predicted first.

    Each block-size combination is lowered at the default buffer depth; a measured search also lowers
    it at the other depth (the cost model can't tell the depths apart, only timing can).
    """
    names = [name for name, _ in options.tune]
    grids = [values for _, values in options.tune]
    candidates = []
    for values in itertools.product(*grids):
        overrides = dict(zip(names, values, strict=True))
        try:
            lowered = lower(nest, fn, specs, options, overrides)
        except AutoKernelError as e:
            candidates.append(Candidate(overrides, options.buffers, None, " ".join(str(e).split())))
            continue
        candidates.append(Candidate(overrides, lowered.buffers, lowered))
        if options.autotune == "measure" and not options.buffers and lowered.buffers == 3:
            candidates.append(Candidate(overrides, 2, lower(nest, fn, specs, options, overrides, 2)))
    candidates.sort(key=lambda c: (c.lowered is None, c.predicted_s))
    if not candidates or candidates[0].lowered is None:
        reasons = "\n  ".join(f"{c.overrides} buffers={c.buffers}: {c.error}" for c in candidates[:8])
        raise AutoKernelError(f"{nest.fn_name}: no tuning candidate is valid:\n  {reasons}")
    return candidates


def measure(candidates: list[Candidate], args: Sequence[Any], top: int) -> Lowered | None:
    """Time the `top` best-predicted block sizes on the attached TPU, then the other buffer depth of the
    winner; the fastest, or None if none ran."""
    valid = [c for c in candidates if c.lowered is not None]
    shapes = list(dict.fromkeys(tuple(sorted(c.overrides.items())) for c in valid))  # predicted order

    def time_candidate(c: Candidate) -> None:
        try:
            c.measured_s = time_kernel(c.lowered, args)
        except Exception as e:  # e.g. Mosaic rejects it: drop the candidate
            c.error = f"failed to run: {type(e).__name__}"

    for shape in shapes[:top]:  # each shape at its best-predicted (default) depth
        time_candidate(next(c for c in valid if tuple(sorted(c.overrides.items())) == shape))
    timed = [c for c in valid if c.measured_s is not None]
    if not timed:
        return None
    best = min(timed, key=lambda c: c.measured_s)
    for c in valid:  # the winner's other depths
        if c.overrides == best.overrides and c.measured_s is None and c.error is None:
            time_candidate(c)
    timed = [c for c in valid if c.measured_s is not None]
    return min(timed, key=lambda c: c.measured_s).lowered


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
