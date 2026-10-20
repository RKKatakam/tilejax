"""Timing on a TPU: wall-clock time per call of a jitted function, after a warm-up call."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax


def time_ms(fn: Callable[..., Any], *args: Any, runs: int = 20, repeats: int = 5) -> float:
    """The median over `repeats` of the mean time per call over `runs` back-to-back calls, in ms."""
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs * 1e3)
    return sorted(samples)[len(samples) // 2]


def require_tpu() -> None:
    if jax.default_backend() != "tpu":
        raise SystemExit("benchmarks time a TPU v5e; this machine has none")
