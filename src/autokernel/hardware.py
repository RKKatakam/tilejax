"""TPU (Mosaic) hardware rules that every generated kernel must satisfy."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

__all__ = [
    "DEFAULT_VMEM_BUDGET",
    "LANES",
    "SMEM_CAPACITY",
    "check_block",
    "check_smem",
    "check_vmem",
    "on_tpu",
    "sublanes",
]

LANES = 128
# Default scoped-VMEM budget Mosaic gives a kernel (v5e: 16 MiB; newer chips allow more).
DEFAULT_VMEM_BUDGET = 16 * 1024 * 1024

# Scalar memory per TensorCore on v4 and newer (v2/v3 have 16 KiB).
SMEM_CAPACITY = 1024 * 1024

Shape = tuple[int, ...]


def sublanes(dtype: Any) -> int:
    """Minimum second-minor tile for a dtype: 8 rows of 32-bit values, packed for narrower types."""
    itemsize = np.dtype(dtype).itemsize
    return 8 * max(1, 4 // itemsize)


def check_block(what: str, block: Shape, full: Shape, dtype: Any) -> list[str]:
    """The TPU tiling violations of one BlockSpec (an empty list if it's fine)."""
    problems: list[str] = []
    if not block:
        return problems
    if block[-1] % LANES and block[-1] != full[-1]:
        problems.append(
            f"{what}: last block dim {block[-1]} (of array dim {full[-1]}) must be a multiple of "
            f"{LANES} or the whole dimension"
        )
    if len(block) >= 2:
        s = sublanes(dtype)
        if block[-2] % s and block[-2] != full[-2]:
            problems.append(
                f"{what}: second-to-last block dim {block[-2]} (of array dim {full[-2]}) must be a "
                f"multiple of {s} for {np.dtype(dtype).name} or the whole dimension"
            )
    return problems


def nbytes(shape: Shape, dtype: Any) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def check_vmem(pipelined: Sequence[tuple[Shape, Any]], scratch: Sequence[tuple[Shape, Any]], budget: int | None) -> int:
    """Estimate VMEM use (pipelined blocks are double-buffered) and warn when it exceeds the budget."""
    used = sum(2 * nbytes(s, d) for s, d in pipelined) + sum(nbytes(s, d) for s, d in scratch)
    limit = budget or DEFAULT_VMEM_BUDGET
    if used > limit:
        warnings.warn(
            f"autokernel: estimated VMEM use {used / 2**20:.1f} MiB exceeds the {limit / 2**20:.0f} MiB "
            "scoped-VMEM budget; use smaller tiles or pass vmem_limit_bytes=",
            stacklevel=2,
        )
    return used


def check_smem(used: int) -> None:
    """Tile tables live in SMEM; reject kernels whose tables can't fit next to Mosaic's own use."""
    if used > SMEM_CAPACITY // 2:
        from .errors import AutoKernelError

        raise AutoKernelError(
            f"tile tables need {used / 1024:.0f} KiB of SMEM (limit {SMEM_CAPACITY // 2 // 1024} KiB); use bigger tiles"
        )


def on_tpu() -> bool:
    return jax.default_backend() == "tpu"
