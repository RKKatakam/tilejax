"""TPU v5e, the one chip autokernel targets: its memories and Mosaic's layout rules.

Numbers are JAX's own (`pltpu.get_tpu_info_for_chip(ChipVersion("v5e"))`) unless a comment says they were
measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

__all__ = [
    "DEVICE_KIND",
    "LANES",
    "SMEM_BUDGET",
    "VMEM_BUDGET",
    "VMEM_LIMIT",
    "layout_problems",
    "on_tpu",
    "padded_bytes",
    "sublanes",
]

DEVICE_KIND = "TPU v5 lite"
LANES = 128
VMEM_BYTES = 128 * 2**20
SMEM_BYTES = 2**20

# Mosaic's scoped-VMEM limit for every kernel. Mosaic takes its own scratch (spilled intermediates) from
# what the kernel's buffers leave under this limit: limits sized from a kernel's own buffers made
# attention-heavy kernels up to 2.7% slower on a v5e (measured).
VMEM_LIMIT = int(VMEM_BYTES * 0.9)
# What a kernel's own buffers (input rings, output slots, carried values) may take.
VMEM_BUDGET = int(VMEM_BYTES * 0.75)
# What step tables may take in SMEM: 961 KiB of tables compiled and ran on a v5e, 1026 KiB ran out of SMEM
# (measured); the rest holds semaphores and Mosaic's own scalars.
SMEM_BUDGET = SMEM_BYTES - 64 * 1024


def on_tpu() -> bool:
    """Is JAX's default backend a TPU?"""
    return jax.default_backend() == "tpu"


def sublanes(dtype: Any) -> int:
    """Rows in one (sublane, lane) tile of `dtype`: 8 for 32-bit types, packed twice as deep per halving."""
    return 8 * max(1, 4 // np.dtype(dtype).itemsize)


def padded_bytes(shape: Sequence[int], dtype: Any) -> int:
    """VMEM bytes a buffer takes: Mosaic pads its last two dims to whole (sublanes, 128) tiles."""
    dims = list(shape) or [1]
    dims[-1] = -(-dims[-1] // LANES) * LANES
    if len(dims) >= 2:
        rows = sublanes(dtype)
        dims[-2] = -(-dims[-2] // rows) * rows
    return math.prod(dims) * np.dtype(dtype).itemsize


def layout_problems(tile: Sequence[int], array: Sequence[int], dtype: Any) -> list[str]:
    """Why a VMEM tile of an array breaks Mosaic's layout rules (an empty list if it doesn't).

    `tile` and `array` are the tile's dims and the array dims they slice. The last dim must be a multiple of
    128 lanes, the second-to-last a multiple of the dtype's sublanes, each unless it is the whole dim.
    """
    problems = []
    if tile[-1] % LANES and tile[-1] != array[-1]:
        problems.append(f"its last dim, {tile[-1]}, must be a multiple of {LANES} or all {array[-1]} elements")
    if len(tile) >= 2:
        rows = sublanes(dtype)
        if tile[-2] % rows and tile[-2] != array[-2]:
            problems.append(
                f"its second-to-last dim, {tile[-2]}, must be a multiple of {rows} for {np.dtype(dtype).name} "
                f"or all {array[-2]} elements"
            )
    return problems
