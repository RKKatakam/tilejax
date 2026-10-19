"""The TPU a kernel targets: its memories, bandwidth and compute rates, and Mosaic's layout rules.

Numbers come from JAX's own `pltpu.get_tpu_info()`: for the attached TPU when there is one, otherwise for
a named chip (`target="v5e"`), so kernels can be planned and tuned on any machine.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax._src import tpu_info as _tpu_info
from jax.experimental.pallas import tpu as pltpu

from .errors import AutoKernelError

__all__ = ["DEFAULT_TARGET", "Hardware", "check_block", "nbytes", "on_tpu", "sublanes", "target"]

DEFAULT_TARGET = "v5e"
# Chips whose logical device spans two TensorCores (megacore); work can be split between them.
_MEGACORE = {"v4": 2, "v5p": 2}


@dataclass(frozen=True)
class Hardware:
    """Per-TensorCore resources of a TPU, and how many TensorCores a kernel can split its work across."""

    name: str
    cores: int
    vmem_bytes: int
    smem_bytes: int
    hbm_bytes_per_s: float
    mxu_flops_per_s: float  # bf16 matmul throughput
    vpu_ops_per_s: float  # elementwise throughput: one op per lane and sublane per cycle
    lanes: int
    sublanes: int

    @property
    def vmem_budget(self) -> int:
        """What a kernel may plan to use; the rest is left for Mosaic's internal scratch."""
        return int(self.vmem_bytes * 0.75)

    @property
    def smem_budget(self) -> int:
        return self.smem_bytes // 2

    def matmul_flops_per_s(self, dtype: Any, highest: bool) -> float:
        """f32 matmuls at HIGHEST precision take about six bf16 MXU passes; 8-bit ints run twice as fast."""
        dtype = np.dtype(dtype)
        if dtype.itemsize == 1:
            return 2 * self.mxu_flops_per_s
        if dtype.itemsize >= 4 and highest:
            return self.mxu_flops_per_s / 6
        return self.mxu_flops_per_s


@functools.cache
def target(name: str | None = None) -> Hardware:
    """The attached TPU (when `name` is None and there is one), else the named chip, e.g. "v5e"."""
    if name is None and on_tpu():
        info = pltpu.get_tpu_info()
        name = str(info.chip_version.value)
    else:
        name = name or DEFAULT_TARGET
        try:
            chip = _tpu_info.ChipVersion(name)
        except ValueError:
            known = ", ".join(c.value for c in _tpu_info.ChipVersion)
            raise AutoKernelError(f"unknown TPU target {name!r}; known: {known}") from None
        info = pltpu.get_tpu_info_for_chip(chip, _MEGACORE.get(name, 1))
    clock = info.bf16_ops_per_second / (info.num_mxus * info.mxu_column_size**2 * 2)
    return Hardware(
        name=name,
        cores=info.num_cores,
        vmem_bytes=info.vmem_capacity_bytes,
        smem_bytes=info.smem_capacity_bytes,
        hbm_bytes_per_s=info.mem_bw_bytes_per_second,
        mxu_flops_per_s=info.bf16_ops_per_second,
        vpu_ops_per_s=clock * info.num_lanes * info.num_sublanes,
        lanes=info.num_lanes,
        sublanes=info.num_sublanes,
    )


def on_tpu() -> bool:
    return jax.default_backend() == "tpu"


# --------------------------------------------------------------------------- layout rules

Shape = tuple[int, ...]


def sublanes(dtype: Any) -> int:
    """Minimum second-minor tile for a dtype: 8 rows of 32-bit values, packed for narrower types."""
    return 8 * max(1, 4 // np.dtype(dtype).itemsize)


def check_block(what: str, block: Shape, full: Shape, dtype: Any, lanes: int = 128) -> list[str]:
    """The TPU tiling violations of one tile (an empty list if it's fine)."""
    problems: list[str] = []
    if not block:
        return problems
    if block[-1] % lanes and block[-1] != full[-1]:
        problems.append(
            f"{what}: last dim {block[-1]} (of array dim {full[-1]}) must be a multiple of {lanes} or the "
            "whole dimension"
        )
    if len(block) >= 2:
        s = sublanes(dtype)
        if block[-2] % s and block[-2] != full[-2]:
            problems.append(
                f"{what}: second-to-last dim {block[-2]} (of array dim {full[-2]}) must be a multiple of {s} for "
                f"{np.dtype(dtype).name} or the whole dimension"
            )
    return problems


def nbytes(shape: Sequence[int], dtype: Any) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
