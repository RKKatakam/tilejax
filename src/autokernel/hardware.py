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

__all__ = [
    "DEFAULT_TARGET",
    "DEFAULT_VPU_INT_BITS",
    "VALU_SLOTS",
    "VPU_INT_BITS",
    "Hardware",
    "check_block",
    "nbytes",
    "on_tpu",
    "sublanes",
    "target",
    "vmem_nbytes",
]

DEFAULT_TARGET = "v5e"
# Chips whose logical device spans two TensorCores (megacore); work can be split between them.
_MEGACORE = {"v4": 2, "v5p": 2}
# Integer widths the VPU does arithmetic on (elementwise ops, comparisons, reductions): narrower integer
# vectors must be widened first (`.astype(jnp.int32)`); loading, storing, converting them and feeding
# them to the MXU is fine. Mosaic on a v5e rejects 8-bit vector arithmetic ("Only vector<i16> and
# vector<i32> are supported"). JAX's own sources don't say what other chips accept (Mosaic's rules live
# in libtpu), so every chip gets the v5e's rule until measured otherwise.
VPU_INT_BITS = {"v4": (16, 32), "v5e": (16, 32), "v5p": (16, 32), "v6e": (16, 32)}
DEFAULT_VPU_INT_BITS = (16, 32)
# Vector-ALU slots per VLIW bundle: whole-vreg elementwise ops the VPU issues per cycle. Fitted to the
# attention timings measured on a v5e (see cost.py); the EUP (transcendentals) issues one more per cycle.
VALU_SLOTS = 4
# TensorCore clocks, Hz. JAX's tables don't list clocks; bf16 rate / (MXUs x MXU size^2 x 2) gives them for the
# chips with 128x128 MXUs (published: v4 1050 MHz; v5e 1.5 GHz; v5p 1.75 GHz), but JAX lists two 256x256
# MXUs for v6e (918 TFLOP/s), which would mean a 3.5 GHz clock: Trillium has four, at about 1.75 GHz.
# Chips not listed here use the formula.
# TODO: newer chips (v7, ...) once their clocks and MXU counts are published.
_CLOCK_HZ = {"v4": 1.05e9, "v5e": 1.5e9, "v5p": 1.75e9, "v6e": 1.75e9}


@dataclass(frozen=True)
class Hardware:
    """Per-TensorCore resources of a TPU, and how many TensorCores a kernel can split its work across."""

    name: str
    cores: int
    vmem_bytes: int
    smem_bytes: int
    hbm_bytes_per_s: float
    mxu_flops_per_s: float  # bf16 matmul throughput
    int8_flops_per_s: float  # 8-bit integer matmul throughput (the bf16 rate on chips without a faster int8 path)
    fp8_flops_per_s: float  # 8-bit float matmul throughput (likewise)
    clock_hz: float
    valu_ops_per_s: float  # vector-ALU throughput, in whole-vreg (sublanes x lanes) ops per second
    eup_ops_per_s: float  # transcendental unit (exp, log, reciprocal, ...) throughput, in vreg ops per second
    vpu_bf16: bool  # the VPU computes on packed 16-bit vregs (v6 and newer); older chips unpack them to 32 bits
    lanes: int
    sublanes: int
    vpu_int_bits: tuple[int, ...] = DEFAULT_VPU_INT_BITS  # integer widths the VPU does arithmetic on

    @property
    def vmem_budget(self) -> int:
        """What a kernel may plan to use; the rest is left for Mosaic's internal scratch."""
        return int(self.vmem_bytes * 0.75)

    @property
    def vmem_limit(self) -> int:
        """Mosaic's scoped-VMEM limit for a kernel: the chip's, not the kernel's estimate. Mosaic takes its own
        scratch (spilled intermediates, relayouts) from what the limit leaves: on a v5e, sizing the limit from
        the estimate made attention-heavy kernels up to 2.7% slower."""
        return int(self.vmem_bytes * 0.9)

    @property
    def smem_budget(self) -> int:
        return self.smem_bytes // 2

    def matmul_flops_per_s(self, dtype: Any, highest: bool) -> float:
        """f32 matmuls at HIGHEST precision take about six bf16 MXU passes; 8-bit types run at the chip's rate."""
        dtype = np.dtype(dtype)
        if dtype.itemsize == 1:
            return self.int8_flops_per_s if np.issubdtype(dtype, np.integer) else self.fp8_flops_per_s
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
    clock = _CLOCK_HZ.get(name) or info.bf16_ops_per_second / (info.num_mxus * info.mxu_column_size**2 * 2)
    return Hardware(
        name=name,
        cores=info.num_cores,
        vmem_bytes=info.vmem_capacity_bytes,
        smem_bytes=info.smem_capacity_bytes,
        hbm_bytes_per_s=info.mem_bw_bytes_per_second,
        mxu_flops_per_s=info.bf16_ops_per_second,
        int8_flops_per_s=max(info.int8_ops_per_second, info.bf16_ops_per_second),
        fp8_flops_per_s=max(info.fp8_ops_per_second, info.bf16_ops_per_second),
        clock_hz=clock,
        valu_ops_per_s=clock * VALU_SLOTS,
        eup_ops_per_s=clock,
        vpu_bf16=info.generation >= 6,
        lanes=info.num_lanes,
        sublanes=info.num_sublanes,
        vpu_int_bits=VPU_INT_BITS.get(name, DEFAULT_VPU_INT_BITS),
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


def vmem_nbytes(shape: Sequence[int], dtype: Any, lanes: int = 128) -> int:
    """Bytes a VMEM buffer of this shape takes: Mosaic pads the last two dims to (sublanes, lanes) tiles."""
    shape = tuple(shape)
    if not shape:
        return nbytes((1,), dtype)  # scalars live in SMEM, but count a word
    padded = list(shape)
    padded[-1] = -(-padded[-1] // lanes) * lanes
    if len(padded) >= 2:
        s = sublanes(dtype)
        padded[-2] = -(-padded[-2] // s) * s
    return nbytes(padded, dtype)
