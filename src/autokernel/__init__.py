"""autokernel: write a TPU v5e kernel as Python loops over tiles; get a gridless Pallas kernel.

- `@autokernel.kernel` compiles a loop nest over output tiles (see README.md); call the result like the
  function: it runs the generated kernel (in the Pallas TPU interpreter off-TPU).
- `kernel.source(*inputs)` is the generated Pallas source; `kernel.verify(*inputs)` checks the kernel
  against the loop nest run as plain JAX.
- `autokernel.profile(kernel, *inputs)` says where the time goes on a v5e, from `jax.named_scope` trace
  regions; `autokernel.optimize(kernel, *inputs)` chooses the kernel's buffering from them.
"""

from __future__ import annotations

from typing import Any

from . import profiling
from .api import Kernel, Lowered, kernel
from .errors import KernelError
from .pipeline import Pipeline
from .profiling import KernelProfile, Optimization, enable_profiling

__all__ = [
    "Kernel",
    "KernelError",
    "KernelProfile",
    "Lowered",
    "Optimization",
    "Pipeline",
    "enable_profiling",
    "kernel",
    "optimize",
    "profile",
]

__version__ = "0.6.0"


def profile(kernel: Kernel, *inputs: Any, runs: int = 5) -> KernelProfile:
    """Where `kernel`'s time goes on a TPU v5e for these inputs: device time, input waits, compute, output
    waits and the tile code's own `jax.named_scope`s. Call `enable_profiling()` first for the breakdown."""
    return profiling.profile(kernel, inputs, runs=runs)


def optimize(kernel: Kernel, *inputs: Any, runs: int = 5) -> Optimization:
    """Choose `kernel`'s buffering for inputs like these from its profiles on a TPU v5e (needs
    `enable_profiling()`); the kernel uses it from then on."""
    return profiling.optimize(kernel, inputs, runs=runs)
