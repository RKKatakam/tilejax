"""autokernel: write a TPU v5e kernel as plain JAX loops over tiles; get a gridless Pallas kernel, tuned.

- `@autokernel.kernel` compiles a loop nest over output tiles (see README.md); the result is called like
  the function. Its last parameter may be a tunable (`@autokernel.tunable`): a small pytree of block sizes
  and other choices. Pass one and the kernel uses it; leave it out and autokernel tunes it on the TPU.
- `kernel.source(...)` is the generated Pallas source; `kernel.verify(...)` checks the kernel against the
  loop nest run as plain JAX; `kernel.tuning(...)` is what tuning tried and chose.
- `autokernel.profile(kernel, ...)` says where the time goes on a v5e, from `jax.named_scope` regions.
- `autokernel.pairs(x[...])` in kernel code reads a 16-bit tile's pairs of rows as the 32-bit words a TPU
  packs them into (`autokernel.unpairs` stores them back); `x[autokernel.rows(start, stop, n), ...]` reads
  (and stores) rows chosen when the kernel runs, as many as are valid.
- `xs = autokernel.all_gather(x, "x")` in a kernel's prologue is every chip's `x` along a mesh axis: the
  kernel, run inside `jax.shard_map`, computes on the rows it has while the others arrive over a ring of the
  chips (a collective matmul).
"""

from __future__ import annotations

from typing import Any

from . import profiling
from .api import Kernel, Lowered, kernel
from .collective import all_gather
from .errors import KernelError
from .packing import pairs, unpairs
from .pipeline import Pipeline
from .profiling import KernelProfile, enable_profiling
from .ragged import rows
from .tunable import choice, tunable
from .tune import Config, Trial, Tuning

__all__ = [
    "Config",
    "Kernel",
    "KernelError",
    "KernelProfile",
    "Lowered",
    "Pipeline",
    "Trial",
    "Tuning",
    "all_gather",
    "choice",
    "enable_profiling",
    "kernel",
    "pairs",
    "profile",
    "rows",
    "tunable",
    "unpairs",
]

__version__ = "0.9.0"

# Tuning reads trace regions, which libtpu records only if told before JAX starts its TPU backend.
if not profiling._backend_started():
    enable_profiling()


def profile(kernel: Kernel, *args: Any, runs: int = 5) -> KernelProfile:
    """Where `kernel`'s time goes on a TPU v5e for these inputs (and tunable, if given): device time, input
    waits, compute, output waits and the tile code's own `jax.named_scope`s."""
    return profiling.profile(kernel, args, runs=runs)
