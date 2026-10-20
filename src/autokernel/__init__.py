"""autokernel: compile Python tile loop nests into Pallas TPU kernels.

Write the tiling as ordinary Python loops and decorate it::

    import jax.numpy as jnp
    import autokernel

    @autokernel.tpu
    def matmul(x, y):
        out = jnp.zeros((M, N))
        for i in range(0, M, TM):
            for j in range(0, N, TN):
                acc = jnp.zeros((TM, TN))
                for k in range(0, K, TK):
                    acc += x[i:i + TM, k:k + TK] @ y[k:k + TK, j:j + TN]
                out = out.at[i:i + TM, j:j + TN].set(acc)
        return out

Calling `matmul(x, y)` runs a generated gridless Pallas kernel; `matmul.source(x, y)` shows its code
and `matmul.verify(x, y)` checks it against the Python loops.

A function with several tile loop nests mixed with ordinary JAX code and Python control flow (a whole
model) becomes a `TpuProgram`: every nest is its own kernel, and the code around them runs as JAX.
"""

from importlib import metadata as _metadata

from .api import TpuKernel, tpu
from .errors import AutoKernelError
from .fusion import FusedLowered
from .lowering import Lowered
from .program import ProgramLowered, ProgramNest, TpuProgram

__all__ = [
    "AutoKernelError",
    "FusedLowered",
    "Lowered",
    "ProgramLowered",
    "ProgramNest",
    "TpuKernel",
    "TpuProgram",
    "__version__",
    "tpu",
]
try:
    __version__ = _metadata.version("autokernel")
except _metadata.PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.4.0"
