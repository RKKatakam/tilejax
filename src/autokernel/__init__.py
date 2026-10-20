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

import tomllib as _tomllib
from importlib import metadata as _metadata
from pathlib import Path as _Path

from .api import TpuKernel, tpu
from .errors import AutoKernelError
from .fusion import FusedLowered
from .lowering import Lowered
from .profiling import KernelProfile, Profile, enable_profiling, profile
from .program import ProgramLowered, ProgramNest, TpuProgram
from .tiles import tile

__all__ = [
    "AutoKernelError",
    "FusedLowered",
    "KernelProfile",
    "Lowered",
    "Profile",
    "ProgramLowered",
    "ProgramNest",
    "TpuKernel",
    "TpuProgram",
    "__version__",
    "enable_profiling",
    "profile",
    "tile",
    "tpu",
]
_PYPROJECT = _Path(__file__).resolve().parents[2] / "pyproject.toml"


def _source_version() -> str:
    """The version in the source tree's pyproject.toml (running from a checkout that isn't installed)."""
    try:
        return _tomllib.loads(_PYPROJECT.read_text())["project"]["version"]
    except (OSError, KeyError, ValueError):
        return "0+unknown"


try:
    __version__ = _metadata.version("autokernel")
except _metadata.PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = _source_version()
