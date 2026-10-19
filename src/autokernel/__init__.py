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
"""

from .api import TpuKernel, tpu
from .errors import AutoKernelError
from .lowering import Lowered

__all__ = ["AutoKernelError", "Lowered", "TpuKernel", "__version__", "tpu"]
__version__ = "0.2.0"
