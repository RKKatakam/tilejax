"""Matmul, `x @ y`, as plain JAX loops over tiles, with its tile sizes left to the autotuner.

Run with `uv run python examples/matmul.py`; without a TPU it runs in the Pallas TPU interpreter, with the
default tiles.
"""

import jax
import jax.numpy as jnp

import autokernel


@autokernel.tunable
class MatmulTiles:
    """(tm, tn) output tiles, each summing tk-deep products."""

    tm: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tn: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tk: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


@autokernel.kernel
def matmul(x, y, tiles=MatmulTiles()):
    M, K = x.shape
    N = y.shape[1]
    tm, tn, tk = min(tiles.tm, M), min(tiles.tn, N), min(tiles.tk, K)
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            acc = jnp.zeros((tm, tn), jnp.float32)
            for k in range(0, K, tk):
                acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
            out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
    return out


if __name__ == "__main__":
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (1024, 1024), jnp.bfloat16)
    out = matmul(x, y)  # tuned on a TPU, the default tiles elsewhere
    print(matmul.tuning(x, y))
    print(matmul.source(x, y))
    print("max |kernel - loop nest|:", matmul.verify(x, y))
    print("with 256^3 tiles:", matmul.verify(x, y, MatmulTiles(tm=256, tn=256, tk=256)))
