"""Causal matmul, `tril(x @ y)`, skipping every output tile above the diagonal.

Output tiles fall into three groups:

    . . . .     above the diagonal: skipped by the mask; the kernel writes them with the zeros of jnp.zeros
    \\ . . .     on it: computed, then the entries above the diagonal zeroed by jnp.where
    # \\ . .     below it: computed
    # # \\ .

autokernel evaluates the mask for every tile when it builds the kernel, so the kernel's step loop visits
only the tiles that run; the skipped ones cost one DMA each, overlapping the compute.

Run with `uv run python examples/causal_matmul.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp
from matmul import MatmulTiles

import autokernel


@autokernel.kernel
def causal_matmul(x, y, tiles=MatmulTiles()):
    M, K = x.shape
    N = y.shape[1]
    tm, tn, tk = min(tiles.tm, M), min(tiles.tn, N), min(tiles.tk, K)
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            if j > i + tm - 1:  # the whole tile is above the diagonal
                continue
            acc = jnp.zeros((tm, tn), jnp.float32)
            for k in range(0, K, tk):
                acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
            rows = i + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0)
            cols = j + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 1)
            out = out.at[i : i + tm, j : j + tn].set(jnp.where(cols <= rows, acc, 0.0).astype(dtype))
    return out


if __name__ == "__main__":
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (1024, 1024), jnp.bfloat16)
    tiles = MatmulTiles(tm=256, tn=256, tk=256)
    lowered = causal_matmul.lower(x, y, tiles)
    print(lowered.source)
    print(f"computes {lowered.tiles} of {lowered.total_tiles} output tiles")
    print("max |kernel - loop nest|:", causal_matmul.verify(x, y, tiles))
