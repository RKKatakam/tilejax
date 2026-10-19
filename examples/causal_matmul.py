"""Causal matmul, tril(x @ y), skipping every tile above the diagonal.

Run with `uv run python examples/causal_matmul.py`. Without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp

import autokernel

# out = tril(x @ y): every entry with col > row is zero.
#
# Output tiles fall into three groups relative to the diagonal:
#
#   . . . .     above: all zero -> `continue`, the tile keeps the zeros from jnp.zeros
#   \ . . .     on:    computed, then entries with col > row zeroed by the jnp.where
#   # \ . .     below: computed, the jnp.where keeps everything
#   # # \ .
#
# autokernel evaluates the `if` for every tile at trace time and the kernel loops over only the tiles
# that survive, so skipped tiles cost no math, no copies and no steps; their zeros are written by
# async DMAs that overlap the compute.
M, K, N = 1024, 1024, 1024
TILE_M, TILE_N, TILE_K = 256, 256, 256


@autokernel.tpu
def causal_matmul(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, TILE_M):
        for j in range(0, N, TILE_N):
            if j > i + TILE_M - 1:  # every entry of this tile is above the diagonal
                continue
            acc = jnp.zeros((TILE_M, TILE_N))
            for k in range(0, K, TILE_K):
                acc += x[i : i + TILE_M, k : k + TILE_K] @ y[k : k + TILE_K, j : j + TILE_N]
            rows = i + jax.lax.broadcasted_iota(jnp.int32, (TILE_M, TILE_N), 0)
            cols = j + jax.lax.broadcasted_iota(jnp.int32, (TILE_M, TILE_N), 1)
            out = out.at[i : i + TILE_M, j : j + TILE_N].set(jnp.where(cols <= rows, acc, 0.0))
    return out


if __name__ == "__main__":
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K))
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N))

    print(causal_matmul.source(x, y))
    lowered = causal_matmul.lower(x, y)
    print(f"computing {lowered.tiles} of {lowered.total_tiles} output tiles")

    ref = jnp.tril(jnp.dot(x, y, precision="highest"))
    print("max diff vs tril(jnp.dot):", jnp.max(jnp.abs(causal_matmul(x, y) - ref)))
    print("max diff vs Python loops:", causal_matmul.verify(x, y))
