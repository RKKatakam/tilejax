"""Dense tiled matmul: print the generated Pallas kernel, run it and check it.

Run with `uv run python examples/tiled_matmul.py`. Without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp

import autokernel

# TPU tiles: the last block dim must be a multiple of 128 (or the whole dim), the second-to-last a
# multiple of 8 for f32 (or the whole dim). autokernel rejects tiles that break this.
M, K, N = 256, 384, 256
TILE_M, TILE_N, TILE_K = 64, 128, 128

assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, "tiles must divide the matrix sizes"


@autokernel.tpu
def tiled_matmul(a, b):
    c = jnp.zeros((M, N))
    for i in range(0, M, TILE_M):
        for j in range(0, N, TILE_N):
            acc = jnp.zeros((TILE_M, TILE_N))
            for k in range(0, K, TILE_K):
                acc += a[i : i + TILE_M, k : k + TILE_K] @ b[k : k + TILE_K, j : j + TILE_N]
            c = c.at[i : i + TILE_M, j : j + TILE_N].set(acc)
    return c


if __name__ == "__main__":
    a = jax.random.normal(jax.random.PRNGKey(0), (M, K))
    b = jax.random.normal(jax.random.PRNGKey(1), (K, N))

    print(tiled_matmul.source(a, b))

    ours = tiled_matmul(a, b)  # the generated Pallas TPU kernel
    ref = jnp.dot(a, b, precision="highest")

    print("max diff vs jnp.dot:", jnp.max(jnp.abs(ours - ref)))
    print("match:", jnp.allclose(ours, ref, atol=1e-4))
    print("max diff vs Python loops:", tiled_matmul.verify(a, b))
