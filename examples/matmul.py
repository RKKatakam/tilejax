"""Matmul, `x @ y`, as tile loops.

Run with `uv run python examples/matmul.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp

import autokernel


def make_matmul(tm: int = 512, tn: int = 512, tk: int = 512) -> autokernel.Kernel:
    """`x @ y` for x of shape (M, K) and y of shape (K, N), in (tm, tn) output tiles summing (tk)-deep
    products. Accumulates in f32; the output has x's dtype."""

    @autokernel.kernel
    def matmul(x, y):
        M, K = x.shape
        N = y.shape[1]
        dtype = x.dtype
        out = jnp.zeros((M, N), dtype)
        for i in range(0, M, tm):
            for j in range(0, N, tn):
                acc = jnp.zeros((tm, tn), jnp.float32)
                for k in range(0, K, tk):
                    acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
                out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
        return out

    return matmul


if __name__ == "__main__":
    matmul = make_matmul(256, 256, 256)
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (1024, 1024), jnp.bfloat16)
    print(matmul.source(x, y))
    print("max |kernel - loop nest|:", matmul.verify(x, y))
