"""RMSNorm fused with the projection after it: `rmsnorm(x) @ w`, in one pass over x.

    rmsnorm(x) = x / sqrt(mean(x**2, axis=-1) + eps) * gamma

The row scale commutes with the matmul, `rmsnorm(x) @ w = (1 / rms(x)) * ((x * gamma) @ w)`, so one
reduction over the model dimension accumulates both `(x * gamma) @ w` on the MXU and each row's sum of
squares, and the tile is scaled once at the end: x is read once per output tile, and the normalized x is
never written to HBM. The sum of squares is kept lane-wide, (tm, 128), as attention's row statistics are.

Run with `uv run python examples/rmsnorm_matmul.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp

import autokernel

EPS = 1e-6
LANES = 128  # TPU vector lanes


@autokernel.tunable
class NormMatmulTiles:
    """(tm, tn) output tiles, each summing tk-deep products (and tm rows' squares) over the model dim."""

    tm: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tn: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tk: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


@autokernel.kernel
def rmsnorm_matmul(x, gamma, w, tiles=NormMatmulTiles()):
    M, D = x.shape
    N = w.shape[1]
    tm, tn, tk = min(tiles.tm, M), min(tiles.tn, N), min(tiles.tk, D)
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            acc = jnp.zeros((tm, tn), jnp.float32)  # (x * gamma) @ w
            sum_sq = jnp.zeros((tm, LANES), jnp.float32)  # each row's sum of squares, lane-wide
            for k in range(0, D, tk):
                xk = x[i : i + tm, k : k + tk].astype(jnp.float32)
                sum_sq += jnp.sum(xk * xk, axis=1, keepdims=True)
                acc += (xk * gamma[0:1, k : k + tk].astype(jnp.float32)).astype(dtype) @ w[k : k + tk, j : j + tn]
            scale = jax.lax.rsqrt(sum_sq / D + EPS)
            out = out.at[i : i + tm, j : j + tn].set((acc * jnp.tile(scale, (1, tn // LANES))).astype(dtype))
    return out


def rmsnorm_then_matmul(x, gamma, w):
    """The same in plain JAX, unfused: what XLA compiles."""
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)
    return jnp.dot(normed.astype(x.dtype), w, preferred_element_type=jnp.float32).astype(x.dtype)


if __name__ == "__main__":
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (1024, 1024), jnp.bfloat16)
    gamma = (1 + 0.1 * jax.random.normal(keys[1], (1, 1024))).astype(jnp.bfloat16)
    w = (jax.random.normal(keys[2], (1024, 512)) / 32).astype(jnp.bfloat16)
    out = rmsnorm_matmul(x, gamma, w)  # tuned on a TPU, the default tiles elsewhere
    print(rmsnorm_matmul.tuning(x, gamma, w))
    print(rmsnorm_matmul.source(x, gamma, w))
    print("max |kernel - loop nest|:", rmsnorm_matmul.verify(x, gamma, w))
    diff = jnp.abs(out.astype(jnp.float32) - rmsnorm_then_matmul(x, gamma, w).astype(jnp.float32))
    print("max |kernel - plain JAX rmsnorm then matmul|:", float(diff.max()))
