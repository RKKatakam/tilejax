"""A SwiGLU feed-forward layer with its RMSNorm and residual, fused: the second half of a Qwen3 / Llama layer.

    n = rmsnorm(x) = x / sqrt(mean(x**2, axis=-1) + eps) * gamma
    out = x + (silu(n @ w_gate) * (n @ w_up)) @ w_down              x: (tokens, d_model)

One tile per block of rows: the code before the reduction loop normalizes the rows once (kept in VMEM, bf16,
for every step); the reduction loop walks the hidden units, `block_f` per step, streaming that block of
`w_gate` and `w_up` (both matmuls in the same step, on the same normalized rows) and of `w_down`, and
accumulating the down projection in f32. The residual is added once, at the end, from the row tile still
in VMEM (re-reading it costs less than carrying an f32 copy). Neither the normalized
rows nor the (rows, ffn) hidden activations ever go to HBM: XLA writes and reads both.

The row tile is every weight's reuse: each tile streams all three weight matrices once, so big row tiles
(1024 rows and up, the tuner's choice) keep it compute-bound; a `block_f` covering all hidden units keeps the
weights in VMEM instead, which only pays for small layers.

`swiglu` and `rms_rows` are shared with the attention kernels that fuse this layer (gqa_attention.py).

Run with `uv run python examples/swiglu.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import autokernel

EPS = 1e-6  # RMSNorm's epsilon (Qwen3: 1e-6)


@autokernel.tunable
class FfnTiles:
    """Rows per output tile, hidden units per step."""

    rows: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=512)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, 16384, default=512)


def rms_rows(x, gamma):
    """RMSNorm over the last dim of f32 rows (..., d) with a (1, d) weight, in f32."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def swiglu(gate, up):
    """silu(gate) * up, in f32: the hidden activations of a block of hidden units."""
    return jax.nn.silu(gate) * up


@autokernel.kernel
def swiglu_ffn(x, gamma, w_gate, w_up, w_down, tiles=FfnTiles()):
    """x + SwiGLU(rmsnorm(x)) over (tokens, d_model) rows."""
    T, D = x.shape
    F = w_up.shape[1]
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        normed = rms_rows(x[i : i + tm, :].astype(jnp.float32), gamma[0:1, :]).astype(dtype)
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):  # bf hidden units per step: gate and up together, then their down projection
            hidden = swiglu(normed @ w_gate[:, f : f + bf], normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        out = out.at[i : i + tm, :].set((x[i : i + tm, :].astype(jnp.float32) + y).astype(dtype))
    return out


def swiglu_ffn_xla(x, gamma, w_gate, w_up, w_down):
    """The same in plain JAX, as XLA compiles it (any leading dims)."""
    normed = rms_rows(x.astype(jnp.float32), gamma).astype(x.dtype)
    gate = jnp.dot(normed, w_gate, preferred_element_type=jnp.float32)
    up = jnp.dot(normed, w_up, preferred_element_type=jnp.float32)
    down = jnp.dot(swiglu(gate, up).astype(x.dtype), w_down, preferred_element_type=jnp.float32)
    return (x.astype(jnp.float32) + down).astype(x.dtype)


def make_inputs(tokens: int, d: int, f: int, dtype=jnp.bfloat16, seed: int = 0) -> tuple[jax.Array, ...]:
    keys = jax.random.split(jax.random.key(seed), 5)
    x = jax.random.normal(keys[0], (tokens, d), dtype)
    gamma = (1 + 0.1 * jax.random.normal(keys[1], (1, d))).astype(dtype)
    w_gate = (jax.random.normal(keys[2], (d, f)) / d**0.5).astype(dtype)
    w_up = (jax.random.normal(keys[3], (d, f)) / d**0.5).astype(dtype)
    w_down = (jax.random.normal(keys[4], (f, d)) / f**0.5).astype(dtype)
    return x, gamma, w_gate, w_up, w_down


if __name__ == "__main__":
    inputs = make_inputs(512, 256, 1024)
    tiles = FfnTiles(rows=256, block_f=256)
    print(swiglu_ffn.source(*inputs, tiles))
    out = swiglu_ffn(*inputs, tiles)
    diff = jnp.abs(out.astype(jnp.float32) - swiglu_ffn_xla(*inputs).astype(jnp.float32))
    print("max |kernel - plain JAX|:", float(diff.max()))
    print("max |kernel - loop nest|:", swiglu_ffn.verify(*inputs, tiles))
