"""Causal multi-head splash attention, written as plain Python tile loops.

softmax(q @ k.T / sqrt(D) + causal mask) @ v for every batch and head, computed flash-attention style:
for each query block, stream over key/value blocks keeping a running row max, row sum and output
accumulator `acc` (the online softmax), then normalise once at the end.

It's "splash" (sparse flash) attention because of the reduction mask: a key/value block entirely in the
future of a query block (`j > i + TQ - 1`) is skipped, so autokernel's generated kernel never fetches or
computes it. Only the diagonal blocks need the elementwise causal mask.

The per-row statistics are kept lane-wide: shape (TQ, 128) with each row's value repeated across the
128 lanes of a TPU vector register, widened with `jnp.tile` where needed. A (TQ, 1) column would need a
cross-lane broadcast on every step; on a v5e the lane-wide form makes this kernel about 1.7x faster.

Run with `uv run python examples/splash_attention.py`. Without a TPU it runs in the Pallas TPU interpreter.
"""

import math

import jax
import jax.numpy as jnp

import autokernel

B, H, S, D = 2, 4, 1024, 128  # batch, heads, sequence length, head dim
TQ, TK = 256, 256  # query / key-value block sizes
SCALE = 1.0 / math.sqrt(D)
LANES = 128  # TPU vector lanes
# Masked scores use a large finite value rather than -inf, so exp(row_max - m_new) never computes -inf + inf.
MASK_VALUE = -1e30


@autokernel.tpu
def causal_attention(q, k, v):
    dtype = q.dtype
    out = jnp.zeros((B, H, S, D), dtype)
    for b in range(0, B, 1):
        for h in range(0, H, 1):
            for i in range(0, S, TQ):
                row_max = jnp.full((TQ, LANES), MASK_VALUE)  # running row max, lane-wide
                row_sum = jnp.zeros((TQ, LANES))  # running sum of exp(s - row_max), lane-wide
                acc = jnp.zeros((TQ, D))  # running sum of exp(s - row_max) @ v
                for j in range(0, S, TK):
                    if j > i + TQ - 1:  # every key in this block is in the future of every query: skip it
                        continue
                    s = (q[b, h, i : i + TQ, :] @ k[b, h, j : j + TK, :].T) * SCALE
                    rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                    cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                    s = jnp.where(cols <= rows, s, MASK_VALUE)
                    m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                    p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                    alpha = jnp.exp(row_max - m_new)
                    row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                    acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + TK, :]
                    row_max = m_new
                out = out.at[b, h, i : i + TQ, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
    return out


def reference_attention(q, k, v):
    """Plain JAX causal attention in f32, for comparison."""
    s = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * SCALE
    causal = jnp.tril(jnp.ones((S, S), bool))
    p = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(jnp.float32))


if __name__ == "__main__":
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(key, (B, H, S, D), jnp.bfloat16) for key in keys)

    lowered = causal_attention.lower(q, k, v)
    print(lowered.source)
    total = B * H * (S // TQ) * (S // TK)
    print(f"{lowered.steps} of {total} (query block, key/value block) steps run; the rest are skipped")

    out = causal_attention(q, k, v)
    err = jnp.max(jnp.abs(out.astype(jnp.float32) - reference_attention(q, k, v)))
    print("max abs error vs plain JAX attention:", err)
    print("max abs error vs Python loops:", causal_attention.verify(q, k, v))
