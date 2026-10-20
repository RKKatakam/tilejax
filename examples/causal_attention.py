"""Causal multi-head attention, flash-attention style, skipping key blocks in the future of a query block.

    out[b, h] = softmax(q[b, h] @ k[b, h].T + causal mask) @ v[b, h]

q is pre-scaled by 1/sqrt(head_dim), as JAX's splash attention expects too. For each query block the
kernel streams the key/value blocks keeping a running row max, row sum and output accumulator
(the online softmax), and normalises once at the end. The reduction mask skips key blocks entirely in
the future of the query block (that's the "splash", sparse flash), so the kernel never fetches them;
only blocks on the diagonal need the elementwise mask.

The row max and sum are kept lane-wide, (block_q, 128) with a row's value repeated in all 128 lanes of a vector
register, and widened with `jnp.tile` where they meet a wider block: a (block_q, 1) column would need a
cross-lane broadcast at every step.

Run with `uv run python examples/causal_attention.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask

import autokernel

LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


def make_causal_attention(block_q: int = 512, block_kv: int = 512) -> autokernel.Kernel:
    """Causal attention over (batch, heads, seq, head_dim) arrays, with q pre-scaled."""

    @autokernel.kernel
    def causal_attention(q, k, v):
        B, H, S, D = q.shape
        dtype = q.dtype
        out = jnp.zeros((B, H, S, D), dtype)
        for b in range(B):
            for h in range(H):
                for i in range(0, S, block_q):
                    row_max = jnp.full((block_q, LANES), MASK_VALUE, jnp.float32)  # running row max
                    row_sum = jnp.zeros((block_q, LANES), jnp.float32)  # running row sum of exp(s - row_max)
                    acc = jnp.zeros((block_q, D), jnp.float32)  # running sum of exp(s - row_max) @ v
                    for j in range(0, S, block_kv):
                        if j > i + block_q - 1:  # every key is in the future of every query
                            continue
                        s = q[b, h, i : i + block_q, :] @ k[b, h, j : j + block_kv, :].T
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (block_q, block_kv), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (block_q, block_kv), 1)
                        s = jnp.where(cols <= rows, s, MASK_VALUE)
                        max_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(max_new, (1, block_kv // LANES)))
                        alpha = jnp.exp(row_max - max_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + block_kv, :]
                        row_max = max_new
                    out = out.at[b, h, i : i + block_q, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
        return out

    return causal_attention


def make_splash_attention(heads: int, seq: int, block_q: int = 512, block_kv: int = 512, interpret: bool = False):
    """JAX's splash attention with a causal mask over (batch, heads, seq, head_dim), at the same blocks."""
    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((seq, seq)) for _ in range(heads)])
    blocks = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    kernel = splash.make_splash_mha_single_device(mask, block_sizes=blocks, interpret=interpret)
    return jax.jit(jax.vmap(kernel))


if __name__ == "__main__":
    B, H, S, D = 1, 2, 1024, 128
    keys = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(key, (B, H, S, D), jnp.bfloat16) for key in keys)
    q = (q * D**-0.5).astype(q.dtype)

    attention = make_causal_attention(256, 256)
    lowered = attention.lower(q, k, v)
    print(lowered.source)
    print(f"runs {lowered.steps} (query block, key block) steps; the future blocks are never visited")
    on_tpu = jax.default_backend() == "tpu"
    reference = make_splash_attention(H, S, 256, 256, interpret=not on_tpu)(q, k, v)
    print("max |autokernel - splash|:", float(jnp.max(jnp.abs(attention(q, k, v).astype(float) - reference))))
    print("max |kernel - loop nest|:", attention.verify(q, k, v))
