"""Causal multi-head attention, flash-attention style, skipping key blocks in the future of a query block.

    out[b, h] = softmax(q[b, h] @ k[b, h].T + causal mask) @ v[b, h]

q is pre-scaled by 1/sqrt(head_dim), as JAX's splash attention expects too. For each query block the
kernel streams the key/value blocks keeping a running row max, row sum and output accumulator (the online
softmax), and normalises once at the end. The reduction mask skips key blocks entirely in the future of
the query block (that's the "splash", sparse flash), so the kernel never fetches them; only blocks on the
diagonal need the elementwise mask.

The row max and sum are kept lane-wide, (block_q, 128) with a row's value repeated in all 128 lanes of a
vector register, and widened with `jnp.tile` where they meet a wider block: a (block_q, 1) column would need
a cross-lane broadcast at every step.

Run with `uv run python examples/causal_attention.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask

import autokernel

LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@autokernel.tunable
class AttentionBlocks:
    """Queries and keys per block."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=512)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=512)


@autokernel.kernel
def causal_attention(q, k, v, blocks=AttentionBlocks()):
    B, H, S, D = q.shape
    bq, bk = min(blocks.block_q, S), min(blocks.block_kv, S)
    dtype = q.dtype
    out = jnp.zeros((B, H, S, D), dtype)
    for b in range(B):
        for h in range(H):
            for i in range(0, S, bq):
                row_max = jnp.full((bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((bq, LANES), jnp.float32)  # of exp(s - row_max)
                acc = jnp.zeros((bq, D), jnp.float32)  # sum of exp(s - row_max) @ v
                for j in range(0, S, bk):
                    if j > i + bq - 1:  # every key is in the future of every query
                        continue
                    s = q[b, h, i : i + bq, :] @ k[b, h, j : j + bk, :].T
                    rows = i + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
                    cols = j + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
                    s = jnp.where(cols <= rows, s, MASK_VALUE)
                    max_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                    p = jnp.exp(s - jnp.tile(max_new, (1, bk // LANES)))
                    alpha = jnp.exp(row_max - max_new)
                    row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                    acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + bk, :]
                    row_max = max_new
                out = out.at[b, h, i : i + bq, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
    return out


def splash_attention(heads: int, seq: int, block_q: int = 512, block_kv: int = 512, interpret: bool = False):
    """JAX's splash attention with a causal mask over (batch, heads, seq, head_dim), at the given blocks."""
    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((seq, seq)) for _ in range(heads)])
    sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    kernel = splash.make_splash_mha_single_device(mask, block_sizes=sizes, interpret=interpret)
    return jax.jit(jax.vmap(kernel))


if __name__ == "__main__":
    B, H, S, D = 1, 2, 1024, 128
    keys = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(key, (B, H, S, D), jnp.bfloat16) for key in keys)
    q = (q * D**-0.5).astype(q.dtype)

    blocks = AttentionBlocks(block_q=256, block_kv=256)
    lowered = causal_attention.lower(q, k, v, blocks)
    print(lowered.source)
    print(f"runs {lowered.steps} (query block, key block) steps; the future blocks are never visited")
    on_tpu = jax.default_backend() == "tpu"
    reference = splash_attention(H, S, 256, 256, interpret=not on_tpu)(q, k, v)
    ours = causal_attention(q, k, v, blocks)
    print("max |autokernel - splash|:", float(jnp.max(jnp.abs(ours.astype(float) - reference.astype(float)))))
    print("max |kernel - loop nest|:", causal_attention.verify(q, k, v, blocks))
