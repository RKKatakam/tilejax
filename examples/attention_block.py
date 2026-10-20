"""A whole attention block in one kernel: causal multi-head attention, the post-attention RMSNorm, and the
feed-forward projections after it.

    a = concat_h(softmax(q_h @ k_h.T + causal mask) @ v_h)      (B, S, D), D = heads * head_dim
    h = rmsnorm(a) = a / sqrt(mean(a**2, axis=-1) + eps) * gamma
    out = gelu(h @ w_up) @ w_down                               (B, S, D)

One loop nest: the parallel loops walk (batch, query block). Each tile then runs two reduction loops, one
after the other: the attention, over the key blocks (skipping those in the future of the query block,
splash attention's causal sparsity), with an online softmax for every head at once (a list comprehension
over heads, so each head is a plain 2-D matmul); then, after the heads are put side by side and normalized
together, the feed-forward layer, over blocks of hidden units, streaming the weights' blocks (or keeping
them in VMEM, when one block covers them all: the tuner decides). The attention output, the normalized
activations and the hidden activations never go to HBM.

q is pre-scaled by 1/sqrt(head_dim), as JAX's splash attention expects too.

Run with `uv run python examples/attention_block.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp
from causal_attention import splash_attention

import autokernel

EPS = 1e-6
LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@autokernel.tunable
class BlockTiles:
    """Queries per output tile, keys per attention step, and hidden units per feed-forward step (a block
    of all of them keeps the feed-forward weights in VMEM; smaller ones stream them)."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, 16384, default=1024)


def widen(stat, n):
    """A lane-wide (..., 128) row statistic as (..., n): repeated across lanes, or its first n lanes."""
    return jnp.tile(stat, (1,) * (stat.ndim - 1) + (n // LANES,)) if n >= LANES else stat[..., :n]


@autokernel.kernel
def attention_block(q, k, v, gamma, w_up, w_down, tiles=BlockTiles()):
    B, H, S, Dh = q.shape
    D, F = H * Dh, w_up.shape[1]
    bq, bk, bf = min(tiles.block_q, S), min(tiles.block_kv, S), min(tiles.block_f, F)
    dtype = q.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for i in range(0, S, bq):
            row_max = jnp.full((H, bq, LANES), MASK_VALUE, jnp.float32)  # per head, lane-wide
            row_sum = jnp.zeros((H, bq, LANES), jnp.float32)  # of exp(s - row_max)
            acc = jnp.zeros((H, bq, Dh), jnp.float32)  # sum of exp(s - row_max) @ v
            for j in range(0, S, bk):  # phase 1: causal attention, every head at once
                if j > i + bq - 1:  # every key is in the future of every query
                    continue
                qt, kt, vt = q[b, :, i : i + bq, :], k[b, :, j : j + bk, :], v[b, :, j : j + bk, :]
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
                cols = j + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
                s = jnp.stack([jnp.where(cols <= rows, qt[h] @ kt[h].T, MASK_VALUE) for h in range(H)])
                max_new = jnp.maximum(row_max, jnp.max(s, axis=2, keepdims=True))
                p = jnp.exp(s - widen(max_new, bk))
                alpha = jnp.exp(row_max - max_new)
                row_sum = alpha * row_sum + jnp.sum(p, axis=2, keepdims=True)
                pv = jnp.stack([p[h].astype(dtype) @ vt[h] for h in range(H)])
                acc = widen(alpha, Dh) * acc + pv
                row_max = max_new
            heads = acc / widen(row_sum, Dh)
            a = jnp.concatenate([heads[h] for h in range(H)], axis=1)  # (bq, D): the heads side by side
            normed = (a * jax.lax.rsqrt(jnp.mean(a * a, axis=1, keepdims=True) + EPS) * gamma[0:1, :]).astype(dtype)
            y = jnp.zeros((bq, D), jnp.float32)
            for f in range(0, F, bf):  # phase 2: the feed-forward layer, bf hidden units at a time
                hidden = jax.nn.gelu(normed @ w_up[:, f : f + bf])
                y += hidden.astype(dtype) @ w_down[f : f + bf, :]
            out = out.at[b, i : i + bq, :].set(y.astype(dtype))
    return out


def norm_and_ffn(a, gamma, w_up, w_down):
    """RMSNorm and the feed-forward layer in plain JAX, on (B, S, D) attention outputs."""
    af = a.astype(jnp.float32)
    normed = af * jax.lax.rsqrt(jnp.mean(af * af, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)
    hidden = jax.nn.gelu(jnp.dot(normed.astype(a.dtype), w_up, preferred_element_type=jnp.float32))
    return jnp.dot(hidden.astype(a.dtype), w_down, preferred_element_type=jnp.float32).astype(a.dtype)


def splash_then_norm_and_ffn(heads: int, seq: int, block_q: int | None = None, block_kv: int | None = None):
    """The unfused block: JAX's splash attention (its default block sizes unless given), the heads put side by
    side, then RMSNorm and the feed-forward layer, compiled by XLA."""
    if block_q is None:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

        mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq, seq)) for _ in range(heads)])
        attend = jax.vmap(splash.make_splash_mha_single_device(mask))
    else:
        attend = splash_attention(heads, seq, block_q, block_kv)

    def block(q, k, v, gamma, w_up, w_down):
        a = attend(q, k, v)  # (B, H, S, Dh)
        a = a.transpose(0, 2, 1, 3).reshape(q.shape[0], seq, -1)
        return norm_and_ffn(a, gamma, w_up, w_down)

    return jax.jit(block)


def make_inputs(B, H, S, Dh, F, dtype=jnp.bfloat16, seed=0):
    keys = jax.random.split(jax.random.key(seed), 6)
    q, k, v = (jax.random.normal(key, (B, H, S, Dh), dtype) for key in keys[:3])
    q = (q * Dh**-0.5).astype(dtype)
    D = H * Dh
    gamma = (1 + 0.1 * jax.random.normal(keys[3], (1, D))).astype(dtype)
    w_up = (jax.random.normal(keys[4], (D, F)) / D**0.5).astype(dtype)
    w_down = (jax.random.normal(keys[5], (F, D)) / F**0.5).astype(dtype)
    return q, k, v, gamma, w_up, w_down


if __name__ == "__main__":
    B, H, S, Dh, F = 1, 2, 512, 128, 1024
    inputs = make_inputs(B, H, S, Dh, F)
    tiles = BlockTiles(block_q=128, block_kv=128)
    lowered = attention_block.lower(*inputs, tiles)
    print(lowered.source)
    print(f"{lowered.steps} steps over {lowered.tiles} query blocks; VMEM {lowered.vmem_bytes / 2**20:.1f} MiB")
    out = attention_block(*inputs, tiles)
    on_tpu = jax.default_backend() == "tpu"
    if on_tpu:
        unfused = splash_then_norm_and_ffn(H, S)(*inputs)
        print("max |fused - splash, norm, ffn|:", float(jnp.max(jnp.abs(out.astype(float) - unfused.astype(float)))))
    print("max |kernel - loop nest|:", attention_block.verify(*inputs, tiles))
