"""A whole decode step of a dense GQA transformer (Qwen3-like) as ONE autokernel kernel: cross-layer prefetch.

    x = embed[tokens]                                                   (B, d_model), one new token per sequence
    per layer:  n = rmsnorm(x);  q, k, v = n @ W_qkv
                q_h = rope(rmsnorm_h(q_h) * q_norm) / sqrt(Dh),  k_g = rope(rmsnorm_h(k_g) * k_norm)   (QK-norm)
                a_h = softmax(q_h . [K_cache[b, :len_b] ; k_b]_g) @ [V_cache[b, :len_b] ; v_b]_g      (GQA)
                x = x + a @ W_o;   x = x + W_down(silu(m @ W_gate) * (m @ W_up)),  m = rmsnorm(x)
    logits = rmsnorm(x) @ W_vocab

Each new token sits at position lengths[b] of its sequence (RoPE turns q and k by it) and attends to the
sequence's lengths[b] cached keys and values and to its own. A decode step is bound by HBM bandwidth: it
reads every layer's weights once (and each sequence's valid cache) for a few FLOPs per byte, so it is fast
when the DMA engines never idle. examples/decode_model.py runs a step as three kernels per layer around a
cache append in XLA; each kernel starts with its pipeline empty and ends by writing its output to HBM.
`decode_megakernel` runs the whole step, every layer and the LM head, as one kernel:

    one tile: all B rows                       (the parallel loop has one iteration)
      h = x                                    the residual stream, f32, in VMEM from the first layer to the last
      REPEAT LOOP over the layers  (weights stacked by layer, (L, ...)):
        phase 1, d: the QKV projection, a block of W_qkv's rows per step  (qkv += n_blocks[d // bd] @ W_qkv[l, d:d+bd])
           then QK-norm, RoPE, the new token's k and v kept for the output, per-sequence attention state
        phase 2, j: attention over the KV cache, a block of one sequence's positions per step, under a RUNTIME
           MASK: only the blocks before each sequence's length run
        phase 3, a: the output projection, a block of W_o's rows per step;  residual, RMSNorm
        phase 4, f: the SwiGLU feed-forward layer, a block of hidden units per step;  residual
      final RMSNorm
      phase 5, c: the LM head, a block of vocab columns per step (logits as (vocab / bv, B, bv) blocks in VMEM)

Windows a repeat loop moves (`w_qkv[layer, d : d + bd, :]`) stream with the phase that reads them, so while
layer l computes its last feed-forward block the pipeline already fetches layer l + 1's first W_qkv blocks
(and the LM head's first vocab block during the last layer): CROSS-LAYER PREFETCH. Nothing returns to HBM
between layers; the kernel's outputs are the logits and every layer's new k and v, (L, B, 2 * Hkv * Dh),
which `megakernel_step` writes into the caches (an XLA scatter, in place on donated caches).

The design, and why:

- Row blocks of W_qkv with the normalized rows held as LEADING-AXIS blocks (`n_blocks` (D / bd, B, bd), a
  carried value built by `lane_blocks` in the repeat loop's code before phase 1, read as `n_blocks[d // bd]`,
  which loads just that block from VMEM) rather than column blocks writing `qkv.at[c // bc].set(...)`: a
  row block is one contiguous DMA (a column block of W_qkv is a strided one, 2 * bc bytes per row), and the
  QK-norm / RoPE epilogue stays straight-line code on the whole (B, (H + 2 Hkv) * Dh) accumulator (which of
  q, k or v a column block holds would be a runtime choice). The accumulator is small: B rows.
- Attention, one SEQUENCE BLOCK per step. The caches are (L, B, S_max, Hkv * Dh), read flattened as
  (L, B * S_max, Hkv * Dh) (a free reshape outside) by a phase over j in steps of `block_kv` positions:
  block j is positions j % S_max .. of sequence j // S_max, and `if j % S_max >= lengths[j // S_max]:
  continue` is a runtime mask, so the kernel visits exactly the blocks holding cached keys (an empty
  sequence takes no step; if all are empty the phase takes one empty step). A step is one contiguous
  (block_kv, Hkv * Dh) DMA of K and of V, as in decode.py's `gqa_decode`: the per-sequence math is that
  kernel's, all heads of the sequence at once with its queries spread BLOCK-DIAGONALLY over the kv heads
  ((H, Hkv * Dh): `q_bd @ k.T` is every head's scores against its own kv head). Rows of different
  sequences never meet: no (B * H, B * bk) score matrices, no batched matmuls, no masking of other
  sequences' rows. (A step over all B sequences at once, (B, bk, Hkv * Dh) windows up to the longest
  length, would read every sequence up to the longest one and need sequence-batched matmuls.)
- Per-sequence state in VMEM, indexed by the sequence: the block-diagonal queries `q_bd` (B, H, Hkv * Dh),
  the online softmax's lane-wide max and sum (B, H, 128) and accumulator (B, H, Hkv * Dh), each read and
  written at `[j // S_max]` (a dynamic index on the untiled leading dim of a VMEM buffer: a plain load and
  store). The state starts as the NEW TOKEN's own term (score q . k_new, weight 1, value v_new), computed
  after phase 1, so the cache blocks fold into it in any number of steps (none for an empty cache).
- Moving between the row layout (sequences on sublanes: what the projections produce and consume) and the
  per-sequence layout (heads on sublanes) uses only iota masks, selects, sublane sums, broadcasts and
  aligned slices and concatenations (`seq_row`, `place_rows`, `block_diagonal`, `gather_heads`): no
  unaligned sublane slicing, no lane-dim reshapes, no transposes, which Mosaic does badly or not at all.
  They cost O(B * H * Hkv * Dh) vector work per layer, small next to the layer's weights.
- The output projection and the FFN are decode_model.py's `decode_mlp` phases; the residual stream stays
  f32 across layers (the per-layer kernels round it to bf16 between kernels).
- The LM head writes `logit_blocks.at[c // bv].set(...)` (a partial update: only that block's VMEM) and
  the tile's last code puts the blocks side by side (aligned lane concatenation) for the (B, vocab) store.

VMEM: every streamed operand has its own ring of slots. At d_model 2048, FFN 6144, vocab 32k and the
defaults (256 rows of W_qkv, 256 cache positions, 512 rows of W_o, 512 hidden units, 1024 vocab columns
per step; 3 input slots) the weight slots take 45 MiB; with 16 layers the kernel's buffers total 52 MiB at
B 1 and 80 MiB at B 64, where the per-sequence state (6 MiB), the logits ((B, vocab) carried and twice as
output slots: 12 MiB) and every layer's new k and v (4 MiB, and 8 as output slots) add up.

Known costs the compiler could remove: windows moved only by the repeat loop (the norm weights) are copied
again at every step of their phase, and a one-tile kernel still gets two output slots per output.

`megakernel_step` wraps the kernel (embedding, RoPE tables, cache appends); `per_layer_step` is the same
step with decode_model.py's kernels (fused RMSNorm + QKV `rmsnorm_matmul`, `gqa_decode`, `decode_mlp`, per
layer; QK-norm, RoPE and the append in XLA between them) and `xla_step` the same in plain JAX (XLA, dense
masked attention over the whole cache), both on per-layer weights and caches (`layer_weights`,
`layer_caches`).

Run with `uv run python examples/decode_megakernel.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from decode import LANES, gqa_decode, online_softmax, widen, xla_gqa_decode
from decode_model import decode_mlp, rope
from kv_cache import append
from layer_stack import rmsnorm
from qwen3 import rope_tables, rotate_half
from rmsnorm_matmul import rmsnorm_matmul

import autokernel

EPS = 1e-6  # RMSNorm and QK-norm epsilon (as layer_stack's and qwen3's)
HEAD_DIM = 128
ROPE_THETA = 1e6  # Qwen3's


@dataclass(frozen=True)
class Config:
    """A dense Qwen3-like model: GQA (`kv_heads` divides `heads`), QK-norm, SwiGLU, head_dim 128."""

    layers: int
    d_model: int
    heads: int
    kv_heads: int
    ffn: int
    vocab: int


@autokernel.tunable
class MegaTiles:
    """Per step: rows of W_qkv, cache positions of a sequence, rows of W_o, hidden units, vocab columns."""

    block_d: int = autokernel.choice(128, 256, 512, 1024, 2048, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, 2048, default=256)
    block_a: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    block_f: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    block_v: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)


# --------------------------------------------------------------------------- helpers (plain JAX, in the kernel)


def lane_blocks(x, size):
    """(rows, n) as (n / size, rows, size): column blocks on a leading axis, so a phase can read block c of a
    carried value as `blocks[c]` (one VMEM load) instead of slicing lanes at a runtime offset."""
    return jnp.stack([x[:, c : c + size] for c in range(0, x.shape[1], size)])


def unblock(blocks):
    """(n / size, rows, size) blocks side by side: (rows, n)."""
    return jnp.concatenate([blocks[c] for c in range(blocks.shape[0])], axis=1)


def seq_row(x, b):
    """Row b of x (rows, n) as (1, n) f32: a masked sum over the row's aligned group of 8 (no unaligned
    sublane slice)."""
    start = b // 8 * 8
    group = x[start : min(start + 8, x.shape[0])].astype(jnp.float32)
    rows = jax.lax.broadcasted_iota(jnp.int32, group.shape, 0)
    return jnp.sum(jnp.where(rows == b - start, group, 0.0), axis=0, keepdims=True)


def place_rows(rows):
    """One (1, n) row per sequence as (len(rows), n): built 8 rows at a time from masked broadcasts and
    stacked at aligned offsets."""
    n, groups = rows[0].shape[1], []
    for start in range(0, len(rows), 8):
        size = min(8, len(rows) - start)
        ids = jax.lax.broadcasted_iota(jnp.int32, (size, n), 0)
        group = jnp.zeros((size, n), jnp.float32)
        for r in range(size):
            group = jnp.where(ids == r, rows[start + r], group)
        groups.append(group)
    return jnp.concatenate(groups, axis=0) if len(groups) > 1 else groups[0]


def block_diagonal(q_row, heads, kv_heads):
    """A sequence's queries, a (1, H * Dh) row, as (H, Hkv * Dh): row h holds head h's query in its kv head
    h // (H / Hkv)'s column block and zeros elsewhere, so `q_bd @ k.T` scores every head against its own
    kv head's keys in one matmul over a whole cache block (decode.py's layout)."""
    group = heads // kv_heads
    ids = jax.lax.broadcasted_iota(jnp.int32, (heads, HEAD_DIM), 0)
    blocks = []
    for g in range(kv_heads):
        block = jnp.zeros((heads, HEAD_DIM), jnp.float32)
        for h in range(g * group, (g + 1) * group):
            block = jnp.where(ids == h, q_row[:, h * HEAD_DIM : (h + 1) * HEAD_DIM], block)
        blocks.append(block)
    return jnp.concatenate(blocks, axis=1) if kv_heads > 1 else blocks[0]


def gather_heads(o, heads, kv_heads):
    """The inverse of `block_diagonal` for outputs: (H, Hkv * Dh), row h's own kv head block being head h's
    output, as a (1, H * Dh) row (each head's sublane picked by a masked sum)."""
    group = heads // kv_heads
    ids = jax.lax.broadcasted_iota(jnp.int32, (heads, HEAD_DIM), 0)
    pieces = []
    for h in range(heads):
        own = o[:, h // group * HEAD_DIM : (h // group + 1) * HEAD_DIM]
        pieces.append(jnp.sum(jnp.where(ids == h, own, 0.0), axis=0, keepdims=True))
    return jnp.concatenate(pieces, axis=1)


def qk_norm_rope(t, weight, cos, sin):
    """QK-norm and RoPE on projected heads t (rows, heads * Dh), f32: each head's RMSNorm over its Dh lanes,
    times the (1, Dh) weight, rotated by the rows' (rows, Dh) cos and sin (rotate-half)."""
    out = []
    for h in range(t.shape[1] // HEAD_DIM):
        x = t[:, h * HEAD_DIM : (h + 1) * HEAD_DIM]
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS) * weight.astype(jnp.float32)
        out.append(x * cos + rotate_half(x, HEAD_DIM // 2) * sin)
    return jnp.concatenate(out, axis=1) if len(out) > 1 else out[0]


def attention_start(qkv, heads, kv_heads):
    """Per-sequence attention state holding just the new token, from each row's [q | k | v] (B, (H + 2 Hkv)
    * Dh): the block-diagonal queries (B, H, Hkv * Dh) (in qkv's dtype), and the online softmax's state:
    lane-wide max (B, H, 128) = the score q_h . k_new, sum (B, H, 128) = 1, accumulator (B, H, Hkv * Dh) =
    v_new (a row's own kv head block is what counts)."""
    hd, kvd = heads * HEAD_DIM, kv_heads * HEAD_DIM
    rows, queries, scores, values = qkv.astype(jnp.float32), [], [], []
    for b in range(qkv.shape[0]):
        row = seq_row(rows, b)
        q_bd = block_diagonal(row[:, :hd], heads, kv_heads)
        queries.append(q_bd.astype(qkv.dtype))
        scores.append(jnp.broadcast_to(jnp.sum(q_bd * row[:, hd : hd + kvd], axis=1, keepdims=True), (heads, LANES)))
        values.append(jnp.broadcast_to(row[:, hd + kvd :], (heads, kvd)))
    row_sum = jnp.ones((qkv.shape[0], heads, LANES), jnp.float32)
    return jnp.stack(queries), jnp.stack(scores), row_sum, jnp.stack(values)


def attention_rows(acc, row_sum, heads, kv_heads):
    """The normalized attention output of every sequence, (B, H * Dh) f32 rows, from the per-sequence
    accumulators (B, H, Hkv * Dh) and lane-wide sums (B, H, 128)."""
    kvd = kv_heads * HEAD_DIM
    rows = [gather_heads(acc[b] * widen(1.0 / row_sum[b], kvd), heads, kv_heads) for b in range(acc.shape[0])]
    return place_rows(rows)


# --------------------------------------------------------------------------- the kernel


@autokernel.kernel
def decode_megakernel(
    x,
    cos,
    sin,
    lengths,
    k_cache,
    v_cache,
    attn_norm,
    w_qkv,
    q_norm,
    k_norm,
    wo,
    ffn_norm,
    w_gate,
    w_up,
    w_down,
    final_norm,
    w_vocab,
    tiles=MegaTiles(),
):
    """A whole decode step: x (B, D) the new tokens' embeddings, cos and sin (B, Dh) RoPE tables at their
    positions, lengths (B,) the cached positions per sequence, caches (L, B * S_max, Hkv * Dh) (flattened),
    weights stacked by layer (norms (L, 1, n)); q's 1/sqrt(Dh) is applied here. Returns the logits (B,
    vocab) and every layer's new [k | v] rows (L, B, 2 * Hkv * Dh)."""
    B, D = x.shape
    L, BS, KVD = k_cache.shape
    S = BS // B
    HD, F, V = wo.shape[1], w_up.shape[2], w_vocab.shape[1]
    H, Hkv = HD // HEAD_DIM, KVD // HEAD_DIM
    bd, bk, ba = min(tiles.block_d, D), min(tiles.block_kv, S), min(tiles.block_a, HD)
    bf, bv = min(tiles.block_f, F), min(tiles.block_v, V)
    scale = HEAD_DIM**-0.5
    dtype = x.dtype
    logits = jnp.zeros((B, V), dtype)
    new_kv = jnp.zeros((L, B, 2 * KVD), dtype)
    for i in range(0, B, B):
        h = x[i : i + B, :].astype(jnp.float32)  # the residual stream, f32, in VMEM throughout
        kv_rows = jnp.zeros((L, B, 2 * KVD), dtype)
        logit_blocks = jnp.zeros((V // bv, B, bv), dtype)
        for layer in range(L):  # a repeat loop: each layer's weights stream, the next layer's prefetched
            n_blocks = lane_blocks(rmsnorm(h, attn_norm[layer, :, :]).astype(dtype), bd)
            qkv = jnp.zeros((B, HD + 2 * KVD), jnp.float32)
            for d in range(0, D, bd):  # phase 1: QKV projection, bd rows of W_qkv per step
                qkv += n_blocks[d // bd] @ w_qkv[layer, d : d + bd, :]
            cos_t, sin_t = cos[i : i + B, :], sin[i : i + B, :]
            q = qk_norm_rope(qkv[:, :HD], q_norm[layer, :, :], cos_t, sin_t) * scale
            k = qk_norm_rope(qkv[:, HD : HD + KVD], k_norm[layer, :, :], cos_t, sin_t)
            new = jnp.concatenate([q, k, qkv[:, HD + KVD :]], axis=1).astype(dtype)
            kv_rows = kv_rows.at[layer].set(new[:, HD:])
            q_bd, row_max, row_sum, acc = attention_start(new, H, Hkv)
            for j in range(0, B * S, bk):  # phase 2: attention, bk cached positions of sequence j // S per step
                if j % S >= lengths[j // S]:  # past the sequence's cache: decided on the device
                    continue
                s = q_bd[j // S] @ k_cache[layer, j : j + bk, :].T
                ok = j % S + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1) < lengths[j // S]
                vt = v_cache[layer, j : j + bk, :]
                m_j, l_j, acc_j = online_softmax(s, vt, ok, row_max[j // S], row_sum[j // S], acc[j // S])
                row_max = row_max.at[j // S].set(m_j)
                row_sum = row_sum.at[j // S].set(l_j)
                acc = acc.at[j // S].set(acc_j)
            attn_blocks = lane_blocks(attention_rows(acc, row_sum, H, Hkv).astype(dtype), ba)
            proj = jnp.zeros((B, D), jnp.float32)
            for a in range(0, HD, ba):  # phase 3: output projection, ba rows of W_o per step
                proj += attn_blocks[a // ba] @ wo[layer, a : a + ba, :]
            h = h + proj
            normed = rmsnorm(h, ffn_norm[layer, :, :]).astype(dtype)
            y = jnp.zeros((B, D), jnp.float32)
            for f in range(0, F, bf):  # phase 4: SwiGLU, bf hidden units per step
                hidden = jax.nn.silu(normed @ w_gate[layer, :, f : f + bf]) * (normed @ w_up[layer, :, f : f + bf])
                y += hidden.astype(dtype) @ w_down[layer, f : f + bf, :]
            h = h + y
        final = rmsnorm(h, final_norm[0:1, :]).astype(dtype)
        for c in range(0, V, bv):  # phase 5: LM head, bv vocab columns per step
            logit_blocks = logit_blocks.at[c // bv].set((final @ w_vocab[:, c : c + bv]).astype(dtype))
        logits = logits.at[i : i + B, :].set(unblock(logit_blocks))
        new_kv = new_kv.at[:, i : i + B, :].set(kv_rows)
    return logits, new_kv


# --------------------------------------------------------------------------- the decode step around it


def kernel_inputs(tokens, k_cache, v_cache, lengths, params) -> tuple[jax.Array, ...]:
    """`decode_megakernel`'s inputs for a step: the new tokens' embeddings (B,) -> (B, D), RoPE's cos and sin
    at positions `lengths`, `lengths`, the caches (L, B, S_max, Hkv * Dh) flattened to (L, B * S_max, Hkv *
    Dh) (free: a reshape of contiguous arrays), and the weights."""
    L, B, S, KVD = k_cache.shape
    cos, sin = rope_tables(lengths, HEAD_DIM, ROPE_THETA)
    caches = (k_cache.reshape(L, B * S, KVD), v_cache.reshape(L, B * S, KVD))
    return (params["embed"][tokens], cos, sin, lengths, *caches, *(params[name] for name in WEIGHTS))


def megakernel_step(tokens, k_cache, v_cache, lengths, params, tiles=None):
    """One decode step with the megakernel: logits (B, vocab) for the new tokens (B,), the caches (L, B,
    S_max, Hkv * Dh) with their keys and values written at positions `lengths`, and lengths + 1.
    lengths[b] < S_max. The kernel's runtime mask reads `lengths`: under jit it tunes on random lengths,
    so call it eagerly once to tune on real ones."""
    kvd = k_cache.shape[3]
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    logits, new_kv = decode_megakernel(*inputs, *(() if tiles is None else (tiles,)))
    rows = jnp.arange(tokens.shape[0])
    k_cache = k_cache.at[:, rows, lengths].set(new_kv[:, :, :kvd], mode="drop")
    v_cache = v_cache.at[:, rows, lengths].set(new_kv[:, :, kvd:], mode="drop")
    return logits, k_cache, v_cache, lengths + 1


WEIGHTS = (
    "attn_norm",
    "w_qkv",
    "q_norm",
    "k_norm",
    "wo",
    "ffn_norm",
    "w_gate",
    "w_up",
    "w_down",
    "final_norm",
    "w_vocab",
)
STACKED = WEIGHTS[:-2]  # the per-layer weights, stacked (L, ...)


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, jax.Array]:
    """Random weights, per-layer ones stacked by layer: norms (L, 1, n), w_qkv (L, D, (H + 2 Hkv) * Dh)
    (q's columns, then k's, then v's), wo (L, H * Dh, D), w_gate and w_up (L, D, F), w_down (L, F, D)."""
    c = config
    d, f, layers = c.d_model, c.ffn, c.layers
    hd, kvd = c.heads * HEAD_DIM, c.kv_heads * HEAD_DIM
    keys = iter(jax.random.split(jax.random.key(seed), 16))

    def normal(shape, scale):
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm_weight(*shape):
        return (1 + 0.1 * jax.random.normal(next(keys), shape)).astype(dtype)

    return {
        "embed": normal((c.vocab, d), 1.0),
        "attn_norm": norm_weight(layers, 1, d),
        "w_qkv": normal((layers, d, hd + 2 * kvd), d**-0.5),
        "q_norm": norm_weight(layers, 1, HEAD_DIM),
        "k_norm": norm_weight(layers, 1, HEAD_DIM),
        "wo": normal((layers, hd, d), hd**-0.5),
        "ffn_norm": norm_weight(layers, 1, d),
        "w_gate": normal((layers, d, f), d**-0.5),
        "w_up": normal((layers, d, f), d**-0.5),
        "w_down": normal((layers, f, d), f**-0.5),
        "final_norm": norm_weight(1, d),
        "w_vocab": normal((d, c.vocab), d**-0.5),
    }


def layer_weights(params: dict[str, jax.Array]) -> list[dict[str, jax.Array]]:
    """The per-layer weights as one dict per layer (separate arrays: what per-layer kernels and XLA read
    without slicing a stacked array)."""
    layers = params["w_qkv"].shape[0]
    return [{name: jnp.copy(params[name][layer]) for name in STACKED} for layer in range(layers)]


def random_caches(config: Config, lengths: Any, max_len: int, seed: int = 0, stacked: bool = True) -> Any:
    """Caches of random bf16 keys and values, zero past each length (a stand-in for prefilled caches):
    stacked, (k_cache, v_cache) of (L, B, max_len, Hkv * Dh), or else decode_model.py's per-layer list of
    {"k", "v"} (B, max_len, Hkv * Dh) holding the same values (each layer is drawn from a key of its own, so
    neither form is made from the other: no transient copy of a big cache)."""
    shape = (len(lengths), max_len, config.kv_heads * HEAD_DIM)
    valid = jnp.asarray((np.arange(max_len)[None, :] < np.asarray(lengths)[:, None])[:, :, None])
    keys = jax.random.split(jax.random.key(seed), (2, config.layers))

    def draw(key):
        return jnp.where(valid, jax.random.normal(key, shape, jnp.bfloat16), 0)

    if stacked:
        return tuple(jax.jit(jax.vmap(draw))(keys[n]) for n in range(2))
    draw = jax.jit(draw)
    return [{"k": draw(keys[0, layer]), "v": draw(keys[1, layer])} for layer in range(config.layers)]


def layer_caches(k_cache: jax.Array, v_cache: jax.Array) -> list[dict[str, jax.Array]]:
    """Stacked caches as decode_model.py's per-layer list of {"k", "v"} (B, S_max, Hkv * Dh)."""
    return [{"k": jnp.copy(k_cache[i]), "v": jnp.copy(v_cache[i])} for i in range(k_cache.shape[0])]


# --------------------------------------------------------------------------- baselines: per-layer kernels, XLA


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _rms(x, gamma):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def _heads(qkv, layer, lengths, heads):
    """q, k, v (B, 1, n, Dh) from a QKV projection (B, (H + 2 Hkv) * Dh) in plain JAX: QK-norm, RoPE at
    positions `lengths`, q scaled by 1/sqrt(Dh)."""
    b = qkv.shape[0]
    t = qkv.reshape(b, 1, -1, HEAD_DIM).astype(jnp.float32)
    kv_heads = (t.shape[2] - heads) // 2
    q, k, v = t[:, :, :heads], t[:, :, heads : heads + kv_heads], t[:, :, heads + kv_heads :]
    positions = lengths[:, None]
    q = rope(_rms(q, layer["q_norm"]), positions) * HEAD_DIM**-0.5
    k = rope(_rms(k, layer["k_norm"]), positions)
    return (t.astype(qkv.dtype) for t in (q, k, v))


def _ffn(x, attn, layer):
    """The layer after attention in plain JAX (decode_model.py's `_mlp`): x + attn @ W_o, then SwiGLU."""
    resid = x.astype(jnp.float32) + _dot(attn, layer["wo"])
    normed = _rms(resid, layer["ffn_norm"]).astype(x.dtype)
    hidden = jax.nn.silu(_dot(normed, layer["w_gate"])) * _dot(normed, layer["w_up"])
    return (resid + _dot(hidden.astype(x.dtype), layer["w_down"])).astype(x.dtype)


def per_layer_step(tokens, caches, lengths, layers, params):
    """The same step with decode_model.py's kernels, three per layer: `rmsnorm_matmul` (RMSNorm + QKV),
    `gqa_decode` (attention over the appended cache), `decode_mlp` (W_o, residual, RMSNorm, SwiGLU,
    residual); QK-norm, RoPE and the cache append in XLA between them; `rmsnorm_matmul` for the logits.
    `caches`: per-layer {"k", "v"} (B, S_max, Hkv * Dh); `layers`: `layer_weights(params)`."""
    b = tokens.shape[0]
    x = params["embed"][tokens]
    new_lengths = lengths + 1
    new_caches = []
    for layer, cache in zip(layers, caches, strict=True):
        heads = layer["wo"].shape[0] // HEAD_DIM
        q, k, v = _heads(rmsnorm_matmul(x, layer["attn_norm"], layer["w_qkv"]), layer, lengths, heads)
        cache = {
            "k": append(cache["k"], lengths, k.reshape(b, 1, -1)),
            "v": append(cache["v"], lengths, v.reshape(b, 1, -1)),
        }
        new_caches.append(cache)
        attn = gqa_decode(q, cache["k"], cache["v"], new_lengths).reshape(b, -1)
        x = decode_mlp(x, attn, layer["wo"], layer["ffn_norm"], layer["w_gate"], layer["w_up"], layer["w_down"])
    return rmsnorm_matmul(x, params["final_norm"], params["w_vocab"]), new_caches, new_lengths


def xla_step(tokens, caches, lengths, layers, params):
    """The same step in plain JAX, compiled by XLA (attention dense over the whole cache, masked:
    decode.py's `xla_gqa_decode`); arguments as `per_layer_step`'s."""
    b = tokens.shape[0]
    x = params["embed"][tokens]
    new_lengths = lengths + 1
    new_caches = []
    for layer, cache in zip(layers, caches, strict=True):
        heads = layer["wo"].shape[0] // HEAD_DIM
        qkv = _dot(_rms(x, layer["attn_norm"]).astype(x.dtype), layer["w_qkv"]).astype(x.dtype)
        q, k, v = _heads(qkv, layer, lengths, heads)
        cache = {
            "k": append(cache["k"], lengths, k.reshape(b, 1, -1)),
            "v": append(cache["v"], lengths, v.reshape(b, 1, -1)),
        }
        new_caches.append(cache)
        attn = xla_gqa_decode(q, cache["k"], cache["v"], new_lengths).reshape(b, -1)
        x = _ffn(x, attn, layer)
    logits = _dot(_rms(x, params["final_norm"]).astype(x.dtype), params["w_vocab"]).astype(x.dtype)
    return logits, new_caches, new_lengths


if __name__ == "__main__":
    config = Config(layers=2, d_model=256, heads=4, kv_heads=2, ffn=512, vocab=512)
    params = init_params(config)
    lengths = jnp.array([0, 1, 130, 250], jnp.int32)
    k_cache, v_cache = random_caches(config, lengths, 256)
    tokens = jnp.array([3, 1, 4, 1], jnp.int32)
    tiles = MegaTiles(block_d=128, block_kv=128, block_a=256, block_f=256, block_v=256)
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    lowered = decode_megakernel.lower(*inputs, tiles)
    print(lowered.source)
    print(f"at most {lowered.steps} steps, {lowered.vmem_bytes / 2**20:.1f} MiB of VMEM")
    print("max |kernel - loop nest|:", decode_megakernel.verify(*inputs, tiles))
    ours, *_ = megakernel_step(tokens, k_cache, v_cache, lengths, params, tiles)
    theirs, *_ = xla_step(tokens, layer_caches(k_cache, v_cache), lengths, layer_weights(params), params)
    print("max |megakernel - XLA| logits:", float(jnp.max(jnp.abs(ours.astype(float) - theirs.astype(float)))))
