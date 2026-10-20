"""The autokernel kernels of Gemma 4's text decoder (examples/gemma4.py composes them into the model).

A Gemma 4 layer (E2B: d_model 1536, 8 query heads sharing 1 kv head, head_dim 256 on sliding-window layers
and 512 on full-attention ones, GELU-tanh MLP, per-layer embeddings):

    n = rmsnorm(h) * w_in
    q = rope(rmsnorm_h(n @ Wq) * q_norm)            per head; RoPE rotate-half over the whole head
    k = rope(rmsnorm_h(n @ Wk) * k_norm),  v = rmsnorm_h(n @ Wv)       (not on KV-shared layers)
    a = attention(q, k, v)                          causal, sliding layers over a window of keys; scale 1
    h = h + rmsnorm(a @ Wo) * w_post_attn
    m = rmsnorm(h) * w_pre_ffn;  h = h + rmsnorm((gelu(m @ Wg) * (m @ Wu)) @ Wd) * w_post_ffn
    g = gelu(h @ Wple_gate) * ple_l;  h = h + rmsnorm(g @ Wple_proj) * w_post_ple
    h = h * layer_scalar

Four kinds of kernel, each one loop nest, used by prefill (B sequences of S tokens) and decode (B new
tokens, one per sequence) alike:

`qkv_projection` / `q_projection`: the input RMSNorm fused with the projections (the row scale commutes
with the matmul: rmsnorm(x) @ W = inv_rms * ((x * w_in) @ W), rmsnorm_matmul.py's trick), then the
per-head norms and RoPE as the epilogue of the reduction over d_model, on the f32 accumulators: q and k
norms with their weights, v's norm without one. RoPE is the rotate-half form over the WHOLE head, cos and
sin being input tables of the rows' positions: the proportional RoPE of the full-attention layers (only
the first quarter of the frequencies turn; the rest have frequency 0, so cos 1 and sin 0) is the same code
with other tables. x is (B, S, D); for decoding, the B new tokens are the rows of one sequence (1, B, D)
and the tables hold row b's position (qwen3_models.py's trick). KV-shared layers project only q.

`make_attention(head_dim, window)`: causal GQA attention (flash attention over key blocks, every head of
a (batch, query block) tile at once; gqa_attention.py's `gqa_step`: a kv head's group of query heads is
stacked along rows, one matmul per key block), skipping key blocks in the future or entirely before the
sliding window; softmax scale 1 (q and k are normed). Output (B, S, H * head_dim), the heads side by side.

`post_attention`: everything of a layer after attention, on rows (tokens, d_model): the output
projection (a phase over row blocks of Wo), the post-attention norm and residual, the pre-FFN norm, the
GELU MLP (a phase over blocks of hidden units, gate and up together, then their part of the down
projection), the post-FFN norm and residual, the per-layer-embedding gate and projection (small, whole
weights fetched once), its norm and residual, and the layer scalar. One tile is `rows` tokens: in prefill
big row tiles reuse each streamed weight block over many tokens (the attention's query blocks would be too
few rows: with 128-token prompts a query block is one prompt); in decode one tile holds all B tokens, so
every weight is read once per step.

`lm_head` / `lm_head_argmax`: the final norm and the tied LM head against the embedding table (V, D),
read transposed in place (`x @ e[c : c + bv, :].T`, no transposed copy of the 805 MB table), with the
final logit soft-cap `tanh(z / 30) * 30`. The argmax kernel walks the vocabulary in blocks keeping each
row's running max and its first index (in f32: exact for indices below 2**24) and returns only the token
ids (greedy decoding never writes the (B, 262144) logits to HBM; the soft-cap is monotonic, so it is
skipped).

Decode attention is decode.py's `gqa_decode` (one tile per sequence, a runtime mask stopping at its
length, every head at once): sliding layers keep a ring buffer of the last `window` positions (softmax
doesn't care about the order of the keys), full layers a contiguous cache.

Run with `uv run python examples/gemma4_kernels.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from gqa_attention import gqa_step, score_bias
from qwen3 import rotate_half, scale_columns, sum_squares
from transformer import widen

import autokernel

EPS = 1e-6  # Gemma 4's RMSNorm epsilon
LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf
SOFTCAP = 30.0  # final_logit_softcapping


@autokernel.tunable
class QkvTiles:
    """Rows per output tile (every column of q, k and v), d_model columns per step."""

    rows: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_d: int = autokernel.choice(128, 256, 512, 768, 1536, default=512)


@autokernel.tunable
class AttentionTiles:
    """Queries per output tile, keys per step."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=512)


@autokernel.tunable
class PostTiles:
    """Rows per tile, rows of Wo per output-projection step, hidden units per MLP step."""

    rows: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    block_a: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=512)


@autokernel.tunable
class HeadTiles:
    """Rows per tile, vocabulary columns per tile (or per step, for the argmax), d_model columns per step."""

    rows: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_v: int = autokernel.choice(512, 1024, 2048, 4096, 8192, default=2048)
    block_d: int = autokernel.choice(256, 512, 768, 1536, default=1536)


# --------------------------------------------------------------------------- helpers (plain JAX, in the kernels)


def gelu(x):
    """GELU, tanh approximation (`gelu_pytorch_tanh`)."""
    return jax.nn.gelu(x, approximate=True)


def rms(x, gamma):
    """RMSNorm over the last dim of f32 rows x with a (1, d) weight, in f32."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def head_norm(t, weight, dh):
    """Each head's RMSNorm over its dh lanes of t (rows, heads * dh), f32, times a (1, dh) weight if given."""
    out = []
    for h in range(t.shape[1] // dh):
        x = t[:, h * dh : (h + 1) * dh]
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS)
        out.append(x if weight is None else x * weight.astype(jnp.float32))
    return jnp.concatenate(out, axis=1) if len(out) > 1 else out[0]


def head_rope(t, cos, sin):
    """RoPE, rotate-half over each whole head of t (rows, heads * dh), f32, with the rows' (rows, dh) tables."""
    dh = cos.shape[1]
    out = []
    for h in range(t.shape[1] // dh):
        x = t[:, h * dh : (h + 1) * dh]
        out.append(x * cos + rotate_half(x, dh // 2) * sin)
    return jnp.concatenate(out, axis=1) if len(out) > 1 else out[0]


def heads_side_by_side(acc, row_sum, bq):
    """The normalized attention output, (bq, H * dh) f32, from the per-kv-head stacks (Hkv, G * bq, dh) of
    `gqa_step` (head g * G + r is row block r of kv head g's stack)."""
    kv_heads, rows, dh = acc.shape
    heads = []
    for g in range(kv_heads):
        out = acc[g] * widen(1.0 / row_sum[g], dh)
        heads += [out[r * bq : (r + 1) * bq] for r in range(rows // bq)]
    return jnp.concatenate(heads, axis=1) if len(heads) > 1 else heads[0]


def first_max(s, c, best, best_index):
    """Fold scores s (rows, bv) of the vocabulary columns c .. c + bv - 1 into each row's running max `best`
    and its first index `best_index` (both lane-wide (rows, 128) f32; ties keep the earlier index)."""
    cols = jax.lax.broadcasted_iota(jnp.int32, s.shape, 1).astype(jnp.float32) + c
    m = jnp.max(s, axis=1, keepdims=True)
    index = jnp.min(jnp.where(s == m, cols, jnp.float32(2**30)), axis=1, keepdims=True)
    better = m > best
    return jnp.where(better, m, best), jnp.where(better, index, best_index)


# --------------------------------------------------------------------------- the kernels


@autokernel.kernel
def qkv_projection(x, gamma, w_q, w_k, w_v, q_norm, k_norm, cos, sin, tiles=QkvTiles()):
    """q, k, v of a layer's input x (B, S, D): input RMSNorm, projections, the per-head norms (q, k with
    their (1, dh) weights, v without), RoPE with the rows' (S, dh) tables. Heads are blocks of columns."""
    B, S, D = x.shape
    HD, KD, Dh = w_q.shape[1], w_k.shape[1], cos.shape[1]
    tm, td = min(tiles.rows, S), min(tiles.block_d, D)
    dtype = x.dtype
    q = jnp.zeros((B, S, HD), dtype)
    k = jnp.zeros((B, S, KD), dtype)
    v = jnp.zeros((B, S, KD), dtype)
    for b in range(B):
        for i in range(0, S, tm):
            acc_q = jnp.zeros((tm, HD), jnp.float32)  # (x * w_in) @ w_q, ...
            acc_k = jnp.zeros((tm, KD), jnp.float32)
            acc_v = jnp.zeros((tm, KD), jnp.float32)
            sum_sq = jnp.zeros((tm, LANES), jnp.float32)  # each row's sum of squares, lane-wide
            for d in range(0, D, td):
                xd = x[b, i : i + tm, d : d + td]
                sum_sq += sum_squares(xd)
                xg = scale_columns(xd, gamma[0:1, d : d + td])
                acc_q += xg @ w_q[d : d + td, :]
                acc_k += xg @ w_k[d : d + td, :]
                acc_v += xg @ w_v[d : d + td, :]
            inv_rms = jax.lax.rsqrt(sum_sq / D + EPS)
            cos_t, sin_t = cos[i : i + tm, :], sin[i : i + tm, :]
            q_t = head_rope(head_norm(acc_q * widen(inv_rms, HD), q_norm[0:1, :], Dh), cos_t, sin_t)
            k_t = head_rope(head_norm(acc_k * widen(inv_rms, KD), k_norm[0:1, :], Dh), cos_t, sin_t)
            v_t = head_norm(acc_v * widen(inv_rms, KD), None, Dh)
            q = q.at[b, i : i + tm, :].set(q_t.astype(dtype))
            k = k.at[b, i : i + tm, :].set(k_t.astype(dtype))
            v = v.at[b, i : i + tm, :].set(v_t.astype(dtype))
    return q, k, v


@autokernel.kernel
def q_projection(x, gamma, w_q, q_norm, cos, sin, tiles=QkvTiles()):
    """`qkv_projection` for a KV-shared layer: q only."""
    B, S, D = x.shape
    HD, Dh = w_q.shape[1], cos.shape[1]
    tm, td = min(tiles.rows, S), min(tiles.block_d, D)
    dtype = x.dtype
    q = jnp.zeros((B, S, HD), dtype)
    for b in range(B):
        for i in range(0, S, tm):
            acc_q = jnp.zeros((tm, HD), jnp.float32)
            sum_sq = jnp.zeros((tm, LANES), jnp.float32)
            for d in range(0, D, td):
                xd = x[b, i : i + tm, d : d + td]
                sum_sq += sum_squares(xd)
                acc_q += scale_columns(xd, gamma[0:1, d : d + td]) @ w_q[d : d + td, :]
            inv_rms = jax.lax.rsqrt(sum_sq / D + EPS)
            q_t = head_norm(acc_q * widen(inv_rms, HD), q_norm[0:1, :], Dh)
            q = q.at[b, i : i + tm, :].set(head_rope(q_t, cos[i : i + tm, :], sin[i : i + tm, :]).astype(dtype))
    return q


@functools.cache
def make_attention(head_dim: int, window: int | None = None) -> autokernel.Kernel:
    """Causal GQA attention (softmax scale 1) for heads of `head_dim`; with `window`, a query sees itself and
    the window - 1 keys before it."""

    @autokernel.kernel
    def attention(q, k, v, tiles=AttentionTiles()):
        """softmax(q_h @ k_g.T) @ v_g for every head h of q (B, S, H * dh), g its kv head of k, v (B, S, Hkv
        * dh): (B, S, H * dh)."""
        B, S, HD = q.shape
        KD = k.shape[2]
        Hkv, groups = KD // head_dim, HD // KD
        bq, bk = min(tiles.block_q, S), min(tiles.block_kv, S)
        dtype = q.dtype
        out = jnp.zeros((B, S, HD), dtype)
        for b in range(B):
            for i in range(0, S, bq):
                row_max = jnp.full((Hkv, groups * bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((Hkv, groups * bq, LANES), jnp.float32)
                acc = jnp.zeros((Hkv, groups * bq, head_dim), jnp.float32)
                for j in range(0, S, bk):
                    if j > i + bq - 1 or (window is not None and j + bk + window <= i + 1):
                        continue  # every key is in the future of (or too far back for) every query
                    bias = score_bias(i, j, bq, bk, groups, window)
                    qt, kt, vt = q[b, i : i + bq, :], k[b, j : j + bk, :], v[b, j : j + bk, :]
                    row_max, row_sum, acc = gqa_step(qt, kt, vt, bias, row_max, row_sum, acc)
                out = out.at[b, i : i + bq, :].set(heads_side_by_side(acc, row_sum, bq).astype(dtype))
        return out

    return attention


@autokernel.kernel
def post_attention(
    x,
    attn,
    ple,
    wo,
    post_attn_norm,
    pre_ffn_norm,
    w_gate,
    w_up,
    w_down,
    post_ffn_norm,
    w_ple_gate,
    w_ple_proj,
    post_ple_norm,
    layer_scale,
    tiles=PostTiles(),
):
    """The rest of a layer on rows: x (M, D) the residual stream, attn (M, H * dh) the attention output, ple
    (M, P) the layer's per-layer input; norms (1, D), layer_scale (1, D) the layer scalar. (M, D)."""
    M, D = x.shape
    A, F = attn.shape[1], w_up.shape[1]
    tm, ba, bf = min(tiles.rows, M), min(tiles.block_a, A), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, tm):
        proj = jnp.zeros((tm, D), jnp.float32)
        for a in range(0, A, ba):  # phase 1: the output projection, ba rows of Wo per step
            proj += attn[i : i + tm, a : a + ba] @ wo[a : a + ba, :]
        h = x[i : i + tm, :].astype(jnp.float32) + rms(proj, post_attn_norm[0:1, :])
        normed = rms(h, pre_ffn_norm[0:1, :]).astype(dtype)
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):  # phase 2: the GELU MLP, bf hidden units per step
            hidden = gelu(normed @ w_gate[:, f : f + bf]) * (normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        h = h + rms(y, post_ffn_norm[0:1, :])
        gate = gelu(h.astype(dtype) @ w_ple_gate[:, :]) * ple[i : i + tm, :].astype(jnp.float32)
        p = gate.astype(dtype) @ w_ple_proj[:, :]
        h = (h + rms(p, post_ple_norm[0:1, :])) * layer_scale[0:1, :].astype(jnp.float32)
        out = out.at[i : i + tm, :].set(h.astype(dtype))
    return out


@autokernel.kernel
def lm_head(x, gamma, embed, tiles=HeadTiles()):
    """Soft-capped logits (M, V) of rows x (M, D): the final RMSNorm, then the tied LM head embed (V, D)."""
    M, D = x.shape
    V = embed.shape[0]
    tm, tv, td = min(tiles.rows, M), min(tiles.block_v, V), min(tiles.block_d, D)
    dtype = x.dtype
    out = jnp.zeros((M, V), dtype)
    for i in range(0, M, tm):
        for c in range(0, V, tv):
            acc = jnp.zeros((tm, tv), jnp.float32)
            sum_sq = jnp.zeros((tm, LANES), jnp.float32)
            for d in range(0, D, td):
                xd = x[i : i + tm, d : d + td]
                sum_sq += sum_squares(xd)
                acc += scale_columns(xd, gamma[0:1, d : d + td]) @ embed[c : c + tv, d : d + td].T
            logits = acc * widen(jax.lax.rsqrt(sum_sq / D + EPS), tv)
            out = out.at[i : i + tm, c : c + tv].set((jnp.tanh(logits / SOFTCAP) * SOFTCAP).astype(dtype))
    return out


@autokernel.kernel
def lm_head_argmax(x, gamma, embed, tiles=HeadTiles()):
    """The greedy next token of rows x (M, D): argmax over the vocabulary of the LM head's logits (the first
    index of the largest), as (M, 128) int32 (every lane the same)."""
    M = x.shape[0]
    V = embed.shape[0]
    tm, tv = min(tiles.rows, M), min(tiles.block_v, V)
    dtype = x.dtype
    out = jnp.zeros((M, LANES), jnp.int32)
    for i in range(0, M, tm):
        normed = rms(x[i : i + tm, :].astype(jnp.float32), gamma[0:1, :]).astype(dtype)
        best = jnp.full((tm, LANES), -jnp.inf, jnp.float32)
        best_index = jnp.zeros((tm, LANES), jnp.float32)
        for c in range(0, V, tv):  # bv vocabulary columns per step
            best, best_index = first_max(normed @ embed[c : c + tv, :].T, c, best, best_index)
        out = out.at[i : i + tm, :].set(best_index.astype(jnp.int32))
    return out
