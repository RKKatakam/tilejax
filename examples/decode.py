"""Decode attention over a KV cache: a new token (or a few) per sequence, against ragged cache lengths.

    out[b, t, h] = softmax(q[b, t, h] @ K[b, kv(h), :n].T) @ V[b, kv(h), :n],   n = lengths[b] - T + t + 1

Generating a token reads every cached key and value of its sequence once and does almost no arithmetic
with them: decode attention is bound by HBM bandwidth (a v5e's ~819 GB/s). Fast means reading only the
valid part of each cache, in big contiguous blocks, with the DMA engines never idle. Three kernels, each
one loop nest with ONE TILE PER SEQUENCE and one reduction loop over its cache:

`gqa_decode(q, k_cache, v_cache, lengths)`: grouped-query attention (Qwen, Llama: H query heads share Hkv
key/value heads; head_dim a multiple of 128) over contiguous caches (B, S_max, Hkv * Dh), a kv head being
a block of Dh columns. The reduction loop walks the cache in blocks of `block_kv` positions under a
RUNTIME MASK, `if j >= lengths[b]: continue`: the masks are evaluated on the device just before the
kernel, which then steps through only the blocks holding valid keys (an empty sequence takes one empty
step). A step streams one (block_kv, Hkv * Dh) block of K and of V: whole cache rows, one contiguous DMA
each, prefetched across sequences. The last, partial block is masked in the tile (`cols < lengths[b]`).

  All heads at once, as one matmul per block: the query rows (H, Dh) are spread BLOCK-DIAGONALLY over
  the kv heads' column blocks once per sequence (`block_diagonal`: row h keeps its values in kv head
  h // G's block, zeros elsewhere; G = H / Hkv), so `q_bd @ k_block.T` (H, Hkv * Dh) x (Hkv * Dh, bk) is
  every head's scores against its own kv head, and `p @ v_block` (H, bk) x (bk, Hkv * Dh) holds each
  head's output in its kv head's column block (`own_blocks` picks it at the end). The MXU loads each K
  and V block once either way, exactly as one small (G, Dh) x (Dh, bk) matmul per kv head would; but the
  softmax runs on dense (H, bk) scores instead of Hkv arrays of G (4-8) rows each, filling half a vreg's
  sublanes, and no array is sliced at a G-row boundary. The zeros cost MXU passes, which decode has to
  spare. The online softmax keeps row statistics lane-wide, (H, 128), as the other examples do.

  Why one tile per sequence, not per (sequence, kv head) as JAX's paged_attention does: a kv head's slice
  of a cache block is a column window, 256 bytes of every 2 KiB row (a strided DMA), and per-head tiles
  take Hkv times more steps, each paying the gridless loop's fixed ~0.2-0.35 us, for Hkv times fewer
  bytes. Per sequence, a 512-position block is 1 MiB of K and 1 MiB of V per step (Hkv 8, Dh 128).

`paged_gqa_decode(q, k_pages, v_pages, block_tables, lengths)`: the same over a paged cache (vLLM's
layout, kv_cache.py): pages (num_pages, page_size, Hkv * Dh), sequence b's i-th page block_tables[b, i].
The reduction loop walks a sequence's pages, one per step, reading `k_pages[block_tables[b, p], :, :]`: a
RUNTIME WINDOW, its page chosen on the device from the table in SMEM. Pages of 128+ positions keep the
fixed cost per step small against the bytes it moves (a step can't gather several pages: a tile read's
index can't come from a loop inside the tile code).

`mla_decode(q_c, q_r, c_kv, k_rope, lengths)`: multi-head latent attention (DeepSeek-V2/V3, Kimi K2) in
the ABSORBED form. The cache holds, per position, a 512-wide latent c, from which every head's key and
value are projected (k_nope_h = c @ W_uk[h], v_h = c @ W_uv[h]), and one 64-wide RoPE key shared by all
heads. q_nope_h . k_nope_h = (q_nope_h @ W_uk[h].T) . c, so the query absorbs W_uk (`absorb_queries`,
q_c (H, 512)), and every head attends to the SAME latents: multi-query attention with 512 + 64-wide keys
and 512-wide values,

    s = q_c @ c.T + q_r @ k_rope.T,    o_c = softmax(s) @ c,    out_h = o_c[h] @ W_uv[h]

All H heads (16-128) are the rows of one matmul against each latent block: MXU-friendly, each block read
once, for every head. The RoPE keys are stored ZERO-PADDED to a 128-lane slot, [k_rope | 0]
(B, S_max, 128), and q_r likewise [q_rope | 0]: the products are exact, and a (bk, 128) block is an
aligned DMA and a plain matmul operand (64-lane windows mean lane shuffles, which Mosaic does worst). It
costs 64 bf16 per position, 11% more bytes than the 576 the math needs. `mla_attention` wraps it: W_uk
and W_uv are applied by XLA, before and after.

A FEW NEW TOKENS per sequence (speculative decoding's drafts, chunked prefill): q is (B, T, H, Dh) and the
kernels stack the T * H query rows (row t * H + h). `lengths` counts the cache positions INCLUDING the T
new tokens, which are appended first (as JAX's ragged_paged_attention expects too): token t sits at
position lengths[b] - T + t and sees the positions up to its own (causal within the new tokens).

q, q_c and q_r come pre-scaled by the softmax scale (1/sqrt(Dh); MLA: 1/sqrt(128 + 64)). lengths[b] must
be in [T, S_max], or 0 for an empty slot (its output is 0). The kernels are plain JAX loop nests, and
`kernel.fn` runs them as written, eagerly (the runtime masks branch on `lengths`); the kernels themselves
run anywhere, also under jit. The baselines (`xla_gqa_decode`, `xla_mla_decode`: dense over the whole
cache, masked; `jax_paged_attention`, `jax_ragged_paged_attention`: JAX's Pallas TPU kernels) compute the
same.

Run with `uv run python examples/decode.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

import autokernel

LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@autokernel.tunable
class DecodeTiles:
    """Cache positions per step."""

    block_kv: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


def widen(stat, n):
    """A lane-wide (rows, 128) row statistic as (rows, n): repeated across lanes, or its first n lanes."""
    return jnp.tile(stat, (1, n // LANES)) if n >= LANES else stat[:, :n]


def _dot(a, b):
    """a @ b accumulating in f32, as `@` in kernel code means. (A helper is plain JAX.)"""
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def query_rows(qt):
    """A sequence's queries (T, H, d) as rows (T * H, d), row t * H + h being token t's head h (in f32,
    where merging the leading dims is a free relabelling of (8, 128) tiles)."""
    t, h, d = qt.shape
    return qt.astype(jnp.float32).reshape(t * h, d)


def _blocks_below(x, size, count):
    """x // size for int32 x in [0, size * count), by comparisons: the TPU's vector units have no integer
    division."""
    out = jnp.zeros(x.shape, jnp.int32)
    for k in range(1, count):
        out += (x >= k * size).astype(jnp.int32)
    return out


def row_index(n_rows, heads, kv_heads):
    """For query rows (row t * heads + h is token t's head h), lane-wide (n_rows, 128) int32: each row's
    token t, and the kv head its query head attends with, h // (heads / kv_heads) (as JAX's GQA kernels
    group heads)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (n_rows, LANES), 0)
    token = _blocks_below(rows, heads, n_rows // heads)
    return token, _blocks_below(rows - token * heads, heads // kv_heads, kv_heads)


def block_diagonal(rows, group, kv_heads, dtype):
    """Query rows (R, Dh) spread over the kv heads' column blocks, (R, kv_heads * Dh): a row keeps its
    values in the block of its kv head (`group`, lane-wide) and is 0 elsewhere, so `block_diagonal(q) @
    k.T` is every head's scores against its own kv head's keys, in one matmul over a whole cache block."""
    group = widen(group, rows.shape[1])
    return jnp.concatenate([jnp.where(group == g, rows, 0.0) for g in range(kv_heads)], axis=1).astype(dtype)


def own_blocks(acc, group, kv_heads):
    """Each row's own kv head's column block of `acc` (R, kv_heads * Dh): (R, Dh)."""
    dh = acc.shape[1] // kv_heads
    group = widen(group, dh)
    out = acc[:, :dh]
    for g in range(1, kv_heads):
        out = jnp.where(group == g, acc[:, g * dh : (g + 1) * dh], out)
    return out


def visible(token, bk, j, length, new_tokens):
    """Which (query row, cache position) pairs attend, for a cache block at position j: positions before
    the sequence's length, and, with several new tokens (token t at position length - new_tokens + t;
    `token` lane-wide, per row), none after the row's own token."""
    cols = j + jax.lax.broadcasted_iota(jnp.int32, (token.shape[0], bk), 1)
    if new_tokens == 1:
        return cols < length
    return cols <= length - new_tokens + widen(token, bk)


def online_softmax(s, v, ok, row_max, row_sum, acc):
    """One online-softmax step: scores `s` (R, bk) (masked where not `ok`) against the values `v` (bk, Dv);
    the lane-wide running max and sum (R, 128) and the accumulator (R, Dv)."""
    bk, dv = s.shape[1], v.shape[1]
    s = jnp.where(ok, s, MASK_VALUE)
    max_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - widen(max_new, bk))
    alpha = jnp.exp(row_max - max_new)
    row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
    acc = widen(alpha, dv) * acc + _dot(p.astype(v.dtype), v)
    return max_new, row_sum, acc


def finish(acc, row_sum, shape, dtype):
    """The normalized output, (T * H, d) rows as `shape` (T, H, d); rows that saw no key (an empty
    sequence) are 0."""
    out = acc * widen(1.0 / jnp.maximum(row_sum, 1e-30), acc.shape[1])
    return out.reshape(shape).astype(dtype)


@autokernel.kernel
def gqa_decode(q, k_cache, v_cache, lengths, tiles=DecodeTiles()):
    """GQA decode attention over a contiguous cache: q (B, T, H, Dh), caches (B, S_max, Hkv * Dh)."""
    B, T, H, Dh = q.shape
    S, W = k_cache.shape[1], k_cache.shape[2]
    Hkv, R = W // Dh, T * H
    bk, dtype = min(tiles.block_kv, S), q.dtype
    out = jnp.zeros((B, T, H, Dh), dtype)
    for b in range(B):
        token, group = row_index(R, H, Hkv)
        qt = block_diagonal(query_rows(q[b, :, :, :]), group, Hkv, dtype)  # once per sequence
        row_max = jnp.full((R, LANES), MASK_VALUE, jnp.float32)
        row_sum = jnp.zeros((R, LANES), jnp.float32)
        acc = jnp.zeros((R, W), jnp.float32)  # every kv head's values; a row keeps its own block
        for j in range(0, S, bk):
            if j >= lengths[b]:  # past the sequence's end: decided on the device, before the kernel runs
                continue
            s = qt @ k_cache[b, j : j + bk, :].T
            ok = visible(token, bk, j, lengths[b], T)
            row_max, row_sum, acc = online_softmax(s, v_cache[b, j : j + bk, :], ok, row_max, row_sum, acc)
        out = out.at[b, :, :, :].set(finish(own_blocks(acc, group, Hkv), row_sum, (T, H, Dh), dtype))
    return out


@autokernel.kernel
def paged_gqa_decode(q, k_pages, v_pages, block_tables, lengths):
    """GQA decode attention over a paged cache: q (B, T, H, Dh), pages (num_pages, page_size, Hkv * Dh),
    sequence b's pages listed in block_tables[b, :]."""
    B, T, H, Dh = q.shape
    P, W = k_pages.shape[1], k_pages.shape[2]
    pages = block_tables.shape[1]
    Hkv, R = W // Dh, T * H
    dtype = q.dtype
    out = jnp.zeros((B, T, H, Dh), dtype)
    for b in range(B):
        token, group = row_index(R, H, Hkv)
        qt = block_diagonal(query_rows(q[b, :, :, :]), group, Hkv, dtype)
        row_max = jnp.full((R, LANES), MASK_VALUE, jnp.float32)
        row_sum = jnp.zeros((R, LANES), jnp.float32)
        acc = jnp.zeros((R, W), jnp.float32)
        for p in range(pages):
            if p * P >= lengths[b]:
                continue
            s = qt @ k_pages[block_tables[b, p], :, :].T  # a runtime window: the page is chosen on the device
            ok = visible(token, P, p * P, lengths[b], T)
            vt = v_pages[block_tables[b, p], :, :]
            row_max, row_sum, acc = online_softmax(s, vt, ok, row_max, row_sum, acc)
        out = out.at[b, :, :, :].set(finish(own_blocks(acc, group, Hkv), row_sum, (T, H, Dh), dtype))
    return out


@autokernel.kernel
def mla_decode(q_c, q_r, c_kv, k_rope, lengths, tiles=DecodeTiles()):
    """Absorbed MLA decode attention: q_c (B, T, H, C), q_r (B, T, H, 128), the latent cache (B, S_max, C)
    and the RoPE key cache (B, S_max, 128); out (B, T, H, C), each head's softmax-weighted latents."""
    B, T, H, C = q_c.shape
    S = c_kv.shape[1]
    R = T * H
    bk, dtype = min(tiles.block_kv, S), q_c.dtype
    out = jnp.zeros((B, T, H, C), dtype)
    for b in range(B):
        token, _ = row_index(R, H, 1)
        qc = query_rows(q_c[b, :, :, :]).astype(dtype)
        qr = query_rows(q_r[b, :, :, :]).astype(dtype)
        row_max = jnp.full((R, LANES), MASK_VALUE, jnp.float32)
        row_sum = jnp.zeros((R, LANES), jnp.float32)
        acc = jnp.zeros((R, C), jnp.float32)
        for j in range(0, S, bk):
            if j >= lengths[b]:
                continue
            ct = c_kv[b, j : j + bk, :]
            s = qc @ ct.T + qr @ k_rope[b, j : j + bk, :].T
            ok = visible(token, bk, j, lengths[b], T)
            row_max, row_sum, acc = online_softmax(s, ct, ok, row_max, row_sum, acc)
        out = out.at[b, :, :, :].set(finish(acc, row_sum, (T, H, C), dtype))
    return out


# --------------------------------------------------------------------------- MLA around the kernel, plain JAX


def absorb_queries(q_nope, q_rope, w_uk):
    """MLA's queries in the absorbed form: q_c = q_nope @ W_uk[h].T (B, T, H, C) and q_rope zero-padded to
    128 lanes (B, T, H, 128), both scaled by the softmax scale 1/sqrt(nope + rope dims). q_nope (B, T, H,
    Dn), q_rope (B, T, H, Dr) with RoPE applied, w_uk (H, C, Dn): head h's keys are k_nope = c @ w_uk[h]."""
    scale = (q_nope.shape[-1] + q_rope.shape[-1]) ** -0.5
    q_c = jnp.einsum("bthd,hcd->bthc", q_nope, w_uk, preferred_element_type=jnp.float32) * scale
    q_r = q_rope.astype(jnp.float32) * scale
    q_r = jnp.pad(q_r, [(0, 0)] * 3 + [(0, LANES - q_r.shape[-1])])
    return q_c.astype(q_nope.dtype), q_r.astype(q_nope.dtype)


def mla_attention(q_nope, q_rope, c_kv, k_rope, lengths, w_uk, w_uv, tiles=None):
    """MLA decode attention, (B, T, H, Dv): absorb W_uk into the queries (XLA), attend to the latent cache
    (`mla_decode`), and project each head's latents by W_uv (H, C, Dv) (XLA): out_h = o_c[h] @ w_uv[h]."""
    q_c, q_r = absorb_queries(q_nope, q_rope, w_uk)
    args = (q_c, q_r, c_kv, k_rope, lengths) + (() if tiles is None else (tiles,))
    latents = mla_decode(*args)
    b, t, h, c = latents.shape
    heads = latents.reshape(b * t, h, c).transpose(1, 0, 2)  # head-major: a plain batched matmul
    out = jnp.einsum("hnc,hcd->hnd", heads, w_uv, preferred_element_type=jnp.float32)
    return out.transpose(1, 0, 2).reshape(b, t, h, -1).astype(q_nope.dtype)


# --------------------------------------------------------------------------- baselines: XLA, JAX's kernels


def _masked_softmax_weights(s, lengths, new_tokens, cache_len):
    """softmax over the last axis of scores s (B, T, ..., S_max), each new token seeing the cache up to its
    own position; rows of empty sequences are 0."""
    position = lengths[:, None] - new_tokens + jnp.arange(new_tokens)[None, :]  # (B, T)
    ok = jnp.arange(cache_len)[None, None, :] <= position[:, :, None]  # (B, T, S_max)
    ok = ok.reshape(ok.shape[:2] + (1,) * (s.ndim - 3) + ok.shape[2:])
    p = jax.nn.softmax(jnp.where(ok, s, MASK_VALUE), axis=-1)
    return jnp.where((lengths > 0).reshape((-1,) + (1,) * (s.ndim - 1)), p, 0.0)


def xla_gqa_decode(q, k_cache, v_cache, lengths):
    """GQA decode attention in plain JAX, compiled by XLA: dense over the whole cache, masked."""
    b, t, h, dh = q.shape
    s_max, kv_heads = k_cache.shape[1], k_cache.shape[2] // dh
    qg = q.reshape(b, t, kv_heads, h // kv_heads, dh)
    k = k_cache.reshape(b, s_max, kv_heads, dh)
    v = v_cache.reshape(b, s_max, kv_heads, dh)
    s = jnp.einsum("btgrd,bsgd->btgrs", qg, k, preferred_element_type=jnp.float32)
    p = _masked_softmax_weights(s, lengths, t, s_max).astype(v.dtype)
    out = jnp.einsum("btgrs,bsgd->btgrd", p, v, preferred_element_type=jnp.float32)
    return out.reshape(b, t, h, dh).astype(q.dtype)


def xla_mla_decode(q_c, q_r, c_kv, k_rope, lengths):
    """Absorbed MLA decode attention in plain JAX (the latents per head, like `mla_decode`), by XLA."""
    s = jnp.einsum("bthc,bsc->bths", q_c, c_kv, preferred_element_type=jnp.float32)
    s = s + jnp.einsum("bthr,bsr->bths", q_r, k_rope, preferred_element_type=jnp.float32)
    p = _masked_softmax_weights(s, lengths, q_c.shape[1], c_kv.shape[1]).astype(c_kv.dtype)
    return jnp.einsum("bths,bsc->bthc", p, c_kv, preferred_element_type=jnp.float32).astype(q_c.dtype)


def jax_paged_attention(q, k_pages, v_pages, block_tables, lengths, pages_per_compute_block: int):
    """JAX's Pallas paged_attention (one new token per sequence) on its own page layout (Hkv, num_pages, P,
    Dh) (`kv_cache.jax_paged_layout`); q (B, 1, H, Dh) as for `gqa_decode`."""
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

    out = paged_attention(
        q[:, 0], k_pages, v_pages, lengths, block_tables, pages_per_compute_block=pages_per_compute_block
    )
    return out[:, None]


def jax_ragged_paged_attention(q, kv_pages, block_tables, lengths, **block_sizes: Any):
    """JAX's Pallas ragged_paged_attention on its own page layout (num_pages, P, 2 * Hkv, Dh)
    (`kv_cache.ragged_kv_pages`); q (B, T, H, Dh) as for `gqa_decode`, `lengths` counting the new tokens."""
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention

    b, t, h, dh = q.shape
    cu_q_lens = jnp.arange(b + 1, dtype=jnp.int32) * t
    num_seqs = jnp.array([b], jnp.int32)
    out = ragged_paged_attention(
        q.reshape(b * t, h, dh), kv_pages, lengths, block_tables, cu_q_lens, num_seqs, sm_scale=1.0, **block_sizes
    )
    return out.reshape(b, t, h, dh)


if __name__ == "__main__":
    B, S, H, KV_HEADS = 4, 1024, 8, 2
    keys = jax.random.split(jax.random.key(0), 3)
    q = (jax.random.normal(keys[0], (B, 1, H, 128)) * 128**-0.5).astype(jnp.bfloat16)
    k_cache, v_cache = (jax.random.normal(key, (B, S, KV_HEADS * 128), jnp.bfloat16) for key in keys[1:])
    lengths = jnp.array([1, 200, 700, 1024], jnp.int32)
    tiles = DecodeTiles(block_kv=256)
    lowered = gqa_decode.lower(q, k_cache, v_cache, lengths, tiles)
    print(lowered.source)
    out = gqa_decode(q, k_cache, v_cache, lengths, tiles)
    steps = int(jax.jit(lowered.entry.__globals__["gqa_decode_steps"])(lengths)[2][0])
    print(f"ran {steps} of at most {lowered.steps} steps: only the cache blocks before each sequence's length")
    diff = jnp.max(jnp.abs(out.astype(jnp.float32) - xla_gqa_decode(q, k_cache, v_cache, lengths).astype(jnp.float32)))
    print("max |autokernel - XLA|:", float(diff))
    print("max |kernel - loop nest|:", gqa_decode.verify(q, k_cache, v_cache, lengths, tiles))
