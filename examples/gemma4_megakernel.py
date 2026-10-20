"""A whole Gemma 4 decode step as ONE autokernel kernel: all 35 layers and the LM head, with cross-layer prefetch.

At small batches a decode step is bound by reading the weights (E2B: 2.2 G parameters, 4.5 GB in bf16 with
the 805 MB tied LM head; ~5.5 ms at the v5e's 819 GB/s), so it is fast when the DMA engines never idle.
examples/gemma4.py's per-layer path runs three kernels per layer, each starting with its pipeline empty and
ending by writing its rows to HBM; `gemma4_megakernel` runs every layer and the greedy LM head in one kernel:

    one tile: all B rows                          (the parallel loop has one iteration)
      h = embeddings                              the residual stream, f32, in VMEM from the first layer to the last
      REPEAT LOOP over the layers:
        P1 q, sliding         d model blocks:  acc += n[d] @ w_q_s[q_slot[layer], d : d + bd]
        P2 k, v, sliding      (non-shared sliding layers only)
        P3 q, full;  P4 k, v, full
           norms, RoPE; the newest k|v of each kind (a KV-shared layer's own-token term), per-sequence state
        P5 attention, sliding: the ring caches, a block of one sequence's slots per step
        P6 attention, full: the full caches
        P7 Wo, sliding;  P8 Wo, full;  post-attention norm, residual, pre-FFN norm
        P9 MLP, narrow (6144);  P10 MLP, wide (12288);  post-FFN norm, residual
        P11 the per-layer-embedding gate and projection, its norm, residual; the layer scalar
      final norm
      P12 the LM head over blocks of the vocabulary, keeping each row's running max and first argmax

Layers differ: sliding (head_dim 256) or full attention (512), KV-shared or not, narrow or wide MLPs. Each
weight lives in a stack per kind (gemma4.py's layout) and every phase reads its layer's block through an
INDEX ARRAY (`w_q_s[q_slot[layer], d : d + bd, :]`: a runtime window, its block chosen on the device), and
phases that don't apply to a layer are RUNTIME MASKS on per-layer index arrays (`if kind[layer] != 0:
continue`): they take one empty step (no copy, no compute), and the pipeline still prefetches across them,
so while layer l computes its last MLP block the next layer's first q block is on its way. The code
between phases runs for both kinds every layer; the inactive kind's accumulators are zeros, which flow
through harmlessly (and are masked out of the result by the selects on `kind[layer]`).

Attention, one block of one sequence's cache per step (decode_megakernel.py's design, for one kv head): the
caches are stacks per kind, (n, B * positions, dh), sequence b's positions b * positions .. ; the runtime
mask `j % R >= seen_s[j // R]` stops each sequence at its cached positions. The NEW token's own key and value
start the online softmax (score q . k_new, weight 1, value v_new); a KV-shared layer's own-token term is the
newest non-shared layer of its kind's k and v (layer 13's / 14's), carried. A sliding layer's ring of R
slots holds positions len - R .. len - 1 before this token is written: slot len % R (position len - R) is
outside the window and masked (`skip_s`). The new keys and values are outputs, written into the caches by
XLA after the kernel (in place, the caches donated).

Moving between rows (sequences on sublanes: what the projections make and take) and per-sequence head
blocks (heads on sublanes) is decode_megakernel.py's iota-mask, select and sublane-sum code, unrolled over
the batch: the kernel is for small batches (gemma4.py's Generator uses it up to MEGA_MAX_BATCH sequences).

The kernel needs kv_heads == 1 (Gemma 4 E2B's). Run with `uv run python examples/gemma4_megakernel.py`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from decode import LANES, online_softmax, widen
from decode_megakernel import lane_blocks, place_rows, seq_row
from gemma4 import (
    I32,
    Config,
    _tables,
    embed_tokens,
    per_layer_inputs,
)
from gemma4_kernels import MegaTiles, first_max, gelu, head_norm, head_rope, rms

import autokernel

# --------------------------------------------------------------------------- helpers (plain JAX, in the kernel)


def sequence_heads(rows, b, heads, dh):
    """Sequence b's heads of query rows (B, heads * dh), f32, as (heads, dh): head h in row h."""
    row = seq_row(rows, b)
    ids = jax.lax.broadcasted_iota(jnp.int32, (heads, dh), 0)
    out = jnp.zeros((heads, dh), jnp.float32)
    for h in range(heads):
        out = jnp.where(ids == h, row[:, h * dh : (h + 1) * dh], out)
    return out


def attention_start(q, kv, heads, dh, dtype):
    """Per-sequence attention state holding just the new token: q (B, heads * dh) and its kv head's [k | v]
    (B, 2 * dh), f32. The queries (B, heads, dh) in `dtype`; the online softmax's lane-wide max (B, heads, 128)
    = the score q_h . k_new, sum (B, heads, 128) = 1, accumulator (B, heads, dh) = v_new."""
    queries, scores, values = [], [], []
    for b in range(q.shape[0]):
        qb = sequence_heads(q, b, heads, dh)
        new = seq_row(kv, b)
        queries.append(qb.astype(dtype))
        scores.append(jnp.broadcast_to(jnp.sum(qb * new[:, :dh], axis=1, keepdims=True), (heads, LANES)))
        values.append(jnp.broadcast_to(new[:, dh:], (heads, dh)))
    row_sum = jnp.ones((q.shape[0], heads, LANES), jnp.float32)
    return jnp.stack(queries), jnp.stack(scores), row_sum, jnp.stack(values)


def attention_rows(acc, row_sum):
    """The normalized attention output of every sequence as rows (B, heads * dh), f32, from the per-sequence
    accumulators (B, heads, dh) and lane-wide sums (B, heads, 128)."""
    batch, heads, dh = acc.shape
    ids = jax.lax.broadcasted_iota(jnp.int32, (heads, dh), 0)
    rows = []
    for b in range(batch):
        out = acc[b] * widen(1.0 / row_sum[b], dh)
        pieces = [jnp.sum(jnp.where(ids == h, out, 0.0), axis=0, keepdims=True) for h in range(heads)]
        rows.append(jnp.concatenate(pieces, axis=1) if heads > 1 else pieces[0])
    return place_rows(rows)


def new_kv(acc_k, acc_v, k_norm, cos, sin, dh):
    """A layer's new [k | v] rows (B, 2 * dh), f32: k normed and rotated, v normed."""
    k = head_rope(head_norm(acc_k, k_norm, dh), cos, sin)
    return jnp.concatenate([k, head_norm(acc_v, None, dh)], axis=1)


# --------------------------------------------------------------------------- the kernel


@autokernel.kernel
def gemma4_megakernel(
    x,
    ple,
    cos_s,
    sin_s,
    cos_f,
    sin_f,
    kind,
    q_slot,
    kv_slot,
    own,
    mlp_slot,
    wide,
    seen_s,
    skip_s,
    seen_f,
    ring_k,
    ring_v,
    full_k,
    full_v,
    attn_norm,
    w_q_s,
    w_k_s,
    w_v_s,
    q_norm_s,
    k_norm_s,
    w_q_f,
    w_k_f,
    w_v_f,
    q_norm_f,
    k_norm_f,
    wo_s,
    wo_f,
    post_attn_norm,
    pre_ffn_norm,
    w_gate_n,
    w_up_n,
    w_down_n,
    w_gate_w,
    w_up_w,
    w_down_w,
    post_ffn_norm,
    w_ple_gate,
    w_ple_proj,
    post_ple_norm,
    layer_scale,
    final_norm,
    embed,
    tiles=MegaTiles(),
):
    """A greedy decode step: x (B, D) the new tokens' embeddings, ple (L, B, P) their per-layer inputs, RoPE
    tables at their positions per kind; per layer (index arrays, (L,)): kind (0 sliding, 1 full), q_slot,
    kv_slot, own (1: computes its own k, v), mlp_slot, wide (1: wide MLP); per sequence (B,): seen_s, skip_s
    (cached ring slots visible, the one to skip or -1), seen_f; caches (n, B * positions, dh) per kind; the
    weight stacks. Returns the next tokens (B, 128) int32 (every lane the same) and every layer's new [k | v]
    rows per kind, (L, B, 2 * dh)."""
    B, D = x.shape
    L, _, P = ple.shape
    DS, DF = cos_s.shape[1], cos_f.shape[1]
    H = w_q_s.shape[2] // DS
    R, S = ring_k.shape[1] // B, full_k.shape[1] // B
    FN, FW, V = w_up_n.shape[2], w_up_w.shape[2], embed.shape[0]
    bd, bks, bkf = min(tiles.block_d, D), min(tiles.block_kv, R), min(tiles.block_kv, S)
    bas, baf = min(tiles.block_a, H * DS), min(tiles.block_a, H * DF)
    bfn, bfw, bp, bv = min(tiles.block_f, FN), min(tiles.block_f, FW), min(tiles.block_p, P), min(tiles.block_v, V)
    dtype = x.dtype
    tokens = jnp.zeros((B, LANES), jnp.int32)
    rows_s = jnp.zeros((L, B, 2 * DS), dtype)
    rows_f = jnp.zeros((L, B, 2 * DF), dtype)
    for i in range(0, B, B):
        h = x[i : i + B, :].astype(jnp.float32)  # the residual stream, f32, in VMEM throughout
        new_s = jnp.zeros((L, B, 2 * DS), dtype)
        new_f = jnp.zeros((L, B, 2 * DF), dtype)
        kv_s = jnp.zeros((B, 2 * DS), jnp.float32)  # the newest sliding / full layer's [k | v]
        kv_f = jnp.zeros((B, 2 * DF), jnp.float32)
        for layer in range(L):  # a repeat loop: the next layer's weights are fetched while this one runs
            n_blocks = lane_blocks(rms(h, attn_norm[layer, :, :]).astype(dtype), bd)
            acc_qs = jnp.zeros((B, H * DS), jnp.float32)
            for d in range(0, D, bd):  # P1: q of a sliding layer
                if kind[layer] != 0:
                    continue
                acc_qs += n_blocks[d // bd] @ w_q_s[q_slot[layer], d : d + bd, :]
            acc_ks = jnp.zeros((B, DS), jnp.float32)
            acc_vs = jnp.zeros((B, DS), jnp.float32)
            for d2 in range(0, D, bd):  # P2: k, v of a non-shared sliding layer
                if kind[layer] != 0 or own[layer] == 0:
                    continue
                acc_ks += n_blocks[d2 // bd] @ w_k_s[kv_slot[layer], d2 : d2 + bd, :]
                acc_vs += n_blocks[d2 // bd] @ w_v_s[kv_slot[layer], d2 : d2 + bd, :]
            acc_qf = jnp.zeros((B, H * DF), jnp.float32)
            for d3 in range(0, D, bd):  # P3: q of a full-attention layer
                if kind[layer] != 1:
                    continue
                acc_qf += n_blocks[d3 // bd] @ w_q_f[q_slot[layer], d3 : d3 + bd, :]
            acc_kf = jnp.zeros((B, DF), jnp.float32)
            acc_vf = jnp.zeros((B, DF), jnp.float32)
            for d4 in range(0, D, bd):  # P4: k, v of a non-shared full-attention layer
                if kind[layer] != 1 or own[layer] == 0:
                    continue
                acc_kf += n_blocks[d4 // bd] @ w_k_f[kv_slot[layer], d4 : d4 + bd, :]
                acc_vf += n_blocks[d4 // bd] @ w_v_f[kv_slot[layer], d4 : d4 + bd, :]
            q_s = head_rope(
                head_norm(acc_qs, q_norm_s[q_slot[layer], :, :], DS), cos_s[i : i + B, :], sin_s[i : i + B, :]
            )
            q_f = head_rope(
                head_norm(acc_qf, q_norm_f[q_slot[layer], :, :], DF), cos_f[i : i + B, :], sin_f[i : i + B, :]
            )
            made_s = new_kv(
                acc_ks, acc_vs, k_norm_s[kv_slot[layer], :, :], cos_s[i : i + B, :], sin_s[i : i + B, :], DS
            )
            made_f = new_kv(
                acc_kf, acc_vf, k_norm_f[kv_slot[layer], :, :], cos_f[i : i + B, :], sin_f[i : i + B, :], DF
            )
            kv_s = jnp.where((kind[layer] == 0) & (own[layer] == 1), made_s, kv_s)
            kv_f = jnp.where((kind[layer] == 1) & (own[layer] == 1), made_f, kv_f)
            new_s = new_s.at[layer].set(made_s.astype(dtype))
            new_f = new_f.at[layer].set(made_f.astype(dtype))
            q_bd_s, max_s, sum_s, acc_s = attention_start(q_s, kv_s, H, DS, dtype)
            q_bd_f, max_f, sum_f, acc_f = attention_start(q_f, kv_f, H, DF, dtype)
            for j in range(0, B * R, bks):  # P5: sliding attention, bks ring slots of sequence j // R per step
                if kind[layer] != 0 or j % R >= seen_s[j // R]:
                    continue
                s = q_bd_s[j // R] @ ring_k[kv_slot[layer], j : j + bks, :].T
                cols = j % R + jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
                ok = (cols < seen_s[j // R]) & (cols != skip_s[j // R])
                vt = ring_v[kv_slot[layer], j : j + bks, :]
                m_j, l_j, a_j = online_softmax(s, vt, ok, max_s[j // R], sum_s[j // R], acc_s[j // R])
                max_s = max_s.at[j // R].set(m_j)
                sum_s = sum_s.at[j // R].set(l_j)
                acc_s = acc_s.at[j // R].set(a_j)
            for j2 in range(0, B * S, bkf):  # P6: full attention, bkf positions of sequence j2 // S per step
                if kind[layer] != 1 or j2 % S >= seen_f[j2 // S]:
                    continue
                s2 = q_bd_f[j2 // S] @ full_k[kv_slot[layer], j2 : j2 + bkf, :].T
                cols2 = j2 % S + jax.lax.broadcasted_iota(jnp.int32, s2.shape, 1)
                vt2 = full_v[kv_slot[layer], j2 : j2 + bkf, :]
                m_k, l_k, a_k = online_softmax(
                    s2, vt2, cols2 < seen_f[j2 // S], max_f[j2 // S], sum_f[j2 // S], acc_f[j2 // S]
                )
                max_f = max_f.at[j2 // S].set(m_k)
                sum_f = sum_f.at[j2 // S].set(l_k)
                acc_f = acc_f.at[j2 // S].set(a_k)
            attn_s = lane_blocks(attention_rows(acc_s, sum_s).astype(dtype), bas)
            attn_f = lane_blocks(attention_rows(acc_f, sum_f).astype(dtype), baf)
            proj = jnp.zeros((B, D), jnp.float32)
            for a in range(0, H * DS, bas):  # P7: output projection of a sliding layer
                if kind[layer] != 0:
                    continue
                proj += attn_s[a // bas] @ wo_s[q_slot[layer], a : a + bas, :]
            for a2 in range(0, H * DF, baf):  # P8: of a full-attention layer
                if kind[layer] != 1:
                    continue
                proj += attn_f[a2 // baf] @ wo_f[q_slot[layer], a2 : a2 + baf, :]
            h = h + rms(proj, post_attn_norm[layer, :, :])
            normed = rms(h, pre_ffn_norm[layer, :, :]).astype(dtype)
            y = jnp.zeros((B, D), jnp.float32)
            for f in range(0, FN, bfn):  # P9: narrow MLP, bfn hidden units per step
                if wide[layer] != 0:
                    continue
                hidden = gelu(normed @ w_gate_n[mlp_slot[layer], :, f : f + bfn]) * (
                    normed @ w_up_n[mlp_slot[layer], :, f : f + bfn]
                )
                y += hidden.astype(dtype) @ w_down_n[mlp_slot[layer], f : f + bfn, :]
            for f2 in range(0, FW, bfw):  # P10: wide MLP
                if wide[layer] != 1:
                    continue
                hidden2 = gelu(normed @ w_gate_w[mlp_slot[layer], :, f2 : f2 + bfw]) * (
                    normed @ w_up_w[mlp_slot[layer], :, f2 : f2 + bfw]
                )
                y += hidden2.astype(dtype) @ w_down_w[mlp_slot[layer], f2 : f2 + bfw, :]
            h = h + rms(y, post_ffn_norm[layer, :, :])
            hb = h.astype(dtype)
            p = jnp.zeros((B, D), jnp.float32)
            for c in range(0, P, bp):  # P11: the per-layer-embedding gate and projection, bp units per step
                gate = gelu(hb @ w_ple_gate[layer, :, c : c + bp]) * ple[layer, i : i + B, c : c + bp].astype(
                    jnp.float32
                )
                p += gate.astype(dtype) @ w_ple_proj[layer, c : c + bp, :]
            h = (h + rms(p, post_ple_norm[layer, :, :])) * layer_scale[layer, :, :].astype(jnp.float32)
        last = rms(h, final_norm[0:1, :]).astype(dtype)
        best = jnp.full((B, LANES), -jnp.inf, jnp.float32)
        best_index = jnp.zeros((B, LANES), jnp.float32)
        for v in range(0, V, bv):  # P12: the LM head, bv vocabulary rows per step
            best, best_index = first_max(last @ embed[v : v + bv, :].T, v, best, best_index)
        tokens = tokens.at[i : i + B, :].set(best_index.astype(jnp.int32))
        rows_s = rows_s.at[:, i : i + B, :].set(new_s)
        rows_f = rows_f.at[:, i : i + B, :].set(new_f)
    return tokens, rows_s, rows_f


# --------------------------------------------------------------------------- the decode step around it


def layer_tables(config: Config) -> tuple[jax.Array, ...]:
    """The per-layer index arrays: kind, q_slot, kv_slot, own, mlp_slot, wide."""
    c, n = config, range(config.layers)
    columns = (
        [int(c.is_full(i)) for i in n],
        [c.q_slot(i) for i in n],
        [c.kv_slot(i) for i in n],
        [int(not c.is_shared(i)) for i in n],
        [c.mlp_slot(i) for i in n],
        [int(c.width(i) == "w") for i in n],
    )
    return tuple(jnp.array(column, I32) for column in columns)


def stack_caches(config: Config, caches: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    """gemma4.py's per-layer caches as the megakernel's stacks per kind: ring_k, ring_v (n_sliding, B, R, dh)
    and full_k, full_v (n_full, B, S, dh) (a stack no layer uses holds one block of zeros)."""
    out = {}
    for kind, name in (("s", "ring"), ("f", "full")):
        layers = config.kv_layers(kind)
        for part in ("k", "v"):
            if layers:
                out[f"{name}_{part}"] = jnp.stack([caches[i][part] for i in layers])
            else:
                other = caches[0][part]
                dh = config.global_head_dim if kind == "f" else config.head_dim
                out[f"{name}_{part}"] = jnp.zeros((1, other.shape[0], LANES, config.kv_heads * dh), other.dtype)
    return out


WEIGHTS = (
    "attn_norm",
    "w_q_s",
    "w_k_s",
    "w_v_s",
    "q_norm_s",
    "k_norm_s",
    "w_q_f",
    "w_k_f",
    "w_v_f",
    "q_norm_f",
    "k_norm_f",
    "wo_s",
    "wo_f",
    "post_attn_norm",
    "pre_ffn_norm",
    "w_gate_n",
    "w_up_n",
    "w_down_n",
    "w_gate_w",
    "w_up_w",
    "w_down_w",
    "post_ffn_norm",
    "w_ple_gate",
    "w_ple_proj",
    "post_ple_norm",
    "layer_scale",
)


def kernel_inputs(params: dict[str, Any], tokens: jax.Array, caches: dict, lengths: jax.Array) -> tuple:
    """`gemma4_megakernel`'s inputs for a step: tokens (B,) at positions `lengths` (B,), stacked caches."""
    c: Config = params["config"]
    if c.kv_heads != 1:
        raise ValueError("the megakernel attends with one kv head")
    b = tokens.shape[0]
    e = embed_tokens(params, tokens)
    ple = per_layer_inputs(params, tokens, e).transpose(1, 0, 2)
    tables = _tables(c, lengths)
    (cos_s, sin_s), (cos_f, sin_f) = (
        tables.get(kind, tables[c.layer_types[0]]) for kind in ("sliding_attention", "full_attention")
    )
    if c.global_head_dim != cos_f.shape[1]:  # no full-attention layer: tables of the right width
        cos_f, sin_f = jnp.ones((b, c.global_head_dim)), jnp.zeros((b, c.global_head_dim))
    if c.head_dim != cos_s.shape[1]:
        cos_s, sin_s = jnp.ones((b, c.head_dim)), jnp.zeros((b, c.head_dim))
    ring = caches["ring_k"].shape[2]
    seen_s = jnp.minimum(lengths, ring)
    skip_s = jnp.where(lengths >= ring, lengths % ring, -1)
    flat = {name: cache.reshape(cache.shape[0], -1, cache.shape[3]) for name, cache in caches.items()}
    caches_in = (flat["ring_k"], flat["ring_v"], flat["full_k"], flat["full_v"])
    w = params["w"]
    return (
        e,
        ple,
        cos_s,
        sin_s,
        cos_f,
        sin_f,
        *layer_tables(c),
        seen_s,
        skip_s,
        lengths,
        *caches_in,
        *(w[name] for name in WEIGHTS),
        params["final_norm"],
        params["embed"],
    )


def write_caches(config: Config, caches: dict, rows_s: jax.Array, rows_f: jax.Array, lengths: jax.Array) -> dict:
    """The stacked caches with each non-shared layer's new [k | v] row (from the kernel's outputs) written at
    position lengths[b] (ring slot lengths[b] % R)."""
    out = dict(caches)
    batch = jnp.arange(lengths.shape[0])
    for kind, name, rows in (("s", "ring", rows_s), ("f", "full", rows_f)):
        layers = config.kv_layers(kind)
        if not layers:
            continue
        cache_k, cache_v = caches[f"{name}_k"], caches[f"{name}_v"]
        dh = cache_k.shape[3]
        position = lengths % cache_k.shape[2] if kind == "s" else lengths
        new = rows[jnp.array(layers)]  # (n, B, 2 * dh)
        slots = jnp.arange(len(layers))[:, None]
        out[f"{name}_k"] = cache_k.at[slots, batch[None, :], position[None, :]].set(new[..., :dh], mode="drop")
        out[f"{name}_v"] = cache_v.at[slots, batch[None, :], position[None, :]].set(new[..., dh:], mode="drop")
    return out


def mega_step(params, tokens, caches, lengths, mega_tiles: MegaTiles | None = None):
    """One greedy decode step with the megakernel: the next tokens (B,), the stacked caches with this step's
    keys and values written, lengths + 1. Its runtime masks read `lengths` and the per-layer tables: under
    jit it tunes on random index data, so call it eagerly once to tune it on real ones."""
    inputs = kernel_inputs(params, tokens, caches, lengths)
    out = gemma4_megakernel(*inputs) if mega_tiles is None else gemma4_megakernel(*inputs, mega_tiles)
    tokens, rows_s, rows_f = out
    return tokens[:, 0], write_caches(params["config"], caches, rows_s, rows_f, lengths), lengths + 1


def mega_loop(params, tokens, caches, lengths, steps: int, mega_tiles: MegaTiles | None = None):
    """`steps` greedy megakernel decode steps in one `lax.fori_loop`: the tokens (B, steps), caches, lengths."""

    def body(t, carry):
        token, caches, lengths, out = carry
        token, caches, lengths = mega_step(params, token, caches, lengths, mega_tiles=mega_tiles)
        return token, caches, lengths, jax.lax.dynamic_update_slice(out, token[:, None], (0, t))

    out = jnp.zeros((tokens.shape[0], steps), I32)
    _, caches, lengths, out = jax.lax.fori_loop(0, steps, body, (tokens, caches, lengths, out))
    return out, caches, lengths


if __name__ == "__main__":
    from gemma4 import TINY, decode_step, init_params, prefill

    params = init_params(TINY)
    prompts = jax.random.randint(jax.random.key(1), (2, 192), 0, TINY.vocab)
    lengths = jnp.array([192, 150], I32)
    logits, caches, lens = jax.jit(prefill, static_argnums=3)(params, prompts, lengths, 256)
    first = jnp.argmax(logits, axis=-1).astype(I32)
    stacked = stack_caches(TINY, caches)
    inputs = kernel_inputs(params, first, stacked, lens)
    lowered = gemma4_megakernel.lower(*inputs)
    print(lowered.source[:3000])
    print(f"at most {lowered.steps} steps, {lowered.vmem_bytes / 2**20:.1f} MiB of VMEM")
    ours, _, _ = mega_step(params, first, stacked, lens)
    theirs, _, _ = jax.jit(decode_step)(params, first, caches, lens)
    print("megakernel tokens", ours.tolist(), "per-layer kernels", theirs.tolist())
