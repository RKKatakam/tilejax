"""Ragged paged attention (vLLM's attention over a paged KV cache, decode and prefill mixed in one batch),
as autokernel kernels. WORK IN PROGRESS: the inputs and the reference.

The ABI is vLLM TorchTPU's RPA v3 (`ragged_paged_attention`): every sequence's new tokens concatenated,
q (T, H, D), k and v (T, Hkv, D); a paged cache (pages, page_size, Hkv, 2, D) holding each head's key and
value side by side; kv_lens (S,) including the new tokens; page_indices (S * pages_per_seq,), sequence s's
pages at s * pages_per_seq; cu_q_lens (S + 1,); distribution (3,): sequences [0, d0) decode (one token),
[d0, d1) prefill chunks, [d1, d2) mixed. The step writes the new tokens' keys and values into the cache at
positions [kv_len - q_len, kv_len) and attends causally over every sequence's cache.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128  # Llama 3 8B / Qwen3 8B
PAGE_SIZE = 128
MAX_LEN = 8192
LANES = 128
MASK_VALUE = -1e30  # finite: exp(old max - new max) never computes -inf + inf


# --------------------------------------------------------------------------- the step's new keys and values


@autokernel.kernel
def rpa_write(kv_cache, keys, values, page_indices, write_page, write_rows, write_first, write_stop, n_writes):
    """The cache (pages, page_size, Hkv, 2, D) updated in place with the step's new keys and values (T, Hkv,
    D), a tile per run of a sequence's new tokens that lands in one page: the tokens from write_rows[w] go
    to positions write_first[w] to write_stop[w] of page page_indices[write_page[w]]."""
    ps, dtype = kv_cache.shape[1], kv_cache.dtype
    out = kv_cache
    for w in range(n_writes[0]):  # the runs, counted as the kernel runs
        k = keys[autokernel.rows(write_rows[w], write_rows[w] + write_stop[w] - write_first[w], ps), :, :]
        v = values[autokernel.rows(write_rows[w], write_rows[w] + write_stop[w] - write_first[w], ps), :, :]
        words = key_value_words(k, v)
        out = out.at[page_indices[write_page[w]], autokernel.rows(write_first[w], write_stop[w], ps), :, :, :].set(
            autokernel.unpairs(words, dtype)
        )
    return out


def writes(kv_lens, page_indices, cu_q_lens, distribution, page_size, max_tokens):
    """`rpa_write`'s runs, in plain JAX (small dense ops): sequence s's new tokens, at positions
    [kv_len - q_len, kv_len), cut at page boundaries. Arrays of a run count's bound, cdiv(T, page_size) +
    sequences."""
    S = kv_lens.shape[0]
    pps = page_indices.shape[0] // S
    seqs = jnp.arange(S, dtype=jnp.int32)
    q_len = jnp.where(seqs < distribution[2], cu_q_lens[1:] - cu_q_lens[:-1], 0)
    first = kv_lens - q_len
    runs = jnp.where(q_len > 0, (kv_lens - 1) // page_size - first // page_size + 1, 0)
    ends = jnp.cumsum(runs)
    w = jnp.arange(-(-max_tokens // page_size) + S, dtype=jnp.int32)
    seq = jnp.minimum(jnp.sum(ends[None, :] <= w[:, None], axis=1, dtype=jnp.int32), S - 1)
    page = first[seq] // page_size + w - (ends[seq] - runs[seq])
    start = jnp.maximum(first[seq], page * page_size)
    return {
        "write_page": seq * pps + page,
        "write_rows": cu_q_lens[seq] + start - first[seq],
        "write_first": start - page * page_size,
        "write_stop": jnp.minimum(kv_lens[seq], (page + 1) * page_size) - page * page_size,
        "n_writes": ends[-1:],
    }


def write_new_tokens(kv_cache, keys, values, kv_lens, page_indices, cu_q_lens, distribution):
    """The cache with the step's new tokens' keys and values at their positions [kv_len - q_len, kv_len),
    in place."""
    runs = writes(kv_lens, page_indices, cu_q_lens, distribution, kv_cache.shape[1], keys.shape[0])
    return rpa_write(kv_cache, keys, values, page_indices, *runs.values())


# --------------------------------------------------------------------------- decode: one new token per sequence


def key_value_words(k, v):
    """Keys and values (..., D) bf16 as the cache's (key | value << 16) words (..., D)."""
    bits = lambda x: jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)  # noqa: E731
    return (bits(k) >> 16) | (bits(v) & jnp.uint32(0xFFFF0000))


def keys_values(words):
    """A page's (key | value << 16) words (positions, Hkv, D) as keys and values (positions * Hkv, D) bf16,
    row t * Hkv + g position t's kv head g. (A bf16 is the high half of the f32 it widens to.)"""
    w = words.reshape(-1, words.shape[-1])
    k = jax.lax.bitcast_convert_type(w << 16, jnp.float32)
    v = jax.lax.bitcast_convert_type(w & jnp.uint32(0xFFFF0000), jnp.float32)
    return k.astype(jnp.bfloat16), v.astype(jnp.bfloat16)


def own_heads(heads, kv_heads, positions, start, length):
    """Which (query head, column) pairs attend, columns t * kv_heads + g being position start + t's kv head g:
    the query head's own kv head (h // (heads / kv_heads), as GQA groups heads), at positions before
    `length`."""
    shape = (heads, positions * kv_heads)
    col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    head = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    return ((col % kv_heads) == head // (heads // kv_heads)) & (start + col // kv_heads < length)


def online_softmax(s, v, row_max, row_sum, acc):
    """One online-softmax step: scores s (R, n) (masked: MASK_VALUE) against values v (n, D); the running
    max and sum, lane-wide (R, 128), and the accumulator (R, D)."""
    n, d = s.shape[1], v.shape[1]
    max_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - jnp.tile(max_new, (1, n // LANES)))
    alpha = jnp.exp(row_max - max_new)
    row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
    acc = jnp.tile(alpha, (1, d // LANES)) * acc + jnp.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
    return max_new, row_sum, acc


def make_decode(sm_scale: float) -> autokernel.Kernel:
    """The decode sequences' attention (RPA's decode case), a tile per sequence."""

    @autokernel.kernel
    def rpa_decode(q, kv_cache, kv_lens, page_indices, distribution):
        """q (T, H, D), the cache (pages, page_size, Hkv, 2, D) holding the new tokens already; sequence s <
        distribution[0] is token s. The output is q updated in place: a decode token's row is its attention
        output, every other row stays the query."""
        T, H, D = q.shape
        ps, Hkv = kv_cache.shape[1], kv_cache.shape[2]
        S = kv_lens.shape[0]
        pps = page_indices.shape[0] // S
        dtype = q.dtype
        out = q
        for s in range(min(S, T)):
            qs = q[s, :, :]
            row_max = jnp.full((H, LANES), MASK_VALUE, jnp.float32)
            row_sum = jnp.zeros((H, LANES), jnp.float32)
            acc = jnp.zeros((H, D), jnp.float32)
            for p in range((kv_lens[s] + ps - 1) // ps if s < distribution[0] else 0):  # its pages, read as it runs
                k, v = keys_values(autokernel.pairs(kv_cache[page_indices[s * pps + p], :, :, :, :]))
                scores = (qs @ k.T) * sm_scale  # every query head against every kv head's keys: (H, ps * Hkv)
                ok = own_heads(H, Hkv, ps, p * ps, kv_lens[s])
                row_max, row_sum, acc = online_softmax(jnp.where(ok, scores, MASK_VALUE), v, row_max, row_sum, acc)
            o = acc * jnp.tile(1.0 / jnp.maximum(row_sum, 1e-30), (1, D // LANES))
            out = out.at[s, :, :].set(jnp.where(s < distribution[0], o.astype(dtype), qs))
        return out

    return rpa_decode


# --------------------------------------------------------------------------- prefill and mixed: a block of queries


BLOCK_Q = 256  # query tokens per tile


@autokernel.tunable
class PrefillTiles:
    pages: int = autokernel.choice(1, 2, 4, 8, default=4)  # cache pages per step


def head_rows(qb, kv_heads):
    """A block of queries (BQ, H, D) as each kv head's rows (Hkv, BQ * G, D), row t * G + i its token t's
    query head g * G + i (G = H / Hkv, as GQA groups heads)."""
    bq, h, d = qb.shape
    g = h // kv_heads
    return jnp.stack([qb[:, i * g : (i + 1) * g, :].reshape(bq * g, d) for i in range(kv_heads)])


def token_heads(rows, heads):
    """`head_rows` undone: (Hkv, BQ * G, D) -> (BQ, H, D)."""
    kv_heads, n, d = rows.shape
    g = heads // kv_heads
    return jnp.concatenate([rows[i].reshape(n // g, g, d) for i in range(kv_heads)], axis=1)


def visible(heads, rows, cols, first_row, first_col, stop):
    """Which (row, column) pairs attend: row t * G + i (a query at position first_row + t) sees the column's
    position first_col + c if it is no later than its own and before `stop`."""
    g = heads
    shape = (1, rows, cols)
    pos = first_row + jax.lax.broadcasted_iota(jnp.int32, shape, 1) // g
    col = first_col + jax.lax.broadcasted_iota(jnp.int32, shape, 2)
    return (col <= pos) & (col < stop)


def online_softmax_heads(s, v, row_max, row_sum, acc):
    """`online_softmax` for every kv head at once: s (Hkv, R, n), v (Hkv, n, D), lane-wide stats (Hkv, R, 128)."""
    n, d = s.shape[-1], v.shape[-1]
    max_new = jnp.maximum(row_max, jnp.max(s, axis=-1, keepdims=True))
    p = jnp.exp(s - jnp.tile(max_new, (1, 1, n // LANES)))
    alpha = jnp.exp(row_max - max_new)
    row_sum = alpha * row_sum + jnp.sum(p, axis=-1, keepdims=True)
    pv = jax.lax.dot_general(p.astype(v.dtype), v, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
    return max_new, row_sum, jnp.tile(alpha, (1, 1, d // LANES)) * acc + pv


def make_prefill(sm_scale: float) -> autokernel.Kernel:
    """The attention of the sequences with several new tokens (RPA's prefill and mixed cases), a tile per
    block of a sequence's queries."""

    @autokernel.kernel
    def rpa_prefill(
        q, kv_cache, page_indices, tile_page, tile_rows, tile_pos, tile_stop, n_tiles, tiles=PrefillTiles()
    ):
        """q (T, H, D), the cache (pages, page_size, Hkv, 2, D) holding the new tokens already. Tile k <
        n_tiles[0] is the queries tile_rows[k] to tile_rows[k + 1] (at most BLOCK_Q rows) of a sequence whose
        pages start at page_indices[tile_page[k]]: its first query sits at position tile_pos[k], and they
        attend to the positions before tile_stop[k]. The output is q updated in place: those rows become
        their attention outputs."""
        _, H, D = q.shape
        ps, Hkv = kv_cache.shape[1], kv_cache.shape[2]
        G, NP, BQ = H // Hkv, tiles.pages, BLOCK_Q
        dtype = q.dtype
        out = q
        for k in range(n_tiles[0]):  # the tiles, counted as the kernel runs
            qh = head_rows(q[autokernel.rows(tile_rows[k], tile_rows[k + 1], BQ), :, :], Hkv)
            row_max = jnp.full((Hkv, BQ * G, LANES), MASK_VALUE, jnp.float32)
            row_sum = jnp.zeros((Hkv, BQ * G, LANES), jnp.float32)
            acc = jnp.zeros((Hkv, BQ * G, D), jnp.float32)
            for j in range(0, tile_stop[k], NP * ps):  # NP pages a step, up to the tile's last query
                # The step's NP pages, as words (NP, ps, Hkv, D).
                words = autokernel.pairs(
                    kv_cache[page_indices[tile_page[k] + j // ps : tile_page[k] + j // ps + NP], :, :, :, :]
                )
                kv = [keys_values(words[:, :, g, :]) for g in range(Hkv)]
                k_heads = jnp.stack([k_ for k_, _ in kv])  # (Hkv, NP * ps, D)
                v_heads = jnp.stack([v_ for _, v_ in kv])
                s = jax.lax.dot_general(qh, k_heads, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                ok = visible(G, BQ * G, NP * ps, tile_pos[k], j, tile_stop[k])
                row_max, row_sum, acc = online_softmax_heads(
                    jnp.where(ok, s * sm_scale, MASK_VALUE), v_heads, row_max, row_sum, acc
                )
            o = acc * jnp.tile(1.0 / jnp.maximum(row_sum, 1e-30), (1, 1, D // LANES))
            out = out.at[autokernel.rows(tile_rows[k], tile_rows[k + 1], BQ), :, :].set(token_heads(o, H).astype(dtype))
        return out

    return rpa_prefill


@functools.cache
def prefill_kernel(sm_scale: float) -> autokernel.Kernel:
    return make_prefill(sm_scale)


def prefill_tiles(kv_lens, page_indices, cu_q_lens, distribution, block_q=BLOCK_Q, max_tokens=None):
    """The prefill kernel's tiles, in plain JAX (small dense ops; a TPU gathers and scatters slowly): a tile
    per block of block_q queries of each sequence in [distribution[0], distribution[2]). Arrays of a tile
    count's bound, cdiv(T, block_q) + sequences; their rows start at tile_rows (one more)."""
    S = kv_lens.shape[0]
    pps = page_indices.shape[0] // S
    seqs = jnp.arange(S, dtype=jnp.int32)
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    mixed = (seqs >= distribution[0]) & (seqs < distribution[2])
    blocks = jnp.where(mixed, (q_len + block_q - 1) // block_q, 0)
    ends = jnp.cumsum(blocks)
    n = -(-max_tokens // block_q) + S
    k = jnp.arange(n, dtype=jnp.int32)
    seq = jnp.minimum(jnp.sum(ends[None, :] <= k[:, None], axis=1, dtype=jnp.int32), S - 1)
    b = k - (ends[seq] - blocks[seq])
    rows = cu_q_lens[seq] + b * block_q
    stop = jnp.minimum(rows + block_q, cu_q_lens[seq + 1])
    pos = kv_lens[seq] - q_len[seq] + b * block_q
    return {
        "tile_page": seq * pps,
        "tile_rows": jnp.concatenate([rows[:1], stop]),  # (a tile's rows end where the next one's start)
        "tile_pos": pos,
        "tile_stop": pos + (stop - rows),
        "n_tiles": ends[-1:],
    }


@functools.cache
def decode_kernel(sm_scale: float) -> autokernel.Kernel:
    return make_decode(sm_scale)


def tune(inputs: dict[str, Any], sm_scale: float) -> Any:
    """Tune the kernels on these inputs (under jax.jit they would tune on random ones); the tuning."""
    q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution = args_of(inputs)
    cache = write_new_tokens(cache, k, v, kv_lens, page_indices, cu_q_lens, distribution)
    decode = decode_kernel(float(sm_scale)).tuning(q, cache, kv_lens, page_indices, distribution)
    tiles = prefill_tiles(kv_lens, page_indices, cu_q_lens, distribution, max_tokens=q.shape[0])
    prefill = prefill_kernel(float(sm_scale)).tuning(q, cache, page_indices, *tiles.values())
    return f"decode: {decode}\nprefill: {prefill}"


@functools.partial(jax.jit, static_argnames=("sm_scale",), donate_argnames=("queries", "kv_cache"))
def autokernel_rpa(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, *, sm_scale=1.0):
    """RPA v3's `ragged_paged_attention` (default options): (out (T, H, D), the updated cache). The decode
    sequences' attention, then the others' (both kernels update the queries in place)."""
    kv_cache = write_new_tokens(kv_cache, keys, values, kv_lens, page_indices, cu_q_lens, distribution)
    out = decode_kernel(float(sm_scale))(queries, kv_cache, kv_lens, page_indices, distribution)
    tiles = prefill_tiles(kv_lens, page_indices, cu_q_lens, distribution, max_tokens=queries.shape[0])
    out = prefill_kernel(float(sm_scale))(out, kv_cache, page_indices, *tiles.values())
    return out, kv_cache


def make_inputs(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    heads: int = HEADS,
    kv_heads: int = KV_HEADS,
    head_dim: int = HEAD_DIM,
    page_size: int = PAGE_SIZE,
    max_len: int = MAX_LEN,
    max_tokens: int | None = None,
    max_seqs: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """One step's inputs in RPA v3's ABI: sequences with these new-token counts and cache lengths (new
    tokens included), decode sequences (q_len 1) first, then the rest (as vLLM orders them). Each sequence's
    pages are distinct random pages of the pool; tokens and sequences are padded to `max_tokens` and
    `max_seqs` (padding sequences have no tokens and an empty cache)."""
    rng = np.random.default_rng(seed)
    S, T = len(q_lens), sum(q_lens)
    assert all(q >= 1 for q in q_lens) and all(k >= q for q, k in zip(q_lens, kv_lens, strict=True))
    decodes = sum(1 for q in q_lens if q == 1)
    assert all(q == 1 for q in q_lens[:decodes]), "decode sequences first"
    max_tokens = T if max_tokens is None else max_tokens
    max_seqs = S if max_seqs is None else max_seqs
    pps = max_len // page_size
    used = [-(-k // page_size) for k in kv_lens]
    pages = sum(used) + 1
    pool = 1 + rng.permutation(pages - 1)  # page 0 stays unused
    table = rng.integers(0, pages, (max_seqs, pps), dtype=np.int32)  # past a sequence's pages: anything
    start = 0
    for s, n in enumerate(used):
        table[s, :n] = pool[start : start + n]
        start += n

    def normal(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.standard_normal(shape), dtype)

    cu = np.zeros(max_seqs + 1, np.int32)
    cu[1 : S + 1] = np.cumsum(q_lens)
    cu[S + 1 :] = T
    lens = np.zeros(max_seqs, np.int32)
    lens[:S] = kv_lens
    return {
        "queries": normal((max_tokens, heads, head_dim)),
        "keys": normal((max_tokens, kv_heads, head_dim)),
        "values": normal((max_tokens, kv_heads, head_dim)),
        "kv_cache": normal((pages, page_size, kv_heads, 2, head_dim)),
        "kv_lens": jnp.asarray(lens),
        "page_indices": jnp.asarray(table.reshape(-1)),
        "cu_q_lens": jnp.asarray(cu),
        "distribution": jnp.asarray([decodes, decodes, S], jnp.int32),
    }


ARGS = ("queries", "keys", "values", "kv_cache", "kv_lens", "page_indices", "cu_q_lens", "distribution")


def args_of(inputs: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(inputs[k] for k in ARGS)


def reference(inputs: dict[str, Any], sm_scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The step in float64 NumPy: (the outputs of the real tokens (T, H, D), the updated cache)."""
    f = lambda x: np.asarray(x, np.float64)  # noqa: E731
    q, k, v, cache = f(inputs["queries"]), f(inputs["keys"]), f(inputs["values"]), f(inputs["kv_cache"]).copy()
    lens, cu = np.asarray(inputs["kv_lens"]), np.asarray(inputs["cu_q_lens"])
    S = int(np.asarray(inputs["distribution"])[2])
    _, H, D = q.shape
    Hkv, ps = k.shape[1], cache.shape[1]
    pps = inputs["page_indices"].shape[0] // lens.shape[0]
    table = np.asarray(inputs["page_indices"]).reshape(-1, pps)
    scale = D**-0.5 if sm_scale is None else sm_scale
    out = np.zeros((int(cu[S]), H, D))
    for s in range(S):
        q0, q1, n = int(cu[s]), int(cu[s + 1]), int(lens[s])
        positions = np.arange(n - (q1 - q0), n)
        pages, rows = table[s, positions // ps], positions % ps
        cache[pages, rows, :, 0, :] = k[q0:q1]
        cache[pages, rows, :, 1, :] = v[q0:q1]
        everything = np.arange(n)
        ks = cache[table[s, everything // ps], everything % ps, :, 0, :]  # (n, Hkv, D)
        vs = cache[table[s, everything // ps], everything % ps, :, 1, :]
        g = H // Hkv
        scores = np.einsum("qhd,khd->hqk", q[q0:q1], np.repeat(ks, g, axis=1)) * scale
        scores = np.where(everything[None, None, :] <= positions[None, :, None], scores, -np.inf)
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[q0:q1] = np.einsum("hqk,khd->qhd", p, np.repeat(vs, g, axis=1))
    return out, cache
