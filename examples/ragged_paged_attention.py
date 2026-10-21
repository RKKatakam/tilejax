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


def write_new_tokens(kv_cache, keys, values, kv_lens, page_indices, cu_q_lens, distribution):
    """The cache with the step's new tokens' keys and values at their positions [kv_len - q_len, kv_len),
    in place (one XLA scatter; padding tokens are dropped)."""
    T = keys.shape[0]
    S, ps = kv_lens.shape[0], kv_cache.shape[1]
    pps = page_indices.shape[0] // S
    t = jnp.arange(T, dtype=jnp.int32)
    seq = jnp.clip(jnp.searchsorted(cu_q_lens, t, side="right").astype(jnp.int32) - 1, 0, S - 1)
    real = t < cu_q_lens[distribution[2]]
    pos = kv_lens[seq] - (cu_q_lens[seq + 1] - cu_q_lens[seq]) + (t - cu_q_lens[seq])
    page = jnp.where(real, page_indices[seq * pps + pos // ps], kv_cache.shape[0])  # (out of bounds: dropped)
    return kv_cache.at[page, pos % ps].set(jnp.stack([keys, values], axis=2).astype(kv_cache.dtype))


# --------------------------------------------------------------------------- decode: one new token per sequence


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
            for p in range(pps):
                if s >= distribution[0] or p * ps >= kv_lens[s]:
                    continue
                k, v = keys_values(autokernel.pairs(kv_cache[page_indices[s * pps + p], :, :, :, :]))
                scores = (qs @ k.T) * sm_scale  # every query head against every kv head's keys: (H, ps * Hkv)
                ok = own_heads(H, Hkv, ps, p * ps, kv_lens[s])
                row_max, row_sum, acc = online_softmax(jnp.where(ok, scores, MASK_VALUE), v, row_max, row_sum, acc)
            o = acc * jnp.tile(1.0 / jnp.maximum(row_sum, 1e-30), (1, D // LANES))
            out = out.at[s, :, :].set(jnp.where(s < distribution[0], o.astype(dtype), qs))
        return out

    return rpa_decode


@functools.cache
def decode_kernel(sm_scale: float) -> autokernel.Kernel:
    return make_decode(sm_scale)


def tune(inputs: dict[str, Any], sm_scale: float) -> Any:
    """Tune the kernels on these inputs (under jax.jit they would tune on random ones); the tuning."""
    args = args_of(inputs)
    cache = write_new_tokens(args[3], args[1], args[2], *args[4:])
    return decode_kernel(float(sm_scale)).tuning(args[0], cache, *args[4:6], args[7])


@functools.partial(jax.jit, static_argnames=("sm_scale",), donate_argnames=("queries", "kv_cache"))
def autokernel_rpa(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, *, sm_scale=1.0):
    """RPA v3's `ragged_paged_attention` (default options): (out (T, H, D), the updated cache). Decode-only
    batches for now."""
    kv_cache = write_new_tokens(kv_cache, keys, values, kv_lens, page_indices, cu_q_lens, distribution)
    out = decode_kernel(float(sm_scale))(queries, kv_cache, kv_lens, page_indices, distribution)
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
