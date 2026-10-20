"""KV caches for decoding, in plain JAX: a contiguous layout and a paged (vLLM-style) one.

Contiguous: per layer, a key and a value cache of shape (B, S_max, Hkv * Dh) (a kv head is a block of Dh
columns, as everywhere in these examples, so a cache block is whole rows: one contiguous DMA), and
`lengths` (B,) int32, the number of valid positions of each sequence. Appending T new tokens writes
positions lengths[b] .. lengths[b] + T - 1 of each sequence (`append`): a scatter of whole rows, which XLA
does in place when the cache is a donated argument of a jitted step.

Paged: the positions of all sequences live in a pool of pages (num_pages, page_size, Hkv * Dh), and
sequence b's i-th page is `block_tables[b, i]` (B, pages_per_seq) int32: position p of sequence b is row
p % page_size of page block_tables[b, p // page_size] (`append_paged`). Sequences grow without reserving
S_max positions each, and pages can be shared; the decode kernels read the pages through runtime windows,
`k_pages[block_tables[b, i], :, :]` (decode.py).

JAX's own TPU decode kernels want other layouts; `jax_paged_layout` and `ragged_kv_pages` convert a paged
cache to them (for comparisons and benchmarks).

The MLA latent cache (decode.py) is two contiguous caches of this kind: the 512-wide latents and the
64-wide RoPE keys zero-padded to 128 lanes (`pad_lanes`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

LANES = 128


def empty_cache(batch: int, max_len: int, width: int, dtype: Any = jnp.bfloat16) -> jax.Array:
    """A contiguous cache (batch, max_len, width) of zeros."""
    return jnp.zeros((batch, max_len, width), dtype)


def append(cache: jax.Array, lengths: jax.Array, new: jax.Array) -> jax.Array:
    """`cache` (B, S_max, W) with each sequence's new tokens `new` (B, T, W) written at its positions
    lengths[b] + t (positions past S_max are dropped). The caller adds T to `lengths`."""
    b, t = new.shape[:2]
    rows = jnp.arange(b)[:, None]
    positions = lengths[:, None] + jnp.arange(t)[None, :]
    return cache.at[rows, positions].set(new.astype(cache.dtype), mode="drop")


def append_paged(pages: jax.Array, block_tables: jax.Array, lengths: jax.Array, new: jax.Array) -> jax.Array:
    """`pages` (num_pages, P, W) with each sequence's new tokens `new` (B, T, W) written at its positions
    lengths[b] + t, in the pages its block table lists."""
    page, row = _page_rows(block_tables, lengths, new.shape[1], pages.shape[1])
    return pages.at[page, row].set(new.astype(pages.dtype), mode="drop")


def page_tables(batch: int, max_len: int, page_size: int, seed: int | None = 0) -> np.ndarray:
    """Block tables (batch, max_len / page_size) for a pool of exactly that many pages: each sequence's
    pages, shuffled over the pool (as an allocator that has run a while leaves them), or in order if
    `seed` is None."""
    pages = batch * (max_len // page_size)
    order = np.arange(pages) if seed is None else np.random.default_rng(seed).permutation(pages)
    return order.reshape(batch, max_len // page_size).astype(np.int32)


def to_pages(cache: jax.Array, block_tables: jax.Array, num_pages: int | None = None) -> jax.Array:
    """A contiguous cache (B, S_max, W) laid out in pages (num_pages, P, W) by `block_tables` (B, S_max / P)."""
    b, s, w = cache.shape
    per_seq = block_tables.shape[1]
    page_size = s // per_seq
    num_pages = b * per_seq if num_pages is None else num_pages
    pages = jnp.zeros((num_pages, page_size, w), cache.dtype)
    return pages.at[block_tables.reshape(-1)].set(cache.reshape(b * per_seq, page_size, w))


def pad_lanes(x: jax.Array, width: int = LANES) -> jax.Array:
    """`x` (..., n) zero-padded to (..., width) lanes."""
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, width - x.shape[-1])])


def jax_paged_layout(pages: jax.Array, kv_heads: int) -> jax.Array:
    """Pages (num_pages, P, Hkv * Dh) as JAX's paged_attention wants them: (Hkv, num_pages, P, Dh)."""
    n, p, w = pages.shape
    return pages.reshape(n, p, kv_heads, w // kv_heads).transpose(2, 0, 1, 3)


def ragged_kv_pages(k_pages: jax.Array, v_pages: jax.Array, kv_heads: int) -> jax.Array:
    """Key and value pages (num_pages, P, Hkv * Dh) as JAX's ragged_paged_attention wants them: one array
    (num_pages, P, 2 * Hkv, Dh), each kv head's keys then its values."""
    n, p, w = k_pages.shape
    k = k_pages.reshape(n, p, kv_heads, 1, w // kv_heads)
    v = v_pages.reshape(n, p, kv_heads, 1, w // kv_heads)
    return jnp.concatenate([k, v], axis=3).reshape(n, p, 2 * kv_heads, w // kv_heads)


def _page_rows(block_tables: jax.Array, lengths: jax.Array, new_tokens: int, page_size: int):
    positions = lengths[:, None] + jnp.arange(new_tokens)[None, :]
    page = jnp.take_along_axis(block_tables, positions // page_size, axis=1, mode="clip")
    return page, positions % page_size


def append_jax_paged(pages: jax.Array, block_tables: jax.Array, lengths: jax.Array, new: jax.Array) -> jax.Array:
    """`append_paged` for JAX paged_attention's layout: pages (Hkv, num_pages, P, Dh), new (B, T, Hkv, Dh)."""
    page, row = _page_rows(block_tables, lengths, new.shape[1], pages.shape[2])
    return pages.at[:, page, row, :].set(new.transpose(2, 0, 1, 3).astype(pages.dtype), mode="drop")


def append_ragged(kv_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array, k: jax.Array, v: jax.Array):
    """`append_paged` for JAX ragged_paged_attention's layout: kv_pages (num_pages, P, 2 * Hkv, Dh), the new
    keys and values (B, T, Hkv, Dh)."""
    page, row = _page_rows(block_tables, lengths, k.shape[1], kv_pages.shape[1])
    kv = jnp.stack([k, v], axis=3).reshape((*k.shape[:2], 2 * k.shape[2], k.shape[3]))
    return kv_pages.at[page, row].set(kv.astype(kv_pages.dtype), mode="drop")
