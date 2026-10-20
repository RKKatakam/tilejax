"""Decoding with a small GQA transformer (Qwen / Llama style): autokernel kernels composed in plain JAX.

    decode_step(tokens, caches, lengths, params) -> (logits, caches, lengths)

    x = embed[tokens]
    per layer:  q, k, v = rmsnorm(x) @ W_qkv;   q, k = rope(q, k at positions lengths[b] + t)
                append k, v to the layer's cache;  a = attention(q, cache[:lengths[b] + T])
                x = x + a @ W_o;   x = x + W_down(silu(h @ W_gate) * (h @ W_up)), h = rmsnorm(x)
    logits = rmsnorm(x) @ W_vocab

T new tokens per sequence (1 when generating; a few to verify speculative drafts), at ragged positions. A
step's activations are only B * T rows, so a step is bound by reading the weights and the KV cache: each
layer is three kernels around a cache append in XLA (a scatter of whole rows, in place on a donated cache):

1. `rmsnorm_matmul` (examples/rmsnorm_matmul.py): the attention RMSNorm fused with the QKV projection. q's
   columns come pre-scaled by 1/sqrt(head_dim) (folded into W_qkv; RoPE is linear, so it commutes). RoPE on
   the one token's q and k is plain JAX (elementwise on a few KiB).
2. `gqa_decode` / `paged_gqa_decode` (decode.py): attention over the valid part of each sequence's cache.
3. `decode_mlp`: the rest of the layer in one kernel of one tile (all B * T rows) and two phases, streaming
   the weights: the output projection over blocks of W_o's rows, then the residual and the RMSNorm, then
   the SwiGLU feed-forward layer over blocks of hidden units, and the second residual.

The final norm and the logits are `rmsnorm_matmul` again. `prefill` fills the caches from prompts in plain
JAX (dense causal attention), and `generate` runs greedy decoding: prefill, then jitted decode steps with
the caches donated. `baseline_step` is the same decode step in plain JAX with XLA's attention over the
whole cache, or JAX's Pallas paged_attention / ragged_paged_attention kernels on their own page layouts.

Head layout: q, k and v are (B, T, heads, 128); query head h attends with kv head h // (H / Hkv); caches
are (B, S_max, Hkv * 128) or pages (num_pages, page_size, Hkv * 128) (kv_cache.py).

Run with `uv run python examples/decode_model.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from decode import (
    gqa_decode,
    jax_paged_attention,
    jax_ragged_paged_attention,
    paged_gqa_decode,
    xla_gqa_decode,
)
from kv_cache import append, append_jax_paged, append_paged, append_ragged, empty_cache, page_tables, to_pages
from rmsnorm_matmul import rmsnorm_matmul

import autokernel

EPS = 1e-6  # as rmsnorm_matmul's
HEAD_DIM = 128
ROPE_THETA = 1e6  # Qwen's


@dataclass(frozen=True)
class Config:
    layers: int
    d_model: int
    heads: int
    kv_heads: int
    ffn: int
    vocab: int


@autokernel.tunable
class MlpTiles:
    """Rows of W_o per output-projection step, hidden units per feed-forward step."""

    block_a: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=512)


@autokernel.kernel
def decode_mlp(x, attn, wo, gamma, w_gate, w_up, w_down, tiles=MlpTiles()):
    """The layer after attention, on (rows, d_model): output projection, residual, RMSNorm, SwiGLU
    feed-forward layer, residual. One tile of every row: each weight is read once."""
    M, D = x.shape
    A, F = attn.shape[1], w_up.shape[1]
    ba, bf = min(tiles.block_a, A), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, M):
        proj = jnp.zeros((M, D), jnp.float32)
        for a in range(0, A, ba):  # phase 1: the output projection, ba rows of W_o per step
            proj += attn[i : i + M, a : a + ba] @ wo[a : a + ba, :]
        resid = x[i : i + M, :].astype(jnp.float32) + proj
        normed = resid * jax.lax.rsqrt(jnp.mean(resid * resid, axis=1, keepdims=True) + EPS) * gamma[0:1, :]
        normed = normed.astype(dtype)
        y = jnp.zeros((M, D), jnp.float32)
        for f in range(0, F, bf):  # phase 2: SwiGLU, bf hidden units per step
            hidden = jax.nn.silu(normed @ w_gate[:, f : f + bf]) * (normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        out = out.at[i : i + M, :].set((resid + y).astype(dtype))
    return out


def rope(x, positions):
    """Rotary position embedding (rotate-half form) of x (B, T, heads, Dh) at positions (B, T)."""
    dh = x.shape[-1]
    inv_freq = ROPE_THETA ** (-jnp.arange(0, dh, 2, dtype=jnp.float32) / dh)
    angles = positions[..., None].astype(jnp.float32) * inv_freq
    angles = jnp.concatenate([angles, angles], axis=-1)[:, :, None, :]
    xf = x.astype(jnp.float32)
    rotated = jnp.concatenate([-xf[..., dh // 2 :], xf[..., : dh // 2]], axis=-1)
    return (xf * jnp.cos(angles) + rotated * jnp.sin(angles)).astype(x.dtype)


def _qkv(qkv, heads, positions):
    """q, k, v (B, T, n, Dh) from a QKV projection (B, T, (H + 2 Hkv) * Dh), RoPE applied to q and k."""
    b, t, _ = qkv.shape
    qkv = qkv.reshape(b, t, -1, HEAD_DIM)
    kv_heads = (qkv.shape[2] - heads) // 2
    q, k, v = jnp.split(qkv, [heads, heads + kv_heads], axis=2)
    return rope(q, positions), rope(k, positions), v


def decode_step(tokens, caches, lengths, params, block_tables=None):
    """One decode step with autokernel kernels: logits (B, T, vocab) for the new tokens (B, T), the caches
    with their keys and values appended, and lengths + T. `caches` holds a {"k", "v"} per layer:
    contiguous (B, S_max, Hkv * Dh), or pages (num_pages, P, Hkv * Dh) if `block_tables` is given.

    The attention kernels' masks read `lengths`: under jit they tune on random lengths, so run a first step
    eagerly (`generate` does) to tune them on real ones."""
    b, t = tokens.shape
    x = params["embed"][tokens].reshape(b * t, -1)
    positions = lengths[:, None] + jnp.arange(t)[None, :]
    new_lengths = lengths + t
    new_caches = []
    for layer, cache in zip(params["layers"], caches, strict=True):
        heads = layer["wo"].shape[0] // HEAD_DIM
        qkv = rmsnorm_matmul(x, layer["attn_norm"], layer["w_qkv"]).reshape(b, t, -1)
        q, k, v = _qkv(qkv, heads, positions)
        k, v = k.reshape(b, t, -1), v.reshape(b, t, -1)
        if block_tables is None:
            cache = {"k": append(cache["k"], lengths, k), "v": append(cache["v"], lengths, v)}
            attn = gqa_decode(q, cache["k"], cache["v"], new_lengths)
        else:
            k_pages = append_paged(cache["k"], block_tables, lengths, k)
            v_pages = append_paged(cache["v"], block_tables, lengths, v)
            cache = {"k": k_pages, "v": v_pages}
            attn = paged_gqa_decode(q, k_pages, v_pages, block_tables, new_lengths)
        new_caches.append(cache)
        mlp = (layer["wo"], layer["ffn_norm"], layer["w_gate"], layer["w_up"], layer["w_down"])
        x = decode_mlp(x, attn.reshape(b * t, -1), *mlp)
    logits = rmsnorm_matmul(x, params["final_norm"], params["w_vocab"])
    return logits.reshape(b, t, -1), new_caches, new_lengths


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights; q's columns of W_qkv are pre-scaled by 1/sqrt(head_dim)."""
    d, f, h, hkv = config.d_model, config.ffn, config.heads, config.kv_heads
    keys = iter(jax.random.split(jax.random.key(seed), 4 + 7 * config.layers))

    def normal(shape, scale):
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm_weight(n):
        return (1 + 0.1 * jax.random.normal(next(keys), (1, n))).astype(dtype)

    layers = []
    for _ in range(config.layers):
        w_qkv = jax.random.normal(next(keys), (d, (h + 2 * hkv) * HEAD_DIM)) / d**0.5
        w_qkv = w_qkv.at[:, : h * HEAD_DIM].multiply(HEAD_DIM**-0.5).astype(dtype)
        layers.append(
            {
                "attn_norm": norm_weight(d),
                "w_qkv": w_qkv,
                "wo": normal((h * HEAD_DIM, d), (h * HEAD_DIM) ** -0.5),
                "ffn_norm": norm_weight(d),
                "w_gate": normal((d, f), d**-0.5),
                "w_up": normal((d, f), d**-0.5),
                "w_down": normal((f, d), f**-0.5),
            }
        )
    return {
        "embed": normal((config.vocab, d), 1.0),
        "layers": layers,
        "final_norm": norm_weight(d),
        "w_vocab": normal((d, config.vocab), d**-0.5),
    }


# --------------------------------------------------------------------------- plain JAX: prefill, baselines


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _rmsnorm(x, gamma):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def _mlp(x, attn, layer):
    """`decode_mlp` in plain JAX, on (..., d_model) rows."""
    resid = x.astype(jnp.float32) + _dot(attn, layer["wo"])
    normed = _rmsnorm(resid, layer["ffn_norm"]).astype(x.dtype)
    hidden = jax.nn.silu(_dot(normed, layer["w_gate"])) * _dot(normed, layer["w_up"])
    return (resid + _dot(hidden.astype(x.dtype), layer["w_down"])).astype(x.dtype)


def _project_qkv(x, layer):
    return _dot(_rmsnorm(x, layer["attn_norm"]).astype(x.dtype), layer["w_qkv"]).astype(x.dtype)


def _logits(x, params):
    return _dot(_rmsnorm(x, params["final_norm"]).astype(x.dtype), params["w_vocab"]).astype(x.dtype)


def empty_caches(config: Config, batch: int, max_len: int, page_size: int | None = None) -> list[dict]:
    """Zero caches for every layer: contiguous, or pages (batch * max_len / page_size of them) if
    `page_size` is given."""
    width = config.kv_heads * HEAD_DIM
    shape = (batch, max_len) if page_size is None else (batch * max_len // page_size, page_size)
    return [{"k": empty_cache(*shape, width), "v": empty_cache(*shape, width)} for _ in range(config.layers)]


def prefill(prompts, prompt_lengths, params, caches, block_tables=None):
    """Run padded prompts (B, S_p) (sequence b's first prompt_lengths[b] tokens) through the model in plain
    JAX, dense causal attention, writing their keys and values to the (empty) caches. Returns the logits
    at each prompt's last token (B, vocab), the caches and the lengths."""
    b, s = prompts.shape
    x = params["embed"][prompts]
    positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
    causal = jnp.tril(jnp.ones((s, s), bool))
    zeros = jnp.zeros(b, jnp.int32)
    new_caches = []
    for layer, cache in zip(params["layers"], caches, strict=True):
        heads = layer["wo"].shape[0] // HEAD_DIM
        q, k, v = _qkv(_project_qkv(x, layer), heads, positions)
        kv_heads = k.shape[2]
        qg = q.reshape(b, s, kv_heads, heads // kv_heads, HEAD_DIM)
        scores = jnp.einsum("bqgrd,bkgd->bgrqk", qg, k, preferred_element_type=jnp.float32)
        p = jax.nn.softmax(jnp.where(causal, scores, -1e30), axis=-1).astype(v.dtype)
        attn = jnp.einsum("bgrqk,bkgd->bqgrd", p, v, preferred_element_type=jnp.float32).astype(x.dtype)
        k, v = k.reshape(b, s, -1), v.reshape(b, s, -1)
        if block_tables is None:
            cache = {"k": append(cache["k"], zeros, k), "v": append(cache["v"], zeros, v)}
        else:
            cache = {name: append_paged(cache[name], block_tables, zeros, new) for name, new in (("k", k), ("v", v))}
        new_caches.append(cache)
        x = _mlp(x, attn.reshape(b, s, -1), layer)
    last = jnp.take_along_axis(x, (prompt_lengths - 1)[:, None, None], axis=1)[:, 0]
    return _logits(last, params), new_caches, prompt_lengths


def generate(prompts, prompt_lengths, params, config: Config, steps: int, max_len: int, page_size=None):
    """Greedy decoding: `steps` new tokens (B, steps) after each prompt, with autokernel's decode step
    (on a paged cache if `page_size` is given). The first decode step runs eagerly, so the attention
    kernels tune on the real cache lengths; the rest are jitted, the caches donated (updated in place)."""
    block_tables = None if page_size is None else jnp.asarray(page_tables(prompts.shape[0], max_len, page_size))
    caches = empty_caches(config, prompts.shape[0], max_len, page_size)
    logits, caches, lengths = jax.jit(prefill)(prompts, prompt_lengths, params, caches, block_tables)
    token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    out = [token]
    step = jax.jit(decode_step, donate_argnums=1)
    for i in range(steps - 1):
        run = decode_step if i == 0 else step
        logits, caches, lengths = run(token[:, None], caches, lengths, params, block_tables)
        token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
        out.append(token)
    return jnp.stack(out, axis=1)


def baseline_step(attention: str = "xla", pages_per_compute_block: int = 4, **block_sizes: Any):
    """`decode_step` in plain JAX, compiled by XLA, jitted with the caches donated, attending with XLA over
    the whole contiguous cache ("xla"), or JAX's Pallas paged_attention ("paged", T = 1; caches {"k", "v"}
    (Hkv, num_pages, P, Dh)) or ragged_paged_attention ("ragged"; caches {"kv"} (num_pages, P, 2 Hkv, Dh))."""

    def step(tokens, caches, lengths, params, block_tables=None):
        b, t = tokens.shape
        x = params["embed"][tokens]
        positions = lengths[:, None] + jnp.arange(t)[None, :]
        new_lengths = lengths + t
        new_caches = []
        for layer, cache in zip(params["layers"], caches, strict=True):
            heads = layer["wo"].shape[0] // HEAD_DIM
            q, k, v = _qkv(_project_qkv(x, layer), heads, positions)
            if attention == "xla":
                k_cache = append(cache["k"], lengths, k.reshape(b, t, -1))
                v_cache = append(cache["v"], lengths, v.reshape(b, t, -1))
                cache = {"k": k_cache, "v": v_cache}
                attn = xla_gqa_decode(q, cache["k"], cache["v"], new_lengths)
            elif attention == "paged":
                k_pages = append_jax_paged(cache["k"], block_tables, lengths, k)
                v_pages = append_jax_paged(cache["v"], block_tables, lengths, v)
                cache = {"k": k_pages, "v": v_pages}
                attn = jax_paged_attention(q, k_pages, v_pages, block_tables, new_lengths, pages_per_compute_block)
            else:
                cache = {"kv": append_ragged(cache["kv"], block_tables, lengths, k, v)}
                attn = jax_ragged_paged_attention(q, cache["kv"], block_tables, new_lengths, **block_sizes)
            new_caches.append(cache)
            x = _mlp(x, attn.reshape(b, t, -1), layer)
        return _logits(x, params), new_caches, new_lengths

    return jax.jit(step, donate_argnums=1)


def random_caches(config: Config, lengths: np.ndarray, max_len: int, seed: int = 0) -> list[dict]:
    """Contiguous caches (B, max_len, Hkv * Dh) of random keys and values (zero past each length): a
    stand-in for prefilled caches, for benchmarks and tests."""
    width = config.kv_heads * HEAD_DIM
    valid = (np.arange(max_len)[None, :] < np.asarray(lengths)[:, None])[:, :, None]
    keys = jax.random.split(jax.random.key(seed), 2 * config.layers)
    caches = []
    for i in range(config.layers):
        k, v = (jax.random.normal(keys[2 * i + n], (len(lengths), max_len, width), jnp.bfloat16) for n in (0, 1))
        caches.append({"k": jnp.where(valid, k, 0), "v": jnp.where(valid, v, 0)})
    return caches


def paged_caches(caches: list[dict], block_tables: jax.Array) -> list[dict]:
    """Contiguous caches as pages (`kv_cache.to_pages`)."""
    return [{name: to_pages(c, block_tables) for name, c in cache.items()} for cache in caches]


if __name__ == "__main__":
    config = Config(layers=2, d_model=256, heads=4, kv_heads=2, ffn=512, vocab=512)
    params = init_params(config)
    prompts = jax.random.randint(jax.random.key(1), (2, 128), 0, config.vocab)
    prompt_lengths = jnp.array([100, 37], jnp.int32)
    ours = generate(prompts, prompt_lengths, params, config, steps=4, max_len=256)
    print("autokernel greedy tokens:", ours.tolist())
    paged = generate(prompts, prompt_lengths, params, config, steps=4, max_len=256, page_size=128)
    print("paged cache, same tokens:", bool((paged == ours).all()))
