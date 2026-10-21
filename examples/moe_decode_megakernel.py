"""A whole decode step of a Qwen3-MoE-like model as ONE kernel: attention, the routing, the chosen experts,
every layer, and the LM head.

    per layer:  q, k, v = rmsnorm(x) @ W_qkv (QK-norm, RoPE);  x += attention(q, [K_cache; k], [V_cache; v]) @ W_o
                n = rmsnorm(x);  experts, weights = top_k(softmax(n @ W_router))
                x += sum_e weights[:, e] * (silu(n @ W_gate[e]) * (n @ W_up[e])) @ W_down[e]
    logits = rmsnorm(x) @ W_vocab

It is decode_megakernel.py's step with moe_megakernel.py's layer in place of the dense FFN:

    one tile: all B rows
      REPEAT LOOP over the layers:
        phase 1, d: the QKV projection;  QK-norm, RoPE, each sequence's attention state
        phase 2, j: attention over the caches, a block of a group of sequences per step, read up to the
           group's longest cache (a runtime mask: `if j % S >= group_max[...]`, the cursor skipping the blocks
           it masks as it moves)
        phase 3, a: the output projection;  residual, RMSNorm
        phase 4, r: the router;  at its last step, the routing (computed index arrays)
        phase 5, c: the chosen experts' hidden units, n[0] * F / be steps reading the experts the routing chose
      final RMSNorm;  phase 6, v: the LM head

Before routing could be computed inside a kernel, a MoE decode step was at least a kernel boundary per layer
(the routing in XLA, then the experts' kernel). `per_layer_step` is that step, kernels per layer: decode
_megakernel.py's (RMSNorm + QKV, attention, the output projection) and moe.py's decode layer (XLA routing,
`moe_decode`).

Run with `uv run python examples/moe_decode_megakernel.py`; without a TPU it runs in the Pallas TPU
interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import moe
from decode import gqa_decode
from decode_megakernel import (
    GROUPS,
    HEAD_DIM,
    ROPE_THETA,
    _heads,
    attention_rows,
    attention_start,
    group_lengths,
    grouped,
    lane_blocks,
    online_softmax_groups,
    qk_norm_rope,
    unblock,
)
from kv_cache import append
from layer_stack import rmsnorm
from moe_megakernel import TOP_K, route
from qwen3 import rope_tables
from qwen3_models import attention_output
from rmsnorm_matmul import rmsnorm_matmul

import autokernel


@dataclass(frozen=True)
class Config:
    """A Qwen3-MoE-like model: GQA with QK-norm (head_dim 128), MoE layers of `experts` SwiGLU experts of
    `ffn` hidden units, TOP_K per token (no shared expert)."""

    layers: int
    d_model: int
    heads: int
    kv_heads: int
    experts: int
    ffn: int
    vocab: int

    @property
    def moe(self) -> moe.MoeConfig:
        return moe.MoeConfig(self.d_model, self.experts, TOP_K, self.ffn)


@autokernel.tunable
class MoeMegaTiles:
    """Per step: rows of W_qkv, cache positions of a group of sequences, rows of W_o, rows of the router,
    hidden units of an expert, vocab columns; and how many sequences an attention step takes."""

    block_d: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_a: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    block_r: int = autokernel.choice(256, 512, 1024, default=512)
    block_e: int = autokernel.choice(128, 256, 384, 512, 768, default=256)
    block_v: int = autokernel.choice(256, 512, 1024, 2048, default=1024)
    group: int = autokernel.choice(*GROUPS, default=1)


@autokernel.kernel
def moe_decode_megakernel(
    x,
    cos,
    sin,
    lengths,
    group_max,
    seq_lengths,
    k_cache,
    v_cache,
    attn_norm,
    w_qkv,
    q_norm,
    k_norm,
    wo,
    ffn_norm,
    w_router,
    w_gate,
    w_up,
    w_down,
    final_norm,
    w_vocab,
    tiles=MoeMegaTiles(),
):
    """A whole MoE decode step (`kernel_inputs`' inputs): the logits (B, vocab) and every layer's new [k | v]
    rows (L, B, 2 * Hkv * Dh)."""
    B, D = x.shape
    L, _, S, KVD = k_cache.shape
    HD, V = wo.shape[1], w_vocab.shape[1]
    E, F = w_gate.shape[1], w_gate.shape[3]
    H, Hkv = HD // HEAD_DIM, KVD // HEAD_DIM
    C = min(E, B * TOP_K)  # the most experts B tokens can choose
    bd, bk, ba = min(tiles.block_d, D), min(tiles.block_kv, S), min(tiles.block_a, HD)
    br, be, bv = min(tiles.block_r, D), min(tiles.block_e, F), min(tiles.block_v, V)
    G, R = tiles.group, GROUPS.index(tiles.group)
    assert S % bk == 0 and B % G == 0 and F % be == 0 and D // br > 1  # (the router takes steps)
    scale = HEAD_DIM**-0.5
    dtype = x.dtype
    logits = jnp.zeros((B, V), dtype)
    new_kv = jnp.zeros((L, B, 2 * KVD), dtype)
    for i in range(0, B, B):
        h = x[i : i + B, :].astype(jnp.float32)
        kv_rows = jnp.zeros((L, B, 2 * KVD), dtype)
        logit_blocks = jnp.zeros((V // bv, B, bv), dtype)
        for layer in range(L):
            n_blocks = lane_blocks(rmsnorm(h, attn_norm[layer, :, :]).astype(dtype), bd)
            qkv = jnp.zeros((B, HD + 2 * KVD), jnp.float32)
            for d in range(0, D, bd):  # phase 1: QKV projection
                qkv += n_blocks[d // bd] @ w_qkv[layer, d : d + bd, :]
            cos_t, sin_t = cos[i : i + B, :], sin[i : i + B, :]
            q = qk_norm_rope(qkv[:, :HD], q_norm[layer, :, :], cos_t, sin_t) * scale
            k = qk_norm_rope(qkv[:, HD : HD + KVD], k_norm[layer, :, :], cos_t, sin_t)
            new = jnp.concatenate([q, k, qkv[:, HD + KVD :]], axis=1).astype(dtype)
            kv_rows = kv_rows.at[layer].set(new[:, HD:])
            q_bd, row_max, row_sum, acc = grouped(attention_start(new, H, Hkv), G)
            lens = seq_lengths[0:B, :, :].reshape(B // G, G, 1, 1)
            for j in range(0, B // G * S, bk):  # phase 2: attention, a group's block of positions per step
                if j % S >= group_max[R, j // S * G]:
                    continue
                kt = k_cache[
                    layer : layer + 1,
                    j // S * G : j // S * G + G,
                    autokernel.rows(j % S, group_max[R, j // S * G], bk),
                    :,
                ][0]
                vt = v_cache[
                    layer : layer + 1,
                    j // S * G : j // S * G + G,
                    autokernel.rows(j % S, group_max[R, j // S * G], bk),
                    :,
                ][0]
                s = jax.lax.dot_general(
                    q_bd[j // S], kt, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32
                )
                ok = j % S + jax.lax.broadcasted_iota(jnp.int32, (G, 1, bk), 2) < lens[j // S]
                valid = j % S + jax.lax.broadcasted_iota(jnp.int32, (G, bk, 1), 1) < lens[j // S]
                m_j, l_j, acc_j = online_softmax_groups(
                    s, jnp.where(valid, vt, 0), ok, row_max[j // S], row_sum[j // S], acc[j // S]
                )
                row_max = row_max.at[j // S].set(m_j)
                row_sum = row_sum.at[j // S].set(l_j)
                acc = acc.at[j // S].set(acc_j)
            rows = attention_rows(acc.reshape(B, H, KVD), row_sum.reshape(B, H, 128), H, Hkv)
            attn_blocks = lane_blocks(rows.astype(dtype), ba)
            proj = jnp.zeros((B, D), jnp.float32)
            for a in range(0, HD, ba):  # phase 3: output projection
                proj += attn_blocks[a // ba] @ wo[layer, a : a + ba, :]
            h = h + proj
            normed = rmsnorm(h, ffn_norm[layer, :, :]).astype(dtype)
            r_blocks = lane_blocks(normed.astype(jnp.float32), br)
            router_logits = jnp.zeros((B, E), jnp.float32)
            for r in range(0, D, br):  # phase 4: the router
                router_logits += r_blocks[r // br] @ w_router[layer, r : r + br, :]
            chosen, n, weights = route(router_logits, TOP_K, C)  # computed index arrays, made at its last step
            y = jnp.zeros((B, D), jnp.float32)
            for c in range(0, n[0] * F, be):  # phase 5: the chosen experts' hidden units
                g = normed @ w_gate[layer, chosen[c // F], :, c % F : c % F + be]
                u = normed @ w_up[layer, chosen[c // F], :, c % F : c % F + be]
                slot = jax.lax.broadcasted_iota(jnp.int32, (B, C), 1) == c // F
                weight = jnp.sum(jnp.where(slot, weights, 0.0), axis=1, keepdims=True)
                y += weight * (
                    (jax.nn.silu(g) * u).astype(dtype) @ w_down[layer, chosen[c // F], c % F : c % F + be, :]
                )
            h = h + y
        final = rmsnorm(h, final_norm[0:1, :]).astype(dtype)
        for v in range(0, V, bv):  # phase 6: LM head
            logit_blocks = logit_blocks.at[v // bv].set((final @ w_vocab[:, v : v + bv]).astype(dtype))
        logits = logits.at[i : i + B, :].set(unblock(logit_blocks))
        new_kv = new_kv.at[:, i : i + B, :].set(kv_rows)
    return logits, new_kv


# --------------------------------------------------------------------------- the step around it


WEIGHTS = (
    "attn_norm",
    "w_qkv",
    "q_norm",
    "k_norm",
    "wo",
    "ffn_norm",
    "w_router",
    "w_gate",
    "w_up",
    "w_down",
    "final_norm",
    "w_vocab",
)


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, jax.Array]:
    """Random weights stacked by layer (norms (L, 1, n); the router f32 (L, D, E); experts (L, E, D, F) and
    (L, E, F, D))."""
    c = config
    d, layers = c.d_model, c.layers
    hd, kvd = c.heads * HEAD_DIM, c.kv_heads * HEAD_DIM
    keys = iter(jax.random.split(jax.random.key(seed), 16))

    def normal(shape, scale, dt=dtype):
        return (jax.random.normal(next(keys), shape, jnp.float32) * scale).astype(dt)

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
        "w_router": normal((layers, d, c.experts), d**-0.5, jnp.float32),
        "w_gate": normal((layers, c.experts, d, c.ffn), d**-0.5),
        "w_up": normal((layers, c.experts, d, c.ffn), d**-0.5),
        "w_down": normal((layers, c.experts, c.ffn, d), c.ffn**-0.5),
        "final_norm": norm_weight(1, d),
        "w_vocab": normal((d, c.vocab), d**-0.5),
    }


def kernel_inputs(tokens, k_cache, v_cache, lengths, params) -> tuple[jax.Array, ...]:
    """The kernel's inputs for a step: embeddings, RoPE tables at positions `lengths`, `lengths` (its groups'
    longest, and as (B, 1, 1)), the caches (L, B, S_max, Hkv * Dh), the weights."""
    cos, sin = rope_tables(lengths, HEAD_DIM, ROPE_THETA)
    by_sequence = (lengths, group_lengths(lengths), lengths.reshape(-1, 1, 1))
    return (params["embed"][tokens], cos, sin, *by_sequence, k_cache, v_cache, *(params[n] for n in WEIGHTS))


def megakernel_step(tokens, k_cache, v_cache, lengths, params, tiles=None):
    """One decode step as one kernel: logits (B, vocab), the caches with the new keys and values written at
    positions `lengths` (an XLA scatter, in place on donated caches), lengths + 1."""
    kvd = k_cache.shape[3]
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    logits, new_kv = moe_decode_megakernel(*inputs, *(() if tiles is None else (tiles,)))
    rows = jnp.arange(tokens.shape[0])
    k_cache = k_cache.at[:, rows, lengths].set(new_kv[:, :, :kvd], mode="drop")
    v_cache = v_cache.at[:, rows, lengths].set(new_kv[:, :, kvd:], mode="drop")
    return logits, k_cache, v_cache, lengths + 1


def layer_weights(params: dict[str, jax.Array]) -> list[dict[str, Any]]:
    """Per-layer weights (separate arrays): the attention's, and moe.py's layer params under "moe"."""
    zero = jnp.zeros((params["w_router"].shape[2],), jnp.float32)
    layers = []
    for layer in range(params["w_qkv"].shape[0]):
        attention = {n: jnp.copy(params[n][layer]) for n in ("attn_norm", "w_qkv", "q_norm", "k_norm", "wo")}
        experts = {n: jnp.copy(params[n][layer]) for n in ("w_gate", "w_up", "w_down")}
        router = {"norm": jnp.copy(params["ffn_norm"][layer]), "router": jnp.copy(params["w_router"][layer])}
        layers.append({**attention, "moe": {**router, "bias": zero, **experts}})
    return layers


def per_layer_step(tokens, caches, lengths, layers, params, config: Config):
    """The same step with kernels per layer: `rmsnorm_matmul` (RMSNorm + QKV), QK-norm, RoPE and the cache
    append in XLA, `gqa_decode`, `attention_output` (x + attn @ W_o), then moe.py's decode layer (XLA
    routing, `moe_decode`); `rmsnorm_matmul` for the logits. `caches`: per-layer {"k", "v"} (B, S_max, Hkv *
    Dh)."""
    b = tokens.shape[0]
    x = params["embed"][tokens]
    new_lengths = lengths + 1
    new_caches = []
    for layer, cache in zip(layers, caches, strict=True):
        q, k, v = _heads(rmsnorm_matmul(x, layer["attn_norm"], layer["w_qkv"]), layer, lengths, config.heads)
        cache = {
            "k": append(cache["k"], lengths, k.reshape(b, 1, -1)),
            "v": append(cache["v"], lengths, v.reshape(b, 1, -1)),
        }
        new_caches.append(cache)
        attn = gqa_decode(q, cache["k"], cache["v"], new_lengths).reshape(b, -1)
        x = moe.moe_layer(attention_output(x, attn, layer["wo"]), layer["moe"], config.moe)
    return rmsnorm_matmul(x, params["final_norm"], params["w_vocab"]), new_caches, new_lengths


if __name__ == "__main__":
    from decode_megakernel import layer_caches, random_caches

    config = Config(layers=2, d_model=512, heads=4, kv_heads=2, experts=16, ffn=256, vocab=512)
    params = init_params(config)
    lengths = jnp.array([0, 1, 130, 250], jnp.int32)
    k_cache, v_cache = random_caches(config, lengths, 256)
    tokens = jnp.array([3, 1, 4, 1], jnp.int32)
    tiles = MoeMegaTiles(block_d=256, block_kv=128, block_a=256, block_r=256, block_e=128, block_v=256)
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    lowered = moe_decode_megakernel.lower(*inputs, tiles)
    print(lowered.source)
    print(f"{lowered.vmem_bytes / 2**20:.1f} MiB of VMEM")
    print("max |kernel - loop nest|:", moe_decode_megakernel.verify(*inputs, tiles))
    ours, *_ = megakernel_step(tokens, k_cache, v_cache, lengths, params, tiles)
    theirs, *_ = per_layer_step(tokens, layer_caches(k_cache, v_cache), lengths, layer_weights(params), params, config)
    print(
        "max |megakernel - per-layer kernels| logits:",
        float(jnp.max(jnp.abs(ours.astype(float) - theirs.astype(float)))),
    )
