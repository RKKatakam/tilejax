"""DeepSeek-V3 / Kimi K2 inference, end to end: prefill, a latent KV cache, and greedy decoding, as autokernel
kernels composed in plain JAX.

The architecture (DeepSeek-V3; Kimi K2 is the same with other sizes): token embedding, then L pre-norm layers

    x = x + MLA(rmsnorm(x))                                  multi-head latent attention (examples/mla.py)
    x = x + FFN(rmsnorm(x))                                  the first `first_k_dense` layers: a dense SwiGLU
    x = x + sum_k w_k * expert_{e_k}(n) + shared(n)          the others: DeepSeekMoE (examples/moe.py), n = rmsnorm(x)

and the final norm and the vocabulary projection. MoE routing: sigmoid scores of the router logits; the
top-k experts chosen by score + a per-expert bias (load balancing without an auxiliary loss), DeepSeek-V3
first keeping the `topk_group` best of `n_group` expert groups; the chosen scores renormalized and scaled by
`routed_scaling_factor`; one shared expert every token goes through. (DeepSeek-V3: 128 heads, 256 experts
in 8 groups of which 4 are kept, top-8, scale 2.5; Kimi K2: 64 heads, 384 experts, no groups, top-8, scale
2.827. `CONFIGS` holds small, realistically shaped stand-ins that fit a 16 GB v5e.)

MLA per layer, in both forms (H heads; latent ranks Rq for q and C = 512 for keys and values):

    c_q = rmsnorm(h @ W_dq),  c = rmsnorm(h @ W_dkv),  k_r = rope(h @ W_kr)          (C + 64 per token: the cache)
    expanded (prefill):  q_h = c_q @ W_uq_h,  k_h = [c @ W_uk_h | k_r],  v_h = c @ W_uv_h,
                         o_h = softmax(q_h . k_h / sqrt(192)) @ v_h,   out = x + sum_h o_h @ W_o_h
    absorbed (decode):   q_c,h = c_q @ (W_uq_nope_h @ W_uk_h^T),   q_r,h = rope(c_q @ W_uq_rope_h)
                         o_c,h = softmax((q_c,h . c + q_r,h . k_r) / sqrt(192)) @ c,
                         out = x + sum_h o_c,h @ (W_uv_h @ W_o_h)

`prefill(tokens, params, max_len)` runs the expanded form (mla.py's kernels: k_nope and v are made once per
token and every query block reuses them) and keeps what decoding needs: per layer, the normalized latents c
(B, S_max, C) and the rotary keys k_r in a 128-lane slot (B, S_max, 128), as `mla_down` writes them,
[k_r | k_r] (decode.py's `mla_decode` reads the slot against queries [q_r | 0], so the copy in the upper
half never counts). `decode_step(tokens, cache, lengths, params)` runs the absorbed form over that cache:
no per-head keys or values are ever formed, a cache position costs C + 128 numbers for every head.

The absorbed form's weights come in two layouts (`pack(params, decode=...)`), made once:

- "folded" (`fold`): W_uk folded into the q up projection, W_uq_absorbed = [W_uq_nope_h @ W_uk_h^T for each
  head | W_uq_rope] (q_rank, H * (C + 64)), so the latent queries are ONE matmul kernel; W_uv into the
  output projection, W_uvo = [W_uv_h @ W_o_h] (H * C, d_model), so the values and the output projection are
  one matmul over the attention's latent outputs.
- "factored" (`factor`): the same products head by head, (c_q @ W_uq_nope_h) @ W_uk_h^T and
  (o_c,h @ W_uv_h) @ W_o_h, in the kernels around attention (`latent_query_heads`, `decode_tail_heads`).
  Folding multiplies the weights decode reads: per head q_rank * (C + 64) + C * d_model numbers instead of
  q_rank * 192 + 2 * C * 128 + 128 * d_model, 2.8x more at these shapes (q_rank 1536, C 512, d_model
  2048); decode at small batches is bound by reading weights. `benchmarks/deepseek.py` times both.

Both carry the softmax scale. Kernels per decode step and layer (all rows of the step in one tile):

1. `mla_down` (mla.py) on the step's B * T rows as one sequence: RMSNorm, the down projections, the latent
   norms and RoPE on the rotary key. Its cos/sin tables are per ROW, so each row rotates at its own
   sequence's position. The new latents and rotary keys are appended to the cache in XLA (a row scatter,
   in place on the donated cache).
2. The latent and rotary queries: folded, `matmul` (matmul.py) against W_uq_absorbed; factored,
   `latent_query_heads` (here: a tile per block of heads, two matmuls per head). RoPE on those few rows is
   plain JAX.
3. `mla_decode` (decode.py): attention over each sequence's valid cache, every head as a row of one matmul
   per cache block, cache blocks past a sequence's length skipped (a runtime mask).
4. `decode_tail` / `decode_tail_heads` (here): the output projection (a phase over W_uvo's rows / over
   blocks of heads, W_uv then W_o), the residual, the RMSNorm, and a SwiGLU feed-forward layer (a phase over
   its hidden units): the dense FFN in dense layers, the SHARED expert in MoE layers, whose routed experts
   then run as moe.py's `expert_ffn` on the rows the router sends them (only the experts some token chose
   are read), combined in XLA. The kernel returns the residual too: the router reads it.

The final norm and the logits are `rmsnorm_matmul`. Prefill per layer: `mla_down`, two `matmul`s (the up
projections), `mla_attention` (attention, output projection, residual), then `swiglu_ffn` (dense layers) or
moe.py's `moe_layer` (routing in XLA, `expert_ffn` on the dispatched rows, `shared_experts`).

Kernels whose masks read index arrays (`mla_decode`'s lengths, `expert_ffn`'s experts) tune on the inputs
they are first called with; under jit those are random. `generate` therefore runs its first prefill and its
first decode step eagerly (which also measures the MoE layout's group size, `moe_group`), then jitted steps
with the cache donated.

`baseline_prefill` and `baseline_decode_step` are the same model in plain JAX for XLA: expanded attention by
XLA or JAX's splash attention (q/k head dim 192) for prefill, absorbed attention over the same latent cache
by XLA for decode (W_uk and W_uv applied per head, unfolded: fewer weight bytes), MoE by XLA's ragged_dot
(or every expert on every token, for decode batches).

Run with `uv run python examples/deepseek.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from decode import absorb_queries, mla_decode, xla_mla_decode
from kv_cache import append, pad_lanes
from matmul import matmul
from mla import LANES, NOPE, QK, ROPE, THETA, MlaConfig, mla_attention, mla_down, rope
from mla import init_params as init_attention
from mla import pack as pack_attention
from moe import (
    DECODE_TOKENS,
    MoeConfig,
    choose_group,
    combine,
    decode_call,
    default_group,
    expert_ffn,
    expert_inputs,
    moe_layer,
)
from moe import baseline as moe_baseline
from moe import baseline_params as moe_baseline_params
from moe import init_params as init_moe
from rmsnorm_matmul import rmsnorm_matmul, rmsnorm_then_matmul
from swiglu import rms_rows, swiglu, swiglu_ffn, swiglu_ffn_xla

import autokernel

EPS = 1e-6  # every RMSNorm's epsilon, as mla.py's, moe.py's and swiglu.py's


@dataclass(frozen=True)
class Config:
    """A DeepSeek-V3-like model: MLA attention (q_rank, kv_rank latents; heads of 128 + 64 dims, values 128),
    the first `first_k_dense` layers with a dense SwiGLU of `dense_ffn` hidden units, the others DeepSeekMoE:
    `experts` routed experts of `expert_ffn` hidden units, `top_k` per token (from the best `topk_group` of
    `n_group` groups), weights scaled by `scale`, and `shared` shared experts."""

    layers: int
    d_model: int
    heads: int  # even: mla_attention holds two heads' rotary queries per 128-lane slot
    q_rank: int
    kv_rank: int
    experts: int
    top_k: int
    expert_ffn: int
    dense_ffn: int
    vocab: int
    n_group: int = 1
    topk_group: int = 1
    scale: float = 2.5  # routed_scaling_factor
    shared: int = 1
    first_k_dense: int = 1

    @property
    def mla(self) -> MlaConfig:
        return MlaConfig(self.d_model, self.heads, self.q_rank, self.kv_rank)

    @property
    def moe(self) -> MoeConfig:
        return MoeConfig(
            self.d_model,
            self.experts,
            self.top_k,
            self.expert_ffn,
            shared=self.shared,
            scoring="sigmoid",
            n_group=self.n_group,
            topk_group=self.topk_group,
            scale=self.scale,
        )


# A Config is static: a pytree with no leaves, so `params` can carry it through jax.jit.
jax.tree_util.register_static(Config)

# Small stand-ins with the real models' per-layer shapes scaled to d_model 2048 (DeepSeek-V3: d_model 7168,
# 128 heads, 256 experts of 2048 in 8 groups (4 kept), dense FFN 18432; Kimi K2: 64 heads, 384 experts, no
# groups), sized for a 16 GB v5e with the baselines' copies of the weights. Kimi K2's expert count keeps
# its 1.5x ratio to DeepSeek-V3's.
CONFIGS = {
    "deepseek-v3": Config(4, 2048, 32, 1536, 512, 128, 8, 512, 6144, 32768, n_group=8, topk_group=4, scale=2.5),
    "kimi-k2": Config(4, 2048, 16, 1536, 512, 192, 8, 512, 6144, 32768, scale=2.827),
}


# --------------------------------------------------------------------------- the decode kernels of this example


def _dot(a, b):
    """a @ b accumulating in f32, as `@` in kernel code means. (A helper is plain JAX.)"""
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _dot_nt(a, b):
    """a @ b.T, the MXU reading b transposed."""
    return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def latent_heads(c_q, w_uq_nope, w_uk):
    """A block of heads' absorbed queries, f32 (heads, rows, C): head g's q_nope = c_q @ w_uq_nope[g] (rounded
    to c_q's dtype, as the expanded form's q is), then q_nope @ w_uk[g].T."""
    return jnp.stack([_dot_nt(_dot(c_q, w_uq_nope[g]).astype(c_q.dtype), w_uk[g]) for g in range(w_uk.shape[0])])


def rope_query_heads(c_q, w_uq_rope):
    """A block of heads' rotary queries before RoPE, f32 (heads, rows, 64)."""
    return jnp.stack([_dot(c_q, w_uq_rope[g]) for g in range(w_uq_rope.shape[0])])


def value_heads(o, w_uv, w_o):
    """A block of heads' share of the output projection, f32 (rows, d_model): the sum over heads g of
    (o[g] @ w_uv[g]) (head g's values, rounded to o's dtype) @ w_o[g]."""
    out = _dot(_dot(o[0], w_uv[0]).astype(o.dtype), w_o[0])
    for g in range(1, o.shape[0]):
        out += _dot(_dot(o[g], w_uv[g]).astype(o.dtype), w_o[g])
    return out


@autokernel.tunable
class TailTiles:
    """Rows of the output projection per step, hidden units of the feed-forward layer per step."""

    block_a: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)
    block_f: int = autokernel.choice(128, 256, 512, 1024, 2048, 4096, default=512)


@autokernel.kernel
def decode_tail(x, attn, w_out, gamma, w_gate, w_up, w_down, tiles=TailTiles()):
    """The rest of a decode layer after attention, on (rows, d_model): resid = x + attn @ w_out (the folded
    W_uv W_o), then resid + SwiGLU(rmsnorm(resid)). Returns both: an MoE layer routes from resid and adds
    its routed experts to the second (its shared expert's output). One tile of every row: each weight is
    read once per step."""
    M, D = x.shape
    A, F = attn.shape[1], w_up.shape[1]
    ba, bf = min(tiles.block_a, A), min(tiles.block_f, F)
    dtype = x.dtype
    resid_out = jnp.zeros((M, D), dtype)
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, M):
        proj = jnp.zeros((M, D), jnp.float32)
        for a in range(0, A, ba):  # phase 1: the output projection, ba rows of W_uvo per step
            proj += attn[i : i + M, a : a + ba] @ w_out[a : a + ba, :]
        resid = x[i : i + M, :].astype(jnp.float32) + proj
        normed = rms_rows(resid, gamma[0:1, :]).astype(dtype)
        y = jnp.zeros((M, D), jnp.float32)
        for f in range(0, F, bf):  # phase 2: SwiGLU, bf hidden units per step
            hidden = swiglu(normed @ w_gate[:, f : f + bf], normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        resid_out = resid_out.at[i : i + M, :].set(resid.astype(dtype))
        out = out.at[i : i + M, :].set((resid + y).astype(dtype))
    return resid_out, out


@autokernel.tunable
class HeadTiles:
    """Heads per tile."""

    heads: int = autokernel.choice(1, 2, 4, 8, 16, default=4)


@autokernel.kernel
def latent_query_heads(c_q, w_uq_nope, w_uq_rope, w_uk, tiles=HeadTiles()):
    """The absorbed queries with W_uk unfolded, head-major: q_c (H, rows, C), q_c[h] = (c_q @ W_uq_nope[h]) @
    W_uk[h]^T, and the rotary queries before RoPE (H, rows, 64), c_q @ W_uq_rope[h]; pre-scaled (in the
    weights). A tile per block of heads, each reading its heads' weights once."""
    M = c_q.shape[0]
    H, C, _ = w_uk.shape
    hb = min(tiles.heads, H)
    dtype = c_q.dtype
    q_c = jnp.zeros((H, M, C), dtype)
    q_rope = jnp.zeros((H, M, ROPE), dtype)
    for h in range(0, H, hb):
        rows = c_q[:, :]
        q_c = q_c.at[h : h + hb, :, :].set(
            latent_heads(rows, w_uq_nope[h : h + hb, :, :], w_uk[h : h + hb, :, :]).astype(dtype)
        )
        q_rope = q_rope.at[h : h + hb, :, :].set(rope_query_heads(rows, w_uq_rope[h : h + hb, :, :]).astype(dtype))
    return q_c, q_rope


@autokernel.tunable
class HeadTailTiles:
    """Heads of the output projection per step, hidden units of the feed-forward layer per step."""

    heads: int = autokernel.choice(1, 2, 4, 8, 16, default=4)
    block_f: int = autokernel.choice(128, 256, 512, 1024, 2048, 4096, default=512)


@autokernel.kernel
def decode_tail_heads(x, heads, w_uv, w_o, gamma, w_gate, w_up, w_down, tiles=HeadTailTiles()):
    """`decode_tail` with W_uv unfolded: from the attention's latent outputs head-major (H, rows, C),
    resid = x + sum_h (heads[h] @ W_uv[h]) @ W_o[h] (a phase over blocks of heads, streaming both weights),
    then resid + SwiGLU(rmsnorm(resid)); both returned."""
    M, D = x.shape
    H, F = heads.shape[0], w_up.shape[1]
    hb, bf = min(tiles.heads, H), min(tiles.block_f, F)
    dtype = x.dtype
    resid_out = jnp.zeros((M, D), dtype)
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, M):
        proj = jnp.zeros((M, D), jnp.float32)
        for hh in range(0, H, hb):  # phase 1: hb heads' values and their output projection per step
            proj += value_heads(heads[hh : hh + hb, i : i + M, :], w_uv[hh : hh + hb, :, :], w_o[hh : hh + hb, :, :])
        resid = x[i : i + M, :].astype(jnp.float32) + proj
        normed = rms_rows(resid, gamma[0:1, :]).astype(dtype)
        y = jnp.zeros((M, D), jnp.float32)
        for f in range(0, F, bf):  # phase 2: SwiGLU, bf hidden units per step
            hidden = swiglu(normed @ w_gate[:, f : f + bf], normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        resid_out = resid_out.at[i : i + M, :].set(resid.astype(dtype))
        out = out.at[i : i + M, :].set((resid + y).astype(dtype))
    return resid_out, out


# --------------------------------------------------------------------------- weights


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights in the model's own layout (and the config, static): per layer "attn" (mla.py's
    `init_params` layout) and "ffn" {norm, w_gate, w_up, w_down} (dense layers) or "moe" (moe.py's)."""
    d, v = config.d_model, config.vocab
    keys = iter(jax.random.split(jax.random.key(seed), 3 + 4 * config.layers))

    def normal(shape: tuple[int, ...], scale: float) -> jax.Array:
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm_weight(n: int) -> jax.Array:
        return (1 + 0.1 * jax.random.normal(next(keys), (1, n))).astype(dtype)

    layers = []
    for i in range(config.layers):
        layer: dict[str, Any] = {"attn": init_attention(config.mla, dtype, seed=1000 * seed + 2 * i + 1)}
        if i < config.first_k_dense:
            f = config.dense_ffn
            layer["ffn"] = {
                "norm": norm_weight(d),
                "w_gate": normal((d, f), d**-0.5),
                "w_up": normal((d, f), d**-0.5),
                "w_down": normal((f, d), f**-0.5),
            }
        else:
            layer["moe"] = init_moe(config.moe, dtype, seed=1000 * seed + 2 * i + 2)
        layers.append(layer)
    return {
        "config": config,
        "embed": normal((v, d), 1.0),
        "layers": layers,
        "final_norm": norm_weight(d),
        "w_vocab": normal((d, v), d**-0.5),
    }


def _float(t: jax.Array) -> jax.Array:
    """t at least f32 (f64 stays f64)."""
    return t.astype(jnp.promote_types(t.dtype, jnp.float32))


def fold(attn: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """The absorbed form's weights, from mla.py's layout (computed at full precision, then rounded once):
    W_uq_absorbed (q_rank, H * C + H * 64) = [W_uq_nope_h @ W_uk_h^T for each head | W_uq_rope_h for each
    head] and W_uvo (H * C, d_model) = [W_uv_h @ W_o_h for each head], head-major; the q weights carry the
    softmax scale 1/sqrt(192)."""
    w_uq = _float(attn["w_uq"])  # (q_rank, H, 192)
    rq, h, _ = w_uq.shape
    highest = jax.lax.Precision.HIGHEST
    q_c = jnp.einsum("rhe,che->rhc", w_uq[..., :NOPE], _float(attn["w_uk"]), precision=highest)
    c = q_c.shape[-1]
    w_uq_absorbed = jnp.concatenate([q_c.reshape(rq, h * c), w_uq[..., NOPE:].reshape(rq, h * ROPE)], axis=1)
    w_uvo = jnp.einsum("che,hed->hcd", _float(attn["w_uv"]), _float(attn["w_o"]), precision=highest)
    dtype = attn["w_uq"].dtype
    return {
        "w_uq_absorbed": (w_uq_absorbed * QK**-0.5).astype(dtype),
        "w_uvo": w_uvo.reshape(h * c, -1).astype(dtype),
    }


def factor(attn: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """The absorbed form's weights unfolded, head-major, for `latent_query_heads` and `decode_tail_heads`:
    W_uq's nope and rope parts (H, q_rank, 128) and (H, q_rank, 64), scaled by 1/sqrt(192); W_uk and W_uv
    (H, C, 128). (W_o is mla.py's (H, 128, d_model) already.)"""
    w_uq = _float(attn["w_uq"]) * QK**-0.5  # (q_rank, H, 192)
    dtype = attn["w_uq"].dtype
    return {
        "w_uq_nope": w_uq[..., :NOPE].transpose(1, 0, 2).astype(dtype),
        "w_uq_rope": w_uq[..., NOPE:].transpose(1, 0, 2).astype(dtype),
        "w_uk_heads": attn["w_uk"].transpose(1, 0, 2),
        "w_uv_heads": attn["w_uv"].transpose(1, 0, 2),
    }


DECODE_FORMS = {"folded": fold, "factored": factor}


def pack(params: dict[str, Any], decode: str = "factored") -> dict[str, Any]:
    """The kernels' layout of the weights (done once, when the model is loaded): each layer's attention
    weights as mla.py's `pack` makes them (prefill), and for decode `factor`'s ("factored": W_uk and W_uv
    applied head by head in the kernels around attention, reading 2.8x fewer weight bytes at CONFIGS' shapes:
    on a v5e 2-31% faster per step than "folded") or `fold`'s ("folded": folded into the projections around
    attention, one matmul each)."""
    make = DECODE_FORMS[decode]
    layers = [layer | {"attn": pack_attention(layer["attn"]) | make(layer["attn"])} for layer in params["layers"]]
    return params | {"layers": layers}


# --------------------------------------------------------------------------- RoPE at any positions


def rope_rows(positions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """cos and sin (n, 128) f32 for rows at `positions` (n,), in mla.py's lane order (`mla.rope_tables` is
    this at positions 0 .. S-1): lane l turns by position * theta ** (-2 (l mod 32) / 64)."""
    inv_freq = THETA ** (-jnp.arange(0, ROPE, 2, dtype=jnp.float32) / ROPE)
    angles = positions.astype(jnp.float32)[:, None] * jnp.tile(inv_freq, LANES // (ROPE // 2))[None, :]
    return jnp.cos(angles), jnp.sin(angles)


# --------------------------------------------------------------------------- the MoE layout's group size

_GROUPS: dict[tuple[int, MoeConfig], int] = {}


def moe_group(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> int:
    """The group size of `expert_ffn`'s padded layout for `x.shape[0]` tokens: measured by moe.py's
    `choose_group` the first time it is called on concrete tokens (on a TPU; `default_group` elsewhere),
    and remembered for the token count, so jitted calls later reuse it (and the kernel's tuning); under
    jit before that, `default_group`."""
    key = (x.shape[0], config)
    if key not in _GROUPS:
        if isinstance(x, jax.core.Tracer):
            return default_group(x.shape[0])
        _GROUPS[key] = choose_group(x, params, config)
    return _GROUPS[key]


# --------------------------------------------------------------------------- prefill


def attention_prefill(x: jax.Array, attn: dict[str, jax.Array], cos: jax.Array, sin: jax.Array) -> tuple:
    """x + MLA(x) for x (B, S, d_model) by mla.py's kernels (expanded form), and the layer's cache entries:
    the latents c (B, S, C) and the rotary keys [k_r | k_r] (B, S, 128)."""
    b, s, _ = x.shape
    c_q, c_kv, k_rope = mla_down(x, attn["attn_norm"], attn["w_down"], attn["q_norm"], attn["kv_norm"], cos, sin)
    q = matmul(c_q.reshape(b * s, -1), attn["w_uq"]).reshape(b, s, -1)
    kv = matmul(c_kv.reshape(b * s, -1), attn["w_ukv"]).reshape(b, s, -1)
    return mla_attention(x, q, kv, k_rope, cos, sin, attn["w_o"]), c_kv, k_rope


def ffn_prefill(x: jax.Array, layer: dict[str, Any], config: Config) -> jax.Array:
    """x + the layer's feed-forward part, on (tokens, d_model) rows: `swiglu_ffn`, or moe.py's `moe_layer`."""
    if "moe" in layer:
        return moe_layer(x, layer["moe"], config.moe, moe_group(x, layer["moe"], config.moe))
    f = layer["ffn"]
    return swiglu_ffn(x, f["norm"], f["w_gate"], f["w_up"], f["w_down"])


def _pad_positions(t: jax.Array, max_len: int) -> jax.Array:
    """(B, S, n) as (B, max_len, n), zeros after S."""
    return jnp.pad(t, ((0, 0), (0, max_len - t.shape[1]), (0, 0)))


def _last(x: jax.Array, lengths: jax.Array) -> jax.Array:
    """Each sequence's row at its last position, lengths[b] - 1: (B, d_model)."""
    return jnp.take_along_axis(x, (lengths - 1)[:, None, None], axis=1)[:, 0]


def prefill(
    tokens: jax.Array, params: dict[str, Any], max_len: int | None = None, prompt_lengths: jax.Array | None = None
) -> tuple[jax.Array, list[dict[str, jax.Array]], jax.Array]:
    """Prompts (B, S) through the model with autokernel kernels, `pack`ed weights: the logits at each prompt's
    last token (B, vocab), the latent cache (per layer {"c_kv": (B, max_len, C), "k_rope": (B, max_len,
    128)}, positions 0 .. S-1 filled) and the lengths. Prompts of different lengths are right-padded, with
    `prompt_lengths` (B,) (default: all S): causal attention keeps the padding out of the real tokens, and
    decoding overwrites it."""
    config: Config = params["config"]
    b, s = tokens.shape
    max_len = max_len or s
    lengths = jnp.full((b,), s, jnp.int32) if prompt_lengths is None else prompt_lengths.astype(jnp.int32)
    x = params["embed"][tokens]
    cos, sin = rope_rows(jnp.arange(s))
    cache = []
    for layer in params["layers"]:
        x, c_kv, k_rope = attention_prefill(x, layer["attn"], cos, sin)
        cache.append({"c_kv": _pad_positions(c_kv, max_len), "k_rope": _pad_positions(k_rope, max_len)})
        x = ffn_prefill(x.reshape(b * s, -1), layer, config).reshape(x.shape)
    logits = rmsnorm_matmul(_last(x, lengths), params["final_norm"], params["w_vocab"])
    return logits, cache, lengths


# --------------------------------------------------------------------------- decode


def attention_decode(x: jax.Array, attn: dict[str, jax.Array], entry: dict, lengths, cos, sin, shape) -> tuple:
    """Absorbed MLA for the step's rows x (B * T, d_model) (`shape` (B, T); cos and sin at each row's
    position): the latent outputs of attention (B, T, H, C), and the cache entry with the new latents and
    rotary keys appended."""
    b, t = shape
    h = attn["w_o"].shape[0]
    down = (attn["attn_norm"], attn["w_down"], attn["q_norm"], attn["kv_norm"], cos, sin)
    c_q, c_kv, k_rope = mla_down(x[None], *down)  # the step's rows as one sequence, each at its own position
    entry = {
        "c_kv": append(entry["c_kv"], lengths, c_kv.reshape(b, t, -1)),
        "k_rope": append(entry["k_rope"], lengths, k_rope.reshape(b, t, -1)),
    }
    if "w_uq_absorbed" in attn:  # folded: one matmul makes every head's latent and rotary queries
        q = matmul(c_q[0], attn["w_uq_absorbed"])
        c = (q.shape[1] - h * ROPE) // h
        q_c, q_rope = q[:, : h * c].reshape(b, t, h, c), q[:, h * c :]
    else:  # factored: head by head, through q_nope
        q_heads, rope_heads = latent_query_heads(c_q[0], attn["w_uq_nope"], attn["w_uq_rope"], attn["w_uk_heads"])
        q_c = q_heads.transpose(1, 0, 2).reshape(b, t, h, -1)
        q_rope = rope_heads.transpose(1, 0, 2).reshape(b * t, h * ROPE)
    q_r = pad_lanes(rope(q_rope, cos, sin).astype(q_c.dtype).reshape(b, t, h, ROPE))  # [q_r | 0]
    return mla_decode(q_c, q_r, entry["c_kv"], entry["k_rope"], lengths + t), entry


def tail(x: jax.Array, o_c: jax.Array, attn: dict[str, jax.Array], ffn: tuple[jax.Array, ...]) -> tuple:
    """`decode_tail` (folded weights) or `decode_tail_heads` (factored) on the attention's latent outputs o_c
    (B, T, H, C), with a feed-forward layer's (norm, w_gate, w_up, w_down): (resid, resid + FFN)."""
    rows = x.shape[0]
    if "w_uvo" in attn:
        return decode_tail(x, o_c.reshape(rows, -1), attn["w_uvo"], *ffn)
    heads = o_c.reshape(rows, *o_c.shape[2:]).transpose(1, 0, 2)  # (H, rows, C)
    return decode_tail_heads(x, heads, attn["w_uv_heads"], attn["w_o"], *ffn)


def moe_decode(x: jax.Array, o_c: jax.Array, attn: dict[str, jax.Array], moe: dict[str, Any], config: MoeConfig):
    """The end of an MoE layer for decode rows: `tail` (the output projection, the residual, and the shared
    expert), then the routed experts on the residual, added: up to moe.DECODE_TOKENS rows, one kernel
    (`moe.decode_call`: a step per expert some row chose); more, the dispatched `expert_ffn`."""
    s = moe["shared"]
    resid, out = tail(x, o_c, attn, (moe["norm"], s["w_gate"], s["w_up"], s["w_down"]))
    if resid.shape[0] <= DECODE_TOKENS:
        kernel, args = decode_call(resid, moe, dataclasses.replace(config, shared=0))
        routed = kernel(*args).astype(jnp.float32) - resid.astype(jnp.float32)  # (it adds the residual)
    else:
        inputs, d, weights = expert_inputs(resid, moe, config, moe_group(resid, moe, config))
        routed = combine(expert_ffn(*inputs), d.pos, weights)
    return (out.astype(jnp.float32) + routed).astype(x.dtype)


def decode_step(
    tokens: jax.Array, cache: list[dict[str, jax.Array]], lengths: jax.Array, params: dict[str, Any]
) -> tuple[jax.Array, list[dict[str, jax.Array]], jax.Array]:
    """One decode step with autokernel kernels (absorbed MLA over the latent cache), `pack`ed weights (either
    decode form): the logits (B, T, vocab) of the new tokens (B, T) (sequence b's at positions lengths[b]
    ..), the cache with their latents and rotary keys appended, and lengths + T.

    `mla_decode`'s and `expert_ffn`'s masks read index arrays: under jit they tune on random ones, so run a
    first step eagerly (`generate` does)."""
    config: Config = params["config"]
    b, t = tokens.shape
    x = params["embed"][tokens.reshape(-1)]  # (B * T, d_model)
    cos, sin = rope_rows((lengths[:, None] + jnp.arange(t)[None, :]).reshape(-1))
    new_cache = []
    for layer, entry in zip(params["layers"], cache, strict=True):
        o_c, entry = attention_decode(x, layer["attn"], entry, lengths, cos, sin, (b, t))
        new_cache.append(entry)
        if "moe" in layer:
            x = moe_decode(x, o_c, layer["attn"], layer["moe"], config.moe)
        else:
            f = layer["ffn"]
            x = tail(x, o_c, layer["attn"], (f["norm"], f["w_gate"], f["w_up"], f["w_down"]))[1]
    logits = rmsnorm_matmul(x, params["final_norm"], params["w_vocab"])
    return logits.reshape(b, t, -1), new_cache, lengths + t


# --------------------------------------------------------------------------- generation

_EAGER_DONE: set[tuple] = set()


def _first_eager(key: tuple) -> bool:
    """True the first time `key` is seen: run that call eagerly, so kernels tune on its index arrays."""
    if key in _EAGER_DONE:
        return False
    _EAGER_DONE.add(key)
    return True


_prefill_jit = jax.jit(prefill, static_argnames="max_len")
_decode_jit = jax.jit(decode_step, donate_argnums=1)


def generate(
    prompts: jax.Array,
    params: dict[str, Any],
    steps: int,
    prompt_lengths: jax.Array | None = None,
    max_len: int | None = None,
) -> jax.Array:
    """Greedy decoding: `steps` new tokens (B, steps) after each prompt (B, S) (`prefill`'s `prompt_lengths`),
    `pack`ed weights. The first prefill and the first decode step for a shape run eagerly (the kernels tune
    on real lengths and routings); the rest are jitted, the cache donated (updated in place)."""
    b, s = prompts.shape
    max_len = max_len or -(-(s + steps) // 128) * 128
    config = params["config"]
    folded = "w_uvo" in params["layers"][0]["attn"]  # the decode form: its kernels tune separately
    run_prefill = prefill if _first_eager(("prefill", config, b, s, max_len)) else _prefill_jit
    logits, cache, lengths = run_prefill(prompts, params, max_len=max_len, prompt_lengths=prompt_lengths)
    token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    out = [token]
    for _ in range(steps - 1):
        run = decode_step if _first_eager(("decode", config, folded, b, max_len)) else _decode_jit
        logits, cache, lengths = run(token[:, None], cache, lengths, params)
        token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
        out.append(token)
    return jnp.stack(out, axis=1)


# --------------------------------------------------------------------------- baselines: plain JAX


def _einsum(spec: str, a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.einsum(spec, a, b, preferred_element_type=jnp.promote_types(a.dtype, jnp.float32))


def _rmsnorm(t: jax.Array, gamma: jax.Array) -> jax.Array:
    t = _float(t)
    return t * jax.lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + EPS) * _float(gamma).reshape(-1)


def _rope_at(t: jax.Array, positions: jax.Array) -> jax.Array:
    """Rotate-half RoPE on the last dim (64) of t (B, S, ..., 64) at positions (B, S)."""
    t = _float(t)
    inv_freq = THETA ** (-jnp.arange(0, ROPE, 2, dtype=t.dtype) / ROPE)
    angles = positions.astype(t.dtype)[..., None] * jnp.concatenate([inv_freq, inv_freq])
    angles = angles.reshape(angles.shape[:2] + (1,) * (t.ndim - 3) + (ROPE,))
    first, second = t[..., : ROPE // 2], t[..., ROPE // 2 :]
    return t * jnp.cos(angles) + jnp.concatenate([-second, first], axis=-1) * jnp.sin(angles)


def latents(x: jax.Array, attn: dict[str, jax.Array], positions: jax.Array) -> tuple[jax.Array, ...]:
    """MLA's per-token projections in plain JAX, for x (B, S, d_model) at positions (B, S): q_nope (B, S, H,
    128), q_rope (B, S, H, 64) (rotated), both unscaled; c_kv (B, S, C); k_rope (B, S, 64), rotated. (mla.py's
    `latents` at any positions.)"""
    dtype = x.dtype
    h = _rmsnorm(x, attn["attn_norm"]).astype(dtype)
    c_q = _rmsnorm(_einsum("bsd,dr->bsr", h, attn["w_dq"]), attn["q_norm"]).astype(dtype)
    q = _einsum("bsr,rhe->bshe", c_q, attn["w_uq"])
    q_rope = _rope_at(q[..., NOPE:].astype(dtype), positions).astype(dtype)
    c_kv = _rmsnorm(_einsum("bsd,dr->bsr", h, attn["w_dkv"]), attn["kv_norm"]).astype(dtype)
    k_rope = _rope_at(_einsum("bsd,de->bse", h, attn["w_kr"]).astype(dtype), positions).astype(dtype)
    return q[..., :NOPE].astype(dtype), q_rope, c_kv, k_rope


def _expanded_attention(x: jax.Array, attn: dict[str, jax.Array], q_nope, q_rope, c_kv, k_rope) -> jax.Array:
    """x + causal MLA in the expanded form, by XLA (the whole (B, H, S, S) score matrix)."""
    dtype = x.dtype
    k_nope = _einsum("bsr,rhe->bshe", c_kv, attn["w_uk"]).astype(dtype)
    v = _einsum("bsr,rhe->bshe", c_kv, attn["w_uv"]).astype(dtype)
    s = _einsum("bqhe,bkhe->bhqk", q_nope, k_nope) + _einsum("bqhe,bke->bhqk", q_rope, k_rope)
    causal = jnp.tril(jnp.ones(s.shape[-2:], bool))
    p = jax.nn.softmax(jnp.where(causal, s * QK**-0.5, -jnp.inf), axis=-1)
    o = _einsum("bhqk,bkhe->bqhe", p.astype(dtype), v).astype(dtype)
    return (_float(x) + _einsum("bqhe,hed->bqd", o, attn["w_o"])).astype(dtype)


def _splash_attention(heads: int, seq: int, block_q: int | None, block_kv: int | None, interpret: bool):
    """x + causal MLA with JAX's splash attention (default block sizes unless given): q and k as [nope | rope]
    (192 wide; the shared rotary key copied to every head), v 128 wide, q pre-scaled. (mla.py's
    `splash_baseline`, from given latents.)"""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

    mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq, seq)) for _ in range(heads)])
    sizes = None
    if block_q is not None:
        sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    attend = jax.vmap(splash.make_splash_mha_single_device(mask, block_sizes=sizes, interpret=interpret))

    def attention(x, attn, q_nope, q_rope, c_kv, k_rope):
        dtype = x.dtype
        b, s, h, _ = q_nope.shape
        k_nope = _einsum("bsr,rhe->bshe", c_kv, attn["w_uk"]).astype(dtype)
        v = _einsum("bsr,rhe->bshe", c_kv, attn["w_uv"]).astype(dtype)
        q = (jnp.concatenate([q_nope, q_rope], -1) * QK**-0.5).astype(dtype)
        k = jnp.concatenate([k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, s, h, ROPE))], -1)
        o = attend(*(t.transpose(0, 2, 1, 3) for t in (q, k, v)))  # (B, H, S, 128)
        return (_float(x) + _einsum("bhqe,hed->bqd", o, attn["w_o"])).astype(dtype)

    return attention


def baseline_params(params: dict[str, Any]) -> dict[str, Any]:
    """The baselines' weights (model layout, made once outside the timed model): moe.py's `baseline_params`
    for MoE layers (gate and up side by side), and W_uk and W_uv head-major (H, C, 128) for decode."""
    layers = []
    for layer in params["layers"]:
        attn = layer["attn"] | {name + "_hcd": layer["attn"][name].transpose(1, 0, 2) for name in ("w_uk", "w_uv")}
        extra = {"moe": moe_baseline_params(layer["moe"])} if "moe" in layer else {}
        layers.append(layer | {"attn": attn} | extra)
    return params | {"layers": layers}


def _baseline_ffn(x: jax.Array, layer: dict[str, Any], config: Config, moe_method: str) -> jax.Array:
    """x + the layer's feed-forward part in plain JAX, on (tokens, d_model) rows."""
    if "moe" in layer:
        return moe_baseline(config.moe, moe_method)(x, layer["moe"])
    f = layer["ffn"]
    return swiglu_ffn_xla(x, f["norm"], f["w_gate"], f["w_up"], f["w_down"])


def baseline_prefill(
    attention: str = "xla",
    block_q: int | None = None,
    block_kv: int | None = None,
    moe_method: str = "ragged_dot",
    interpret: bool = False,
):
    """`prefill` in plain JAX, jitted, on `baseline_params`: `(tokens, params, max_len, prompt_lengths=None)
    -> (logits, cache, lengths)`, the cache as `prefill`'s (the rotary keys as [k_r | 0]). Attention by XLA
    ("xla") or JAX's splash attention ("splash", at the given block sizes); MoE by moe.py's baseline."""

    def run(tokens, params, max_len, prompt_lengths=None):
        config: Config = params["config"]
        b, s = tokens.shape
        lengths = jnp.full((b,), s, jnp.int32) if prompt_lengths is None else prompt_lengths.astype(jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
        attend = (
            _expanded_attention
            if attention == "xla"
            else _splash_attention(config.heads, s, block_q, block_kv, interpret)
        )
        x = params["embed"][tokens]
        cache = []
        for layer in params["layers"]:
            q_nope, q_rope, c_kv, k_rope = latents(x, layer["attn"], positions)
            cache.append({"c_kv": _pad_positions(c_kv, max_len), "k_rope": _pad_positions(pad_lanes(k_rope), max_len)})
            x = attend(x, layer["attn"], q_nope, q_rope, c_kv, k_rope)
            x = _baseline_ffn(x.reshape(b * s, -1), layer, config, moe_method).reshape(x.shape)
        logits = rmsnorm_then_matmul(_last(x, lengths), params["final_norm"], params["w_vocab"])
        return logits, cache, lengths

    return jax.jit(run, static_argnames="max_len")


def baseline_decode_step(moe_method: str = "ragged_dot"):
    """`decode_step` in plain JAX, jitted with the cache donated, on `baseline_params`: absorbed attention over
    the latent cache by XLA (decode.py's `absorb_queries` and `xla_mla_decode`, dense over the whole cache,
    masked; W_uv and W_o applied after it), MoE by moe.py's baseline ("ragged_dot", or "dense": every
    expert on every token)."""

    def step(tokens, cache, lengths, params):
        config: Config = params["config"]
        b, t = tokens.shape
        positions = lengths[:, None] + jnp.arange(t)[None, :]
        new_lengths = lengths + t
        x = params["embed"][tokens]
        new_cache = []
        for layer, entry in zip(params["layers"], cache, strict=True):
            attn = layer["attn"]
            q_nope, q_rope, c_kv, k_rope = latents(x, attn, positions)
            entry = {
                "c_kv": append(entry["c_kv"], lengths, c_kv),
                "k_rope": append(entry["k_rope"], lengths, pad_lanes(k_rope)),
            }
            new_cache.append(entry)
            q_c, q_r = absorb_queries(q_nope, q_rope, attn["w_uk_hcd"])
            o_c = xla_mla_decode(q_c, q_r, entry["c_kv"], entry["k_rope"], new_lengths)
            heads = o_c.reshape(b * t, *o_c.shape[2:]).transpose(1, 0, 2)  # head-major: a plain batched matmul
            o = _einsum("hnc,hce->hne", heads, attn["w_uv_hcd"]).astype(x.dtype)
            x = (_float(x) + _einsum("hne,hed->nd", o, attn["w_o"]).reshape(x.shape)).astype(x.dtype)
            x = _baseline_ffn(x.reshape(b * t, -1), layer, config, moe_method).reshape(x.shape)
        logits = rmsnorm_then_matmul(x, params["final_norm"], params["w_vocab"])
        return logits, new_cache, new_lengths

    return jax.jit(step, donate_argnums=1)


def kv_bytes_per_token(config: Config, dtype_bytes: int = 2) -> dict[str, int]:
    """Cache bytes per token over all layers: the latent cache as the math needs it (C + 64) and as stored
    here (C + a 128-lane rotary slot), against what multi-head attention would cache (k 192 + v 128 per
    head)."""
    per_layer = {
        "latent": config.kv_rank + ROPE,
        "latent, stored": config.kv_rank + LANES,
        "MHA": config.heads * (QK + NOPE),
    }
    return {name: n * dtype_bytes * config.layers for name, n in per_layer.items()}


if __name__ == "__main__":
    config = Config(2, 256, 2, 256, 512, 8, 2, 128, 512, 512, n_group=4, topk_group=2)
    params = init_params(config)
    packed = pack(params)
    prompts = jax.random.randint(jax.random.key(1), (2, 128), 0, config.vocab)
    tokens = generate(prompts, packed, steps=4)  # tuned on a TPU, the default tiles elsewhere
    print("autokernel greedy tokens:", tokens.tolist())
    factored = generate(prompts, pack(params, decode="factored"), steps=4)
    print("the factored decode form, same tokens:", bool((factored == tokens).all()))
    logits, cache, lengths = prefill(prompts, packed, max_len=256)
    plain, _, _ = baseline_prefill()(prompts, baseline_params(params), 256)
    diff = jnp.max(jnp.abs(logits.astype(jnp.float32) - plain.astype(jnp.float32)))
    print("prefill: max |autokernel - plain JAX| logits:", float(diff))
    print("latent cache per layer:", {name: t.shape for name, t in cache[0].items()})
    print("cache bytes per token:", kv_bytes_per_token(config))
