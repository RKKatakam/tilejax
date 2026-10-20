"""End-to-end inference for two Qwen3-family models, every matmul and attention in autokernel kernels:
a dense Qwen3 (Qwen3-1.7B-like) and a Qwen3-MoE (Qwen3-30B-A3B-like), prefill, KV-cache decoding and greedy
generation.

    tokens -> embedding -> L x [ q, k, v = qkv(rmsnorm(x));  x += Wo(attention(q, k, v));
                                 x += FFN(rmsnorm(x)) ]  -> rmsnorm -> logits

    q_h = rope(rmsnorm_h(n @ Wq)_h * q_norm) / sqrt(Dh),  k_g = rope(rmsnorm_h(n @ Wk)_g * k_norm),
    v_g = (n @ Wv)_g    (QK-norm; H query heads, Hkv kv heads, head h attending with kv head h // (H / Hkv))
    dense FFN:  x += (silu(n @ W_gate) * (n @ W_up)) @ W_down
    MoE FFN:    p = softmax(n @ W_router);  the top-k experts e, weights p_e / sum(top-k p);
                x += sum_e w_e * (silu(n @ W_gate[e]) * (n @ W_up[e])) @ W_down[e]       (no shared expert)

The kernels are the other examples' (this module composes them, with plain-JAX glue: the embedding gather,
the router's top-k and the sort that dispatches tokens to experts, the cache appends):

PREFILL (`prefill`: prompts (B, S) -> the last token's logits and the KV cache), per layer:
1. qwen3.py's `make_qkv` projection: RMSNorm + QKV projection + QK-norm + RoPE in one pass over x; q comes
   out pre-scaled by 1/sqrt(head_dim). Its k and v ARE the cache's contents (RoPE applied), padded to the
   cache length in XLA.
2. dense: gqa_attention.py's GQA attention + output projection + residual, with the SwiGLU FFN fused into
   the same kernel or as swiglu.py's kernel of its own, whichever their tuned device times say
   (`qwen3.fuse_ffn`); MoE: the attention kernel, then moe.py's layer: router and dispatch in XLA, every
   expert's rows through `expert_ffn` (one grouped SwiGLU kernel whose tiles read their expert's weights
   through runtime windows), combine. The group size (rows per expert tile) is measured on the TPU on the
   first layer's real routing (`prefill_group`).
3. rmsnorm_matmul.py's kernel for the final norm and the logits of each prompt's last token.

DECODE (`decode_step`: one new token per sequence (B,), at its own position lengths[b]), per layer:
1. The SAME `make_qkv` projection kernel, on the B tokens as the rows of one sequence (1, B, D): its RoPE
   tables are inputs indexed by row, so row b gets the cos and sin of position lengths[b], and QK-norm and
   RoPE stay fused for decoding too. k and v are appended to the cache in place (XLA scatter, the cache
   donated to the jitted step).
2. decode.py's `gqa_decode`: one tile per sequence, a runtime mask stopping its walk over the cache at
   lengths[b], every head at once (block-diagonal queries).
3. dense: decode_model.py's `decode_mlp` (output projection, residual, RMSNorm, SwiGLU, residual: one tile
   of all B rows, two phases streaming the weights); MoE: `attention_output` (below: output projection and
   residual), then moe.py's layer on the B tokens: each expert some token chose is one tile of `expert_ffn`
   (group ~ B rows), so a step reads only the chosen experts' weights (8 of 128 for one token).
4. rmsnorm_matmul for the logits.

`attention_output` is the one kernel of this module: x + attn @ W_o for a few rows, one tile of every row
and one phase over blocks of W_o's rows (decode is bound by reading W_o once).

`generate` runs greedy decoding: prefill and the first decode step eagerly, so every kernel is tuned on
real inputs (the decode attention on real cache lengths, the experts on real routing; under jax.jit,
kernels with runtime masks tune on random index data), then jitted decode steps with the caches donated.

`baseline_prefill` and `baseline_decode` are the same models in plain JAX: JAX's splash attention (GQA via
its MQA kernel per kv head, or MHA with k and v repeated) for prefill, XLA's attention over the whole cache
or JAX's ragged_paged_attention for decode, XLA's dense FFN, and for MoE `jax.lax.ragged_dot` (or JAX's
Pallas megablox gmm, or every expert on every token for decode batches) via moe.py's baselines.

Run with `uv run python examples/qwen3_models.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import moe
import numpy as np
from decode import gqa_decode, jax_ragged_paged_attention, xla_gqa_decode
from decode_model import decode_mlp
from gqa_attention import make_attention, splash_gqa
from kv_cache import append, append_ragged
from qwen3 import (
    Config,
    attention_input,
    fuse_ffn,
    fused_layer,
    make_qkv,
    qk_heads_xla,
    qk_weights,
    rope_tables,
    split_layer,
)
from qwen3 import init_params as init_dense_params
from rmsnorm_matmul import rmsnorm_matmul
from swiglu import rms_rows, swiglu_ffn, swiglu_ffn_xla

import autokernel

B16 = jnp.bfloat16
F32 = jnp.float32
FFN_WEIGHTS = ("ffn_norm", "w_gate", "w_up", "w_down")  # a dense layer's feed-forward weights


@dataclass(frozen=True)
class ModelConfig:
    """A Qwen3 model: dense if `experts` is 0, else a mixture of `experts` SwiGLU experts of `ffn` hidden
    units each, `top_k` per token (softmax router, top-k weights renormalized, no shared expert)."""

    layers: int
    d_model: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    experts: int = 0
    top_k: int = 0
    rope_theta: float = 1e6

    @property
    def attention(self) -> Config:
        """The attention side, as qwen3.py's Config (QK-norm, full RoPE, causal)."""
        return Config(
            self.layers, self.d_model, self.heads, self.kv_heads, self.head_dim, self.ffn, self.vocab, self.rope_theta
        )

    @property
    def moe(self) -> moe.MoeConfig | None:
        """The feed-forward side of a MoE model, as moe.py's MoeConfig (None if dense)."""
        return moe.MoeConfig(self.d_model, self.experts, self.top_k, self.ffn) if self.experts else None


# A ModelConfig is static: a pytree with no leaves, so `params` can carry it through jax.jit.
jax.tree_util.register_static(ModelConfig)

# Fake small models with the real per-layer dims (and a 32k vocab): weights ~0.7 GB (dense) and ~5 GB (MoE).
CONFIGS = {
    "dense": ModelConfig(layers=4, d_model=2048, heads=16, kv_heads=8, head_dim=128, ffn=6144, vocab=32768),
    "moe": ModelConfig(
        layers=4, d_model=2048, heads=32, kv_heads=4, head_dim=128, ffn=768, vocab=32768, experts=128, top_k=8
    ),
}


# --------------------------------------------------------------------------- the one new kernel


@autokernel.tunable
class OutputTiles:
    """Rows of W_o per step."""

    block_a: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)


@autokernel.kernel
def attention_output(x, attn, wo, tiles=OutputTiles()):
    """x + attn @ wo on (rows, d_model) rows: one tile of every row, so W_o (H * Dh, d_model) is read once,
    `block_a` of its rows per step."""
    M, D = x.shape
    A = attn.shape[1]
    ba = min(tiles.block_a, A)
    dtype = x.dtype
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, M):
        acc = jnp.zeros((M, D), jnp.float32)
        for a in range(0, A, ba):  # the heads' columns, ba per step, against the matching rows of W_o
            acc += attn[i : i + M, a : a + ba] @ wo[a : a + ba, :]
        out = out.at[i : i + M, :].set((x[i : i + M, :].astype(jnp.float32) + acc).astype(dtype))
    return out


# --------------------------------------------------------------------------- parameters and caches


def init_params(config: ModelConfig, dtype: Any = B16, seed: int = 0) -> dict[str, Any]:
    """Random weights: qwen3.py's (attention, embedding, dense FFN, final norm, vocab projection); a MoE
    model's layers hold moe.py's layer weights under "moe" instead of the dense FFN's."""
    params = init_dense_params(config.attention, dtype, seed)
    layers = params["layers"]
    if config.moe is not None:
        layers = [
            {name: w for name, w in layer.items() if name not in FFN_WEIGHTS}
            | {"moe": moe.init_params(config.moe, dtype, seed=seed + 1 + n)}
            for n, layer in enumerate(layers)
        ]
    return params | {"config": config, "layers": layers}


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def random_cache(config: ModelConfig, lengths: Any, max_len: int, seed: int = 0) -> list[dict[str, jax.Array]]:
    """Per layer, a contiguous key and value cache (B, max_len, Hkv * Dh) of random bf16 values, zero past
    each sequence's length: a stand-in for prefilled caches (decode benchmarks, tests)."""
    width = config.kv_heads * config.head_dim
    valid = jnp.asarray(np.arange(max_len)[None, :] < np.asarray(lengths)[:, None])[:, :, None]
    keys = jax.random.split(jax.random.key(seed), 2 * config.layers)
    shape = (len(lengths), max_len, width)
    return [
        {name: jnp.where(valid, jax.random.normal(keys[2 * n + m], shape, B16), 0) for m, name in enumerate("kv")}
        for n in range(config.layers)
    ]


# --------------------------------------------------------------------------- prefill


def prefill_group(tokens: jax.Array, params: dict[str, Any]) -> int | None:
    """The MoE group size (rows per expert tile of `expert_ffn`) for prefilling `tokens`: measured on a TPU on
    the first layer's real routing (moe.choose_group, which tunes the kernel for each candidate); None (moe.py's
    default) for a dense model or off a TPU. Call eagerly."""
    config: ModelConfig = params["config"]
    if config.moe is None or jax.default_backend() != "tpu":
        return None
    x = params["embed"][tokens]
    b, s, d = x.shape
    cos, sin = rope_tables(jnp.arange(s), config.head_dim, config.rope_theta)
    layer = params["layers"][0]
    q, k, v = attention_input(x, layer, config.attention, cos, sin)
    x = make_attention(config.head_dim).attention(x, q, k, v, layer["wo"])
    return moe.choose_group(x.reshape(b * s, d), layer["moe"], config.moe)


def _layers(tokens: jax.Array, params: dict[str, Any], max_len: int, group: int | None) -> tuple[jax.Array, list]:
    """Every layer over whole prompts: the last layer's output (B, S, D) and the caches (keys and values at
    positions 0 .. S-1, zeros up to max_len)."""
    config: ModelConfig = params["config"]
    x = params["embed"][tokens]
    b, s, d = x.shape
    cos, sin = rope_tables(jnp.arange(s), config.head_dim, config.rope_theta)
    attention = make_attention(config.head_dim)
    pad = ((0, 0), (0, max_len - s), (0, 0))
    cache = []
    for layer in params["layers"]:
        qkv = attention_input(x, layer, config.attention, cos, sin)
        cache.append({"k": jnp.pad(qkv[1], pad), "v": jnp.pad(qkv[2], pad)})
        if config.moe is None:
            form = fused_layer if fuse_ffn(x, qkv, layer, config.attention) else split_layer
            x = form(x, qkv, layer, config.attention)
        else:
            x = attention.attention(x, *qkv, layer["wo"])
            x = moe.moe_layer(x.reshape(b * s, d), layer["moe"], config.moe, group).reshape(b, s, d)
    return x, cache


def prefill(
    tokens: jax.Array,
    params: dict[str, Any],
    max_len: int | None = None,
    prompt_lengths: jax.Array | None = None,
    group: int | None = None,
) -> tuple[jax.Array, list, jax.Array]:
    """Run prompts (B, S) through the model: the logits (B, vocab) at each prompt's last token, the KV cache
    (per layer {"k", "v"}, (B, max_len, Hkv * Dh), RoPE applied to k) and the cache lengths. Prompts shorter
    than S (`prompt_lengths`) are padded at the end: causal attention keeps the padding out of the real
    tokens, and decoding overwrites its cache rows. `group`: the MoE group size (`prefill_group`).
    Static arguments under jax.jit: max_len (default S) and group."""
    b, s = tokens.shape
    x, cache = _layers(tokens, params, max_len or s, group)
    lengths = jnp.full((b,), s, jnp.int32) if prompt_lengths is None else prompt_lengths.astype(jnp.int32)
    last = jnp.take_along_axis(x, (lengths - 1)[:, None, None], axis=1)[:, 0]
    return rmsnorm_matmul(last, params["final_norm"], params["w_vocab"]), cache, lengths


def forward(tokens: jax.Array, params: dict[str, Any], group: int | None = None) -> jax.Array:
    """The logits (B, S, vocab) at every position of prompts (B, S)."""
    x, _ = _layers(tokens, params, tokens.shape[1], group)
    b, s, d = x.shape
    return rmsnorm_matmul(x.reshape(b * s, d), params["final_norm"], params["w_vocab"]).reshape(b, s, -1)


# --------------------------------------------------------------------------- decode


def decode_step(
    token: jax.Array, cache: list, lengths: jax.Array, params: dict[str, Any], group: int | None = None
) -> tuple[jax.Array, list, jax.Array]:
    """One greedy-decoding step: the logits (B, vocab) of one new token per sequence (B,) at positions
    `lengths`, the caches with its keys and values appended, and lengths + 1. `group`: the MoE group size
    (default: about the B tokens, moe.default_group).

    gqa_decode's mask reads `lengths` (and expert_ffn's the routing): under jax.jit they tune on random
    index data, so run a first step eagerly to tune them on real data (`generate` does)."""
    config: ModelConfig = params["config"]
    b = token.shape[0]
    x = params["embed"][token]  # (B, D)
    cos, sin = rope_tables(lengths, config.head_dim, config.rope_theta)  # row b: position lengths[b]
    new_lengths = lengths + 1
    new_cache = []
    for layer, c in zip(params["layers"], cache, strict=True):
        q, k, v = attention_input(x[None], layer, config.attention, cos, sin)  # the B tokens as one sequence's rows
        c = {"k": append(c["k"], lengths, k.reshape(b, 1, -1)), "v": append(c["v"], lengths, v.reshape(b, 1, -1))}
        new_cache.append(c)
        attn = gqa_decode(q.reshape(b, 1, config.heads, config.head_dim), c["k"], c["v"], new_lengths)
        attn = attn.reshape(b, -1)
        if config.moe is None:
            x = decode_mlp(x, attn, layer["wo"], *(layer[w] for w in FFN_WEIGHTS))
        else:
            x = moe.moe_layer(attention_output(x, attn, layer["wo"]), layer["moe"], config.moe, group)
    return rmsnorm_matmul(x, params["final_norm"], params["w_vocab"]), new_cache, new_lengths


# decode_step jitted, the caches donated: XLA appends to them in place.
jitted_decode_step = jax.jit(decode_step, donate_argnums=1, static_argnames="group")


def generate(
    prompts: jax.Array,
    params: dict[str, Any],
    steps: int,
    max_len: int | None = None,
    prompt_lengths: jax.Array | None = None,
) -> jax.Array:
    """Greedy decoding: `steps` new tokens (B, steps) after each prompt (B, S). Prefill and the first decode
    step run eagerly, so every kernel tunes on real inputs (cache lengths, routing); the other steps are
    jitted, the caches donated (appended in place). The cache holds max_len positions (default: S + steps,
    rounded up to 128)."""
    max_len = max_len or round_up(prompts.shape[1] + steps, 128)
    logits, cache, lengths = prefill(prompts, params, max_len, prompt_lengths, prefill_group(prompts, params))
    token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    out = [token]
    for i in range(steps - 1):
        logits, cache, lengths = (decode_step if i == 0 else jitted_decode_step)(token, cache, lengths, params)
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        out.append(token)
    return jnp.stack(out, axis=1)


# --------------------------------------------------------------------------- the kernels a model runs


class KernelCall(NamedTuple):
    """A kernel a model calls, the shapes it calls it with, and what part of the model it is."""

    phase: str  # "projections", "attention", "ffn" or "moe"
    name: str
    kernel: autokernel.Kernel
    specs: tuple[jax.ShapeDtypeStruct, ...]


def _specs(*shapes: tuple[int, ...], dtype: Any = B16) -> tuple[jax.ShapeDtypeStruct, ...]:
    return tuple(jax.ShapeDtypeStruct(shape, dtype) for shape in shapes)


def _qkv_call(config: ModelConfig, x: tuple[int, ...], positions: int) -> KernelCall:
    d, dh = config.d_model, config.head_dim
    specs = _specs(x, (1, d), (d, config.heads * dh), (d, config.kv_heads * dh), (d, config.kv_heads * dh))
    specs += _specs((1, dh), (1, dh), (positions, dh), (positions, dh), dtype=F32)
    return KernelCall("projections", "qkv (rmsnorm, QK-norm, RoPE)", make_qkv(True, None).projection, specs)


def _experts_call(config: ModelConfig, tokens: int, group: int | None) -> KernelCall:
    group = group or moe.default_group(tokens)
    kernel, specs = moe.experts_specs(config.moe, tokens, group)
    return KernelCall("moe", f"experts (group {group})", kernel, specs)


def _logits_call(config: ModelConfig, rows: int) -> KernelCall:
    specs = _specs((rows, config.d_model), (1, config.d_model), (config.d_model, config.vocab))
    return KernelCall("projections", "final norm + logits", rmsnorm_matmul, specs)


def prefill_calls(
    config: ModelConfig, batch: int, seq: int, group: int | None = None, fused: bool | None = None
) -> list[KernelCall]:
    """The kernels `prefill` runs per layer (and once, the logits), with their input shapes. For a dense
    model, `fused` picks the layer's form (attention + FFN in one kernel, or two); None: both."""
    d, dh, f = config.d_model, config.head_dim, config.ffn
    hd, kd = config.heads * dh, config.kv_heads * dh
    x = (batch, seq, d)
    attention = make_attention(dh)
    qkv = (x, (batch, seq, hd), (batch, seq, kd), (batch, seq, kd))
    calls = [_qkv_call(config, x, seq)]
    if config.moe is not None:
        calls.append(KernelCall("attention", "attention + Wo + residual", attention.attention, _specs(*qkv, (hd, d))))
        calls.append(_experts_call(config, batch * seq, group))
    else:
        ffn = ((1, d), (d, f), (d, f), (f, d))
        if fused is not False:
            specs = _specs(*qkv, (hd, d), *ffn)
            calls.append(KernelCall("attention", "attention + Wo + FFN (fused layer)", attention.layer, specs))
        if fused is not True:
            calls.append(
                KernelCall("attention", "attention + Wo + residual", attention.attention, _specs(*qkv, (hd, d)))
            )
            calls.append(KernelCall("ffn", "SwiGLU FFN", swiglu_ffn, _specs((batch * seq, d), *ffn)))
    return [*calls, _logits_call(config, batch)]


def decode_calls(config: ModelConfig, batch: int, max_len: int, group: int | None = None) -> list[KernelCall]:
    """The kernels `decode_step` runs per layer (and once, the logits), with their input shapes."""
    d, dh, f = config.d_model, config.head_dim, config.ffn
    hd, kd = config.heads * dh, config.kv_heads * dh
    cache = _specs((batch, 1, config.heads, dh), (batch, max_len, kd), (batch, max_len, kd))
    calls = [
        _qkv_call(config, (1, batch, d), batch),
        KernelCall("attention", "gqa_decode", gqa_decode, cache + _specs((batch,), dtype=jnp.int32)),
    ]
    rows = ((batch, d), (batch, hd), (hd, d))
    if config.moe is None:
        specs = _specs(*rows, (1, d), (d, f), (d, f), (f, d))
        calls.append(KernelCall("ffn", "Wo + residual + SwiGLU FFN (decode_mlp)", decode_mlp, specs))
    else:
        calls.append(KernelCall("projections", "Wo + residual", attention_output, _specs(*rows)))
        calls.append(_experts_call(config, batch, group))
    return [*calls, _logits_call(config, batch)]


def device_ms(calls: list[KernelCall], layers: int) -> dict[str, float]:
    """The tuned device time of a model's kernels, ms per phase (each per-layer kernel times `layers`):
    from the tunings of the shapes the model ran (on a TPU; elsewhere nothing is timed: inf)."""
    out: dict[str, float] = {}
    for call in calls:
        n = 1 if call.name == "final norm + logits" else layers
        out[call.phase] = out.get(call.phase, 0.0) + n * call.kernel.tuning(*call.specs).best.seconds * 1e3
    return out


# --------------------------------------------------------------------------- baselines: plain JAX


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def baseline_params(params: dict[str, Any]) -> dict[str, Any]:
    """The baselines' weights: each layer's q, k and v weights side by side (`w_in`: one matmul), and a MoE
    layer's expert gate and up weights side by side (moe.baseline_params: one grouped matmul), WITHOUT the
    separate expert weights (so a benchmark holds the experts' weights about once: drop the original
    params). Shares every other array with `params`."""
    layers = []
    for layer in params["layers"]:
        layer = layer | {"w_in": jnp.concatenate([layer["w_q"], layer["w_k"], layer["w_v"]], axis=1)}
        if "moe" in layer:
            fused = moe.baseline_params(layer["moe"])
            layer["moe"] = {name: w for name, w in fused.items() if name not in ("w_gate", "w_up")}
        layers.append(layer)
    return params | {"layers": layers}


def _baseline_ffn(config: ModelConfig, experts: str, tiling: Any = None) -> Any:
    """The layer's feed-forward half in plain JAX, (x (T, D), layer) -> x + FFN(rmsnorm(x))."""
    if config.moe is None:
        return lambda x, layer: swiglu_ffn_xla(x, *(layer[w] for w in FFN_WEIGHTS))
    moe_layer = moe.baseline(config.moe, experts, tiling)
    return lambda x, layer: moe_layer(x, layer["moe"])


def _baseline_qkv(x: jax.Array, layer: dict[str, Any], config: ModelConfig, cos: jax.Array, sin: jax.Array):
    """q (pre-scaled), k, v (B, S, heads * Dh) in plain JAX: one projection matmul, then QK-norm and RoPE."""
    hd, kd = config.heads * config.head_dim, config.kv_heads * config.head_dim
    n = rms_rows(x.astype(F32), layer["attn_norm"]).astype(x.dtype)
    q, k, v = jnp.split(_dot(n, layer["w_in"]).astype(x.dtype), [hd, hd + kd], axis=-1)
    q_weight, k_weight = qk_weights(layer, config.attention)
    q = qk_heads_xla(q, q_weight, cos, sin, config.attention).astype(x.dtype)
    k = qk_heads_xla(k, k_weight, cos, sin, config.attention).astype(x.dtype)
    return q, k, v


def _baseline_logits(x: jax.Array, params: dict[str, Any]) -> jax.Array:
    n = rms_rows(x.astype(F32), params["final_norm"]).astype(x.dtype)
    return _dot(n, params["w_vocab"]).astype(x.dtype)


def baseline_prefill(
    config: ModelConfig,
    seq: int,
    block_q: int | None = None,
    block_kv: int | None = None,
    form: str = "mqa",
    experts: str = "ragged_dot",
    tiling: Any = None,
    interpret: bool = False,
) -> Any:
    """`prefill` in plain JAX, jitted, taking (tokens, baseline_params(params), max_len): JAX's splash
    attention for GQA (gqa_attention.splash_gqa: `form` "mqa" or "mha", its default blocks unless given),
    XLA's dense FFN, or moe.py's MoE baseline (`experts`: "ragged_dot", or "megablox" with `tiling`)."""
    attend = splash_gqa(config.heads, config.kv_heads, seq, None, block_q, block_kv, form, interpret)
    ffn = _baseline_ffn(config, experts, tiling)

    def run(tokens, params, max_len):
        x = params["embed"][tokens]
        b, s, d = x.shape
        cos, sin = rope_tables(jnp.arange(s), config.head_dim, config.rope_theta)
        pad = ((0, 0), (0, max_len - s), (0, 0))
        cache = []
        for layer in params["layers"]:
            q, k, v = _baseline_qkv(x, layer, config, cos, sin)
            cache.append({"k": jnp.pad(k, pad), "v": jnp.pad(v, pad)})
            x = (x.astype(F32) + _dot(attend(q, k, v), layer["wo"])).astype(x.dtype)
            x = ffn(x.reshape(b * s, d), layer).reshape(b, s, d)
        return _baseline_logits(x[:, -1], params), cache, jnp.full((b,), s, jnp.int32)

    return jax.jit(run, static_argnums=2)


def baseline_decode(config: ModelConfig, attention: str = "xla", experts: str = "ragged_dot", **block_sizes: Any):
    """`decode_step` in plain JAX, jitted with the caches donated: (token, cache, lengths,
    baseline_params(params), block_tables) -> (logits, cache, lengths). Attention with XLA over the whole
    contiguous cache ("xla"; cache per layer {"k", "v"} as `prefill`'s), or JAX's Pallas
    ragged_paged_attention ("ragged"; per layer {"kv"} (num_pages, P, 2 Hkv, Dh), kv_cache.ragged_kv_pages;
    `block_sizes` its block sizes). MoE experts by `experts`: "ragged_dot" or "dense" (every expert on every
    token)."""
    ffn = _baseline_ffn(config, experts)

    def step(token, cache, lengths, params, block_tables=None):
        x = params["embed"][token]  # (B, D)
        b = x.shape[0]
        cos, sin = rope_tables(lengths, config.head_dim, config.rope_theta)
        new_lengths = lengths + 1
        new_cache = []
        for layer, c in zip(params["layers"], cache, strict=True):
            q, k, v = (t[0] for t in _baseline_qkv(x[None], layer, config, cos, sin))  # tokens as rows
            q = q.reshape(b, 1, config.heads, config.head_dim)
            if attention == "xla":
                c = {"k": append(c["k"], lengths, k[:, None]), "v": append(c["v"], lengths, v[:, None])}
                attn = xla_gqa_decode(q, c["k"], c["v"], new_lengths)
            else:
                k, v = (t.reshape(b, 1, config.kv_heads, config.head_dim) for t in (k, v))
                c = {"kv": append_ragged(c["kv"], block_tables, lengths, k, v)}
                attn = jax_ragged_paged_attention(q, c["kv"], block_tables, new_lengths, **block_sizes)
            new_cache.append(c)
            x = (x.astype(F32) + _dot(attn.reshape(b, -1), layer["wo"])).astype(x.dtype)
            x = ffn(x, layer)
        return _baseline_logits(x, params), new_cache, new_lengths

    return jax.jit(step, donate_argnums=1)


if __name__ == "__main__":
    tiny = {
        "dense": ModelConfig(layers=2, d_model=256, heads=4, kv_heads=2, head_dim=128, ffn=512, vocab=512),
        "moe": ModelConfig(
            layers=2, d_model=256, heads=4, kv_heads=1, head_dim=128, ffn=256, vocab=512, experts=8, top_k=2
        ),
    }
    prompts = jax.random.randint(jax.random.key(1), (2, 128), 0, 512)
    for name, config in tiny.items():
        params = init_params(config)
        tokens = generate(prompts, params, steps=4)  # tuned on a TPU, the default tiles elsewhere
        print(f"{name}: greedy tokens {tokens.tolist()}")
    print(attention_output.source(*_specs((8, 256), (8, 512), (512, 256)), OutputTiles(block_a=256)))
