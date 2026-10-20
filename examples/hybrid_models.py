"""Hybrid linear/full-attention models, end to end: a Qwen3-Next-like and a Kimi-Linear-like decoder, prefill and
decode, as autokernel kernels composed in plain JAX (the layers: hybrid_layers.py).

    tokens -> embedding -> L x [ x = mixer(x);  x = MoE(x) ] -> rmsnorm -> logits

Every fourth layer's mixer is full attention, the others linear attention (3:1); each mixer and each MoE
layer has its own pre-norm and residual. Small fake configs that fit a 16 GB v5e (`CONFIGS`, ~4 GB of
bf16 weights each, mostly experts):

    qwen3-next-small    d_model 2048, 8 layers: Gated DeltaNet (16 query/key heads, 32 value heads, 128
                        dims) and gated GQA (16 heads, 2 kv heads of 256, QK-norm, RoPE on a quarter of
                        each head); MoE of 64 experts (Qwen3-Next: 512), top 8, 512 hidden units each,
                        softmax router, a shared expert of 512 behind a sigmoid gate
    kimi-linear-small   d_model 2048, 8 layers: KDA (32 heads of 128) and MLA (32 heads, q_rank 1024,
                        kv_rank 512, 128 + 64 dims); MoE of 64 experts (Kimi Linear: 256), top 8, 512
                        each, sigmoid router (Kimi K2's), a shared expert. (Kimi Linear's own MLA layers
                        use no positional encoding and no q latent; these keep DeepSeek-V3's, as mla.py.)

`prefill(tokens, params)` runs prompts (B, S) and returns the logits of each prompt's last token and the
decode state, per layer: a linear layer's recurrent state (B, H, d_k, d_v) f32 and its convolution tail
(B, 3, conv channels); a full-attention layer's KV cache (GQA: k and v (B, S_max, kv_heads * head_dim);
MLA: the latents (B, S_max, kv_rank) and rope keys (B, S_max, 128)). `decode_step(token, state, lengths,
params)` runs one new token per sequence (sequence b at position lengths[b]) and returns its logits, the
state with the token in it, and lengths + 1. `generate(prompt, params, n)` is greedy decoding.

Prefill is where linear attention pays: a linear layer's work grows with S, not S^2, and its state is a
fixed 64 KiB per head. Decoding a linear layer reads and writes that state once per token, whatever the
context; only the full-attention layers' caches grow.

Each function takes `impl`: autokernel's kernels (`AUTOKERNEL`, the default) or the same model in plain JAX
(`Impl(kernels=False)`: the chunked delta rule and splash attention for prefill, XLA's attention over the
caches for decode, ragged_dot experts). `baseline_prefill` and `baseline_decode_step` are those, jitted.

Run with `uv run python examples/hybrid_models.py`; without a TPU the kernels run in the Pallas TPU
interpreter.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hybrid_layers import (
    AUTOKERNEL,
    HybridConfig,
    Impl,
    gdn_decode_layer,
    gdn_prefill_layer,
    gqa_decode_layer,
    gqa_prefill_layer,
    init_gqa,
    init_kda,
    init_mla,
    init_moe,
    kda_decode_layer,
    kda_prefill_layer,
    mla_decode_layer,
    mla_prefill_layer,
    moe_ffn,
    norm_matmul,
)
from linear_attention import HEAD_DIM
from linear_attention_layer import CONV
from linear_attention_layer import init_params as init_gdn
from mla import LANES
from moe import MoeConfig

CONFIGS = {
    "qwen3-next-small": HybridConfig(
        kind="qwen3-next",
        layers=8,
        d_model=2048,
        vocab=32768,
        k_heads=16,
        v_heads=32,
        heads=16,
        kv_heads=2,
        head_dim=256,
        moe=MoeConfig(2048, 64, 8, 512, shared=1),
    ),
    "kimi-linear-small": HybridConfig(
        kind="kimi-linear",
        layers=8,
        d_model=2048,
        vocab=32768,
        k_heads=32,
        v_heads=32,
        heads=32,
        q_rank=1024,
        kv_rank=512,
        moe=MoeConfig(2048, 64, 8, 512, shared=1, scoring="sigmoid", scale=2.446),
    ),
}


def tiny_config(kind: str, top_k: int = 2) -> HybridConfig:
    """A tiny model of either kind (d_model 256, 4 layers, 4 experts of 128 hidden units, `top_k` per token),
    for tests and quick runs: the kernels' smallest tiles, in the interpreter in seconds."""
    if kind == "qwen3-next":
        moe = MoeConfig(256, 4, top_k, 128, shared=1)
        return HybridConfig(kind, 4, 256, 512, k_heads=1, v_heads=2, heads=2, kv_heads=1, head_dim=256, moe=moe)
    moe = MoeConfig(256, 4, top_k, 128, shared=1, scoring="sigmoid", scale=2.0)
    return HybridConfig(kind, 4, 256, 512, k_heads=2, v_heads=2, heads=2, q_rank=256, kv_rank=256, moe=moe)


def init_params(config: HybridConfig, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights: per layer a "mixer" and a "moe"; the config rides along (a static pytree node)."""
    layers = []
    for i in range(config.layers):
        s = seed + 100 * (i + 1)
        if config.kind == "qwen3-next":
            mixer = init_gqa(config, s, dtype) if config.is_full(i) else init_gdn(config.linear, dtype, s)
        else:
            mixer = init_mla(config, s, dtype) if config.is_full(i) else init_kda(config, s, dtype)
        layers.append({"mixer": mixer, "moe": init_moe(config, s + 1, dtype)})
    keys = jax.random.split(jax.random.key(seed), 3)
    d = config.d_model
    return {
        "config": config,
        "embed": jax.random.normal(keys[0], (config.vocab, d)).astype(dtype),
        "layers": layers,
        "final_norm": (1 + 0.1 * jax.random.normal(keys[1], (1, d))).astype(dtype),
        "w_vocab": (jax.random.normal(keys[2], (d, config.vocab)) * d**-0.5).astype(dtype),
    }


def round_up(n: int, multiple: int = 512) -> int:
    return -(-n // multiple) * multiple


def prefill(tokens: jax.Array, params: dict[str, Any], max_len: int | None = None, impl: Impl = AUTOKERNEL):
    """Prompts tokens (B, S) (S a multiple of 128) through the model: the logits of each prompt's last token
    (B, vocab) and the decode state (see the module docstring), with KV caches of `max_len` positions (S if
    None). The prompts' lengths are S: decode from `jnp.full((B,), S)`."""
    config: HybridConfig = params["config"]
    B, S = tokens.shape
    max_len = max_len or S
    x = params["embed"][tokens]
    state = []
    for i, layer in enumerate(params["layers"]):
        mixer = layer["mixer"]
        if config.is_full(i):
            full = gqa_prefill_layer if config.kind == "qwen3-next" else mla_prefill_layer
            x, cache = full(x, mixer, config, impl, max_len)
        else:
            linear = gdn_prefill_layer if config.kind == "qwen3-next" else kda_prefill_layer
            x, cache = linear(x, mixer, config, impl)
        x = moe_ffn(x.reshape(B * S, -1), layer["moe"], config, impl).reshape(x.shape)
        state.append(cache)
    return norm_matmul(x[:, -1], params["final_norm"], params["w_vocab"], impl), state


def decode_step(token: jax.Array, state: list[dict], lengths: jax.Array, params: dict[str, Any], impl=AUTOKERNEL):
    """One new token per sequence, token (B,) at positions `lengths` (B,): its logits (B, vocab), the state
    with it, and lengths + 1. (The attention and expert kernels' masks read `lengths` and the routing: under
    jit they tune on random ones, so run a first step eagerly to tune them on real ones, as `generate` does.)"""
    config: HybridConfig = params["config"]
    x = params["embed"][token]
    new_state = []
    for i, (layer, cache) in enumerate(zip(params["layers"], state, strict=True)):
        mixer = layer["mixer"]
        if config.is_full(i):
            full = gqa_decode_layer if config.kind == "qwen3-next" else mla_decode_layer
            x, cache = full(x, mixer, config, cache, lengths, impl)
        else:
            linear = gdn_decode_layer if config.kind == "qwen3-next" else kda_decode_layer
            x, cache = linear(x, mixer, config, cache, impl)
        x = moe_ffn(x, layer["moe"], config, impl)
        new_state.append(cache)
    return norm_matmul(x, params["final_norm"], params["w_vocab"], impl), new_state, lengths + 1


def generate(prompt: jax.Array, params: dict[str, Any], n: int, impl: Impl = AUTOKERNEL) -> jax.Array:
    """Greedy decoding: n new tokens (B, n) after prompts (B, S). Prefill is jitted; on a TPU the first decode
    step runs eagerly (its kernels tune on the real lengths and routing), the rest jitted with the state
    donated (the caches updated in place)."""
    B, S = prompt.shape
    logits, state = jax.jit(prefill, static_argnames=("max_len", "impl"))(prompt, params, round_up(S + n), impl)
    lengths = jnp.full((B,), S, jnp.int32)
    token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    out = [token]
    step = jax.jit(decode_step, static_argnames="impl", donate_argnums=1)
    tune_eagerly = jax.default_backend() == "tpu"  # (off a TPU nothing is tuned)
    for i in range(n - 1):
        run = decode_step if i == 0 and tune_eagerly else step
        logits, state, lengths = run(token, state, lengths, params, impl=impl)
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        out.append(token)
    return jnp.stack(out, axis=1)


def baseline_prefill(attention: str = "splash", interpret: bool = False) -> Any:
    """`prefill` in plain JAX, jitted: (tokens, params, max_len=None) -> (logits, state); JAX's splash attention
    (`attention="splash"`, in its interpreter if `interpret`) or dense XLA attention ("xla")."""
    impl = Impl(kernels=False, attention=attention, interpret=interpret)
    return jax.jit(functools.partial(prefill, impl=impl), static_argnames="max_len")


def baseline_decode_step(experts: str = "ragged_dot") -> Any:
    """`decode_step` in plain JAX (XLA's attention over the whole caches, masked; `jax.lax.ragged_dot`
    experts, or "dense"), jitted with the state donated: (token, state, lengths, params)."""
    impl = Impl(kernels=False, experts=experts)
    return jax.jit(functools.partial(decode_step, impl=impl), donate_argnums=1)


def random_state(config: HybridConfig, lengths: Any, max_len: int, seed: int = 0) -> list[dict]:
    """A stand-in for a prefilled decode state (benchmarks): random recurrent states and conv tails, and
    random caches valid up to each sequence's length (zeros after)."""
    lengths = np.asarray(lengths)
    b = len(lengths)
    valid = jnp.asarray(np.arange(max_len)[None, :, None] < lengths[:, None, None])
    keys = iter(jax.random.split(jax.random.key(seed), 2 * config.layers))
    state = []
    for i in range(config.layers):
        k1, k2 = next(keys), next(keys)
        if config.is_full(i) and config.kind == "qwen3-next":
            width = config.kv_heads * config.head_dim
            k, v = (jax.random.normal(key, (b, max_len, width), jnp.bfloat16) for key in (k1, k2))
            cache = {"k": jnp.where(valid, k, 0), "v": jnp.where(valid, v, 0)}
        elif config.is_full(i):
            c_kv = jax.random.normal(k1, (b, max_len, config.kv_rank), jnp.bfloat16)
            k_rope = jax.random.normal(k2, (b, max_len, LANES), jnp.bfloat16)
            k_rope = jnp.where(jnp.arange(LANES) < LANES // 2, k_rope, 0)
            cache = {"c_kv": jnp.where(valid, c_kv, 0), "k_rope": jnp.where(valid, k_rope, 0)}
        else:
            heads = 2 * config.k_heads + config.v_heads if config.kind == "qwen3-next" else 3 * config.v_heads
            channels = heads * HEAD_DIM  # the convolution's: q, k and v
            shape = (b, config.v_heads, HEAD_DIM, HEAD_DIM)
            cache = {
                "state": 0.05 * jax.random.normal(k1, shape, jnp.float32),
                "conv": jax.random.normal(k2, (b, CONV - 1, channels), jnp.bfloat16),
            }
        state.append(cache)
    return state


if __name__ == "__main__":
    for kind in ("qwen3-next", "kimi-linear"):
        config = tiny_config(kind, top_k=4)  # every expert chosen: no routing near-ties between the two
        params = init_params(config)
        prompt = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
        ours = generate(prompt, params, 4)  # kernels tuned on a TPU, the default tiles elsewhere
        base = generate(prompt, params, 4, Impl(kernels=False, attention="xla"))
        print(f"{kind}: autokernel {ours.tolist()}, plain JAX {base.tolist()}")
