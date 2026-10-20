"""A Qwen3-style dense decoder (prefill: whole sequences, tokens to logits) as autokernel kernels.

    tokens -> embedding -> L x [ q, k, v = qkv(rmsnorm(x));  x += Wo(gqa_attention(q, k, v));
                                 x += SwiGLU(rmsnorm(x)) ] -> rmsnorm -> logits

Qwen3's attention input, per layer, is one kernel (`make_qkv`):

    n = rmsnorm(x)                                    (B, S, D)
    q_h = rope(rmsnorm_h(n @ Wq)_h * q_norm) / sqrt(Dh)   for each of H query heads (Dh = head_dim)
    k_g = rope(rmsnorm_h(n @ Wk)_g * k_norm)              for each of Hkv kv heads (GQA: Hkv divides H)
    v_g = (n @ Wv)_g
    gate = sigmoid(n @ Wgate)                          gated attention only (Qwen3-Next)

RMSNorm commutes with the projections (rmsnorm_matmul.py's trick: `rmsnorm(x) @ w = (1 / rms(x)) *
((x * gamma) @ w)`), so one reduction over d_model accumulates every projection of a block of rows on the
MXU together with the rows' sums of squares; everything else is the epilogue, once per row tile, in f32 on
the accumulators: the row scale, the per-head RMSNorm over head_dim (QK-norm) and its weight, RoPE (the
rotate-half convention: `t * cos + rotate_half(t) * sin`, rotate_half lane rolls within each head and a
sign), and q's 1/sqrt(head_dim) (folded into q's norm weight: a scalar, which RoPE commutes with). The tile
is (rows, all columns of q, k and v), so which projection a column holds is static and the epilogue is
straight-line code for each (column tiles would make it a runtime choice); the tunable picks the rows and
the model dims per step. q, k and v come out as separate (B, S, heads * head_dim) arrays, each head a
block of columns, in the layout the attention kernel reads, and nothing between x and them goes to HBM.
(XLA writes the projections, then reads and writes q and k again to norm and rotate them.)

RoPE's cos and sin are inputs, (S, head_dim) tables of the rows' positions, so the kernel serves any
positions (an offset for decoding, say). Partial RoPE (Qwen3-Next rotates the first quarter of head_dim)
is the same code: the tables hold cos 1 and sin 0 past the rotary dims, and rotate_half rotates only them.
Without QK-norm (Qwen2.5, Llama) the per-head norm is skipped and the weight vectors only scale q.

The rest of each layer is gqa_attention.py's kernels (attention + output projection + residual, then
swiglu.py's feed-forward kernel; or all of it in one kernel, whichever their tuned device times say is
faster for the layer's shapes), and the final norm and the logits are `rmsnorm_matmul`. `baseline` is the
same model in plain JAX with JAX's splash attention.

Run with `uv run python examples/qwen3.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from gqa_attention import make_attention, splash_gqa
from rmsnorm_matmul import rmsnorm_matmul
from swiglu import rms_rows, swiglu_ffn, swiglu_ffn_xla
from transformer import widen

import autokernel

EPS = 1e-6  # Qwen3's RMSNorm epsilon (input norms and QK-norm)
LANES = 128  # TPU vector lanes


@dataclass(frozen=True)
class Config:
    """A Qwen3-like model. `window`: sliding-window attention over that many keys (None: causal);
    `gated`: Qwen3-Next's attention output gate; `rotary_fraction`: the share of head_dim RoPE rotates."""

    layers: int
    d_model: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    rope_theta: float = 1e6
    qk_norm: bool = True
    window: int | None = None
    gated: bool = False
    rotary_fraction: float = 1.0

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_fraction)


# A Config is static: a pytree with no leaves, so `params` can carry it through jax.jit.
jax.tree_util.register_static(Config)


@autokernel.tunable
class ProjectionTiles:
    """Rows per output tile (every column of q, k and v), and model dims per step."""

    rows: int = autokernel.choice(128, 256, 512, 1024, 2048, default=256)
    block_d: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


def rope_tables(positions: jax.Array, head_dim: int, theta: float = 1e6, rotary_dim: int | None = None):
    """RoPE's (len(positions), head_dim) f32 cos and sin tables for the rotate-half convention: lane l of a
    head (l < rotary_dim) turns by positions * theta ** (-2 (l mod rotary_dim/2) / rotary_dim); lanes past
    rotary_dim don't turn (cos 1, sin 0)."""
    rotary_dim = rotary_dim or head_dim
    half = rotary_dim // 2
    inv_freq = theta ** (-jnp.arange(half, dtype=jnp.float32) * 2 / rotary_dim)
    angles = positions.astype(jnp.float32)[:, None] * jnp.concatenate([inv_freq, inv_freq])[None, :]
    still = jnp.zeros((positions.shape[0], head_dim - rotary_dim), jnp.float32)
    cos = jnp.concatenate([jnp.cos(angles), still + 1], axis=1)
    sin = jnp.concatenate([jnp.sin(angles), still], axis=1)
    return cos, sin


def sum_squares(x):
    """Each row's sum of squares of a tile, in f32, (rows, 1)."""
    xf = x.astype(jnp.float32)
    return jnp.sum(xf * xf, axis=1, keepdims=True)


def scale_columns(x, gamma):
    """x * gamma (a (1, cols) weight), computed in f32, in x's dtype: the MXU's operand."""
    return (x.astype(jnp.float32) * gamma.astype(jnp.float32)).astype(x.dtype)


def rotate_half(t, half):
    """RoPE's rotate_half on each row of a (rows, head_dim) head, for its first 2 * half lanes: lane l < half
    gets -t[l + half], lane half <= l < 2 * half gets t[l - half] (lanes past those are don't-cares: their
    sin is 0). Lane rolls: a rotation within the head's vector registers, no lane slicing."""
    lane = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    if 2 * half == t.shape[1]:
        r = jnp.roll(t, half, axis=1)  # by half of the lanes: the same roll either way
        return jnp.where(lane < half, -r, r)
    return jnp.where(lane < half, -jnp.roll(t, -half, axis=1), jnp.roll(t, half, axis=1))


def qk_heads(acc, inv_rms, weight, cos, sin, qk_norm, half):
    """q or k from its accumulated projection `acc` (rows, heads * Dh), f32: each row scaled by its inverse
    RMS (the input RMSNorm), then per head the RMSNorm over its Dh lanes (if `qk_norm`) and the (1, Dh)
    weight, then RoPE with the rows' (rows, Dh) cos and sin."""
    dh = cos.shape[1]
    row_scale = widen(inv_rms, dh)
    out = []
    for h in range(acc.shape[1] // dh):
        t = acc[:, h * dh : (h + 1) * dh] * row_scale
        if qk_norm:
            t = t * jax.lax.rsqrt(jnp.mean(t * t, axis=1, keepdims=True) + EPS)
        t = t * weight.astype(jnp.float32)
        out.append(t * cos + rotate_half(t, half) * sin)
    return jnp.concatenate(out, axis=1) if len(out) > 1 else out[0]


class QkvKernels(NamedTuple):
    """The attention-input kernels for one variant (`make_qkv`)."""

    projection: autokernel.Kernel  # (x, gamma, w_q, w_k, w_v, q_weight, k_weight, cos, sin) -> q, k, v
    gated_projection: autokernel.Kernel  # (..., w_v, w_gate, q_weight, ...) -> q, k, v, gate


@functools.cache
def make_qkv(qk_norm: bool = True, rotary_dim: int | None = None) -> QkvKernels:
    """The fused RMSNorm + QKV projection + QK-norm + RoPE kernels: with QK-norm (Qwen3) or without (Qwen2.5,
    Llama: `q_weight`, `k_weight` then only scale), rotating `rotary_dim` lanes of each head (None: all).

    `q_weight` is q's QK-norm weight times 1/sqrt(head_dim), `k_weight` k's, both (1, head_dim); cos and sin
    are (S, head_dim) tables (`rope_tables`) of the positions of x's rows."""

    @autokernel.kernel
    def qkv_projection(x, gamma, w_q, w_k, w_v, q_weight, k_weight, cos, sin, tiles=ProjectionTiles()):
        B, S, D = x.shape
        half = (rotary_dim or cos.shape[1]) // 2
        HD, KD = w_q.shape[1], w_k.shape[1]
        tm, td = min(tiles.rows, S), min(tiles.block_d, D)
        dtype = x.dtype
        q = jnp.zeros((B, S, HD), dtype)
        k = jnp.zeros((B, S, KD), dtype)
        v = jnp.zeros((B, S, KD), dtype)
        for b in range(B):
            for i in range(0, S, tm):
                acc_q = jnp.zeros((tm, HD), jnp.float32)  # (x * gamma) @ w_q, ...
                acc_k = jnp.zeros((tm, KD), jnp.float32)
                acc_v = jnp.zeros((tm, KD), jnp.float32)
                sum_sq = jnp.zeros((tm, LANES), jnp.float32)  # each row's sum of squares, lane-wide
                for d in range(0, D, td):
                    xd = x[b, i : i + tm, d : d + td]
                    sum_sq += sum_squares(xd)
                    xg = scale_columns(xd, gamma[0:1, d : d + td])
                    acc_q += xg @ w_q[d : d + td, :]
                    acc_k += xg @ w_k[d : d + td, :]
                    acc_v += xg @ w_v[d : d + td, :]
                inv_rms = jax.lax.rsqrt(sum_sq / D + EPS)
                cos_t, sin_t = cos[i : i + tm, :], sin[i : i + tm, :]
                q_t = qk_heads(acc_q, inv_rms, q_weight[0:1, :], cos_t, sin_t, qk_norm, half)
                k_t = qk_heads(acc_k, inv_rms, k_weight[0:1, :], cos_t, sin_t, qk_norm, half)
                q = q.at[b, i : i + tm, :].set(q_t.astype(dtype))
                k = k.at[b, i : i + tm, :].set(k_t.astype(dtype))
                v = v.at[b, i : i + tm, :].set((acc_v * widen(inv_rms, KD)).astype(dtype))
        return q, k, v

    @autokernel.kernel
    def gated_qkv_projection(x, gamma, w_q, w_k, w_v, w_gate, q_weight, k_weight, cos, sin, tiles=ProjectionTiles()):
        """qkv_projection and the attention output gate, sigmoid(rmsnorm(x) @ w_gate)."""
        B, S, D = x.shape
        half = (rotary_dim or cos.shape[1]) // 2
        HD, KD = w_q.shape[1], w_k.shape[1]
        tm, td = min(tiles.rows, S), min(tiles.block_d, D)
        dtype = x.dtype
        q = jnp.zeros((B, S, HD), dtype)
        k = jnp.zeros((B, S, KD), dtype)
        v = jnp.zeros((B, S, KD), dtype)
        gate = jnp.zeros((B, S, HD), dtype)
        for b in range(B):
            for i in range(0, S, tm):
                acc_q = jnp.zeros((tm, HD), jnp.float32)
                acc_k = jnp.zeros((tm, KD), jnp.float32)
                acc_v = jnp.zeros((tm, KD), jnp.float32)
                acc_g = jnp.zeros((tm, HD), jnp.float32)
                sum_sq = jnp.zeros((tm, LANES), jnp.float32)
                for d in range(0, D, td):
                    xd = x[b, i : i + tm, d : d + td]
                    sum_sq += sum_squares(xd)
                    xg = scale_columns(xd, gamma[0:1, d : d + td])
                    acc_q += xg @ w_q[d : d + td, :]
                    acc_k += xg @ w_k[d : d + td, :]
                    acc_v += xg @ w_v[d : d + td, :]
                    acc_g += xg @ w_gate[d : d + td, :]
                inv_rms = jax.lax.rsqrt(sum_sq / D + EPS)
                cos_t, sin_t = cos[i : i + tm, :], sin[i : i + tm, :]
                q_t = qk_heads(acc_q, inv_rms, q_weight[0:1, :], cos_t, sin_t, qk_norm, half)
                k_t = qk_heads(acc_k, inv_rms, k_weight[0:1, :], cos_t, sin_t, qk_norm, half)
                q = q.at[b, i : i + tm, :].set(q_t.astype(dtype))
                k = k.at[b, i : i + tm, :].set(k_t.astype(dtype))
                v = v.at[b, i : i + tm, :].set((acc_v * widen(inv_rms, KD)).astype(dtype))
                g_t = jax.nn.sigmoid(acc_g * widen(inv_rms, HD))
                gate = gate.at[b, i : i + tm, :].set(g_t.astype(dtype))
        return q, k, v, gate

    return QkvKernels(qkv_projection, gated_qkv_projection)


# --------------------------------------------------------------------------- the model


def qk_weights(layer: dict[str, Any], config: Config) -> tuple[jax.Array, jax.Array]:
    """The (1, head_dim) f32 weights the QKV kernel multiplies q's and k's heads by: the QK-norm weights (ones
    without QK-norm), q's times 1/sqrt(head_dim)."""
    if config.qk_norm:
        q_weight, k_weight = layer["q_norm"].astype(jnp.float32), layer["k_norm"].astype(jnp.float32)
    else:
        q_weight = k_weight = jnp.ones((1, config.head_dim), jnp.float32)
    return q_weight * config.head_dim**-0.5, k_weight


def attention_input(x: jax.Array, layer: dict[str, Any], config: Config, cos: jax.Array, sin: jax.Array) -> tuple:
    """(q, k, v) or, with gated attention, (q, k, v, gate) for a layer's input x (B, S, D)."""
    kernels = make_qkv(config.qk_norm, config.rotary_dim)
    q_weight, k_weight = qk_weights(layer, config)
    weights = (layer["w_q"], layer["w_k"], layer["w_v"])
    if config.gated:
        return kernels.gated_projection(
            x, layer["attn_norm"], *weights, layer["w_attn_gate"], q_weight, k_weight, cos, sin
        )
    return kernels.projection(x, layer["attn_norm"], *weights, q_weight, k_weight, cos, sin)


def ffn_weights(layer: dict[str, Any]) -> tuple[jax.Array, ...]:
    """A layer's feed-forward weights, in the order the kernels take them."""
    return layer["ffn_norm"], layer["w_gate"], layer["w_up"], layer["w_down"]


def fused_layer(x: jax.Array, qkv: tuple, layer: dict[str, Any], config: Config) -> jax.Array:
    """The layer after its QKV projection as one kernel: attention, projection, residual, SwiGLU, residual."""
    kernels = make_attention(config.head_dim, config.window)
    kernel = kernels.gated_layer if config.gated else kernels.layer
    return kernel(x, *qkv, layer["wo"], *ffn_weights(layer))


def split_layer(x: jax.Array, qkv: tuple, layer: dict[str, Any], config: Config) -> jax.Array:
    """The layer after its QKV projection as two kernels: attention (+ projection, residual), then SwiGLU."""
    kernels = make_attention(config.head_dim, config.window)
    kernel = kernels.gated_attention if config.gated else kernels.attention
    resid = kernel(x, *qkv, layer["wo"])
    b, s, d = x.shape
    return swiglu_ffn(resid.reshape(b * s, d), *ffn_weights(layer)).reshape(b, s, d)


def fuse_ffn(x: Any, qkv: tuple, layer: dict[str, Any], config: Config) -> bool:
    """Should the feed-forward layer run in the attention kernel (`fused_layer`) rather than a kernel of its
    own (`split_layer`)? Fusing saves a round trip of the activations through HBM, but its row tiles are the
    attention's query blocks, and with big weights a kernel of its own reuses them over more rows: tuning
    measures both (off a TPU nothing is measured: fused). As transformer.py's."""
    kernels = make_attention(config.head_dim, config.window)
    layer_kernel, attention = (
        (kernels.gated_layer, kernels.gated_attention) if config.gated else (kernels.layer, kernels.attention)
    )
    fused = layer_kernel.tuning(x, *qkv, layer["wo"], *ffn_weights(layer))
    if not fused.trials:
        return True
    b, s, d = x.shape
    rows = jax.ShapeDtypeStruct((b * s, d), x.dtype)
    split = (
        attention.tuning(x, *qkv, layer["wo"]).best.seconds + swiglu_ffn.tuning(rows, *ffn_weights(layer)).best.seconds
    )
    return fused.best.seconds <= split


def decoder_layer(x: jax.Array, layer: dict[str, Any], config: Config, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """One layer, x (B, S, D) -> (B, S, D): the QKV kernel, then the rest fused or split, whichever is faster."""
    qkv = attention_input(x, layer, config, cos, sin)
    return (fused_layer if fuse_ffn(x, qkv, layer, config) else split_layer)(x, qkv, layer, config)


def forward(tokens: jax.Array, params: dict[str, Any]) -> jax.Array:
    """The model with autokernel kernels: logits (B, S, vocab) for tokens (B, S) at positions 0 .. S-1."""
    config: Config = params["config"]
    x = params["embed"][tokens]
    B, S, D = x.shape
    cos, sin = rope_tables(jnp.arange(S), config.head_dim, config.rope_theta, config.rotary_dim)
    for layer in params["layers"]:
        x = decoder_layer(x, layer, config, cos, sin)
    return rmsnorm_matmul(x.reshape(B * S, D), params["final_norm"], params["w_vocab"]).reshape(B, S, -1)


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights (and the config, a static pytree node that jit passes through)."""
    d, f, dh = config.d_model, config.ffn, config.head_dim
    hd, kd = config.heads * dh, config.kv_heads * dh
    keys = iter(jax.random.split(jax.random.key(seed), 4 + 13 * config.layers))

    def normal(shape, scale):
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm_weight(n):
        return (1 + 0.1 * jax.random.normal(next(keys), (1, n))).astype(dtype)

    layers = []
    for _ in range(config.layers):
        layer = {
            "attn_norm": norm_weight(d),
            "w_q": normal((d, hd), d**-0.5),
            "w_k": normal((d, kd), d**-0.5),
            "w_v": normal((d, kd), d**-0.5),
            "wo": normal((hd, d), hd**-0.5),
            "ffn_norm": norm_weight(d),
            "w_gate": normal((d, f), d**-0.5),
            "w_up": normal((d, f), d**-0.5),
            "w_down": normal((f, d), f**-0.5),
        }
        if config.qk_norm:
            layer |= {"q_norm": norm_weight(dh), "k_norm": norm_weight(dh)}
        if config.gated:
            layer["w_attn_gate"] = normal((d, hd), d**-0.5)
        layers.append(layer)
    return {
        "config": config,
        "embed": normal((config.vocab, d), 1.0),
        "layers": layers,
        "final_norm": norm_weight(d),
        "w_vocab": normal((d, config.vocab), d**-0.5),
    }


# --------------------------------------------------------------------------- the baseline: plain JAX + splash


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def qk_heads_xla(t: jax.Array, weight: jax.Array, cos: jax.Array, sin: jax.Array, config: Config) -> jax.Array:
    """QK-norm and RoPE on a projection (B, S, heads * Dh) in plain JAX, the Hugging Face way: rotate_half by
    slicing the rotary dims and concatenating."""
    b, s, hd = t.shape
    dh, rd = config.head_dim, config.rotary_dim
    t = t.reshape(b, s, hd // dh, dh).astype(jnp.float32)
    if config.qk_norm:
        t = t * jax.lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + EPS)
    t = t * weight.astype(jnp.float32)
    rot, still = t[..., :rd], t[..., rd:]
    rotated = jnp.concatenate([-rot[..., rd // 2 :], rot[..., : rd // 2]], axis=-1)
    rot = rot * cos[:, None, :rd] + rotated * sin[:, None, :rd]
    return jnp.concatenate([rot, still], axis=-1).reshape(b, s, hd)


def baseline_layer(
    config: Config,
    seq: int,
    block_q: int | None = None,
    block_kv: int | None = None,
    form: str = "mqa",
    interpret: bool = False,
) -> Any:
    """One layer in plain JAX with JAX's splash attention for GQA (`gqa_attention.splash_gqa`: its MQA kernel
    per kv head, or MHA with k and v repeated; its default blocks unless given): `(x, layer, cos, sin) -> x`.
    With `w_in` in the layer (`with_fused_projections`), one matmul makes q, k, v (and the gate)."""
    attend = splash_gqa(config.heads, config.kv_heads, seq, config.window, block_q, block_kv, form, interpret)
    hd, kd = config.heads * config.head_dim, config.kv_heads * config.head_dim

    def layer_fn(x, layer, cos, sin):
        n = rms_rows(x.astype(jnp.float32), layer["attn_norm"]).astype(x.dtype)
        if "w_in" in layer:
            projected = _dot(n, layer["w_in"]).astype(x.dtype)
            q, k, v, gate = jnp.split(projected, [hd, hd + kd, hd + 2 * kd], axis=-1)
        else:
            q, k, v = (_dot(n, layer[w]).astype(x.dtype) for w in ("w_q", "w_k", "w_v"))
            gate = _dot(n, layer["w_attn_gate"]) if config.gated else None
        q_weight, k_weight = qk_weights(layer, config)
        q = qk_heads_xla(q, q_weight, cos, sin, config).astype(x.dtype)
        k = qk_heads_xla(k, k_weight, cos, sin, config).astype(x.dtype)
        a = attend(q, k, v)
        if config.gated:
            a = (a * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(x.dtype)
        x = (x.astype(jnp.float32) + _dot(a, layer["wo"])).astype(x.dtype)
        return swiglu_ffn_xla(x, *ffn_weights(layer))

    return layer_fn


def with_fused_projections(params: dict[str, Any]) -> dict[str, Any]:
    """`params` with each layer's q, k, v (and gate) weights also side by side, `w_in`: one matmul for the
    baseline (made once, outside the timed model)."""
    names = ("w_q", "w_k", "w_v", "w_attn_gate")
    layers = [
        layer | {"w_in": jnp.concatenate([layer[w] for w in names if w in layer], axis=1)} for layer in params["layers"]
    ]
    return params | {"layers": layers}


def baseline(config: Config, seq: int, *args: Any, **kwargs: Any) -> Any:
    """The same model in plain JAX, compiled by XLA, with `baseline_layer`'s layers (same arguments): a jitted
    `(tokens, params) -> logits`."""
    layer_fn = baseline_layer(config, seq, *args, **kwargs)

    def model(tokens, params):
        x = params["embed"][tokens]
        cos, sin = rope_tables(jnp.arange(x.shape[1]), config.head_dim, config.rope_theta, config.rotary_dim)
        for layer in params["layers"]:
            x = layer_fn(x, layer, cos, sin)
        n = rms_rows(x.astype(jnp.float32), params["final_norm"]).astype(x.dtype)
        return _dot(n, params["w_vocab"]).astype(x.dtype)

    return jax.jit(model)


if __name__ == "__main__":
    config = Config(layers=2, d_model=256, heads=4, kv_heads=2, head_dim=128, ffn=512, vocab=512)
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (1, 256), 0, config.vocab)
    logits = jax.jit(forward)(tokens, params)  # tuned on a TPU, the default tiles elsewhere
    x = params["embed"][tokens]
    cos, sin = rope_tables(jnp.arange(256), config.head_dim, config.rope_theta)
    layer = params["layers"][0]
    args = (x, layer["attn_norm"], layer["w_q"], layer["w_k"], layer["w_v"], *qk_weights(layer, config), cos, sin)
    projection = make_qkv().projection
    print(projection.source(*args))
    print("QKV projection: max |kernel - loop nest|:", projection.verify(*args))
    print(f"logits {logits.shape}")
    if jax.default_backend() == "tpu":
        reference = baseline(config, 256)(tokens, params)
        print(
            "max |autokernel - splash + XLA|:", float(jnp.max(jnp.abs(logits.astype(float) - reference.astype(float))))
        )
