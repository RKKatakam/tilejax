"""The layers of hybrid linear/full-attention models (Qwen3-Next, Kimi Linear) for prefill and decode, as
autokernel kernels composed in plain JAX; hybrid_models.py stacks them into whole models.

Four token mixers and one feed-forward layer, each with a prefill form (a whole prompt, (B, S, d_model),
returning what decoding needs: a recurrent state or a KV cache) and a decode form (one new token per
sequence, (B, d_model), updating that state):

    Gated DeltaNet (Qwen3-Next's linear attention)   linear_attention_layer.py's layer: norm + projections,
                                                    short convolution, gates, delta rule, gated RMSNorm,
                                                    output projection, residual
    KDA (Kimi Linear's linear attention)             the same with a decay per key channel from a low-rank
                                                    projection, f = softplus(h W_fa W_fb + dt_bias), and an
                                                    output gate sigmoid(h W_ga W_gb) (rank head_dim each)
    gated GQA (Qwen3-Next's full attention)          qwen3.py's gated QKV kernel (QK-norm, partial RoPE)
                                                    and gqa_attention.py's gated attention kernel
    MLA (DeepSeek-V3 / Kimi's full attention)        mla.py's kernels
    MoE + shared expert                             moe.py's grouped `expert_ffn` kernel, softmax (Qwen) or
                                                    sigmoid (Kimi) router; Qwen3-Next's shared expert is
                                                    scaled by a gate of its own (`gated_shared_experts`)

**The delta rule with its final state** (`gdn_prefill`, `kda_prefill`): linear_attention.py's chunked
kernels compute a prompt's outputs; decoding also needs every head's state after its last token. These
kernels are the same chunk math (`linear_attention.delta_block`), one tile per (batch, group of heads), one
reduction loop over blocks of tokens carrying the heads' states in VMEM, with two differences: each step
stores its block of outputs straight to HBM (a per-step output tile: the copy overlaps the next steps, and
the sequence needs no carried buffers), and the tile ends by storing its heads' final states, side by side
as (d_v, heads * d_k) columns of a (B, d_v, H * d_k) f32 output. The output gate (Qwen3-Next: RMSNorm per
head * weight * silu(z); KDA: * sigmoid(z)) is fused at the end of each step.

**Decoding a linear layer** is the recurrence for one token, in plain JAX, per head (S: d_k x d_v, f32):

    S' = Diag(a) S;  u = beta (v - S'^T k);  S'' = S' + k u^T;  o = S''^T q = S^T (a q) + (k . q) u

so one batched matrix-vector product reads S for both S^T (a k) and S^T (a q), and one elementwise pass
writes S'': the state (64 MiB per layer at batch 32, 32 heads) is read twice and written once per token,
bandwidth-bound and exact in f32 (HIGHEST precision: TPU f32 matmuls are one bf16 pass by default). The
short convolution keeps its last CONV - 1 = 3 inputs per sequence (the "conv tail").

**Decoding a full-attention layer** appends the token's key and value (GQA) or latent and RoPE key (MLA)
to the layer's cache and attends with decode.py's `gqa_decode` / `mla_decode` (runtime length masks: only
the valid blocks of each cache are read). The projection kernels serve one token per sequence by taking
the batch as the rows of one sequence, (1, B, d_model), with RoPE tables of each row's own position.

**Baselines**: every layer also runs as plain JAX (`Impl(kernels=False)`): XLA's RMSNorm and matmuls, the
chunked delta rule in plain JAX (`chunked_with_state`: linear_attention.baseline_chunked, also returning
the state), JAX's splash attention (or dense XLA attention), `jax.lax.ragged_dot` for the experts, and
XLA's attention over the whole cache for decoding. The glue (gates, convolution, routing, dispatch) is the
same code in both, so a comparison measures the kernels.

Run with `uv run python examples/hybrid_models.py`; without a TPU the kernels run in the Pallas TPU
interpreter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from decode import absorb_queries, gqa_decode, mla_decode, xla_gqa_decode, xla_mla_decode
from gqa_attention import make_attention, splash_gqa
from kv_cache import append, pad_lanes
from linear_attention import (
    CHUNK,
    HEAD_DIM,
    KDA_GATE_MIN,
    NORM_EPS,
    DeltaTiles,
    chunk_output,
    chunk_update,
    delta_block,
    gated_rmsnorm,
    gdn_intra,
    kda_intra,
    l2_normalize,
)
from linear_attention_layer import CONV, GATES, prepare, residual_matmul
from linear_attention_layer import Config as LinearConfig
from matmul import matmul
from mla import LANES, NOPE, ROPE, THETA, MlaConfig, mla_attention, mla_down, pack
from mla import init_params as mla_init_params
from mla import rope_tables as mla_rope_tables
from moe import (
    DECODE_TOKENS,
    EPS,
    MoeConfig,
    SharedTiles,
    combine,
    decode_call,
    default_group,
    dispatch,
    expert_ffn,
    expert_inputs,
    gather,
    moe_layer,
    ragged_rows,
    rmsnorm,
    route,
)
from moe import init_params as moe_init_params
from qwen3 import Config as Qwen3Config
from qwen3 import attention_input, qk_heads_xla, qk_weights, rope_tables
from rmsnorm_matmul import rmsnorm_matmul, rmsnorm_then_matmul
from transformer import widen

import autokernel

LOW_RANK = HEAD_DIM  # KDA: the rank of the decay and output-gate projections (Kimi Linear's: head_dim)
MASK_VALUE = -1e30


@dataclass(frozen=True)
class HybridConfig:
    """A hybrid model: `layers` layers, each a token mixer then a MoE feed-forward layer; every `full_every`-th
    layer's mixer is full attention, the others linear attention (3:1 for full_every 4).

    kind "qwen3-next": Gated DeltaNet (k_heads query/key heads, v_heads value heads, HEAD_DIM each) and gated
    GQA (heads, kv_heads of head_dim, QK-norm, RoPE on rotary_fraction of head_dim), Qwen's MoE with a gated
    shared expert. kind "kimi-linear": KDA (v_heads heads) and MLA (heads, q_rank, kv_rank), a sigmoid
    router and a shared expert."""

    kind: str
    layers: int
    d_model: int
    vocab: int
    k_heads: int
    v_heads: int
    heads: int
    moe: MoeConfig
    kv_heads: int = 1
    head_dim: int = 128
    rotary_fraction: float = 0.25
    rope_theta: float = 1e7  # Qwen3-Next's
    q_rank: int = 1024
    kv_rank: int = 512
    full_every: int = 4

    def is_full(self, layer: int) -> bool:
        """Is layer `layer`'s mixer full attention?"""
        return layer % self.full_every == self.full_every - 1

    @property
    def linear(self) -> LinearConfig:
        """The linear-attention layer's dims (linear_attention_layer.Config)."""
        return LinearConfig(self.d_model, self.k_heads, self.v_heads)

    @property
    def attention(self) -> Qwen3Config:
        """The gated GQA layer's dims (qwen3.Config: its QKV kernel and plain-JAX QK-norm + RoPE read them)."""
        return Qwen3Config(
            layers=self.layers,
            d_model=self.d_model,
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            ffn=self.moe.ffn,
            vocab=self.vocab,
            rope_theta=self.rope_theta,
            gated=True,
            rotary_fraction=self.rotary_fraction,
        )


# A config is static: a pytree with no leaves, so params can carry it through jax.jit.
jax.tree_util.register_static(HybridConfig)


@dataclass(frozen=True)
class Impl:
    """How layers run: autokernel's kernels, or plain JAX (`kernels=False`, the baseline) with JAX's splash
    attention for prefill (`attention="splash"`, in its interpreter if `interpret`) or dense XLA attention
    ("xla"), and the experts by `jax.lax.ragged_dot` ("ragged_dot") or every expert on every token, one-hot
    weighted ("dense")."""

    kernels: bool = True
    attention: str = "splash"
    experts: str = "ragged_dot"
    interpret: bool = False


AUTOKERNEL = Impl()
jax.tree_util.register_static(Impl)


# --------------------------------------------------------------------------- kernels


def sigmoid_gated_rmsnorm(o, z, weight, eps=NORM_EPS):
    """Kimi Linear's output gate on (rows, heads * HEAD_DIM): each head of `o` RMS-normalized, times `weight`
    (1, HEAD_DIM), times sigmoid of the gate `z`; f32."""
    heads = []
    for lanes in range(0, o.shape[1], HEAD_DIM):
        oh = o[:, lanes : lanes + HEAD_DIM].astype(jnp.float32)
        zh = z[:, lanes : lanes + HEAD_DIM].astype(jnp.float32)
        normed = oh * jax.lax.rsqrt(jnp.mean(oh * oh, axis=1, keepdims=True) + eps)
        heads.append(normed * weight.astype(jnp.float32) * jax.nn.sigmoid(zh))
    return jnp.concatenate(heads, axis=1)


def state_columns(state):
    """A group's states (G, d_v, d_k) side by side, (d_v, G * d_k): the tile's columns of the state output."""
    return jnp.concatenate([state[h] for h in range(state.shape[0])], axis=1) if state.shape[0] > 1 else state[0]


def gdn_step(q, k, v, g, beta, z, weight, col, state):
    """A step of `gdn_prefill`: a block of tokens for a group of heads (linear_attention.delta_block), its
    outputs through Qwen3-Next's gated RMSNorm; and the heads' states after it."""
    o, state = delta_block(gdn_intra, q, k, v, g, beta, col, state)
    return gated_rmsnorm(o, z, weight), state


def kda_step(q, k, v, g, beta, z, weight, col, state):
    """A step of `kda_prefill`: as `gdn_step`, KDA's chunks and Kimi Linear's sigmoid output gate."""
    o, state = delta_block(kda_intra, q, k, v, g, beta, col, state)
    return sigmoid_gated_rmsnorm(o, z, weight), state


@autokernel.kernel
def gdn_prefill(q, k, v, g, beta, z, weight, tiles=DeltaTiles()):
    """Gated DeltaNet over whole prompts, with Qwen3-Next's output gate: q, k, v (B, S, H * HEAD_DIM) (q and
    k not yet normalized), log decays g and betas (B, H, S) f32, the gate z (B, S, >= H * HEAD_DIM: its
    first columns), weight (1, HEAD_DIM). The gated outputs (B, S, H * HEAD_DIM), and every head's final
    state, transposed, as columns: (B, d_v, H * d_k) f32."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    T = min(tiles.block, S)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    final = jnp.zeros((B, HEAD_DIM, D), jnp.float32)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)  # per head, transposed: (d_v, d_k)
            for c in range(0, S, T):  # a block of tokens per step: its outputs stored by the step
                qt, kt, vt = q[b, c : c + T, j : j + W], k[b, c : c + T, j : j + W], v[b, c : c + T, j : j + W]
                gt, bt, zt = g[b, :, c : c + T], beta[b, :, c : c + T], z[b, c : c + T, j : j + W]
                o, state = gdn_step(qt, kt, vt, gt, bt, zt, weight[:, :], j, state)
                out = out.at[b, c : c + T, j : j + W].set(o.astype(dtype))
            final = final.at[b, :, j : j + W].set(state_columns(state))
    return out, final


@autokernel.kernel
def kda_prefill(q, k, v, g, beta, z, weight, tiles=DeltaTiles()):
    """KDA over whole prompts, with Kimi Linear's output gate: as `gdn_prefill`, with g (B, S, H * HEAD_DIM)
    f32, log decays per key channel (clamped to [KDA_GATE_MIN, 0]), and z like v."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    T = min(tiles.block, S)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    final = jnp.zeros((B, HEAD_DIM, D), jnp.float32)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            for c in range(0, S, T):
                qt, kt, vt = q[b, c : c + T, j : j + W], k[b, c : c + T, j : j + W], v[b, c : c + T, j : j + W]
                gt, bt, zt = g[b, c : c + T, j : j + W], beta[b, :, c : c + T], z[b, c : c + T, j : j + W]
                o, state = kda_step(qt, kt, vt, gt, bt, zt, weight[:, :], j, state)
                out = out.at[b, c : c + T, j : j + W].set(o.astype(dtype))
            final = final.at[b, :, j : j + W].set(state_columns(state))
    return out, final


@autokernel.kernel
def gated_shared_experts(x, gamma, w_gate, w_up, w_down, w_shared_gate, routed, tiles=SharedTiles()):
    """The end of Qwen3-Next's MoE layer on (tokens, d_model) rows: x + routed + sigmoid(n @ w_g) * SwiGLU(n),
    n = rmsnorm(x). The gate's weight column comes repeated in 128 lanes, `w_shared_gate` (d_model, 128):
    the MXU computes 128 columns either way, and the gate comes out lane-wide."""
    T, D = x.shape
    F = w_gate.shape[1]
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        resid = x[i : i + tm, :].astype(jnp.float32)
        normed = resid * jax.lax.rsqrt(jnp.mean(resid * resid, axis=1, keepdims=True) + EPS) * gamma[0:1, :]
        normed = normed.astype(dtype)
        gate = jax.nn.sigmoid(normed @ w_shared_gate[:, :])  # (tm, 128), every lane the same
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):  # the shared expert's hidden units, bf per step
            hidden = jax.nn.silu(normed @ w_gate[:, f : f + bf]) * (normed @ w_up[:, f : f + bf])
            y += hidden.astype(dtype) @ w_down[f : f + bf, :]
        routed_t = routed[i : i + tm, :].astype(jnp.float32)
        out = out.at[i : i + tm, :].set((resid + routed_t + widen(gate, D) * y).astype(dtype))
    return out


# --------------------------------------------------------------------------- plain JAX pieces


def _bf16(x: jax.Array) -> jax.Array:
    return x.astype(jnp.bfloat16)


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def norm_matmul(x: jax.Array, gamma: jax.Array, w: jax.Array, impl: Impl) -> jax.Array:
    """rmsnorm(x) @ w on (rows, d_model): `rmsnorm_matmul`, or XLA."""
    return rmsnorm_matmul(x, gamma, w) if impl.kernels else rmsnorm_then_matmul(x, gamma, w)


def residual_proj(x: jax.Array, a: jax.Array, w: jax.Array, impl: Impl) -> jax.Array:
    """x + a @ w (an output projection and the residual): `residual_matmul`, or XLA."""
    if impl.kernels:
        return residual_matmul(x, a, w)
    return (x.astype(jnp.float32) + _dot(a, w)).astype(x.dtype)


def short_conv(pre: jax.Array, w: jax.Array, tail: jax.Array | None = None) -> jax.Array:
    """The causal depthwise convolution and SiLU over tokens (Qwen3-Next's and Kimi Linear's short conv):
    pre (B, T, C), its weights w (CONV, C) (tap j weighs token t - CONV + 1 + j); `tail` (B, CONV - 1, C)
    holds the inputs before pre's first token (zeros if None). f32 math, in pre's dtype."""
    t = pre.shape[1]
    before = jnp.zeros((pre.shape[0], CONV - 1, pre.shape[2]), pre.dtype) if tail is None else tail.astype(pre.dtype)
    padded = jnp.concatenate([before, pre], axis=1).astype(jnp.float32)
    wf = w.astype(jnp.float32)
    return jax.nn.silu(sum(wf[j] * padded[:, j : j + t] for j in range(CONV))).astype(pre.dtype)


def delta_step(q, k, v, decay, beta, state):
    """One token of the delta rule for every sequence and head, f32: q, k (normalized; q scaled), v
    (B, H, d), decays (B, H, d_k), or (B, H, 1) for one per head, betas (B, H), states (B, H, d_k, d_v).
    The outputs (B, H, d_v) and the states after the token (see the module docstring)."""
    highest = jax.lax.Precision.HIGHEST
    both = jnp.stack([decay * k, decay * q], axis=-1)  # (B, H, d_k, 2)
    read = jnp.einsum("bhkv,bhkn->bhnv", state, both, precision=highest)  # S^T (a k), S^T (a q)
    u = beta[..., None] * (v - read[:, :, 0])
    out = read[:, :, 1] + jnp.sum(k * q, axis=-1, keepdims=True) * u
    return out, decay[..., None] * state + k[..., :, None] * u[..., None, :]


def states_from_columns(final: jax.Array) -> jax.Array:
    """The delta-rule kernels' final states (B, d_v, H * d_k) as (B, H, d_k, d_v)."""
    b, dv, d = final.shape
    return final.reshape(b, dv, d // HEAD_DIM, HEAD_DIM).transpose(0, 2, 3, 1)


@jax.jit
def chunked_with_state(q, k, v, g, beta):
    """linear_attention.baseline_chunked, also returning the final states: the chunked delta rule in plain
    JAX (Gated DeltaNet for g (B, H, S), KDA for g like v). Outputs (B, S, H * HEAD_DIM) in v's dtype, states
    (B, H, d_k, d_v) f32."""
    B, S, D = v.shape
    H, n = D // HEAD_DIM, S // CHUNK

    def chunks(x):  # (B, S, H * d) -> (n, B, H, C, d)
        return x.reshape(B, n, CHUNK, H, HEAD_DIM).transpose(1, 0, 3, 2, 4)

    def rows(x):  # (B, H, S) -> (n, B, H, 1, C)
        return x.reshape(B, H, n, 1, CHUNK).transpose(2, 0, 1, 3, 4)

    def normalized(x, scale=1.0):
        return _bf16(l2_normalize(x.reshape(B, S, H, HEAD_DIM), scale).reshape(B, S, D))

    qc, kc, vc = chunks(normalized(q, HEAD_DIM**-0.5)), chunks(normalized(k)), chunks(_bf16(v))
    beta_rows = rows(jnp.clip(beta, 0.0, 1.0))
    if g.shape == v.shape:
        intra = kda_intra(qc, kc, vc, chunks(jnp.clip(g.astype(jnp.float32), KDA_GATE_MIN, 0.0)), beta_rows)
    else:
        intra = gdn_intra(qc, kc, vc, rows(jnp.minimum(g, 0.0)), beta_rows)
    w, u, q_gamma, qk, k_hat, decay_last = intra

    def step(state, chunk):
        u, after = chunk_update(*chunk, state)
        return after, (u, _bf16(state))

    state = jnp.zeros((B, H, HEAD_DIM, HEAD_DIM), jnp.float32)
    final, (u, states) = jax.lax.scan(step, state, (w, u, k_hat, decay_last))
    out = chunk_output(q_gamma, qk, u, states)  # (n, B, H, C, d)
    return out.transpose(1, 0, 3, 2, 4).reshape(B, S, D).astype(v.dtype), jnp.swapaxes(final, -1, -2)


def _heads(x: jax.Array, heads: int) -> jax.Array:
    """(B, heads * d) -> (B, heads, d)."""
    return x.reshape(x.shape[0], heads, -1)


def linear_decode_core(q, k, v, decay, beta, cache):
    """Normalize q and k (B, H, HEAD_DIM), clamp decays and betas as the kernels do, and run `delta_step` on
    the cached state: the outputs (B, H * HEAD_DIM) f32 and the new state."""
    q = l2_normalize(q, HEAD_DIM**-0.5)
    k = l2_normalize(k)
    o, state = delta_step(q, k, v.astype(jnp.float32), decay, jnp.clip(beta, 0.0, 1.0), cache["state"])
    return o.reshape(o.shape[0], -1), state


# --------------------------------------------------------------------------- Gated DeltaNet (Qwen3-Next)


def gdn_prefill_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, impl: Impl) -> tuple[jax.Array, dict]:
    """The Gated DeltaNet layer (linear_attention_layer.py's) on prompts x (B, S, d_model): x + its output,
    and its decode state {"state": (B, v_heads, d_k, d_v) f32, "conv": the last CONV - 1 conv inputs}."""
    B, S, D = x.shape
    lin = config.linear
    rows = x.reshape(B * S, D)
    proj = norm_matmul(rows, p["norm"], p["w_qkvz"], impl).reshape(B, S, -1)
    gates = norm_matmul(rows, p["norm"], p["w_ba"], impl).reshape(B, S, GATES)
    q, k, v, g, beta = prepare(proj, gates, p, lin)
    if impl.kernels:
        o, final = gdn_prefill(q, k, v, g, beta, proj, p["out_norm"])  # z: proj's first value_dim columns
        state = states_from_columns(final)
    else:
        o, state = chunked_with_state(q, k, v, g, beta)
        z = proj[..., : lin.value_dim].reshape(B * S, -1)
        o = gated_rmsnorm(o.reshape(B * S, -1), z, p["out_norm"]).astype(x.dtype)
    out = residual_proj(rows, o.reshape(B * S, lin.value_dim), p["w_out"], impl).reshape(B, S, D)
    return out, {"state": state, "conv": proj[:, S - (CONV - 1) :, lin.value_dim :]}


def gdn_decode_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, cache: dict, impl: Impl):
    """The Gated DeltaNet layer for one new token per sequence, x (B, d_model): x + its output, and the
    updated state."""
    lin = config.linear
    vd, kd, hv, r = lin.value_dim, lin.key_dim, lin.v_heads, lin.v_heads // lin.k_heads
    proj = norm_matmul(x, p["norm"], p["w_qkvz"], impl)
    gates = norm_matmul(x, p["norm"], p["w_ba"], impl).astype(jnp.float32)
    pre = proj[:, None, vd:]
    conv = short_conv(pre, p["conv"], cache["conv"])[:, 0]
    q, k = (jnp.repeat(_heads(t, lin.k_heads), r, axis=1) for t in (conv[:, :kd], conv[:, kd : 2 * kd]))
    v = _heads(conv[:, 2 * kd :], hv)
    beta = jax.nn.sigmoid(gates[:, :hv])
    g = -jnp.exp(p["A_log"]) * jax.nn.softplus(gates[:, hv : 2 * hv] + p["dt_bias"])
    decay = jnp.exp(jnp.minimum(g, 0.0))[..., None]  # (B, H, 1): one per head
    o, state = linear_decode_core(q, k, v, decay, beta, cache)
    gated = gated_rmsnorm(o, proj[:, :vd], p["out_norm"]).astype(x.dtype)
    out = residual_proj(x, gated, p["w_out"], impl)
    conv_tail = jnp.concatenate([cache["conv"], pre.astype(cache["conv"].dtype)], axis=1)[:, 1:]
    return out, {"state": state, "conv": conv_tail}


# --------------------------------------------------------------------------- KDA (Kimi Linear)


def kda_gates(low: jax.Array, p: dict[str, Any], heads: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """From the low-rank projection [h W_fa | h W_ga | h W_b] (..., 3 * LOW_RANK): the log decays per key
    channel (..., heads * HEAD_DIM) f32, -exp(A_log) softplus(h W_fa W_fb + dt_bias), clamped to
    [KDA_GATE_MIN, 0] as the kernel clamps them; the output gate h W_ga W_gb (..., heads * HEAD_DIM) in
    low's dtype; the betas sigmoid(h W_b) (..., heads) f32."""
    f = _dot(low[..., :LOW_RANK], p["f_b"])
    a = jnp.repeat(jnp.exp(p["A_log"]), HEAD_DIM)
    g = jnp.clip(-a * jax.nn.softplus(f + p["dt_bias"]), KDA_GATE_MIN, 0.0)
    z = _dot(low[..., LOW_RANK : 2 * LOW_RANK], p["g_b"]).astype(low.dtype)
    beta = jax.nn.sigmoid(low[..., 2 * LOW_RANK : 2 * LOW_RANK + heads].astype(jnp.float32))
    return g, z, beta


def kda_prefill_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, impl: Impl) -> tuple[jax.Array, dict]:
    """Kimi Linear's KDA layer on prompts x (B, S, d_model): x + its output, and its decode state."""
    B, S, D = x.shape
    H = config.v_heads
    rows = x.reshape(B * S, D)
    proj = norm_matmul(rows, p["norm"], p["w_qkv"], impl).reshape(B, S, -1)
    low = norm_matmul(rows, p["norm"], p["w_low"], impl).reshape(B, S, -1)
    q, k, v = jnp.split(short_conv(proj, p["conv"]), 3, axis=-1)
    g, z, beta = kda_gates(low, p, H)
    beta = beta.transpose(0, 2, 1)
    if impl.kernels:
        o, final = kda_prefill(q, k, v, g, beta, z, p["out_norm"])
        state = states_from_columns(final)
    else:
        o, state = chunked_with_state(q, k, v, g, beta)
        o = sigmoid_gated_rmsnorm(o.reshape(B * S, -1), z.reshape(B * S, -1), p["out_norm"]).astype(x.dtype)
    out = residual_proj(rows, o.reshape(B * S, -1), p["w_out"], impl).reshape(B, S, D)
    return out, {"state": state, "conv": proj[:, S - (CONV - 1) :]}


def kda_decode_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, cache: dict, impl: Impl):
    """The KDA layer for one new token per sequence, x (B, d_model): x + its output, and the updated state."""
    H = config.v_heads
    proj = norm_matmul(x, p["norm"], p["w_qkv"], impl)
    low = norm_matmul(x, p["norm"], p["w_low"], impl)
    conv = short_conv(proj[:, None], p["conv"], cache["conv"])[:, 0]
    q, k, v = (_heads(t, H) for t in jnp.split(conv, 3, axis=-1))
    g, z, beta = kda_gates(low, p, H)
    o, state = linear_decode_core(q, k, v, _heads(jnp.exp(g), H), beta, cache)
    gated = sigmoid_gated_rmsnorm(o, z, p["out_norm"]).astype(x.dtype)
    out = residual_proj(x, gated, p["w_out"], impl)
    conv_tail = jnp.concatenate([cache["conv"], proj[:, None].astype(cache["conv"].dtype)], axis=1)[:, 1:]
    return out, {"state": state, "conv": conv_tail}


# --------------------------------------------------------------------------- gated GQA (Qwen3-Next)


def dense_gqa(q: jax.Array, k: jax.Array, v: jax.Array, heads: int, kv_heads: int) -> jax.Array:
    """Causal GQA attention in plain XLA over the whole (S, S) scores, (B, S, heads * Dh) layout; q
    pre-scaled."""
    b, s, hd = q.shape
    dh = hd // heads
    qg = q.reshape(b, s, kv_heads, heads // kv_heads, dh)
    kh, vh = (t.reshape(b, s, kv_heads, dh) for t in (k, v))
    scores = jnp.einsum("bqgrd,bkgd->bgrqk", qg, kh, preferred_element_type=jnp.float32)
    causal = jnp.tril(jnp.ones((s, s), bool))
    p = jax.nn.softmax(jnp.where(causal, scores, MASK_VALUE), axis=-1).astype(v.dtype)
    out = jnp.einsum("bgrqk,bkgd->bqgrd", p, vh, preferred_element_type=jnp.float32)
    return out.reshape(b, s, hd).astype(q.dtype)


def gqa_inputs(x, p, config: HybridConfig, cos, sin, impl: Impl) -> tuple[jax.Array, ...]:
    """q (pre-scaled), k, v and the output gate (B, S, heads * head_dim) for rows x (B, S, d_model) at the
    positions of cos and sin (S, head_dim): qwen3.py's gated QKV kernel, or the same in plain JAX."""
    acfg = config.attention
    if impl.kernels:
        return attention_input(x, p, acfg, cos, sin)
    n = rmsnorm(x, p["attn_norm"])
    q, k, v = (_dot(n, p[w]).astype(x.dtype) for w in ("w_q", "w_k", "w_v"))
    gate = jax.nn.sigmoid(_dot(n, p["w_attn_gate"])).astype(x.dtype)
    q_weight, k_weight = qk_weights(p, acfg)
    q = qk_heads_xla(q, q_weight, cos, sin, acfg).astype(x.dtype)
    k = qk_heads_xla(k, k_weight, cos, sin, acfg).astype(x.dtype)
    return q, k, v, gate


def gqa_prefill_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, impl: Impl, max_len: int):
    """Qwen3-Next's gated attention layer on prompts x (B, S, d_model): x + its output, and the KV cache
    {"k", "v"} (B, max_len, kv_heads * head_dim) holding the prompts' keys and values."""
    S = x.shape[1]
    cos, sin = rope_tables(jnp.arange(S), config.head_dim, config.rope_theta, config.attention.rotary_dim)
    q, k, v, gate = gqa_inputs(x, p, config, cos, sin, impl)
    if impl.kernels:
        out = make_attention(config.head_dim).gated_attention(x, q, k, v, gate, p["wo"])
    else:
        if impl.attention == "splash":
            a = splash_gqa(config.heads, config.kv_heads, S, interpret=impl.interpret)(q, k, v)
        else:
            a = dense_gqa(q, k, v, config.heads, config.kv_heads)
        a = (a.astype(jnp.float32) * gate.astype(jnp.float32)).astype(x.dtype)
        out = (x.astype(jnp.float32) + _dot(a, p["wo"])).astype(x.dtype)
    pad = ((0, 0), (0, max_len - S), (0, 0))
    return out, {"k": jnp.pad(k, pad), "v": jnp.pad(v, pad)}


def gqa_decode_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, cache: dict, lengths, impl: Impl):
    """The gated attention layer for one new token per sequence (x (B, d_model), at positions `lengths`):
    x + its output, and the cache with the token's key and value appended."""
    B = x.shape[0]
    cos, sin = rope_tables(lengths, config.head_dim, config.rope_theta, config.attention.rotary_dim)
    q, k, v, gate = gqa_inputs(x[None], p, config, cos, sin, impl)  # the batch as one sequence's rows
    cache = {"k": append(cache["k"], lengths, k[0][:, None]), "v": append(cache["v"], lengths, v[0][:, None])}
    q = q[0].reshape(B, 1, config.heads, config.head_dim)
    attend = gqa_decode if impl.kernels else xla_gqa_decode
    attn = attend(q, cache["k"], cache["v"], lengths + 1)
    a = (attn.reshape(B, -1).astype(jnp.float32) * gate[0].astype(jnp.float32)).astype(x.dtype)
    return residual_proj(x, a, p["wo"], impl), cache


# --------------------------------------------------------------------------- MLA (Kimi Linear's full attention)


def rope_at(t: jax.Array, positions: jax.Array) -> jax.Array:
    """mla.py's rotate-half RoPE on the last dim (64) of t (B, T, ..., 64) at positions (B, T); f32."""
    inv_freq = THETA ** (-jnp.arange(0, ROPE, 2, dtype=jnp.float32) / ROPE)
    angles = positions.astype(jnp.float32)[..., None] * inv_freq
    angles = jnp.concatenate([angles, angles], axis=-1)
    angles = angles.reshape(angles.shape[:-1] + (1,) * (t.ndim - angles.ndim) + (ROPE,))
    t = t.astype(jnp.float32)
    rotated = jnp.concatenate([-t[..., ROPE // 2 :], t[..., : ROPE // 2]], axis=-1)
    return t * jnp.cos(angles) + rotated * jnp.sin(angles)


def mla_tables_at(positions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`mla.rope_tables` for rows at `positions` (n,): cos and sin (n, 128)."""
    inv_freq = THETA ** (-jnp.arange(0, ROPE, 2, dtype=jnp.float32) / ROPE)
    angles = positions.astype(jnp.float32)[:, None] * jnp.tile(inv_freq, LANES // (ROPE // 2))[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def rope_slot(k_slot: jax.Array) -> jax.Array:
    """mla_down's rope key slot [k_rope | k_rope] (..., 128) as the decode cache stores it, [k_rope | 0]."""
    return jnp.where(jnp.arange(LANES) < ROPE, k_slot, 0)


def mla_project_xla(x: jax.Array, raw: dict[str, jax.Array], positions: jax.Array) -> tuple[jax.Array, ...]:
    """mla.latents at any positions (B, T), in plain JAX: q_nope (B, T, H, 128) and q_rope (B, T, H, 64)
    (rotated, unscaled), c_kv (B, T, kv_rank), k_rope (B, T, 64) (rotated), in x's dtype."""
    dtype = x.dtype
    h = rmsnorm(x, raw["attn_norm"])
    c_q = rmsnorm(_dot(h, raw["w_dq"]), raw["q_norm"]).astype(dtype)
    q = jnp.einsum("btr,rhe->bthe", c_q, raw["w_uq"], preferred_element_type=jnp.float32)
    q_rope = rope_at(q[..., NOPE:].astype(dtype), positions).astype(dtype)
    c_kv = rmsnorm(_dot(h, raw["w_dkv"]), raw["kv_norm"]).astype(dtype)
    k_rope = rope_at(_dot(h, raw["w_kr"]).astype(dtype), positions).astype(dtype)
    return q[..., :NOPE].astype(dtype), q_rope, c_kv, k_rope


def mla_attend_xla(q_nope, q_rope, k_nope, k_rope, v, impl: Impl) -> jax.Array:
    """Causal MLA attention (expanded form) in plain JAX: splash attention (192-wide q and k, the rope key
    copied to every head) or dense XLA; (B, S, H, 128)."""
    b, s, heads, _ = q_nope.shape
    scale = (NOPE + ROPE) ** -0.5
    q = (jnp.concatenate([q_nope, q_rope], -1).astype(jnp.float32) * scale).astype(q_nope.dtype)
    k = jnp.concatenate([k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, s, heads, ROPE))], -1)
    if impl.attention == "splash":
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

        mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((s, s)) for _ in range(heads)])
        attend = jax.vmap(splash.make_splash_mha_single_device(mask, interpret=impl.interpret))
        out = attend(*(t.transpose(0, 2, 1, 3) for t in (q, k, v)))  # (B, H, S, 128)
        return out.transpose(0, 2, 1, 3)
    scores = jnp.einsum("bqhe,bkhe->bhqk", q, k, preferred_element_type=jnp.float32)
    causal = jnp.tril(jnp.ones((s, s), bool))
    p = jax.nn.softmax(jnp.where(causal, scores, MASK_VALUE), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhe->bqhe", p, v, preferred_element_type=jnp.float32).astype(v.dtype)


def mla_prefill_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, impl: Impl, max_len: int):
    """The MLA layer on prompts x (B, S, d_model): x + its output, and the latent cache {"c_kv":
    (B, max_len, kv_rank), "k_rope": (B, max_len, 128) as [k_rope | 0]} (decode.py's layout)."""
    B, S, _ = x.shape
    if impl.kernels:
        pk = p["packed"]
        cos, sin = mla_rope_tables(S)
        c_q, c_kv, k_slot = mla_down(x, pk["attn_norm"], pk["w_down"], pk["q_norm"], pk["kv_norm"], cos, sin)
        q = matmul(c_q.reshape(B * S, -1), pk["w_uq"]).reshape(B, S, -1)
        kv = matmul(c_kv.reshape(B * S, -1), pk["w_ukv"]).reshape(B, S, -1)
        out = mla_attention(x, q, kv, k_slot, cos, sin, pk["w_o"])
        k_rope = rope_slot(k_slot)
    else:
        raw = p["raw"]
        q_nope, q_rope, c_kv, k_r = mla_project_xla(x, raw, jnp.broadcast_to(jnp.arange(S), (B, S)))
        k_nope = jnp.einsum("bsr,rhe->bshe", c_kv, raw["w_uk"], preferred_element_type=jnp.float32).astype(x.dtype)
        v = jnp.einsum("bsr,rhe->bshe", c_kv, raw["w_uv"], preferred_element_type=jnp.float32).astype(x.dtype)
        a = mla_attend_xla(q_nope, q_rope, k_nope, k_r, v, impl)
        o = jnp.einsum("bqhe,hed->bqd", a, raw["w_o"], preferred_element_type=jnp.float32)
        out = (x.astype(jnp.float32) + o).astype(x.dtype)
        k_rope = pad_lanes(k_r)
    pad = ((0, 0), (0, max_len - S), (0, 0))
    return out, {"c_kv": jnp.pad(c_kv, pad), "k_rope": jnp.pad(k_rope, pad)}


def mla_decode_layer(x: jax.Array, p: dict[str, Any], config: HybridConfig, cache: dict, lengths, impl: Impl):
    """The MLA layer for one new token per sequence, in the absorbed form (decode.py): x + its output, and
    the cache with the token's latent and rope key appended."""
    B, D = x.shape
    raw = p["raw"]
    if impl.kernels:
        pk = p["packed"]
        cos, sin = mla_tables_at(lengths)
        c_q, c_new, k_slot = mla_down(x[None], pk["attn_norm"], pk["w_down"], pk["q_norm"], pk["kv_norm"], cos, sin)
        q = matmul(c_q[0], raw["w_uq"].reshape(config.q_rank, -1)).reshape(B, 1, config.heads, NOPE + ROPE)
        q_nope, q_rope = q[..., :NOPE], rope_at(q[..., NOPE:], lengths[:, None]).astype(x.dtype)
        c_new, r_new = c_new[0][:, None], rope_slot(k_slot[0])[:, None]
    else:
        q_nope, q_rope, c_new, k_r = mla_project_xla(x[:, None], raw, lengths[:, None])
        r_new = pad_lanes(k_r)
    cache = {"c_kv": append(cache["c_kv"], lengths, c_new), "k_rope": append(cache["k_rope"], lengths, r_new)}
    q_c, q_r = absorb_queries(q_nope, q_rope, p["w_uk_h"])
    attend = mla_decode if impl.kernels else xla_mla_decode
    latents = attend(q_c, q_r, cache["c_kv"], cache["k_rope"], lengths + 1)  # (B, 1, H, kv_rank)
    heads = latents[:, 0].transpose(1, 0, 2)  # head-major: a plain batched matmul (as decode.mla_attention)
    o = jnp.einsum("hbc,hcd->hbd", heads, p["w_uv_h"], preferred_element_type=jnp.float32)
    o = o.astype(x.dtype).transpose(1, 0, 2).reshape(B, -1)
    return residual_proj(x, o, raw["w_o"].reshape(-1, D), impl), cache


# --------------------------------------------------------------------------- MoE feed-forward layer


def moe_ffn(x: jax.Array, p: dict[str, Any], config: HybridConfig, impl: Impl) -> jax.Array:
    """x + MoE(rmsnorm(x)) + the shared expert (gated, for Qwen3-Next) on rows x (tokens, d_model): moe.py's
    routing, dispatch and `expert_ffn` kernel; or plain JAX with `jax.lax.ragged_dot` (or dense experts)."""
    mc = config.moe
    gated = config.kind == "qwen3-next"
    if impl.kernels:
        if not gated:  # moe.py's layer: one kernel at decode sizes, dispatched expert tiles above
            return moe_layer(x, p, mc)
        if x.shape[0] <= DECODE_TOKENS:  # one kernel for the routed experts (it adds the residual)
            kernel, args = decode_call(x, p, dataclasses.replace(mc, shared=0))
            routed = kernel(*args).astype(jnp.float32) - x.astype(jnp.float32)
        else:
            inputs, d, weights = expert_inputs(x, p, mc, default_group(x.shape[0]))
            routed = combine(expert_ffn(*inputs), d.pos, weights)
        s = p["shared"]
        return gated_shared_experts(
            x, p["norm"], s["w_gate"], s["w_up"], s["w_down"], p["shared_gate"], routed.astype(x.dtype)
        )
    tokens = x.shape[0]
    normed = rmsnorm(x, p["norm"])
    ids, weights = route(normed, p["router"], p["bias"], mc)
    if impl.experts == "dense":
        hidden = jax.nn.silu(jnp.einsum("td,edf->etf", normed, p["w_gate"], preferred_element_type=jnp.float32))
        hidden = hidden * jnp.einsum("td,edf->etf", normed, p["w_up"], preferred_element_type=jnp.float32)
        y = jnp.einsum("etf,efd->etd", hidden.astype(x.dtype), p["w_down"], preferred_element_type=jnp.float32)
        dense = jnp.zeros((tokens, mc.experts), jnp.float32).at[jnp.arange(tokens)[:, None], ids].add(weights)
        routed = jnp.einsum("te,etd->td", dense, y.astype(x.dtype).astype(jnp.float32))
    else:
        d = dispatch(ids, mc.experts, 1, ragged_rows(tokens * mc.top_k))
        xs = gather(normed, d.src)

        def grouped(lhs, rhs):
            return jax.lax.ragged_dot(lhs, rhs, d.counts, preferred_element_type=lhs.dtype)

        hidden = jax.nn.silu(grouped(xs, p["w_gate"]).astype(jnp.float32)) * grouped(xs, p["w_up"])
        routed = combine(grouped(hidden.astype(x.dtype), p["w_down"]), d.pos, weights)
    s = p["shared"]
    hidden = (jax.nn.silu(_dot(normed, s["w_gate"])) * _dot(normed, s["w_up"])).astype(x.dtype)
    shared = _dot(hidden, s["w_down"])
    if gated:
        shared = shared * jax.nn.sigmoid(_dot(normed, p["shared_gate"][:, :1]))
    return (x.astype(jnp.float32) + routed.astype(x.dtype).astype(jnp.float32) + shared).astype(x.dtype)


# --------------------------------------------------------------------------- weights


def _normal(key: jax.Array, shape: tuple[int, ...], fan_in: int, dtype: Any = jnp.bfloat16) -> jax.Array:
    return (jax.random.normal(key, shape) * fan_in**-0.5).astype(dtype)


def _norm_weight(key: jax.Array, n: int, dtype: Any = jnp.bfloat16) -> jax.Array:
    return (1 + 0.1 * jax.random.normal(key, (1, n))).astype(dtype)


def init_kda(config: HybridConfig, seed: int = 0, dtype: Any = jnp.bfloat16) -> dict[str, jax.Array]:
    """Random KDA weights: the qkv projection [q | k | v] (d_model, 3 H * HEAD_DIM); the low-rank projection
    [W_fa | W_ga | W_b, padded to LOW_RANK columns] (d_model, 3 * LOW_RANK) and W_fb, W_gb (LOW_RANK,
    H * HEAD_DIM); A in [1, 16] per head, dt_bias per channel (-4: decays near 1, long memories)."""
    d, h = config.d_model, config.v_heads
    vd = h * HEAD_DIM
    keys = jax.random.split(jax.random.key(seed), 10)
    w_low = jnp.zeros((d, 3 * LOW_RANK), dtype)
    w_low = w_low.at[:, : 2 * LOW_RANK].set(_normal(keys[1], (d, 2 * LOW_RANK), d, dtype))
    w_low = w_low.at[:, 2 * LOW_RANK : 2 * LOW_RANK + h].set(_normal(keys[2], (d, h), d, dtype))
    return {
        "norm": _norm_weight(keys[0], d, dtype),
        "w_qkv": _normal(keys[3], (d, 3 * vd), d, dtype),
        "w_low": w_low,
        "f_b": _normal(keys[4], (LOW_RANK, vd), LOW_RANK, dtype),
        "g_b": _normal(keys[5], (LOW_RANK, vd), LOW_RANK, dtype),
        "conv": _normal(keys[6], (CONV, 3 * vd), CONV, dtype),
        "A_log": jnp.log(jax.random.uniform(keys[7], (h,), minval=1.0, maxval=16.0)),
        "dt_bias": jnp.full((vd,), -4.0),
        "out_norm": _norm_weight(keys[8], HEAD_DIM, dtype),
        "w_out": _normal(keys[9], (vd, d), vd, dtype),
    }


def init_gqa(config: HybridConfig, seed: int = 0, dtype: Any = jnp.bfloat16) -> dict[str, jax.Array]:
    """Random gated-attention weights in qwen3.py's layout (its QKV kernel and plain-JAX path read them)."""
    d, dh = config.d_model, config.head_dim
    hd, kd = config.heads * dh, config.kv_heads * dh
    keys = jax.random.split(jax.random.key(seed), 8)
    return {
        "attn_norm": _norm_weight(keys[0], d, dtype),
        "w_q": _normal(keys[1], (d, hd), d, dtype),
        "w_k": _normal(keys[2], (d, kd), d, dtype),
        "w_v": _normal(keys[3], (d, kd), d, dtype),
        "w_attn_gate": _normal(keys[4], (d, hd), d, dtype),
        "q_norm": _norm_weight(keys[5], dh, dtype),
        "k_norm": _norm_weight(keys[6], dh, dtype),
        "wo": _normal(keys[7], (hd, d), hd, dtype),
    }


def init_mla(config: HybridConfig, seed: int = 0, dtype: Any = jnp.bfloat16) -> dict[str, Any]:
    """Random MLA weights: mla.py's own layout ("raw", the plain-JAX path's and decoding's), the prefill
    kernels' ("packed", `mla.pack`), and W_uk, W_uv head-major (H, kv_rank, 128) for the absorbed form."""
    raw = mla_init_params(MlaConfig(config.d_model, config.heads, config.q_rank, config.kv_rank), dtype, seed)
    return {
        "raw": raw,
        "packed": pack(raw),
        "w_uk_h": raw["w_uk"].transpose(1, 0, 2),
        "w_uv_h": raw["w_uv"].transpose(1, 0, 2),
    }


def init_moe(config: HybridConfig, seed: int = 0, dtype: Any = jnp.bfloat16) -> dict[str, Any]:
    """Random MoE weights (moe.py's); Qwen3-Next's shared-expert gate as one column repeated in 128 lanes."""
    params = moe_init_params(config.moe, dtype, seed)
    if config.kind == "qwen3-next":
        column = _normal(jax.random.key(seed + 10**6), (config.d_model, 1), config.d_model, dtype)
        params["shared_gate"] = jnp.tile(column, (1, LANES))
    return params
