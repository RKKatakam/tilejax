"""Hybrid linear/full-attention models (examples/hybrid_layers.py, examples/hybrid_models.py): the new kernels in
the interpreter with the race detector against float64 NumPy; whole tiny models (prefill logits and state,
decode steps, greedy generation) against a float64 NumPy model whose linear layers run the recurrence token by
token; the plain-JAX baselines likewise; every kernel the models run cross-lowered for a v5e at the benchmark
configs' shapes; on a TPU, the tuned models against the baselines."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ON_TPU
from decode import gqa_decode, mla_decode
from gqa_attention import make_attention
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from hybrid_layers import (
    LOW_RANK,
    HybridConfig,
    Impl,
    delta_step,
    gated_shared_experts,
    gdn_prefill,
    init_moe,
    kda_prefill,
    moe_ffn,
    states_from_columns,
)
from hybrid_models import (
    CONFIGS,
    baseline_decode_step,
    baseline_prefill,
    decode_step,
    generate,
    init_params,
    prefill,
    random_state,
    round_up,
    tiny_config,
)
from linear_attention import HEAD_DIM, KDA_GATE_MIN, DeltaTiles, l2_normalize, make_inputs
from linear_attention_layer import GATES, residual_matmul
from matmul import matmul
from mla import LANES, NOPE, ROPE, mla_attention, mla_down
from moe import MoeConfig, SharedTiles, capacity, default_group, expert_ffn, shared_experts
from qwen3 import make_qkv
from rmsnorm_matmul import rmsnorm_matmul

B16 = jnp.bfloat16
EPS = 1e-6
XLA = Impl(kernels=False, attention="xla")

# Tiny models. Every expert is chosen (top_k = experts): the routing is then a smooth function of its input,
# so bf16 rounding can't flip a token to another expert between the model and NumPy (that is tested on one
# layer, below, with top-2 routing).
TINY = {kind: tiny_config(kind, top_k=4) for kind in ("qwen3-next", "kimi-linear")}


# --------------------------------------------------------------------------- float64 NumPy references


def _rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * f64(w).reshape(-1)


def _silu(x):
    return x / (1 + np.exp(-x))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _l2n(x, scale=1.0):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + EPS) * scale


def _heads(x, d=HEAD_DIM):
    return x.reshape(*x.shape[:-1], -1, d)


def _conv(pre, w):
    """The causal short convolution and SiLU over tokens of pre (B, S, C)."""
    s = pre.shape[1]
    padded = np.pad(pre, ((0, 0), (3, 0), (0, 0)))
    return _silu(sum(f64(w)[j] * padded[:, j : j + s] for j in range(4)))


def _delta(q, k, v, decay, beta):
    """The delta rule token by token: q, k, v (B, S, H, d) (q and k normalized, q scaled), decay (B, S, H,
    d_k), beta (B, S, H). Outputs (B, S, H, d_v), final states (B, H, d_k, d_v)."""
    b, s, h, d = v.shape
    state = np.zeros((b, h, d, d))
    out = np.empty_like(v)
    for t in range(s):
        state = decay[:, t, :, :, None] * state
        u = beta[:, t, :, None] * (v[:, t] - np.einsum("bhkv,bhk->bhv", state, k[:, t]))
        state = state + k[:, t, :, :, None] * u[:, :, None, :]
        out[:, t] = np.einsum("bhkv,bhk->bhv", state, q[:, t])
    return out, state


def _gated(o, z, weight, gate):
    """RMSNorm per head of o (B, S, H, d), times weight, times gate(z), as (B, S, H * d)."""
    normed = o / np.sqrt(np.mean(o * o, axis=-1, keepdims=True) + EPS) * f64(weight).reshape(-1)
    return (normed * gate(z)).reshape(*o.shape[:2], -1)


def _gdn(x, p, config):
    """Qwen3-Next's Gated DeltaNet layer: x + output, the final states, the last 3 conv inputs."""
    vd, kd, hv = config.v_heads * HEAD_DIM, config.k_heads * HEAD_DIM, config.v_heads
    r = config.v_heads // config.k_heads
    h = _rms(x, p["norm"])
    proj, gates = h @ f64(p["w_qkvz"]), h @ f64(p["w_ba"])
    z, pre = proj[..., :vd], proj[..., vd:]
    conv = _conv(pre, p["conv"])
    q, k = (np.repeat(_heads(t), r, axis=2) for t in (conv[..., :kd], conv[..., kd : 2 * kd]))
    v = _heads(conv[..., 2 * kd :])
    beta = _sigmoid(gates[..., :hv])
    g = -np.exp(f64(p["A_log"])) * np.logaddexp(0, gates[..., hv : 2 * hv] + f64(p["dt_bias"]))
    decay = np.broadcast_to(np.exp(np.minimum(g, 0))[..., None], v.shape)
    o, state = _delta(_l2n(q, HEAD_DIM**-0.5), _l2n(k), v, decay, beta)
    out = x + _gated(o, _heads(z), p["out_norm"], _silu) @ f64(p["w_out"])
    return out, {"state": state, "conv": pre[:, -3:]}


def _kda(x, p, config):
    """Kimi Linear's KDA layer: x + output, the final states, the last 3 conv inputs."""
    hv = config.v_heads
    h = _rms(x, p["norm"])
    proj, low = h @ f64(p["w_qkv"]), h @ f64(p["w_low"])
    q, k, v = (_heads(t) for t in np.split(_conv(proj, p["conv"]), 3, axis=-1))
    f = low[..., :LOW_RANK] @ f64(p["f_b"])
    g = -np.repeat(np.exp(f64(p["A_log"])), HEAD_DIM) * np.logaddexp(0, f + f64(p["dt_bias"]))
    decay = _heads(np.exp(np.clip(g, KDA_GATE_MIN, 0)))
    z = low[..., LOW_RANK : 2 * LOW_RANK] @ f64(p["g_b"])
    beta = _sigmoid(low[..., 2 * LOW_RANK : 2 * LOW_RANK + hv])
    o, state = _delta(_l2n(q, HEAD_DIM**-0.5), _l2n(k), v, decay, beta)
    out = x + _gated(o, _heads(z), p["out_norm"], _sigmoid) @ f64(p["w_out"])
    return out, {"state": state, "conv": proj[:, -3:]}


def _rope_half(t, positions, rotary_dim, theta):
    """Rotate-half RoPE of t (B, S, H, d) on its first rotary_dim dims, at positions (S,)."""
    half = rotary_dim // 2
    inv_freq = theta ** (-np.arange(half) * 2 / rotary_dim)
    angles = np.asarray(positions, np.float64)[:, None] * np.concatenate([inv_freq, inv_freq])[None]
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    r, still = t[..., :rotary_dim], t[..., rotary_dim:]
    rotated = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * cos + rotated * sin, still], axis=-1)


def _causal(scores):
    s = scores.shape[-1]
    scores = np.where(np.tril(np.ones((s, s), bool)), scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def _gqa(x, p, config):
    """Qwen3-Next's gated attention layer: x + output, and the keys and values (B, S, kv_heads * head_dim)."""
    dh, s = config.head_dim, x.shape[1]
    rd = config.attention.rotary_dim
    n = _rms(x, p["attn_norm"])
    q = _rms(_heads(n @ f64(p["w_q"]), dh), p["q_norm"])
    k = _rms(_heads(n @ f64(p["w_k"]), dh), p["k_norm"])
    q = _rope_half(q, np.arange(s), rd, config.rope_theta) * dh**-0.5
    k = _rope_half(k, np.arange(s), rd, config.rope_theta)
    v = _heads(n @ f64(p["w_v"]), dh)
    groups = config.heads // config.kv_heads
    kr, vr = np.repeat(k, groups, axis=2), np.repeat(v, groups, axis=2)
    a = np.einsum("bhqk,bkhd->bqhd", _causal(np.einsum("bqhd,bkhd->bhqk", q, kr)), vr).reshape(*x.shape[:2], -1)
    a = a * _sigmoid(n @ f64(p["w_attn_gate"]))
    flat = (*x.shape[:2], -1)
    return x + a @ f64(p["wo"]), {"k": k.reshape(flat), "v": v.reshape(flat)}


def _mla(x, p, config):
    """The MLA layer (expanded form): x + output, and the latents and rope keys (B, S, 64)."""
    raw, s = p["raw"], x.shape[1]
    h = _rms(x, raw["attn_norm"])
    c_q = _rms(h @ f64(raw["w_dq"]), raw["q_norm"])
    q = np.einsum("bsr,rhe->bshe", c_q, f64(raw["w_uq"]))
    q_rope = _rope_half(q[..., 128:], np.arange(s), 64, 10000.0)
    c_kv = _rms(h @ f64(raw["w_dkv"]), raw["kv_norm"])
    k_rope = _rope_half((h @ f64(raw["w_kr"]))[:, :, None], np.arange(s), 64, 10000.0)[:, :, 0]
    k_nope = np.einsum("bsr,rhe->bshe", c_kv, f64(raw["w_uk"]))
    v = np.einsum("bsr,rhe->bshe", c_kv, f64(raw["w_uv"]))
    scores = np.einsum("bqhe,bkhe->bhqk", q[..., :128], k_nope) + np.einsum("bqhe,bke->bhqk", q_rope, k_rope)
    o = np.einsum("bhqk,bkhe->bqhe", _causal(scores * 192**-0.5), v)
    return x + np.einsum("bqhe,hed->bqd", o, f64(raw["w_o"])), {"c_kv": c_kv, "k_rope": k_rope}


def _moe(x, p, config):
    """The MoE layer and its shared expert (gated for Qwen3-Next) on rows x (T, D)."""
    mc = config.moe
    normed = _rms(x, p["norm"])
    logits = normed @ f64(p["router"])
    if mc.scoring == "softmax":
        scores = np.exp(logits - logits.max(1, keepdims=True))
        scores /= scores.sum(1, keepdims=True)
        choice = scores
    else:
        scores = _sigmoid(logits)
        choice = scores + f64(p["bias"])
    ids = np.argsort(-choice, axis=1, kind="stable")[:, : mc.top_k]
    weights = np.take_along_axis(scores, ids, 1)
    weights = weights / weights.sum(1, keepdims=True) * mc.scale

    def swiglu(rows, w_gate, w_up, w_down):
        return (_silu(rows @ f64(w_gate)) * (rows @ f64(w_up))) @ f64(w_down)

    out = x.copy()
    for e in range(mc.experts):
        chosen = np.sum(np.where(ids == e, weights, 0.0), axis=1, keepdims=True)
        out += chosen * swiglu(normed, p["w_gate"][e], p["w_up"][e], p["w_down"][e])
    s = p["shared"]
    shared = swiglu(normed, s["w_gate"], s["w_up"], s["w_down"])
    if config.kind == "qwen3-next":
        shared = shared * _sigmoid(normed @ f64(p["shared_gate"][:, :1]))
    return out + shared


def _model(tokens, params):
    """Logits (B, S, vocab) at every position and the decode state after the last."""
    config = params["config"]
    x = f64(params["embed"])[np.asarray(tokens)]
    b, s, d = x.shape
    state = []
    for i, layer in enumerate(params["layers"]):
        if config.is_full(i):
            mixer = _gqa if config.kind == "qwen3-next" else _mla
        else:
            mixer = _gdn if config.kind == "qwen3-next" else _kda
        x, cache = mixer(x, layer["mixer"], config)
        x = _moe(x.reshape(b * s, d), layer["moe"], config).reshape(b, s, d)
        state.append(cache)
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"]), state


def _close(got, want, tol=3e-2):
    want = f64(want)
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


@functools.cache
def _tiny(kind):
    config = TINY[kind]
    params = init_params(config)
    prompt = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
    return config, params, prompt


# --------------------------------------------------------------------------- the delta-rule kernels with states


def _gate_inputs(batch, heads, seq, seed=5):
    z = rand((batch, seq, heads * HEAD_DIM), seed, B16)
    weight = (1 + 0.1 * rand((1, HEAD_DIM), seed + 1)).astype(B16)
    return z, weight


def _delta_reference(q, k, v, g, beta, per_channel):
    """The recurrence (inputs as the kernels take them) in float64: outputs (B, S, H, d), states."""
    q, k, v, g, beta = (f64(t) for t in (q, k, v, g, beta))
    if per_channel:
        decay = _heads(np.exp(np.clip(g, KDA_GATE_MIN, 0)))
    else:
        decay = np.broadcast_to(np.exp(np.minimum(g, 0)).transpose(0, 2, 1)[..., None], _heads(v).shape)
    return _delta(_l2n(_heads(q), HEAD_DIM**-0.5), _l2n(_heads(k)), _heads(v), decay, beta.transpose(0, 2, 1))


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize(
    ("batch", "heads", "seq", "tiles"),
    [(1, 2, 512, DeltaTiles(heads=2, block=128)), (2, 2, 256, DeltaTiles(heads=1, block=256))],
)
def test_delta_kernels_store_outputs_per_step_and_final_states(per_channel, batch, heads, seq, tiles):
    kernel = kda_prefill if per_channel else gdn_prefill
    q, k, v, g, beta = make_inputs(batch, heads, seq, per_channel)
    z, weight = _gate_inputs(batch, heads, seq)
    out, final = run_checking_races(kernel, q, k, v, g, beta, z, weight, tiles)
    assert out.shape == v.shape and out.dtype == B16 and final.shape == (batch, HEAD_DIM, heads * HEAD_DIM)
    o, state = _delta_reference(q, k, v, g, beta, per_channel)
    _close(out, _gated(o, _heads(f64(z)), weight, _sigmoid if per_channel else _silu))
    _close(states_from_columns(final), state, 2e-2)
    kernel.verify(q, k, v, g, beta, z, weight, tiles)
    lowered = kernel.lower(q, k, v, g, beta, z, weight, tiles)
    # One phase over blocks of tokens storing a block of outputs per step; the state carried; the final
    # state stored as the tile ends.
    assert [p.range.var for p in lowered.plan.phases] == ["c"]
    assert set(lowered.plan.carried) == {"state"}
    assert lowered.steps == batch * heads // tiles.heads * seq // tiles.block


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_chunked_baseline_returns_the_recurrences_state(per_channel):
    from hybrid_layers import chunked_with_state

    q, k, v, g, beta = make_inputs(2, 2, 512, per_channel)
    out, state = chunked_with_state(q, k, v, g, beta)
    o, want = _delta_reference(q, k, v, g, beta, per_channel)
    _close(out, o.reshape(out.shape), 2e-2)
    _close(state, want, 2e-2)


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_decode_recurrence_continues_the_prefill_state(per_channel):
    # The kernel's state after 256 tokens, then delta_step token by token, equals the recurrence over 260.
    q, k, v, g, beta = make_inputs(1, 2, 260, per_channel)
    z, weight = _gate_inputs(1, 2, 256)
    g_prompt = g[:, :256] if per_channel else g[..., :256]
    kernel = kda_prefill if per_channel else gdn_prefill
    tiles = DeltaTiles(heads=2, block=256)
    _, final = kernel(q[:, :256], k[:, :256], v[:, :256], g_prompt, beta[..., :256], z, weight, tiles)
    state = states_from_columns(final)
    o_want, s_want = _delta_reference(q, k, v, g, beta, per_channel)
    for t in range(256, 260):
        qt, kt, vt = (x[:, t].reshape(1, 2, HEAD_DIM).astype(jnp.float32) for x in (q, k, v))
        if per_channel:
            decay = jnp.exp(jnp.clip(g[:, t], KDA_GATE_MIN, 0.0)).reshape(1, 2, HEAD_DIM)
        else:
            decay = jnp.exp(jnp.minimum(g[:, :, t], 0.0))[..., None]
        o, state = delta_step(l2_normalize(qt, HEAD_DIM**-0.5), l2_normalize(kt), vt, decay, beta[:, :, t], state)
        _close(o, o_want[:, t], 2e-2)
    _close(state, s_want, 2e-2)


# --------------------------------------------------------------------------- the MoE layer


def test_gated_shared_experts_against_numpy():
    t, d, f = 256, 256, 512
    x, routed = rand((t, d), 0, B16), rand((t, d), 1, B16)
    gamma = (1 + 0.1 * rand((1, d), 2)).astype(B16)
    w_gate, w_up = rand((d, f), 3, B16) / 16, rand((d, f), 4, B16) / 16
    w_down = rand((f, d), 5, B16) / 23
    w_sg = jnp.tile(rand((d, 1), 6, B16) / 16, (1, 128))
    args = (x, gamma, w_gate, w_up, w_down, w_sg, routed, SharedTiles(rows=256, block_f=256))
    out = run_checking_races(gated_shared_experts, *args)
    n = _rms(f64(x), gamma)
    hidden = _silu(n @ f64(w_gate)) * (n @ f64(w_up))
    want = f64(x) + f64(routed) + _sigmoid(n @ f64(w_sg[:, :1])) * (hidden @ f64(w_down))
    _close(out, want)
    gated_shared_experts.verify(*args)


@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
def test_moe_layer_with_top_k_routing_against_numpy(kind):
    config = HybridConfig(
        kind=kind, layers=1, d_model=256, vocab=512, k_heads=1, v_heads=1, heads=2,
        moe=MoeConfig(256, 8, 2, 128, shared=1, scoring="softmax" if kind == "qwen3-next" else "sigmoid"),
    )  # fmt: skip
    params = init_moe(config, 3)
    x = rand((64, 256), 7, B16)
    want = _moe(f64(x), params, config)
    for impl in (Impl(), XLA, Impl(kernels=False, experts="dense")):
        _close(jax.jit(moe_ffn, static_argnums=(2, 3))(x, params, config, impl), want)


# --------------------------------------------------------------------------- whole models


@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
@pytest.mark.parametrize("impl", [Impl(), XLA], ids=["autokernel", "plain-jax"])
def test_prefill_matches_numpy(kind, impl):
    config, params, prompt = _tiny(kind)
    logits, state = jax.jit(prefill, static_argnames=("max_len", "impl"))(prompt, params, 512, impl)
    want_logits, want_state = _model(prompt, params)
    assert logits.shape == (2, config.vocab)
    _close(logits, want_logits[:, -1], 4e-2)
    for i, (got, want) in enumerate(zip(state, want_state, strict=True)):
        if not config.is_full(i):
            # The prefill state equals running the recurrence token by token, and the conv tail the last inputs.
            assert got["state"].dtype == jnp.float32 and got["state"].shape == (2, config.v_heads, 128, 128)
            _close(got["state"], want["state"], 3e-2)
            _close(got["conv"], want["conv"])
        elif kind == "qwen3-next":
            for name in ("k", "v"):
                assert got[name].shape == (2, 512, config.kv_heads * config.head_dim)
                _close(got[name][:, :256], want[name])
                assert not np.asarray(got[name][:, 256:]).any()
        else:
            _close(got["c_kv"][:, :256], want["c_kv"])
            _close(got["k_rope"][:, :256, :64], want["k_rope"])
            assert not np.asarray(got["k_rope"][:, :, 64:]).any()


@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
def test_decode_steps_match_numpy(kind):
    # Prefill, then three decode steps of given tokens: each step's logits are NumPy's at that position.
    config, params, prompt = _tiny(kind)
    new = jax.random.randint(jax.random.key(2), (2, 3), 0, config.vocab)
    want, _ = _model(jnp.concatenate([prompt, new], axis=1), params)
    ours = jax.jit(decode_step, static_argnames="impl")
    for impl, step in ((Impl(), ours), (XLA, baseline_decode_step())):
        _, state = jax.jit(prefill, static_argnames=("max_len", "impl"))(prompt, params, 512, impl)
        lengths = jnp.full((2,), 256, jnp.int32)
        for t in range(3):
            logits, state, lengths = step(new[:, t], state, lengths, params)
            _close(logits, want[:, 256 + t], 4e-2)
        assert lengths.tolist() == [259, 259]


@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
def test_greedy_generation_is_numpys_argmax(kind):
    _, params, prompt = _tiny(kind)
    tokens = generate(prompt, params, 4)
    assert tokens.shape == (2, 4) and tokens.dtype == jnp.int32
    logits, _ = _model(jnp.concatenate([prompt, tokens[:, :-1]], axis=1), params)
    logits = logits[:, 255:]  # the logits each new token was chosen from
    chosen = np.take_along_axis(logits, np.asarray(tokens)[..., None], axis=-1)[..., 0]
    # Each token is NumPy's argmax, or within bf16 noise of it (a near-tie).
    assert (chosen >= logits.max(-1) - 0.03 * np.abs(logits).max()).all()
    assert (np.asarray(tokens) == logits.argmax(-1)).mean() >= 0.75


def test_random_state_has_the_prefill_states_layout():
    for kind in TINY:
        config, params, prompt = _tiny(kind)
        _, state = jax.eval_shape(functools.partial(prefill, max_len=512), prompt, params)
        fake = random_state(config, [100, 300], 512)
        assert jax.tree.map(lambda t: (t.shape, t.dtype), fake) == jax.tree.map(lambda t: (t.shape, t.dtype), state)


# --------------------------------------------------------------------------- Mosaic for a v5e, at the benchmark shapes


def _model_specs(config, batch, seq):
    """(kernel, input specs) for every kernel a benchmark config runs: prefill of (batch, seq) prompts, and
    decode steps of `batch` sequences over caches of `seq` positions."""
    d, mc, b16 = config.d_model, config.moe, functools.partial(spec, dtype=B16)
    rows, vd, kd, h = batch * seq, config.v_heads * HEAD_DIM, config.k_heads * HEAD_DIM, config.heads
    gamma, act, gates = spec(1, d, dtype=B16), b16(batch, seq, vd), spec(batch, config.v_heads, seq)
    lengths = spec(batch, dtype=jnp.int32)
    attn = h * (config.head_dim if config.kind == "qwen3-next" else NOPE)
    cases = [(rmsnorm_matmul, (b16(batch, d), gamma, b16(d, config.vocab)))]  # the logits
    for n in (rows, batch):  # prefill's rows, decode's
        group = default_group(n)
        slots = capacity(n * mc.top_k, mc.experts, group)
        experts = (b16(mc.experts, d, mc.ffn), b16(mc.experts, d, mc.ffn), b16(mc.experts, mc.ffn, d))
        shared = (gamma, b16(d, mc.ffn), b16(d, mc.ffn), b16(mc.ffn, d))
        cases += [
            (expert_ffn, (b16(slots, d), *experts, spec(slots // group, dtype=jnp.int32))),
            (residual_matmul, (b16(n, d), b16(n, vd), b16(vd, d))),  # the linear layers' output projection
            (residual_matmul, (b16(n, d), b16(n, attn), b16(attn, d))),  # decode attention's
        ]
        if config.kind == "qwen3-next":
            projections = (2 * vd + 2 * kd, GATES)
            cases.append((gated_shared_experts, (b16(n, d), *shared, b16(d, LANES), b16(n, d))))
        else:
            projections = (3 * vd, 3 * LOW_RANK)
            cases.append((shared_experts, (b16(n, d), *shared, b16(n, d))))
            cases.append((matmul, (b16(n, config.q_rank), b16(config.q_rank, h * (NOPE + ROPE)))))
        cases += [(rmsnorm_matmul, (b16(n, d), gamma, b16(d, w))) for w in projections]
    if config.kind == "qwen3-next":
        proj = b16(batch, seq, 2 * vd + 2 * kd)
        cases += [(gdn_prefill, (act, act, act, gates, gates, proj, b16(1, HEAD_DIM), tiles)) for tiles in DELTA]
        dh, hd, kvd = config.head_dim, h * config.head_dim, config.kv_heads * config.head_dim
        qkv = make_qkv(True, config.attention.rotary_dim).gated_projection
        weights = (b16(d, hd), b16(d, kvd), b16(d, kvd), b16(d, hd), spec(1, dh), spec(1, dh))
        for b, s in ((batch, seq), (1, batch)):  # prefill; decode: the batch as one sequence's rows
            cases.append((qkv, (b16(b, s, d), gamma, *weights, spec(s, dh), spec(s, dh))))
        q, kv = b16(batch, seq, hd), b16(batch, seq, kvd)
        cases.append((make_attention(dh).gated_attention, (b16(batch, seq, d), q, kv, kv, q, b16(hd, d))))
        cases.append((gqa_decode, (b16(batch, 1, h, dh), kv, kv, lengths)))
    else:
        g = spec(batch, seq, vd)
        cases += [(kda_prefill, (act, act, act, g, gates, act, b16(1, HEAD_DIM), tiles)) for tiles in DELTA]
        rq, rkv = config.q_rank, config.kv_rank
        down = (gamma, b16(d, rq + rkv + LANES), b16(1, rq), b16(1, rkv))
        for b, s in ((batch, seq), (1, batch)):
            cases.append((mla_down, (b16(b, s, d), *down, spec(s, LANES), spec(s, LANES))))
        cases.append((matmul, (b16(rows, rkv), b16(rkv, 2 * h * NOPE))))
        q, kv, slot = b16(batch, seq, h * (NOPE + ROPE)), b16(batch, seq, 2 * h * NOPE), b16(batch, seq, LANES)
        tables = (spec(seq, LANES), spec(seq, LANES))
        cases.append((mla_attention, (b16(batch, seq, d), q, kv, slot, *tables, b16(h, NOPE, d))))
        latents = (b16(batch, 1, h, rkv), b16(batch, 1, h, LANES), b16(batch, seq, rkv), b16(batch, seq, LANES))
        cases.append((mla_decode, (*latents, lengths)))
    return cases


DELTA = (DeltaTiles(), DeltaTiles(heads=4, block=512), DeltaTiles(heads=1, block=128))  # (defaults, extremes)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize(("batch", "seq"), [(1, 4096), (32, 1024)])
def test_kernels_lower_to_mosaic_for_v5e(name, batch, seq):
    for kernel, specs in _model_specs(CONFIGS[name], batch, seq):
        arrays = specs[: len(kernel.nest.params)]
        assert "tpu_custom_call" in lower_for_v5e(kernel.lower(*specs), *arrays), kernel.nest.name


# --------------------------------------------------------------------------- on a TPU v5e


def _medium(kind):
    """A 4-layer, d_model 1024 version of a benchmark config, every one of its 16 experts chosen (no routing
    near-ties between the models)."""
    small = CONFIGS[f"{kind}-small"]
    moe = dataclasses.replace(small.moe, d_model=1024, experts=16, top_k=16)
    return dataclasses.replace(small, layers=4, d_model=1024, vocab=4096, moe=moe)


@pytest.mark.tpu
@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
def test_prefill_and_decode_match_the_baselines_on_tpu(kind):
    config = _medium(kind)
    params = init_params(config)
    prompt = jax.random.randint(jax.random.key(1), (2, 2048), 0, config.vocab)
    logits, state = jax.jit(prefill, static_argnames="max_len")(prompt, params, 2560)
    want, want_state = baseline_prefill()(prompt, params, 2560)
    _close(logits, want, 5e-2)
    for got, base in zip(state, want_state, strict=True):
        for name in got:
            _close(got[name], base[name], 5e-2)
    lengths = jnp.full((2,), 2048, jnp.int32)
    token = jnp.argmax(want, axis=-1).astype(jnp.int32)
    ours, _, _ = decode_step(token, state, lengths, params)  # eagerly: tunes on the real lengths
    base, _, _ = baseline_decode_step()(token, want_state, lengths, params)
    _close(ours, base, 5e-2)
    assert generate(prompt[:1, :512], params, 4).shape == (1, 4)


def test_round_up():
    assert (round_up(260), round_up(512), round_up(513, 128)) == (512, 512, 640)


@pytest.mark.parametrize("kind", ["qwen3-next", "kimi-linear"])
def test_splash_baseline_prefill_matches_numpy(kind):
    # The benchmark's baseline: JAX's splash attention (in its interpreter off a TPU).
    _, params, prompt = _tiny(kind)
    logits, _ = baseline_prefill(interpret=not ON_TPU)(prompt[:1], params)
    _close(logits, _model(prompt[:1], params)[0][:, -1], 4e-2)
