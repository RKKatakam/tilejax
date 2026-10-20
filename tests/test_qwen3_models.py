"""End-to-end Qwen3 dense and Qwen3-MoE inference (examples/qwen3_models.py): prefill logits and caches, decode
steps and greedy generation against a float64 NumPy model; the plain-JAX baselines; the module's kernel and
the decode use of the QKV kernel in the interpreter with the race detector; every kernel the models run,
cross-lowered to Mosaic for a v5e at the benchmark's shapes; and (on a TPU) the models against the
baselines."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, rand, run_checking_races
from kv_cache import page_tables, ragged_kv_pages, to_pages
from qwen3 import ProjectionTiles, make_qkv, qk_weights, rope_tables
from qwen3_models import (
    CONFIGS,
    ModelConfig,
    OutputTiles,
    attention_output,
    baseline_decode,
    baseline_params,
    baseline_prefill,
    decode_calls,
    decode_step,
    device_ms,
    forward,
    generate,
    init_params,
    prefill,
    prefill_calls,
    random_cache,
)

B16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
EPS = 1e-6

DENSE = ModelConfig(layers=2, d_model=256, heads=4, kv_heads=2, head_dim=128, ffn=512, vocab=512)
MOE = ModelConfig(layers=2, d_model=256, heads=4, kv_heads=1, head_dim=128, ffn=256, vocab=512, experts=8, top_k=2)
# The MoE model in f32 for exact comparisons: in bf16 a router's near-tie can pick another expert than
# float64 does (a different but equally valid output for that token), so bf16 is checked on the whole.
MODELS = [(DENSE, B16), (MOE, F32)]
IDS = ["dense", "moe"]


# --------------------------------------------------------------------------- the float64 NumPy model


def _rms(x, gamma):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * f64(gamma).reshape(-1)


def _rope(x, positions, theta):
    """Rotate-half RoPE of x (n, heads, Dh) at positions (n,), as Hugging Face's Qwen3."""
    dh = x.shape[-1]
    angles = np.asarray(positions, np.float64)[:, None] * theta ** (-np.arange(0, dh, 2) / dh)
    cos, sin = np.cos(np.concatenate([angles] * 2, -1)), np.sin(np.concatenate([angles] * 2, -1))
    rotated = np.concatenate([-x[..., dh // 2 :], x[..., : dh // 2]], axis=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def _softmax(s):
    p = np.exp(s - s.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def _silu(x):
    return x / (1 + np.exp(-x))


def _moe(x, p, config):
    """Qwen3-MoE's feed-forward layer: softmax router, top-k experts, their weights renormalized."""
    normed = _rms(x, p["norm"])
    probs = _softmax(normed @ f64(p["router"]))
    ids = np.argsort(-probs, axis=1)[:, : config.top_k]
    weights = np.take_along_axis(probs, ids, axis=1)
    weights /= weights.sum(1, keepdims=True)
    hidden = _silu(np.einsum("td,edf->etf", normed, f64(p["w_gate"]))) * np.einsum(
        "td,edf->etf", normed, f64(p["w_up"])
    )
    every = np.einsum("etf,efd->etd", hidden, f64(p["w_down"]))  # every expert on every token
    return sum(weights[:, j, None] * every[ids[:, j], np.arange(len(x))] for j in range(config.top_k))


def _model(tokens, params):
    """One sequence's logits (n, vocab) for its tokens (n,), and each layer's keys and values (n, Hkv * Dh)
    (keys normed and rotated, as cached)."""
    config: ModelConfig = params["config"]
    dh, heads, kv_heads = config.head_dim, config.heads, config.kv_heads
    x = f64(params["embed"])[np.asarray(tokens)]
    n = len(x)
    positions = np.arange(n)
    keys, values = [], []
    for layer in params["layers"]:
        h = _rms(x, layer["attn_norm"])
        q = _rms((h @ f64(layer["w_q"])).reshape(n, heads, dh), layer["q_norm"])
        k = _rms((h @ f64(layer["w_k"])).reshape(n, kv_heads, dh), layer["k_norm"])
        v = (h @ f64(layer["w_v"])).reshape(n, kv_heads, dh)
        q, k = _rope(q, positions, config.rope_theta) / np.sqrt(dh), _rope(k, positions, config.rope_theta)
        keys.append(k.reshape(n, -1))
        values.append(v.reshape(n, -1))
        attn = np.zeros((n, heads, dh))
        causal = np.tril(np.ones((n, n), bool))
        for head in range(heads):
            g = head // (heads // kv_heads)
            attn[:, head] = _softmax(np.where(causal, q[:, head] @ k[:, g].T, -np.inf)) @ v[:, g]
        x = x + attn.reshape(n, -1) @ f64(layer["wo"])
        if config.moe is not None:
            x = x + _moe(x, layer["moe"], config)
        else:
            f = _rms(x, layer["ffn_norm"])
            x = x + (_silu(f @ f64(layer["w_gate"])) * (f @ f64(layer["w_up"]))) @ f64(layer["w_down"])
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"]), keys, values


def _close(got, want, tol):
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


def _tol(dtype):
    return 5e-2 if dtype == B16 else 2e-3


def _rows_agreeing(got, want, tol=3e-2):
    """The share of rows of logits (n, vocab) within `tol` relative RMS error of `want`'s."""
    got, want = f64(got), f64(want)
    return float(np.mean(np.sqrt(np.mean((got - want) ** 2, -1) / np.mean(want**2, -1)) < tol))


# --------------------------------------------------------------------------- the kernels, in the interpreter


def test_attention_output_matches_numpy():
    x, attn = rand((8, 256), 0, B16), rand((8, 512), 1, B16)
    wo = (rand((512, 256), 2) / 512**0.5).astype(B16)
    tiles = OutputTiles(block_a=256)  # two steps over W_o's rows
    out = run_checking_races(attention_output, x, attn, wo, tiles)
    _close(out, f64(x) + f64(attn) @ f64(wo), 2e-2)
    attention_output.verify(x, attn, wo, tiles)
    assert [p.range.var for p in attention_output.lower(x, attn, wo, tiles).plan.phases] == ["a"]


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_qkv_kernel_rotates_each_decode_row_at_its_own_position(batch):
    # Decoding runs the prefill QKV kernel on the B new tokens as one sequence's rows, row b with the RoPE
    # tables of position lengths[b].
    params = init_params(DENSE)
    layer = params["layers"][0]
    x = rand((1, batch, 256), 3, B16)
    positions = jnp.asarray(np.random.default_rng(0).integers(0, 5000, batch), I32)
    cos, sin = rope_tables(positions, 128)
    args = (x, layer["attn_norm"], layer["w_q"], layer["w_k"], layer["w_v"], *qk_weights(layer, DENSE.attention))
    args += (cos, sin, ProjectionTiles(rows=128, block_d=128))
    kernel = make_qkv(True, None).projection
    q, k, v = run_checking_races(kernel, *args)
    h = _rms(f64(x[0]), layer["attn_norm"])
    want_q = _rms((h @ f64(layer["w_q"])).reshape(batch, 4, 128), layer["q_norm"])
    want_k = _rms((h @ f64(layer["w_k"])).reshape(batch, 2, 128), layer["k_norm"])
    _close(q[0], _rope(want_q, np.asarray(positions), 1e6).reshape(batch, -1) / np.sqrt(128), 3e-2)
    _close(k[0], _rope(want_k, np.asarray(positions), 1e6).reshape(batch, -1), 3e-2)
    _close(v[0], h @ f64(layer["w_v"]), 3e-2)
    kernel.verify(*args)


# --------------------------------------------------------------------------- the models against NumPy


@pytest.mark.cpu  # (exact f32 matmuls: the interpreter's)
@pytest.mark.parametrize(("config", "dtype"), MODELS, ids=IDS)
def test_prefill_matches_numpy(config, dtype):
    params = init_params(config, dtype)
    tokens = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
    prompt_lengths = jnp.array([256, 77], I32)  # the second prompt padded
    logits = jax.jit(forward)(tokens, params)
    last, cache, lengths = jax.jit(prefill, static_argnums=2)(tokens, params, 384, prompt_lengths)
    assert logits.shape == (2, 256, config.vocab) and last.shape == (2, config.vocab)
    assert lengths.tolist() == [256, 77] and cache[0]["k"].shape == (2, 384, config.kv_heads * 128)
    for b, n in enumerate([256, 77]):
        want, keys, values = _model(tokens[b, :n], params)
        _close(logits[b, :n], want, _tol(dtype))  # (causal: the padding after a prompt doesn't change it)
        _close(last[b], want[-1], _tol(dtype))
        for layer in (0, -1):  # the cache holds the layers' keys (normed, rotated) and values
            _close(cache[layer]["k"][b, :n], keys[layer], _tol(dtype))
            _close(cache[layer]["v"][b, :n], values[layer], _tol(dtype))
            assert not np.asarray(cache[layer]["k"][b, 256:]).any()


def test_moe_model_in_bf16_matches_numpy_on_the_whole():
    params = init_params(MOE)
    tokens = jax.random.randint(jax.random.key(1), (1, 256), 0, MOE.vocab)
    logits = jax.jit(forward)(tokens, params)
    # Most rows agree to bf16 accuracy (~0.6% RMS); a few tokens whose router is near a tie took another
    # expert than float64 did, and differ by ~30%.
    assert _rows_agreeing(logits[0], _model(tokens[0], params)[0]) > 0.95


@pytest.mark.cpu
@pytest.mark.parametrize(("config", "dtype"), MODELS, ids=IDS)
def test_decode_steps_match_numpy(config, dtype):
    params = init_params(config, dtype)
    prompts = jax.random.randint(jax.random.key(1), (3, 128), 0, config.vocab)
    prompt_lengths = jnp.array([128, 5, 77], I32)
    new = jax.random.randint(jax.random.key(2), (3, 2), 0, config.vocab)
    first, cache, lengths = prefill(prompts, params, 256, prompt_lengths)
    logits = []
    for t in range(2):  # the first step eagerly (the kernels' masks read the lengths), then jitted
        step = decode_step if t == 0 else jax.jit(decode_step, donate_argnums=1)
        out, cache, lengths = step(new[:, t], cache, lengths, params)
        logits.append(out)
    assert lengths.tolist() == [130, 7, 79]
    for b, n in enumerate([128, 5, 77]):
        want, keys, _ = _model(np.concatenate([np.asarray(prompts[b, :n]), np.asarray(new[b])]), params)
        _close(first[b], want[n - 1], _tol(dtype))
        _close(jnp.stack([logits[0][b], logits[1][b]]), want[n:], _tol(dtype))
        _close(cache[-1]["k"][b, : n + 2], keys[-1], _tol(dtype))  # appended at each token's position


@pytest.mark.cpu
@pytest.mark.parametrize(("config", "dtype"), MODELS, ids=IDS)
def test_greedy_generation_matches_numpy(config, dtype):
    params = init_params(config, dtype)
    prompts = jax.random.randint(jax.random.key(3), (2, 128), 0, config.vocab)
    generated = generate(prompts, params, steps=4)
    assert generated.shape == (2, 4)
    for b in range(2):
        tokens = np.concatenate([np.asarray(prompts[b]), np.asarray(generated[b])])
        logits = _model(tokens[:-1], params)[0][127:]  # what predicts each generated token
        chosen = logits[np.arange(4), np.asarray(generated[b])]
        # Greedy: each token is NumPy's argmax, up to rounding of near ties.
        assert (logits.max(-1) - chosen <= _tol(dtype) * np.abs(logits).max()).all()


@pytest.mark.cpu
@pytest.mark.parametrize(("config", "dtype"), MODELS, ids=IDS)
def test_baselines_match_numpy(config, dtype):
    # The benchmark's baselines: plain JAX with splash attention (its interpreter off a TPU) for prefill, and
    # XLA attention over the cache for decode; MoE experts by ragged_dot and, for decode, every expert.
    params = init_params(config, dtype)
    base = baseline_params(params)
    assert "w_gate" not in base["layers"][0].get("moe", {})  # the experts' weights held once
    tokens = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
    last, cache, lengths = baseline_prefill(config, 256, interpret=True)(tokens, base, 384)
    new = jax.random.randint(jax.random.key(2), (2,), 0, config.vocab)
    experts = ("ragged_dot", "dense") if config.moe else ("ragged_dot",)
    steps = [baseline_decode(config, "xla", e)(new, jax.tree.map(jnp.copy, cache), lengths, base)[0] for e in experts]
    for b in range(2):
        want = _model(np.concatenate([np.asarray(tokens[b]), np.asarray(new[b : b + 1])]), params)[0]
        _close(last[b], want[255], _tol(dtype))
        for logits in steps:
            _close(logits[b], want[256], _tol(dtype))


# --------------------------------------------------------------------------- Mosaic for a v5e, at scale


def _unique(calls):
    seen, out = set(), []
    for call in calls:
        key = (call.kernel, call.specs)
        if key not in seen:
            seen.add(key)
            out.append(call)
    return out


@pytest.mark.parametrize("name", ["dense", "moe"])
def test_every_kernel_lowers_for_v5e_at_benchmark_shapes(name):
    config = CONFIGS[name]
    calls = [c for b, s in ((1, 4096), (4, 2048)) for c in prefill_calls(config, b, s)]
    calls += [c for b in (1, 8, 32) for c in decode_calls(config, b, 4096)]
    for call in _unique(calls):
        lowered = call.kernel.lower(*call.specs)  # the default schedule (off a TPU, tuning keeps it)
        assert "tpu_custom_call" in lower_for_v5e(lowered, *call.specs), call.name
    phases = device_ms(decode_calls(config, 8, 4096), config.layers)
    assert set(phases) == ({"projections", "attention", "moe"} if config.moe else {"projections", "attention", "ffn"})


# --------------------------------------------------------------------------- on a TPU v5e: against the baselines


@pytest.mark.tpu
@pytest.mark.parametrize("name", ["dense", "moe"])
def test_models_match_the_plain_jax_baselines_on_tpu(name):
    config = dataclasses.replace(CONFIGS[name], layers=2)
    params = init_params(config)
    base = baseline_params(params)
    # (MoE: a token's router near a tie may pick another expert in one model than in the other.)
    share = 1.0 if config.moe is None else 0.75
    tokens = jax.random.randint(jax.random.key(1), (8, 512), 0, config.vocab)
    ours, _, _ = prefill(tokens, params, 1024)
    theirs, _, _ = baseline_prefill(config, 512)(tokens, base, 1024)
    assert _rows_agreeing(ours, theirs) >= share

    lengths = jnp.asarray(np.random.default_rng(0).integers(512, 1536, 8), I32)
    cache = random_cache(config, lengths, 2048)
    token = jax.random.randint(jax.random.key(2), (8,), 0, config.vocab)
    ours, _, _ = decode_step(token, jax.tree.map(jnp.copy, cache), lengths, params)
    theirs, _, _ = baseline_decode(config, "xla")(token, jax.tree.map(jnp.copy, cache), lengths, base)
    assert _rows_agreeing(ours, theirs) >= share
    tables = jnp.asarray(page_tables(8, 2048, 128))
    pages = [
        {"kv": ragged_kv_pages(to_pages(c["k"], tables), to_pages(c["v"], tables), config.kv_heads)} for c in cache
    ]
    ragged, _, _ = baseline_decode(config, "ragged")(token, pages, lengths, base, tables)
    assert _rows_agreeing(ours, ragged) >= share
