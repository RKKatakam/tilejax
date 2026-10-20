"""The Qwen3 kernels (examples/qwen3.py, gqa_attention.py, swiglu.py) against float64 NumPy, their own loop
nests, Mosaic for a v5e, and (on a TPU) JAX's splash attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ON_TPU
from gqa_attention import AttentionTiles, LayerTiles, make_attention, make_inputs, splash_gqa
from helpers import f64, lower_for_v5e, run_checking_races, spec
from qwen3 import (
    Config,
    ProjectionTiles,
    baseline,
    forward,
    init_params,
    make_qkv,
    qk_weights,
    rope_tables,
    with_fused_projections,
)
from swiglu import FfnTiles, swiglu_ffn

EPS = 1e-6
B16 = jnp.bfloat16

# --------------------------------------------------------------------------- float64 NumPy references


def _rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * f64(w).reshape(-1)


def _heads(t, dh):
    """(B, S, n * Dh) -> (B, n, S, Dh)."""
    return t.reshape(t.shape[0], t.shape[1], -1, dh).transpose(0, 2, 1, 3)


def _unheads(t):
    return t.transpose(0, 2, 1, 3).reshape(t.shape[0], t.shape[2], -1)


def _rope(t, positions, rotary_dim, theta):
    """Rotate-half RoPE of (..., S, Dh) on the first rotary_dim dims, as Hugging Face's Qwen3 and Qwen3-Next."""
    half = rotary_dim // 2
    inv_freq = theta ** (-np.arange(half) * 2 / rotary_dim)
    angles = np.asarray(positions, np.float64)[:, None] * np.concatenate([inv_freq, inv_freq])[None]
    r, still = t[..., :rotary_dim], t[..., rotary_dim:]
    rotated = np.concatenate([-r[..., half:], r[..., :half]], axis=-1)
    return np.concatenate([r * np.cos(angles) + rotated * np.sin(angles), still], axis=-1)


def _qkv(x, gamma, w_q, w_k, w_v, q_norm, k_norm, dh, rotary_dim, theta=1e6, w_gate=None, offset=0):
    """q (scaled by 1/sqrt(dh)), k, v (and the gate) of a layer input, (B, S, n * Dh) each, its rows at
    positions offset, offset + 1, ...; no QK-norm if q_norm is None."""
    n = _rms(f64(x), gamma)
    q, k = _heads(n @ f64(w_q), dh), _heads(n @ f64(w_k), dh)
    if q_norm is not None:
        q, k = _rms(q, q_norm), _rms(k, k_norm)
    positions = offset + np.arange(x.shape[1])
    q, k = _rope(q, positions, rotary_dim, theta) * dh**-0.5, _rope(k, positions, rotary_dim, theta)
    out = (_unheads(q), _unheads(k), n @ f64(w_v))
    return out + ((1 / (1 + np.exp(-(n @ f64(w_gate)))),) if w_gate is not None else ())


def _attention(x, q, k, v, wo, dh, window=None, gate=None):
    """x + GQA attention (q pre-scaled; causal, or sliding-window) @ wo."""
    q, k, v = (_heads(f64(t), dh) for t in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
    s = q.shape[2]
    rows, cols = np.arange(s)[:, None], np.arange(s)[None, :]
    visible = cols <= rows
    if window is not None:
        visible &= rows - cols < window
    scores = np.where(visible, q @ k.transpose(0, 1, 3, 2), -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    a = _unheads((p / p.sum(-1, keepdims=True)) @ v)
    if gate is not None:
        a = a * f64(gate)
    return f64(x) + a @ f64(wo)


def _swiglu(x, gamma, w_gate, w_up, w_down):
    n = _rms(f64(x), gamma)
    g, u = n @ f64(w_gate), n @ f64(w_up)
    return f64(x) + (g / (1 + np.exp(-g)) * u) @ f64(w_down)


def _model(tokens, params):
    """The model in float64 NumPy, from the layer references above."""
    config: Config = params["config"]
    dh = config.head_dim
    x = f64(params["embed"])[np.asarray(tokens)]
    for layer in params["layers"]:
        norms = (layer["q_norm"], layer["k_norm"]) if config.qk_norm else (None, None)
        qkv = _qkv(
            x,
            layer["attn_norm"],
            layer["w_q"],
            layer["w_k"],
            layer["w_v"],
            *norms,
            dh,
            config.rotary_dim,
            config.rope_theta,
            layer.get("w_attn_gate"),
        )
        x = _attention(x, *qkv[:3], layer["wo"], dh, config.window, qkv[3] if config.gated else None)
        x = _swiglu(x, layer["ffn_norm"], layer["w_gate"], layer["w_up"], layer["w_down"])
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"])


def _close(got, want, tol=3e-2):
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


# --------------------------------------------------------------------------- RoPE tables


@pytest.mark.cpu  # a TPU computes f32 cos/sin of large angles (2000 rad) to ~2e-3, not to NumPy's float64
@pytest.mark.parametrize("rotary_dim", [None, 32])
def test_rope_tables_rotate_like_numpy(rotary_dim):
    t = np.random.default_rng(0).normal(size=(64, 128))
    cos, sin = rope_tables(jnp.arange(64) + 1000, 128, 1e6, rotary_dim)  # positions with an offset
    rd = rotary_dim or 128
    half = rd // 2
    rotated = np.concatenate([-t[:, half:rd], t[:, :half], t[:, rd:]], axis=1)  # past rd: anything (sin 0)
    got = t * f64(cos) + rotated * f64(sin)
    # (f32 tables, as Hugging Face's: angles of positions near 1000 are good to ~1e-4)
    np.testing.assert_allclose(got, _rope(t, np.arange(64) + 1000, rd, 1e6), atol=5e-4)


# --------------------------------------------------------------------------- the attention-input kernel


def _projection_inputs(b, s, d, heads, kv_heads, dh, seed=0):
    keys = jax.random.split(jax.random.key(seed), 8)
    hd, kd = heads * dh, kv_heads * dh
    x = jax.random.normal(keys[0], (b, s, d), B16)
    gamma = (1 + 0.1 * jax.random.normal(keys[1], (1, d))).astype(B16)
    w_q, w_gate = ((jax.random.normal(key, (d, hd)) / d**0.5).astype(B16) for key in keys[2:4])
    w_k, w_v = ((jax.random.normal(key, (d, kd)) / d**0.5).astype(B16) for key in keys[4:6])
    q_norm, k_norm = ((1 + 0.1 * jax.random.normal(key, (1, dh))).astype(B16) for key in keys[6:8])
    return x, gamma, w_q, w_k, w_v, w_gate, q_norm, k_norm


@pytest.mark.parametrize(
    ("qk_norm", "rotary_dim", "gated", "block_d", "offset"),
    [
        (True, None, False, 128, 0),  # Qwen3; two steps over d_model: carried accumulators
        (True, None, True, 256, 0),  # gated attention's gate too; d_model in one step
        (False, None, False, 128, 0),  # Qwen2.5 / Llama: no QK-norm
        (True, 32, True, 128, 0),  # Qwen3-Next: partial RoPE, gated
        (True, None, False, 256, 777),  # rows at positions 777 ...: the tables carry the positions
    ],
)
def test_qkv_projection_matches_numpy(qk_norm, rotary_dim, gated, block_d, offset):
    dh = 128
    x, gamma, w_q, w_k, w_v, w_gate, q_norm, k_norm = _projection_inputs(1, 256, 256, 4, 2, dh)
    cos, sin = rope_tables(offset + jnp.arange(256), dh, 1e6, rotary_dim)
    config = Config(1, 256, 4, 2, dh, 512, 256, qk_norm=qk_norm, rotary_fraction=(rotary_dim or dh) / dh)
    q_weight, k_weight = qk_weights({"q_norm": q_norm, "k_norm": k_norm}, config)
    kernels = make_qkv(qk_norm, rotary_dim)
    tiles = ProjectionTiles(rows=128, block_d=block_d)
    if gated:
        args = (x, gamma, w_q, w_k, w_v, w_gate, q_weight, k_weight, cos, sin, tiles)
        kernel = kernels.gated_projection
    else:
        args = (x, gamma, w_q, w_k, w_v, q_weight, k_weight, cos, sin, tiles)
        kernel = kernels.projection
    got = run_checking_races(kernel, *args)
    norms = (q_norm, k_norm) if qk_norm else (None, None)
    want = _qkv(x, gamma, w_q, w_k, w_v, *norms, dh, rotary_dim or dh, w_gate=w_gate if gated else None, offset=offset)
    assert len(got) == len(want) == (4 if gated else 3)
    for g, w in zip(got, want, strict=True):
        _close(g, w)
    kernel.verify(*args)
    lowered = kernel.lower(*args)
    assert set(lowered.plan.carried) >= ({"acc_q", "acc_k", "acc_v", "sum_sq"} if block_d < 256 else set())


def test_qkv_projection_head_dim_256_partial_rope():
    # Qwen3-Next's attention: head_dim 256, the first 64 dims rotated.
    dh = 256
    x, gamma, w_q, w_k, w_v, _, q_norm, k_norm = _projection_inputs(1, 128, 256, 2, 1, dh, seed=1)
    cos, sin = rope_tables(jnp.arange(128), dh, 1e6, 64)
    config = Config(1, 256, 2, 1, dh, 512, 256, rotary_fraction=0.25)
    q_weight, k_weight = qk_weights({"q_norm": q_norm, "k_norm": k_norm}, config)
    kernel = make_qkv(True, 64).projection
    args = (x, gamma, w_q, w_k, w_v, q_weight, k_weight, cos, sin, ProjectionTiles(rows=128, block_d=256))
    got = run_checking_races(kernel, *args)
    for g, w in zip(got, _qkv(x, gamma, w_q, w_k, w_v, q_norm, k_norm, dh, 64), strict=True):
        _close(g, w)


# --------------------------------------------------------------------------- attention


@pytest.mark.parametrize(
    ("heads", "kv_heads", "dh", "window", "tiles"),
    [
        (4, 2, 128, None, AttentionTiles(block_q=128, block_kv=128, block_o=256)),  # 2 heads per kv head
        (4, 1, 128, None, AttentionTiles(block_q=256, block_kv=128, block_o=512)),  # MQA; wo in VMEM
        (4, 2, 128, 192, AttentionTiles(block_q=128, block_kv=128, block_o=128)),  # sliding window
        (4, 2, 64, 128, AttentionTiles(block_q=128, block_kv=256, block_o=128)),  # head_dim 64
    ],
)
def test_gqa_attention_matches_numpy(heads, kv_heads, dh, window, tiles):
    x, q, k, v, _, wo = make_inputs(1, 512, 256, heads, kv_heads, dh)
    kernel = make_attention(dh, window).attention
    out = run_checking_races(kernel, x, q, k, v, wo, tiles)
    _close(out, _attention(x, q, k, v, wo, dh, window))
    kernel.verify(x, q, k, v, wo, tiles)


def test_gated_gqa_attention_matches_numpy():
    x, q, k, v, gate, wo = make_inputs(1, 256, 256, 4, 2)
    tiles = AttentionTiles(block_q=128, block_kv=128, block_o=256)
    kernel = make_attention(128, None).gated_attention
    out = run_checking_races(kernel, x, q, k, v, gate, wo, tiles)
    _close(out, _attention(x, q, k, v, wo, 128, gate=gate))
    kernel.verify(x, q, k, v, gate, wo, tiles)


@pytest.mark.parametrize("window", [None, 256, 1000])
def test_attention_skips_blocks_outside_the_mask(window):
    inputs = make_inputs(2, 1024, 256, 4, 2)
    x, q, k, v, _, wo = inputs
    blocks, bq = 128, 128
    lowered = make_attention(128, window).attention.lower(x, q, k, v, wo, AttentionTiles(block_q=bq, block_kv=blocks))
    visited = 0
    for i in range(0, 1024, bq):
        for j in range(0, 1024, blocks):
            future = j > i + bq - 1
            too_old = window is not None and j + blocks - 1 <= i - window
            visited += not (future or too_old)
    assert lowered.steps == 2 * visited  # (batch 2); the steps that run, and only those
    # block_o 512 is all of wo's rows: the output projection's loop of one block is inlined, wo kept in VMEM.
    assert [p.range.var for p in lowered.plan.phases] == ["j"]


def test_attention_kernels_are_cached_per_variant():
    assert make_attention(128, None) is make_attention(128, None)
    assert make_attention(128, 256) is not make_attention(128, None)
    assert make_qkv(True, None) is make_qkv(True, None)


# --------------------------------------------------------------------------- feed-forward and whole layers


@pytest.mark.parametrize(("rows", "block_f"), [(128, 256), (256, 1024)])
def test_swiglu_ffn_matches_numpy(rows, block_f):
    from swiglu import make_inputs as ffn_inputs

    inputs = ffn_inputs(256, 256, 1024)
    tiles = FfnTiles(rows=rows, block_f=block_f)
    out = run_checking_races(swiglu_ffn, *inputs, tiles)
    _close(out, _swiglu(*inputs))
    swiglu_ffn.verify(*inputs, tiles)
    phases = [p.range.var for p in swiglu_ffn.lower(*inputs, tiles).plan.phases]
    assert phases == (["f"] if block_f < 1024 else [])


@pytest.mark.parametrize(("gated", "window"), [(False, None), (True, 128)])
def test_gqa_layer_in_three_phases(gated, window):
    from swiglu import make_inputs as ffn_inputs

    x, q, k, v, gate, wo = make_inputs(1, 256, 256, 4, 2)
    _, ffn_norm, w_gate, w_up, w_down = ffn_inputs(256, 256, 512, seed=1)
    kernels = make_attention(128, window)
    tiles = LayerTiles(block_q=128, block_kv=128, block_o=256, block_f=256)
    ffn = (ffn_norm, w_gate, w_up, w_down)
    if gated:
        kernel, args = kernels.gated_layer, (x, q, k, v, gate, wo, *ffn, tiles)
    else:
        kernel, args = kernels.layer, (x, q, k, v, wo, *ffn, tiles)
    out = run_checking_races(kernel, *args)
    attended = _attention(x, q, k, v, wo, 128, window, gate if gated else None)
    _close(out, _swiglu(attended, *ffn))
    kernel.verify(*args)
    assert [p.range.var for p in kernel.lower(*args).plan.phases] == ["j", "c", "f"]


# --------------------------------------------------------------------------- the model


MODELS = [
    Config(layers=2, d_model=256, heads=4, kv_heads=2, head_dim=128, ffn=512, vocab=512),  # Qwen3
    Config(  # Qwen2.5-style attention with a sliding window
        layers=1, d_model=256, heads=4, kv_heads=1, head_dim=128, ffn=512, vocab=256, qk_norm=False, window=128
    ),
    Config(  # Qwen3-Next-style attention: gated, partial RoPE
        layers=1, d_model=256, heads=2, kv_heads=1, head_dim=256, ffn=512, vocab=256, gated=True, rotary_fraction=0.25
    ),
]


@pytest.mark.parametrize("config", MODELS, ids=["qwen3", "window", "gated"])
def test_model_matches_numpy(config):
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
    logits = jax.jit(forward)(tokens, params)
    assert logits.shape == (2, 256, config.vocab)
    _close(logits, _model(tokens, params), 5e-2)


@pytest.mark.parametrize("config", MODELS, ids=["qwen3", "window", "gated"])
def test_baseline_matches_numpy(config):
    # The benchmarks' baseline: plain JAX, and splash attention (in its interpreter off a TPU), both GQA forms,
    # with q, k, v weights apart and side by side.
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (1, 256), 0, config.vocab)
    want = _model(tokens, params)
    for form, weights in (("mqa", params), ("mha", with_fused_projections(params))):
        _close(baseline(config, 256, form=form, interpret=not ON_TPU)(tokens, weights), want, 5e-2)


def test_split_layer_matches_fused():
    from qwen3 import attention_input, fuse_ffn, fused_layer, split_layer

    config = MODELS[0]
    params = init_params(config)
    x = params["embed"][jax.random.randint(jax.random.key(2), (1, 256), 0, config.vocab)]
    cos, sin = rope_tables(jnp.arange(256), 128)
    layer = params["layers"][0]
    qkv = attention_input(x, layer, config, cos, sin)
    fused, split = fused_layer(x, qkv, layer, config), split_layer(x, qkv, layer, config)
    want = _swiglu(_attention(x, *qkv, layer["wo"], 128), *(layer[w] for w in ("ffn_norm", "w_gate", "w_up", "w_down")))
    _close(fused, want)
    _close(split, want)
    if not ON_TPU:
        assert fuse_ffn(x, qkv, layer, config)  # nothing measured off a TPU: fused


# --------------------------------------------------------------------------- Mosaic for a v5e, at scale


SCALES = [  # B, S, d_model, heads, kv_heads, ffn
    (2, 2048, 1024, 16, 8, 3072),
    (1, 4096, 2048, 16, 4, 6144),
    (2, 2048, 2560, 32, 8, 9728),
    (1, 4096, 4096, 32, 8, 12288),
]


@pytest.mark.parametrize(("b", "s", "d", "heads", "kv_heads", "f"), SCALES)
def test_kernels_lower_to_mosaic_at_scale(b, s, d, heads, kv_heads, f):
    dh = 128
    hd, kd = heads * dh, kv_heads * dh
    x, gamma = spec(b, s, d, dtype=B16), spec(1, d, dtype=B16)
    w_q, w_kv = spec(d, hd, dtype=B16), spec(d, kd, dtype=B16)
    tables = (spec(1, dh), spec(1, dh), spec(s, dh), spec(s, dh))
    q, kv = spec(b, s, hd, dtype=B16), spec(b, s, kd, dtype=B16)
    ffn = (gamma, spec(d, f, dtype=B16), spec(d, f, dtype=B16), spec(f, d, dtype=B16))
    qwen3, qwen3_next, window = make_qkv(True, None), make_qkv(True, dh // 4), make_attention(dh, 1024)
    attention = make_attention(dh, None)
    cases = [
        (qwen3.projection, (x, gamma, w_q, w_kv, w_kv, *tables)),
        (make_qkv(False, None).projection, (x, gamma, w_q, w_kv, w_kv, *tables)),
        (qwen3_next.gated_projection, (x, gamma, w_q, w_kv, w_kv, w_q, *tables)),
        (attention.attention, (x, q, kv, kv, spec(hd, d, dtype=B16))),
        (window.attention, (x, q, kv, kv, spec(hd, d, dtype=B16))),
        (attention.gated_attention, (x, q, kv, kv, q, spec(hd, d, dtype=B16))),
        (attention.layer, (x, q, kv, kv, spec(hd, d, dtype=B16), *ffn)),
        (window.gated_layer, (x, q, kv, kv, q, spec(hd, d, dtype=B16), *ffn)),
        (swiglu_ffn, (spec(b * s, d, dtype=B16), *ffn)),
    ]
    for n, (kernel, specs) in enumerate(cases):
        lowered = kernel.lower(*specs)
        if n % 2:  # with trace regions, as tuning profiles it
            lowered = kernel.lowered(kernel.signature(*specs), lowered.config, trace=True)
        assert "tpu_custom_call" in lower_for_v5e(lowered, *specs), kernel


# --------------------------------------------------------------------------- on a TPU v5e: against splash


@pytest.mark.tpu
@pytest.mark.parametrize(("window", "form"), [(None, "mqa"), (None, "mha"), (1024, "mqa")])
def test_gqa_attention_matches_splash_on_tpu(window, form):
    b, s, d, heads, kv_heads = 2, 2048, 1024, 16, 4
    x, q, k, v, _, wo = make_inputs(b, s, d, heads, kv_heads)
    out = make_attention(128, window).attention(x, q, k, v, wo, AttentionTiles(block_q=256, block_kv=512))
    a = splash_gqa(heads, kv_heads, s, window, 512, 512, form)(q, k, v)
    want = (x.astype(jnp.float32) + jnp.dot(a, wo, preferred_element_type=jnp.float32)).astype(B16)
    np.testing.assert_allclose(f64(out), f64(want), atol=3e-2 * float(jnp.abs(want.astype(jnp.float32)).max()))


@pytest.mark.tpu
def test_model_matches_splash_baseline_on_tpu():
    config = Config(layers=2, d_model=1024, heads=16, kv_heads=8, head_dim=128, ffn=3072, vocab=4096)
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (2, 1024), 0, config.vocab)
    logits = jax.jit(forward)(tokens, params)
    want = baseline(config, 1024)(tokens, params)
    np.testing.assert_allclose(f64(logits), f64(want), atol=5e-2 * float(jnp.abs(want.astype(jnp.float32)).max()))
