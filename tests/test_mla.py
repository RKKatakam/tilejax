"""Multi-head latent attention (examples/mla.py): each kernel and the whole block against float64 NumPy, the
absorbed form against the expanded one, the baselines, and cross-lowering at realistic sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ON_TPU
from helpers import f64, lower_for_v5e, run_checking_races, spec
from matmul import MatmulTiles, matmul
from mla import (
    LANES,
    NOPE,
    QK,
    ROPE,
    AttentionTiles,
    DownTiles,
    MlaConfig,
    baseline,
    init_params,
    mla_absorbed,
    mla_attention,
    mla_block,
    mla_down,
    pack,
    rope_tables,
    splash_baseline,
)

B16 = jnp.bfloat16

# --------------------------------------------------------------------------- float64 NumPy references


def _rms(t, gamma, eps=1e-6):
    return t / np.sqrt(np.mean(t * t, axis=-1, keepdims=True) + eps) * f64(gamma).reshape(-1)


def _rope(t, seq):
    """Rotate-half RoPE on the last dim (64) of t (B, S, ..., 64)."""
    inv_freq = 10000.0 ** (-np.arange(0, ROPE, 2) / ROPE)
    angles = np.arange(seq)[:, None] * np.concatenate([inv_freq, inv_freq])[None, :]
    shape = (seq,) + (1,) * (t.ndim - 3) + (ROPE,)
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    rotated = np.concatenate([-t[..., ROPE // 2 :], t[..., : ROPE // 2]], axis=-1)
    return t * cos + rotated * sin


def _attention(q_nope, q_rope, k_nope, k_rope, v, scale):
    """Causal attention with a shared rope key: (B, S, H, e) per-head arrays, k_rope (B, S, 64)."""
    s = np.einsum("bqhe,bkhe->bhqk", q_nope, k_nope) + np.einsum("bqhe,bke->bhqk", q_rope, k_rope)
    s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s * scale, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    return np.einsum("bhqk,bkhe->bqhe", p / p.sum(-1, keepdims=True), v)


def _down(x, p):
    """c_q, c_kv, k_rope (B, S, 64) of the block, in float64."""
    h = _rms(f64(x), p["attn_norm"])
    s = x.shape[1]
    return _rms(h @ f64(p["w_dq"]), p["q_norm"]), _rms(h @ f64(p["w_dkv"]), p["kv_norm"]), _rope(h @ f64(p["w_kr"]), s)


def _mla(x, p):
    """The MLA block in float64 NumPy, from the model's weights."""
    c_q, c_kv, k_rope = _down(x, p)
    q = np.einsum("bsr,rhe->bshe", c_q, f64(p["w_uq"]))
    k_nope, v = (np.einsum("bsr,rhe->bshe", c_kv, f64(p[w])) for w in ("w_uk", "w_uv"))
    o = _attention(q[..., :NOPE], _rope(q[..., NOPE:], x.shape[1]), k_nope, k_rope, v, QK**-0.5)
    return f64(x) + np.einsum("bqhe,hed->bqd", o, f64(p["w_o"]))


def _attention_layer(x, q, kv, k_rope, w_o):
    """mla_attention's math in float64 NumPy, from its packed inputs (q pre-scaled, its rope part unrotated)."""
    b, s, _ = x.shape
    h = w_o.shape[0]
    q, kv = f64(q), f64(kv)
    q_nope, q_rope = q[..., : h * NOPE].reshape(b, s, h, NOPE), q[..., h * NOPE :].reshape(b, s, h, ROPE)
    k_nope, v = kv[..., : h * NOPE].reshape(b, s, h, NOPE), kv[..., h * NOPE :].reshape(b, s, h, NOPE)
    o = _attention(q_nope, _rope(q_rope, s), k_nope, f64(k_rope)[..., :ROPE], v, 1.0)
    return f64(x) + np.einsum("bqhe,hed->bqd", o, f64(w_o))


def _close(out, want, tol=3e-2):
    np.testing.assert_allclose(f64(out), want, atol=tol * np.abs(want).max())


# --------------------------------------------------------------------------- inputs


def _x(config, batch, seq, seed=1):
    return jax.random.normal(jax.random.key(seed), (batch, seq, config.d_model), B16)


def _attention_inputs(batch, seq, heads, d_model, seed=0):
    """Random x, q (pre-scaled), kv, [k_rope | k_rope] and W_o for mla_attention."""
    keys = jax.random.split(jax.random.key(seed), 5)
    x = jax.random.normal(keys[0], (batch, seq, d_model), B16)
    q = (jax.random.normal(keys[1], (batch, seq, heads * QK)) * QK**-0.5).astype(B16)
    kv = jax.random.normal(keys[2], (batch, seq, 2 * heads * NOPE), B16)
    k_rope = jax.random.normal(keys[3], (batch, seq, ROPE), B16)
    w_o = (jax.random.normal(keys[4], (heads, NOPE, d_model)) / (heads * NOPE) ** 0.5).astype(B16)
    return x, q, kv, jnp.concatenate([k_rope, k_rope], axis=-1), w_o


# --------------------------------------------------------------------------- the kernels


@pytest.mark.parametrize(
    ("q_rank", "tiles"),
    [
        (256, DownTiles(rows=128, block_d=128)),  # two reduction steps
        (384, DownTiles(rows=256, block_d=256)),  # the whole model dim at once: W_down resident
    ],
)
def test_mla_down(q_rank, tiles):
    config = MlaConfig(d_model=256, heads=2, q_rank=q_rank, kv_rank=256)
    params = init_params(config)
    packed = pack(params)
    x = _x(config, 2, 256)
    cos, sin = rope_tables(256)
    args = (x, packed["attn_norm"], packed["w_down"], packed["q_norm"], packed["kv_norm"], cos, sin, tiles)
    c_q, c_kv, k_rope = run_checking_races(mla_down, *args)
    want_q, want_kv, want_rope = _down(x, params)
    assert (c_q.shape, c_kv.shape, k_rope.shape) == ((2, 256, q_rank), (2, 256, 256), (2, 256, LANES))
    _close(c_q, want_q)
    _close(c_kv, want_kv)
    _close(k_rope, np.concatenate([want_rope, want_rope], axis=-1))  # the rope key, twice per lane slot
    mla_down.verify(*args)
    lowered = mla_down.lower(*args)
    resident = tiles.block_d == config.d_model
    assert {op.name: op.kind.value for op in lowered.plan.operands}["w_down"] == ("once" if resident else "stream")


@pytest.mark.parametrize(
    ("heads", "tiles"),
    [
        (2, AttentionTiles(block_q=128, block_kv=128)),
        (4, AttentionTiles(block_q=256, block_kv=128)),
        (4, AttentionTiles(block_q=128, block_kv=256)),
    ],
)
def test_mla_attention(heads, tiles):
    inputs = _attention_inputs(1, 512, heads, 256)
    cos, sin = rope_tables(512)
    x, q, kv, k_rope, w_o = inputs
    args = (x, q, kv, k_rope, cos, sin, w_o, tiles)
    out = run_checking_races(mla_attention, *args)
    _close(out, _attention_layer(*inputs))
    mla_attention.verify(*args)
    lowered = mla_attention.lower(*args)
    assert [phase.range.var for phase in lowered.plan.phases] == ["j", "hh"]
    bq, bk = tiles.block_q, tiles.block_kv
    assert lowered.steps == sum(i // bk + 1 + heads for i in range(bq - 1, 512, bq))  # future key blocks skipped
    kinds = {op.name: op.kind.value for op in lowered.plan.operands}
    assert (kinds["kv"], kinds["k_rope"], kinds["w_o"], kinds["q"]) == ("stream", "stream", "stream", "tile")


def test_mla_attention_f32():
    x, q, kv, k_rope, w_o = (t.astype(jnp.float32) for t in _attention_inputs(1, 256, 2, 256))
    cos, sin = rope_tables(256)
    out = run_checking_races(mla_attention, x, q, kv, k_rope, cos, sin, w_o, AttentionTiles(block_q=128, block_kv=128))
    _close(out, _attention_layer(x, q, kv, k_rope, w_o), 1e-4)


def test_up_projections_are_matmuls():
    config = MlaConfig(d_model=256, heads=2, q_rank=256, kv_rank=256)
    params = init_params(config)
    packed = pack(params)
    c = jax.random.normal(jax.random.key(3), (256, 256), B16)
    tiles = MatmulTiles(tm=128, tn=128, tk=256)
    q = run_checking_races(matmul, c, packed["w_uq"], tiles)
    want = np.einsum("sr,rhe->she", f64(c), f64(params["w_uq"])) * QK**-0.5
    _close(q, np.concatenate([want[..., :NOPE].reshape(256, -1), want[..., NOPE:].reshape(256, -1)], axis=1))
    kv = run_checking_races(matmul, c, packed["w_ukv"], tiles)
    k_nope, v = (np.einsum("sr,rhe->she", f64(c), f64(params[w])).reshape(256, -1) for w in ("w_uk", "w_uv"))
    _close(kv, np.concatenate([k_nope, v], axis=1))


# --------------------------------------------------------------------------- the whole block and the references


@pytest.mark.parametrize(("heads", "q_rank"), [(2, 256), (4, 384)])
def test_mla_block_matches_numpy(heads, q_rank):
    config = MlaConfig(d_model=256, heads=heads, q_rank=q_rank, kv_rank=256)
    params = init_params(config)
    x = _x(config, 2, 256)
    out = jax.jit(mla_block)(x, pack(params))  # the default tiles off a TPU, tuned on one
    want = _mla(x, params)
    assert out.shape == x.shape and out.dtype == x.dtype
    _close(out, want, 5e-2)
    _close(baseline(x, params), want, 5e-2)


def test_absorbed_form_equals_the_expanded_form_in_float64():
    config = MlaConfig(d_model=256, heads=4, q_rank=128, kv_rank=128)
    params = init_params(config, jnp.float32)
    with jax.enable_x64(True):
        params = {name: jnp.asarray(f64(w)) for name, w in params.items()}
        x = jnp.asarray(f64(_x(config, 2, 128)))
        expanded, absorbed = baseline(x, params), mla_absorbed(x, params)
        assert expanded.dtype == absorbed.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(absorbed), np.asarray(expanded), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(expanded), _mla(x, params), rtol=1e-10, atol=1e-10)


def test_absorbed_form_in_bf16():
    config = MlaConfig(d_model=256, heads=2, q_rank=128, kv_rank=256)
    params = init_params(config)
    x = _x(config, 1, 256)
    _close(jax.jit(mla_absorbed)(x, params), _mla(x, params), 5e-2)


@pytest.mark.parametrize("pad", [False, True])
def test_splash_baseline(pad):
    config = MlaConfig(d_model=256, heads=2, q_rank=128, kv_rank=256)
    params = init_params(config)
    x = _x(config, 1, 256)
    out = splash_baseline(config.heads, 256, 128, 128, pad=pad, interpret=not ON_TPU)(x, params)
    _close(out, _mla(x, params), 5e-2)


def test_pack_layouts():
    config = MlaConfig(d_model=256, heads=2, q_rank=128, kv_rank=256)
    packed = pack(init_params(config))
    assert {name: w.shape for name, w in packed.items()} == {
        "attn_norm": (1, 256),
        "w_down": (256, 128 + 256 + LANES),
        "q_norm": (1, 128),
        "kv_norm": (1, 256),
        "w_uq": (128, 2 * QK),
        "w_ukv": (256, 2 * 2 * NOPE),
        "w_o": (2, NOPE, 256),
    }
    cos, sin = rope_tables(16)
    assert cos.shape == sin.shape == (16, LANES)
    np.testing.assert_allclose(f64(cos[:, :ROPE]), f64(cos[:, ROPE:]))  # two 64-dim groups per lane slot


# --------------------------------------------------------------------------- cross-lowering at realistic sizes


@pytest.mark.parametrize(
    ("batch", "seq", "d_model", "q_rank", "tiles"),
    [
        (2, 2048, 2048, 768, DownTiles()),
        (1, 4096, 2048, 1536, DownTiles(rows=512, block_d=2048)),  # W_down resident
        (2, 2048, 4096, 1536, DownTiles(rows=256, block_d=1024)),
    ],
)
def test_mla_down_lowers_at_scale(batch, seq, d_model, q_rank, tiles):
    specs = (spec(batch, seq, d_model, dtype=B16), spec(1, d_model, dtype=B16))
    specs += (spec(d_model, q_rank + 512 + LANES, dtype=B16), spec(1, q_rank, dtype=B16), spec(1, 512, dtype=B16))
    specs += (spec(seq, LANES), spec(seq, LANES))
    assert "tpu_custom_call" in lower_for_v5e(mla_down.lower(*specs, tiles), *specs)


@pytest.mark.parametrize(
    ("batch", "seq", "d_model", "heads", "tiles"),
    [
        (2, 2048, 2048, 16, AttentionTiles()),
        (1, 4096, 2048, 16, AttentionTiles(block_q=512, block_kv=512)),
        (1, 2048, 2048, 32, AttentionTiles(block_q=256, block_kv=512)),  # Kimi K2's ratio of heads to d_model
        (1, 2048, 4096, 64, AttentionTiles(block_q=128, block_kv=256)),
    ],
)
def test_mla_attention_lowers_at_scale(batch, seq, d_model, heads, tiles):
    specs = (spec(batch, seq, d_model, dtype=B16), spec(batch, seq, heads * QK, dtype=B16))
    specs += (spec(batch, seq, 2 * heads * NOPE, dtype=B16), spec(batch, seq, LANES, dtype=B16))
    specs += (spec(seq, LANES), spec(seq, LANES), spec(heads, NOPE, d_model, dtype=B16))
    assert "tpu_custom_call" in lower_for_v5e(mla_attention.lower(*specs, tiles), *specs)


@pytest.mark.parametrize(("rank", "columns"), [(768, 16 * QK), (1536, 32 * QK), (512, 2 * 32 * NOPE)])
def test_up_projections_lower_at_scale(rank, columns):
    specs = (spec(4096, rank, dtype=B16), spec(rank, columns, dtype=B16))
    assert "tpu_custom_call" in lower_for_v5e(matmul.lower(*specs), *specs)


# --------------------------------------------------------------------------- on a TPU v5e


@pytest.mark.tpu
@pytest.mark.parametrize(
    ("batch", "seq", "d_model", "heads", "q_rank"), [(1, 2048, 2048, 16, 768), (1, 2048, 2048, 32, 1536)]
)
def test_mla_block_matches_the_baselines_on_tpu(batch, seq, d_model, heads, q_rank):
    config = MlaConfig(d_model=d_model, heads=heads, q_rank=q_rank)
    params = init_params(config)
    x = _x(config, batch, seq)
    out = f64(jax.jit(mla_block)(x, pack(params)))
    scale = np.abs(out).max()
    for name, want in (
        ("XLA", jax.jit(baseline)(x, params)),
        ("splash", splash_baseline(heads, seq)(x, params)),
        ("splash, padded", splash_baseline(heads, seq, pad=True)(x, params)),
    ):
        diff = np.abs(out - f64(want)).max()
        assert diff < 3e-2 * scale, (name, diff, scale)
