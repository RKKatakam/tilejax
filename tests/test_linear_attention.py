"""Linear attention with the delta rule (Gated DeltaNet, KDA) and Qwen3-Next's Gated DeltaNet layer, against the
recurrence in float64 NumPy and the chunked algorithm in plain JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from linear_attention import (
    HEAD_DIM,
    KDA_GATE_MIN,
    DeltaTiles,
    baseline_chunked,
    baseline_recurrent,
    gated_delta_rule,
    gated_delta_rule_gated,
    kimi_delta_attention,
    make_inputs,
    reference_recurrent,
)

KERNELS = {False: gated_delta_rule, True: kimi_delta_attention}  # by: decays per key channel?


def _assert_close(got, want, tol=2e-2):
    """Within `tol` of the largest value: bf16 rounds at every step of the chunked algorithm."""
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize(
    ("batch", "heads", "seq", "tiles"),
    [
        (1, 2, 1024, DeltaTiles(heads=2, block=128)),  # two steps per phase: the state crosses steps and phases
        (2, 2, 512, DeltaTiles(heads=1, block=128)),  # a head per tile, a step per phase
        (1, 4, 1024, DeltaTiles(heads=2, block=256)),  # two chunks per step
    ],
)
def test_kernels_match_the_recurrence(per_channel, batch, heads, seq, tiles):
    kernel = KERNELS[per_channel]
    inputs = make_inputs(batch, heads, seq, per_channel)
    out = run_checking_races(kernel, *inputs, tiles)
    assert out.shape == (batch, seq, heads * HEAD_DIM) and out.dtype == jnp.bfloat16
    _assert_close(out, reference_recurrent(*inputs))
    kernel.verify(*inputs, tiles)


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_kernel_structure(per_channel):
    kernel = KERNELS[per_channel]
    lowered = kernel.lower(*make_inputs(2, 4, 2048, per_channel), DeltaTiles(heads=2, block=256))
    # A tile per (batch, pair of heads); a block of tokens per step, its outputs stored by the step.
    assert (lowered.tiles, lowered.steps) == (4, 4 * 2048 // 256)
    assert [phase.range.var for phase in lowered.plan.phases] == ["c"]
    assert {op.kind.value for op in lowered.plan.operands} == {"stream"}
    assert set(lowered.plan.carried) == {"state"} and lowered.plan.outputs[0].phase == 0
    assert lowered.plan.carried["state"].shape == (2, HEAD_DIM, HEAD_DIM)


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_baselines_match_the_recurrence(per_channel):
    inputs = make_inputs(2, 2, 512, per_channel)
    want = reference_recurrent(*inputs)
    _assert_close(baseline_chunked(*inputs), want)
    _assert_close(baseline_recurrent(*inputs), want, tol=1e-2)
    # The chunked baseline is the kernels' math: they agree to a bf16 rounding or two.
    _assert_close(KERNELS[per_channel](*inputs, DeltaTiles(heads=1, block=128)), f64(baseline_chunked(*inputs)), 1e-2)


@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_out_of_range_gates_are_clamped(per_channel):
    # Tuning runs kernels on normal random inputs: positive log decays, betas outside [0, 1], decays below
    # KDA's bound. The kernels clamp them (as the recurrence here does), so outputs stay finite and exact.
    b, h, s, d = 1, 2, 512, 2 * HEAD_DIM
    q, k, v = (rand((b, s, d), seed, jnp.bfloat16) for seed in range(3))
    g = 3 * rand((b, s, d) if per_channel else (b, h, s), 3)
    beta = 2 * rand((b, h, s), 4)
    if per_channel:
        assert float(g.min()) < KDA_GATE_MIN
    out = KERNELS[per_channel](q, k, v, g, beta, DeltaTiles(heads=2, block=128))
    assert np.isfinite(f64(out)).all()
    _assert_close(out, reference_recurrent(q, k, v, g, beta))


def _gated_rmsnorm(o, z, weight, eps=1e-6):
    """Qwen3-Next's gated RMSNorm per head in float64 NumPy: (B, S, H * HEAD_DIM)."""
    shape = o.shape
    o, z = f64(o).reshape(*shape[:-1], -1, HEAD_DIM), f64(z).reshape(*shape[:-1], -1, HEAD_DIM)
    normed = o / np.sqrt(np.mean(o * o, axis=-1, keepdims=True) + eps) * f64(weight)[0]
    return (normed * z / (1 + np.exp(-z))).reshape(shape)


def test_gated_delta_rule_gated():
    q, k, v, g, beta = make_inputs(1, 2, 1024)
    z = rand((1, 1024, 3 * HEAD_DIM), 5, jnp.bfloat16)  # wider than v: its first columns are the gate
    weight = (1 + 0.1 * rand((1, HEAD_DIM), 6)).astype(jnp.bfloat16)
    tiles = DeltaTiles(heads=1, block=128)
    out = run_checking_races(gated_delta_rule_gated, q, k, v, g, beta, z, weight, tiles)
    want = _gated_rmsnorm(reference_recurrent(q, k, v, g, beta), z[..., : 2 * HEAD_DIM], weight)
    _assert_close(out, want, 3e-2)
    gated_delta_rule_gated.verify(q, k, v, g, beta, z, weight, tiles)


@pytest.mark.parametrize(
    ("batch", "heads", "seq", "tiles"),
    [
        (1, 16, 4096, DeltaTiles()),
        (2, 32, 4096, DeltaTiles(heads=4, block=512)),
        (1, 32, 8192, DeltaTiles(heads=1, block=128)),
    ],
)
def test_kernels_lower_to_mosaic_for_v5e_at_scale(batch, heads, seq, tiles):
    b16, d = jnp.bfloat16, heads * HEAD_DIM
    qkv = (spec(batch, seq, d, dtype=b16),) * 3
    rows = spec(batch, heads, seq)
    cases = [
        (gated_delta_rule, (*qkv, rows, rows)),
        (kimi_delta_attention, (*qkv, spec(batch, seq, d), rows)),
        (gated_delta_rule_gated, (*qkv, rows, rows, spec(batch, seq, 3 * d, dtype=b16), spec(1, HEAD_DIM, dtype=b16))),
    ]
    for kernel, specs in cases:
        assert "tpu_custom_call" in lower_for_v5e(kernel.lower(*specs, tiles), *specs)


# --------------------------------------------------------------------------- Qwen3-Next's Gated DeltaNet layer


def _layer_reference(x, params, config):
    """The layer in float64 NumPy (see examples/linear_attention_layer.py)."""
    from linear_attention_layer import CONV

    x = f64(x)
    B, S, _ = x.shape
    vd, kd, hv, r = config.value_dim, config.key_dim, config.v_heads, config.v_heads // config.k_heads
    h = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * f64(params["norm"])
    proj, gates = h @ f64(params["w_qkvz"]), h @ f64(params["w_ba"])
    z, qkv = proj[..., :vd], np.pad(proj[..., vd:], ((0, 0), (CONV - 1, 0), (0, 0)))
    conv = sum(f64(params["conv"])[j] * qkv[:, j : j + S] for j in range(CONV))
    conv = conv / (1 + np.exp(-conv))
    q, k, v = conv[..., :kd], conv[..., kd : 2 * kd], conv[..., 2 * kd :]
    q, k = (np.repeat(t.reshape(B, S, -1, HEAD_DIM), r, axis=2).reshape(B, S, vd) for t in (q, k))
    beta = 1 / (1 + np.exp(-gates[..., :hv]))
    g = -np.exp(f64(params["A_log"])) * np.log1p(np.exp(gates[..., hv : 2 * hv] + f64(params["dt_bias"])))
    o = reference_recurrent(q, k, v, g.transpose(0, 2, 1), beta.transpose(0, 2, 1))
    return x + _gated_rmsnorm(o, z, params["out_norm"]) @ f64(params["w_out"])


@pytest.mark.parametrize(("k_heads", "v_heads"), [(1, 2), (2, 2)])
def test_gated_deltanet_layer_matches_numpy(k_heads, v_heads):
    from linear_attention_layer import Config, baseline_layer, gated_deltanet_layer, init_params

    config = Config(d_model=256, k_heads=k_heads, v_heads=v_heads)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (2, 512, config.d_model), jnp.bfloat16)
    want = _layer_reference(x, params, config)
    out = jax.jit(gated_deltanet_layer, static_argnames="config")(x, params, config)
    assert out.shape == x.shape and out.dtype == x.dtype
    _assert_close(out, want, 3e-2)
    _assert_close(baseline_layer(x, params, config), want, 3e-2)


def test_gated_deltanet_layer_kernels():
    from linear_attention_layer import Config, init_params, prepare, residual_matmul
    from matmul import MatmulTiles
    from rmsnorm_matmul import rmsnorm_matmul

    config = Config(d_model=256, k_heads=1, v_heads=2)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (512, config.d_model), jnp.bfloat16)
    proj = rmsnorm_matmul(x, params["norm"], params["w_qkvz"]).reshape(1, 512, -1)
    gates = rmsnorm_matmul(x, params["norm"], params["w_ba"]).reshape(1, 512, -1)
    args = (*prepare(proj, gates, params, config), proj, params["out_norm"], DeltaTiles(heads=2, block=128))
    o = run_checking_races(gated_delta_rule_gated, *args)
    gated_delta_rule_gated.verify(*args)
    tiles = MatmulTiles(tm=256, tn=128, tk=128)
    out = run_checking_races(residual_matmul, x, o.reshape(512, -1), params["w_out"], tiles)
    want = f64(x) + f64(o.reshape(512, -1)) @ f64(params["w_out"])
    _assert_close(out, want)


def test_gated_deltanet_layer_lowers_to_mosaic_for_v5e():
    from linear_attention_layer import Config, residual_matmul

    b16 = jnp.bfloat16
    config = Config(d_model=2048, k_heads=16, v_heads=32)  # Qwen3-Next-80B-A3B's
    B, S, vd = 1, 4096, config.value_dim
    specs = (spec(B, S, vd, dtype=b16),) * 3 + (spec(B, 32, S),) * 2
    specs += (spec(B, S, 2 * vd + 2 * config.key_dim, dtype=b16), spec(1, HEAD_DIM, dtype=b16))
    assert "tpu_custom_call" in lower_for_v5e(gated_delta_rule_gated.lower(*specs), *specs)
    specs = (spec(B * S, 2048, dtype=b16), spec(B * S, vd, dtype=b16), spec(vd, 2048, dtype=b16))
    assert "tpu_custom_call" in lower_for_v5e(residual_matmul.lower(*specs), *specs)


# --------------------------------------------------------------------------- on a TPU v5e


@pytest.mark.tpu
@pytest.mark.parametrize("per_channel", [False, True], ids=["gdn", "kda"])
def test_tuned_kernels_match_the_chunked_baseline_on_tpu(per_channel):
    inputs = make_inputs(2, 16, 2048, per_channel)
    kernel = KERNELS[per_channel]
    out = jax.jit(kernel)(*inputs)  # tunes on the first call
    assert kernel.tuning(*inputs).trials
    _assert_close(out, f64(baseline_chunked(*inputs)))


@pytest.mark.tpu
def test_gated_deltanet_layer_on_tpu():
    from linear_attention_layer import Config, baseline_layer, gated_deltanet_layer, init_params

    config = Config(d_model=1024, k_heads=4, v_heads=8)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (1, 2048, config.d_model), jnp.bfloat16)
    out = jax.jit(gated_deltanet_layer, static_argnames="config")(x, params, config)
    _assert_close(out, f64(baseline_layer(x, params, config)), 3e-2)
