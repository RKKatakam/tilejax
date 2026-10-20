"""The example kernels, against independent references: NumPy, and JAX's own splash attention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from causal_attention import AttentionBlocks, causal_attention, splash_attention
from causal_matmul import causal_matmul
from conftest import ON_TPU
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from matmul import MatmulTiles, matmul


def tiles(tm, tn, tk):
    return MatmulTiles(tm=tm, tn=tn, tk=tk)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (jnp.float32, (128, 128, 128)),
        (jnp.bfloat16, (128, 256, 128)),
        (jnp.float32, (256, 128, 512)),  # one reduction step: no accumulator
        (jnp.bfloat16, (128, 128, 128)),
    ],
)
def test_matmul(dtype, shape):
    x, y = rand((256, 384), 0, dtype), rand((384, 256), 1, dtype)
    out = run_checking_races(matmul, x, y, tiles(*shape))
    assert out.dtype == dtype
    np.testing.assert_allclose(f64(out), f64(x) @ f64(y), rtol=2e-2 if dtype == jnp.bfloat16 else 1e-4, atol=0.2)
    assert matmul.verify(x, y, tiles(*shape)) < 1e-3


def test_three_reduction_steps_a_ring_that_isnt_a_power_of_two():
    x, y = rand((256, 384), 0), rand((384, 256), 1)
    lowered = matmul.lower(x, y, tiles(128, 128, 128))
    assert lowered.schedule.steps_per_tile == 3 and "jax.lax.rem(s, 3)" in lowered.source
    out = run_checking_races(matmul, x, y, tiles(128, 128, 128))
    np.testing.assert_allclose(f64(out), f64(x) @ f64(y), rtol=1e-4, atol=1e-4)


def test_plain_jax_with_the_default_tiles():
    # Without a tunable, off a TPU, the kernel uses the defaults (clamped to the arrays by its prologue).
    x, y = rand((256, 256)), rand((256, 256), 1)
    np.testing.assert_allclose(f64(matmul(x, y)), f64(x) @ f64(y), rtol=1e-4, atol=1e-3)
    if not ON_TPU:
        assert matmul.tuning(x, y).config.tunable == MatmulTiles()
    # The function itself is plain JAX: the tunable is a pytree with no leaves, so jit passes it through.
    assert jax.tree_util.tree_leaves(MatmulTiles()) == []
    np.testing.assert_allclose(f64(jax.jit(matmul)(x, y, tiles(128, 128, 128))), f64(x) @ f64(y), rtol=1e-4, atol=1e-3)
    with jax.default_matmul_precision("highest"):  # plain JAX's f32 `@` is one bf16 pass on a TPU
        np.testing.assert_allclose(f64(jax.jit(matmul.fn)(x, y)), f64(x) @ f64(y), rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_causal_matmul(dtype):
    x, y = rand((512, 256), 0, dtype), rand((256, 512), 1, dtype)
    out = run_checking_races(causal_matmul, x, y, tiles(128, 128, 128))
    np.testing.assert_allclose(f64(out), np.tril(f64(x) @ f64(y)), rtol=2e-2, atol=0.2)
    causal_matmul.verify(x, y, tiles(128, 128, 128))
    lowered = causal_matmul.lower(x, y, tiles(128, 128, 128))
    assert (lowered.tiles, lowered.total_tiles, lowered.steps) == (10, 16, 20)


def _attention_inputs(b, h, s, d, dtype=jnp.bfloat16):
    q, k, v = (rand((b, h, s, d), seed, dtype) for seed in range(3))
    return (q * d**-0.5).astype(dtype), k, v


def _attention(q, k, v):
    """Causal softmax attention in float64 NumPy (q already scaled)."""
    q, k, v = f64(q), f64(k), f64(v)
    s = np.einsum("bhqd,bhkd->bhqk", q, k)
    s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    return np.einsum("bhqk,bhkd->bhqd", p / p.sum(-1, keepdims=True), v)


@pytest.mark.parametrize(("block_q", "block_kv"), [(128, 128), (256, 128), (128, 256)])
def test_causal_attention_matches_splash(block_q, block_kv):
    q, k, v = _attention_inputs(2, 2, 512, 128)
    blocks = AttentionBlocks(block_q=block_q, block_kv=block_kv)
    out = run_checking_races(causal_attention, q, k, v, blocks)
    splash = splash_attention(2, 512, block_q, block_kv, interpret=not ON_TPU)(q, k, v)
    np.testing.assert_allclose(f64(out), f64(splash), atol=2e-2)
    np.testing.assert_allclose(f64(out), _attention(q, k, v), atol=2e-2)
    causal_attention.verify(q, k, v, blocks)


def test_causal_attention_skips_future_blocks():
    lowered = causal_attention.lower(*_attention_inputs(2, 4, 512, 128), AttentionBlocks(block_q=128, block_kv=128))
    assert lowered.steps == 2 * 4 * (1 + 2 + 3 + 4)
    # The masks don't read the batch and head loops: the step tables hold one batch-head's pattern.
    assert lowered.schedule.repeats == 8 and len(lowered.schedule.tables["step_tile"]) == 10


def test_causal_attention_f32():
    q, k, v = _attention_inputs(1, 1, 256, 128, jnp.float32)
    out = run_checking_races(causal_attention, q, k, v, AttentionBlocks(block_q=128, block_kv=128))
    np.testing.assert_allclose(f64(out), _attention(q, k, v), atol=1e-4)


@pytest.mark.parametrize("trace", [False, True])
def test_examples_lower_to_mosaic_for_v5e(trace):
    b16 = jnp.bfloat16
    big = (spec(1024, 1024, dtype=b16), spec(1024, 1024, dtype=b16))
    cases = [
        (matmul, big, tiles(256, 256, 256)),
        (matmul, big, tiles(256, 256, 1024)),
        (causal_matmul, (spec(1024, 1024), spec(1024, 1024)), tiles(256, 256, 256)),
        (causal_attention, (spec(2, 4, 1024, 128, dtype=b16),) * 3, AttentionBlocks(block_q=256, block_kv=256)),
    ]
    for kernel, specs, tunable in cases:
        config = kernel.lower(*specs, tunable).config
        lowered = kernel.lowered(kernel.signature(*specs), config, trace=trace)
        assert ("named_scope" in lowered.source) == trace
        assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


def _rmsnorm_matmul(x, gamma, w, eps=1e-6):
    """rmsnorm(x) @ w in float64 NumPy."""
    x, gamma, w = f64(x), f64(gamma), f64(w)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma @ w


def _norm_inputs(m, d, n, dtype):
    x = rand((m, d), 0, dtype)
    gamma = (1 + 0.1 * rand((1, d), 1)).astype(dtype)
    w = (rand((d, n), 2) / np.sqrt(d)).astype(dtype)
    return x, gamma, w


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (jnp.float32, (128, 128, 128)),
        (jnp.bfloat16, (256, 128, 256)),
        (jnp.bfloat16, (128, 256, 512)),  # the whole model dim in one step: no carried values
    ],
)
def test_rmsnorm_matmul(dtype, shape):
    from rmsnorm_matmul import NormMatmulTiles, rmsnorm_matmul, rmsnorm_then_matmul

    x, gamma, w = _norm_inputs(256, 512, 512, dtype)
    tiles = NormMatmulTiles(tm=shape[0], tn=shape[1], tk=shape[2])
    out = run_checking_races(rmsnorm_matmul, x, gamma, w, tiles)
    want = _rmsnorm_matmul(x, gamma, w)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(f64(out), want, rtol=tol, atol=tol * np.abs(want).max())
    with jax.default_matmul_precision("highest"):  # plain JAX's f32 `@` is one bf16 pass on a TPU
        plain = rmsnorm_then_matmul(x, gamma, w)
    np.testing.assert_allclose(f64(out), f64(plain), rtol=tol, atol=tol * np.abs(want).max())
    rmsnorm_matmul.verify(x, gamma, w, tiles)
    lowered = rmsnorm_matmul.lower(x, gamma, w, tiles)
    assert set(lowered.plan.carried) == ({"acc", "sum_sq"} if shape[2] < 512 else set())


def test_rmsnorm_matmul_lowers_to_mosaic_for_v5e():
    from rmsnorm_matmul import NormMatmulTiles, rmsnorm_matmul

    b16 = jnp.bfloat16
    specs = (spec(2048, 1024, dtype=b16), spec(1, 1024, dtype=b16), spec(1024, 1024, dtype=b16))
    for tiles in (NormMatmulTiles(), NormMatmulTiles(tm=256, tn=1024, tk=1024)):
        assert "tpu_custom_call" in lower_for_v5e(rmsnorm_matmul.lower(*specs, tiles), *specs)


def _attention_block(q, k, v, gamma, w_up, w_down, eps=1e-6):
    """Causal attention, then RMSNorm and the feed-forward layer (tanh GELU), in float64 NumPy."""
    a = _attention(q, k, v)  # (B, H, S, Dh)
    b, h, s, dh = a.shape
    a = a.transpose(0, 2, 1, 3).reshape(b, s, h * dh)
    normed = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps) * f64(gamma)
    up = normed @ f64(w_up)
    hidden = 0.5 * up * (1 + np.tanh(np.sqrt(2 / np.pi) * (up + 0.044715 * up**3)))
    return hidden @ f64(w_down)


@pytest.mark.parametrize(("block_q", "block_kv"), [(128, 128), (256, 128)])
def test_attention_block(block_q, block_kv):
    from attention_block import BlockTiles, attention_block, make_inputs

    inputs = make_inputs(1, 2, 512, 128, 512)
    tiles = BlockTiles(block_q=block_q, block_kv=block_kv)
    out = run_checking_races(attention_block, *inputs, tiles)
    want = _attention_block(*inputs)
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    attention_block.verify(*inputs, tiles)
    lowered = attention_block.lower(*inputs, tiles)
    assert {op.name: op.kind.value for op in lowered.plan.operands} == {
        "q": "tile",
        "k": "stream",
        "v": "stream",
        "gamma": "once",
        "w_up": "once",
        "w_down": "once",
    }
    assert lowered.steps == sum(i // block_kv + 1 for i in range(block_q - 1, 512, block_q))


def test_attention_block_lowers_to_mosaic_for_v5e():
    from attention_block import BlockTiles, attention_block

    b16 = jnp.bfloat16
    specs = (spec(4, 8, 2048, 128, dtype=b16),) * 3
    specs += (spec(1, 1024, dtype=b16), spec(1024, 4096, dtype=b16), spec(4096, 1024, dtype=b16))
    lowered = attention_block.lower(*specs, BlockTiles(block_q=256, block_kv=512))
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(("heads", "head_dim"), [(2, 128), (4, 64)])
def test_attention_block_streaming_its_weights(heads, head_dim):
    from attention_block import BlockTiles, attention_block, make_inputs

    inputs = make_inputs(1, heads, 256, head_dim, 1024)
    tiles = BlockTiles(block_q=128, block_kv=128, block_f=256)  # the feed-forward layer in four steps
    out = run_checking_races(attention_block, *inputs, tiles)
    want = _attention_block(*inputs)
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    attention_block.verify(*inputs, tiles)
    lowered = attention_block.lower(*inputs, tiles)
    assert [phase.range.var for phase in lowered.plan.phases] == ["j", "f"]
    assert {op.name: (op.kind.value, op.phase) for op in lowered.plan.operands if op.name.startswith("w_")} == {
        "w_up": ("stream", 1),
        "w_down": ("stream", 1),
    }


@pytest.mark.parametrize(("heads", "head_dim", "ffn"), [(16, 128, 8192), (16, 64, 4096), (32, 128, 16384)])
def test_attention_block_lowers_at_scale(heads, head_dim, ffn):
    from attention_block import attention_block

    b16, d = jnp.bfloat16, heads * head_dim
    specs = (spec(2, heads, 2048, head_dim, dtype=b16),) * 3
    specs += (spec(1, d, dtype=b16), spec(d, ffn, dtype=b16), spec(ffn, d, dtype=b16))
    assert "tpu_custom_call" in lower_for_v5e(attention_block.lower(*specs), *specs)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _rms(x, gamma, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * f64(gamma)


def _transformer_layer(x, qkv, wo, gamma, w_up, w_down):
    """A layer after its QKV projection, in float64 NumPy: attention, output projection, residual, RMSNorm,
    feed-forward layer, residual."""
    b, s, _ = x.shape
    h, dh = wo.shape[0], wo.shape[1]
    q, k, v = (f64(t).reshape(b, s, h, dh).transpose(0, 2, 1, 3) for t in np.split(f64(qkv), 3, axis=-1))
    heads = _attention(q, k, v)  # (B, H, S, Dh); q is pre-scaled
    resid = f64(x) + np.einsum("bhsd,hde->bse", heads, f64(wo))
    return resid + _gelu(_rms(resid, gamma) @ f64(w_up)) @ f64(w_down)


def _transformer(tokens, params):
    x = f64(params["embed"])[np.asarray(tokens)]
    for layer in params["layers"]:
        qkv = _rms(x, layer["attn_norm"]) @ f64(layer["w_qkv"])
        x = _transformer_layer(x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"])


def test_transformer_matches_numpy():
    from transformer import Config, init_params, transformer

    config = Config(layers=2, d_model=256, heads=2, ffn=512, vocab=512)
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (2, 256), 0, config.vocab)
    logits = jax.jit(transformer)(tokens, params)
    want = _transformer(tokens, params)
    assert logits.shape == (2, 256, 512)
    np.testing.assert_allclose(f64(logits), want, atol=5e-2 * np.abs(want).max())


@pytest.mark.parametrize(("heads", "head_dim"), [(2, 128), (4, 64)])
def test_transformer_layer_in_three_phases(heads, head_dim):
    from transformer import LayerTiles, transformer_layer

    d, s = heads * head_dim, 256
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(keys[0], (1, s, d), jnp.bfloat16)
    qkv = jax.random.normal(keys[1], (1, s, 3 * d), jnp.bfloat16)
    qkv = qkv.at[:, :, :d].multiply(head_dim**-0.5)
    wo = (jax.random.normal(keys[2], (heads, head_dim, d)) / d**0.5).astype(jnp.bfloat16)
    gamma = jnp.ones((1, d), jnp.bfloat16)
    w_up = (jax.random.normal(keys[3], (d, 1024)) / d**0.5).astype(jnp.bfloat16)
    w_down = (jax.random.normal(keys[4], (1024, d)) / 32).astype(jnp.bfloat16)
    args = (x, qkv, wo, gamma, w_up, w_down, LayerTiles(block_q=128, block_kv=128, block_f=256))
    out = run_checking_races(transformer_layer, *args)
    want = _transformer_layer(*args[:6])
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    transformer_layer.verify(*args)
    lowered = transformer_layer.lower(*args)
    assert [phase.range.var for phase in lowered.plan.phases] == ["j", "hh", "f"]
    assert "heads_ref[hh]" in lowered.source  # a head of the carried attention output, read from VMEM directly


@pytest.mark.parametrize(
    ("d", "heads", "ffn"), [(1024, 8, 4096), (2048, 16, 8192), (1024, 16, 4096), (4096, 32, 16384)]
)
def test_transformer_layer_lowers_at_scale(d, heads, ffn):
    from transformer import transformer_layer

    b16, dh = jnp.bfloat16, d // heads
    specs = (spec(2, 2048, d, dtype=b16), spec(2, 2048, 3 * d, dtype=b16), spec(heads, dh, d, dtype=b16))
    specs += (spec(1, d, dtype=b16), spec(d, ffn, dtype=b16), spec(ffn, d, dtype=b16))
    assert "tpu_custom_call" in lower_for_v5e(transformer_layer.lower(*specs), *specs)


def test_transformer_split_layer_matches_numpy():
    from transformer import Config, FfnTiles, LayerTiles, attention_layer, ffn_layer, fuse_ffn, init_params, split_layer

    config = Config(layers=1, d_model=256, heads=2, ffn=1024, vocab=256)
    layer = init_params(config)["layers"][0]
    keys = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(keys[0], (1, 256, 256), jnp.bfloat16)
    qkv = jax.random.normal(keys[1], (1, 256, 768), jnp.bfloat16).at[:, :, :256].multiply(128**-0.5)
    want = _transformer_layer(x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])
    out = split_layer(x, qkv, layer)
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    tiles = LayerTiles(block_q=128, block_kv=128)
    run_checking_races(attention_layer, x, qkv, layer["wo"], tiles)
    attention_layer.verify(x, qkv, layer["wo"], tiles)
    rows = x.reshape(256, 256)
    ffn_tiles = FfnTiles(rows=128, block_f=256)
    run_checking_races(ffn_layer, rows, layer["ffn_norm"], layer["w_up"], layer["w_down"], ffn_tiles)
    ffn_layer.verify(rows, layer["ffn_norm"], layer["w_up"], layer["w_down"], ffn_tiles)
    if not ON_TPU:
        assert fuse_ffn(x, qkv, layer)  # nothing measured off a TPU: fused
