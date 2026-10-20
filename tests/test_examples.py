"""The example kernels, against independent references: NumPy, and JAX's own splash attention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from causal_attention import make_causal_attention, make_splash_attention
from causal_matmul import make_causal_matmul
from conftest import ON_TPU
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from matmul import make_matmul


@pytest.mark.parametrize(
    ("dtype", "tiles"),
    [
        (jnp.float32, (128, 128, 128)),
        (jnp.bfloat16, (128, 256, 128)),
        (jnp.float32, (256, 128, 384)),  # one reduction step: no accumulator
        (jnp.bfloat16, (128, 128, 384)),  # three steps per tile: a ring that isn't a power of two
    ],
)
def test_matmul(dtype, tiles):
    x, y = rand((256, 384), 0, dtype), rand((384, 256), 1, dtype)
    kernel = make_matmul(*tiles)
    out = run_checking_races(kernel, x, y)
    assert out.dtype == dtype
    np.testing.assert_allclose(f64(out), f64(x) @ f64(y), rtol=2e-2 if dtype == jnp.bfloat16 else 1e-4, atol=0.2)
    assert kernel.verify(x, y) < 1e-3


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_causal_matmul(dtype):
    x, y = rand((512, 256), 0, dtype), rand((256, 512), 1, dtype)
    kernel = make_causal_matmul(128, 128, 128)
    out = run_checking_races(kernel, x, y)
    np.testing.assert_allclose(f64(out), np.tril(f64(x) @ f64(y)), rtol=2e-2, atol=0.2)
    kernel.verify(x, y)
    lowered = kernel.lower(x, y)
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
    kernel = make_causal_attention(block_q, block_kv)
    out = run_checking_races(kernel, q, k, v)
    splash = make_splash_attention(2, 512, block_q, block_kv, interpret=not ON_TPU)(q, k, v)
    np.testing.assert_allclose(f64(out), f64(splash), atol=2e-2)
    np.testing.assert_allclose(f64(out), _attention(q, k, v), atol=2e-2)
    kernel.verify(q, k, v)


def test_causal_attention_skips_future_blocks():
    lowered = make_causal_attention(128, 128).lower(*_attention_inputs(2, 4, 512, 128))
    assert lowered.steps == 2 * 4 * (1 + 2 + 3 + 4)
    # The masks don't read the batch and head loops: the step tables hold one batch-head's pattern.
    assert lowered.schedule.repeats == 8 and len(lowered.schedule.tables["step_tile"]) == 10


def test_causal_attention_f32():
    q, k, v = _attention_inputs(1, 1, 256, 128, jnp.float32)
    out = run_checking_races(make_causal_attention(128, 128), q, k, v)
    np.testing.assert_allclose(f64(out), _attention(q, k, v), atol=1e-4)


@pytest.mark.parametrize("trace", [False, True])
def test_examples_lower_to_mosaic_for_v5e(trace):
    b16 = jnp.bfloat16
    cases = [
        (make_matmul(256, 256, 256), (spec(1024, 1024, dtype=b16), spec(1024, 1024, dtype=b16))),
        (make_matmul(256, 256, 1024), (spec(1024, 1024, dtype=b16), spec(1024, 1024, dtype=b16))),
        (make_causal_matmul(256, 256, 256), (spec(1024, 1024), spec(1024, 1024))),
        (make_causal_attention(256, 256), (spec(2, 4, 1024, 128, dtype=b16),) * 3),
    ]
    for kernel, specs in cases:
        signature = kernel.signature(*specs)
        lowered = kernel.lowered(signature, kernel.pipeline(signature), trace=trace)
        assert ("named_scope" in lowered.source) == trace
        assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


def test_examples_run_under_jit():
    x, y = rand((256, 256)), rand((256, 256), 1)
    out = jax.jit(make_matmul(128, 128, 128))(x, y)
    np.testing.assert_allclose(f64(out), f64(x) @ f64(y), rtol=1e-4, atol=1e-3)
