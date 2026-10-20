"""On a TPU v5e: the examples run natively and right, and profiles read their trace regions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from causal_attention import make_causal_attention, make_splash_attention
from causal_matmul import make_causal_matmul
from helpers import f64, make_kernel, rand
from matmul import make_matmul

import autokernel
from autokernel import Pipeline, profiling

pytestmark = pytest.mark.tpu


def test_regions_are_enabled():
    assert profiling.regions_enabled()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_matmuls_natively(dtype):
    x, y = rand((1024, 512), 0, dtype), rand((512, 1024), 1, dtype)
    want = f64(x) @ f64(y)
    np.testing.assert_allclose(f64(make_matmul(256, 256, 256)(x, y)), want, rtol=2e-2, atol=0.5)
    np.testing.assert_allclose(f64(make_causal_matmul(256, 256, 256)(x, y)), np.tril(want), rtol=2e-2, atol=0.5)
    make_matmul(256, 256, 512).verify(x, y)
    make_causal_matmul(256, 256, 256).verify(x, y)


def test_attention_natively_matches_splash():
    q, k, v = (rand((2, 4, 1024, 128), seed, jnp.bfloat16) for seed in range(3))
    q = (q * 128**-0.5).astype(q.dtype)
    attention = make_causal_attention(256, 256)
    splash = make_splash_attention(4, 1024, 256, 256)
    np.testing.assert_allclose(f64(attention(q, k, v)), f64(splash(q, k, v)), atol=2e-2)
    attention.verify(q, k, v)


def test_profile_reads_regions_and_scopes():
    kernel = make_kernel(
        """
        def f(x, y):
            out = jnp.zeros((1024, 1024))
            for i in range(0, 1024, 256):
                for j in range(0, 1024, 256):
                    acc = jnp.zeros((256, 256))
                    for k in range(0, 1024, 256):
                        with jax.named_scope("mxu"):
                            acc += x[i : i + 256, k : k + 256] @ y[k : k + 256, j : j + 256]
                    with jax.named_scope("epilogue"):
                        out = out.at[i : i + 256, j : j + 256].set(jnp.maximum(acc, 0))
            return out
        """
    )
    x, y = rand((1024, 1024), 0, jnp.bfloat16), rand((1024, 1024), 1, jnp.bfloat16)
    p = autokernel.profile(kernel, x, y, runs=3)
    assert p.calls == 3 and p.regions and p.steps == kernel.lower(x, y).steps
    assert 0 < p.compute_s < p.device_s and p.scopes["mxu"] > 0 and p.scopes["epilogue"] > 0
    assert p.bottleneck in ("dma", "overhead", "compute")


def test_optimize_finds_more_buffering_for_small_tiles():
    kernel = make_matmul(128, 128, 128)
    x, y = rand((4096, 4096), 0, jnp.bfloat16), rand((4096, 4096), 1, jnp.bfloat16)
    result = autokernel.optimize(kernel, x, y, runs=3)
    assert result.tried[0].pipeline == Pipeline(3, 2)
    assert kernel.lower(x, y).pipeline == result.pipeline
    assert result.profile.device_s <= result.tried[0].device_s
    jax.block_until_ready(kernel(x, y))
