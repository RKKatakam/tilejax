"""On a TPU v5e: the examples run natively and right, profiles read their trace regions, and tuning
finds schedules faster than the defaults."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from causal_attention import AttentionBlocks, causal_attention, splash_attention
from causal_matmul import causal_matmul
from helpers import f64, make_kernel, rand
from matmul import MatmulTiles, matmul

import autokernel
from autokernel import profiling

pytestmark = pytest.mark.tpu

TILES = MatmulTiles(tm=256, tn=256, tk=256)


def test_regions_are_enabled():
    assert profiling.regions_enabled()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_matmuls_natively(dtype):
    x, y = rand((1024, 512), 0, dtype), rand((512, 1024), 1, dtype)
    want = f64(x) @ f64(y)
    np.testing.assert_allclose(f64(matmul(x, y, TILES)), want, rtol=2e-2, atol=0.5)
    np.testing.assert_allclose(f64(causal_matmul(x, y, TILES)), np.tril(want), rtol=2e-2, atol=0.5)
    matmul.verify(x, y, MatmulTiles(tm=256, tn=256, tk=512))
    causal_matmul.verify(x, y, TILES)


def test_attention_natively_matches_splash():
    q, k, v = (rand((2, 4, 1024, 128), seed, jnp.bfloat16) for seed in range(3))
    q = (q * 128**-0.5).astype(q.dtype)
    blocks = AttentionBlocks(block_q=256, block_kv=256)
    splash = splash_attention(4, 1024, 256, 256)
    np.testing.assert_allclose(f64(causal_attention(q, k, v, blocks)), f64(splash(q, k, v)), atol=2e-2)
    causal_attention.verify(q, k, v, blocks)


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


def test_tuning_beats_small_default_tiles():
    @autokernel.tunable
    class Small:
        tm: int = autokernel.choice(128, 256, 512, 1024)
        tn: int = autokernel.choice(128, 256, 512, 1024)
        tk: int = autokernel.choice(128, 256, 512, 1024)

    @autokernel.kernel
    def small_matmul(x, y, tiles=Small()):
        M, K = x.shape
        N = y.shape[1]
        dtype = x.dtype
        out = jnp.zeros((M, N), dtype)
        for i in range(0, M, tiles.tm):
            for j in range(0, N, tiles.tn):
                acc = jnp.zeros((tiles.tm, tiles.tn), jnp.float32)
                for k in range(0, K, tiles.tk):
                    acc += x[i : i + tiles.tm, k : k + tiles.tk] @ y[k : k + tiles.tk, j : j + tiles.tn]
                out = out.at[i : i + tiles.tm, j : j + tiles.tn].set(acc.astype(dtype))
        return out

    x, y = rand((4096, 4096), 0, jnp.bfloat16), rand((4096, 4096), 1, jnp.bfloat16)
    tuning = small_matmul.tuning(x, y)
    print(tuning)
    first = tuning.trials[0]
    assert first.config.tunable == Small() and first.profile.bottleneck in ("dma", "overhead")
    assert tuning.best.seconds < 0.7 * first.seconds  # 128^3 tiles are per-step-overhead bound on a v5e
    assert small_matmul.lower(x, y).config == tuning.config
    small_matmul.verify(x, y)
    jax.block_until_ready(jax.jit(small_matmul)(x, y))


def test_tuning_under_jit():
    @autokernel.tunable
    class Rows:
        tm: int = autokernel.choice(128, 256, 512)

    @autokernel.kernel
    def scale(x, tiles=Rows()):
        M, N = x.shape
        out = jnp.zeros((M, N), x.dtype)
        for i in range(0, M, tiles.tm):
            out = out.at[i : i + tiles.tm, :].set(x[i : i + tiles.tm, :] * 2)
        return out

    x = rand((2048, 1024))
    np.testing.assert_allclose(f64(jax.jit(scale)(x)), 2 * f64(x))
    assert len(scale.tuning(x).trials) >= 2
