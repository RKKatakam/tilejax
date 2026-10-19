"""Splash attention and the features it needs: integer (batch/head) indexing, `x @ y.T`, reduction masks."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError

# Masked scores use a large finite value, not -inf: a row masked out of a whole block then gives finite
# garbage that the next block's rescale (exp(m - m_new) == 0) wipes out, instead of exp(-inf + inf) = nan.
MASK_VALUE = -1e30
LANES = 128  # row statistics are kept lane-wide: (TQ, 128), each row's value repeated across lanes


def make_attention(B, H, S, D, TQ, TK, window=None, **options):
    """Flash attention with a causal (and optionally sliding-window) mask on key/value blocks."""
    scale = 1.0 / math.sqrt(D)
    W = window if window is not None else S

    def attention(q, k, v):
        dtype = q.dtype
        out = jnp.zeros((B, H, S, D), dtype)
        for b in range(0, B, 1):
            for h in range(0, H, 1):
                for i in range(0, S, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)
                    row_sum = jnp.zeros((TQ, LANES))
                    acc = jnp.zeros((TQ, D))
                    for j in range(0, S, TK):
                        if j > i + TQ - 1 or j + TK - 1 < i - W + 1:  # all future, or all out of the window
                            continue
                        s = (q[b, h, i : i + TQ, :] @ k[b, h, j : j + TK, :].T) * scale
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                        s = jnp.where((cols <= rows) & (cols > rows - W), s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + TK, :]
                        row_max = m_new
                    out = out.at[b, h, i : i + TQ, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
        return out

    return autokernel.tpu(attention, **options)


def reference(q, k, v, window=None):
    """Plain JAX attention in full f32 precision (TPU einsums otherwise default to a single bf16 pass)."""
    S, D = q.shape[-2:]
    W = window if window is not None else S
    f32 = jnp.float32
    s = jnp.einsum("bhqd,bhkd->bhqk", q.astype(f32), k.astype(f32), precision="highest") / math.sqrt(D)
    rows, cols = np.indices((S, S))
    allowed = (cols <= rows) & (cols > rows - W)
    p = jax.nn.softmax(jnp.where(allowed, s, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(f32), precision="highest")


def qkv(B, H, S, D, dtype):
    return (rand((B, H, S, D), seed, dtype) for seed in range(3))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("TQ, TK", [(128, 128), (256, 128), (128, 256)])
def test_causal_splash_attention(dtype, TQ, TK):
    B, H, S, D = 2, 3, 512, 128
    kern = make_attention(B, H, S, D, TQ, TK)
    q, k, v = qkv(B, H, S, D, dtype)
    tol = 1e-4 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(kern(q, k, v), np.float32), reference(q, k, v), atol=tol, rtol=tol)
    kern.verify(q, k, v)


def test_causal_mask_skips_future_kv_blocks():
    B, H, S, D, T = 2, 3, 512, 128, 128
    lw = make_attention(B, H, S, D, T, T).lower(*qkv(B, H, S, D, jnp.bfloat16))
    n = S // T
    assert lw.steps == B * H * n * (n + 1) // 2  # lower triangle of (query block, key block) pairs
    assert "_mxu_dot_t(q_ref[...], k_ref[...])" in lw.source  # q @ k.T without materialising k.T


def test_sliding_window_skips_blocks_on_both_sides():
    B, H, S, D, T, W = 1, 2, 1024, 128, 128, 256
    kern = make_attention(B, H, S, D, T, T, window=W)
    q, k, v = qkv(B, H, S, D, jnp.float32)
    np.testing.assert_allclose(kern(q, k, v), reference(q, k, v, window=W), atol=1e-4, rtol=1e-4)
    # Query block i needs key blocks i-2 .. i (W / T + 1 at most) instead of 0 .. i.
    assert kern.lower(q, k, v).steps == H * sum(min(i, W // T) + 1 for i in range(S // T))


def test_splash_attention_has_no_races():
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
    from jax.experimental.pallas import tpu as pltpu

    B, H, S, D, T = 1, 2, 512, 128, 128
    kern = make_attention(B, H, S, D, T, T)
    q, k, v = qkv(B, H, S, D, jnp.float32)
    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    out = kern.lower(q, k, v).entry(q, k, v, interpret=params).block_until_ready()
    assert interpreter.races is not None and not interpreter.races.races_found
    np.testing.assert_allclose(out, reference(q, k, v), atol=1e-4, rtol=1e-4)


def test_reduction_masks_need_the_manual_pipeline():
    kern = make_attention(1, 1, 256, 128, 128, 128, pipeline="grid")
    with pytest.raises(AutoKernelError, match="need pipeline='manual'"):
        kern.lower(*qkv(1, 1, 256, 128, jnp.float32))


def test_reduction_mask_that_empties_a_tile_is_rejected():
    @autokernel.tpu
    def bad(x, y):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    if i > 0:
                        continue
                    acc += x[i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
                out = out.at[i : i + 128, j : j + 128].set(acc)
        return out

    with pytest.raises(AutoKernelError, match="skips every step of the tile at i=128"):
        bad.lower(rand((256, 256)), rand((256, 256), 1))


def test_integer_indexing_by_batch_loop(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def batched_matmul(x, y):
        out = jnp.zeros((3, 256, 256))
        for b in range(0, 3, 1):
            for i in range(0, 256, 128):
                acc = jnp.zeros((128, 256))
                for k in range(0, 256, 128):
                    acc += x[b, i : i + 128, k : k + 128] @ y[b, k : k + 128, :]
                out = out.at[b, i : i + 128, :].set(acc)
        return out

    x, y = rand((3, 256, 256)), rand((3, 256, 256), 1)
    batched_matmul.verify(x, y)
    np.testing.assert_allclose(batched_matmul(x, y), jnp.matmul(x, y, precision="highest"), atol=1e-3, rtol=1e-3)


def test_constant_integer_index(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def second_channel(x):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            out = out.at[i : i + 128, :].set(x[1, i : i + 128, :] * 2)
        return out

    x = rand((4, 256, 256))
    second_channel.verify(x)


def test_transposed_matmul(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def gram(x):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 384, 128):
                    acc += x[i : i + 128, k : k + 128] @ x[j : j + 128, k : k + 128].T
                out = out.at[i : i + 128, j : j + 128].set(acc)
        return out

    x = rand((256, 384))
    gram.verify(x)
    np.testing.assert_allclose(gram(x), jnp.dot(x, x.T, precision="highest"), atol=1e-3, rtol=1e-3)


def test_indexing_by_a_loop_with_step_above_one_is_rejected():
    @autokernel.tpu
    def bad(x):
        out = jnp.zeros((4, 256))
        for b in range(0, 4, 2):
            out = out.at[b : b + 2, :].set(x[b, :][None].repeat(2, 0))
        return out

    with pytest.raises(AutoKernelError, match="only loops stepping by 1 can index a dim"):
        bad.lower(rand((4, 256)))
