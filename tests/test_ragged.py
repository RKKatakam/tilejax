"""Ragged loops: `range(0, M, T)` with `M` not a multiple of `T`.

The last tile is a whole tile running past `M`: it reads zeros past the end of an input and drops what it
writes past the end of an output (the kernel's wrapper zero-pads the inputs and crops the outputs). The
reference reads and writes tiles the same way (`reference.read_tile`, `reference.store_tile`), so `verify`
checks the kernel against plain Python with that one rule.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, reference

M, K, N = 200, 300, 260
TM, TN, TK = 64, 128, 128


def matmul(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc)
    return out


def _f64(a):
    return np.asarray(a, np.float64)


def test_ragged_matmul_matches_numpy():
    kern = autokernel.tpu(matmul)
    x, y = rand((M, K)), rand((K, N), 1)
    got = kern(x, y)
    assert got.shape == (M, N)
    np.testing.assert_allclose(got, _f64(x) @ _f64(y), rtol=1e-4, atol=1e-3)
    kern.verify(x, y)
    lw = kern.lower(x, y)
    # 4 x 3 tiles of 3 steps each: ceil(200 / 64), ceil(260 / 128), ceil(300 / 128)
    assert (lw.tiles, lw.steps) == (12, 36)
    assert lw.plan.spec.pads == {"x": (256, 384), "y": (384, 384)}
    assert lw.outputs[0].padded == (256, 384) and lw.outputs[0].kernel_out.shape == (256, 384)
    assert lw.plan.spec.ragged and lw.plan.spec.loops["i"].ragged and lw.plan.spec.loops["i"].end == M


def test_ragged_source_pads_inputs_and_crops_outputs():
    kern = autokernel.tpu(matmul)
    x, y = rand((M, K)), rand((K, N), 1)
    src = kern.source(x, y)
    assert "x = jnp.pad(x, ((0, 56), (0, 84)))" in src
    assert "y = jnp.pad(y, ((0, 84), (0, 124)))" in src
    assert "out_shape=jax.ShapeDtypeStruct((256, 384), jnp.float32)" in src
    assert "[:200, :260]" in src
    # The source runs on its own and computes what the kernel does.
    namespace = {}
    exec(src, namespace)
    fn = next(v for k, v in namespace.items() if k.endswith("_pallas"))
    from jax.experimental.pallas import tpu as pltpu

    np.testing.assert_allclose(fn(x, y, interpret=pltpu.InterpretParams()), kern(x, y), rtol=1e-6)


def test_aligned_loops_have_no_padding():
    def aligned(x):
        out = jnp.zeros((256, 128))
        for i in range(0, 256, 64):
            out = out.at[i : i + 64, :].set(x[i : i + 64, :] * 2)
        return out

    lw = autokernel.tpu(aligned).lower(rand((256, 128)))
    assert not lw.plan.spec.ragged and "jnp.pad" not in lw.source and lw.outputs[0].kernel_out == lw.outputs[0].out


def test_ragged_parallel_loop_only():
    R, C, TR = 100, 256, 16

    @autokernel.tpu
    def rowsum(a):
        out = jnp.zeros((R, 1))
        for r in range(0, R, TR):
            out = out.at[r : r + TR, :].set(jnp.sum(jnp.exp(a[r : r + TR, :]), axis=1, keepdims=True))
        return out

    a = rand((R, C), 2)
    np.testing.assert_allclose(rowsum(a), np.exp(_f64(a)).sum(1, keepdims=True), rtol=1e-5)
    rowsum.verify(a)


def test_ragged_reduction_reads_zeros_past_the_end():
    """A max over a ragged reduction sees the zeros past the end, in the kernel and the reference alike;
    masking with the loop variable gives the max of the real elements."""
    R, C, TC = 64, 200, 128

    def plain_max(a):
        out = jnp.zeros((R, 128))
        for r in range(0, R, 8):
            m = jnp.full((8, 128), -jnp.inf)
            for c in range(0, C, TC):
                m = jnp.maximum(m, jnp.max(a[r : r + 8, c : c + TC], axis=1, keepdims=True))
            out = out.at[r : r + 8, :].set(m)
        return out

    def masked_max(a):
        out = jnp.zeros((R, 128))
        for r in range(0, R, 8):
            m = jnp.full((8, 128), -jnp.inf)
            for c in range(0, C, TC):
                t = jnp.where(c + jnp.arange(TC)[None, :] < C, a[r : r + 8, c : c + TC], -jnp.inf)
                m = jnp.maximum(m, jnp.max(t, axis=1, keepdims=True))
            out = out.at[r : r + 8, :].set(m)
        return out

    a = -jnp.abs(rand((R, C), 3)) - 1.0  # every element negative
    plain, masked = autokernel.tpu(plain_max), autokernel.tpu(masked_max)
    plain.verify(a)
    masked.verify(a)
    assert float(jnp.max(plain(a))) == 0.0  # the padding's zeros
    np.testing.assert_allclose(masked(a)[:, 0], _f64(a).max(1), rtol=1e-6)


def test_ragged_with_a_tile_mask():
    S, T = 300, 128

    @autokernel.tpu
    def causal(x, y):
        out = jnp.zeros((S, S))
        for i in range(0, S, T):
            for j in range(0, S, T):
                if j > i:
                    continue
                out = out.at[i : i + T, j : j + T].set(x[i : i + T, :] @ y[:, j : j + T])
        return out

    x, y = rand((S, 128)), rand((128, S), 1)
    causal.verify(x, y)
    want = _f64(x) @ _f64(y)
    blocks = np.arange(S) // T
    want = np.where(blocks[None, :] > blocks[:, None], 0.0, want)
    np.testing.assert_allclose(causal(x, y), want, rtol=1e-4, atol=1e-3)


def test_ragged_output_keeping_its_values_pads_the_aliased_allocation():
    S, T = 200, 64

    @autokernel.tpu(background_fill=False)
    def masked(x):
        out = jnp.ones((S, 128))
        for i in range(0, S, T):
            if i == 64:
                continue
            out = out.at[i : i + T, :].set(x[i : i + T, :] * 2)
        return out

    x = rand((S, 128))
    lw = masked.lower(x)
    assert lw.fill == "prefill" and "jnp.pad(jnp.ones((S, 128)" in lw.source
    want = np.where((np.arange(S) // T == 1)[:, None], 1.0, 2 * _f64(x))
    np.testing.assert_allclose(masked(x), want, rtol=1e-6)
    masked.verify(x)


def test_ragged_tuple_outputs():
    S, T = 72, 16

    @autokernel.tpu
    def two(x):
        a = jnp.zeros((S, 128))
        b = jnp.zeros((S, 128), jnp.int32)
        for i in range(0, S, T):
            a = a.at[i : i + T, :].set(x[i : i + T, :] + 1)
            b = b.at[i : i + T, :].set((x[i : i + T, :] > 0).astype(jnp.int32))
        return a, b

    x = rand((S, 128))
    a, b = two(x)
    assert a.shape == b.shape == (S, 128)
    np.testing.assert_allclose(a, _f64(x) + 1, rtol=1e-6)
    np.testing.assert_array_equal(b, (_f64(x) > 0).astype(np.int32))
    two.verify(x)


def test_ragged_kernels_transform():
    kern = autokernel.tpu(matmul)
    x, y = rand((M, K)), rand((K, N), 1)
    jitted = jax.jit(kern)(x, y)
    np.testing.assert_allclose(jitted, kern(x, y), rtol=1e-6)
    gx = jax.grad(lambda x: kern(x, y).sum())(x)
    np.testing.assert_allclose(gx, np.ones((M, N)) @ _f64(y).T, rtol=1e-4, atol=1e-3)  # not jnp: one bf16 pass on TPU
    xs = jnp.stack([x, 2 * x])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*can't batch")  # one kernel for the batch, not a loop
        batched = jax.vmap(kern, in_axes=(0, None))(xs, y)
    np.testing.assert_allclose(batched[1], 2 * kern(x, y), rtol=1e-5, atol=1e-4)


def test_tiles_must_start_inside_the_array():
    def overrun(x):
        out = jnp.zeros((128, 128))
        for i in range(0, 256, 64):  # tiles at 128 and 192 start past x's end
            out = out.at[i : i + 64, :].set(x[i : i + 64, :])
        return out

    with pytest.raises(AutoKernelError, match="runs to 256 but dim 0 only has size 128"):
        autokernel.tpu(overrun).lower(rand((128, 128)))


def test_a_loop_must_still_start_on_a_tile():
    def misaligned(x):
        out = jnp.zeros((200, 128))
        for i in range(8, 200, 64):
            out = out.at[i : i + 64, :].set(x[i : i + 64, :])
        return out

    with pytest.raises(AutoKernelError, match="start divisible by step"):
        autokernel.tpu(misaligned).lower(rand((200, 128)))


def test_ragged_nest_in_a_program_runs_unfused():
    S, T = 200, 64

    @autokernel.tpu
    def prog(x):
        h = jnp.zeros((S, 128))
        for i in range(0, S, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * 2)
        out = jnp.zeros((S, 128))
        for i in range(0, S, T):
            out = out.at[i : i + T, :].set(h[i : i + T, :] + 1)
        return out

    x = rand((S, 128))
    np.testing.assert_allclose(prog(x), 2 * _f64(x) + 1, rtol=1e-6)
    prog.verify(x)
    assert len(prog.lower(x)) == 2  # aligned, the two nests would be one fused kernel


def test_tuning_keeps_loops_aligned():
    T = 64

    def f(x):
        out = jnp.zeros((256, 128))
        for i in range(0, 256, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] + 1)
        return out

    kern = autokernel.tpu(f, tune={"T": (64, 96)})
    x = rand((256, 128))
    kern(x)
    by_value = {c.overrides["T"]: c for c in kern.tuning(x)}
    assert by_value[64].error is None
    assert "would be ragged" in by_value[96].error


def test_tuning_a_ragged_loop_as_written():
    T = 64

    def f(x):
        out = jnp.zeros((200, 128))
        for i in range(0, 200, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] + 1)
        return out

    kern = autokernel.tpu(f, tune={"T": (64, 128)})
    x = rand((200, 128))
    assert all(c.error is None for c in kern.tuning(x))
    kern.verify(x)


# --------------------------------------------------------------------------- the reference's tile reads


def test_read_tile_pads_with_zeros():
    x = jnp.arange(10.0).reshape(2, 5)
    np.testing.assert_array_equal(
        reference.read_tile(x, (slice(0, 2), slice(3, 8))), [[3, 4, 0, 0, 0], [8, 9, 0, 0, 0]]
    )
    np.testing.assert_array_equal(reference.read_tile(x, (1, slice(4, 6))), [9, 0])  # integer-indexed dim dropped
    np.testing.assert_array_equal(reference.read_tile(x, slice(1, 3)), [[5, 6, 7, 8, 9], [0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(reference.read_tile(x, (0, slice(np.int64(0), np.int64(2)))), [0, 1])
    assert reference.read_tile(x, (slice(None), slice(None))).shape == (2, 5)  # nothing past the end


def test_read_tile_with_too_many_indices_is_plain_indexing():
    x = jnp.arange(4.0)
    with pytest.raises(IndexError):
        reference.read_tile(x, (0, 1))


def test_store_tile_drops_what_lies_past_the_end():
    out = jnp.zeros((3, 4))
    got = reference.store_tile(out, (slice(2, 4), slice(0, 4)), jnp.ones((2, 4)))
    np.testing.assert_array_equal(got, [[0] * 4, [0] * 4, [1] * 4])
    got = reference.store_tile(out, (slice(2, 4), slice(0, 4)), 5.0)  # a scalar broadcasts
    np.testing.assert_array_equal(got[2], [5] * 4)
    got = reference.store_tile(out, (slice(2, 4), slice(0, 4)), jnp.ones((1, 4)))  # broadcast along the dim
    np.testing.assert_array_equal(got[2], [1] * 4)
    got = reference.store_tile(out, slice(0, 2), jnp.ones((2, 4)))  # in bounds
    np.testing.assert_array_equal(got[:2], np.ones((2, 4)))
    got = reference.store_tile(jnp.zeros((2, 3, 4)), (0, slice(2, 4)), jnp.ones((2, 4)))
    np.testing.assert_array_equal(got[0, 2], [1] * 4)
