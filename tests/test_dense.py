"""Kernels without masks: matmuls, elementwise ops, accumulators, precision, buffering and core splits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races

import autokernel

M, K, N = 256, 384, 256
TM, TN, TK = 64, 128, 128


@autokernel.tpu
def matmul(a, b):
    c = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):
                acc += a[i : i + TM, k : k + TK] @ b[k : k + TK, j : j + TN]
            c = c.at[i : i + TM, j : j + TN].set(acc)
    return c


def specs():
    return jax.ShapeDtypeStruct((M, K), jnp.float32), jax.ShapeDtypeStruct((K, N), jnp.float32)


def test_matmul_matches_reference_and_jnp():
    a, b = rand((M, K)), rand((K, N), 1)
    matmul.verify(a, b)
    np.testing.assert_allclose(matmul(a, b), jnp.dot(a, b, precision="highest"), atol=1e-4, rtol=1e-4)


def test_lowering_structure():
    lw = matmul.lower(*specs())
    assert (lw.schedule, lw.steps, lw.tiles, lw.cores, lw.buffers) == ("uniform", 24, 8, 1, 3)
    assert [op.block.kernel_shape for op in lw.operands] == [(TM, TK), (TK, TN)]
    assert [op.phase for op in lw.operands] == [0, 0]
    assert lw.state["acc"].aval.shape == (TM, TN)
    assert "grid=" not in lw.source and "pltpu.VMEM((3, 64, 128), jnp.float32)" in lw.source
    assert "cost_estimate=_COST" in lw.source and lw.estimate.flops >= 2 * M * N * K


def test_lowering_is_cached_per_signature():
    a, b = rand((M, K)), rand((K, N), 1)
    assert matmul.lower(a, b) is matmul.lower(jax.ShapeDtypeStruct((M, K), jnp.float32), b)


def test_works_under_jit():
    a, b = rand((M, K)), rand((K, N), 1)
    np.testing.assert_allclose(jax.jit(matmul)(a, b), matmul.reference(a, b), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("buffers", [2, 3, 4])
def test_deeper_input_buffering(buffers):
    kern = autokernel.tpu(matmul.fn, buffers=buffers)
    a, b = rand((M, K)), rand((K, N), 1)
    assert kern.lower(a, b).buffers == buffers
    np.testing.assert_allclose(run_checking_races(kern, a, b), matmul.reference(a, b), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cores", [2, 3])
def test_work_split_across_tensorcores(cores):
    kern = autokernel.tpu(matmul.fn, cores=cores)
    a, b = rand((M, K)), rand((K, N), 1)
    lw = kern.lower(a, b)
    assert lw.cores == cores and f"grid=({cores},)" in lw.source and 'dimension_semantics=("parallel",)' in lw.source
    kern.verify(a, b)


def test_elementwise_without_reduction():
    @autokernel.tpu(verify=True)
    def add_relu(x, y):
        out = jnp.zeros((256, 512))
        for i in range(0, 256, 64):
            for j in range(0, 512, 256):
                tile = x[i : i + 64, j : j + 256] + y[i : i + 64, j : j + 256]
                out = out.at[i : i + 64, j : j + 256].set(jnp.maximum(tile, 0))
        return out

    add_relu(rand((256, 512)), rand((256, 512), 1))
    assert add_relu.lower(rand((256, 512)), rand((256, 512), 1)).steps == 8


def test_per_tile_operand_and_fused_epilogue():
    """`bias` only changes between output tiles, so it's fetched once per tile, not once per step."""

    @autokernel.tpu
    def linear_relu(x, w, bias):
        B, D = x.shape
        H = w.shape[1]
        out = jnp.zeros((B, H))
        for i in range(0, B, 8):
            for j in range(0, H, 128):
                acc = jnp.zeros((8, 128))
                for k in range(0, D, 128):
                    acc += x[i : i + 8, k : k + 128] @ w[k : k + 128, j : j + 128]
                y = acc + bias[:, j : j + 128]
                out = out.at[i : i + 8, j : j + 128].set(jnp.maximum(y, 0.0))
        return out

    for B, D, H in [(16, 256, 256), (32, 128, 384)]:
        x, w, bias = rand((B, D)), rand((D, H), 1), rand((1, H), 2)
        linear_relu.verify(x, w, bias)
        assert [op.phase for op in linear_relu.lower(x, w, bias).operands] == [0, 0, None]


def test_operand_that_never_changes():
    @autokernel.tpu
    def scale_rows(x, scale):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 64):
            out = out.at[i : i + 64, :].set(x[i : i + 64, :] * scale[0:8, :][0:1])
        return out

    scale_rows.verify(rand((256, 256)), rand((8, 256), 1))


def test_reduction_over_part_of_an_axis():
    @autokernel.tpu
    def partial_matmul(a, b):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(128, 384, 128):
                    acc += a[i : i + 128, k : k + 128] @ b[k : k + 128, j : j + 128]
                out = out.at[i : i + 128, j : j + 128].set(acc)
        return out

    partial_matmul.verify(rand((256, 512)), rand((512, 256), 1))


def test_single_step_reduction():
    @autokernel.tpu
    def one_k(a, b):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 128, 128):
                    acc += a[i : i + 128, k : k + 128] @ b[k : k + 128, j : j + 128]
                out = out.at[i : i + 128, j : j + 128].set(acc)
        return out

    one_k.verify(rand((256, 128)), rand((128, 256), 1))


def test_loop_index_as_value():
    @autokernel.tpu
    def add_row_block_offset(x):
        out = jnp.zeros((64, 128))
        for i in range(0, 64, 16):
            out = out.at[i : i + 16, :].set(x[i : i + 16, :] + i)
        return out

    x = rand((64, 128))
    add_row_block_offset.verify(x)
    assert "i = i_blk * 16" in add_row_block_offset.source(x)


def test_same_input_with_two_tilings():
    @autokernel.tpu
    def square(a):
        c = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += a[i : i + 128, k : k + 128] @ a[k : k + 128, j : j + 128]
                c = c.at[i : i + 128, j : j + 128].set(acc)
        return c

    a = rand((256, 256))
    square.verify(a)
    assert len(square.lower(a).operands) == 2


def test_bf16_inputs_accumulate_in_f32():
    @autokernel.tpu
    def mm_bf16(a, b):
        c = jnp.zeros((128, 256), jnp.bfloat16)
        for i in range(0, 128, 32):
            for j in range(0, 256, 128):
                acc = jnp.zeros((32, 128), jnp.bfloat16)  # widened to an f32 VMEM accumulator
                for k in range(0, 256, 128):
                    acc += a[i : i + 32, k : k + 128] @ b[k : k + 128, j : j + 128]
                c = c.at[i : i + 32, j : j + 128].set(acc.astype(jnp.bfloat16))
        return c

    a, b = rand((128, 256), dtype=jnp.bfloat16), rand((256, 256), 1, dtype=jnp.bfloat16)
    assert mm_bf16.lower(a, b).state["acc"].aval.dtype == jnp.float32
    assert mm_bf16(a, b).dtype == jnp.bfloat16
    mm_bf16.verify(a, b)


def test_default_precision_is_the_faster_path():
    a, b = rand((M, K)), rand((K, N), 1)
    fast = autokernel.tpu(matmul.fn, matmul_precision="default")
    assert "precision = None if" in fast.source(a, b)
    assert "precision = jax.lax.Precision.HIGHEST if" in matmul.source(a, b)
    fast.verify(a, b)


def test_kernel_has_no_races():
    a, b = rand((M, K)), rand((K, N), 1)
    out = run_checking_races(matmul, a, b)
    np.testing.assert_allclose(out, matmul.reference(a, b), atol=1e-4, rtol=1e-4)
