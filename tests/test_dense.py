"""Kernels without tile masks: matmuls, elementwise ops, accumulators and precision, on both backends."""

import jax
import jax.numpy as jnp
import numpy as np
from helpers import rand

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


def test_matmul_matches_reference_and_jnp(pipeline):
    kern = autokernel.tpu(matmul.fn, pipeline=pipeline)
    a, b = rand((M, K)), rand((K, N), 1)
    kern.verify(a, b)
    np.testing.assert_allclose(kern(a, b), jnp.dot(a, b, precision="highest"), atol=1e-4, rtol=1e-4)


def test_manual_is_the_default_and_has_no_grid():
    lw = matmul.lower(*specs())
    assert (lw.pipeline, lw.grid, lw.steps) == ("manual", (), (M // TM) * (N // TN) * (K // TK))
    assert "grid=" not in lw.source and "pl.loop(0, 24" in lw.source
    assert "pltpu.VMEM((2, 64, 128), jnp.float32)" in lw.source  # double-buffered x tiles


def test_grid_lowering_structure():
    lw = autokernel.tpu(matmul.fn, pipeline="grid").lower(*specs())
    assert (lw.pipeline, lw.grid) == ("grid", (M // TM, N // TN, K // TK))
    assert [op.block.shape for op in lw.operands] == [(TM, TK), (TK, TN)]
    assert [op.block.index for op in lw.operands] == [("i", "k"), ("k", "j")]
    assert lw.out_block.shape == (TM, TN)
    assert "dimension_semantics=('parallel', 'parallel', 'arbitrary')" in lw.source
    assert "pltpu.VMEM((64, 128), jnp.float32)" in lw.source


def test_lowering_is_cached_per_signature():
    a, b = rand((M, K)), rand((K, N), 1)
    assert matmul.lower(a, b) is matmul.lower(jax.ShapeDtypeStruct((M, K), jnp.float32), b)


def test_works_under_jit(pipeline):
    kern = autokernel.tpu(matmul.fn, pipeline=pipeline)
    a, b = rand((M, K)), rand((K, N), 1)
    np.testing.assert_allclose(jax.jit(kern)(a, b), kern.reference(a, b), atol=1e-4, rtol=1e-4)


def test_elementwise_without_reduction(pipeline):
    @autokernel.tpu(verify=True, pipeline=pipeline)
    def add_relu(x, y):
        out = jnp.zeros((256, 512))
        for i in range(0, 256, 64):
            for j in range(0, 512, 256):
                tile = x[i : i + 64, j : j + 256] + y[i : i + 64, j : j + 256]
                out = out.at[i : i + 64, j : j + 256].set(jnp.maximum(tile, 0))
        return out

    add_relu(rand((256, 512)), rand((256, 512), 1))
    assert add_relu.lower(rand((256, 512)), rand((256, 512), 1)).steps == 8


def test_per_tile_operand_and_fused_epilogue(pipeline):
    """`bias` only changes between output tiles, so the manual backend fetches it once per column."""

    @autokernel.tpu(pipeline=pipeline)
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
        linear_relu.verify(rand((B, D)), rand((D, H), 1), rand((1, H), 2))


def test_operand_that_never_changes(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def scale_rows(x, scale):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 64):
            out = out.at[i : i + 64, :].set(x[i : i + 64, :] * scale[0:8, :][0:1])
        return out

    scale_rows.verify(rand((256, 256)), rand((8, 256), 1))


def test_reduction_over_part_of_an_axis(pipeline):
    """A reduction loop that starts past 0 offsets every reduction block."""

    @autokernel.tpu(pipeline=pipeline)
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


def test_single_step_reduction(pipeline):
    @autokernel.tpu(pipeline=pipeline)
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


def test_batched_three_dim_tiles(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def batched_scale(x):
        out = jnp.zeros((4, 64, 256))
        for b in range(0, 4, 1):
            for i in range(0, 64, 32):
                out = out.at[b : b + 1, i : i + 32, :].set(x[b : b + 1, i : i + 32, :] * (b + 1))
        return out

    batched_scale.verify(rand((4, 64, 256)))


def test_loop_index_as_value(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def add_row_block_offset(x):
        out = jnp.zeros((64, 128))
        for i in range(0, 64, 16):
            out = out.at[i : i + 16, :].set(x[i : i + 16, :] + i)
        return out

    x = rand((64, 128))
    add_row_block_offset.verify(x)
    expected = "i = pl.program_id(0) * 16" if pipeline == "grid" else "i = i_blocks[tile] * 16"
    assert expected in add_row_block_offset.source(x)


def test_same_input_with_two_tilings(pipeline):
    @autokernel.tpu(pipeline=pipeline)
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


def test_bf16_inputs_accumulate_in_f32(pipeline):
    @autokernel.tpu(pipeline=pipeline)
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
    lw = mm_bf16.lower(a, b)
    assert lw.scratch["acc"].dtype == jnp.float32
    assert mm_bf16(a, b).dtype == jnp.bfloat16
    mm_bf16.verify(a, b)


def test_default_precision_is_the_faster_path():
    a, b = rand((M, K)), rand((K, N), 1)
    fast = autokernel.tpu(matmul.fn, matmul_precision="default")
    assert "precision = None if" in fast.source(a, b)
    assert "precision = jax.lax.Precision.HIGHEST if" in matmul.source(a, b)
    fast.verify(a, b)


def test_manual_kernel_has_no_races():
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
    from jax.experimental.pallas import tpu as pltpu

    a, b = rand((M, K)), rand((K, N), 1)
    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    out = matmul.lower(a, b).entry(a, b, interpret=params).block_until_ready()
    assert interpreter.races is not None and not interpreter.races.races_found
    np.testing.assert_allclose(out, matmul.reference(a, b), atol=1e-4, rtol=1e-4)
