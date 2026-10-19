"""Feature combinations, each run in the interpreter with the race detector and cross-lowered to Mosaic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races
from test_mosaic_lowering import f32, i32, lower_for

import autokernel

M, T = 512, 128


@autokernel.tpu(buffers=3)
def linear_relu_3_buffers(x, w, bias):
    """A per-tile operand (bias) next to 3-deep input rings."""
    out = jnp.zeros((16, 256))
    for i in range(0, 16, 8):
        for j in range(0, 256, 128):
            acc = jnp.zeros((8, 128))
            for k in range(0, 256, 128):
                acc += x[i : i + 8, k : k + 128] @ w[k : k + 128, j : j + 128]
            out = out.at[i : i + 8, j : j + 128].set(jnp.maximum(acc + bias[:, j : j + 128], 0.0))
    return out


@autokernel.tpu(cores=2)
def two_masked_phases(x, w):
    """Two reduction loops, each with its own static mask."""
    out = jnp.zeros((M, M))
    for i in range(0, M, T):
        for j in range(0, M, T):
            sq = jnp.zeros((T, 128))
            for k in range(0, M, T):
                if k > i:
                    continue
                sq += jnp.sum(x[i : i + T, k : k + T] ** 2, axis=1, keepdims=True)
            inv = jax.lax.rsqrt(sq / M + 1e-6)
            acc = jnp.zeros((T, T))
            for kk in range(0, M, T):
                if kk > j:
                    continue
                acc += (x[i : i + T, kk : kk + T] * inv) @ w[kk : kk + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


@autokernel.tpu(cores=2)
def batched_causal(x, y):
    """Three parallel loops (one squeezed) with a static tile mask and background fills."""
    out = jnp.zeros((3, 256, 256))
    for b in range(3):
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                if j > i:
                    continue
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += x[b, i : i + 128, k : k + 128] @ y[b, k : k + 128, j : j + 128]
                out = out.at[b, i : i + 128, j : j + 128].set(acc)
    return out


@autokernel.tpu(cores=2)
def half_batches(x):
    """An output with a squeezed dim, a static tile mask over two loops and a non-zero fill."""
    out = jnp.full((4, 256, 256), -1.0)
    for b in range(4):
        for i in range(0, 256, 128):
            if b % 2 == 1 or i > 0:
                continue
            out = out.at[b, i : i + 128, :].set(x[b, i : i + 128, :] * 2)
    return out


def make_both_runtime(**options):
    """A runtime tile mask and a runtime reduction mask in one kernel."""

    def both_runtime(x, y, tile_on, blk):
        out = jnp.full((M, M), -1.0)
        for i in range(0, M, T):
            for j in range(0, M, T):
                if not tile_on[i // T, j // T]:
                    continue
                acc = jnp.zeros((T, T))
                for k in range(0, M, T):
                    if blk[k // T] == 0:
                        continue
                    acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
                out = out.at[i : i + T, j : j + T].set(acc)
        return out

    return autokernel.tpu(both_runtime, **options)


@autokernel.tpu
def offset_batches(x):
    """A reduction loop starting above 0 used as an integer index."""
    out = jnp.zeros((128, 256))
    for i in range(0, 128, 128):
        acc = jnp.zeros((128, 256))
        for b in range(1, 3):
            acc += x[b, i : i + 128, :]
        out = out.at[i : i + 128, :].set(acc)
    return out


@autokernel.tpu
def mm_i8(a, b):
    """int8 inputs, int32 accumulation (the MXU has no int32 x int32 matmul)."""
    c = jnp.zeros((256, 256), jnp.int32)
    for i in range(0, 256, 128):
        for j in range(0, 256, 128):
            acc = jnp.zeros((128, 128), jnp.int32)
            for k in range(0, 256, 128):
                acc += a[i : i + 128, k : k + 128] @ b[k : k + 128, j : j + 128]
            c = c.at[i : i + 128, j : j + 128].set(acc)
    return c


BOTH_RUNTIME_ARGS = (
    rand((M, M)),
    rand((M, M), 1),
    jnp.asarray(np.random.default_rng(2).random((4, 4)) < 0.5),
    jnp.asarray([1, 0, 1, 1], jnp.int32),
)


def test_per_tile_operand_with_three_buffers():
    args = rand((16, 256)), rand((256, 256), 1), rand((1, 256), 2)
    assert [op.phase for op in linear_relu_3_buffers.lower(*args).operands] == [0, 0, None]
    np.testing.assert_allclose(
        run_checking_races(linear_relu_3_buffers, *args), linear_relu_3_buffers.reference(*args), atol=1e-4
    )


def test_two_masked_phases():
    x, w = rand((M, M)), rand((M, M), 1)
    lw = two_masked_phases.lower(x, w)
    assert lw.schedule == "table" and lw.steps == 80
    np.testing.assert_allclose(
        run_checking_races(two_masked_phases, x, w), two_masked_phases.reference(x, w), atol=1e-3
    )


def test_three_parallel_loops_with_tile_mask():
    x, y = rand((3, 256, 256)), rand((3, 256, 256), 1)
    lw = batched_causal.lower(x, y)
    assert (lw.fill, lw.tiles, lw.total_tiles) == ("background", 9, 12)
    np.testing.assert_allclose(run_checking_races(batched_causal, x, y), batched_causal.reference(x, y), atol=1e-3)


def test_squeezed_output_dim_with_fill():
    x = rand((4, 256, 256))
    lw = half_batches.lower(x)
    assert (lw.fill, lw.tiles, lw.total_tiles, lw.steps) == ("background", 2, 8, 2)
    np.testing.assert_array_equal(run_checking_races(half_batches, x), half_batches.reference(x))


@pytest.mark.parametrize("cores", [1, 2])
def test_runtime_tile_and_reduction_masks(cores):
    kern = make_both_runtime(cores=cores)
    assert kern.lower(*BOTH_RUNTIME_ARGS).schedule == "runtime"
    np.testing.assert_allclose(
        run_checking_races(kern, *BOTH_RUNTIME_ARGS), kern.reference(*BOTH_RUNTIME_ARGS), atol=1e-3
    )


@pytest.mark.parametrize("cores", [1, 2, 3])
def test_background_fill_split_across_cores(cores):
    from test_masks import make_masked_matmul

    kern = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.full(s, -1.0), cores=cores)
    x, y = rand((512, 256)), rand((256, 512), 1)
    np.testing.assert_allclose(run_checking_races(kern, x, y), kern.reference(x, y), atol=1e-4, rtol=1e-4)


def test_reduction_loop_starting_above_zero():
    x = rand((3, 128, 256))
    assert "b_blk = (d + 1)" in offset_batches.source(x)
    np.testing.assert_allclose(run_checking_races(offset_batches, x), x[1] + x[2], atol=1e-6)


def test_integer_matmul_is_exact():
    rng = np.random.default_rng(0)
    a, b = (jnp.asarray(rng.integers(-50, 50, (256, 256)), jnp.int8) for _ in range(2))
    assert mm_i8.verify(a, b) == 0.0
    np.testing.assert_array_equal(mm_i8(a, b), np.asarray(a, np.int32) @ np.asarray(b, np.int32))


MOSAIC_CASES = {
    "per-tile-3-buffers": (linear_relu_3_buffers, (f32(16, 256), f32(256, 256), f32(1, 256))),
    "two-masked-phases": (two_masked_phases, (f32(M, M), f32(M, M))),
    "three-parallel-tile-mask": (batched_causal, (f32(3, 256, 256), f32(3, 256, 256))),
    "squeezed-output-fill": (half_batches, (f32(4, 256, 256),)),
    "both-runtime-masks": (
        make_both_runtime(),
        (f32(M, M), f32(M, M), jax.ShapeDtypeStruct((4, 4), jnp.bool_), i32(4)),
    ),
    "offset-reduction-index": (offset_batches, (f32(3, 128, 256),)),
    "int8-matmul": (mm_i8, (jax.ShapeDtypeStruct((256, 256), jnp.int8),) * 2),
}


@pytest.mark.parametrize("case", MOSAIC_CASES)
def test_lowers_for_v5e(case):
    kernel, specs = MOSAIC_CASES[case]
    assert "tpu_custom_call" in lower_for("v5e", kernel, *specs).as_text()


@pytest.mark.parametrize("case", ["two-masked-phases", "three-parallel-tile-mask", "both-runtime-masks"])
def test_lowers_for_v5p(case):
    kernel, specs = MOSAIC_CASES[case]
    assert "tpu_custom_call" in lower_for("v5p", kernel, *specs).as_text()
