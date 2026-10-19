"""Loop nests beyond one reduction and one output: hoisted statements, several outputs, several and nested
reduction loops, batched matmuls, scalar state, and masks that read input arrays at run time."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races

import autokernel


@autokernel.tpu
def scale_per_batch(x):
    out = jnp.zeros((4, 256, 256))
    for b in range(0, 4, 1):
        scale = b + 1.0  # runs once per tile: it only reads the outer loop
        for i in range(0, 256, 128):
            out = out.at[b, i : i + 128, :].set(x[b, i : i + 128, :] * scale)
    return out


@autokernel.tpu
def matmul_and_relu(a, b):
    c = jnp.zeros((256, 256))
    r = jnp.zeros((256, 256), jnp.bfloat16)
    for i in range(0, 256, 128):
        for j in range(0, 256, 128):
            acc = jnp.zeros((128, 128))
            for k in range(0, 256, 128):
                acc += a[i : i + 128, k : k + 128] @ b[k : k + 128, j : j + 128]
            c = c.at[i : i + 128, j : j + 128].set(acc)
            r = r.at[i : i + 128, j : j + 128].set(jnp.maximum(acc, 0).astype(jnp.bfloat16))
    return c, r


RM, RK, RN, RT = 256, 512, 256, 128


@autokernel.tpu
def rmsnorm_matmul(x, w):
    """First reduction: sum of squares over the row. Second: (x * rsqrt(mean)) @ w over the same row."""
    out = jnp.zeros((RM, RN))
    for i in range(0, RM, RT):
        for j in range(0, RN, RT):
            sq = jnp.zeros((RT, 128))
            for k in range(0, RK, RT):
                sq += jnp.sum(x[i : i + RT, k : k + RT] ** 2, axis=1, keepdims=True)
            inv = jax.lax.rsqrt(sq / RK + 1e-6)
            acc = jnp.zeros((RT, RT))
            for kk in range(0, RK, RT):
                acc += (x[i : i + RT, kk : kk + RT] * inv) @ w[kk : kk + RT, j : j + RT]
            out = out.at[i : i + RT, j : j + RT].set(acc)
    return out


@autokernel.tpu
def sum_of_matmuls(x, y):
    """sum_b x[b] @ y[b]: a perfect nest of two reduction loops."""
    out = jnp.zeros((256, 256))
    for i in range(0, 256, 128):
        for j in range(0, 256, 128):
            acc = jnp.zeros((128, 128))
            for b in range(0, 3, 1):
                for k in range(0, 256, 128):
                    acc += x[b, i : i + 128, k : k + 128] @ y[b, k : k + 128, j : j + 128]
            out = out.at[i : i + 128, j : j + 128].set(acc)
    return out


@autokernel.tpu
def bmm(x, y):
    out = jnp.zeros((4, 256, 256))
    for b in range(0, 4, 2):
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((2, 128, 128))
                for k in range(0, 256, 128):
                    acc += x[b : b + 2, i : i + 128, k : k + 128] @ y[b : b + 2, k : k + 128, j : j + 128]
                out = out.at[b : b + 2, i : i + 128, j : j + 128].set(acc)
    return out


@autokernel.tpu
def scaled_by_block_count(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, 128):
        n = 0.0
        acc = jnp.zeros((128, 256))
        for k in range(0, 512, 128):
            n = n + 1.0
            acc += x[i : i + 128, k : k + 128] @ jnp.ones((128, 256))
        out = out.at[i : i + 128, :].set(acc / n)
    return out


def make_block_sparse(M, K, N, T, **options):
    """x @ y using only the (row block, k block) pairs that block_mask marks: a runtime reduction mask."""

    def block_sparse(x, y, block_mask):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc = jnp.zeros((T, T))
                for k in range(0, K, T):
                    if block_mask[i // T, k // T] == 0:
                        continue
                    acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
                out = out.at[i : i + T, j : j + T].set(acc)
        return out

    return autokernel.tpu(block_sparse, **options)


def make_masked_copy(**options):
    """Copies (doubled) only the tiles tile_on marks: a runtime tile mask."""

    def masked_copy(x, tile_on):
        out = jnp.full((512, 256), -1.0)
        for i in range(0, 512, 128):
            for j in range(0, 256, 128):
                if not tile_on[i // 128, j // 128]:
                    continue
                out = out.at[i : i + 128, j : j + 128].set(x[i : i + 128, j : j + 128] * 2)
        return out

    return autokernel.tpu(masked_copy, **options)


def block_sparse_reference(x, y, block_mask, T):
    keep = np.kron(np.asarray(block_mask), np.ones((T, T)))
    return jnp.dot(x * keep, y, precision="highest")


# --------------------------------------------------------------------------- tests


def test_statements_between_parallel_loops():
    x = rand((4, 256, 256))
    scale_per_batch.verify(x)
    np.testing.assert_allclose(scale_per_batch(x), x * jnp.arange(1, 5)[:, None, None])


def test_multiple_outputs():
    a, b = rand((256, 256)), rand((256, 256), 1)
    c, r = matmul_and_relu(a, b)
    assert (c.dtype, r.dtype) == (jnp.float32, jnp.bfloat16)
    matmul_and_relu.verify(a, b)
    np.testing.assert_allclose(np.asarray(r, np.float32), np.maximum(np.asarray(c), 0), atol=2e-2, rtol=1e-2)


def test_two_reduction_loops_fused_rmsnorm_matmul():
    x, w = rand((RM, RK)), rand((RK, RN), 1)
    assert rmsnorm_matmul.lower(x, w).steps == (RM // RT) * (RN // RT) * 2 * (RK // RT)  # two phases per tile
    rmsnorm_matmul.verify(x, w)
    norm = x / np.sqrt(np.mean(np.asarray(x) ** 2, axis=1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(rmsnorm_matmul(x, w), jnp.dot(norm, w, precision="highest"), atol=1e-3, rtol=1e-3)


def test_nested_reduction_loops():
    x, y = rand((3, 256, 256)), rand((3, 256, 256), 1)
    assert sum_of_matmuls.lower(x, y).steps == 4 * 3 * 2
    sum_of_matmuls.verify(x, y)
    want = jnp.einsum("bik,bkj->ij", x, y, precision="highest")
    np.testing.assert_allclose(sum_of_matmuls(x, y), want, atol=1e-3, rtol=1e-3)


def test_batched_matmul_on_3d_tiles():
    x, y = rand((4, 256, 256)), rand((4, 256, 256), 1)
    bmm.verify(x, y)
    np.testing.assert_allclose(bmm(x, y), jnp.matmul(x, y, precision="highest"), atol=1e-3, rtol=1e-3)


def test_scalar_state_lives_in_smem():
    x = rand((256, 512))
    assert scaled_by_block_count.lower(x).state["n"].smem
    scaled_by_block_count.verify(x)


@pytest.mark.parametrize("cores", [1, 2])
def test_runtime_reduction_mask(cores):
    M, K, N, T = 256, 512, 256, 128
    kern = make_block_sparse(M, K, N, T, cores=cores)
    x, y = rand((M, K)), rand((K, N), 1)
    rng = np.random.default_rng(0)
    for density in (0.5, 0.0, 1.0):
        mask = jnp.asarray((rng.random((M // T, K // T)) < density).astype(np.int32))
        np.testing.assert_allclose(kern(x, y, mask), block_sparse_reference(x, y, mask, T), atol=1e-3, rtol=1e-3)
    lw = kern.lower(x, y, mask)
    assert lw.schedule == "runtime" and "_runtime_tables" in lw.source


def test_runtime_reduction_mask_has_no_races():
    M, K, N, T = 256, 512, 256, 128
    kern = make_block_sparse(M, K, N, T)
    x, y = rand((M, K)), rand((K, N), 1)
    mask = jnp.asarray([[1, 0, 1, 1], [0, 0, 1, 0]], jnp.int32)
    np.testing.assert_allclose(run_checking_races(kern, x, y, mask), block_sparse_reference(x, y, mask, T), atol=1e-3)


@pytest.mark.parametrize("cores", [1, 2])
def test_runtime_tile_mask(cores):
    kern = make_masked_copy(cores=cores)
    x = rand((512, 256))
    on = jnp.asarray([[1, 0], [0, 0], [1, 1], [0, 1]], bool)
    want = np.where(np.kron(np.asarray(on), np.ones((128, 128))) > 0, np.asarray(x) * 2, -1.0)
    np.testing.assert_allclose(kern(x, on), want)
    np.testing.assert_allclose(kern(x, jnp.zeros_like(on)), -np.ones((512, 256)))  # no tile computed
    assert kern.lower(x, on).fill == "prefill"


def test_runtime_mask_combined_with_static_mask():
    """Causal (static) and a per-k-block keep list (runtime) in one reduction mask."""
    M, T = 512, 128

    @autokernel.tpu
    def causal_sparse(x, y, keep):
        out = jnp.zeros((M, M))
        for i in range(0, M, T):
            for j in range(0, M, T):
                acc = jnp.zeros((T, T))
                for k in range(0, M, T):
                    if k > i or keep[k // T] == 0:
                        continue
                    acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
                out = out.at[i : i + T, j : j + T].set(acc)
        return out

    x, y = rand((M, M)), rand((M, M), 1)
    causal_sparse.verify(x, y, jnp.asarray([1, 0, 1, 1], jnp.int32))
