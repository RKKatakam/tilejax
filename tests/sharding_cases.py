"""Cases for test_sharding.py, which runs this file with pytest in a process with four CPU devices.

Run it directly with

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu pytest tests/sharding_cases.py
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import autokernel

DEVICES = 4
pytestmark = pytest.mark.skipif(jax.device_count() < DEVICES, reason=f"needs {DEVICES} devices")

M, K, N = 256, 256, 256
TM = 64


def matmul(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, 128):
            acc = jnp.zeros((TM, 128))
            for k in range(0, K, 128):
                acc += x[i : i + TM, k : k + 128] @ y[k : k + 128, j : j + 128]
            out = out.at[i : i + TM, j : j + 128].set(acc)
    return out


def plain_matmul(x, y):
    return jnp.dot(x, y, precision="highest")


def mesh(shape=(DEVICES,), names=("d",), explicit=True):
    kind = AxisType.Explicit if explicit else AxisType.Auto
    return jax.make_mesh(shape, names, axis_types=(kind,) * len(shape))


def shard(x, m, *spec):
    return jax.device_put(x, NamedSharding(m, P(*spec)))


def close(got, want, tol=1e-4):
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tol, atol=tol)


@pytest.mark.parametrize("explicit", [True, False])
def test_row_sharded_matmul_runs_per_shard(explicit):
    kernel = autokernel.tpu(matmul)
    m = mesh(explicit=explicit)
    x, y = rand((M, K)), rand((K, N), 1)
    xs = shard(x, m, "d", None)
    out = kernel(xs, y)
    close(out, plain_matmul(x, y))
    assert out.sharding.spec == P("d", None)  # the output comes out sharded like the rows
    (local,) = kernel._local.values()
    lowered = local.lower(jax.ShapeDtypeStruct((M // DEVICES, K), x.dtype), y)
    assert lowered.total_tiles == (M // DEVICES // TM) * (N // 128)  # each device runs a quarter of the tiles
    if explicit:  # under jit the sharding is part of the traced value's type
        out = jax.jit(kernel)(xs, y)
        close(out, plain_matmul(x, y))
        assert out.sharding.spec == P("d", None)


def test_grad_and_jvp_with_sharded_inputs():
    kernel = autokernel.tpu(matmul)
    m = mesh()
    x, y = rand((M, K)), rand((K, N), 1) / 16
    xs = shard(x, m, "d", None)
    loss = lambda f: lambda x, y: jnp.sum(jnp.tanh(f(x, y)) ** 2)  # noqa: E731
    got = jax.jit(jax.grad(loss(kernel), argnums=(0, 1)))(xs, y)
    want = jax.grad(loss(plain_matmul), argnums=(0, 1))(x, y)
    for g, w in zip(got, want, strict=True):
        close(g, w, tol=1e-3)
    tangents = (rand((M, K), 2), rand((K, N), 3))
    _, t = jax.jvp(kernel, (xs, y), tangents)
    close(t, jax.jvp(plain_matmul, (x, y), tangents)[1], tol=1e-3)


def test_reduction_sharded_inputs_are_gathered_with_a_warning():
    kernel = autokernel.tpu(matmul)
    m = mesh()
    x, y = rand((M, K)), rand((K, N), 1)
    with pytest.warns(UserWarning, match=r"`y` dim 0 \(tiled by reduction loop `k`\)"):
        out = kernel(x, shard(y, m, "d", None))
    close(out, plain_matmul(x, y))


B, H, S, D = 2, 2, 256, 128
T = 128


def heads(q, k):
    # Per batch and head: a causal tile mask on the rows, and the loops over b and h only index.
    out = jnp.zeros((B, H, S, S))
    for b in range(B):
        for h in range(H):
            for i in range(0, S, T):
                for j in range(0, S, T):
                    if j > i:
                        continue
                    out = out.at[b, h, i : i + T, j : j + T].set(q[b, h, i : i + T, :] @ k[b, h, j : j + T, :].T)
    return out


def plain_heads(q, k):
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision="highest")
    blocks = np.kron(np.tril(np.ones((S // T, S // T))), np.ones((T, T)))
    return jnp.where(blocks > 0, s, 0.0)


def test_batch_and_heads_split_over_a_2d_mesh():
    kernel = autokernel.tpu(heads)
    m = mesh((2, 2), ("data", "model"))
    q, k = rand((B, H, S, D)), rand((B, H, S, D), 1)
    qs, ks = shard(q, m, "data", "model"), shard(k, m, "data", None)  # k's heads get split as well
    out = jax.jit(kernel)(qs, ks)
    close(out, plain_heads(q, k), tol=1e-3)
    assert out.sharding.spec == P("data", "model", None, None)
    (local,) = kernel._local.values()
    lowered = local.lower(jax.ShapeDtypeStruct((1, 1, S, D), q.dtype), jax.ShapeDtypeStruct((1, 1, S, D), k.dtype))
    assert (lowered.tiles, lowered.total_tiles) == (3, 4)  # one (b, h) per device, the mask unchanged
    from test_mosaic_lowering import lower_for

    lower_for("v5e", local, *[jax.ShapeDtypeStruct((1, 1, S, D), jnp.float32)] * 2)  # compiles for a TPU too


def row_offsets(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TM, N), 0)  # reads the global row index
        out = out.at[i : i + TM, :].set(x[i : i + TM, :] + rows)
    return out


def test_a_loop_used_as_a_value_is_not_split():
    kernel = autokernel.tpu(row_offsets)
    x = rand((M, N))
    with pytest.warns(UserWarning, match="loop `i` is used for more than indexing"):
        out = kernel(shard(x, mesh(), "d", None))
    close(out, x + np.arange(M)[:, None])


def causal(x, y):
    out = jnp.full((M, N), -1.0)
    for i in range(0, M, 128):
        for j in range(0, N, 128):
            if j > i:
                continue
            acc = jnp.zeros((128, 128))
            for k in range(0, K, 128):
                acc += x[i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
            out = out.at[i : i + 128, j : j + 128].set(acc)
    return out


def test_vmap_over_a_sharded_batch():
    kernel = autokernel.tpu(causal)
    m = mesh()
    xs, ys = jnp.stack([rand((M, K), s) for s in range(4)]), jnp.stack([rand((K, N), s + 4) for s in range(4)])
    want = jax.vmap(kernel.reference)(xs, ys)
    out = jax.jit(jax.vmap(kernel))(shard(xs, m, "d", None, None), shard(ys, m, "d", None, None))
    close(out, want, tol=1e-3)
    assert out.sharding.spec == P("d", None, None)
    (batched,) = kernel._batched.values()
    assert batched._local  # the batched kernel split its batch loop across the devices


def two_layers(x, w1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, 128):
            acc = jnp.zeros((TM, 128))
            for k in range(0, K, 128):
                acc += x[i : i + TM, k : k + 128] @ w1[k : k + 128, j : j + 128]
            h = h.at[i : i + TM, j : j + 128].set(jnp.tanh(acc))
    out = jnp.zeros((M, N))  # fused with the first nest when nothing is sharded
    for i in range(0, M, TM):
        for j in range(0, N, 128):
            acc = jnp.zeros((TM, 128))
            for k in range(0, N, 128):
                acc += h[i : i + TM, k : k + 128] @ w2[k : k + 128, j : j + 128]
            out = out.at[i : i + TM, j : j + 128].set(acc)
    return out


def test_a_fused_program_on_sharded_rows():
    prog = autokernel.tpu(two_layers)
    m = mesh()
    x, w1, w2 = rand((M, K)), rand((K, N), 1) / 16, rand((N, N), 2) / 16
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing gathered
        warnings.filterwarnings("ignore", message=".*interpreter")
        out = prog(shard(x, m, "d", None), w1, w2)
    want = jnp.dot(jnp.tanh(jnp.dot(x, w1, precision="highest")), w2, precision="highest")
    close(out, want, tol=1e-3)
    assert out.sharding.spec == P("d", None)


def block_masked(x, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, 128):
        for j in range(0, N, 128):
            acc = jnp.zeros((128, 128))
            for k in range(0, K, 128):
                if keep[i // 128, k // 128] == 0:  # a runtime mask
                    continue
                acc += x[i : i + 128, k : k + 128] * 2.0
            out = out.at[i : i + 128, j : j + 128].set(acc)
    return out


def test_vmap_with_a_sharded_runtime_mask():
    kernel = autokernel.tpu(block_masked)
    m = mesh()
    xs = jnp.stack([rand((M, K), s) for s in range(4)])
    keeps = jnp.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [1, 0]], [[0, 1], [1, 1]]], jnp.int32)
    want = jnp.stack([kernel(x, keep) for x, keep in zip(xs, keeps, strict=True)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the batch splits: nothing gathered
        warnings.filterwarnings("ignore", message=".*interpreter")
        out = jax.jit(jax.vmap(kernel))(shard(xs, m, "d", None, None), shard(keeps, m, "d", None, None))
    close(out, want)
    assert out.sharding.spec == P("d", None, None)
    loss = jax.vmap(jax.grad(lambda x, keep: jnp.sum(kernel(x, keep))))
    grads = jax.jit(loss)(shard(xs, m, "d", None, None), shard(keeps, m, "d", None, None))
    for g, keep in zip(grads, keeps, strict=True):
        kept = np.repeat(np.repeat(np.asarray(keep), 128, 0), 128, 1).astype(np.float32)
        close(g, kept * 2.0 * (N // 128))
