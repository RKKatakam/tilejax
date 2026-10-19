"""Kernels and programs under JAX transformations: `jax.vmap` (one batched kernel), forward-mode
differentiation (`jax.jvp`, `jax.jacfwd`, `jax.linearize`) and their compositions with `jit` and `grad`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel

M, K, N = 256, 256, 256
T = 128


def matmul_tanh(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(jnp.tanh(acc))
    return out


def plain_matmul_tanh(x, y):
    return jnp.tanh(jnp.dot(x, y, precision="highest"))


def close(got, want, tol=1e-4):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(w, np.float32), rtol=tol, atol=tol)


def pallas_calls(fn, *args):
    """How many pallas_calls tracing `fn(*args)` runs, counting nested jaxprs (jit, custom rules, scans)."""

    def count(jaxpr):
        n = 0
        for eqn in jaxpr.eqns:
            n += eqn.primitive.name == "pallas_call"
            for p in eqn.params.values():
                for sub in p if isinstance(p, (tuple, list)) else [p]:
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        n += count(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        n += count(sub)
        return n

    return count(jax.make_jaxpr(fn)(*args).jaxpr)


def batch(shape, n=3, seed=0, scale=1.0):
    return jnp.stack([rand(shape, seed + b) * scale for b in range(n)])


# --------------------------------------------------------------------------- jax.vmap


def test_vmap_runs_one_batched_kernel():
    kernel = autokernel.tpu(matmul_tanh)
    xs, y = batch((M, K)), rand((K, N), 9) / 16
    f = jax.vmap(kernel, in_axes=(0, None))
    close(f(xs, y), jax.vmap(plain_matmul_tanh, in_axes=(0, None))(xs, y))
    assert pallas_calls(f, xs, y) == 1
    (batched,) = kernel._batched.values()
    lowered = batched.lower(xs, y)
    assert lowered.total_tiles == 3 * 4  # the batch is the outermost tile loop
    batched.verify(xs, y)  # the batched kernel against its own Python loops


@pytest.mark.parametrize("in_axes", [(None, 0), (1, 2), (0, 0)])
def test_vmap_over_any_inputs_and_axes(in_axes):
    kernel = autokernel.tpu(matmul_tanh)
    x, y = rand((M, K)), rand((K, N), 1) / 16
    xs = x if in_axes[0] is None else jnp.stack([x, -x], axis=in_axes[0])
    ys = y if in_axes[1] is None else jnp.stack([y, y * 0.5], axis=in_axes[1])
    got = jax.vmap(kernel, in_axes=in_axes)(xs, ys)
    close(got, jax.vmap(plain_matmul_tanh, in_axes=in_axes)(xs, ys))


def test_nested_vmap_and_jit():
    kernel = autokernel.tpu(matmul_tanh)
    xs = jnp.stack([batch((M, K), 2), batch((M, K), 2, seed=5)])  # (2, 2, M, K)
    y = rand((K, N), 9) / 16
    want = jax.vmap(jax.vmap(plain_matmul_tanh, in_axes=(0, None)), in_axes=(0, None))(xs, y)
    nested = jax.vmap(jax.vmap(kernel, in_axes=(0, None)), in_axes=(0, None))
    close(nested(xs, y), want)
    assert pallas_calls(nested, xs, y) == 1
    close(jax.jit(nested)(xs, y), want)  # jit of vmap
    close(jax.vmap(jax.vmap(jax.jit(kernel), in_axes=(0, None)), in_axes=(0, None))(xs, y), want)  # vmap of jit


def test_vmap_of_grad_and_grad_of_vmap():
    kernel = autokernel.tpu(matmul_tanh)
    xs, y = batch((M, K)), rand((K, N), 9) / 16

    def loss(f):
        return lambda x, y: jnp.sum(f(x, y) ** 2)

    got = jax.vmap(jax.grad(loss(kernel), argnums=(0, 1)), in_axes=(0, None))(xs, y)
    want = jax.vmap(jax.grad(loss(plain_matmul_tanh), argnums=(0, 1)), in_axes=(0, None))(xs, y)
    close(got, want)

    def batch_loss(f):
        return lambda x, y: jnp.sum(jax.vmap(f, in_axes=(0, None))(x, y) ** 2)

    got = jax.jit(jax.grad(batch_loss(kernel), argnums=(0, 1)))(xs, y)
    close(got, jax.grad(batch_loss(plain_matmul_tanh), argnums=(0, 1))(xs, y), tol=5e-4)


def causal(x, y):
    out = jnp.full((M, N), -1.0)
    for i in range(0, M, T):
        for j in range(0, N, T):
            if j > i:
                continue
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def test_vmap_with_a_static_tile_mask():
    kernel = autokernel.tpu(causal)
    xs, ys = batch((M, K)), batch((K, N), seed=7)
    got = jax.vmap(kernel)(xs, ys)
    np.testing.assert_allclose(np.asarray(got), np.asarray(jax.vmap(kernel.reference)(xs, ys)), rtol=1e-4, atol=1e-4)
    (batched,) = kernel._batched.values()
    lowered = batched.lower(xs, ys)
    assert (lowered.tiles, lowered.total_tiles) == (3 * 3, 3 * 4)  # three batches of the same three tiles


def block_masked(x, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                if keep[i // T, k // T] == 0:  # a runtime mask: read from an input
                    continue
                acc += x[i : i + T, k : k + T] * 2.0
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def test_vmap_with_a_batched_runtime_mask():
    kernel = autokernel.tpu(block_masked)
    xs = batch((M, K))
    keeps = jnp.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [1, 0]]], jnp.int32)
    got = jax.vmap(kernel)(xs, keeps)
    want = jnp.stack([kernel(x, keep) for x, keep in zip(xs, keeps, strict=True)])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    assert pallas_calls(jax.vmap(kernel), xs, keeps) == 1
    # Only the mask batched: x is read unbatched by every batch element.
    got = jax.vmap(kernel, in_axes=(None, 0))(xs[0], keeps)
    want = jnp.stack([kernel(xs[0], keep) for keep in keeps])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_vmap_of_grad_with_a_batched_runtime_mask():
    kernel = autokernel.tpu(block_masked)
    xs = batch((M, K))
    keeps = jnp.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [1, 0]]], jnp.int32)
    got = jax.vmap(jax.grad(lambda x, keep: jnp.sum(kernel(x, keep))))(xs, keeps)
    for g, keep in zip(got, keeps, strict=True):
        kept = np.repeat(np.repeat(np.asarray(keep), T, 0), T, 1).astype(np.float32)
        np.testing.assert_allclose(np.asarray(g), kept * 2.0 * (N // T), rtol=1e-6)  # each x block feeds N // T tiles


def shapes_from_inputs(x, scale):
    rows, cols = x.shape  # the prologue reads the inputs' shapes: per batch element under vmap
    out = jnp.zeros((rows, cols), x.dtype)
    for i in range(0, rows, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * scale[0:1, :])
    return out


def test_vmap_keeps_shapes_per_element():
    kernel = autokernel.tpu(shapes_from_inputs)
    xs, scale = batch((M, N)), rand((1, N), 3)
    got = jax.vmap(kernel, in_axes=(0, None))(xs, scale)
    np.testing.assert_allclose(np.asarray(got), np.asarray(xs * scale[0:1]), rtol=1e-6)
    scales = batch((1, N), seed=4)
    got = jax.vmap(kernel)(xs, scales)
    np.testing.assert_allclose(np.asarray(got), np.asarray(xs * scales[:, 0:1]), rtol=1e-6)


def indexed(w, layer):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(w[layer, i : i + T, :] + 1.0)
    return out


def test_vmap_over_a_runtime_scalar_runs_one_kernel_per_element():
    kernel = autokernel.tpu(indexed)
    w = batch((M, N), n=4)
    layers = jnp.array([3, 0, 2], jnp.int32)
    with pytest.warns(UserWarning, match="runtime scalar `layer` is batched"):
        got = jax.vmap(kernel, in_axes=(None, 0))(w, layers)
    np.testing.assert_allclose(np.asarray(got), np.asarray(w[layers] + 1.0))
    ws = batch((4, M, N), n=2)  # batching only the array keeps one kernel
    got = jax.vmap(kernel, in_axes=(0, None))(ws, 1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ws[:, 1] + 1.0))


# --------------------------------------------------------------------------- forward mode


def test_jvp_jacfwd_and_linearize_match_plain_jax():
    kernel = autokernel.tpu(matmul_tanh)
    x, y = rand((M, K)), rand((K, N), 1) / 16
    tx, ty = rand((M, K), 2), rand((K, N), 3)
    out, tangent = jax.jvp(kernel, (x, y), (tx, ty))
    want_out, want_tangent = jax.jvp(plain_matmul_tanh, (x, y), (tx, ty))
    close((out, tangent), (want_out, want_tangent))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(kernel(x, y)))  # the primal is the kernel's
    assert pallas_calls(lambda x, y: jax.jvp(kernel, (x, y), (tx, ty)), x, y) == 1

    _, lin = jax.linearize(kernel, x, y)
    close(lin(tx, ty), want_tangent)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda x, y: jax.jvp(kernel, (x, y), (tx, ty))[0])(x, y)), out)

    small = autokernel.tpu(shapes_from_inputs)
    xs, scale = rand((M, N)), rand((1, N), 3)
    f = lambda s: jnp.sum(jnp.sin(small(xs, s)), axis=0)  # noqa: E731
    g = lambda s: jnp.sum(jnp.sin(xs * s), axis=0)  # noqa: E731
    close(jax.jacfwd(f)(scale), jax.jacfwd(g)(scale))
    close(jax.jacrev(f)(scale), jax.jacrev(g)(scale))
    h = lambda s: jnp.sum(jnp.sin(small(xs, s)))  # noqa: E731
    close(jax.hessian(h)(scale), jax.hessian(lambda s: jnp.sum(jnp.sin(xs * s)))(scale), tol=1e-3)


def test_grad_runs_the_kernel_once():
    kernel = autokernel.tpu(matmul_tanh)
    x, y = rand((M, K)), rand((K, N), 1) / 16
    grad = jax.grad(lambda x, y: jnp.sum(kernel(x, y) ** 2), argnums=(0, 1))
    assert pallas_calls(grad, x, y) == 1
    value_and_grad = jax.value_and_grad(lambda x, y: jnp.sum(kernel(x, y) ** 2))
    assert pallas_calls(value_and_grad, x, y) == 1


def test_jvp_of_vmap_and_vmap_of_jvp():
    kernel = autokernel.tpu(matmul_tanh)
    xs, y = batch((M, K)), rand((K, N), 9) / 16
    txs = batch((M, K), seed=20)
    f = jax.vmap(kernel, in_axes=(0, None))
    g = jax.vmap(plain_matmul_tanh, in_axes=(0, None))
    close(jax.jvp(lambda x: f(x, y), (xs,), (txs,)), jax.jvp(lambda x: g(x, y), (xs,), (txs,)))
    vj = jax.vmap(lambda x, t: jax.jvp(lambda x: kernel(x, y), (x,), (t,)))
    pj = jax.vmap(lambda x, t: jax.jvp(lambda x: plain_matmul_tanh(x, y), (x,), (t,)))
    close(vj(xs, txs), pj(xs, txs))


def test_backward_option_keeps_reverse_mode_only():
    kernel = autokernel.tpu(matmul_tanh, backward=lambda x, y, ct: (ct, None))
    x, y = rand((M, K)), rand((K, N), 1) / 16
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: jnp.sum(kernel(x, y)))(x)), np.ones((M, K)))
    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jvp(lambda x: kernel(x, y), (x,), (x,))
    xs = batch((M, K))  # vmap still runs one batched kernel, and batches `backward`
    close(jax.vmap(kernel, in_axes=(0, None))(xs, y), jax.vmap(plain_matmul_tanh, in_axes=(0, None))(xs, y))
    grads = jax.vmap(jax.grad(lambda x: jnp.sum(kernel(x, y))))(xs)
    np.testing.assert_array_equal(np.asarray(grads), np.ones((3, M, K)))


# --------------------------------------------------------------------------- programs


def two_layers(x, w1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc))
    h = h * 0.5  # glue code: plain JAX
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, N, T):
                acc += h[i : i + T, k : k + T] @ w2[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def two_fused_layers(x, w1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc) * 0.5)
    out = jnp.zeros((M, N))  # no code between the nests: one fused kernel
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, N, T):
                acc += h[i : i + T, k : k + T] @ w2[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def plain_two_layers(x, w1, w2):
    h = jnp.tanh(jnp.dot(x, w1, precision="highest")) * 0.5
    return jnp.dot(h, w2, precision="highest")


@pytest.mark.parametrize("fn", [two_layers, two_fused_layers])
def test_programs_under_vmap_and_jvp(fn):
    prog = autokernel.tpu(fn)
    xs, w1, w2 = batch((M, K)), rand((K, N), 1) / 16, rand((N, N), 2) / 16
    f = jax.vmap(prog, in_axes=(0, None, None))
    close(f(xs, w1, w2), jax.vmap(plain_two_layers, in_axes=(0, None, None))(xs, w1, w2))
    assert pallas_calls(f, xs, w1, w2) == 2  # each nest one batched kernel (a fused pair runs unfused)
    tangents = (xs[0], w1, w2)
    got = jax.jvp(prog, (xs[0], w1, w2), tangents)
    close(got, jax.jvp(plain_two_layers, (xs[0], w1, w2), tangents))
    grads = jax.vmap(jax.grad(lambda x: jnp.sum(prog(x, w1, w2) ** 2)))(xs)
    close(grads, jax.vmap(jax.grad(lambda x: jnp.sum(plain_two_layers(x, w1, w2) ** 2)))(xs))


L = 2


def deep(x, w):
    for layer in range(L):  # a jax.lax.fori_loop: `layer` is a runtime scalar of the kernel
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc = jnp.zeros((T, T))
                for k in range(0, K, T):
                    acc += x[i : i + T, k : k + T] @ w[layer, k : k + T, j : j + T]
                h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc))
        x = x + h
    return x


def plain_deep(x, w):
    for layer in range(L):
        x = x + jnp.tanh(jnp.dot(x, w[layer], precision="highest"))
    return x


def test_vmap_and_jvp_of_a_program_with_a_layer_loop():
    prog = autokernel.tpu(deep)
    xs, w = batch((M, K)), rand((L, K, N), 2) / 16
    f = jax.vmap(prog, in_axes=(0, None))
    close(f(xs, w), jax.vmap(plain_deep, in_axes=(0, None))(xs, w))
    assert pallas_calls(f, xs, w) == 1  # one batched kernel, in the loop body
    assert prog.unrolled == []
    tangents = (xs[1], w)
    close(jax.jvp(prog, (xs[0], w), tangents), jax.jvp(plain_deep, (xs[0], w), tangents))


@pytest.mark.parametrize("chip", ["v5e", "v5p"])
def test_batched_kernels_lower_to_mosaic(chip):
    from test_mosaic_lowering import lower_for

    kernel = autokernel.tpu(block_masked)
    xs, keeps = batch((M, K)), jnp.ones((3, M // T, K // T), jnp.int32)
    jax.vmap(kernel)(xs, keeps)
    (batched,) = kernel._batched.values()
    lower_for(chip, batched, jax.ShapeDtypeStruct(xs.shape, xs.dtype), jax.ShapeDtypeStruct(keeps.shape, keeps.dtype))
    kernel = autokernel.tpu(causal)
    jax.vmap(kernel, in_axes=(0, None))(xs, xs[0])
    (batched,) = kernel._batched.values()
    lower_for(chip, batched, jax.ShapeDtypeStruct(xs.shape, xs.dtype), jax.ShapeDtypeStruct(xs.shape[1:], xs.dtype))
