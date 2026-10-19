"""Differentiating through kernels and programs with `jax.grad`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError

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


def loss(f):
    return lambda *args: jnp.sum(f(*args) ** 2)


def assert_grads_close(got, want, tol=1e-4):
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=tol, atol=tol)


def inputs():
    return rand((M, K)), rand((K, N), seed=1) / 16


def test_grad_of_a_kernel_differentiates_its_loops():
    kernel = autokernel.tpu(matmul_tanh)
    x, y = inputs()
    got = jax.jit(jax.grad(loss(kernel), argnums=(0, 1)))(x, y)
    want = jax.grad(loss(plain_matmul_tanh), argnums=(0, 1))(x, y)
    assert_grads_close(got, want)


def test_backward_option_is_used():
    calls = []

    def backward(x, y, ct):
        calls.append(1)
        t = kernel(x, y)  # backward passes may call kernels too
        d = ct * (1 - t * t)
        return jnp.dot(d, y.T, precision="highest"), None  # no gradient for y

    kernel = autokernel.tpu(matmul_tanh, backward=backward)
    x, y = inputs()
    dx, dy = jax.grad(loss(kernel), argnums=(0, 1))(x, y)
    want_dx, _ = jax.grad(loss(plain_matmul_tanh), argnums=(0, 1))(x, y)
    assert calls
    assert_grads_close([dx], [want_dx])
    assert not np.any(np.asarray(dy))


def test_backward_with_the_wrong_number_of_gradients_fails():
    kernel = autokernel.tpu(matmul_tanh, backward=lambda x, y, ct: (ct,))
    with pytest.raises(ValueError, match="1 gradients for 2 inputs"):
        jax.grad(loss(kernel))(*inputs())


def two_layers(x, w1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc))
    h = h * 0.5  # glue code between kernels is differentiated by JAX
    out = jnp.zeros((M, N))
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


def test_grad_of_a_program_goes_through_every_kernel():
    prog = autokernel.tpu(two_layers)
    assert isinstance(prog, autokernel.TpuProgram)
    x, w1 = inputs()
    w2 = rand((N, N), seed=2) / 16
    got = jax.jit(jax.grad(loss(prog), argnums=(0, 1, 2)))(x, w1, w2)
    want = jax.grad(loss(plain_two_layers), argnums=(0, 1, 2))(x, w1, w2)
    assert_grads_close(got, want)


def test_backward_option_on_a_program():
    prog = autokernel.tpu(two_layers, backward=lambda x, w1, w2, ct: (None, None, jnp.ones_like(w2)))
    x, w1 = inputs()
    w2 = rand((N, N), seed=2) / 16
    dw2 = jax.grad(loss(prog), argnums=2)(x, w1, w2)
    np.testing.assert_array_equal(np.asarray(dw2), np.ones((N, N), np.float32))


def masked_rows(x, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if keep[i // T, 0] == 0:
                continue
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 3.0)
    return out


def test_integer_inputs_get_no_gradient():
    kernel = autokernel.tpu(masked_rows)
    x = rand((M, N))
    keep = jnp.array([[1], [0]], jnp.int32)
    dx = jax.grad(lambda x: jnp.sum(kernel(x, keep)))(x)
    want = np.concatenate([np.full((T, N), 3.0), np.zeros((T, N))]).astype(np.float32)
    np.testing.assert_allclose(np.asarray(dx), want)


def test_static_program_arguments_reject_backward():
    def scaled(x, w, scale):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * scale)
        return h @ w

    prog = autokernel.tpu(scaled, backward=lambda *a: a)
    with pytest.raises(AutoKernelError, match="needs every argument to be an array"):
        prog(rand((M, N)), rand((N, N)), 2.0)


def two_fused_layers(x, w1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc) * 0.5)
    out = jnp.zeros((M, N))  # no code between the nests: they run as one fused kernel
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, N, T):
                acc += h[i : i + T, k : k + T] @ w2[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def test_grad_goes_through_fused_kernels():
    prog = autokernel.tpu(two_fused_layers)
    x, w1 = inputs()
    w2 = rand((N, N), seed=2) / 16
    assert len(prog.lower(x, w1, w2)) == 1  # one fused kernel
    got = jax.jit(jax.grad(loss(prog), argnums=(0, 1, 2)))(x, w1, w2)
    want = jax.grad(loss(plain_two_layers), argnums=(0, 1, 2))(x, w1, w2)
    assert_grads_close(got, want)


def test_grad_under_jit_with_a_runtime_mask():
    """A mask reading an input is traced as an array (`jnp.where`) when differentiating under jit."""
    kernel = autokernel.tpu(masked_rows)
    x = rand((M, N))
    keep = jnp.array([[0], [1]], jnp.int32)
    dx = jax.jit(jax.grad(lambda x, keep: jnp.sum(kernel(x, keep))))(x, keep)
    want = np.concatenate([np.zeros((T, N)), np.full((T, N), 3.0)]).astype(np.float32)
    np.testing.assert_allclose(np.asarray(dx), want)
    np.testing.assert_allclose(np.asarray(jax.jit(kernel.reference)(x, keep)), np.asarray(kernel(x, keep)))
