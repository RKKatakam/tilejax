"""Runtime scalars: float scalars computed by JAX code, and traced indices out of range (clamped by the
kernel, reported under `jax.experimental.checkify`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental import checkify
from test_fusion import check_every_kernel, lower_for_v5e

import autokernel

M = N = 256
T = 128


def scaled(x, w, d):
    scale = 1 / jnp.sqrt(d)  # a traced 0-d float: a runtime scalar of both nests, read from SMEM
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * scale)
    y = jnp.zeros((M, N), x.dtype)
    for i in range(0, M, T):
        y = y.at[i : i + T, :].set(h[i : i + T, :] + w[i : i + T, :] * scale)
    return y


def test_a_float_computed_by_jax_is_a_runtime_scalar():
    prog = autokernel.tpu(scaled)
    x, w = rand((M, N), 1), rand((M, N), 2)
    for d in (4.0, 9.0):
        np.testing.assert_allclose(prog(x, w, jnp.float32(d)), (x + w) / np.sqrt(d), rtol=1e-5, atol=1e-6)
    lowered = check_every_kernel(prog, x, w, jnp.float32(4.0))
    assert lowered.names == ["scaled_h+scaled_y"]
    assert lowered["scaled_h"].scalars == ["scale"]
    assert "scale = fscalars[0]" in lowered.source  # an f32 (not weakly typed: `d` is an f32 array)
    assert sum(len(n.kernels) for n in prog.kernels) == 2  # one kernel per nest, whatever `d` is
    assert prog.verify(x, w, jnp.float32(4.0)) < 1e-5


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_float_scalars_mix_with_tiles_as_jax_says(dtype):
    def f(x, s):
        return (x * s).astype(x.dtype)

    @autokernel.tpu
    def prog(x, s):
        t = s * 1  # glue: a program, `t` a runtime scalar of the nest
        out = jnp.zeros((M, N), x.dtype)
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set((x[i : i + T, :] * t).astype(jnp.bfloat16))
        return out

    x = rand((M, N), 1, jnp.bfloat16)
    s = jnp.asarray(1.5, dtype)  # a bf16 tile times an f32 scalar is f32; times a bf16 one, bf16
    np.testing.assert_allclose(np.float32(prog(x, s)), np.float32(f(x, s)), rtol=1e-2)
    assert prog.verify(x, s) < 1e-2
    specs = prog.lower(x, s).inputs[0]
    assert "tpu_custom_call" in lower_for_v5e(prog.lower(x, s)[0][1], specs)


def test_a_float_read_by_an_allocation():
    @autokernel.tpu
    def f(x, s):
        t = s + 0
        out = jnp.full((M, N), t)  # the allocation reads `t`: the skipped tiles keep it
        for i in range(0, M, T):
            if i > 0:
                continue
            out = out.at[i : i + T, :].set(x[i : i + T, :])
        return out

    x = rand((M, N), 1)
    got = f(x, jnp.float32(2.0))
    np.testing.assert_allclose(got, jnp.where(jnp.arange(M)[:, None] < T, x, 2.0))


# --------------------------------------------------------------------------- traced indices out of range


def pick(x, w, scores):
    idx = jnp.argmax(scores) + 2  # traced: can't be checked before the kernel runs
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + w[idx, i : i + T, :])
    return out


def pick_two(x, w, scores):
    idx = jnp.argmax(scores) + 2
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] + w[idx, i : i + T, :])
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(h[i : i + T, :] * w[idx - 1, i : i + T, :])
    return out


def test_an_index_out_of_range_is_clamped_and_reported_under_checkify():
    prog = autokernel.tpu(pick)
    x, w = rand((M, N), 1), rand((3, M, N), 2)
    bad = jnp.zeros(4).at[3].set(1)  # idx = 5, outside w's 3 layers: the kernel reads the last one
    np.testing.assert_allclose(prog(x, w, bad), x + w[2], rtol=1e-6)
    err, out = checkify.checkify(prog)(x, w, bad)
    with pytest.raises(checkify.JaxRuntimeError, match=r"pick_out: `w\[idx, i:i \+ T, :\]`: index 5 \(`idx`\)"):
        err.throw()
    np.testing.assert_allclose(out, x + w[2], rtol=1e-6)
    err, _ = checkify.checkify(prog)(x, w, jnp.zeros(4).at[0].set(1))  # idx = 2: fine
    err.throw()
    assert "jnp.minimum(jnp.maximum(" in prog.source(x, w, bad)


def test_fused_kernels_clamp_and_report_too():
    prog = autokernel.tpu(pick_two)
    x, w = rand((M, N), 1), rand((3, M, N), 2)
    bad = jnp.zeros(4).at[3].set(1)  # idx = 5, idx - 1 = 4: both clamped to 2
    assert prog.lower(x, w, bad).names == ["pick_two_h+pick_two_out"]
    np.testing.assert_allclose(prog(x, w, bad), (x + w[2]) * w[2], rtol=1e-5)
    err, _ = jax.jit(checkify.checkify(prog))(x, w, bad)
    with pytest.raises(checkify.JaxRuntimeError, match="index 5"):
        err.throw()


def test_a_single_kernel_clamps_traced_indices_under_jit():
    @autokernel.tpu
    def layer(w, i):
        out = jnp.zeros((M, N))
        for r in range(0, M, T):
            out = out.at[r : r + T, :].set(w[i, r : r + T, :] * 2)
        return out

    w = rand((3, M, N), 1)
    np.testing.assert_allclose(jax.jit(layer)(w, jnp.int32(7)), w[2] * 2, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(layer)(w, jnp.int32(-9)), w[0] * 2, rtol=1e-6)  # -9 + 3 < 0: clamped to 0
    err, _ = checkify.checkify(jax.jit(layer))(w, jnp.int32(7))
    with pytest.raises(checkify.JaxRuntimeError, match="index 7"):
        err.throw()
