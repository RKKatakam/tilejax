"""The public API: building, running and verifying kernels; tolerances; buffering; small units."""

import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import make_kernel, rand, spec
from matmul import MatmulTiles, matmul

import autokernel
from autokernel import KernelError, Pipeline, mxu, v5e
from autokernel.api import compare
from autokernel.pipeline import default_pipeline

TILES = MatmulTiles(tm=128, tn=128, tk=128)


def test_lowered_describes_the_kernel():
    lowered = matmul.lower(spec(256, 384), spec(384, 256), TILES)
    assert (lowered.steps, lowered.tiles, lowered.total_tiles) == (12, 4, 4)
    assert lowered.pipeline == Pipeline(3, 2) and str(lowered.pipeline) == "3 input / 2 output buffers"
    assert lowered.config.tunable == TILES and str(lowered.config).startswith("MatmulTiles(tm=128")
    # x and y: 3 slots of 128x128 f32 each; out: 2 slots; acc: one.
    assert lowered.vmem_bytes == (3 + 3 + 2 + 1) * 128 * 128 * 4
    assert matmul.lower(spec(256, 384), spec(384, 256), TILES) is lowered  # built once per signature
    assert repr(matmul) == "<autokernel.Kernel matmul>" and matmul.__name__ == "matmul"
    assert matmul.source(spec(256, 384), spec(384, 256), TILES) == lowered.source
    assert "MatmulTiles(tm=128, tn=128, tk=128); 3 input / 2 output buffers" in lowered.source


def test_inputs_are_arrays_of_the_right_count():
    with pytest.raises(KernelError, match=r"takes 2 input arrays \(x, y\), then optionally `tiles`; got 1"):
        matmul(rand((256, 256)))
    with pytest.raises(KernelError, match="input `y` must be an array, got float"):
        matmul(rand((256, 256)), 1.0)
    with pytest.raises(KernelError, match="input `y` must be an array"):
        matmul(rand((256, 256)), jnp.float32(1))
    with pytest.raises(KernelError, match="`tiles` must be a MatmulTiles, not tuple"):
        matmul(rand((256, 256)), rand((256, 256)), (128, 128, 128))
    with pytest.raises(KernelError, match=r"takes 1 input arrays \(x\); got 2"):
        make_kernel(SCALE_SOURCE, SCALE=2.0)(rand((8, 128)), rand((8, 128)))


SCALE_SOURCE = """
def f(x):
    out = jnp.zeros((8, 128))
    for i in range(0, 8, 8):
        out = out.at[i : i + 8, :].set(x[i : i + 8, :] * SCALE)
    return out
"""


def test_constants_are_read_when_a_signature_is_first_built():
    kernel = make_kernel(SCALE_SOURCE, SCALE=2.0)
    x = rand((8, 128))
    first = kernel(x)
    kernel.fn.__globals__["SCALE"] = 3.0
    np.testing.assert_array_equal(np.asarray(kernel(x)), np.asarray(first))  # as jax.jit does
    np.testing.assert_array_equal(np.asarray(kernel.reference(x)), np.asarray(first))
    np.testing.assert_allclose(np.asarray(kernel(rand((16, 128)))[:8]), 3 * np.asarray(rand((16, 128))[:8]))


def test_closures_are_read_too():
    def make(scale):
        @autokernel.kernel
        def scaled(x):
            out = jnp.zeros(x.shape)
            for i in range(0, x.shape[0], 8):
                out = out.at[i : i + 8, :].set(x[i : i + 8, :] * scale)
            return out

        return scaled

    x = rand((16, 128))
    np.testing.assert_allclose(np.asarray(make(4.0)(x)), 4 * np.asarray(x))
    assert make(4.0).verify(x) == 0.0


def test_verify_returns_the_largest_difference():
    kernel = matmul
    x, y = rand((256, 256)), rand((256, 256), 1)
    assert kernel.verify(x, y, TILES, rtol=0.0, atol=0.0) == 0.0  # the interpreter computes exactly what JAX does


def test_compare_tolerances():
    a = jnp.array([1.0, 2.0, jnp.nan, jnp.inf])
    assert compare("f32", a, a) == 0.0
    assert compare("f32", a + jnp.array([1e-5, 0, 0, 0]), a) == pytest.approx(1e-5, rel=1e-2)
    with pytest.raises(AssertionError, match="1 of 4 elements"):
        compare("f32", a + jnp.array([1.0, 0, 0, 0]), a)
    b = jnp.array([100.0, 1.0], jnp.bfloat16)
    assert compare("bf16", b + jnp.array([0, 1.0], jnp.bfloat16), b) == 1.0  # 2e-2 of the largest value
    i = jnp.array([1, 2], jnp.int32)
    with pytest.raises(AssertionError):
        compare("i32", i + 1, i)
    assert compare("i32", i, i) == 0.0
    nan = jnp.array([jnp.nan], jnp.bfloat16)
    assert compare("all nan", nan, nan) == 0.0


def test_buffering_falls_back_to_two_slots_then_fails():
    def big(tile):
        kernel = make_kernel(
            f"""
            def f(x):
                out = jnp.zeros((2 * {tile}, 128))
                for i in range(0, 2 * {tile}, {tile}):
                    acc = jnp.zeros(({tile}, 128))
                    for k in range(0, 256, 128):
                        acc += x[i : i + {tile}, k : k + 128]
                    out = out.at[i : i + {tile}, :].set(acc)
                return out
            """
        )
        return default_pipeline(*kernel.build(kernel.signature(spec(2 * tile, 256)), None))

    assert big(32768) == Pipeline(3, 2)  # six 16 MiB buffers: exactly the 96 MiB budget
    assert big(36864) == Pipeline(2, 2)  # 18 MiB buffers: five fit, six don't
    with pytest.raises(KernelError, match="MiB of VMEM double-buffered"):
        big(49152)


def test_v5e_layout():
    assert v5e.sublanes(jnp.float32) == 8 and v5e.sublanes(jnp.bfloat16) == 16 and v5e.sublanes(jnp.int8) == 32
    assert v5e.padded_bytes((100,), jnp.float32) == 128 * 4
    assert v5e.padded_bytes((3, 100), jnp.bfloat16) == 16 * 128 * 2
    assert v5e.padded_bytes((), jnp.float32) == 128 * 4
    assert v5e.layout_problems((100,), (200,), jnp.float32) == [
        "its last dim, 100, must be a multiple of 128 or all 200 elements"
    ]
    assert v5e.layout_problems((7, 128), (7, 256), jnp.float32) == []
    assert not v5e.on_tpu() or jax.devices()[0].device_kind == v5e.DEVICE_KIND


def test_errors_quote_the_line_when_they_can():
    node = ast.Pass(lineno=1)
    assert str(KernelError("boom", filename=__file__, node=node)).startswith(f"{__file__}:1: boom\n    ")
    assert str(KernelError("boom", filename="<nowhere>", node=node)) == "<nowhere>:1: boom"
    assert str(KernelError("boom")) == "boom"


def test_mxu_semantics():
    x, y = rand((8, 128), dtype=jnp.bfloat16), rand((16, 128), 1, dtype=jnp.bfloat16)
    assert mxu.FUNCTIONS[mxu.DOT](x, y.T).dtype == jnp.float32
    np.testing.assert_allclose(mxu.FUNCTIONS[mxu.DOT_NT](x, y), mxu.FUNCTIONS[mxu.DOT](x, y.T), rtol=1e-6)
    ints = jnp.ones((8, 128), jnp.int8)
    assert mxu.FUNCTIONS[mxu.DOT](ints, ints.T).dtype == jnp.int32
    mixed = mxu.FUNCTIONS[mxu.DOT](x, rand((128, 4)))
    assert mixed.dtype == jnp.float32
    batched = mxu.FUNCTIONS[mxu.DOT_NT](rand((2, 8, 128)), rand((4, 128, 2)))  # x @ y.T, y.T: (2, 128, 4)
    assert batched.shape == (2, 8, 4)


def test_matmul_assignment():
    kernel = make_kernel(
        """
        def f(x, w):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, 128):
                acc = x[i : i + 128, :]
                for k in range(0, 2):
                    acc @= w[k, :, :]
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    x, w = rand((128, 128)), rand((2, 128, 128), 1) / 8
    want = np.asarray(x, np.float64) @ np.asarray(w[0], np.float64) @ np.asarray(w[1], np.float64)
    np.testing.assert_allclose(np.asarray(kernel(x, w)), want, rtol=1e-4, atol=1e-4)
    assert "acc_ref[...] = _mxu_dot(acc_ref[...], w_ref[...])" in kernel.source(x, w)


def test_pipelines_are_checked():
    with pytest.raises(KernelError, match="1 to 4 input buffers"):
        Pipeline(5, 2)
    with pytest.raises(KernelError, match="1 to 3 output buffers"):
        Pipeline(2, 0)


def test_prologue_comprehensions_see_globals():
    kernel = make_kernel(
        """
        def f(x):
            R = min(d // T for d in x.shape)
            out = jnp.zeros((R * T, 128))
            for i in range(0, R * T, T):
                out = out.at[i : i + T, :].set(x[i : i + T, :])
            return out
        """,
        T=8,
    )
    assert kernel.lower(spec(16, 128)).tiles == 2
