"""Cross-lower kernels to Mosaic for a TPU v5e, on any machine.

The TPU interpreter skips Pallas's Mosaic lowering, so these catch kernels that run in the interpreter
but can't be compiled for real hardware (unsupported ops, DMAs, semaphores, dynamic loop bounds).
"""

import jax
import jax.numpy as jnp
import pytest
from test_dense import matmul
from test_masks import make_causal, make_masked_matmul

import autokernel

V5E = jax.sharding.AbstractDevice(device_kind="TPU v5 lite", num_cores=1, platform="tpu")


def lower_for_v5e(kernel, *specs):
    entry = kernel.lower(*specs).entry
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=V5E)
    with jax.sharding.use_abstract_mesh(mesh):
        return jax.jit(lambda *xs: entry(*xs, interpret=False)).trace(*specs).lower(lowering_platforms=("tpu",))


def f32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def bf16(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.bfloat16)


CASES = {
    "dense": (lambda p: autokernel.tpu(matmul.fn, pipeline=p), (f32(256, 384), f32(384, 256))),
    "causal-f32": (lambda p: make_causal(512, 256, 512, 128, 128, 128, jnp.float32, p), (f32(512, 256), f32(256, 512))),
    "causal-bf16": (
        lambda p: make_causal(512, 256, 512, 128, 128, 128, jnp.bfloat16, p),
        (bf16(512, 256), bf16(256, 512)),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_lowers_to_mosaic(case, pipeline):
    make, specs = CASES[case]
    assert "tpu_custom_call" in lower_for_v5e(make(pipeline), *specs).as_text()


def test_per_tile_operand_lowers_to_mosaic(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def linear(x, w, bias):
        out = jnp.zeros((16, 256))
        for i in range(0, 16, 8):
            for j in range(0, 256, 128):
                acc = jnp.zeros((8, 128))
                for k in range(0, 256, 128):
                    acc += x[i : i + 8, k : k + 128] @ w[k : k + 128, j : j + 128]
                out = out.at[i : i + 8, j : j + 128].set(acc + bias[:, j : j + 128])
        return out

    assert "tpu_custom_call" in lower_for_v5e(linear, f32(16, 256), f32(256, 256), f32(1, 256)).as_text()


def _before_kernel(hlo):
    return hlo.split("tpu_custom_call")[0]


def test_background_fill_drops_the_output_fill(pipeline):
    kernel = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s), pipeline=pipeline)
    hlo = lower_for_v5e(kernel, f32(512, 256), f32(256, 512)).as_text()
    # No fill pass before the kernel: grid kernels alias an uninitialised buffer, manual ones write
    # their output directly.
    assert "broadcast_in_dim" not in _before_kernel(hlo)
    assert ("AllocateBuffer" in hlo) == (pipeline == "grid")


def test_prefill_keeps_the_output_fill(pipeline):
    kernel = make_masked_matmul(
        512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s), background_fill=False, pipeline=pipeline
    )
    hlo = lower_for_v5e(kernel, f32(512, 256), f32(256, 512)).as_text()
    assert "AllocateBuffer" not in hlo and "broadcast_in_dim" in _before_kernel(hlo)
