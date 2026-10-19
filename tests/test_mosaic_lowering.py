"""Cross-lower kernels to Mosaic for a TPU v5e, on any machine.

The TPU interpreter skips Pallas's Mosaic lowering, so these catch kernels that run in the interpreter
but can't be compiled for real hardware (unsupported ops, DMAs, semaphores, dynamic loop bounds).
"""

import jax
import jax.numpy as jnp
import pytest
from test_dense import matmul
from test_masks import make_causal, make_masked_matmul

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


@pytest.mark.parametrize(
    "kernel, specs",
    [
        (matmul, (f32(256, 384), f32(384, 256))),
        (make_causal(512, 256, 512, 128, 128, 128, jnp.float32), (f32(512, 256), f32(256, 512))),
        (make_causal(512, 256, 512, 128, 128, 128, jnp.bfloat16), (bf16(512, 256), bf16(256, 512))),
    ],
    ids=["dense", "causal-f32", "causal-bf16"],
)
def test_lowers_to_mosaic(kernel, specs):
    assert "tpu_custom_call" in lower_for_v5e(kernel, *specs).as_text()


def test_background_fill_drops_the_output_fill():
    kernel = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s))
    hlo = lower_for_v5e(kernel, f32(512, 256), f32(256, 512)).as_text()
    assert "AllocateBuffer" in hlo  # the output starts uninitialised: no fill pass before the kernel


def test_prefill_keeps_the_output_fill():
    kernel = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s), background_fill=False)
    hlo = lower_for_v5e(kernel, f32(512, 256), f32(256, 512)).as_text()
    assert "AllocateBuffer" not in hlo and "broadcast_in_dim" in hlo
