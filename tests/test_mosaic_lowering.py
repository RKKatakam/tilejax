"""Cross-lower kernels to Mosaic for real TPUs, on any machine.

The TPU interpreter skips Pallas's Mosaic lowering, so these catch kernels that run in the interpreter
but can't be compiled for hardware (unsupported ops, DMAs, semaphores, dynamic loop bounds, layouts).
"""

import jax
import jax.numpy as jnp
import pytest
import test_generality as gen
from test_attention import make_attention
from test_dense import matmul
from test_masks import make_causal, make_masked_matmul

import autokernel

DEVICES = {
    "v5e": jax.sharding.AbstractDevice(device_kind="TPU v5 lite", num_cores=1, platform="tpu"),
    "v5p": jax.sharding.AbstractDevice(device_kind="TPU v5", num_cores=2, platform="tpu"),
}


def lower_for(chip, kernel, *specs):
    entry = kernel.lower(*specs).entry
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=DEVICES[chip])
    with jax.sharding.use_abstract_mesh(mesh):
        return jax.jit(lambda *xs: entry(*xs, interpret=False)).trace(*specs).lower(lowering_platforms=("tpu",))


def f32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def bf16(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.bfloat16)


def i32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.int32)


CASES = {
    "dense": (matmul, (f32(256, 384), f32(384, 256))),
    "dense-3-buffers": (autokernel.tpu(matmul.fn, buffers=3), (f32(256, 384), f32(384, 256))),
    "causal-f32": (make_causal(512, 256, 512, 128, 128, 128, jnp.float32), (f32(512, 256), f32(256, 512))),
    "causal-bf16": (make_causal(512, 256, 512, 128, 128, 128, jnp.bfloat16), (bf16(512, 256), bf16(256, 512))),
    "prefill": (
        make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s), background_fill=False),
        (f32(512, 256), f32(256, 512)),
    ),
    "splash-attention": (make_attention(2, 2, 512, 128, 128, 128), (bf16(2, 2, 512, 128),) * 3),
    "sliding-window": (make_attention(1, 2, 1024, 128, 128, 128, window=256), (bf16(1, 2, 1024, 128),) * 3),
    "hoisted": (gen.scale_per_batch, (f32(4, 256, 256),)),
    "two-outputs": (gen.matmul_and_relu, (f32(256, 256), f32(256, 256))),
    "two-reductions": (gen.rmsnorm_matmul, (f32(gen.RM, gen.RK), f32(gen.RK, gen.RN))),
    "nested-reduction": (gen.sum_of_matmuls, (f32(3, 256, 256), f32(3, 256, 256))),
    "batched-matmul": (gen.bmm, (f32(4, 256, 256), f32(4, 256, 256))),
    "smem-scalar-state": (gen.scaled_by_block_count, (f32(256, 512),)),
    "runtime-reduction-mask": (gen.make_block_sparse(256, 512, 256, 128), (f32(256, 512), f32(512, 256), i32(2, 4))),
    "runtime-tile-mask": (gen.make_masked_copy(), (f32(512, 256), jax.ShapeDtypeStruct((4, 2), jnp.bool_))),
}


@pytest.mark.parametrize("case", CASES)
def test_lowers_for_v5e(case):
    kernel, specs = CASES[case]
    assert "tpu_custom_call" in lower_for("v5e", kernel, *specs).as_text()


@pytest.mark.parametrize("case", ["dense", "causal-f32", "splash-attention", "runtime-reduction-mask"])
def test_lowers_split_across_two_tensorcores(case):
    kernel, specs = CASES[case]
    split = autokernel.tpu(kernel.fn, target="v5p")
    assert split.lower(*specs).cores == 2
    assert "tpu_custom_call" in lower_for("v5p", split, *specs).as_text()


def _before_kernel(hlo):
    return hlo.split("tpu_custom_call")[0]


def test_background_fill_drops_the_output_fill():
    kernel = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.zeros(s))
    hlo = lower_for("v5e", kernel, f32(512, 256), f32(256, 512)).as_text()
    assert "broadcast_in_dim" not in _before_kernel(hlo)  # the kernel writes every tile itself


def test_prefill_keeps_the_output_fill():
    kernel, specs = CASES["prefill"]
    assert "broadcast_in_dim" in _before_kernel(lower_for("v5e", kernel, *specs).as_text())
