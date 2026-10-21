"""examples/kda_decode.py: KDA's decode step as one autokernel kernel, against vLLM TorchTPU's fused
decode_kda kernel (vendored in benchmarks/vs_vllm_kda, run in the Pallas TPU interpreter) and a float64
NumPy reference: paged pools updated in place, the null slot, sequences on their first step."""

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, run_checking_races
from jax.experimental.pallas import tpu as pltpu
from kda_decode import (
    HEAD_DIM,
    HEADS,
    KdaTiles,
    args_of,
    kda_decode,
    kda_decode_step,
    make_inputs,
    reference,
    step_args,
    step_kernel,
    xla_decode,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks" / "vs_vllm_kda"))
from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda


def vllm(inputs, monkeypatch, **options):
    """vLLM's kernel on these inputs, in the interpreter (it sizes its VMEM limit from the TPU it runs on)."""
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: types.SimpleNamespace(vmem_capacity_bytes=128 << 20))
    with pltpu.force_tpu_interpret_mode():
        return jax.block_until_ready(decode_kda(*args_of(inputs), **options))


def inputs_with_padding(batch=4, heads=HEADS, seed=0):
    """A batch with a padded entry (slot 0) and a sequence on its first step."""
    inputs = make_inputs(batch, seed=seed, heads=heads)
    inputs["state_indices"] = inputs["state_indices"].at[1].set(0)
    inputs["has_initial_state"] = inputs["has_initial_state"].at[2].set(0).at[3].set(1)
    return inputs


def check_against_reference(inputs, result, **options):
    idx = np.asarray(inputs["state_indices"])
    out, conv, state = (f64(r) for r in result)
    want_out, want_conv, want_state = reference(inputs, **options)
    np.testing.assert_allclose(out, want_out, rtol=8e-3, atol=1e-3 * np.abs(want_out).max())  # bf16: one ulp
    np.testing.assert_array_equal(conv[idx], want_conv)  # the conv states are copies: exact
    np.testing.assert_allclose(state[idx], want_state, rtol=0, atol=1e-6)
    # Slots no sequence owns (and the null slot) are untouched.
    others = np.setdiff1d(np.arange(state.shape[0]), idx[idx > 0])
    np.testing.assert_array_equal(state[others], f64(inputs["state"])[others])
    np.testing.assert_array_equal(conv[others], f64(inputs["conv_state"])[others])


@pytest.mark.cpu  # (tolerances of the interpreter's exact f32: a v5e's MXU rounds f32 operands to bf16)
@pytest.mark.parametrize(
    "options", [{}, {"state_transposed": True}, {"lower_bound": -5.0}, {"lower_bound": -5.0, "state_transposed": True}]
)
def test_kda_decode_matches_float64_and_vllms_kernel(monkeypatch, options):
    inputs = inputs_with_padding()
    ours = jax.block_until_ready(kda_decode(*args_of(inputs), **options))
    check_against_reference(inputs, ours, **options)
    theirs = vllm(inputs, monkeypatch, **options)
    check_against_reference(inputs, theirs, **options)
    assert float(jnp.abs(ours[0][1]).max()) == 0.0  # the padded entry's output: zeros
    for mine, vllms in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(f64(mine), f64(vllms), rtol=8e-3 if mine.dtype == jnp.bfloat16 else 1e-6, atol=1e-7)
    # With vLLM's tiles (all heads at once) the same arithmetic in the same order gives the same bits. (On
    # a v5e, at any tiles. The interpreter's f32 matmuls of transposed states may sum in another order.)
    whole = step_kernel(**options)(*step_args(*args_of(inputs)), KdaTiles(heads=HEADS))
    for mine, vllms in zip(whole, theirs, strict=True):
        if options.get("state_transposed"):
            np.testing.assert_allclose(f64(mine), f64(vllms), rtol=8e-3 if mine.dtype == jnp.bfloat16 else 1e-6)
        else:
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(vllms))


@pytest.mark.cpu
def test_kda_decode_in_the_interpreter_checking_races():
    inputs = inputs_with_padding(batch=3)
    args = step_args(*args_of(inputs))
    for heads in (16, 32):
        result = run_checking_races(kda_decode_step, *args, KdaTiles(heads=heads))
        check_against_reference(inputs, result)
    kda_decode_step.verify(*args)


@pytest.mark.cpu
def test_kda_decode_as_a_donated_decode_loop():
    inputs = inputs_with_padding(batch=2, heads=16)
    step = jax.jit(kda_decode, donate_argnums=(3, 5))
    args = list(args_of(inputs))
    args[3], args[5] = jnp.copy(args[3]), jnp.copy(args[5])
    out, conv, state = step(*args)
    check_against_reference(inputs, (out, conv, state))
    # The next step continues from the states just written: as the plain-JAX step does.
    args[3], args[5] = conv, state
    want = xla_decode(*args)
    out2, conv2, state2 = step(*args)
    np.testing.assert_allclose(f64(out2), f64(want[0]), rtol=8e-3, atol=1e-3 * float(jnp.abs(want[0]).max()))
    np.testing.assert_array_equal(f64(conv2), f64(want[1]))
    np.testing.assert_allclose(f64(state2), f64(want[2]), rtol=0, atol=1e-5)


@pytest.mark.parametrize("batch", [1, 64, 256])
def test_kda_decode_lowers_for_a_v5e_at_kimi_linears_shapes(batch):
    slots = 2 * batch + 1
    d = HEADS * HEAD_DIM
    sds = jax.ShapeDtypeStruct
    bf16, f32, i32 = jnp.bfloat16, jnp.float32, jnp.int32
    specs = [
        sds((batch, 3, HEADS, HEAD_DIM), bf16),
        sds((batch, HEADS, HEAD_DIM), bf16),
        sds((batch, HEADS, 1), bf16),
        sds((slots, 3, 3, HEADS, HEAD_DIM), bf16),
        sds((4, 3, HEADS, HEAD_DIM), bf16),
        sds((slots, HEADS, HEAD_DIM, HEAD_DIM), f32),
        sds((HEADS, 1), f32),
        sds((HEADS, HEAD_DIM), f32),
        sds((batch,), i32),
        sds((batch,), i32),
    ]
    assert d == 4096
    for heads in (16, 32):
        lowered = kda_decode_step.lower(*specs, KdaTiles(heads=heads))
        assert lowered.steps == batch * HEADS // heads
        assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.tpu
@pytest.mark.parametrize("options", [{}, {"state_transposed": True}, {"lower_bound": -5.0}])
@pytest.mark.parametrize("heads", [16, 32])
def test_kda_decode_is_bit_for_bit_vllms_on_a_v5e(options, heads):
    inputs = inputs_with_padding(batch=8)
    ours = step_kernel(**options)(*step_args(*args_of(inputs)), KdaTiles(heads=heads))
    theirs = decode_kda(*args_of(inputs), **options)
    for mine, vllms in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(vllms))
