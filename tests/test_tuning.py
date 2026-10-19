"""Hardware targets, the cost model, and block-size tuning."""

import jax
import jax.numpy as jnp
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError
from autokernel.hardware import target

S = 2048
TM = TN = TK = 128  # overridden by tuning


def big_matmul(a, b):
    c = jnp.zeros((S, S), jnp.bfloat16)
    for i in range(0, S, TM):
        for j in range(0, S, TN):
            acc = jnp.zeros((TM, TN), jnp.float32)
            for k in range(0, S, TK):
                acc += a[i : i + TM, k : k + TK] @ b[k : k + TK, j : j + TN]
            c = c.at[i : i + TM, j : j + TN].set(acc.astype(jnp.bfloat16))
    return c


def bf16(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.bfloat16)


def test_hardware_targets_come_from_jax():
    v5e, v5p = target("v5e"), target("v5p")
    assert (v5e.cores, v5e.vmem_bytes, v5e.lanes) == (1, 128 * 2**20, 128)
    assert v5e.hbm_bytes_per_s == 820e9 and v5e.mxu_flops_per_s == 197e12
    assert v5p.cores == 2  # megacore: work is split between two TensorCores
    with pytest.raises(AutoKernelError, match="unknown TPU target"):
        autokernel.tpu(big_matmul, target="v99")


def test_cost_estimate_counts_the_matmul():
    lw = autokernel.tpu(big_matmul).lower(bf16(S, S), bf16(S, S))
    assert lw.estimate.flops == pytest.approx(2 * S**3, rel=0.01)
    assert lw.estimate.hbm_bytes >= 2 * S * S * 2  # at least reads both inputs once
    assert "cost_estimate=_COST" in lw.source


def test_vmem_limit_follows_the_estimate():
    lw = autokernel.tpu(big_matmul).lower(bf16(S, S), bf16(S, S))
    assert lw.vmem_limit_bytes >= lw.estimate.vmem_bytes
    assert f"vmem_limit_bytes={lw.vmem_limit_bytes}" in lw.source


def test_tuning_picks_big_tiles_for_a_big_matmul():
    kern = autokernel.tpu(big_matmul, tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (128, 256, 512)})
    lw = kern.lower(bf16(S, S), bf16(S, S))
    assert lw.overrides["TM"] >= 256 and lw.overrides["TN"] >= 256 and "# Tuned:" in lw.source
    candidates = kern.tuning(bf16(S, S), bf16(S, S))
    assert len(candidates) == 27 and candidates[0].lowered is lw
    predicted = [c.predicted_s for c in candidates]
    assert predicted == sorted(predicted)
    small = next(c for c in candidates if c.overrides == {"TM": 128, "TN": 128, "TK": 128})
    assert small.predicted_s > 3 * candidates[0].predicted_s  # per-step overhead dominates small tiles


def test_tuned_kernel_is_correct():
    S2 = 512

    def matmul(a, b):
        c = jnp.zeros((S2, S2))
        for i in range(0, S2, TM):
            for j in range(0, S2, TN):
                acc = jnp.zeros((TM, TN))
                for k in range(0, S2, TK):
                    acc += a[i : i + TM, k : k + TK] @ b[k : k + TK, j : j + TN]
                c = c.at[i : i + TM, j : j + TN].set(acc)
        return c

    kern = autokernel.tpu(matmul, tune={"TM": (128, 256), "TN": (128, 256), "TK": (128, 256)})
    a, b = rand((S2, S2)), rand((S2, S2), 1)
    kern.verify(a, b)


def test_invalid_candidates_are_dropped_with_a_reason():
    kern = autokernel.tpu(big_matmul, tune={"TM": (128, 96), "TN": (32, 256), "TK": (256,)})
    lw = kern.lower(bf16(S, S), bf16(S, S))
    assert lw.overrides == {"TM": 128, "TN": 256, "TK": 256}
    rejected = [c for c in kern.tuning(bf16(S, S), bf16(S, S)) if c.lowered is None]
    assert len(rejected) == 3 and all(c.error for c in rejected)
    assert any("multiple of 128" in c.error for c in rejected)  # TN=32 breaks the lane rule
    assert any("aligned" in c.error for c in rejected)  # TM=96 doesn't divide 2048


def test_no_valid_candidate_is_an_error():
    kern = autokernel.tpu(big_matmul, tune={"TN": (32, 64)})
    with pytest.raises(AutoKernelError, match="no tuning candidate is valid"):
        kern.lower(bf16(S, S), bf16(S, S))


def test_too_much_vmem_is_rejected():
    def huge(a):
        out = jnp.zeros((8192, 8192))
        for i in range(0, 8192, 8192):
            out = out.at[i : i + 8192, :].set(a[i : i + 8192, :] * 2)
        return out

    with pytest.raises(AutoKernelError, match="MiB of VMEM"):
        autokernel.tpu(huge).lower(jax.ShapeDtypeStruct((8192, 8192), jnp.float32))


def test_measure_falls_back_to_the_model_without_a_tpu():
    if jax.default_backend() == "tpu":
        pytest.skip("measures on a real TPU")
    kern = autokernel.tpu(big_matmul, tune={"TM": (256, 512)}, autotune="measure")
    lw = kern.lower(bf16(S, S), bf16(S, S))
    assert all(c.measured_s is None for c in kern.tuning(bf16(S, S), bf16(S, S)))
    assert lw is kern.tuning(bf16(S, S), bf16(S, S))[0].lowered
    assert {c.buffers for c in kern.tuning(bf16(S, S), bf16(S, S))} == {2, 3}  # measure also tries 3 buffers


@pytest.mark.parametrize(
    "options, match",
    [
        (dict(buffers=1), "buffers must be at least 2"),
        (dict(autotune="guess"), "autotune must be one of"),
        (dict(cores=0), "cores must be at least 1"),
        (dict(tune={"TM": ()}), "has no candidate values"),
        (dict(matmul_precision="fastest"), "matmul_precision"),
    ],
)
def test_bad_options_fail_at_decoration_time(options, match):
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(**options)


S8 = 8192


def matmul_8192(a, b):
    c = jnp.zeros((S8, S8), jnp.bfloat16)
    for i in range(0, S8, TM):
        for j in range(0, S8, TN):
            acc = jnp.zeros((TM, TN), jnp.float32)
            for k in range(0, S8, TK):
                acc += a[i : i + TM, k : k + TK] @ b[k : k + TK, j : j + TN]
            c = c.at[i : i + TM, j : j + TN].set(acc.astype(jnp.bfloat16))
    return c


def test_model_ranks_tiles_like_the_hardware():
    """Measured on a v5e (README): 1024x1024x512 < 512^3 < 256^3 < 128^3 in run time, for dense bf16."""
    kern = autokernel.tpu(
        matmul_8192, tune={"TM": (128, 256, 512, 1024), "TN": (128, 256, 512, 1024), "TK": (128, 256, 512)}
    )
    predicted = {
        tuple(c.overrides[k] for k in ("TM", "TN", "TK")): c.predicted_s
        for c in kern.tuning(bf16(S8, S8), bf16(S8, S8))
    }
    measured_order = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 512)]
    times = [predicted[t] for t in measured_order]
    assert times == sorted(times, reverse=True)
