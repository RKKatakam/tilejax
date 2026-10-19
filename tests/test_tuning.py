"""Hardware targets, the cost model, and block-size tuning."""

import itertools

import jax
import jax.numpy as jnp
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, frontend, hardware, lowering
from autokernel import tune as tune_lib
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


def test_vmem_limit_is_the_chips():
    # Mosaic takes its own scratch from what the limit leaves, so kernels get the chip's scoped VMEM, not
    # just their estimate; `vmem_limit_bytes` overrides it.
    lw = autokernel.tpu(big_matmul).lower(bf16(S, S), bf16(S, S))
    assert lw.vmem_limit_bytes == hardware.target().vmem_limit >= lw.estimate.vmem_bytes
    assert f"vmem_limit_bytes={lw.vmem_limit_bytes}" in lw.source
    lw = autokernel.tpu(big_matmul, vmem_limit_bytes=40 << 20).lower(bf16(S, S), bf16(S, S))
    assert lw.vmem_limit_bytes == 40 << 20


def test_tuning_picks_big_tiles_for_a_big_matmul():
    kern = autokernel.tpu(big_matmul, tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (128, 256, 512)})
    lw = kern.lower(bf16(S, S), bf16(S, S))
    assert lw.overrides["TM"] >= 256 and lw.overrides["TN"] >= 256 and "# Tuned:" in lw.source
    candidates = kern.tuning(bf16(S, S), bf16(S, S))
    assert len(candidates) == 27 and candidates[0].lowered is lw
    scores = [c.score for c in candidates]
    assert scores == sorted(scores)
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
    rejected = [c for c in kern.tuning(bf16(S, S), bf16(S, S)) if c.error is not None]
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
        (dict(tune="fast"), "tune must be 'auto' or a mapping"),
        (dict(tune=(128, 256)), "tune must be 'auto' or a mapping"),
        (dict(tune={"TM": "fast"}), r"tune\['TM'\] must be 'auto' or a sequence"),
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


# --------------------------------------------------------------------------- tune="auto"

RM, RN, RK = 256, 384, 1024  # a matmul whose dims aren't all powers of two
BM = BN = BK = 128


def small_matmul(a, b):
    c = jnp.zeros((RM, RN))
    for i in range(0, RM, BM):
        for j in range(0, RN, BN):
            acc = jnp.zeros((BM, BN))
            for k in range(0, RK, BK):
                acc += a[i : i + BM, k : k + BK] @ b[k : k + BK, j : j + BN]
            c = c.at[i : i + BM, j : j + BN].set(acc)
    return c


SMALL = (jax.ShapeDtypeStruct((RM, RK), jnp.float32), jax.ShapeDtypeStruct((RK, RN), jnp.float32))


def tried(kern, *specs):
    """{name: values tried} over every tuning candidate."""
    values = {}
    for c in kern.tuning(*specs):
        for name, v in c.overrides.items():
            values.setdefault(name, set()).add(v)
    return {name: sorted(v) for name, v in values.items()}


def test_auto_generates_block_sizes_from_the_loops():
    # BM tiles the rows of `a` and `c` (f32 sublanes: from 8); BN the lanes of `b` and `c` (from 128, or all
    # 384); BK the lanes of `a` (from 128). Only values dividing their loop are kept.
    kern = autokernel.tpu(small_matmul, tune="auto")
    assert tried(kern, *SMALL) == {
        "BM": [8, 16, 32, 64, 128, 256],
        "BN": [128, 384],
        "BK": [128, 256, 512, 1024],
    }
    assert len(kern.tuning(*SMALL)) == 6 * 2 * 4
    sizes, _ = tune_lib.block_sizes(frontend.parse(small_matmul), small_matmul, SMALL)
    assert sizes["BN"] == (128, 384)
    lw = kern.lower(*SMALL)
    assert lw is kern.tuning(*SMALL)[0].lowered and "# Tuned:" in lw.source


def test_auto_minimum_follows_the_dtype():
    bf = (jax.ShapeDtypeStruct((RM, RK), jnp.bfloat16), jax.ShapeDtypeStruct((RK, RN), jnp.bfloat16))
    sizes, _ = tune_lib.block_sizes(frontend.parse(small_matmul), small_matmul, bf)
    assert sizes["BM"][0] == 16  # bf16 inputs pack 16 rows per tile (the f32 output needs only 8)


def test_auto_and_explicit_values_mix():
    kern = autokernel.tpu(small_matmul, tune={"BM": "auto", "BK": (256, 512)})
    assert tried(kern, *SMALL) == {"BM": [8, 16, 32, 64, 128, 256], "BK": [256, 512]}  # BN keeps its value


def test_auto_keeps_the_current_value_and_caps_the_range():
    def wide(a):
        out = jnp.zeros((8, 16384))
        for j in range(0, 16384, WIDE):
            out = out.at[:, j : j + WIDE].set(a[:, j : j + WIDE] + 1)
        return out

    WIDE = 4096  # bigger than tune_lib.AUTO_MAX, kept because it's the user's own choice
    sizes, _ = tune_lib.block_sizes(frontend.parse(wide), wide, [jax.ShapeDtypeStruct((8, 16384), jnp.float32)])
    assert sizes == {"WIDE": (128, 256, 512, 1024, 2048, 4096)}


B2, ONE, TR = 2, 1, 8


def batched(x):
    _, R, C = x.shape
    TC = C // 2  # derived from the input's shape: not a tunable constant
    out = jnp.zeros((B2, R, C))
    for b in range(0, B2, ONE):  # a step of 1 indexes a dim; it isn't a block size
        for i in range(0, R, TR):
            for j in range(0, C, TC):
                out = out.at[b, i : i + TR, j : j + TC].set(x[b, i : i + TR, j : j + TC] * 2)
    return out


X3 = jax.ShapeDtypeStruct((B2, 64, 256), jnp.float32)


def test_steps_from_shapes_or_of_one_are_not_tuned():
    sizes, why_not = tune_lib.block_sizes(frontend.parse(batched), batched, [X3])
    assert sizes == {"TR": (8, 16, 32, 64)}
    assert "prologue" in why_not["TC"] and "stepping by 1" in why_not["ONE"]
    kern = autokernel.tpu(batched, tune="auto")
    assert tried(kern, X3) == {"TR": [8, 16, 32, 64]}
    x = rand((B2, 64, 256))
    kern.verify(x)


@pytest.mark.parametrize(
    "tune, match",
    [
        ({"TC": "auto"}, "`TC` is computed in the function's prologue"),
        ({"ONE": "auto"}, "loops stepping by 1 index a dim"),
        ({"B2": "auto"}, "`B2` isn't the step of any loop"),
        ({"TC": (64, 128)}, "computed in the function's prologue, so overriding it has no effect"),
        ({"NOPE": (1, 2)}, "`NOPE` isn't used by the function"),
    ],
)
def test_auto_tuning_errors_name_the_constant(tune, match):
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(batched, tune=tune).lower(X3)


def test_auto_with_nothing_to_tune_is_an_error():
    def fixed(x):
        out = jnp.zeros((64, 256))
        for i in range(0, 64, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] + 1)
        return out

    with pytest.raises(AutoKernelError, match="found no block sizes to tune"):
        autokernel.tpu(fixed, tune="auto").lower(jax.ShapeDtypeStruct((64, 256), jnp.float32))


def test_screening_ranks_like_exhaustive_lowering():
    """Only the best few candidates are generated; the ranking matches lowering every combination."""
    kern = autokernel.tpu(small_matmul, tune="auto")
    candidates = kern.tuning(*SMALL)
    built = [c for c in candidates if c.lowered is not None]
    valid = [c for c in candidates if c.error is None]
    assert len(built) == lowering.Options().measure_top and built == tune_lib.diverse(valid, len(built))
    assert built[0] is candidates[0]
    assert all(c.estimate is not None for c in valid)

    nest, options = frontend.parse(small_matmul), lowering.Options()
    grid = tune_lib.grid(nest, small_matmul, SMALL, lowering.Options(tune=lowering.Options.make_tune("auto")))
    exhaustive = []
    for values in itertools.product(*(v for _, v in grid)):
        overrides = dict(zip([name for name, _ in grid], values, strict=True))
        try:
            lw = lowering.lower(nest, small_matmul, SMALL, options, overrides)
        except AutoKernelError:
            continue
        exhaustive.append((overrides, lw.estimate.score, lw.buffers))
    exhaustive.sort(key=lambda e: e[1])
    assert [(c.overrides, c.score, c.buffers) for c in valid] == exhaustive
    assert all(
        c.lowered.source == lowering.lower(nest, small_matmul, SMALL, options, c.overrides).source for c in built[:2]
    )


def test_measure_times_the_best_auto_candidates(monkeypatch):
    kern = autokernel.tpu(small_matmul, tune="auto", autotune="measure", interpret=True)
    model_pick = kern.lower(*SMALL)
    built = [c for c in kern.tuning(*SMALL) if c.lowered is not None]
    assert {c.buffers for c in built} == {2, 3}  # the best block sizes at both depths
    # Pretend to be on a TPU where the model's eighth choice is the fastest.
    eighth = list(dict.fromkeys(tuple(sorted(c.overrides.items())) for c in built))[7]
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    monkeypatch.setattr(
        tune_lib, "time_kernel", lambda lw, args: 0.5 if tuple(sorted(lw.overrides.items())) == eighth else 1.0
    )
    a, b = rand((RM, RK)), rand((RK, RN), 1)
    picked = kern.lower(a, b)
    assert picked is not model_pick and tuple(sorted(picked.overrides.items())) == eighth
    assert len([c for c in kern.tuning(a, b) if c.measured_s is not None]) == 8 + 1  # and the other depth


# --------------------------------------------------------------------------- searching big spaces


def search_best(fn, specs, tune, monkeypatch, exhaustive_max):
    monkeypatch.setattr(tune_lib, "EXHAUSTIVE_MAX", exhaustive_max)
    options = lowering.Options(tune=lowering.Options.make_tune(tune), measure_top=1)
    candidates = tune_lib.search(frontend.parse(fn), fn, specs, options)
    return candidates[0], candidates


@pytest.mark.parametrize(
    "fn, specs",
    [(small_matmul, SMALL), (matmul_8192, (bf16(S8, S8), bf16(S8, S8))), (big_matmul, (bf16(S, S), bf16(S, S)))],
)
def test_descent_finds_what_exhaustive_search_finds(fn, specs, monkeypatch):
    full, every = search_best(fn, specs, "auto", monkeypatch, 10**9)
    found, planned = search_best(fn, specs, "auto", monkeypatch, 0)
    assert found.overrides == full.overrides and found.score == full.score
    assert len(planned) < len(every)


R12 = 12288
T1 = T2 = TA = TB = 512


def two_reductions(x, y, z, w):
    out = jnp.zeros((R12, R12), jnp.bfloat16)
    for i in range(0, R12, TA):
        for j in range(0, R12, TB):
            a1 = jnp.zeros((TA, TB))
            for k in range(0, R12, T1):
                a1 += x[i : i + TA, k : k + T1] @ y[k : k + T1, j : j + TB]
            a2 = jnp.zeros((TA, TB))
            for k2 in range(0, R12, T2):
                a2 += z[i : i + TA, k2 : k2 + T2] @ w[k2 : k2 + T2, j : j + TB]
            out = out.at[i : i + TA, j : j + TB].set((a1 + a2).astype(jnp.bfloat16))
    return out


def test_four_names_of_eight_values_are_searched_quickly(monkeypatch):
    """8^4 = 4096 combinations: planning them all takes ~12 s; the descent plans under 300 (about 0.8 s on a
    laptop). Counted in plans rather than timed, so a slow machine doesn't fail it."""
    plans = []
    real_plan = tune_lib.plan
    monkeypatch.setattr(tune_lib, "plan", lambda *a, **k: plans.append(a[4]) or real_plan(*a, **k))
    lanes = (128, 256, 512, 768, 1024, 1536, 2048, 3072)
    tune = {"TA": (16, 32, 64, 128, 256, 512, 1024, 2048), "TB": lanes, "T1": lanes, "T2": lanes}
    specs = [bf16(R12, R12)] * 4
    best, planned = search_best(two_reductions, specs, tune, monkeypatch, tune_lib.EXHAUSTIVE_MAX)
    assert len(planned) < 300 and best.overrides == {"T1": 2048, "T2": 2048, "TA": 1024, "TB": 768}
    assert len(plans) <= len(planned) + 1  # each combination planned once, plus the one built


def test_blocks_bigger_than_one_that_does_not_fit_are_not_planned(monkeypatch):
    calls = []
    real_plan = tune_lib.plan
    monkeypatch.setattr(tune_lib, "plan", lambda *a, **k: calls.append(a[4]) or real_plan(*a, **k))
    tune = {"TM": (512, 1024, 2048, 4096), "TN": (512, 1024, 2048, 4096), "TK": (512, 1024, 2048)}
    candidates = autokernel.tpu(matmul_8192, tune=tune).tuning(bf16(S8, S8), bf16(S8, S8))
    assert len(candidates) == 48  # a small space: every combination is listed
    pruned = [c for c in candidates if c.error and c.error.startswith("too much VMEM")]
    assert pruned and len(calls) == 48 - len(pruned) + lowering.Options().measure_top
    assert all(c.overrides["TM"] * c.overrides["TN"] >= 2048 * 4096 for c in pruned)
