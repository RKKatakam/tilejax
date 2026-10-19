"""Regression tests for code generation fixes: what `@` means at every rank, generated names that can't
hide the user's, SMEM and VMEM budgets, the TensorCore split, and generated source that runs on its own."""

import itertools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races
from jax.experimental.pallas import tpu as pltpu
from test_generality import block_sparse_reference, make_block_sparse, make_masked_copy

import autokernel
from autokernel import AutoKernelError, hardware, mxu
from autokernel.lowering import VmemError
from autokernel.schedule import core_bounds

T = 128

interpreter_only = pytest.mark.skipif(
    jax.default_backend() == "tpu", reason="simulates two TensorCores in the CPU interpreter"
)


# --------------------------------------------------------------------------- what `@` means


def _operand(shape, dtype, seed):
    if dtype == jnp.int8:
        return jax.random.randint(jax.random.PRNGKey(seed), shape, -8, 8).astype(jnp.int8)
    return rand(shape, seed, dtype)


# (x shape, y shape, `x @ y.T` instead of `x @ y`)
MATMUL_SHAPES = [
    ((8, 16), (16, 24), False),
    ((8, 16), (24, 16), True),
    ((2, 8, 16), (16, 24), False),
    ((2, 8, 16), (24, 16), True),  # was contracted against the wrong dim of x
    ((3, 2, 8, 16), (24, 16), True),
    ((2, 8, 16), (2, 16, 24), False),
    ((2, 8, 16), (24, 16, 2), True),  # .T of a 3-D array reverses all of its dims
    ((2, 3, 8, 16), (2, 3, 16, 24), False),
    ((2, 3, 8, 16), (24, 16, 3, 2), True),
    ((2, 8, 16), (1, 16, 24), False),  # broadcast batch
    ((16,), (16,), False),  # vectors: dot_general used to fail
    ((16,), (16, 24), False),
    ((8, 16), (16,), False),
    ((8, 16), (16,), True),
]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int8])
@pytest.mark.parametrize("helpers", ["kernel", "reference"])
def test_matmul_helpers_follow_numpy_at_every_rank(helpers, dtype):
    fns = mxu.helpers("highest") if helpers == "kernel" else mxu.reference_helpers()
    for n, (xs, ys, transposed) in enumerate(MATMUL_SHAPES):
        x, y = _operand(xs, dtype, 2 * n), _operand(ys, dtype, 2 * n + 1)
        wide = np.int64 if dtype == jnp.int8 else np.float64
        xn, yn = np.asarray(x, wide), np.asarray(y, wide)
        want = xn @ (yn.T if transposed else yn)
        with jax.default_matmul_precision("highest"):  # as `reference` runs
            got = fns[mxu.CALLS["_mxu_dot_t" if transposed else "_mxu_dot"]](x, y)
        assert got.shape == want.shape, (xs, ys, transposed)
        assert got.dtype == (jnp.int32 if dtype == jnp.int8 else jnp.float32)  # the MXU's accumulator
        np.testing.assert_allclose(np.asarray(got, wide), want, rtol=1e-5, atol=1e-4, err_msg=str((xs, ys)))


def test_stack_of_matrices_times_a_transposed_matrix():
    @autokernel.tpu
    def batched_qkT(q, k):
        out = jnp.zeros((2, 128, 256))
        for j in range(0, 256, T):
            out = out.at[:, :, j : j + T].set(q[:, :, :] @ k[j : j + T, :].T)
        return out

    q, k = rand((2, 128, 128)), rand((256, 128), 1)
    want = np.einsum("bmk,nk->bmn", np.asarray(q, np.float64), np.asarray(k, np.float64))
    np.testing.assert_allclose(run_checking_races(batched_qkT, q, k), want, rtol=1e-5, atol=1e-4)
    assert batched_qkT.verify(q, k) < 1e-4


def test_verify_does_not_share_the_kernels_matmul(monkeypatch):
    # A kernel whose `@` is wrong: the reference (jnp.matmul) must not be wrong with it.
    broken = mxu._PRELUDE.replace(
        "return jnp.dot(x, y, preferred_element_type", "return 2 * jnp.dot(x, y, preferred_element_type"
    )
    assert broken != mxu._PRELUDE
    monkeypatch.setattr(mxu, "_PRELUDE", broken)
    mxu.helpers.cache_clear()
    try:

        def mm(x, y):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, T):
                out = out.at[i : i + T, :].set(x[i : i + T, :] @ y[:, :])
            return out

        kern = autokernel.tpu(mm)
        x, y = rand((128, 128)), rand((128, 128), 1)
        np.testing.assert_allclose(kern.reference(x, y), np.asarray(x) @ np.asarray(y), rtol=1e-5, atol=1e-4)
        with pytest.raises(AssertionError, match="disagrees"):
            kern.verify(x, y)
    finally:
        mxu.helpers.cache_clear()


# --------------------------------------------------------------------------- generated names


def _rem(a, b):  # named like a generated helper: Python's modulo, not jax.lax.rem
    return jnp.mod(a, b)


def _blocks(a):  # named like the kernel's block-index function
    return a * 2


def _decode(a):  # and its step decoder
    return a + 1


def _mxu_dot(a, b):  # and the MXU helper
    return a - b


_COST = 3.0  # and the cost estimate


def named_like_generated(x):
    out = jnp.zeros((256, 128))
    for i in range(0, 256, T):
        t = x[i : i + T, :]
        out = out.at[i : i + T, :].set(_rem(t, 3.0) + _blocks(t) + _decode(t) + _mxu_dot(t, t) * _COST)
    return out


def test_user_globals_named_like_generated_code():
    x = rand((256, 128)) * 5
    xn = np.asarray(x)
    want = np.mod(xn, 3.0) + xn * 2 + (xn + 1)
    kern = autokernel.tpu(named_like_generated)
    np.testing.assert_allclose(run_checking_races(kern, x), want, rtol=1e-5, atol=1e-5)
    assert kern.verify(x) < 1e-4


def test_matmul_beside_user_functions_named_like_its_helpers():
    def mm(x, y):
        out = jnp.zeros((256, 128))
        for i in range(0, 256, T):
            out = out.at[i : i + T, :].set(_mxu_dot(x[i : i + T, :] @ y[:, :], x[i : i + T, :]))
        return out

    x, y = rand((256, 128)), rand((128, 128), 1)
    kern = autokernel.tpu(mm)
    want = np.asarray(x) @ np.asarray(y) - np.asarray(x)
    np.testing.assert_allclose(run_checking_races(kern, x, y), want, rtol=1e-5, atol=1e-4)


pl = "not pallas"  # the generated kernel's import name


def reads_pl(x):
    out = jnp.zeros((128, 128))
    for i in range(0, 128, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * len(pl))
    return out


def test_user_global_named_like_an_import_of_the_kernel_is_renamed():
    x = rand((128, 128))
    lowered = autokernel.tpu(reads_pl).lower(x)
    assert "pl_ = 'not pallas'" in lowered.source  # the kernel reads the user's `pl` under a fresh name
    want = np.asarray(x) * len(pl)
    np.testing.assert_allclose(lowered.entry(x, interpret=pltpu.InterpretParams()), want, rtol=1e-6)
    np.testing.assert_allclose(_standalone(lowered.source, (x,)), want, rtol=1e-6)


# --------------------------------------------------------------------------- types


def test_single_step_accumulator_keeps_a_python_floats_weak_type():
    def weak_acc(x):
        out = jnp.zeros((256, 128), jnp.bfloat16)
        for i in range(0, 256, T):
            acc = 0.0
            for _ in range(0, 1):
                acc += 0.1 * (i + 1)
            out = out.at[i : i + T, :].set((x[i : i + T, :] * acc) * 3.3)  # bf16 throughout, as in Python
        return out

    kern = autokernel.tpu(weak_acc)
    x = rand((256, 128), 0, jnp.bfloat16)
    assert re.search(r"\bacc = _weak\w*\(", kern.source(x))  # a Python float's weak type, not a strong f32
    out = run_checking_races(kern, x)
    assert out.dtype == jnp.bfloat16
    if jax.default_backend() == "tpu":  # Mosaic and XLA may round fused bf16 math differently: one ulp
        np.testing.assert_allclose(np.float32(out), np.float32(kern.reference(x)), rtol=2**-7, atol=2**-7)
    else:
        np.testing.assert_array_equal(np.float32(out), np.float32(kern.reference(x)))


# --------------------------------------------------------------------------- budgets


def causal_tiles(x):
    out = jnp.zeros((128, 4096, 4096), jnp.bfloat16)
    for b in range(128):
        for i in range(0, 4096, T):
            for j in range(0, 4096, T):
                if j > i:
                    continue
                out = out.at[b, i : i + T, j : j + T].set(x[b, i : i + T, j : j + T])
    return out


def test_tile_and_fill_tables_count_against_smem():
    x = jax.ShapeDtypeStruct((128, 4096, 4096), jnp.bfloat16)
    match = r"tile tables .* fill tables .*use background_fill=False.*; or bigger blocks"
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(causal_tiles).lower(x)


def big_matmul(x, y):
    out = jnp.zeros((2048, 2048))
    for i in range(0, 2048, 1024):
        for j in range(0, 2048, 1024):
            acc = jnp.zeros((1024, 1024))
            for k in range(0, 2048, 1024):
                acc += x[i : i + 1024, k : k + 1024] @ y[k : k + 1024, j : j + 1024]
            out = out.at[i : i + 1024, j : j + 1024].set(acc)
    return out


def test_vmem_limit_bytes_is_the_budget():
    specs = [jax.ShapeDtypeStruct((2048, 2048), jnp.float32)] * 2
    default = autokernel.tpu(big_matmul).lower(*specs)
    assert default.buffers == 3
    need = default.estimate.vmem_bytes
    tight = autokernel.tpu(big_matmul, vmem_limit_bytes=need - 1).lower(*specs)
    assert tight.buffers == 2 and tight.estimate.vmem_bytes <= need - 1  # three buffers don't fit the limit
    with pytest.raises(VmemError, match="vmem_limit_bytes"):
        autokernel.tpu(big_matmul, vmem_limit_bytes=8 * 2**20).lower(*specs)


# --------------------------------------------------------------------------- the TensorCore split


def test_core_split_is_optimal_for_any_core_count():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, cores = int(rng.integers(1, 10)), int(rng.integers(1, 6))
        starts = np.concatenate([[0], np.cumsum(rng.integers(1, 10, n))])
        best = min(
            max(starts[b] - starts[a] for a, b in itertools.pairwise([0, *mid, n]))
            for mid in itertools.combinations_with_replacement(range(n + 1), cores - 1)
        )
        for xp in (np, jnp):
            cuts = np.asarray(core_bounds(xp.asarray(starts), cores, xp))
            assert cuts[0] == 0 and cuts[-1] == n and np.all(np.diff(cuts) >= 0) and len(cuts) == cores + 1
            assert max(starts[b] - starts[a] for a, b in itertools.pairwise(cuts)) == best, (starts, cores, cuts)


@interpreter_only
def test_runtime_split_over_four_cores():
    kern = make_block_sparse(512, 512, 256, 128, cores=4)
    x, y = rand((512, 512)), rand((512, 256), 1)
    mask = jnp.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], jnp.int32)
    params = pltpu.InterpretParams(detect_races=True, num_cores_or_threads=4)
    lowered = kern.lower(x, y, mask)
    np.testing.assert_allclose(
        lowered.entry(x, y, mask, interpret=params), block_sparse_reference(x, y, mask, 128), atol=1e-3
    )


# --------------------------------------------------------------------------- runtime scalars


def test_float_runtime_scalars_are_not_indices():
    from test_runtime_scalars import scaled

    prog = autokernel.tpu(scaled)
    x, w = rand((256, 256), 1), rand((256, 256), 2)
    lowered = prog.lower(x, w, jnp.float32(4.0))
    lowered["scaled_h"].check_indices({"scale": jnp.float32(np.inf)})  # used to raise OverflowError
    lowered["scaled_h"].check_indices({"scale": jnp.float32(np.nan)})
    prog.verify_nests(x, w, jnp.float32(0.0))  # scale = inf


# --------------------------------------------------------------------------- generated source


def helper_fn(t):  # a function the tile code calls: its source goes into the kernel's
    return jnp.maximum(t, 0) * SCALE


SCALE = 3


def dense(x, y):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            acc = jnp.zeros((T, T))
            for k in range(0, 256, T):
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(helper_fn(acc))
    return out


def causal(x, y):
    out = jnp.full((512, 512), -jnp.inf)
    for i in range(0, 512, T):
        for j in range(0, 512, T):
            if j > i:
                continue
            acc = jnp.zeros((T, T))
            for k in range(0, 512, T):
                if k > i:
                    continue
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def _standalone(source, args):
    namespace = {}
    exec(source, namespace)  # nothing but the source
    entry = next(v for k, v in namespace.items() if k.endswith("_pallas"))
    return entry(*args, interpret=pltpu.InterpretParams())


def _kernels():
    x, y = rand((256, 256)), rand((256, 256), 1)
    yield autokernel.tpu(dense), (x, y)
    yield autokernel.tpu(causal), (rand((512, 512)), rand((512, 512), 1))
    mask = jnp.asarray([[1, 0], [1, 1]], jnp.int32)
    yield make_block_sparse(256, 256, 256, 128), (x, y, mask)
    yield make_masked_copy(), (rand((512, 256)), jnp.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], jnp.int32))


@pytest.mark.parametrize("n", range(4))
def test_generated_source_runs_on_its_own(n):
    kern, args = list(_kernels())[n]
    lowered = kern.lower(*args)
    got = _standalone(lowered.source, args)
    want = lowered.entry(*args, interpret=pltpu.InterpretParams())
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert " = pl.CostEstimate(" in lowered.source


def test_fused_source_runs_on_its_own():
    from test_fusion import K, M, N, masks_and_fills

    prog = autokernel.tpu(masks_and_fills)
    args = {"x": rand((M, N), 1), "y": rand((K, N), 2), "block_mask": jnp.array([[1, 0], [1, 1]], jnp.int32)}
    [(_, fused)] = list(prog.lower(*args.values()))
    inputs = [args[name] for name in fused.inputs]
    got = _standalone(fused.source, inputs)
    want = fused.entry(*inputs, interpret=pltpu.InterpretParams())
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_wrapper_comment_says_how_inputs_are_buffered():
    x, y = rand((256, 256)), rand((256, 256), 1)
    assert "inputs 3-buffered by hand" in autokernel.tpu(dense).lower(x, y).source
    per_tile = autokernel.tpu(named_like_generated).lower(rand((256, 128))).source
    assert "inputs fetched once per tile, 2-buffered by hand" in per_tile and "3-buffered" not in per_tile


# --------------------------------------------------------------------------- hardware


def test_vector_unit_clocks():
    assert hardware.target("v5e").clock_hz == 1.5e9
    assert hardware.target("v6e").clock_hz == 1.75e9  # 918 TFLOP/s from four 256x256 MXUs, not 3.5 GHz
