"""Edge cases of the kernel API, tuning, batching and differentiation: error reports, fallbacks, and the
branches that only run on a TPU (simulated by patching `hardware.on_tpu` and `tune.time_kernel`)."""

import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import AutoKernelError, analysis, batching, frontend, hardware, lowering, sharding
from autokernel import tune as tune_lib
from autokernel.api import check_outputs

M, N = 256, 256
T = 64
TR = 64
SCALE = 2.0
FLOAT_STEP = 64.0


def f32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def close(got, want, tol=1e-5):
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tol, atol=tol)


def double(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
    return out


# --------------------------------------------------------------------------- arguments and interpreting


def test_non_array_arguments_are_rejected():
    kernel = autokernel.tpu(double, interpret=True)
    with pytest.raises(AutoKernelError, match="kernel arguments must be arrays, got str"):
        kernel.lower("not an array")


def make_closure_kernel(delete_bias):
    bias = rand((M, N), 5)

    def add_bias(x):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] + bias[i : i + T, :])
        return out

    kernel = autokernel.tpu(add_bias, interpret=True)
    if delete_bias:
        del bias
        return kernel, None
    return kernel, bias


def test_closure_arrays_are_inputs_read_on_every_call():
    kernel, bias = make_closure_kernel(delete_bias=False)
    x = rand((M, N))
    close(kernel(x), x + bias)


def test_an_unset_closure_array_is_reported():
    kernel, _ = make_closure_kernel(delete_bias=True)
    with pytest.raises(AutoKernelError, match="closure variable `bias` has no value yet"):
        kernel(rand((M, N)))


def test_interpret_params_are_passed_through():
    kernel = autokernel.tpu(double, interpret=pltpu.InterpretParams())
    assert isinstance(kernel._interpret_mode(), pltpu.InterpretParams)
    x = rand((M, N))
    close(kernel(x), 2 * x)


def test_on_a_tpu_kernels_run_natively_by_default(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    assert autokernel.tpu(double)._interpret_mode() is False


TS = 8


def tiny_steps(x):
    out = jnp.zeros((4096, 128))
    for i in range(0, 4096, TS):
        out = out.at[i : i + TS, :].set(x[i : i + TS, :] + 1.0)
    return out


def test_overhead_bound_kernels_warn_on_a_tpu(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    with pytest.warns(UserWarning, match='larger blocks would run faster: try tune="auto"'):
        autokernel.tpu(tiny_steps, target="v5e").lower(f32(4096, 128))


def test_check_outputs_reports_mismatched_structures_and_shapes():
    a = jnp.zeros((8, 128))
    with pytest.raises(AssertionError, match="kernels returned"):
        check_outputs("k", (a, a), a, "highest")
    with pytest.raises(AssertionError, match=r"kernel returned float32\(8, 128\), reference returned"):
        check_outputs("k", a, jnp.zeros((16, 128)), "highest")
    assert check_outputs("k", a, a, "highest") == 0.0


# --------------------------------------------------------------------------- jax.vmap


def shape_attrs(x):
    """A docstring stays first in the batched function."""
    rows = len(x)
    cols = x.shape[1]
    rank = x.ndim
    per_row = x.size // rows
    dtype = x.dtype
    out = jnp.zeros((rows, per_row + cols - N), dtype)
    for i in range(0, rows, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * rank)
    return out


def test_vmap_gives_shapes_per_batch_element():
    kernel = autokernel.TpuKernel(shape_attrs, interpret=True)
    xs = jnp.stack([rand((M, N), s) for s in range(3)])
    close(jax.vmap(kernel)(xs), xs * 2)
    assert len(kernel._batched) == 1 and all(isinstance(k, autokernel.TpuKernel) for k in kernel._batched.values())


def whole_shape(x):
    rows = jnp.shape(x)[0]  # reads `x` itself, not just its shape attributes
    out = jnp.zeros((rows, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
    return out


def test_vmap_of_a_kernel_reading_a_whole_input_runs_per_element():
    kernel = autokernel.TpuKernel(whole_shape, interpret=True)  # a single nest, so a kernel
    xs = jnp.stack([rand((M, N), s) for s in range(2)])
    with pytest.warns(UserWarning, match="`x` is used as a whole array"):
        out = jax.vmap(kernel)(xs)
    close(out, xs * 2)
    out = jax.vmap(kernel, in_axes=1)(xs.transpose(1, 0, 2))  # remembered: no new attempt, no new warning
    close(out, xs * 2)
    (failed,) = kernel._batched.values()
    assert isinstance(failed, batching.BatchingError)


def in_place(x):
    x = x
    for i in range(0, M, T):
        x = x.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
    return x


def test_vmap_of_an_in_place_update_runs_per_element():
    # Batched, the output is a stacked allocation rather than the input itself, which the nest then reads.
    kernel = autokernel.tpu(in_place, interpret=True)
    xs = jnp.stack([rand((M, N), s) for s in range(2)])
    with pytest.warns(UserWarning, match="rebinds parameter `x`"):
        out = jax.vmap(kernel)(xs)
    close(out, xs * 2)


def update_corner(x, y):
    out = y
    for i in range(0, M, T):
        out = out.at[i : i + T, 0:128].set(x[i : i + T, 0:128] * 2.0)
    return out


def test_vmap_of_a_partial_update_runs_per_element():
    # The batched output no longer keeps `y`'s values, so its uncovered columns can't be produced.
    kernel = autokernel.tpu(update_corner, interpret=True)
    xs, ys = jnp.stack([rand((M, N), s) for s in range(2)]), jnp.stack([rand((M, N), s + 2) for s in range(2)])
    with pytest.warns(UserWarning, match="the tiles don't write all of `out`'s dim 2"):
        out = jax.vmap(kernel)(xs, ys)
    close(out, ys.at[:, :, :128].set(xs[:, :, :128] * 2))
    with pytest.warns(UserWarning):  # a new kernel warns again
        autokernel.tpu(update_corner, interpret=True)
        jax.vmap(autokernel.tpu(update_corner, interpret=True))(xs, ys)


# --------------------------------------------------------------------------- differentiation


def to_int(x):
    out = jnp.zeros((M, N), jnp.int32)
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set((x[i : i + T, :] * 4.0).astype(jnp.int32))
    return out


def int_double(x):
    out = jnp.zeros((M, N), jnp.int32)
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * 2)
    return out


def test_jvp_of_integer_outputs_is_float0():
    kernel = autokernel.tpu(to_int, interpret=True)
    x = rand((M, N))
    out, tangent = jax.jvp(kernel, (x,), (jnp.ones_like(x),))
    np.testing.assert_array_equal(np.asarray(out), np.asarray((x * 4.0).astype(jnp.int32)))
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == (M, N)


def test_jvp_of_an_integer_kernel():
    kernel = autokernel.tpu(int_double, interpret=True)
    x = jnp.arange(M * N, dtype=jnp.int32).reshape(M, N)
    out, tangent = jax.jvp(kernel, (x,), (np.zeros((M, N), jax.dtypes.float0),))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x * 2))
    assert tangent.dtype == jax.dtypes.float0


def scale_by_counts(x, counts):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * counts[i : i + T, :].astype(jnp.float32))
    return out


def scale_by_counts_backward(x, counts, cotangent):
    return cotangent * counts.astype(jnp.float32), None  # no gradient for the integer counts


def test_a_backward_without_a_gradient_for_an_integer_input():
    kernel = autokernel.tpu(scale_by_counts, interpret=True, backward=scale_by_counts_backward)
    x, counts = rand((M, N)), jnp.arange(M * N, dtype=jnp.int32).reshape(M, N) % 5
    gx, gc = jax.grad(lambda x, c: jnp.sum(kernel(x, c)), argnums=(0, 1), allow_int=True)(x, counts)
    close(gx, counts.astype(jnp.float32))
    assert gc.dtype == jax.dtypes.float0 and gc.shape == (M, N)


# --------------------------------------------------------------------------- tuning


def test_tune_values_must_be_a_sequence():
    # `autokernel.tpu` makes tuples of them; `Options` built directly may hold anything.
    options = lowering.Options(tune=(("T", [64, 128]),))
    with pytest.raises(AutoKernelError, match=r"tune\['T'\] must be 'auto' or a sequence of candidate values"):
        tune_lib.grid(frontend.parse(double), double, [f32(M, N)], options)


def float_step(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, FLOAT_STEP):
        out = out.at[i : i + 64, :].set(x[i : i + 64, :])
    return out


def test_auto_reports_steps_that_are_not_ints():
    with pytest.raises(AutoKernelError, match=r"`FLOAT_STEP` is 64.0, not an int"):
        autokernel.tpu(float_step, tune="auto").lower(f32(M, N))


def test_block_sizes_report_steps_that_are_not_constants():
    nest = frontend.parse(double)
    env = analysis.Environment({"M": M, "N": N}, __file__)  # `T` unbound
    sizes, why_not = tune_lib.block_sizes(nest, env, [f32(M, N)])
    assert sizes == {} and why_not == {"T": "isn't a module global or closure variable"}


def bad_allocation(x):
    out = jnp.zeros((M, N), "not a dtype")
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :])
    return out


def test_auto_tuning_a_kernel_whose_allocation_fails():
    with pytest.raises(AutoKernelError, match="couldn't evaluate the output allocation"):
        autokernel.tpu(bad_allocation, tune="auto").lower(f32(M, N))


def failing_prologue(x):
    assert x.shape[0] == 3 * M
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :])
    return out


def test_tuning_a_kernel_whose_prologue_fails():
    with pytest.raises(AutoKernelError, match="no tuning candidate is valid"):
        autokernel.tpu(failing_prologue, tune={"T": (64, 128)}).lower(f32(M, N))


R, C = 65536, 4096


def wide_rows(x):
    out = jnp.zeros((R, C))
    for i in range(0, R, TR):
        out = out.at[i : i + TR, :].set(x[i : i + TR, :] * SCALE)
    return out


def test_vmem_pruning_bisects_the_block_sizes():
    kernel = autokernel.tpu(wide_rows, tune={"TR": (8, 16, 32, 16384)})
    lowered = kernel.lower(f32(R, C))
    by_size = {c.overrides["TR"]: c for c in kernel.tuning(f32(R, C))}
    assert lowered.overrides["TR"] in (8, 16, 32)
    assert by_size[16384].error is not None and "VMEM" in by_size[16384].error


def test_vmem_pruning_stops_at_a_block_size_failing_otherwise():
    # 24 doesn't divide the rows: it fails as a ragged loop, which says nothing about VMEM.
    kernel = autokernel.tpu(wide_rows, tune={"TR": (8, 24, 16384)})
    candidates = {c.overrides["TR"]: c for c in kernel.tuning(f32(R, C))}
    assert "ragged" in candidates[24].error and candidates[8].error is None


def test_descent_starts_from_the_nearest_values(monkeypatch):
    # TR (64) isn't a candidate: the descent starts from its nearest one; SCALE isn't a block size, so
    # from the middle of its values.
    monkeypatch.setattr(tune_lib, "EXHAUSTIVE_MAX", 0)
    kernel = autokernel.tpu(wide_rows, tune={"TR": (32, 128, 256), "SCALE": (1.0, 3.0)})
    lowered = kernel.lower(f32(R, C))
    assert lowered.overrides["TR"] in (32, 128, 256) and lowered.overrides["SCALE"] in (1.0, 3.0)


def test_a_searched_space_without_a_valid_candidate_says_why(monkeypatch):
    monkeypatch.setattr(tune_lib, "EXHAUSTIVE_MAX", 0)
    kernel = autokernel.tpu(double, tune={"T": (3, 5, 7)})
    with pytest.raises(AutoKernelError, match="no tuning candidate is valid"):
        kernel.lower(f32(M, N))


def test_measure_drops_candidates_that_fail_to_run(monkeypatch):
    kernel = autokernel.tpu(double, tune={"T": (64, 128)}, autotune="measure", interpret=True, target="v5e")
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)

    def time_kernel(lowered, args):
        if lowered.overrides["T"] == 64:
            raise RuntimeError("Mosaic rejects it")
        return 1.0

    monkeypatch.setattr(tune_lib, "time_kernel", time_kernel)
    x = rand((M, N))
    assert kernel.lower(x).overrides["T"] == 128
    errors = [c.error for c in kernel.tuning(x) if c.overrides["T"] == 64]
    assert "failed to run: RuntimeError" in errors


def test_measure_keeps_the_model_pick_when_nothing_runs(monkeypatch):
    kernel = autokernel.tpu(double, tune={"T": (64, 128)}, autotune="measure", interpret=True)
    model_pick = kernel.lower(f32(M, N))
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)

    def time_kernel(lowered, args):
        raise RuntimeError("no TPU after all")

    monkeypatch.setattr(tune_lib, "time_kernel", time_kernel)
    x = rand((M, N))
    assert kernel.lower(x) is model_pick
    assert all(c.measured_s is None for c in kernel.tuning(x))


# --------------------------------------------------------------------------- sharding helpers


def test_strided_and_open_slices_are_not_split():
    loops = {"i"}
    for src, want in [("x[i : i + T : 2, j]", [None, None]), ("x[i : 2 * i, :]", [None, None])]:
        index = ast.parse(src, mode="eval").body.slice
        assert sharding._index_vars(index, loops) == (want, set())
