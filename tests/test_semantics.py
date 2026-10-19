"""What `@` means, what state variables are, what `verify` compares, and the kernel's step split: cases the
compiler used to get wrong."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races

import autokernel
from autokernel import AutoKernelError, schedule
from autokernel import hardware as hardware_lib
from autokernel import tune as tune_lib
from autokernel.analysis import specialize

M, T = 512, 128
STEP_TILE = 2.0  # named like one of the kernel's SMEM tables, on purpose


@autokernel.tpu
def causal_scaled(x, y):
    out = jnp.zeros((M, M))
    for i in range(0, M, T):
        for j in range(0, M, T):
            acc = jnp.zeros((T, T))
            for k in range(0, M, T):
                if k > i:
                    continue
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc * STEP_TILE)
    return out


def test_user_constants_named_like_generated_tables():
    x, y = rand((M, M)), rand((M, M), 1)
    assert causal_scaled.lower(x, y).schedule == "table"  # the kernel has a STEP_TILE table of its own
    causal_scaled.verify(x, y)


def test_integer_index_and_one_wide_slice_are_different_operands():
    @autokernel.tpu
    def both_ways(x):
        out = jnp.zeros((4, 64, 128))
        for b in range(4):
            for i in range(0, 64, 8):
                a = x[b, i : i + 8, :]  # (8, 128)
                c = x[b : b + 1, i : i + 8, :]  # (1, 8, 128): the same window, seen with another rank
                out = out.at[b, i : i + 8, :].set(a + jnp.sum(c, axis=0))
        return out

    x = rand((4, 64, 128))
    ops = both_ways.lower(x).operands
    assert [op.block.kernel_shape for op in ops] == [(8, 128), (1, 8, 128)]
    np.testing.assert_allclose(both_ways(x), 2 * x)


def test_matmul_state_is_typed_like_the_kernel_computes_it():
    """`xw = x @ w` in bf16 is an f32 value in the kernel (MXU accumulation), so its scratch must be f32."""

    @autokernel.tpu
    def project_then_reduce(x, w, y):
        out = jnp.zeros((256, 256), jnp.bfloat16)
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                xw = x[i : i + 128, :] @ w[:, j : j + 128]  # per-tile projection, read inside the reduction
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += xw * jnp.sum(y[i : i + 128, k : k + 128], axis=1, keepdims=True)
                out = out.at[i : i + 128, j : j + 128].set(acc.astype(jnp.bfloat16))
        return out

    x, w, y = (rand(s, n, jnp.bfloat16) for n, s in enumerate([(256, 128), (128, 256), (256, 256)]))
    assert project_then_reduce.lower(x, w, y).state["xw"].aval.dtype == jnp.float32
    project_then_reduce.verify(x, w, y)


def test_state_initialised_from_a_python_literal():
    @autokernel.tpu
    def counted(x):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            n = 0  # an int literal, promoted to the f32 the updates need
            acc = jnp.zeros((128, 256))
            for k in range(0, 512, 128):
                n = n + 0.5
                acc += x[i : i + 128, k : k + 128] @ jnp.ones((128, 256))
            out = out.at[i : i + 128, :].set(acc / n)
        return out

    x = rand((256, 512))
    assert counted.lower(x).state["n"].aval.dtype == jnp.float32
    counted.verify(x)


def test_loops_without_explicit_start_or_step():
    @autokernel.tpu
    def scale(x):
        out = jnp.zeros((4, 128, 256))
        for b in range(4):
            for i in range(0, 128, 64):
                out = out.at[b, i : i + 64, :].set(x[b, i : i + 64, :] * 2)
        return out

    x = rand((4, 128, 256))
    assert scale.lower(x).steps == 8
    scale.verify(x)


# --------------------------------------------------------------------------- verify


def test_bf16_inputs_with_an_f32_output_verify():
    """Plain JAX would round every partial product to bf16; the reference accumulates like the kernel."""

    @autokernel.tpu
    def mm(x, y):
        out = jnp.zeros((M, M))
        for i in range(0, M, T):
            for j in range(0, M, T):
                acc = jnp.zeros((T, T))
                for k in range(0, M, T):
                    acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
                out = out.at[i : i + T, j : j + T].set(acc)
        return out

    x, y = rand((M, M), 0, jnp.bfloat16), rand((M, M), 1, jnp.bfloat16)
    assert mm.verify(x, y) < 1e-3
    exact = jnp.dot(x.astype(jnp.float32), y.astype(jnp.float32), precision="highest")
    np.testing.assert_allclose(mm.reference(x, y), exact, atol=1e-3)


def test_int8_matmul_accumulates_in_i32():
    @autokernel.tpu
    def mm8(a, b):
        c = jnp.zeros((256, 256), jnp.int32)
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                acc = jnp.zeros((128, 128), jnp.int32)
                for k in range(0, 256, 128):
                    acc += a[i : i + 128, k : k + 128] @ b[k : k + 128, j : j + 128]
                c = c.at[i : i + 128, j : j + 128].set(acc)
        return c

    rng = np.random.default_rng(0)
    a, b = (jnp.asarray(rng.integers(-100, 100, (256, 256)), jnp.int8) for _ in range(2))
    exact = np.asarray(a, np.int64) @ np.asarray(b, np.int64)
    assert np.abs(exact).max() > 2**15  # plain int8 arithmetic would wrap
    np.testing.assert_array_equal(mm8(a, b), exact)
    assert mm8.verify(a, b) == 0.0


def test_verify_option_runs_on_the_first_concrete_call(monkeypatch):
    @autokernel.tpu(verify=True)
    def double(x):
        out = jnp.zeros((64, 128))
        for i in range(0, 64, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * 2)
        return out

    checks = []
    monkeypatch.setattr(double, "_check", lambda *a, **k: checks.append(a) or 0.0)
    x = rand((64, 128))
    jax.jit(double)(x)  # traced: nothing concrete to compare
    assert checks == []
    double(x)
    double(x)
    assert len(checks) == 1


def test_measurement_happens_on_the_first_concrete_call(monkeypatch):
    """Lowering from specs first must not lock in the cost model's pick when `autotune="measure"`."""
    S = 512

    def mm(a, b):
        c = jnp.zeros((S, S))
        for i in range(0, S, TM):
            for j in range(0, S, TN):
                acc = jnp.zeros((TM, TN))
                for k in range(0, S, 128):
                    acc += a[i : i + TM, k : k + 128] @ b[k : k + 128, j : j + TN]
                c = c.at[i : i + TM, j : j + TN].set(acc)
        return c

    TM = TN = 128
    kern = autokernel.tpu(mm, tune={"TM": (128, 256), "TN": (128, 256)}, autotune="measure", interpret=True)
    specs = (jax.ShapeDtypeStruct((S, S), jnp.float32),) * 2
    model_pick = kern.lower(*specs)  # no arrays yet: the cost model's pick
    # Pretend to be on a TPU where every other candidate beats the model's pick.
    monkeypatch.setattr(hardware_lib, "on_tpu", lambda: True)
    monkeypatch.setattr(tune_lib, "time_kernel", lambda lowered, args: 1.0 if lowered is model_pick else 0.5)
    a, b = rand((S, S)), rand((S, S), 1)
    out = kern(a, b)
    measured_pick = kern.lower(a, b)
    assert measured_pick is not model_pick  # measure mode also tries 3 buffers, so the tiles may agree
    timed = [c for c in kern.tuning(a, b) if c.measured_s is not None]
    assert len(timed) == 4 + 1  # the four block sizes, then the winner's other buffer depth
    np.testing.assert_allclose(out, kern.reference(a, b), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(kern(a, b), out)  # recompiled for the new pick


# --------------------------------------------------------------------------- the two-core split


def test_two_cores_split_steps_not_tiles():
    """With a causal reduction mask, late tiles have more steps; the split must balance steps."""
    from test_attention import make_attention

    B, H, S, D = 1, 1, 1024, 128
    kern = make_attention(B, H, S, D, 128, 128, target="v5p")
    lw = kern.lower(*(jax.ShapeDtypeStruct((B, H, S, D), jnp.bfloat16),) * 3)
    per_core = [e - b for b, e in lw.core_steps]
    assert lw.cores == 2 and sum(per_core) == lw.steps == 36
    assert max(per_core) - min(per_core) <= 8  # within one tile's steps of even (was 10 vs 26)
    q, k, v = (rand((B, H, S, D), n, jnp.float32) for n in range(3))
    run_checking_races(kern, q, k, v)


def test_runtime_split_balances_steps():
    from test_generality import block_sparse_reference, make_block_sparse

    kern = make_block_sparse(512, 512, 256, 128, cores=2)
    x, y = rand((512, 512)), rand((512, 256), 1)
    mask = jnp.asarray(
        [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], jnp.int32
    )  # 8 tiles: 4,4,1,1,1,1,1,1 steps
    np.testing.assert_allclose(kern(x, y, mask), block_sparse_reference(x, y, mask, 128), atol=1e-3)
    spec = specialize(kern.nest, kern.fn, [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (x, y, mask)])
    sched = schedule.plan(spec, 2, hardware_lib.target("v5e").smem_budget)
    tables = schedule.runtime_tables(spec, sched, {"x": x, "y": y, "block_mask": mask})
    step_bounds = np.asarray(tables["bounds"][:3])
    assert step_bounds.tolist() == [0, 8, 14]  # 8 + 6 steps, instead of 10 + 4 from splitting the tiles evenly


# --------------------------------------------------------------------------- rejected code


def test_one_loop_tiling_two_output_dims_is_rejected():
    @autokernel.tpu
    def diagonal(x):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            out = out.at[i : i + 128, i : i + 128].set(x[i : i + 128, :128])
        return out

    with pytest.raises(AutoKernelError, match="tiles two dims"):
        diagonal.lower(rand((256, 256)))


def test_integer_index_expressions_are_explained():
    @autokernel.tpu
    def shifted(x):
        out = jnp.zeros((2, 128, 256))
        for b in range(2, 4):
            out = out.at[b - 2, :, :].set(x[b, :, :])
        return out

    with pytest.raises(AutoKernelError, match="must be a loop variable or a constant, not `b - 2`"):
        shifted.lower(rand((4, 128, 256)))


def test_tile_mask_on_an_outer_loop_is_explained():
    def outer_mask(x):
        out = jnp.zeros((4, 256, 256))
        for b in range(4):
            if b > 1:
                continue
            for i in range(0, 256, 128):
                out = out.at[b, i : i + 128, :].set(x[b, i : i + 128, :])
        return out

    with pytest.raises(AutoKernelError, match="tile mask goes on the innermost parallel loop"):
        autokernel.tpu(outer_mask)
