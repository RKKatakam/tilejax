"""Index arrays: inputs read element by element, in SMEM. Their elements are values in tile code, conditions
in reduction masks (evaluated on the device before the kernel runs) and indices of tile reads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

from autokernel import Config, KernelError, Pipeline, profiling, tune
from autokernel.profiling import KernelProfile


def check(kernel, *inputs, pipeline=None):
    """Run `kernel` checking races (with `pipeline`, if given), verify it, and cross-lower it for a v5e."""
    if pipeline is not None:
        lowered = kernel.lowered(kernel.signature(*inputs), Config(None, pipeline))
        out = lowered.run(*inputs)
    else:
        out = run_checking_races(kernel, *inputs)
        lowered = kernel.lower(*inputs)
    kernel.verify(*inputs)
    specs = [spec(*x.shape, dtype=x.dtype) for x in inputs]
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)
    return out


def i32(*values):
    return jnp.array(values, jnp.int32)


DECODE = """
def f(cache, lengths):
    B, S, D = cache.shape
    out = jnp.zeros((B, 8, D))
    for b in range(B):
        acc = jnp.zeros((8, D))
        for j in range(0, S, 128):
            if j >= lengths[b]:  # cache blocks past this sequence's end: decided when the kernel runs
                continue
            rows = j + jax.lax.broadcasted_iota(jnp.int32, (128, D), 0)
            block = jnp.where(rows < lengths[b], cache[b, j : j + 128, :], 0.0)
            acc += jnp.sum(block.reshape(16, 8, D), axis=0)
        out = out.at[b, :, :].set(acc)
    return out
"""


def _decode(cache, lengths):
    out = np.zeros((cache.shape[0], 8, cache.shape[2]))
    for b, n in enumerate(np.asarray(lengths)):
        rows = np.concatenate([f64(cache[b, :n]), np.zeros((-n % 8, cache.shape[2]))])
        out[b] = rows.reshape(-1, 8, cache.shape[2]).sum(0)
    return out


@pytest.mark.parametrize("pipeline", [None, Pipeline(4, 3), Pipeline(1, 1)])
def test_runtime_masks_skip_the_steps_the_index_arrays_say(pipeline):
    kernel = make_kernel(DECODE)
    cache, lengths = rand((4, 512, 128)), i32(0, 1, 300, 512)  # an empty sequence takes one empty step
    out = check(kernel, cache, lengths, pipeline=pipeline)
    np.testing.assert_allclose(f64(out), _decode(cache, lengths), rtol=1e-5, atol=1e-4)
    lowered = kernel.lower(cache, lengths)
    sch = lowered.schedule
    assert sch.runtime and sch.empty and lowered.steps == 16  # at most
    assert [index.name for index in lowered.plan.indices] == ["lengths"] and lowered.plan.runtime
    src = lowered.source
    assert "def f_steps(lengths):" in src and "jnp.logical_not(j >= lengths[b])" in src
    assert "@pl.loop(0, n_steps)" in src and "lengths_smem[b]" in src and "def f(cache, lengths, *" in src
    tables = jax.jit(lowered.entry.__globals__["f_steps"])(lengths)
    assert int(tables[2][0]) == 1 + 1 + 3 + 4  # steps that run (the empty one included)
    np.testing.assert_allclose(f64(kernel.fn(cache, lengths)), _decode(cache, lengths), rtol=1e-5, atol=1e-4)


def test_a_runtime_mask_under_jit_and_tables_repeated_over_outer_loops():
    # The mask reads `b`, not `h`: the tables hold one head's steps, repeated per head.
    kernel = make_kernel(
        """
        def f(cache, lengths):
            H, B, S, D = cache.shape
            out = jnp.zeros((H, B, 8, D))
            for h in range(H):
                for b in range(B):
                    acc = jnp.zeros((8, D))
                    for j in range(0, S, 128):
                        if j >= lengths[b] or not (0 <= j < S) or -j > 0:
                            continue
                        acc += jnp.sum(cache[h, b, j : j + 128, :].reshape(16, 8, D), axis=0)
                    out = out.at[h, b, :, :].set(acc)
            return out
        """
    )
    cache, lengths = rand((3, 2, 384, 128)), i32(130, 384)
    out = jax.jit(kernel)(cache, lengths)
    want = np.stack(
        [
            [f64(cache[h, b, : -(-int(n) // 128) * 128]).reshape(-1, 8, 128).sum(0) for b, n in enumerate(lengths)]
            for h in range(3)
        ]
    )
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    check(kernel, cache, lengths)
    lowered = kernel.lower(cache, lengths)
    assert lowered.schedule.repeats == 3 and "jax.lax.div(s, step_count[0])" in lowered.source
    vectorized = "jnp.logical_not(jnp.logical_and(0 <= j, j < S))), -j > 0)"
    assert "jnp.logical_or(jnp.logical_or(j >= lengths[b], " + vectorized in lowered.source


GMM = """
def gmm(x, w, starts, ends):
    M, K = x.shape
    E, _, N = w.shape
    out = jnp.zeros((M, N), jnp.float32)
    for j in range(0, N, 128):
        for i in range(0, M, 128):
            acc = jnp.zeros((128, 128), jnp.float32)
            for e in range(E):  # only the experts whose rows meet this tile's
                if ends[e] <= i or starts[e] >= i + 128:
                    continue
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (128, K), 0)
                mine = (rows >= starts[e]) & (rows < ends[e])
                acc += jnp.where(mine, x[i : i + 128, :], 0.0) @ w[e, :, j : j + 128]
            out = out.at[i : i + 128, j : j + 128].set(acc)
    return out
"""


def test_a_grouped_matmul_over_the_experts_each_tile_meets():
    kernel = make_kernel(GMM)
    x, w = rand((512, 256)), rand((4, 256, 256), 1)
    sizes = np.array([100, 0, 300, 112])
    ends = np.cumsum(sizes)
    starts, ends = i32(*(ends - sizes)), i32(*ends)
    out = check(kernel, x, w, starts, ends)
    want = np.zeros((512, 256))
    for e in range(4):
        want[starts[e] : ends[e]] = f64(x[starts[e] : ends[e]]) @ f64(w[e])
    np.testing.assert_allclose(f64(out), want, rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(x, w, starts, ends)
    assert lowered.schedule.repeats == 2 and lowered.steps == 2 * 4 * 4


def test_runtime_windows_index_tiles_by_elements_of_index_arrays():
    kernel = make_kernel(
        """
        def f(x, w, ids):
            T, S, _, K = x.shape
            N = w.shape[2]
            out = jnp.zeros((T, 128, N))
            for t in range(T):
                acc = jnp.zeros((128, N))
                for s in range(S):
                    acc += x[t, s, 0:128, :] @ w[ids[t, s], :, :]
                out = out.at[t, :, :].set(acc + w[ids[t, 0], 0:8, :].sum() + w[ids[0, 1], 0:8, :].sum())
            return out
        """
    )
    x, w = rand((3, 2, 128, 128)), rand((5, 128, 256), 1)
    ids = jnp.array([[4, 0], [1, 1], [2, 9]], jnp.int32)  # 9 is past the 5 experts: clamped, as JAX clamps
    out = check(kernel, x, w, ids)
    clamped = np.clip(np.asarray(ids), 0, 4)
    extra = f64(w[clamped[0, 1], :8]).sum()
    want = [
        sum(f64(x[t, s]) @ f64(w[clamped[t, s]]) for s in range(2)) + f64(w[clamped[t, 0], :8]).sum() + extra
        for t in range(3)
    ]
    np.testing.assert_allclose(f64(out), np.stack(want), rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(x, w, ids)
    assert {op.name: op.kind.value for op in lowered.plan.operands} == {
        "x": "stream",
        "w": "stream",
        "w2": "tile",
        "w3": "once",
    }
    assert "w_hbm.at[jnp.clip(ids_smem[t_blk, kb], 0, 4), :, :]" in lowered.source


def test_runtime_masks_with_static_and_unmasked_phases():
    kernel = make_kernel(
        """
        def f(x, lengths):
            out = jnp.zeros((512, 128))
            for i in range(0, 512, 128):
                a = jnp.zeros((128, 128))
                for k in range(0, 512, 128):
                    if k > i:
                        continue
                    a += x[i : i + 128, k : k + 128]
                b = a * 1
                for k2 in range(0, 512, 128):
                    if k2 >= lengths[i // 128] if i > 0 else k2 > 0:
                        continue
                    b += x[i : i + 128, k2 : k2 + 128] * 2
                for k3 in range(0, 256, 128):
                    b += x[i : i + 128, k3 : k3 + 128]
                out = out.at[i : i + 128, :].set(b)
            return out
        """
    )
    x, lengths = rand((512, 512)), i32(0, 0, 200, 512)
    out = check(kernel, x, lengths)
    blocks = [f64(x[:, k : k + 128]) for k in range(0, 512, 128)]
    want = np.zeros((512, 128))
    for t, n in enumerate([128, 0, 200, 512]):
        rows = slice(t * 128, (t + 1) * 128)
        a = sum(blocks[k][rows] for k in range(t + 1))
        b = a + sum(2 * blocks[k][rows] for k in range(-(-n // 128))) + blocks[0][rows] + blocks[1][rows]
        want[rows] = b
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    sch = kernel.lower(x, lengths).schedule
    assert list(sch.static_runs) == [0] and sch.static_runs[0].tolist()[1] == [True, True, False, False]
    assert "PHASE0_RUNS" in kernel.source(x, lengths) and "np.ones((4, 2), bool)" in kernel.source(x, lengths)


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("a = ids[0] + ids[0:8, :].sum()", "read element by element (an index array, kept in SMEM) and as tiles"),
        ("a = ids[ids[0, 0], 0]", "elements are indexed by loop variables and constants"),
        ("a = ids[0]", "index every one of ids's 2 dims"),
        ("a = ids[i // 128 + 8, 0]", "reads element 9 of a dim of 9 at i=128"),
    ],
)
def test_index_array_errors(code, message):
    kernel = make_kernel(
        f"""
        def f(x, ids):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                {code}
                out = out.at[i : i + 128, :].set(x[i : i + 128, :] + a)
            return out
        """
    )
    with pytest.raises(KernelError, match=message.replace("(", r"\(").replace(")", r"\)")):
        kernel.lower(rand((256, 128)), jnp.zeros((9, 128), jnp.int32))


@pytest.mark.parametrize(
    ("tile_mask", "step_mask", "message"),
    [
        ("lengths[i // 128] > 0", "False", "masks are evaluated when the kernel is built, on loop variables and"),
        ("False", "x[0:8, :].sum() > 0", "a reduction loop's mask reads loop variables, constants and elements of"),
        ("False", "lengths.shape[0] > 9", "not `lengths` itself"),
    ],
)
def test_mask_errors(tile_mask, step_mask, message):
    with pytest.raises(KernelError, match=message):
        make_kernel(
            f"""
            def f(x, lengths):
                out = jnp.zeros((256, 128))
                for i in range(0, 256, 128):
                    if {tile_mask}:
                        continue
                    acc = jnp.zeros((128, 128))
                    for k in range(0, 256, 128):
                        if {step_mask}:
                            continue
                        acc += x[i : i + 128, k : k + 128]
                    out = out.at[i : i + 128, :].set(acc)
                return out
            """
        )


def test_index_arrays_and_tables_must_fit_in_smem():
    kernel = make_kernel(DECODE)
    with pytest.raises(KernelError, match="step tables and index arrays take"):
        kernel.lower(spec(2, 128, 128), spec(300_000, dtype=jnp.int32))
    with pytest.raises(KernelError, match="step tables and index arrays take"):
        kernel.lower(spec(4096, 4096, 128), spec(4096, dtype=jnp.int32))  # 32768 steps could run


@pytest.mark.cpu
def test_tuning_runs_on_the_callers_inputs(monkeypatch):
    seen = []

    def measure(lowered, inputs, runs):
        seen.append([np.asarray(x) for x in inputs])
        return KernelProfile(name="f", config=lowered.config, calls=1, device_s=1e-3, steps=lowered.steps)

    monkeypatch.setattr(tune, "_can_time", lambda: True)
    monkeypatch.setattr(profiling, "measure", measure)
    cache, lengths = rand((4, 512, 128)), i32(0, 1, 300, 512)
    kernel = make_kernel(DECODE)
    np.testing.assert_allclose(f64(kernel(cache, lengths)), _decode(cache, lengths), rtol=1e-5, atol=1e-4)
    assert (seen[0][1] == np.asarray(lengths)).all()  # the lengths given, not random ones
    traced = make_kernel(DECODE)
    jax.jit(traced)(cache, lengths)  # traced: tuned on random inputs of these shapes
    assert len(seen) == 2 and seen[1][1].shape == (4,)
