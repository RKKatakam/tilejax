"""Outputs updated in place: an output allocated as an input (`new = pool`) is that input's buffer
(`input_output_aliases`), keeping its values wherever no tile stores; such an output may be stored at
windows index arrays choose (a scatter: `new = new.at[slots[b], ...].set(...)`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

from autokernel import Config, KernelError, Pipeline, profiling, tune
from autokernel.profiling import KernelProfile
from autokernel.tune import _index_bounds, random_inputs

# A paged decode step's shape: each sequence reads its slot of a pool and writes it back updated.
SCATTER = """
def f(x, pool, slots):
    B, H, D = x.shape
    new = pool
    out = jnp.zeros((B, H, D))
    for b in range(B):
        for h in range(0, H, 8):
            old = pool[slots[b], h : h + 8, :]
            out = out.at[b, h : h + 8, :].set(old * 2)
            new = new.at[slots[b], h : h + 8, :].set(old + x[b, h : h + 8, :])
    return out, new
"""


def _scatter(x, pool, slots):
    pool, x = f64(pool), f64(x)
    out, new = np.zeros(x.shape), pool.copy()
    for b, s in enumerate(np.asarray(slots)):
        out[b], new[s] = 2 * pool[s], pool[s] + x[b]
    return out, new


def i32(*values):
    return jnp.array(values, jnp.int32)


@pytest.mark.parametrize("pipeline", [None, Pipeline(2, 1), Pipeline(4, 3)])
def test_a_scatter_into_an_input_updated_in_place(pipeline):
    kernel = make_kernel(SCATTER)
    x, pool, slots = rand((3, 16, 128)), rand((7, 16, 128), 1), i32(5, 0, 2)
    if pipeline is None:
        out, new = run_checking_races(kernel, x, pool, slots)
        lowered = kernel.lower(x, pool, slots)
    else:
        lowered = kernel.lowered(kernel.signature(x, pool, slots), Config(None, pipeline))
        out, new = lowered.run(x, pool, slots)
    want_out, want_new = _scatter(x, pool, slots)
    np.testing.assert_allclose(f64(out), want_out, rtol=1e-6)
    np.testing.assert_allclose(f64(new), want_new, rtol=1e-6)  # slots 1, 3, 4, 6 as they were
    kernel.verify(x, pool, slots)
    src = lowered.source
    assert "input_output_aliases={2: 1},  # outputs updated in place" in src  # (after the SMEM index array)
    assert "new_hbm.at[jnp.clip(slots_smem[b_blk], 0, 6), pl.ds(h_blk * 8, 8), :]" in src
    assert lowered.plan.outputs[1].aliases == 1 and lowered.plan.outputs[1].fill is None
    assert "tpu_custom_call" in lower_for_v5e(lowered, spec(3, 16, 128), spec(7, 16, 128), spec(3, dtype=jnp.int32))


def test_in_place_under_jit_with_the_pool_donated():
    kernel = make_kernel(SCATTER)
    x, pool, slots = rand((3, 16, 128)), rand((7, 16, 128), 1), i32(1, 4, 6)
    want = _scatter(x, pool, slots)[1]
    step = jax.jit(lambda x, pool, slots: kernel(x, pool, slots)[1], donate_argnums=1)
    new = step(x, jnp.copy(pool), slots)
    np.testing.assert_allclose(f64(new), want, rtol=1e-6)


def test_masked_tiles_of_an_input_updated_in_place_keep_its_values():
    # Two outputs: `out` (zeros) has its skipped tiles filled, `new` (in place) keeps the input's there.
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((256, 256))
            new = x
            for i in range(0, 256, 128):
                for j in range(0, 256, 128):
                    if j > i:
                        continue
                    out = out.at[i : i + 128, j : j + 128].set(x[i : i + 128, j : j + 128] + 1)
                    new = new.at[i : i + 128, j : j + 128].set(x[i : i + 128, j : j + 128] * 3)
            return out, new
        """
    )
    x = rand((256, 256))
    out, new = run_checking_races(kernel, x)
    keep = np.kron(np.tril(np.ones((2, 2))), np.ones((128, 128))).astype(bool)
    np.testing.assert_allclose(f64(out), np.where(keep, f64(x) + 1, 0.0), rtol=1e-6)
    np.testing.assert_allclose(f64(new), np.where(keep, f64(x) * 3, f64(x)), rtol=1e-6)
    lowered = kernel.lower(x)
    assert lowered.schedule.fills == 1 and "out_fill_copy" in lowered.source and "new_fill" not in lowered.source
    kernel.verify(x)


def test_only_in_place_outputs_fill_nothing():
    kernel = make_kernel(
        """
        def f(x):
            new = x
            for i in range(0, 256, 128):
                if i > 0:
                    continue
                new = new.at[i : i + 128, :].set(x[i : i + 128, :] * 3)
            return new
        """
    )
    x = rand((256, 128))
    new = run_checking_races(kernel, x)
    np.testing.assert_allclose(f64(new), np.concatenate([f64(x[:128]) * 3, f64(x[128:])]), rtol=1e-6)
    assert "_fill" not in kernel.source(x)


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            "out = jnp.zeros((7, 8, 128))\n    for b in range(3):\n"
            "        out = out.at[slots[b], :, :].set(x[b, :, :])",
            "an index array places this tile, so no tile might store the rest of `out`: allocate it as an input "
            "updated in place",
        ),
        (
            "out = slots\n    for b in range(3):\n        out = out.at[b].set(slots[b])",
            "output `out` updates `slots` in place, but `slots` is an index array",
        ),
        ("out = x\n    for b in range(3):\n        out = out.at[b, :, :].set(x[b, :, :])", None),
        (
            "out = wrong\n    for b in range(3):\n        out = out.at[b, :, :].set(x[b, :, :])",
            "allocate output `out` with jnp.zeros(shape, dtype) or jnp.full(shape, value, dtype), or as an input "
            "updated in place (`out = x`)",
        ),
    ],
)
def test_in_place_errors(code, message):
    kernel = make_kernel(f"def f(x, slots):\n    {code}\n    return out\n", wrong=3)
    x, slots = rand((3, 8, 128)), i32(0, 1, 2)
    if message is None:
        np.testing.assert_allclose(f64(run_checking_races(kernel, x, slots)), f64(x))
        return
    with pytest.raises(KernelError, match=message.replace("(", r"\(").replace(")", r"\)")):
        kernel.lower(x, slots)


def test_one_output_updates_an_input_in_place():
    kernel = make_kernel(
        """
        def f(x):
            a = x
            b = x
            for i in range(0, 256, 128):
                a = a.at[i : i + 128, :].set(x[i : i + 128, :])
                b = b.at[i : i + 128, :].set(x[i : i + 128, :])
            return a, b
        """
    )
    with pytest.raises(KernelError, match="two outputs update `x` in place: one may"):
        kernel.lower(rand((256, 128)))


def test_random_index_arrays_that_place_stores_hold_distinct_values():
    kernel = make_kernel(SCATTER)
    signature = kernel.signature(rand((3, 16, 128)), rand((7, 16, 128)), i32(0, 1, 2))
    bounds, distinct = _index_bounds(kernel, signature, kernel.default_config(signature))
    assert bounds == {2: 6} and distinct == {2}  # the pool's last slot (the loops go further: 16)
    for seed in range(5):
        slots = np.asarray(random_inputs(signature, bounds, seed, distinct)[2])
        assert len(set(slots.tolist())) == 3 and slots.min() >= 0 and slots.max() <= 6
    many = random_inputs(((((8,), np.dtype(np.int32))),), {0: 3}, distinct=frozenset({0}))[0]
    assert many.shape == (8,) and int(many.min()) >= 0 and int(many.max()) <= 3  # can't be distinct


@pytest.mark.cpu
def test_tuning_a_traced_scatter(monkeypatch):
    monkeypatch.setattr(tune, "_can_time", lambda: True)
    monkeypatch.setattr(profiling, "measure", lambda lowered, inputs, runs: KernelProfile("f", lowered.config, 1, 1e-3))
    kernel = make_kernel(SCATTER)
    x, pool, slots = rand((3, 16, 128)), rand((7, 16, 128), 1), i32(5, 0, 2)
    _, new = jax.jit(kernel)(x, pool, slots)  # tunes on random inputs: distinct slots, so no races
    assert all(t.error is None for t in kernel.tuning(x, pool, slots).trials)
    np.testing.assert_allclose(f64(new), _scatter(x, pool, slots)[1], rtol=1e-6)
