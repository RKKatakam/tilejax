"""Outputs stored by every step of a reduction loop: a scan writes a tile per step, carrying its state."""

import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

from autokernel import Config, KernelError, Pipeline

SCAN = """
def scan(x, w):
    S, D = x.shape
    out = jnp.zeros((S, D), jnp.float32)
    last = jnp.zeros((8, D), jnp.float32)
    for j in range(0, D, 128):
        state = jnp.zeros((128, 128), jnp.float32)
        for i in range(0, S, 128):  # a chunk per step: its outputs stored by the step
            state = state * 0.5 + x[i : i + 128, j : j + 128] @ w[:, :]
            out = out.at[i : i + 128, j : j + 128].set(state)
        last = last.at[:, j : j + 128].set(state[0:8, :])
    return out, last
"""


def _scan(x, w):
    x, w = f64(x), f64(w)
    out = np.zeros_like(x)
    for j in range(0, x.shape[1], 128):
        state = np.zeros((128, 128))
        for i in range(0, x.shape[0], 128):
            state = state * 0.5 + x[i : i + 128, j : j + 128] @ w
            out[i : i + 128, j : j + 128] = state
    return out


@pytest.mark.parametrize("pipeline", [None, Pipeline(2, 1), Pipeline(3, 3)])
def test_a_scan_stores_a_tile_per_step(pipeline):
    kernel = make_kernel(SCAN)
    x, w = rand((512, 256)), rand((128, 128), 1) / 16
    if pipeline is None:
        out, last = run_checking_races(kernel, x, w)
        lowered = kernel.lower(x, w)
    else:
        lowered = kernel.lowered(kernel.signature(x, w), Config(None, pipeline))
        out, last = lowered.run(x, w)
    want = _scan(x, w)
    np.testing.assert_allclose(f64(out), want, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        f64(last), np.concatenate([want[-128:-120, j : j + 128] for j in (0, 128)], 1), rtol=1e-4, atol=1e-4
    )
    kernel.verify(x, w)
    assert [out.phase for out in lowered.plan.outputs] == [0, None]
    assert "tpu_custom_call" in lower_for_v5e(lowered, spec(512, 256), spec(128, 128))


def test_a_reduction_loop_of_one_step_that_stores_stays_a_phase():
    kernel = make_kernel(SCAN)
    x, w = rand((128, 256)), rand((128, 128), 1) / 16
    out, _ = run_checking_races(kernel, x, w)
    np.testing.assert_allclose(f64(out), _scan(x, w), rtol=1e-4, atol=1e-4)
    assert [p.range.var for p in kernel.lower(x, w).plan.phases] == ["i"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("for j in range(0, D, 128):", "for j in range(0, D, 128):\n        if j > 0:\n            continue"),
            "no mask may skip",
        ),
        (
            ("out = out.at[i : i + 128, j : j + 128]", "out = out.at[0:128, j : j + 128]"),
            "`i`'s steps each store each output tile",
        ),
    ],
)
def test_step_output_errors(change, message):
    with pytest.raises(KernelError, match=message):
        make_kernel(SCAN.replace(*change)).lower(spec(512, 256), spec(128, 128))
