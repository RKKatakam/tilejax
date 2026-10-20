"""Repeat loops: a loop around reduction loops runs their steps iteration after iteration, carrying values
from one iteration to the next, and fetches the next iteration's inputs while this one's run."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

from autokernel import Config, KernelError, Pipeline
from autokernel.tune import _stats


def check(kernel, *inputs, pipeline=None):
    """Run `kernel` (with `pipeline` if given, else checking races), verify it, cross-lower it for a v5e."""
    if pipeline is None:
        out = run_checking_races(kernel, *inputs)
        lowered = kernel.lower(*inputs)
    else:
        lowered = kernel.lowered(kernel.signature(*inputs), Config(None, pipeline))
        out = lowered.run(*inputs)
    kernel.verify(*inputs)
    specs = [spec(*x.shape, dtype=x.dtype) for x in inputs]
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)
    return out


FFN_STACK = """
def ffn_stack(x, gamma, w_up, w_down):
    M, D = x.shape
    L, _, F = w_up.shape
    out = jnp.zeros((M, D), jnp.float32)
    for i in range(0, M, 128):
        h = x[i : i + 128, :]
        for l in range(L):  # a layer per iteration, its weights streamed
            n = h * gamma[l, 0:1, :]
            y = jnp.zeros((128, D))
            for f in range(0, F, 128):
                y += jnp.maximum(n @ w_up[l, :, f : f + 128], 0.0) @ w_down[l, f : f + 128, :]
            h = h + y * (l + 1)
        out = out.at[i : i + 128, :].set(h)
    return out
"""


def _ffn_stack(x, gamma, w_up, w_down):
    h = f64(x)
    for layer in range(w_up.shape[0]):
        n = h * f64(gamma[layer])
        h = h + (layer + 1) * np.maximum(n @ f64(w_up[layer]), 0) @ f64(w_down[layer])
    return h


def _ffn_inputs(layers=3):
    x, gamma = rand((256, 128)), 1 + 0.1 * rand((layers, 1, 128), 1)
    return x, gamma, rand((layers, 128, 256), 2) / 12, rand((layers, 256, 128), 3) / 16


@pytest.mark.parametrize("pipeline", [None, Pipeline(1, 1), Pipeline(4, 3)])
def test_layers_in_one_kernel(pipeline):
    kernel = make_kernel(FFN_STACK)
    inputs = _ffn_inputs()
    out = check(kernel, *inputs, pipeline=pipeline)
    np.testing.assert_allclose(f64(out), _ffn_stack(*inputs), rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(*inputs)
    plan, sch = lowered.plan, lowered.schedule
    assert [(r.range.var, r.first, r.last) for r in plan.repeats] == [("l", 0, 1)]
    assert set(plan.carried) == {"h", "n", "y"} and lowered.steps == 2 * 3 * 2
    # The layer's weights stream with its loop, and its norm is fetched once per layer: an iteration ahead.
    assert {op.name: (op.kind.value, op.phase) for op in plan.operands} == {
        "x": ("tile", None),
        "gamma": ("iteration", 0),  # fetched once per layer
        "w_up": ("stream", 0),
        "w_down": ("stream", 0),
    }
    assert [sch.iteration_at(s) for s in range(lowered.steps)] == [0, 0, 1, 1, 2, 2] * 2
    src = lowered.source
    ahead = lowered.pipeline.input_buffers - 1  # (tuned on a TPU)
    assert f"ahead_tile, ahead_ph, ahead_rb, ahead_kb, ahead_first, _ = step(s + {ahead})" in src
    # Copies the tuner counts: x and out per tile, gamma per layer of a tile, the weights per step.
    tile, layer, block = 128 * 128 * 4, 128 * 4, 2 * 128 * 128 * 4
    assert _stats(plan, sch).dma_bytes == 2 * tile + 2 * 3 * layer + 12 * block + 2 * tile
    assert "w_up_hbm.at[rb, :, pl.ds(kb * 128, 128)]" in src and "@pl.when(rb == 2)" in src


LAYERED = """
def f(x, w, limits):
    M, K = x.shape
    L = w.shape[0]
    out = jnp.zeros((M, 128), jnp.float32)
    for i in range(0, M, 128):
        a = jnp.zeros((128, 128))
        for k in range(0, K, 128):  # a phase before the repeat loop
            a += x[i : i + 128, k : k + 128]
        h = a * 0.5
        for l in range(L):
            y = jnp.zeros((128, 128))
            for f in range(0, 384, 128):  # layer l runs l + 1 blocks: a mask on the iteration
                if f > l * 128:
                    continue
                y += h @ w[l, f : f + 128, :]
            h = h + y * 0.1
            z = h * 0
            for g in range(0, 256, 128):  # a mask read when the kernel runs (empty for limits[l] == 0)
                if g >= limits[l]:
                    continue
                z += x[i : i + 128, g : g + 128] * (l + 1)
            h = h + z
        t = h * 2
        for c in range(0, 256, 128):  # a phase after it
            t = t + x[i : i + 128, c : c + 128]
        out = out.at[i : i + 128, :].set(t)
    return out
"""


def _layered(x, w, limits):
    x, w = f64(x), f64(w)
    out = []
    for i in range(0, x.shape[0], 128):
        rows = x[i : i + 128]
        blocks = [rows[:, k : k + 128] for k in range(0, rows.shape[1], 128)]
        h = sum(blocks) * 0.5
        for layer in range(w.shape[0]):
            y = sum(h @ w[layer, f : f + 128] for f in range(0, 128 * (layer + 1), 128) if f < 384)
            h = h + y * 0.1
            h = h + sum((layer + 1) * blocks[g // 128] for g in range(0, 256, 128) if g < limits[layer])
        out.append(2 * h + blocks[0] + blocks[1])
    return np.concatenate(out)


@pytest.mark.parametrize("limits", [(256, 0, 100), (0, 0, 0)])
def test_phases_around_and_in_a_repeat_loop_with_masks(limits):
    kernel = make_kernel(LAYERED)
    x, w, limits = rand((256, 256)), rand((3, 384, 128), 1) / 20, jnp.array(limits, jnp.int32)
    out = check(kernel, x, w, limits)
    np.testing.assert_allclose(f64(out), _layered(x, w, np.asarray(limits)), rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(x, w, limits)
    plan, sch = lowered.plan, lowered.schedule
    assert [p.range.var for p in plan.phases] == ["k", "f", "g", "c"] and sch.runtime and sch.iterations == 3
    assert [(r.first, r.last) for r in plan.repeats] == [(1, 3)] and plan.phases[1].repeat == 0
    # The masks don't read `i`: the tables hold one tile's steps, repeated.
    assert (
        sch.repeats == 2
        and sch.static_runs[1].shape == (3, 1, 3)
        and sch.static_runs[1][:, 0].tolist()
        == [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    assert lowered.steps == 2 * (2 + 3 * (3 + 2) + 2)  # at most
    src = lowered.source
    assert "l = np.arange(3)[:, None, None]" in src and "g = np.arange(0, 256, 128)[None, :]" in src
    assert "def _end_repeat2():" in src and "@pl.when(rb == 2)" in src


def test_two_repeat_loops_and_static_tables():
    kernel = make_kernel(
        """
        def f(x, w):
            out = jnp.zeros((128, 128), jnp.float32)
            for i in range(0, 128, 128):
                h = x[i : i + 128, :]
                for l in range(2):
                    for f in range(0, 256, 128):
                        if f > l * 128:
                            continue
                        h = h + w[l, f : f + 128, :]
                u = h * 1
                for m in range(2):
                    v = u * 2
                    for g in range(0, 256, 128):
                        v = v + w[m, g : g + 128, :] * (m + 1)
                    u = v
                out = out.at[i : i + 128, :].set(u)
            return out
        """
    )
    x, w = rand((128, 128)), rand((2, 256, 128), 1)
    out = check(kernel, x, w)
    xf, wf = f64(x), f64(w)
    h = xf + wf[0, :128] + wf[1, :128] + wf[1, 128:]
    for m in range(2):
        h = 2 * h + (m + 1) * (wf[m, :128] + wf[m, 128:])
    np.testing.assert_allclose(f64(out), h, rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(x, w)
    sch = lowered.schedule
    assert not sch.runtime and sch.iterations == 2 and lowered.steps == 1 + 2 + 2 * 2
    assert [sch.iteration_at(s) for s in range(lowered.steps)] == [0, 1, 1, 0, 0, 1, 1]
    assert "def _before_phase1():" in lowered.source  # the second loop's code before its phase


CHOSEN = """
def f(x, w, ids, n):
    M, K = x.shape
    C = ids.shape[0]
    out = jnp.zeros((M, K), jnp.float32)
    for i in range(0, M, 128):
        h = x[i : i + 128, :]
        acc = jnp.zeros((128, K))
        for c in range(C):  # the experts some row chose, then padding
            for f in range(0, K, K):
                if c >= n[0]:  # padding: an empty step, its expert's weights never fetched
                    continue
                acc += h @ w[ids[c], :, :] * (c + 1)
        out = out.at[i : i + 128, :].set(acc)
    return out
"""


@pytest.mark.parametrize("pipeline", [None, Pipeline(4, 2)])
def test_an_empty_iteration_skips_what_only_its_steps_read(pipeline):
    kernel = make_kernel(CHOSEN)
    x, w = rand((256, 128)), rand((6, 128, 128), 1) / 10
    ids, n = jnp.array([4, 1, 0, 0], jnp.int32), jnp.array([2], jnp.int32)
    out = check(kernel, x, w, ids, n, pipeline=pipeline)
    want = f64(x) @ f64(w[4]) + 2 * f64(x) @ f64(w[1])
    np.testing.assert_allclose(f64(out), want, rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(x, w, ids, n)
    [op] = [op for op in lowered.plan.operands if op.name == "w"]
    assert op.kind.value == "iteration" and op.in_body
    assert "jnp.logical_not" in lowered.source and "def _iteration_inputs0_body():" in lowered.source
    # A layer's norm, read before its phase's steps, is fetched for every iteration.
    plan = make_kernel(FFN_STACK).lower(*_ffn_inputs()).plan
    assert [op.in_body for op in plan.operands if op.name == "gamma"] == [False]


def test_a_repeat_loop_takes_a_step_per_iteration_at_least():
    # Its one reduction loop has one iteration (a whole layer's weights per step): it stays a phase.
    kernel = make_kernel(FFN_STACK.replace("range(0, F, 128)", "range(0, F, F)").replace("f + 128", "f + F"))
    inputs = _ffn_inputs()
    out = check(kernel, *inputs)
    np.testing.assert_allclose(f64(out), _ffn_stack(*inputs), rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(*inputs)
    assert [p.range.count for p in lowered.plan.phases] == [1] and lowered.steps == 2 * 3


def test_a_repeat_loop_of_one_iteration_runs_inline():
    kernel = make_kernel(FFN_STACK)
    inputs = _ffn_inputs(layers=1)
    out = check(kernel, *inputs)
    np.testing.assert_allclose(f64(out), _ffn_stack(*inputs), rtol=1e-4, atol=1e-3)
    lowered = kernel.lower(*inputs)
    assert lowered.plan.repeats == () and "\n        l = 0\n" in lowered.source  # bound at every step


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            def f(x):
                out = jnp.zeros((128, 128))
                for i in range(0, 128, 128):
                    h = x[i : i + 128, :]
                    for l in range(2):
                        for f in range(2):
                            h = h + 1
                    out = out.at[i : i + 128, :].set(h + l)
                return out
            """,
            "`l` is only defined inside its repeat loop",
        ),
        (
            """
            def f(x):
                out = jnp.zeros((128, 128))
                for i in range(0, 128, 128):
                    h = x[i : i + 128, :]
                    for k in range(2):
                        if l > 0:
                            continue
                        h = h + 1
                    for l in range(2):
                        for f in range(2):
                            h = h + 1
                    out = out.at[i : i + 128, :].set(h)
                return out
            """,
            "a mask reads its own and enclosing loops' variables, not `l`",
        ),
    ],
)
def test_repeat_loop_errors(source, message):
    with pytest.raises(KernelError, match=message):
        make_kernel(source).lower(spec(128, 128))
