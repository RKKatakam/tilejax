"""Computed index arrays: values the tile code computes (int32) whose elements place later phases' windows
and bound their loops, made and read as the kernel runs (routing inside a kernel); and runtime ranges in
repeat loops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

from autokernel import Config, KernelError, Pipeline

E, K = 8, 2  # experts, chosen per layer at most


def route(logits):
    """Plain JAX (Mosaic-friendly): the K experts with the highest summed logits over the tokens, how many of
    them to use (the second only if its score is positive), and each token's weight for each, (K, T, 1)."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, logits.shape[1]), 1)
    score = jnp.where(lanes < E, jnp.sum(logits, axis=0, keepdims=True), -jnp.inf)
    chosen, weights, scores = [], [], []
    for _ in range(K):
        best = jnp.max(score, axis=1, keepdims=True)
        pick = jnp.min(jnp.where(score == best, lanes, logits.shape[1]), axis=1, keepdims=True)  # (1, 1)
        chosen.append(pick)
        scores.append(best)
        weights.append(jax.nn.sigmoid(jnp.sum(jnp.where(lanes == pick, logits, 0.0), axis=1, keepdims=True)))
        score = jnp.where(lanes == pick, -jnp.inf, score)
    ids = jnp.concatenate(chosen, axis=1).reshape(K).astype(jnp.int32)
    n = (1 + (scores[1] > 0).astype(jnp.int32)).reshape(1)
    return ids, n, jnp.stack(weights)


MOE = """
def f(x, router, w):
    T, D = x.shape
    L = w.shape[0]
    out = jnp.zeros((T, D))
    for i in range(0, T, T):
        h = x[i : i + T, :]
        for layer in range(L):
            blocks = jnp.stack([h[:, 0:128], h[:, 128:256]])
            logits = jnp.zeros((T, 128))
            for d in range(0, D, 128):  # the router, a block of rows a step
                logits += blocks[d // 128] @ router[layer, d : d + 128, :]
            chosen, n, weights = route(logits)  # made at the router's last step
            acc = jnp.zeros((T, D))
            for c in range(0, n[0] * D, D):  # the chosen experts: as many as the routing says
                acc += weights[c // D] * (h @ w[layer, chosen[c // D], :, :])
            h = h + acc
        out = out.at[i : i + T, :].set(h)
    return out
"""


def numpy_moe(x, router, w):
    h = f64(x)
    for layer in range(w.shape[0]):
        logits = h @ f64(router[layer])
        ids, n, weights = (np.asarray(a) for a in route(jnp.asarray(logits, jnp.float32)))
        h = h + sum(f64(weights[k]) * (h @ f64(w[layer, ids[k]])) for k in range(int(n[0])))
    return h


def moe_inputs(seed=0):
    x = rand((8, 256), seed)
    router = rand((3, 256, 128), seed + 1) * 0.1
    w = rand((3, E, 256, 256), seed + 2) * 0.05
    return x, router, w


@pytest.mark.parametrize("pipeline", [None, Pipeline(2, 1), Pipeline(4, 1)])
def test_routing_inside_the_kernel_places_and_counts_later_steps(pipeline):
    kernel = make_kernel(MOE, route=route)
    x, router, w = moe_inputs()
    if pipeline is None:
        out = run_checking_races(kernel, x, router, w)
        lowered = kernel.lower(x, router, w)
    else:
        lowered = kernel.lowered(kernel.signature(x, router, w), Config(None, pipeline))
        out = lowered.run(x, router, w)
    np.testing.assert_allclose(f64(out), numpy_moe(x, router, w), rtol=2e-4, atol=2e-4)
    assert kernel.verify(x, router, w) < 1e-4
    plan = lowered.plan
    assert [(c.name, c.shape, c.phase, c.repeat, c.consumers) for c in plan.computed] == [
        ("chosen", (K,), 0, 0, (1,)),
        ("n", (1,), 0, 0, (1,)),
    ]
    src = lowered.source
    assert "chosen_smem[0, jnp.clip(" in src and "counters[1] >= " in src and "def _catch_up0():" in src
    specs = [spec(*a.shape) for a in (x, router, w)]
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


def test_layers_route_differently():
    x, router, w = moe_inputs(seed=3)
    counts = []
    h = f64(x)
    for layer in range(3):
        logits = h @ f64(router[layer])
        _, n, _ = route(jnp.asarray(logits, jnp.float32))
        counts.append(int(n[0]))
        h = numpy_moe(x, router[: layer + 1], w[: layer + 1])
    assert len(set(counts)) > 1 or counts == [2, 2, 2]  # (the inputs exercise a count that changes or not)
    out = run_checking_races(make_kernel(MOE, route=route), x, router, w)
    np.testing.assert_allclose(f64(out), numpy_moe(x, router, w), rtol=2e-4, atol=2e-4)


OFFSETS = (0, 2)  # (a module constant read in a bound)


def test_indices_made_once_per_tile_and_constants_in_bounds():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((16, 128))
            for i in range(0, 16, 8):
                acc = jnp.zeros((8, 128))
                for d in range(0, 256, 128):
                    acc += x[0, i : i + 8, d : d + 128]
                n = (jnp.sum(acc > 0) % 3).astype(jnp.int32).reshape(1)  # made once per tile
                for c in range(1, n[0] + OFFSETS[1] + 1):
                    acc += x[c, i : i + 8, 0:128]
                out = out.at[i : i + 8, :].set(acc)
            return out
        """,
        OFFSETS=OFFSETS,
    )
    x = rand((5, 16, 256))
    out = run_checking_races(kernel, x)
    xs = f64(x)
    for i in range(0, 16, 8):
        acc = xs[0, i : i + 8, :128] + xs[0, i : i + 8, 128:]
        n = int(np.sum(np.asarray(jnp.asarray(acc, jnp.float32)) > 0) % 3)
        want = acc + sum(xs[1 + c, i : i + 8, :128] for c in range(n + 2))
        np.testing.assert_allclose(f64(out[i : i + 8]), want, rtol=1e-5, atol=1e-5)
    assert [c.repeat for c in kernel.lower(x).plan.computed] == [None]


def test_a_runtime_range_in_a_repeat_loop():
    kernel = make_kernel(
        """
        def f(x, n):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                acc = jnp.zeros((8, 128))
                for j in range(n[0]):
                    acc += x[i : i + 8, 0:128]
                for l in range(3):
                    for k in range(0, n[1 + l], 128):
                        acc += x[i : i + 8, k : k + 128] * (l + 1)
                out = out.at[i : i + 8, :].set(acc)
            return out
        """
    )
    x, n = rand((8, 384)), jnp.array([2, 384, 0, 256], jnp.int32)
    out = run_checking_races(kernel, x, n)
    xs = f64(x)
    want = 2 * xs[:, :128] + 1 * (xs[:, :128] + xs[:, 128:256] + xs[:, 256:]) + 3 * (xs[:, :128] + xs[:, 128:256])
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-5)
    assert kernel.lower(x, n).schedule.cursor


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (  # made in the code before the first phase: no earlier step to learn it at
            "ids = jnp.zeros((2,), jnp.int32)\n        acc = jnp.zeros((8, 128))\n"
            "        for c in range(2):\n            acc += x[ids[c], :, :]",
            "it is computed after a reduction loop",
        ),
        (  # read by a mask
            "acc = jnp.zeros((8, 128))\n        for d in range(2):\n            acc += x[d, :, :]\n"
            "        n = jnp.ones((1,), jnp.int32)\n        for c in range(2):\n            if c >= n[0]:\n"
            "                continue\n            acc += x[c, :, :]",
            "a mask can't read values the tile code computes",
        ),
        (  # not int32
            "acc = jnp.zeros((8, 128))\n        for d in range(2):\n            acc += x[d, :, :]\n"
            "        ids = jnp.zeros((2,), jnp.float32)\n"
            "        for c in range(2):\n            acc += x[ids[c], :, :]",
            "it must be int32 of 1 or 2 dims, not float32",
        ),
        (  # read by a window no reduction loop moves
            "acc = jnp.zeros((8, 128))\n        for d in range(2):\n            acc += x[d, :, :]\n"
            "        ids = jnp.zeros((2,), jnp.int32)\n        acc = acc + x[ids[0], :, :]",
            "only windows a reduction loop moves may be placed by `ids`",
        ),
        (  # assigned twice
            "acc = jnp.zeros((8, 128))\n        for d in range(2):\n            acc += x[d, :, :]\n"
            "        ids = jnp.zeros((2,), jnp.int32)\n        ids = ids + 1\n"
            "        for c in range(2):\n            acc += x[ids[c], :, :]",
            "assign it once",
        ),
        (  # made in each iteration of a repeat loop, read after the loop
            "acc = jnp.zeros((8, 128))\n        for l in range(2):\n            for d in range(2):\n"
            "                acc += x[d, :, :]\n            ids = jnp.zeros((2,), jnp.int32) + l\n"
            "        for c in range(2):\n            acc += x[ids[c], :, :]",
            "only later loops in the same iteration of its repeat loop may read it",
        ),
    ],
)
def test_computed_index_errors(source, message):
    kernel = make_kernel(
        f"def f(x):\n    out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        {source}\n"
        "        out = out.at[i : i + 8, :].set(acc)\n    return out\n"
    )
    with pytest.raises(KernelError, match=message):
        kernel.lower(rand((2, 8, 128)))


@pytest.mark.tpu
def test_routing_inside_the_kernel_natively():
    kernel = make_kernel(MOE, route=route)
    x, router, w = moe_inputs()
    out = kernel(x, router, w)
    want = numpy_moe(x, router, w)
    np.testing.assert_allclose(f64(out), want, rtol=1e-3, atol=1e-3 * float(np.abs(want).max()))
