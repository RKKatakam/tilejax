"""A stack of MoE layers decoding in one kernel, the routing inside it (examples/moe_megakernel.py): against
a float64 NumPy stack, its loop nest and moe.py's per-layer decode layers; its plan; cross-lowering at real
sizes; natively on a v5e."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, run_checking_races, spec
from moe_megakernel import (
    TOP_K,
    Config,
    MoeTiles,
    init_params,
    layer_params,
    megakernel_layers,
    moe_megakernel,
    per_layer,
    xla_layers,
)

CONFIG = Config(layers=2, d_model=512, experts=16, top_k=TOP_K, ffn=256)
SMALL = MoeTiles(block_d=256, block_e=128)  # two router steps, two steps per expert


def _bf16(x):
    return f64(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16))


def numpy_stack(x, params, config):
    """The stack in float64, rounding where the kernel rounds to bf16 (the normalized rows, the hidden units,
    the output)."""
    h = f64(x)
    for layer in range(config.layers):
        normed = _bf16(h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1e-6) * f64(params["gamma"][layer]))
        logits = normed @ f64(params["router"][layer])
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        top = np.argsort(-scores, axis=1, kind="stable")[:, : config.top_k]
        weights = np.zeros_like(scores)
        np.put_along_axis(weights, top, np.take_along_axis(scores, top, axis=1), axis=1)
        weights /= weights.sum(axis=1, keepdims=True)
        acc = np.zeros_like(h)
        for e in np.flatnonzero(weights.max(axis=0) > 0):
            g = normed @ f64(params["w_gate"][layer, e])
            u = normed @ f64(params["w_up"][layer, e])
            hidden = _bf16(g / (1 + np.exp(-g)) * u)
            acc += weights[:, e : e + 1] * (hidden @ f64(params["w_down"][layer, e]))
        h = h + acc
    return h


def _inputs(tokens, config=CONFIG, seed=0):
    params = init_params(config, seed=seed)
    x = jax.random.normal(jax.random.key(seed + 1), (tokens, config.d_model), jnp.bfloat16)
    return x, params


@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_megakernel_matches_numpy_and_its_loop_nest(tokens):
    x, params = _inputs(tokens)
    args = (x, params["gamma"], params["router"], params["w_gate"], params["w_up"], params["w_down"])
    out = run_checking_races(moe_megakernel, *args, SMALL)
    want = numpy_stack(x, params, CONFIG)
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    assert moe_megakernel.verify(*args, SMALL) == 0.0


def test_megakernel_agrees_with_moe_py_per_layer_and_xla():
    x, params = _inputs(4)
    ours = jax.block_until_ready(megakernel_layers(x, params, SMALL))
    layers = layer_params(params, CONFIG, gate_up=True)
    for stack in (per_layer, xla_layers):
        theirs = jax.block_until_ready(jax.jit(stack, static_argnums=2)(x, layers, CONFIG))
        np.testing.assert_allclose(f64(ours), f64(theirs), atol=3e-2 * float(jnp.abs(theirs).max()))


def test_plan_routes_inside_the_kernel():
    x, params = _inputs(8)
    args = (x, params["gamma"], params["router"], params["w_gate"], params["w_up"], params["w_down"])
    lowered = moe_megakernel.lower(*args, SMALL)
    plan = lowered.plan
    assert [phase.range.var for phase in plan.phases] == ["d", "c"]
    assert [(c.name, c.shape, c.phase, c.consumers) for c in plan.computed] == [
        ("chosen", (16,), 0, (1,)),
        ("n", (1,), 0, (1,)),
    ]
    assert plan.phases[1].range.bounds is not None and lowered.schedule.cursor  # steps found as it runs
    kinds = {op.name: (op.kind.value, op.phase) for op in plan.operands}
    assert kinds["w_router"] == ("stream", 0) and kinds["w_gate"] == ("stream", 1)


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_megakernel_lowers_for_v5e(tokens):
    layers, d, e, f = 4, 2048, 128, 768  # Qwen3-30B-A3B's layers
    specs = (
        spec(tokens, d, dtype=jnp.bfloat16),
        spec(layers, 1, d, dtype=jnp.bfloat16),
        spec(layers, d, e),
        spec(layers, e, d, f, dtype=jnp.bfloat16),
        spec(layers, e, d, f, dtype=jnp.bfloat16),
        spec(layers, e, f, d, dtype=jnp.bfloat16),
    )
    lowered = moe_megakernel.lower(*specs)
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.tpu
@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_megakernel_natively_matches_the_per_layer_layers(tokens):
    config = Config(layers=3, d_model=2048, experts=128, top_k=TOP_K, ffn=768)
    x, params = _inputs(tokens, config)
    ours = megakernel_layers(x, params)
    theirs = per_layer(x, layer_params(params, config), config)
    np.testing.assert_allclose(f64(ours), f64(theirs), atol=5e-2 * float(jnp.abs(theirs).max()))
