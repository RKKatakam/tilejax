"""A stack of feed-forward layers in one kernel (a repeat loop over the layers), against NumPy."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, run_checking_races, spec
from layer_stack import StackTiles, ffn_layer, ffn_stack, layer_by_layer, make_inputs, plain_jax


def _stack(x, gamma, w_gate, w_up, w_down, eps=1e-6):
    h = f64(x)
    for layer in range(w_up.shape[0]):
        n = h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + eps) * f64(gamma[layer])
        gate, up = n @ f64(w_gate[layer]), n @ f64(w_up[layer])
        h = h + (gate / (1 + np.exp(-gate)) * up) @ f64(w_down[layer])
    return h


@pytest.mark.parametrize(("rows", "block_f"), [(128, 256), (64, 512)])
def test_ffn_stack(rows, block_f):
    inputs = make_inputs(128, 256, 512, 3)
    tiles = StackTiles(rows=rows, block_f=block_f)
    out = run_checking_races(ffn_stack, *inputs, tiles)
    want = _stack(*inputs)
    np.testing.assert_allclose(f64(out), want, atol=3e-2 * np.abs(want).max())
    ffn_stack.verify(*inputs, tiles)
    lowered = ffn_stack.lower(*inputs, tiles)
    assert [r.range.var for r in lowered.plan.repeats] == ["layer"] and set(lowered.plan.carried) == {"h", "n", "y"}
    np.testing.assert_allclose(f64(layer_by_layer(*inputs)), want, atol=3e-2 * np.abs(want).max())
    np.testing.assert_allclose(f64(plain_jax(*inputs)), want, atol=3e-2 * np.abs(want).max())
    run_checking_races(ffn_layer, inputs[0], inputs[1][0], *(w[0] for w in inputs[2:]), tiles)


@pytest.mark.parametrize(("tokens", "d", "f", "layers"), [(64, 2048, 8192, 8), (4096, 2048, 8192, 4)])
def test_ffn_stack_lowers_at_scale(tokens, d, f, layers):
    b16 = jnp.bfloat16
    specs = (spec(tokens, d, dtype=b16), spec(layers, 1, d, dtype=b16), spec(layers, d, f, dtype=b16))
    specs += (spec(layers, d, f, dtype=b16), spec(layers, f, d, dtype=b16))
    assert "tpu_custom_call" in lower_for_v5e(ffn_stack.lower(*specs), *specs)
