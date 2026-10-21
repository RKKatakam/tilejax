"""A whole MoE decode step in one kernel (examples/moe_decode_megakernel.py): attention, routing inside the
kernel, the chosen experts and the LM head, against its loop nest and the same step with kernels per layer
(XLA routing between them), over chained steps; cross-lowering at real sizes; natively on a v5e."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decode_megakernel import layer_caches, random_caches
from helpers import f64, lower_for_v5e, run_checking_races
from moe_decode_megakernel import (
    Config,
    MoeMegaTiles,
    init_params,
    kernel_inputs,
    layer_weights,
    megakernel_step,
    moe_decode_megakernel,
    per_layer_step,
)

CONFIG = Config(layers=2, d_model=512, heads=4, kv_heads=2, experts=16, ffn=256, vocab=512)
SMALL = MoeMegaTiles(block_d=256, block_kv=128, block_a=256, block_r=256, block_e=128, block_v=256)


def _inputs(lengths, config=CONFIG, max_len=256, seed=0):
    params = init_params(config, seed=seed)
    lengths = jnp.asarray(lengths, jnp.int32)
    k_cache, v_cache = random_caches(config, lengths, max_len, seed=seed + 1)
    tokens = jax.random.randint(jax.random.key(seed + 2), (len(lengths),), 0, config.vocab)
    return tokens, k_cache, v_cache, lengths, params


def _close(ours, theirs):
    np.testing.assert_allclose(f64(ours), f64(theirs), atol=4e-2 * float(jnp.abs(theirs).max()))


def test_megakernel_matches_its_loop_nest():
    tokens, k_cache, v_cache, lengths, params = _inputs([0, 1, 130, 250])
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    run_checking_races(moe_decode_megakernel, *inputs, SMALL)
    assert moe_decode_megakernel.verify(*inputs, SMALL) == 0.0
    plan = moe_decode_megakernel.lower(*inputs, SMALL).plan
    assert [p.range.var for p in plan.phases] == ["d", "j", "a", "r", "c", "v"]
    assert [(c.name, c.phase, c.repeat, c.consumers) for c in plan.computed] == [
        ("chosen", 3, 0, (4,)),
        ("n", 3, 0, (4,)),
    ]


def test_steps_match_the_per_layer_kernels():
    """Two chained steps (the caches appended in between), from an empty and a one-token cache."""
    tokens, k_cache, v_cache, lengths, params = _inputs([0, 1, 127, 250])
    caches, layers = layer_caches(k_cache, v_cache), layer_weights(params)
    for _ in range(2):
        ours, k_cache, v_cache, new_lengths = jax.block_until_ready(
            megakernel_step(tokens, k_cache, v_cache, lengths, params, SMALL)
        )
        theirs, caches, _ = jax.block_until_ready(per_layer_step(tokens, caches, lengths, layers, params, CONFIG))
        _close(ours, theirs)
        for layer, cache in enumerate(caches):
            np.testing.assert_allclose(f64(cache["k"]), f64(k_cache[layer]), atol=3e-2)
        tokens, lengths = jnp.argmax(theirs, axis=-1).astype(jnp.int32), new_lengths


@pytest.mark.parametrize("group", [2, 4])
def test_attention_over_groups_of_sequences(group):
    tokens, k_cache, v_cache, lengths, params = _inputs([0, 1, 130, 250])
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    tiles = MoeMegaTiles(block_d=256, block_kv=128, block_a=256, block_r=256, block_e=128, block_v=256, group=group)
    ours, _ = run_checking_races(moe_decode_megakernel, *inputs, tiles)
    theirs, *_ = per_layer_step(tokens, layer_caches(k_cache, v_cache), lengths, layer_weights(params), params, CONFIG)
    _close(ours, theirs)


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_megakernel_lowers_for_v5e(batch):
    config = Config(layers=2, d_model=2048, heads=32, kv_heads=4, experts=128, ffn=768, vocab=32768)  # Qwen3-30B-A3B
    params = jax.eval_shape(lambda: init_params(config))
    lengths = jax.ShapeDtypeStruct((batch,), jnp.int32)
    cache = jax.ShapeDtypeStruct((2, batch, 2048, 512), jnp.bfloat16)
    specs = jax.eval_shape(kernel_inputs, lengths, cache, cache, lengths, params)
    lowered = moe_decode_megakernel.lower(*specs)
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.tpu
@pytest.mark.parametrize("batch", [1, 8, 32])
def test_megakernel_natively_matches_the_per_layer_kernels(batch):
    config = Config(layers=2, d_model=2048, heads=32, kv_heads=4, experts=128, ffn=768, vocab=32768)
    lengths = np.random.default_rng(0).integers(0, 1500, batch)
    tokens, k_cache, v_cache, lengths, params = _inputs(lengths, config, max_len=2048)
    ours, k_new, _, _ = megakernel_step(tokens, k_cache, v_cache, lengths, params)
    theirs, caches, _ = per_layer_step(
        tokens, layer_caches(k_cache, v_cache), lengths, layer_weights(params), params, config
    )
    _close(ours, theirs)
    np.testing.assert_allclose(f64(k_new[1]), f64(caches[1]["k"]), atol=5e-2)
