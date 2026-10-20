"""A whole decode step in one kernel (examples/decode_megakernel.py): the megakernel against a float64 NumPy
decode step over several steps with growing ragged lengths (empty caches and one-token caches included),
its loop nest, the per-layer-kernels and plain-JAX steps, its plan, and cross-lowering for a v5e at real
model sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decode_megakernel import (
    HEAD_DIM,
    WEIGHTS,
    Config,
    MegaTiles,
    decode_megakernel,
    init_params,
    kernel_inputs,
    layer_caches,
    layer_weights,
    megakernel_step,
    per_layer_step,
    random_caches,
    xla_step,
)
from helpers import f64, lower_for_v5e, run_checking_races, spec

from autokernel import v5e

B16 = jnp.bfloat16
I32 = jnp.int32
CONFIG = Config(layers=2, d_model=256, heads=4, kv_heads=2, ffn=512, vocab=512)
SMALL = MegaTiles(block_d=128, block_kv=128, block_a=256, block_f=256, block_v=256)  # two steps per phase


# --------------------------------------------------------------------------- a float64 NumPy decode step


def _rms(x, gamma=None, eps=1e-6):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return out if gamma is None else out * f64(gamma)


def _rope(x, positions, theta=1e6):
    """x (B, heads, Dh) at positions (B,), rotate-half."""
    dh = x.shape[-1]
    angles = positions[:, None] * theta ** (-np.arange(0, dh, 2) / dh)
    angles = np.concatenate([angles, angles], -1)[:, None, :]
    rotated = np.concatenate([-x[..., dh // 2 :], x[..., : dh // 2]], axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def _softmax(s):
    p = np.exp(s - s.max())
    return p / p.sum()


def numpy_step(tokens, k_cache, v_cache, lengths, params, config):
    """The decode step in float64: logits (B, vocab) and each layer's new keys and values (L, B, Hkv * Dh).
    Token b sits at position lengths[b] and sees the first lengths[b] cached positions and itself."""
    p = {name: f64(value) for name, value in params.items()}
    lengths = np.asarray(lengths)
    k_cache, v_cache = f64(k_cache), f64(v_cache)
    b_, h_, g_ = len(lengths), config.heads, config.kv_heads
    hd, kvd, group = h_ * HEAD_DIM, g_ * HEAD_DIM, h_ // g_
    x = p["embed"][np.asarray(tokens)]
    new_k, new_v = [], []
    for layer in range(config.layers):
        qkv = _rms(x, p["attn_norm"][layer]) @ p["w_qkv"][layer]
        q = qkv[:, :hd].reshape(b_, h_, HEAD_DIM)
        k = qkv[:, hd : hd + kvd].reshape(b_, g_, HEAD_DIM)
        v = qkv[:, hd + kvd :].reshape(b_, g_, HEAD_DIM)
        q = _rope(_rms(q, p["q_norm"][layer]), lengths) / np.sqrt(HEAD_DIM)
        k = _rope(_rms(k, p["k_norm"][layer]), lengths)
        attn = np.zeros((b_, h_, HEAD_DIM))
        for b, n in enumerate(lengths):
            for h in range(h_):
                g, cols = h // group, slice(h // group * HEAD_DIM, (h // group + 1) * HEAD_DIM)
                keys = np.concatenate([k_cache[layer, b, :n, cols], k[b, g][None]])
                values = np.concatenate([v_cache[layer, b, :n, cols], v[b, g][None]])
                attn[b, h] = _softmax(keys @ q[b, h]) @ values
        x = x + attn.reshape(b_, hd) @ p["wo"][layer]
        m = _rms(x, p["ffn_norm"][layer])
        gate = m @ p["w_gate"][layer]
        x = x + (gate / (1 + np.exp(-gate)) * (m @ p["w_up"][layer])) @ p["w_down"][layer]
        new_k.append(k.reshape(b_, kvd))
        new_v.append(v.reshape(b_, kvd))
    return _rms(x, p["final_norm"]) @ p["w_vocab"], np.stack(new_k), np.stack(new_v)


def _close(got, want, tol=3e-2):
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


def _step_inputs(batch, lengths, max_len, config=CONFIG, seed=0):
    params = init_params(config, seed=seed)
    lengths = jnp.array(lengths, I32)
    k_cache, v_cache = random_caches(config, lengths, max_len, seed=seed + 1)
    tokens = jax.random.randint(jax.random.key(seed + 2), (batch,), 0, config.vocab)
    return tokens, k_cache, v_cache, lengths, params


# --------------------------------------------------------------------------- the megakernel


def test_megakernel_matches_numpy_and_its_loop_nest():
    tokens, k_cache, v_cache, lengths, params = _step_inputs(4, [0, 1, 130, 250], 256)
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    logits, new_kv = run_checking_races(decode_megakernel, *inputs, SMALL)
    want, want_k, want_v = numpy_step(tokens, k_cache, v_cache, lengths, params, CONFIG)
    _close(logits, want)
    kvd = CONFIG.kv_heads * HEAD_DIM
    _close(new_kv[:, :, :kvd], want_k, 2e-2)
    _close(new_kv[:, :, kvd:], want_v, 2e-2)
    assert decode_megakernel.verify(*inputs, SMALL) == 0.0  # same ops in the same order


def test_decode_steps_with_growing_lengths_match_numpy():
    """Three chained steps (the caches appended in between), from an empty and a one-token cache."""
    tokens, k_cache, v_cache, lengths, params = _step_inputs(4, [0, 1, 127, 254 - 2], 256)
    for step in range(3):
        want, want_k, want_v = numpy_step(tokens, k_cache, v_cache, lengths, params, CONFIG)
        logits, k_new, v_new, new_lengths = megakernel_step(tokens, k_cache, v_cache, lengths, params, SMALL)
        _close(logits, want)
        assert new_lengths.tolist() == [step + 1, step + 2, 128 + step, 253 + step]
        rows = np.arange(4)
        _close(np.asarray(k_new, np.float64)[:, rows, np.asarray(lengths)], want_k, 2e-2)
        _close(np.asarray(v_new, np.float64)[:, rows, np.asarray(lengths)], want_v, 2e-2)
        # Nothing else in the caches changed.
        changed = np.zeros(k_cache.shape[:3], bool)
        changed[:, rows, np.asarray(lengths)] = True
        np.testing.assert_array_equal(f64(k_new)[~changed], f64(k_cache)[~changed])
        tokens = jnp.argmax(logits, axis=-1).astype(I32)
        k_cache, v_cache, lengths = k_new, v_new, new_lengths


@pytest.mark.parametrize(
    ("batch", "lengths", "layers"),
    [(1, [0], 2), (1, [200], 2), (10, [5, 0, 128, 129, 255, 1, 64, 250, 3, 100], 1)],
)
def test_one_sequence_and_a_batch_past_eight_rows(batch, lengths, layers):
    config = Config(layers=layers, d_model=256, heads=4, kv_heads=2, ffn=512, vocab=512)
    tokens, k_cache, v_cache, lengths, params = _step_inputs(batch, lengths, 256, config, seed=batch)
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    logits, new_kv = run_checking_races(decode_megakernel, *inputs, SMALL)
    want, want_k, _ = numpy_step(tokens, k_cache, v_cache, lengths, params, config)
    _close(logits, want)
    _close(new_kv[:, :, : config.kv_heads * HEAD_DIM], want_k, 2e-2)
    if layers == 1:  # a repeat loop of one iteration runs inline
        assert decode_megakernel.lower(*inputs, SMALL).plan.repeats == ()


def test_megakernel_agrees_with_the_per_layer_kernels_and_xla():
    tokens, k_cache, v_cache, lengths, params = _step_inputs(4, [0, 1, 130, 250], 256)
    # Off a TPU, nothing may be dispatched while an interpreted kernel runs: its callbacks run JAX
    # computations of their own, and with XLA:CPU's executions in flight exhausted that deadlocks. So wait for
    # the kernel, and run each baseline as one jitted computation.
    ours, k_new, v_new, _ = jax.block_until_ready(megakernel_step(tokens, k_cache, v_cache, lengths, params, SMALL))
    layers = layer_weights(params)
    for step in (per_layer_step, xla_step):
        caches = jax.block_until_ready(layer_caches(k_cache, v_cache))
        theirs, caches, new_lengths = jax.block_until_ready(jax.jit(step)(tokens, caches, lengths, layers, params))
        _close(theirs, f64(ours), 3e-2)
        assert new_lengths.tolist() == [1, 2, 131, 251]
        for layer, cache in enumerate(caches):
            np.testing.assert_allclose(f64(cache["k"]), f64(k_new[layer]), atol=3e-2)
            np.testing.assert_allclose(f64(cache["v"]), f64(v_new[layer]), atol=3e-2)
    # The per-layer caches random_caches draws hold the stacked ones' values.
    per_layer = random_caches(CONFIG, lengths, 256, seed=1, stacked=False)
    for layer, cache in enumerate(per_layer):
        np.testing.assert_array_equal(f64(cache["k"]), f64(k_cache[layer]))
        np.testing.assert_array_equal(f64(cache["v"]), f64(v_cache[layer]))


def test_megakernel_plan_streams_every_layer_and_visits_only_cached_blocks():
    tokens, k_cache, v_cache, lengths, params = _step_inputs(4, [0, 1, 130, 250], 256)
    inputs = kernel_inputs(tokens, k_cache, v_cache, lengths, params)
    lowered = decode_megakernel.lower(*inputs, SMALL)
    plan, schedule = lowered.plan, lowered.schedule
    assert [phase.range.var for phase in plan.phases] == ["d", "j", "a", "f", "c"]
    assert [(r.range.var, r.first, r.last) for r in plan.repeats] == [("layer", 0, 4)]
    kinds = {op.name: (op.kind.value, op.phase) for op in plan.operands}
    for name, phase in (("w_qkv", 0), ("k_cache", 1), ("v_cache", 1), ("wo", 2), ("w_down", 3), ("w_vocab", 4)):
        assert kinds[name] == ("stream", phase)
    assert kinds["x"][0] == "once" and schedule.runtime  # one tile: fetched once
    src = lowered.source
    for line in (
        "n_blocks_ref[d // bd]",  # a block of a carried value: one VMEM load
        "q_bd_ref[j // S]",
        "acc_ref[j // S] = ",  # a partial update: only that sequence's state is stored
        "logit_blocks_ref[c // bv] = ",
        "jnp.logical_not(j % S >= lengths[j // S])",  # the runtime mask
    ):
        assert line in src
    # Steps that run: per layer 2 QKV blocks, the cache blocks before each length (0, 1, 2 and 2 of 2), 2 of
    # W_o, 2 of the FFN; then 2 vocab blocks.
    count = jax.jit(lowered.entry.__globals__["decode_megakernel_steps"])(lengths)[2][0]
    assert int(count) == 2 * (2 + (0 + 1 + 2 + 2) + 2 + 2) + 2
    empty = jax.jit(lowered.entry.__globals__["decode_megakernel_steps"])(jnp.zeros(4, I32))[2][0]
    assert int(empty) == 2 * (2 + 1 + 2 + 2) + 2  # all caches empty: one empty attention step per layer


def test_empty_caches_attend_to_the_new_token_only():
    tokens, k_cache, v_cache, _, params = _step_inputs(4, [0, 0, 0, 0], 256)
    lengths = jnp.zeros(4, I32)
    logits, *_ = megakernel_step(tokens, k_cache, v_cache, lengths, params, SMALL)
    _close(logits, numpy_step(tokens, k_cache, v_cache, lengths, params, CONFIG)[0])


# --------------------------------------------------------------------------- cross-lowering at real sizes


def _specs(layers, d, heads, kv_heads, ffn, vocab, batch, max_len):
    hd, kvd = heads * HEAD_DIM, kv_heads * HEAD_DIM
    shapes = {
        "attn_norm": (layers, 1, d),
        "w_qkv": (layers, d, hd + 2 * kvd),
        "q_norm": (layers, 1, HEAD_DIM),
        "k_norm": (layers, 1, HEAD_DIM),
        "wo": (layers, hd, d),
        "ffn_norm": (layers, 1, d),
        "w_gate": (layers, d, ffn),
        "w_up": (layers, d, ffn),
        "w_down": (layers, ffn, d),
        "final_norm": (1, d),
        "w_vocab": (d, vocab),
    }
    rope = spec(batch, HEAD_DIM)
    cache = spec(layers, batch * max_len, kvd, dtype=B16)
    head = (spec(batch, d, dtype=B16), rope, rope, spec(batch, dtype=I32), cache, cache)
    return head + tuple(spec(*shapes[name], dtype=B16) for name in WEIGHTS)


@pytest.mark.parametrize(
    ("layers", "batch", "max_len", "tiles"),
    [
        (16, 1, 4096, None),
        (16, 8, 4096, None),
        (8, 64, 4096, None),
        (8, 32, 4096, MegaTiles(block_d=512, block_kv=512, block_a=1024, block_f=1024, block_v=2048)),
    ],
)
def test_megakernel_lowers_for_v5e(layers, batch, max_len, tiles):
    specs = _specs(layers, 2048, 16, 8, 6144, 32768, batch, max_len)
    lowered = decode_megakernel.lower(*specs, *(() if tiles is None else (tiles,)))
    assert lowered.vmem_bytes <= v5e.VMEM_BUDGET
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


# --------------------------------------------------------------------------- on a TPU v5e


@pytest.mark.tpu
@pytest.mark.parametrize(("batch", "s_max"), [(1, 2048), (8, 2048), (64, 2048), (8, 1792), (8, 7168)])
def test_megakernel_natively_matches_the_per_layer_kernels_and_xla(batch, s_max):
    config = Config(layers=4, d_model=2048, heads=16, kv_heads=8, ffn=6144, vocab=32768)
    lengths = np.random.default_rng(0).integers(0, s_max * 3 // 4, batch).astype(np.int32)
    tokens, k_cache, v_cache, lengths, params = _step_inputs(batch, lengths, s_max, config)
    ours, k_new, _, _ = megakernel_step(tokens, k_cache, v_cache, lengths, params)
    layers = layer_weights(params)
    for step in (per_layer_step, xla_step):
        theirs, caches, _ = step(tokens, layer_caches(k_cache, v_cache), lengths, layers, params)
        np.testing.assert_allclose(f64(ours), f64(theirs), atol=5e-2 * float(jnp.abs(theirs).max()))
        np.testing.assert_allclose(f64(k_new[1]), f64(caches[1]["k"]), atol=5e-2)
