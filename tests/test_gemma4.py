"""Gemma 4 text inference (examples/gemma4.py, examples/gemma4_kernels.py): every kernel in the interpreter with
the race detector and against its loop nest and float64 NumPy; the model (plain-JAX reference and kernels)
against a float64 NumPy model: prefill logits, decode steps over ragged lengths with sliding-window ring
caches that wrap, greedy generation; the checkpoint reader and packer on a checkpoint written here; every
kernel cross-lowered to Mosaic for a v5e at Gemma 4 E2B's shapes (decode batches 1-256, a 2048-token
prefill). (benchmarks/gemma4_hf_tiny.py checks the same NumPy model against Hugging Face's.)"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decode import DecodeTiles, gqa_decode
from gemma4 import (
    E2B,
    FULL,
    SLIDING,
    TINY,
    Config,
    Generator,
    Tiles,
    config_from_json,
    decode_logits,
    decode_loop,
    decode_step,
    embed_tokens,
    fill_caches,
    forward,
    generate,
    init_params,
    load,
    prefill,
    prefill_hidden,
    read_safetensors,
    reference_forward,
    rope_tables,
)
from gemma4_kernels import (
    AttentionTiles,
    HeadTiles,
    PostTiles,
    QkvTiles,
    lm_head,
    lm_head_argmax,
    make_attention,
    post_attention,
    q_projection,
    qkv_projection,
)
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec

from autokernel import v5e

B16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
EPS = 1e-6
# Two steps per phase at TINY's sizes (d_model 256, heads of 128 and 256, ffn 512 / 1024, vocab 512).
SMALL = Tiles(
    qkv=QkvTiles(rows=128, block_d=128),
    attention=AttentionTiles(block_q=128, block_kv=128),
    decode=DecodeTiles(block_kv=128),
    post=PostTiles(rows=128, block_a=256, block_f=256),
    head=HeadTiles(rows=128, block_v=512, block_d=128),
)


# --------------------------------------------------------------------------- the float64 NumPy model


def _rms(x, gamma=None):
    out = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return out if gamma is None else out * f64(gamma).reshape(-1)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _rope(x, positions, theta, fraction):
    """Rotate-half RoPE of x (n, heads, dh) at positions (n,); only the first fraction * dh / 2 frequencies
    turn (Gemma 4's proportional RoPE)."""
    dh = x.shape[-1]
    turning = int(fraction * dh // 2)
    inv_freq = np.concatenate([theta ** (-np.arange(0, 2 * turning, 2) / dh), np.zeros(dh // 2 - turning)])
    angles = np.asarray(positions, np.float64)[:, None] * np.concatenate([inv_freq, inv_freq])[None]
    rotated = np.concatenate([-x[..., dh // 2 :], x[..., : dh // 2]], -1)
    return x * np.cos(angles)[:, None] + rotated * np.sin(angles)[:, None]


def _softmax(s):
    p = np.exp(s - s.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def numpy_model(tokens, params):
    """One sequence's soft-capped logits (n, vocab), float64, for its tokens (n,) at positions 0 .. n-1."""
    c: Config = params["config"]
    tokens = np.asarray(tokens)
    n = len(tokens)
    x = f64(params["embed"])[tokens] * np.sqrt(c.d_model)
    token_part = f64(params["embed_ple"])[tokens].reshape(n, c.layers, c.ple_dim) * np.sqrt(c.ple_dim)
    context = (x @ f64(params["ple_proj"])).reshape(n, c.layers, c.ple_dim) / np.sqrt(c.d_model)
    ple = (_rms(context, params["ple_norm"]) + token_part) / np.sqrt(2)
    positions = np.arange(n)
    kv = {}
    for i, p in enumerate(params["layers"]):
        dh, theta, fraction = c.rope(i)
        h = _rms(x, p["attn_norm"])
        q = _rope(_rms((h @ f64(p["w_q"])).reshape(n, c.heads, dh), p["q_norm"]), positions, theta, fraction)
        if not c.is_shared(i):
            k = _rms((h @ f64(p["w_k"])).reshape(n, c.kv_heads, dh), p["k_norm"])
            kv[i] = (_rope(k, positions, theta, fraction), _rms((h @ f64(p["w_v"])).reshape(n, c.kv_heads, dh)))
        k, v = kv[c.kv_layer(i)]
        rows, cols = positions[:, None], positions[None, :]
        visible = (cols <= rows) & ((rows - cols < c.window) | c.is_full(i))
        attn = np.zeros((n, c.heads, dh))
        for head in range(c.heads):
            g = head // (c.heads // c.kv_heads)
            attn[:, head] = _softmax(np.where(visible, q[:, head] @ k[:, g].T, -np.inf)) @ v[:, g]
        x = x + _rms(attn.reshape(n, -1) @ f64(p["wo"]), p["post_attn_norm"])
        m = _rms(x, p["pre_ffn_norm"])
        x = x + _rms((_gelu(m @ f64(p["w_gate"])) * (m @ f64(p["w_up"]))) @ f64(p["w_down"]), p["post_ffn_norm"])
        gate = _gelu(x @ f64(p["w_ple_gate"])) * ple[:, i]
        x = (x + _rms(gate @ f64(p["w_ple_proj"]), p["post_ple_norm"])) * f64(p["layer_scale"])[0, 0]
    logits = _rms(x, params["final_norm"]) @ f64(params["embed"]).T
    return np.tanh(logits / c.softcap) * c.softcap


def _close(got, want, tol):
    np.testing.assert_allclose(f64(got), want, atol=tol * np.abs(want).max())


def _relative_rms(got, want):
    got, want = f64(got), f64(want)
    return float(np.sqrt(np.mean((got - want) ** 2) / np.mean(want**2)))


# --------------------------------------------------------------------------- the kernels, in the interpreter


def _qkv_inputs(dh, rows, seed=0, d=256, heads=2):
    keys = jax.random.split(jax.random.key(seed), 8)
    x = jax.random.normal(keys[0], (1, rows, d), B16)
    gamma = (1 + 0.1 * jax.random.normal(keys[1], (1, d))).astype(B16)
    w_q, w_k, w_v = (
        (jax.random.normal(k, (d, n)) * d**-0.5).astype(B16)
        for k, n in zip(keys[2:5], (heads * dh, dh, dh), strict=True)
    )
    q_norm, k_norm = ((1 + 0.1 * jax.random.normal(k, (1, dh))).astype(B16) for k in keys[5:7])
    positions = jax.random.randint(keys[7], (rows,), 0, 3000)
    return x, gamma, w_q, w_k, w_v, q_norm, k_norm, positions


@pytest.mark.parametrize(("dh", "theta", "fraction"), [(128, 1e4, 1.0), (256, 1e6, 0.25)])
def test_qkv_kernels_match_numpy(dh, theta, fraction):
    # Decode-style rows: each row at its own position (the tables are per row); proportional RoPE on full layers.
    x, gamma, w_q, w_k, w_v, q_norm, k_norm, positions = _qkv_inputs(dh, 256)
    cos, sin = rope_tables(positions, dh, theta, fraction)
    tiles = QkvTiles(rows=128, block_d=128)
    args = (x, gamma, w_q, w_k, w_v, q_norm, k_norm, cos, sin, tiles)
    q, k, v = run_checking_races(qkv_projection, *args)
    h = _rms(f64(x[0]), gamma)
    want_q = _rope(_rms((h @ f64(w_q)).reshape(256, 2, dh), q_norm), positions, theta, fraction)
    want_k = _rope(_rms((h @ f64(w_k)).reshape(256, 1, dh), k_norm), positions, theta, fraction)
    _close(q[0], want_q.reshape(256, -1), 2e-2)
    _close(k[0], want_k.reshape(256, -1), 2e-2)
    _close(v[0], _rms(h @ f64(w_v)), 2e-2)
    assert qkv_projection.verify(*args) == 0.0
    q_only = run_checking_races(q_projection, x, gamma, w_q, q_norm, cos, sin, tiles)
    np.testing.assert_array_equal(f64(q_only), f64(q))
    assert q_projection.verify(x, gamma, w_q, q_norm, cos, sin, tiles) == 0.0


@pytest.mark.parametrize("window", [None, 128])
def test_attention_kernel_matches_numpy(window):
    b, s, heads, dh = 2, 384, 4, 128
    q, k, v = (rand((b, s, n * dh), seed, B16) for seed, n in ((0, heads), (1, 2), (2, 2)))
    q = (q.astype(F32) * dh**-0.5).astype(B16)
    tiles = AttentionTiles(block_q=128, block_kv=128)
    kernel = make_attention(dh, window)
    out = run_checking_races(kernel, q, k, v, tiles)
    positions = np.arange(s)
    visible = positions[None, :] <= positions[:, None]
    if window is not None:
        visible &= positions[:, None] - positions[None, :] < window
    want = np.zeros((b, s, heads, dh))
    for bb in range(b):
        for h in range(heads):
            g = slice(h // 2 * dh, (h // 2 + 1) * dh)
            scores = f64(q[bb, :, h * dh : (h + 1) * dh]) @ f64(k[bb, :, g]).T
            want[bb, :, h] = _softmax(np.where(visible, scores, -np.inf)) @ f64(v[bb, :, g])
    _close(out, want.reshape(b, s, -1), 2e-2)
    assert kernel.verify(q, k, v, tiles) == 0.0
    # Key blocks entirely in the future (and, with the window, entirely before it) take no step.
    lowered = kernel.lower(q, k, v, tiles)
    assert lowered.steps == b * (1 + 2 + 2 if window else 1 + 2 + 3)


def _post_inputs(m, d=256, a=512, f=512, p=128, seed=0):
    keys = iter(jax.random.split(jax.random.key(seed), 16))

    def normal(shape, scale=1.0):
        return (jax.random.normal(next(keys), shape) * scale).astype(B16)

    def norm():
        return (1 + 0.1 * jax.random.normal(next(keys), (1, d))).astype(B16)

    return (
        normal((m, d)),
        normal((m, a)),
        normal((m, p)),
        normal((a, d), a**-0.5),
        norm(),
        norm(),
        normal((d, f), d**-0.5),
        normal((d, f), d**-0.5),
        normal((f, d), f**-0.5),
        norm(),
        normal((d, p), d**-0.5),
        normal((p, d), p**-0.5),
        norm(),
        jnp.full((1, d), 0.75, B16),
    )


@pytest.mark.parametrize("rows", [256, 8])
def test_post_attention_kernel_matches_numpy(rows):
    args = _post_inputs(rows)
    x, attn, ple, wo, n1, n2, wg, wu, wd, n3, wpg, wpp, n4, scale = map(f64, args)
    tiles = PostTiles(rows=128, block_a=256, block_f=256)
    out = run_checking_races(post_attention, *args, tiles)
    h = x + _rms(attn @ wo, n1)
    m = _rms(h, n2)
    h = h + _rms((_gelu(m @ wg) * (m @ wu)) @ wd, n3)
    want = (h + _rms((_gelu(h @ wpg) * ple) @ wpp, n4)) * scale
    _close(out, want, 2e-2)
    assert post_attention.verify(*args, tiles) == 0.0
    assert [phase.range.var for phase in post_attention.lower(*args, tiles).plan.phases] == ["a", "f"]


def test_lm_head_kernels_match_numpy():
    x, gamma, embed = rand((256, 256), 0, B16), (1 + 0.1 * rand((1, 256), 1)).astype(B16), rand((1024, 256), 2, B16)
    # Ties: rows 0-3's largest logits are equal at two columns (two vocabulary rows the same).
    embed = embed.at[700].set(embed[300])
    x = x.at[:4].set(jnp.broadcast_to(embed[300].astype(F32) * 3, (4, 256)).astype(B16))
    tiles = HeadTiles(rows=128, block_v=512, block_d=128)
    logits = run_checking_races(lm_head, x, gamma, embed, tiles)
    raw = _rms(f64(x), gamma) @ f64(embed).T
    _close(logits, np.tanh(raw / 30) * 30, 2e-2)
    assert lm_head.verify(x, gamma, embed, tiles) == 0.0
    best = run_checking_races(lm_head_argmax, x, gamma, embed, tiles)
    assert (np.asarray(best) == np.asarray(best)[:, :1]).all()  # every lane the same
    assert best[:4, 0].tolist() == [300] * 4  # the first of the tied columns
    # The kernel's argmax is exact on the f32 logits it computes; float64's may differ only at near-ties.
    top2 = np.sort(raw, axis=1)[:, -2:]
    differ = best[:, 0] != np.argmax(raw, axis=1)
    assert (top2[differ, 1] - top2[differ, 0] < 0.05).all() and differ.mean() < 0.05
    assert lm_head_argmax.verify(x, gamma, embed, tiles) == 0.0


def test_decode_attention_over_a_ring_matches_numpy():
    # A sliding layer's ring of 128 slots: a sequence past the window sees every slot, in any order.
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (3, 1, 8, 256), B16)
    k_cache, v_cache = (jax.random.normal(key, (3, 128, 256), B16) for key in keys[1:])
    seen = jnp.array([128, 5, 77], I32)
    out = run_checking_races(gqa_decode, q, k_cache, v_cache, seen, DecodeTiles(block_kv=128))
    for b, n in enumerate(seen.tolist()):
        p = _softmax(f64(q[b, 0]) @ f64(k_cache[b, :n]).T)
        _close(out[b, 0], p @ f64(v_cache[b, :n]), 2e-2)


# --------------------------------------------------------------------------- the model against NumPy


@pytest.fixture(scope="module")
def tiny_f32():
    return init_params(TINY, seed=0, dtype=F32)


@pytest.fixture(scope="module")
def tiny_bf16():
    return init_params(TINY, seed=0)


def test_config_layers():
    assert E2B.first_shared == 15 and [E2B.kv_layer(i) for i in (3, 4, 15, 19, 33, 34)] == [3, 4, 13, 14, 13, 14]
    assert [E2B.dh(i) for i in (0, 4, 34)] == [256, 512, 512]
    assert [E2B.ffn_of(i) for i in (14, 15)] == [6144, 12288]
    assert E2B.ring(4096) == 512 and E2B.ring(384) == 384
    assert TINY.kv_layer(3) == 1 and TINY.kv_layer(4) == 2


@pytest.mark.cpu  # f32 matmuls are exact in the interpreter (on a TPU they are bf16 passes)
def test_forward_and_reference_match_numpy_in_f32(tiny_f32):
    ids = jax.random.randint(jax.random.key(1), (2, 256), 0, TINY.vocab)
    ours = jax.jit(forward, static_argnums=2)(tiny_f32, ids, SMALL)
    ref = jax.jit(reference_forward)(tiny_f32, ids)
    for b in range(2):
        want = numpy_model(ids[b], tiny_f32)
        _close(ours[b], want, 1e-4)
        _close(ref[b], want, 1e-4)


def test_forward_and_reference_match_numpy_in_bf16(tiny_bf16):
    ids = jax.random.randint(jax.random.key(2), (1, 200), 0, TINY.vocab)  # padded to 256 inside
    ours = jax.jit(forward)(tiny_bf16, ids)
    ref = jax.jit(reference_forward)(tiny_bf16, ids)
    want = numpy_model(ids[0], tiny_bf16)
    assert ours.shape == (1, 200, TINY.vocab)
    assert _relative_rms(ours[0], want) < 1.5e-2
    assert _relative_rms(ref[0], want) < 2.5e-2  # rounds to bf16 between ops, as transformers does
    assert np.mean(np.argmax(f64(ours[0]), -1) == np.argmax(want, -1)) > 0.97


@pytest.mark.cpu
def test_prefill_and_decode_steps_match_numpy(tiny_f32):
    # Ragged prompts in a padded batch; the sliding layers' rings (128 slots) wrap for the long prompt.
    ids = jax.random.randint(jax.random.key(3), (3, 300), 0, TINY.vocab)
    lengths = jnp.array([300, 140, 3], I32)
    max_len = 384
    logits, caches, lens = jax.jit(prefill, static_argnums=(3, 4))(tiny_f32, ids, lengths, max_len, SMALL)
    assert [c["k"].shape for c in caches] == [(3, 128, 128), (3, 128, 128), (3, 384, 256)]
    sequences = [list(np.asarray(ids[b, :n])) for b, n in enumerate(lengths.tolist())]
    for b, sequence in enumerate(sequences):
        _close(logits[b], numpy_model(sequence, tiny_f32)[-1], 1e-4)
    step = jax.jit(decode_logits, static_argnums=4)
    for t in range(3):
        tokens = jax.random.randint(jax.random.key(10 + t), (3,), 0, TINY.vocab).astype(I32)
        logits, caches, lens = step(tiny_f32, tokens, caches, lens, SMALL)
        for b, sequence in enumerate(sequences):
            sequence.append(int(tokens[b]))
            _close(logits[b], numpy_model(sequence, tiny_f32)[-1], 1e-4)
    assert lens.tolist() == [303, 143, 6]


def test_fill_caches_keeps_each_prompts_last_window():
    config = Config(**{**TINY.__dict__, "window": 128})
    k = jnp.arange(2 * 300, dtype=F32).reshape(2, 300, 1) * jnp.ones((1, 1, 128))
    kv = {i: (k, -k) for i in range(config.first_shared)}
    lengths = jnp.array([300, 70], I32)
    caches = fill_caches(config, kv, lengths, 384)
    ring = np.asarray(caches[0]["k"][:, :, 0])
    for slot in range(128):  # the last position p < 300 with p % 128 == slot
        assert ring[0, slot] == max(p for p in range(300) if p % 128 == slot)
    assert ring[1, :70].tolist() == [300 + p for p in range(70)]
    assert caches[2]["k"].shape == (2, 384, 128) and not np.asarray(caches[2]["k"][:, 300:]).any()


def _greedy_numpy_agrees(params, prompt, tokens, tie=0.05):
    """Our greedy tokens after `prompt` against float64 argmaxes on the same prefix: equal, or a near-tie."""
    sequence = list(np.asarray(prompt))
    for token in np.asarray(tokens).tolist():
        logits = numpy_model(sequence, params)[-1]
        best = int(np.argmax(logits))
        assert token == best or logits[best] - logits[token] < tie * np.abs(logits).max(), (token, best)
        sequence.append(token)


def test_greedy_generation_matches_numpy(tiny_bf16):
    prompts = jax.random.randint(jax.random.key(4), (2, 160), 0, TINY.vocab)
    lengths = jnp.array([160, 90], I32)
    tokens, timings = generate(tiny_bf16, prompts, 6, lengths)
    assert tokens.shape == (2, 6) and timings.decode_tokens_per_s > 0
    for b, n in enumerate(lengths.tolist()):
        _greedy_numpy_agrees(tiny_bf16, prompts[b, :n], tokens[b])


def test_decode_loop_runs_the_steps():
    # The fori_loop over steps (one jit, caches donated) gives what the steps one by one give.
    params = init_params(TINY, seed=5)
    prompts = jax.random.randint(jax.random.key(5), (2, 128), 0, TINY.vocab)
    lengths = jnp.array([128, 100], I32)
    logits, caches, lens = jax.jit(prefill, static_argnums=3)(params, prompts, lengths, 256)
    first = jnp.argmax(logits, axis=-1).astype(I32)
    looped, _, looped_lens = jax.jit(decode_loop, static_argnums=4)(params, first, caches, lens, 3)
    token, step = first, jax.jit(decode_step)
    for t in range(3):
        token, caches, lens = step(params, token, caches, lens)
        assert looped[:, t].tolist() == token.tolist()
    assert looped_lens.tolist() == lens.tolist() == [131, 103]


def test_generator_reuses_its_compiled_functions(tiny_bf16):
    prompts = jax.random.randint(jax.random.key(6), (1, 128), 0, TINY.vocab)
    generator = Generator(tiny_bf16, 1, 128, 3)
    first, _ = generator(prompts)
    again, timings = generator(prompts)
    assert first.tolist() == again.tolist() and generator.max_len == 256 and timings.prefill_ms > 0


# --------------------------------------------------------------------------- weights


def _to_hf(params):
    """`params` as Hugging Face's text tensors (names after model.language_model.), the inverse of `pack`."""
    c: Config = params["config"]
    out = {
        "embed_tokens.weight": params["embed"],
        "embed_tokens_per_layer.weight": params["embed_ple"],
        "per_layer_model_projection.weight": params["ple_proj"].T,
        "per_layer_projection_norm.weight": params["ple_norm"][0],
        "norm.weight": params["final_norm"][0],
    }
    names = {
        "attn_norm": "input_layernorm.weight",
        "w_q": "self_attn.q_proj.weight",
        "w_k": "self_attn.k_proj.weight",
        "w_v": "self_attn.v_proj.weight",
        "q_norm": "self_attn.q_norm.weight",
        "k_norm": "self_attn.k_norm.weight",
        "wo": "self_attn.o_proj.weight",
        "post_attn_norm": "post_attention_layernorm.weight",
        "pre_ffn_norm": "pre_feedforward_layernorm.weight",
        "w_gate": "mlp.gate_proj.weight",
        "w_up": "mlp.up_proj.weight",
        "w_down": "mlp.down_proj.weight",
        "post_ffn_norm": "post_feedforward_layernorm.weight",
        "w_ple_gate": "per_layer_input_gate.weight",
        "w_ple_proj": "per_layer_projection.weight",
        "post_ple_norm": "post_per_layer_input_norm.weight",
    }
    for i, layer in enumerate(params["layers"]):
        for ours, theirs in names.items():
            if ours in layer:
                w = layer[ours]
                out[f"layers.{i}.{theirs}"] = w[0] if w.shape[0] == 1 else w.T
        out[f"layers.{i}.layer_scalar"] = layer["layer_scale"][0, :1]
        if c.is_shared(i):  # the checkpoint has (unused) k/v projections on KV-shared layers too
            out[f"layers.{i}.self_attn.k_proj.weight"] = jnp.zeros((c.dh(i), c.d_model), B16)
    return out


def _write_safetensors(path, tensors):
    header, offset, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for name, value in tensors.items():
        data = np.asarray(value).tobytes()
        dtype = {np.dtype(jnp.bfloat16): "BF16", np.dtype(np.float32): "F32"}[np.asarray(value).dtype]
        header[name] = {"dtype": dtype, "shape": list(np.shape(value)), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))


def _text_config(config):
    return {
        "num_hidden_layers": config.layers,
        "hidden_size": config.d_model,
        "num_attention_heads": config.heads,
        "num_key_value_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "global_head_dim": config.global_head_dim,
        "intermediate_size": config.ffn,
        "vocab_size": config.vocab,
        "hidden_size_per_layer_input": config.ple_dim,
        "num_kv_shared_layers": config.kv_shared_layers,
        "layer_types": list(config.layer_types),
        "use_double_wide_mlp": config.double_wide_mlp,
        "sliding_window": config.window,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "num_global_key_value_heads": None,
        "final_logit_softcapping": 30.0,
        "rope_parameters": {
            FULL: {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            SLIDING: {"rope_theta": 1e4, "rope_type": "default"},
        },
    }


def test_config_from_json_reads_e2b(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": _text_config(E2B)}))
    assert config_from_json(tmp_path / "config.json") == E2B


def test_load_reads_and_packs_a_checkpoint(tmp_path, tiny_bf16):
    tensors = {"model.language_model." + k: v for k, v in _to_hf(tiny_bf16).items()}
    tensors["model.vision_tower.ignored"] = jnp.ones((2, 2), F32)
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": _text_config(TINY)}))
    assert set(read_safetensors(tmp_path / "model.safetensors", "model.vision_tower.")) == {"ignored"}
    for path in (tmp_path, tmp_path / "model.safetensors"):
        loaded = load(path)
        assert loaded["config"] == TINY
        same = jax.tree.map(lambda a, b: bool((np.asarray(a) == np.asarray(b)).all()), loaded, tiny_bf16)
        assert all(jax.tree.leaves(same))
        assert set(loaded["layers"][3]) == set(tiny_bf16["layers"][3])  # no k/v weights on shared layers


def test_embedding_multiplier_is_rounded_to_bf16():
    # sqrt(1536) = 39.19 is 39.25 in bf16 (transformers multiplies by the bf16 scalar).
    params = {"config": E2B, "embed": jnp.ones((4, 8), B16)}
    assert float(embed_tokens(params, jnp.array([1]))[0, 0]) == 39.25


# --------------------------------------------------------------------------- cross-lowering at E2B's sizes


def _lowers(kernel, *specs):
    lowered = kernel.lower(*specs)
    assert lowered.vmem_bytes <= v5e.VMEM_BUDGET
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


def _b16(*shape):
    return spec(*shape, dtype=B16)


@pytest.mark.parametrize("rows", [1, 8, 64, 256, 2048])  # decode batches; a 2048-token prefill
@pytest.mark.parametrize("dh", [256, 512])
def test_projection_and_post_attention_kernels_lower_for_v5e(rows, dh):
    d, h, p = E2B.d_model, E2B.heads, E2B.ple_dim
    rope = spec(rows, dh)
    _lowers(
        qkv_projection,
        _b16(1, rows, d),
        _b16(1, d),
        _b16(d, h * dh),
        *[_b16(d, dh)] * 2,
        *[_b16(1, dh)] * 2,
        rope,
        rope,
    )
    _lowers(q_projection, _b16(1, rows, d), _b16(1, d), _b16(d, h * dh), _b16(1, dh), rope, rope)
    norm = _b16(1, d)
    for f in (6144, 12288):
        weights = (
            _b16(h * dh, d),
            norm,
            norm,
            _b16(d, f),
            _b16(d, f),
            _b16(f, d),
            norm,
            _b16(d, p),
            _b16(p, d),
            norm,
            norm,
        )
        _lowers(post_attention, _b16(rows, d), _b16(rows, h * dh), _b16(rows, p), *weights)


@pytest.mark.parametrize(("batch", "seq"), [(1, 2048), (64, 128)])
@pytest.mark.parametrize(("dh", "window"), [(256, 512), (512, None)])
def test_attention_kernels_lower_for_v5e(batch, seq, dh, window):
    _lowers(make_attention(dh, window), _b16(batch, seq, 8 * dh), _b16(batch, seq, dh), _b16(batch, seq, dh))


@pytest.mark.parametrize("batch", [1, 8, 64, 256])
def test_decode_kernels_lower_for_v5e(batch):
    for dh, positions in ((256, 512), (512, 4096)):
        cache = _b16(batch, positions, dh)
        _lowers(gqa_decode, _b16(batch, 1, 8, dh), cache, cache, spec(batch, dtype=I32))
    _lowers(lm_head_argmax, _b16(batch, E2B.d_model), _b16(1, E2B.d_model), _b16(E2B.vocab, E2B.d_model))
    _lowers(lm_head, _b16(batch, E2B.d_model), _b16(1, E2B.d_model), _b16(E2B.vocab, E2B.d_model))


# --------------------------------------------------------------------------- on a TPU v5e


@pytest.mark.tpu
def test_model_at_e2b_sizes_natively_matches_the_reference():
    config = Config(**{**E2B.__dict__, "layers": 7, "kv_shared_layers": 2, "layer_types": (*E2B.layer_types[:6], FULL)})
    params = init_params(config, seed=0)
    ids = jax.random.randint(jax.random.key(1), (2, 512), 0, config.vocab)
    ours = jax.jit(forward)(params, ids)
    want = jax.jit(reference_forward)(params, ids)
    assert _relative_rms(ours, want) < 3e-2
    x, _ = jax.jit(prefill_hidden)(params, ids)
    assert x.shape == (2, 512, config.d_model)
    tokens, _ = generate(params, ids, 8)
    assert tokens.shape == (2, 8)
