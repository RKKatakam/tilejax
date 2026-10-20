"""KV-cache decoding (examples/decode.py, kv_cache.py, decode_model.py): the decode attention kernels (GQA
over contiguous and paged caches, absorbed MLA) against float64 NumPy at ragged lengths, the cache layouts,
the post-attention kernel, and whole decode steps and greedy generation against a float64 NumPy model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decode import (
    DecodeTiles,
    gqa_decode,
    jax_paged_attention,
    jax_ragged_paged_attention,
    mla_attention,
    mla_decode,
    paged_gqa_decode,
    xla_gqa_decode,
    xla_mla_decode,
)
from decode_model import (
    Config,
    MlpTiles,
    baseline_step,
    decode_mlp,
    decode_step,
    empty_caches,
    generate,
    init_params,
    paged_caches,
    prefill,
    random_caches,
)
from helpers import V5E, f64, lower_for_v5e, rand, run_checking_races, spec
from kv_cache import (
    append,
    append_jax_paged,
    append_paged,
    append_ragged,
    jax_paged_layout,
    pad_lanes,
    page_tables,
    ragged_kv_pages,
    to_pages,
)

B16 = jnp.bfloat16
I32 = jnp.int32


def i32(*values):
    return jnp.array(values, I32)


# --------------------------------------------------------------------------- float64 NumPy references


def _softmax_rows(s):
    p = np.exp(s - s.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def _gqa(q, k_cache, v_cache, lengths):
    """GQA decode attention: q (B, T, H, Dh), caches (B, S, Hkv * Dh); token t of sequence b sits at
    position lengths[b] - T + t and sees the positions up to it; empty sequences give 0."""
    q, k_cache, v_cache = f64(q), f64(k_cache), f64(v_cache)
    _, t_, h_, dh = q.shape
    group = h_ // (k_cache.shape[2] // dh)
    out = np.zeros(q.shape)
    for b, n in enumerate(np.asarray(lengths)):
        for t in range(t_ if n else 0):
            for h in range(h_):
                cols = slice(h // group * dh, (h // group + 1) * dh)
                seen = n - t_ + t + 1
                p = _softmax_rows(q[b, t, h] @ k_cache[b, :seen, cols].T)
                out[b, t, h] = p @ v_cache[b, :seen, cols]
    return out


def _mla(q_nope, q_rope, c_kv, k_rope, lengths, w_uk, w_uv):
    """MLA decode attention in its plain (non-absorbed) form, DeepSeek-V2 eq. (10)-(19): head h's keys
    [c @ w_uk[h] | k_rope] and values c @ w_uv[h], scaled by 1/sqrt(nope + rope dims)."""
    q_nope, q_rope, c_kv, k_rope, w_uk, w_uv = map(f64, (q_nope, q_rope, c_kv, k_rope, w_uk, w_uv))
    b_, t_, h_, dn = q_nope.shape
    dr = q_rope.shape[-1]
    out = np.zeros((b_, t_, h_, w_uv.shape[-1]))
    for b, n in enumerate(np.asarray(lengths)):
        for t in range(t_ if n else 0):
            seen = n - t_ + t + 1
            for h in range(h_):
                keys = np.concatenate([c_kv[b, :seen] @ w_uk[h], k_rope[b, :seen, :dr]], axis=1)
                query = np.concatenate([q_nope[b, t, h], q_rope[b, t, h]])
                p = _softmax_rows(keys @ query / np.sqrt(dn + dr))
                out[b, t, h] = p @ (c_kv[b, :seen] @ w_uv[h])
    return out


def _rms(x, gamma, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * f64(gamma)


def _rope(x, positions, theta=1e6):
    """x (n, heads, Dh) at positions (n,), rotate-half form."""
    dh = x.shape[-1]
    angles = positions[:, None] * theta ** (-np.arange(0, dh, 2) / dh)
    cos, sin = np.cos(np.concatenate([angles] * 2, -1)), np.sin(np.concatenate([angles] * 2, -1))
    rotated = np.concatenate([-x[..., dh // 2 :], x[..., : dh // 2]], axis=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def _silu(x):
    return x / (1 + np.exp(-x))


def _mlp(x, attn, layer):
    resid = x + attn @ f64(layer["wo"])
    h = _rms(resid, layer["ffn_norm"])
    return resid + (_silu(h @ f64(layer["w_gate"])) * (h @ f64(layer["w_up"]))) @ f64(layer["w_down"])


def _model(tokens, params, heads):
    """The model's logits (n, vocab) for one sequence's tokens (n,), causal, in float64."""
    x = f64(params["embed"])[np.asarray(tokens)]
    n = len(tokens)
    positions = np.arange(n, dtype=np.float64)
    for layer in params["layers"]:
        qkv = (_rms(x, layer["attn_norm"]) @ f64(layer["w_qkv"])).reshape(n, -1, 128)
        kv_heads = (qkv.shape[1] - heads) // 2
        q, k, v = (
            _rope(qkv[:, :heads], positions),
            _rope(qkv[:, heads : heads + kv_heads], positions),
            qkv[:, -kv_heads:],
        )
        attn = np.zeros((n, heads, 128))
        for h in range(heads):
            g = h // (heads // kv_heads)
            s = np.where(np.tril(np.ones((n, n), bool)), q[:, h] @ k[:, g].T, -np.inf)
            attn[:, h] = _softmax_rows(s) @ v[:, g]
        x = _mlp(x, attn.reshape(n, -1), layer)
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"])


# --------------------------------------------------------------------------- inputs


def _gqa_inputs(b, t, h, kv_heads, s, seed=0):
    q = (rand((b, t, h, 128), seed, B16) * 128**-0.5).astype(B16)
    k = rand((b, s, kv_heads * 128), seed + 1, B16)
    v = rand((b, s, kv_heads * 128), seed + 2, B16)
    return q, k, v


def _mla_inputs(b, t, h, s, seed=0):
    keys = jax.random.split(jax.random.key(seed), 6)
    q_nope = jax.random.normal(keys[0], (b, t, h, 128), B16)
    q_rope = jax.random.normal(keys[1], (b, t, h, 64), B16)
    c_kv = jax.random.normal(keys[2], (b, s, 512), B16)
    k_rope = pad_lanes(jax.random.normal(keys[3], (b, s, 64), B16))  # [k_rope | 0]
    w_uk = (jax.random.normal(keys[4], (h, 512, 128)) / 512**0.5).astype(B16)
    w_uv = (jax.random.normal(keys[5], (h, 512, 128)) / 512**0.5).astype(B16)
    return q_nope, q_rope, c_kv, k_rope, w_uk, w_uv


def _ragged(t, s):
    """Lengths: an empty sequence, one with just the new tokens, a partial last block and a full cache."""
    return i32(0, t, 300, s)


def _close(out, want, tol=2e-2):
    np.testing.assert_allclose(f64(out), want, atol=tol * np.abs(want).max())


# --------------------------------------------------------------------------- decode attention kernels


@pytest.mark.parametrize(("t", "heads", "kv_heads", "block_kv"), [(1, 8, 2, 128), (1, 4, 4, 256), (3, 8, 2, 128)])
def test_gqa_decode_matches_numpy(t, heads, kv_heads, block_kv):
    q, k, v = _gqa_inputs(4, t, heads, kv_heads, 512)
    lengths, tiles = _ragged(t, 512), DecodeTiles(block_kv=block_kv)
    out = run_checking_races(gqa_decode, q, k, v, lengths, tiles)
    want = _gqa(q, k, v, lengths)
    _close(out, want)
    assert not np.asarray(out[0]).any()  # the empty sequence
    gqa_decode.verify(q, k, v, lengths, tiles)
    _close(xla_gqa_decode(q, k, v, lengths), want)  # the XLA baseline computes the same
    lowered = gqa_decode.lower(q, k, v, lengths, tiles)
    assert lowered.schedule.runtime and lowered.steps == 4 * 512 // block_kv  # at most
    assert {op.name: op.kind.value for op in lowered.plan.operands} == {
        "q": "tile",
        "k_cache": "stream",
        "v_cache": "stream",
    }
    assert "jnp.logical_not(j >= lengths[b])" in lowered.source


def test_gqa_decode_reads_only_the_valid_blocks():
    q, k, v = _gqa_inputs(4, 1, 8, 2, 512)
    lowered = gqa_decode.lower(q, k, v, _ragged(1, 512), DecodeTiles(block_kv=128))
    tables = jax.jit(lowered.entry.__globals__["gqa_decode_steps"])(_ragged(1, 512))
    assert int(tables[2][0]) == 1 + 1 + 3 + 4  # the empty sequence takes one empty step


def test_paged_gqa_decode_matches_numpy_and_the_contiguous_kernel():
    q, k, v = _gqa_inputs(4, 1, 8, 2, 512)
    lengths = _ragged(1, 512)
    tables = jnp.asarray(page_tables(4, 512, 128, seed=1))
    k_pages, v_pages = to_pages(k, tables), to_pages(v, tables)
    out = run_checking_races(paged_gqa_decode, q, k_pages, v_pages, tables, lengths)
    _close(out, _gqa(q, k, v, lengths))
    np.testing.assert_array_equal(f64(out), f64(gqa_decode(q, k, v, lengths, DecodeTiles(block_kv=128))))
    paged_gqa_decode.verify(q, k_pages, v_pages, tables, lengths)
    lowered = paged_gqa_decode.lower(q, k_pages, v_pages, tables, lengths)
    assert "k_pages_hbm.at[jnp.clip(block_tables_smem[b_blk, kb], 0, 15), :, :]" in lowered.source


def test_paged_gqa_decode_with_new_tokens_and_small_pages():
    q, k, v = _gqa_inputs(3, 2, 4, 2, 256, seed=3)
    lengths = i32(2, 64, 200)
    tables = jnp.asarray(page_tables(3, 256, 64, seed=2))
    out = run_checking_races(paged_gqa_decode, q, to_pages(k, tables), to_pages(v, tables), tables, lengths)
    _close(out, _gqa(q, k, v, lengths))


@pytest.mark.parametrize(("t", "heads", "block_kv"), [(1, 16, 128), (2, 16, 256)])
def test_mla_decode_absorbed_matches_the_plain_form(t, heads, block_kv):
    q_nope, q_rope, c_kv, k_rope, w_uk, w_uv = _mla_inputs(4, t, heads, 512)
    lengths, tiles = _ragged(t, 512), DecodeTiles(block_kv=block_kv)
    want = _mla(q_nope, q_rope, c_kv, k_rope, lengths, w_uk, w_uv)
    _close(mla_attention(q_nope, q_rope, c_kv, k_rope, lengths, w_uk, w_uv, tiles), want, 3e-2)
    from decode import absorb_queries

    q_c, q_r = absorb_queries(q_nope, q_rope, w_uk)
    assert q_r.shape[-1] == 128 and not np.asarray(q_r[..., 64:]).any()
    latents = run_checking_races(mla_decode, q_c, q_r, c_kv, k_rope, lengths, tiles)
    mla_decode.verify(q_c, q_r, c_kv, k_rope, lengths, tiles)
    np.testing.assert_allclose(f64(latents), f64(xla_mla_decode(q_c, q_r, c_kv, k_rope, lengths)), atol=2e-2)


# --------------------------------------------------------------------------- cross-lowering at realistic sizes


@pytest.mark.parametrize(
    ("b", "t", "heads", "kv_heads", "s", "block_kv"),
    [(8, 1, 32, 8, 8192, 512), (64, 1, 32, 8, 2048, 256), (16, 4, 32, 8, 4096, 1024), (32, 1, 16, 2, 4096, 2048)],
)
def test_gqa_decode_lowers_for_v5e(b, t, heads, kv_heads, s, block_kv):
    specs = (spec(b, t, heads, 128, dtype=B16), spec(b, s, kv_heads * 128, dtype=B16))
    specs += (specs[1], spec(b, dtype=I32))
    lowered = gqa_decode.lower(*specs, DecodeTiles(block_kv=block_kv))
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(("b", "t", "page_size", "s"), [(32, 1, 128, 4096), (8, 1, 256, 8192), (16, 2, 64, 2048)])
def test_paged_gqa_decode_lowers_for_v5e(b, t, page_size, s):
    pages = b * s // page_size
    specs = (
        spec(b, t, 32, 128, dtype=B16),
        spec(pages, page_size, 1024, dtype=B16),
        spec(pages, page_size, 1024, dtype=B16),
    )
    specs += (spec(b, s // page_size, dtype=I32), spec(b, dtype=I32))
    assert "tpu_custom_call" in lower_for_v5e(paged_gqa_decode.lower(*specs), *specs)


@pytest.mark.parametrize(("b", "t", "heads", "s"), [(32, 1, 16, 4096), (8, 1, 128, 8192), (16, 2, 64, 2048)])
def test_mla_decode_lowers_for_v5e(b, t, heads, s):
    specs = (spec(b, t, heads, 512, dtype=B16), spec(b, t, heads, 128, dtype=B16))
    specs += (spec(b, s, 512, dtype=B16), spec(b, s, 128, dtype=B16), spec(b, dtype=I32))
    assert "tpu_custom_call" in lower_for_v5e(mla_decode.lower(*specs), *specs)


@pytest.mark.parametrize(("rows", "d", "heads", "ffn"), [(8, 2048, 16, 6144), (64, 4096, 32, 12288)])
def test_decode_mlp_lowers_for_v5e(rows, d, heads, ffn):
    specs = (spec(rows, d, dtype=B16), spec(rows, heads * 128, dtype=B16), spec(heads * 128, d, dtype=B16))
    specs += (spec(1, d, dtype=B16), spec(d, ffn, dtype=B16), spec(d, ffn, dtype=B16), spec(ffn, d, dtype=B16))
    assert "tpu_custom_call" in lower_for_v5e(decode_mlp.lower(*specs), *specs)


def test_the_jax_kernel_baselines_lower_for_v5e():
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=V5E)
    pages = spec(8, 64, 128, 128, dtype=B16)  # (Hkv, num_pages, P, Dh)
    paged = (spec(8, 1, 32, 128, dtype=B16), pages, pages, spec(8, 8, dtype=I32), spec(8, dtype=I32))
    ragged = (paged[0], spec(64, 128, 16, 128, dtype=B16), paged[3], paged[4])
    sizes = {"num_kv_pages_per_block": 4, "num_queries_per_block": 8}  # (else they're looked up on the TPU)
    with jax.sharding.use_abstract_mesh(mesh):
        for fn, specs in (
            (lambda *a: jax_paged_attention(*a, 4), paged),
            (lambda *a: jax_ragged_paged_attention(*a, **sizes), ragged),
        ):
            lowered = jax.jit(fn).trace(*specs).lower(lowering_platforms=("tpu",))
            assert "tpu_custom_call" in lowered.as_text()


# --------------------------------------------------------------------------- caches


def test_appending_to_contiguous_and_paged_caches():
    cache = jnp.zeros((3, 256, 256), B16)
    lengths = i32(0, 100, 254)
    new = rand((3, 2, 256), 1, B16)
    appended = append(cache, lengths, new)
    want = np.zeros(cache.shape)
    for b, n in enumerate([0, 100, 254]):
        want[b, n : n + 2] = f64(new[b])
    np.testing.assert_array_equal(f64(appended), want)
    tables = jnp.asarray(page_tables(3, 256, 64, seed=5))
    pages = append_paged(to_pages(cache, tables), tables, lengths, new)
    np.testing.assert_array_equal(f64(pages), f64(to_pages(appended, tables)))
    # Past S_max: dropped.
    np.testing.assert_array_equal(f64(append(cache, i32(0, 0, 255), new))[2, 255], f64(new[2, 0]))


def test_jax_kernel_layouts():
    k, v = rand((2, 256, 256), 0, B16), rand((2, 256, 256), 1, B16)
    tables = jnp.asarray(page_tables(2, 256, 128, seed=None))
    k_pages, v_pages = to_pages(k, tables), to_pages(v, tables)
    assert (tables == jnp.arange(4).reshape(2, 2)).all()
    jk = jax_paged_layout(k_pages, 2)
    np.testing.assert_array_equal(f64(jk[1, 3]), f64(k[1, 128:, 128:]))  # kv head 1, page 3 = sequence 1's second
    kv = ragged_kv_pages(k_pages, v_pages, 2)
    np.testing.assert_array_equal(f64(kv[2, :, 2]), f64(k[1, :128, 128:]))
    np.testing.assert_array_equal(f64(kv[2, :, 3]), f64(v[1, :128, 128:]))
    lengths, new_k, new_v = i32(3, 130), rand((2, 1, 2, 128), 2, B16), rand((2, 1, 2, 128), 3, B16)
    flat = append(k, lengths, new_k.reshape(2, 1, 256))
    np.testing.assert_array_equal(
        f64(append_jax_paged(jk, tables, lengths, new_k)), f64(jax_paged_layout(to_pages(flat, tables), 2))
    )
    kv_new = append_ragged(kv, tables, lengths, new_k, new_v)
    flat_v = append(v, lengths, new_v.reshape(2, 1, 256))
    np.testing.assert_array_equal(
        f64(kv_new), f64(ragged_kv_pages(to_pages(flat, tables), to_pages(flat_v, tables), 2))
    )


# --------------------------------------------------------------------------- the model


CONFIG = Config(layers=2, d_model=256, heads=4, kv_heads=2, ffn=512, vocab=512)


def test_decode_mlp_matches_numpy():
    params = init_params(CONFIG)
    layer = params["layers"][0]
    x, attn = rand((8, 256), 0, B16), rand((8, 512), 1, B16)
    args = (x, attn, layer["wo"], layer["ffn_norm"], layer["w_gate"], layer["w_up"], layer["w_down"])
    tiles = MlpTiles(block_a=256, block_f=256)  # two steps of each phase
    out = run_checking_races(decode_mlp, *args, tiles)
    want = _mlp(f64(x), f64(attn), layer)
    _close(out, want)
    decode_mlp.verify(*args, tiles)
    assert [phase.range.var for phase in decode_mlp.lower(*args, tiles).plan.phases] == ["a", "f"]


def _prompts():
    prompts = jax.random.randint(jax.random.key(1), (3, 128), 0, CONFIG.vocab)
    return prompts, i32(5, 128, 77)


@pytest.mark.parametrize("page_size", [None, 64])
def test_decode_step_matches_a_numpy_model(page_size):
    params = init_params(CONFIG)
    prompts, prompt_lengths = _prompts()
    tables = None if page_size is None else jnp.asarray(page_tables(3, 256, page_size))
    caches = empty_caches(CONFIG, 3, 256, page_size)
    first, caches, lengths = prefill(prompts, prompt_lengths, params, caches, tables)
    new = jax.random.randint(jax.random.key(2), (3, 2), 0, CONFIG.vocab)  # two new tokens per sequence
    logits, caches, lengths = decode_step(new, caches, lengths, params, tables)
    assert lengths.tolist() == [7, 130, 79] and logits.shape == (3, 2, CONFIG.vocab)
    for b, n in enumerate([5, 128, 77]):
        tokens = np.concatenate([np.asarray(prompts[b, :n]), np.asarray(new[b])])
        want = _model(tokens, params, CONFIG.heads)
        _close(first[b], want[n - 1], 5e-2)
        _close(logits[b], want[n:], 5e-2)
    if page_size is None:  # the plain-JAX baseline step agrees
        caches = empty_caches(CONFIG, 3, 256)
        _, caches, lengths = prefill(prompts, prompt_lengths, params, caches)
        plain, _, _ = baseline_step("xla")(new, caches, lengths, params)
        _close(plain, f64(logits), 5e-2)


def test_greedy_generation_matches_a_numpy_model():
    params = init_params(CONFIG)
    prompts, prompt_lengths = _prompts()
    generated = generate(prompts, prompt_lengths, params, CONFIG, steps=4, max_len=256)
    assert generated.shape == (3, 4)
    for b, n in enumerate([5, 128, 77]):
        tokens = np.concatenate([np.asarray(prompts[b, :n]), np.asarray(generated[b])])
        logits = _model(tokens[:-1], params, CONFIG.heads)[n - 1 :]  # predicting each generated token
        chosen = logits[np.arange(4), np.asarray(generated[b])]
        # Greedy: each token is NumPy's argmax, up to bf16 rounding of near ties.
        assert (logits.max(-1) - chosen <= 3e-2 * np.abs(logits).max()).all()
    paged = generate(prompts, prompt_lengths, params, CONFIG, steps=4, max_len=256, page_size=128)
    np.testing.assert_array_equal(np.asarray(paged), np.asarray(generated))


def test_random_caches_are_zero_past_each_length():
    caches = random_caches(CONFIG, np.array([3, 0]), 128)
    assert len(caches) == 2 and caches[0]["k"].shape == (2, 128, 256)
    assert np.asarray(caches[1]["v"][0, :3]).any() and not np.asarray(caches[1]["v"][0, 3:]).any()
    assert not np.asarray(caches[0]["k"][1]).any()
    tables = jnp.asarray(page_tables(2, 128, 64))
    assert paged_caches(caches, tables)[0]["k"].shape == (4, 64, 256)


# --------------------------------------------------------------------------- on a TPU v5e: against JAX's kernels


def _ragged_lengths(b, mean, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(mean // 2, 3 * mean // 2, b), I32)


@pytest.mark.tpu
@pytest.mark.parametrize(("b", "mean", "page_size"), [(8, 1024, 128), (32, 2048, 64)])
def test_gqa_decode_natively_matches_jax_paged_attention(b, mean, page_size):
    s = 4 * mean
    q, k, v = _gqa_inputs(b, 1, 32, 8, s)
    lengths = _ragged_lengths(b, mean)
    tables = jnp.asarray(page_tables(b, s, page_size))
    k_pages, v_pages = to_pages(k, tables), to_pages(v, tables)
    theirs = jax_paged_attention(q, jax_paged_layout(k_pages, 8), jax_paged_layout(v_pages, 8), tables, lengths, 4)
    ragged = jax_ragged_paged_attention(q, ragged_kv_pages(k_pages, v_pages, 8), tables, lengths)
    ours = gqa_decode(q, k, v, lengths)
    paged = paged_gqa_decode(q, k_pages, v_pages, tables, lengths)
    for got in (ours, paged, ragged):
        np.testing.assert_allclose(f64(got), f64(theirs), atol=2e-2)
    gqa_decode.verify(q, k, v, lengths, DecodeTiles())


@pytest.mark.tpu
@pytest.mark.parametrize("heads", [16, 128])
def test_mla_decode_natively_matches_xla(heads):
    q_nope, q_rope, c_kv, k_rope, w_uk, w_uv = _mla_inputs(16, 1, heads, 4096)
    lengths = _ragged_lengths(16, 2048)
    out = mla_attention(q_nope, q_rope, c_kv, k_rope, lengths, w_uk, w_uv)
    from decode import absorb_queries

    q_c, q_r = absorb_queries(q_nope, q_rope, w_uk)
    want = jnp.einsum("bthc,hcd->bthd", xla_mla_decode(q_c, q_r, c_kv, k_rope, lengths), w_uv)
    np.testing.assert_allclose(f64(out), f64(want), atol=3e-2)


@pytest.mark.tpu
def test_decode_step_natively_matches_the_jax_baselines():
    config = Config(layers=2, d_model=1024, heads=8, kv_heads=4, ffn=2048, vocab=4096)
    params = init_params(config)
    lengths = _ragged_lengths(8, 512)
    caches = random_caches(config, np.asarray(lengths), 2048)
    tables = jnp.asarray(page_tables(8, 2048, 128))
    tokens = jax.random.randint(jax.random.key(3), (8, 1), 0, config.vocab)
    ours, _, _ = decode_step(tokens, caches, lengths, params)
    for attention, layout in (
        ("xla", caches),
        ("paged", [{n: jax_paged_layout(to_pages(c[n], tables), 4) for n in c} for c in caches]),
        ("ragged", [{"kv": ragged_kv_pages(to_pages(c["k"], tables), to_pages(c["v"], tables), 4)} for c in caches]),
    ):
        theirs, _, _ = baseline_step(attention)(tokens, layout, lengths, params, tables)
        np.testing.assert_allclose(f64(ours), f64(theirs), atol=5e-2 * float(jnp.abs(theirs).max()))
