"""DeepSeek-V3 / Kimi K2 inference (examples/deepseek.py): prefill, the latent cache, decode steps (both decode
forms) and greedy generation against a float64 NumPy model written in the plain (expanded) DeepSeek form; the
absorbed weights, folded and factored, in float64; the decode kernels; the plain-JAX baselines; and every
kernel cross-lowered for a v5e at the benchmark's shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ON_TPU
from decode import DecodeTiles, mla_decode
from deepseek import (
    CONFIGS,
    Config,
    HeadTailTiles,
    HeadTiles,
    TailTiles,
    baseline_decode_step,
    baseline_params,
    baseline_prefill,
    decode_step,
    decode_tail,
    decode_tail_heads,
    factor,
    fold,
    generate,
    init_params,
    kv_bytes_per_token,
    latent_query_heads,
    pack,
    prefill,
    rope_rows,
)
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from matmul import MatmulTiles, matmul
from mla import LANES, NOPE, QK, ROPE, AttentionTiles, mla_attention, mla_down, rope_tables
from moe import ExpertTiles, SharedTiles, capacity, expert_ffn, shared_experts
from rmsnorm_matmul import NormMatmulTiles, rmsnorm_matmul
from swiglu import swiglu_ffn

B16 = jnp.bfloat16
I32 = jnp.int32

# Three layers (one dense, two MoE: sigmoid routing in the better of two groups), the real latent width. (The
# data below is clear of routing near-ties: see `_model`.)
TINY = Config(3, 256, 4, 256, 512, 8, 2, 128, 512, 512, n_group=2, topk_group=1, scale=2.5)


# --------------------------------------------------------------------------- the float64 NumPy model


def _rms(t, gamma, eps=1e-6):
    return t / np.sqrt(np.mean(t * t, axis=-1, keepdims=True) + eps) * f64(gamma).reshape(-1)


def _rope(t, positions):
    """Rotate-half RoPE on the last dim (64) of t (n, ..., 64) at positions (n,)."""
    inv_freq = 10000.0 ** (-np.arange(0, ROPE, 2) / ROPE)
    angles = positions[:, None] * np.concatenate([inv_freq, inv_freq])[None, :]
    shape = (len(positions),) + (1,) * (t.ndim - 2) + (ROPE,)
    rotated = np.concatenate([-t[..., ROPE // 2 :], t[..., : ROPE // 2]], axis=-1)
    return t * np.cos(angles).reshape(shape) + rotated * np.sin(angles).reshape(shape)


def _silu(v):
    return v / (1 + np.exp(-v))


def _swiglu(x, w_gate, w_up, w_down):
    return (_silu(x @ f64(w_gate)) * (x @ f64(w_up))) @ f64(w_down)


def _softmax_rows(s):
    p = np.exp(s - s.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def _down(x, attn, positions):
    """c_q, c_kv and the rotated k_rope (n, 64) of a sequence's rows x (n, d)."""
    h = _rms(x, attn["attn_norm"])
    c_q = _rms(h @ f64(attn["w_dq"]), attn["q_norm"])
    c_kv = _rms(h @ f64(attn["w_dkv"]), attn["kv_norm"])
    return c_q, c_kv, _rope(h @ f64(attn["w_kr"]), positions)


def _mla(x, attn):
    """MLA's output (without the residual) for a sequence's rows x (n, d), DeepSeek's plain form: per head,
    q = [q_nope | rope(q_rope)], k = [c @ W_uk | k_rope], v = c @ W_uv, causal softmax(q.k / sqrt(192))."""
    n = x.shape[0]
    positions = np.arange(n, dtype=np.float64)
    c_q, c_kv, k_rope = _down(x, attn, positions)
    q = np.einsum("sr,rhe->she", c_q, f64(attn["w_uq"]))
    q_nope, q_rope = q[..., :NOPE], _rope(q[..., NOPE:], positions)
    k_nope, v = (np.einsum("sr,rhe->she", c_kv, f64(attn[w])) for w in ("w_uk", "w_uv"))
    s = np.einsum("qhe,khe->hqk", q_nope, k_nope) + np.einsum("qhe,ke->hqk", q_rope, k_rope)
    s = np.where(np.tril(np.ones((n, n), bool)), s * QK**-0.5, -np.inf)
    o = np.einsum("hqk,khe->qhe", _softmax_rows(s), v)
    return np.einsum("qhe,hed->qd", o, f64(attn["w_o"]))


def _route(normed, moe, config):
    """DeepSeek-V3's router: sigmoid scores; experts chosen by score + bias among the best groups (each
    scored by its two best experts); the chosen scores renormalized and scaled. Also each token's margin:
    how far its choice is from a tie (between the last kept and the first dropped group, or expert)."""
    scores = 1 / (1 + np.exp(-(normed @ f64(moe["router"]))))
    choice = scores + f64(moe["bias"])
    t, e = choice.shape
    margin = np.full(t, np.inf)
    if config.n_group > 1:
        group_scores = np.sort(choice.reshape(t, config.n_group, -1), axis=2)[:, :, -2:].sum(2)
        ranked = -np.sort(-group_scores, axis=1)
        margin = ranked[:, config.topk_group - 1] - ranked[:, config.topk_group]
        best = np.argsort(-group_scores, axis=1)[:, : config.topk_group]
        kept = np.zeros((t, config.n_group), bool)
        kept[np.arange(t)[:, None], best] = True
        choice = np.where(np.repeat(kept, e // config.n_group, axis=1), choice, -np.inf)
    ranked = -np.sort(-choice, axis=1)
    margin = np.minimum(margin, ranked[:, config.top_k - 1] - ranked[:, config.top_k])
    ids = np.argsort(-choice, axis=1, kind="stable")[:, : config.top_k]
    weights = np.take_along_axis(scores, ids, 1)
    return ids, weights / weights.sum(1, keepdims=True) * config.scale, margin


def _moe(x, moe, config):
    """The routed experts and the shared expert (without the residual), and the routing margins."""
    normed = _rms(x, moe["norm"])
    ids, weights, margin = _route(normed, moe, config)
    s = moe["shared"]
    out = _swiglu(normed, s["w_gate"], s["w_up"], s["w_down"])
    for t in range(x.shape[0]):
        for e, w in zip(ids[t], weights[t], strict=True):
            out[t] += w * _swiglu(normed[t], moe["w_gate"][e], moe["w_up"][e], moe["w_down"][e])
    return out, margin


def _model(tokens, params, checked=1):
    """The model's logits (n, vocab) for one sequence's tokens (n,), in float64. The routing of the last
    `checked` tokens, whose logits a test compares, must be clear of ties (bf16 rounding could choose other
    experts than float64 at a near tie, so test data avoids them; elsewhere a flip reaches the compared
    tokens only through attention, diluted)."""
    config = params["config"]
    x = f64(params["embed"])[np.asarray(tokens)]
    for layer in params["layers"]:
        x = x + _mla(x, layer["attn"])
        if "moe" in layer:
            y, margins = _moe(x, layer["moe"], config.moe)
            closest = margins[-checked:].min()
            assert closest > 3e-3, f"a near tie in routing ({closest:.2g}): pick other test data"
            x = x + y
        else:
            f = layer["ffn"]
            x = x + _swiglu(_rms(x, f["norm"]), f["w_gate"], f["w_up"], f["w_down"])
    return _rms(x, params["final_norm"]) @ f64(params["w_vocab"])


def _close(out, want, tol=5e-2):
    np.testing.assert_allclose(f64(out), want, atol=tol * np.abs(want).max())


# --------------------------------------------------------------------------- the absorbed form


@pytest.mark.parametrize("form", ["folded", "factored"])
def test_absorbed_weights_give_the_expanded_attention_in_float64(form):
    config = Config(1, 256, 4, 128, 256, 8, 2, 128, 256, 256)
    with jax.enable_x64(True):
        attn = {name: jnp.asarray(f64(w)) for name, w in init_params(config, jnp.float32)["layers"][0]["attn"].items()}
        weights = {name: f64(w) for name, w in (fold if form == "folded" else factor)(attn).items()}
    n, h, c = 64, config.heads, config.kv_rank
    x = f64(rand((n, 256), 3))
    positions = np.arange(n, dtype=np.float64)
    c_q, c_kv, k_rope = _down(x, attn, positions)
    if form == "folded":  # q_c = c_q @ (W_uq_nope W_uk^T), one matmul for every head (and the rotary part)
        assert weights["w_uq_absorbed"].shape == (128, h * (c + ROPE)) and weights["w_uvo"].shape == (h * c, 256)
        q = c_q @ weights["w_uq_absorbed"]  # pre-scaled
        q_c, q_rope = q[:, : h * c].reshape(n, h, c), q[:, h * c :].reshape(n, h, ROPE)
    else:  # q_c = (c_q @ W_uq_nope) @ W_uk^T, head by head
        q_nope = np.einsum("qr,hre->qhe", c_q, weights["w_uq_nope"])
        q_c = np.einsum("qhe,hce->qhc", q_nope, weights["w_uk_heads"])
        q_rope = np.einsum("qr,hre->qhe", c_q, weights["w_uq_rope"])
    s = np.einsum("qhc,kc->hqk", q_c, c_kv) + np.einsum("qhe,ke->hqk", _rope(q_rope, positions), k_rope)
    s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    o_c = np.einsum("hqk,kc->qhc", _softmax_rows(s), c_kv)  # every head attends to the same latents
    if form == "folded":
        out = o_c.reshape(n, h * c) @ weights["w_uvo"]
    else:
        out = np.einsum("qhe,hed->qd", np.einsum("qhc,hce->qhe", o_c, weights["w_uv_heads"]), f64(attn["w_o"]))
    np.testing.assert_allclose(out, _mla(x, attn), rtol=1e-10, atol=1e-10)


def test_rope_rows_are_mlas_tables_at_any_positions():
    cos, sin = rope_rows(jnp.array([0, 5, 77, 3], I32))
    want_cos, want_sin = rope_tables(80)
    np.testing.assert_allclose(f64(cos), f64(want_cos[jnp.array([0, 5, 77, 3])]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f64(sin), f64(want_sin[jnp.array([0, 5, 77, 3])]), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- the decode kernel


@pytest.mark.parametrize(("rows", "tiles"), [(8, TailTiles(block_a=256, block_f=128)), (3, TailTiles())])
def test_decode_tail_matches_numpy(rows, tiles):
    x, attn = rand((rows, 256), 0, B16), rand((rows, 512), 1, B16)
    w_out = (rand((512, 256), 2) / 23).astype(B16)
    gamma = (1 + 0.1 * rand((1, 256), 3)).astype(B16)
    w_gate, w_up = (rand((256, 256), 4) / 16).astype(B16), (rand((256, 256), 5) / 16).astype(B16)
    w_down = (rand((256, 256), 6) / 16).astype(B16)
    args = (x, attn, w_out, gamma, w_gate, w_up, w_down, tiles)
    resid, out = run_checking_races(decode_tail, *args)
    want_resid = f64(x) + f64(attn) @ f64(w_out)
    _close(resid, want_resid, 2e-2)
    _close(out, want_resid + _swiglu(_rms(want_resid, gamma), w_gate, w_up, w_down), 2e-2)
    decode_tail.verify(*args)
    phases = [phase.range.var for phase in decode_tail.lower(*args).plan.phases]
    assert phases == (["a", "f"] if rows == 8 else [])  # loops of one block run inline: the weights stay put


@pytest.mark.parametrize(("rows", "heads"), [(8, 2), (1, 4)])
def test_latent_query_heads_matches_numpy(rows, heads):
    c_q = rand((rows, 256), 0, B16)
    w_uq_nope, w_uq_rope = (rand((4, 256, 128), 1) / 16).astype(B16), (rand((4, 256, 64), 2) / 16).astype(B16)
    w_uk = (rand((4, 512, 128), 3) / 11).astype(B16)
    args = (c_q, w_uq_nope, w_uq_rope, w_uk, HeadTiles(heads=heads))
    q_c, q_rope = run_checking_races(latent_query_heads, *args)
    q_nope = np.einsum("mr,hre->hme", f64(c_q), f64(w_uq_nope))
    _close(q_c, np.einsum("hme,hce->hmc", q_nope, f64(w_uk)), 2e-2)
    _close(q_rope, np.einsum("mr,hre->hme", f64(c_q), f64(w_uq_rope)), 2e-2)
    latent_query_heads.verify(*args)
    lowered = latent_query_heads.lower(*args)
    assert lowered.tiles == 4 // heads and not lowered.plan.phases  # a tile per block of heads, one step each


def test_decode_tail_heads_matches_numpy():
    x, heads = rand((8, 256), 0, B16), rand((4, 8, 512), 1, B16)
    w_uv, w_o = (rand((4, 512, 128), 2) / 22).astype(B16), (rand((4, 128, 256), 3) / 22).astype(B16)
    gamma = (1 + 0.1 * rand((1, 256), 4)).astype(B16)
    w_gate, w_up = (rand((256, 256), 5) / 16).astype(B16), (rand((256, 256), 6) / 16).astype(B16)
    w_down = (rand((256, 256), 7) / 16).astype(B16)
    args = (x, heads, w_uv, w_o, gamma, w_gate, w_up, w_down, HeadTailTiles(heads=2, block_f=128))
    resid, out = run_checking_races(decode_tail_heads, *args)
    values = np.einsum("hmc,hce->hme", f64(heads), f64(w_uv))
    want_resid = f64(x) + np.einsum("hme,hed->md", values, f64(w_o))
    _close(resid, want_resid, 2e-2)
    _close(out, want_resid + _swiglu(_rms(want_resid, gamma), w_gate, w_up, w_down), 2e-2)
    decode_tail_heads.verify(*args)
    assert [phase.range.var for phase in decode_tail_heads.lower(*args).plan.phases] == ["hh", "f"]


# --------------------------------------------------------------------------- the model against NumPy


@pytest.fixture(scope="module")
def model():
    params = init_params(TINY)
    return params, pack(params)


@pytest.fixture(scope="module")
def factored(model):
    return pack(model[0], decode="factored")


def _prompts():
    prompts = jax.random.randint(jax.random.key(1), (2, 128), 0, TINY.vocab)
    return prompts, jnp.array([128, 77], I32)


def test_prefill_matches_numpy(model):
    params, packed = model
    prompts, lengths = _prompts()
    logits, cache, out_lengths = prefill(prompts, packed, max_len=256, prompt_lengths=lengths)
    assert logits.shape == (2, TINY.vocab) and out_lengths.tolist() == [128, 77]
    assert len(cache) == 3 and cache[0]["c_kv"].shape == (2, 256, 512) and cache[0]["k_rope"].shape == (2, 256, 128)
    for b, n in enumerate([128, 77]):
        _close(logits[b], _model(prompts[b, :n], params)[-1])
    # The first layer's cache: the normalized latents and [k_rope | k_rope], zeros past the prompt.
    attn = params["layers"][0]["attn"]
    x = f64(params["embed"])[np.asarray(prompts[0])]
    _, c_kv, k_rope = _down(x, attn, np.arange(128, dtype=np.float64))
    _close(cache[0]["c_kv"][0, :128], c_kv, 2e-2)
    _close(cache[0]["k_rope"][0, :128], np.concatenate([k_rope, k_rope], axis=1), 2e-2)
    assert not np.asarray(cache[0]["c_kv"][:, 128:]).any()


@pytest.mark.parametrize("form", ["folded", "factored"])
def test_decode_steps_match_numpy(model, factored, form):
    params, packed = model
    packed = packed if form == "folded" else factored
    prompts, lengths = _prompts()
    _, cache, lengths = prefill(prompts, packed, max_len=256, prompt_lengths=lengths)
    new = jax.random.randint(jax.random.key(2), (2, 3), 0, TINY.vocab)
    two, cache, lengths = decode_step(new[:, :2], cache, lengths, packed)  # two new tokens per sequence
    one, cache, lengths = jax.jit(decode_step, donate_argnums=1)(new[:, 2:], cache, lengths, packed)
    assert two.shape == (2, 2, TINY.vocab) and one.shape == (2, 1, TINY.vocab) and lengths.tolist() == [131, 80]
    for b, n in enumerate([128, 77]):
        want = _model(np.concatenate([np.asarray(prompts[b, :n]), np.asarray(new[b])]), params, checked=3)
        _close(two[b], want[n : n + 2])
        _close(one[b, 0], want[n + 2])


def test_greedy_generation_matches_numpy(model):
    params, packed = model
    prompts, lengths = _prompts()
    generated = generate(prompts, packed, steps=3, prompt_lengths=lengths)
    assert generated.shape == (2, 3)
    for b, n in enumerate([128, 77]):
        tokens = np.concatenate([np.asarray(prompts[b, :n]), np.asarray(generated[b])])
        logits = _model(tokens[:-1], params, checked=3)[n - 1 :]  # predicting each generated token
        chosen = logits[np.arange(3), np.asarray(generated[b])]
        # Greedy: each token is NumPy's argmax, up to bf16 rounding of near ties.
        assert (logits.max(-1) - chosen <= 3e-2 * np.abs(logits).max()).all()
    again = generate(prompts, packed, steps=3, prompt_lengths=lengths)  # now jitted throughout
    np.testing.assert_array_equal(np.asarray(again), np.asarray(generated))


def test_the_baselines_match_numpy_and_share_the_cache_layout(model):
    params, packed = model
    plain = baseline_params(params)
    prompts, lengths = _prompts()
    logits, cache, lengths = baseline_prefill()(prompts, plain, 256, lengths)
    for b, n in enumerate([128, 77]):
        _close(logits[b], _model(prompts[b, :n], params)[-1])
    new = jax.random.randint(jax.random.key(2), (2, 1), 0, TINY.vocab)
    want = [
        _model(np.concatenate([np.asarray(prompts[b, :n]), np.asarray(new[b])]), params)[-1]
        for b, n in enumerate([128, 77])
    ]
    for method in ("ragged_dot", "dense"):
        copy = jax.tree.map(jnp.copy, cache)
        step, _, _ = baseline_decode_step(method)(new, copy, lengths, plain)
        for b in range(2):
            _close(step[b, 0], want[b])
    # autokernel's decode step reads the baseline's cache ([k_rope | 0] rotary slots) as its own.
    ours, _, _ = decode_step(new, cache, lengths, packed)
    for b in range(2):
        _close(ours[b, 0], want[b])


def test_splash_baseline_matches_numpy(model):
    params, _ = model
    prompts = jax.random.randint(jax.random.key(1), (1, 256), 0, TINY.vocab)
    logits, _, _ = baseline_prefill("splash", 128, 128, interpret=not ON_TPU)(prompts, baseline_params(params), 256)
    _close(logits[0], _model(prompts[0], params)[-1])


def test_kv_cache_bytes_per_token():
    sizes = kv_bytes_per_token(CONFIGS["deepseek-v3"])
    assert sizes == {"latent": 4 * 576 * 2, "latent, stored": 4 * 640 * 2, "MHA": 4 * 32 * 320 * 2}


# --------------------------------------------------------------------------- Mosaic, at the benchmark's shapes


def _b16(*shape):
    return spec(*shape, dtype=B16)


def _lowers(kernel, specs, tiles=None):
    lowered = kernel.lower(*specs) if tiles is None else kernel.lower(*specs, tiles)
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize("name", ["deepseek-v3", "kimi-k2"])
@pytest.mark.parametrize(("batch", "seq"), [(1, 4096), (2, 2048)])
def test_prefill_kernels_lower_for_v5e(name, batch, seq):
    c = CONFIGS[name]
    d, h, rq, rkv, t = c.d_model, c.heads, c.q_rank, c.kv_rank, batch * seq
    down = (_b16(batch, seq, d), _b16(1, d), _b16(d, rq + rkv + LANES), _b16(1, rq), _b16(1, rkv))
    _lowers(mla_down, (*down, spec(seq, LANES), spec(seq, LANES)))
    _lowers(matmul, (_b16(t, rq), _b16(rq, h * QK)))
    _lowers(matmul, (_b16(t, rkv), _b16(rkv, 2 * h * NOPE)))
    attention = (_b16(batch, seq, d), _b16(batch, seq, h * QK), _b16(batch, seq, 2 * h * NOPE))
    attention += (_b16(batch, seq, LANES), spec(seq, LANES), spec(seq, LANES), _b16(h, NOPE, d))
    _lowers(mla_attention, attention, AttentionTiles(block_q=512, block_kv=512))
    _lowers(swiglu_ffn, (_b16(t, d), _b16(1, d), _b16(d, c.dense_ffn), _b16(d, c.dense_ffn), _b16(c.dense_ffn, d)))
    f, e, group = c.expert_ffn, c.experts, 256
    rows = capacity(t * c.top_k, e, group)
    experts = (_b16(rows, d), _b16(e, d, f), _b16(e, d, f), _b16(e, f, d), spec(rows // group, dtype=I32))
    _lowers(expert_ffn, experts, ExpertTiles(block_f=512))
    _lowers(shared_experts, (_b16(t, d), _b16(1, d), _b16(d, f), _b16(d, f), _b16(f, d), _b16(t, d)), SharedTiles())
    _lowers(rmsnorm_matmul, (_b16(batch, d), _b16(1, d), _b16(d, c.vocab)))


@pytest.mark.parametrize("name", ["deepseek-v3", "kimi-k2"])
@pytest.mark.parametrize(("batch", "s_max"), [(1, 4608), (8, 2560), (32, 4608)])
def test_decode_kernels_lower_for_v5e(name, batch, s_max):
    c = CONFIGS[name]
    d, h, rq, rkv = c.d_model, c.heads, c.q_rank, c.kv_rank
    down = (_b16(1, batch, d), _b16(1, d), _b16(d, rq + rkv + LANES), _b16(1, rq), _b16(1, rkv))
    _lowers(mla_down, (*down, spec(batch, LANES), spec(batch, LANES)))
    _lowers(matmul, (_b16(batch, rq), _b16(rq, h * (rkv + ROPE))), MatmulTiles(tm=512, tn=1024, tk=512))
    queries = (_b16(batch, 1, h, rkv), _b16(batch, 1, h, LANES), _b16(batch, s_max, rkv), _b16(batch, s_max, LANES))
    _lowers(mla_decode, (*queries, spec(batch, dtype=I32)), DecodeTiles(block_kv=512))
    heads = (_b16(batch, rq), _b16(h, rq, NOPE), _b16(h, rq, ROPE), _b16(h, rkv, NOPE))
    _lowers(latent_query_heads, heads, HeadTiles(heads=4))
    tail = (_b16(batch, d), _b16(batch, h * rkv), _b16(h * rkv, d), _b16(1, d))
    tail_heads = (_b16(batch, d), _b16(h, batch, rkv), _b16(h, rkv, NOPE), _b16(h, NOPE, d), _b16(1, d))
    for f in (c.dense_ffn, c.expert_ffn * c.shared):
        _lowers(decode_tail, (*tail, _b16(d, f), _b16(d, f), _b16(f, d)))
        _lowers(decode_tail_heads, (*tail_heads, _b16(d, f), _b16(d, f), _b16(f, d)))
    f, e, group = c.expert_ffn, c.experts, 16 if batch < 16 else 32
    rows = capacity(batch * c.top_k, e, group)
    experts = (_b16(rows, d), _b16(e, d, f), _b16(e, d, f), _b16(e, f, d), spec(rows // group, dtype=I32))
    _lowers(expert_ffn, experts)
    _lowers(rmsnorm_matmul, (_b16(batch, d), _b16(1, d), _b16(d, c.vocab)), NormMatmulTiles())


# --------------------------------------------------------------------------- on a TPU v5e


def _agree(ours, theirs):
    """Logits rows that agree: correlated to 0.999 (bf16 differences can route a token to another expert
    at a near tie, which moves its logits by more than rounding, so elementwise bounds would be brittle)."""
    for a, b in zip(f64(ours).reshape(-1, ours.shape[-1]), f64(theirs).reshape(-1, theirs.shape[-1]), strict=True):
        assert np.corrcoef(a, b)[0, 1] > 0.999


@pytest.mark.tpu
def test_prefill_and_decode_natively_match_the_baselines():
    config = Config(4, 2048, 16, 768, 512, 64, 8, 512, 4096, 32768, n_group=8, topk_group=4)
    params = init_params(config)
    plain = baseline_params(params)
    prompts = jax.random.randint(jax.random.key(1), (2, 2048), 0, config.vocab)
    # Plain JAX's f32 router matmul is one bf16 pass on a TPU (ours is accurate to 2^-17): at full precision,
    # both route the same tokens to the same experts.
    with jax.default_matmul_precision("highest"):
        theirs, their_cache, lengths = baseline_prefill()(prompts, plain, 2304)
        token = jnp.argmax(theirs, axis=-1).astype(I32)[:, None]
        their_step, _, _ = baseline_decode_step()(token, jax.tree.map(jnp.copy, their_cache), lengths, plain)
    for form in ("folded", "factored"):
        packed = pack(params, decode=form)
        ours, cache, lengths = prefill(prompts, packed, max_len=2304)
        _agree(ours, theirs)
        np.testing.assert_allclose(f64(cache[0]["c_kv"]), f64(their_cache[0]["c_kv"]), atol=5e-2)
        ours, _, _ = decode_step(token, cache, lengths, packed)
        _agree(ours, their_step)
