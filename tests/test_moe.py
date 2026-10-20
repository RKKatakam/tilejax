"""Mixture-of-experts layers (examples/moe.py): routers, dispatch, the grouped kernels and whole layers against
float64 NumPy; the kernels cross-lowered for a v5e at realistic sizes; on a TPU, against XLA's ragged_dot and
JAX's megablox."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, rand, run_checking_races, spec
from moe import (
    BLOCKS,
    CONFIGS,
    CombineTiles,
    DecodeTiles,
    ExpertTiles,
    GmmTiles,
    MoeConfig,
    RouterTiles,
    SharedTiles,
    baseline,
    baseline_params,
    best_groups,
    block_dispatch,
    capacity,
    choose_group,
    combine_rows,
    counts_before,
    decode_call,
    default_group,
    dispatch,
    expert_ffn,
    expert_ffn_blocks,
    expert_inputs,
    experts_specs,
    gather,
    gmm_down,
    gmm_swiglu,
    group_candidates,
    init_params,
    megablox_tiling,
    moe_decode,
    moe_decode_shared,
    moe_layer,
    moe_layer_ragged,
    norm_router,
    ragged_rows,
    rmsnorm,
    route,
    route_tokens,
    shared_combine,
    shared_experts,
    split_router,
    tile_experts,
    top_k,
)

SIGMOID = MoeConfig(256, 8, 2, 256, shared=1, scoring="sigmoid", n_group=4, topk_group=2, scale=2.5)
SOFTMAX = MoeConfig(256, 8, 2, 256)


# --------------------------------------------------------------------------- float64 NumPy references


def _silu(v):
    return v / (1 + np.exp(-v))


def _swiglu(x, w_gate, w_up, w_down):
    return (_silu(x @ f64(w_gate)) * (x @ f64(w_up))) @ f64(w_down)


def _route(normed, w_router, bias, config):
    """The router in float64: experts (T, k) in decreasing order of choice, and their weights."""
    logits = f64(normed) @ f64(w_router)
    if config.scoring == "softmax":
        scores = np.exp(logits - logits.max(1, keepdims=True))
        scores /= scores.sum(1, keepdims=True)
        choice = scores.copy()
    else:
        scores = 1 / (1 + np.exp(-logits))
        choice = scores + f64(bias)
    t, e = choice.shape
    if config.n_group > 1:
        groups = choice.reshape(t, config.n_group, -1)
        group_scores = np.sort(groups, axis=2)[:, :, -2:].sum(2)
        best = np.argsort(-group_scores, axis=1)[:, : config.topk_group]
        kept = np.zeros((t, config.n_group), bool)
        kept[np.arange(t)[:, None], best] = True
        choice = np.where(np.repeat(kept, e // config.n_group, axis=1), choice, -np.inf)
    ids = np.argsort(-choice, axis=1, kind="stable")[:, : config.top_k]
    weights = np.take_along_axis(scores, ids, 1)
    if config.normalize:
        weights /= weights.sum(1, keepdims=True)
    return ids, weights * config.scale


def _moe(x, params, config):
    """The whole layer in float64 (the router sees the bf16 rounding of the normalized rows, as the layer's)."""
    xf = f64(x)
    normed = xf / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + 1e-6) * f64(params["norm"])
    ids, weights = _route(f64(normed.astype(np.float32).astype(jnp.bfloat16)), params["router"], params["bias"], config)
    out = xf.copy()
    for t in range(x.shape[0]):
        for e, w in zip(ids[t], weights[t], strict=True):
            out[t] += w * _swiglu(normed[t], params["w_gate"][e], params["w_up"][e], params["w_down"][e])
    if config.shared:
        s = params["shared"]
        out += _swiglu(normed, s["w_gate"], s["w_up"], s["w_down"])
    return out


def _groups(sizes, group=1):
    """starts, ends (int32) of experts with these row counts, each starting at a multiple of `group`."""
    padded = -(-np.array(sizes) // group) * group
    ends = np.cumsum(padded)
    starts = ends - padded
    return jnp.array(starts, jnp.int32), jnp.array(starts + np.array(sizes), jnp.int32)


def _grouped(x, ws, starts, ends, fn, width):
    """fn(rows of expert e, its weights) for each expert's rows, zeros elsewhere, in float64."""
    out = np.zeros((x.shape[0], width))
    for e, (s, n) in enumerate(zip(np.asarray(starts), np.asarray(ends), strict=True)):
        out[s:n] = fn(f64(x[s:n]), *(w[e] for w in ws))
    return out


def _steps(lowered, *indices):
    """The (tile, reduction block) of every non-empty step the kernel runs, from its device-made tables."""
    tile, code, count = jax.jit(lowered.entry.__globals__[f"{lowered.plan.nest.name}_steps"])(*indices)
    n = int(count[0])
    tile, code = np.asarray(tile[:n]), np.asarray(code[:n])
    full = (code & 4) == 0
    return list(zip(tile[full].tolist(), (code[full] >> 3).tolist(), strict=True))


# --------------------------------------------------------------------------- routers


@pytest.mark.parametrize(
    "config",
    [
        SOFTMAX,  # Qwen3-MoE
        MoeConfig(256, 16, 4, 256, normalize=False),  # DeepSeek-V2
        SIGMOID,  # DeepSeek-V3: group-limited
        MoeConfig(256, 32, 8, 256, scoring="sigmoid", scale=2.827),  # Kimi K2
    ],
)
def test_routers_choose_and_weigh_experts_as_numpy(config):
    params = init_params(config, jnp.float32)
    normed = rand((64, config.d_model), 3)
    ids, weights = route(normed, params["router"], params["bias"], config)
    want_ids, want_weights = _route(normed, params["router"], params["bias"], config)
    assert ids.dtype == jnp.int32 and ids.shape == (64, config.top_k)
    np.testing.assert_array_equal(np.asarray(ids), want_ids)  # random scores: no ties
    np.testing.assert_allclose(f64(weights), want_weights, rtol=1e-5, atol=1e-6)
    if config.n_group > 1:  # every expert chosen is in one of the token's best groups
        per_group = config.experts // config.n_group
        assert all(len(set(row)) <= config.topk_group for row in np.asarray(ids) // per_group)


def test_top_k_and_best_groups_choose_as_lax_top_k_with_ties():
    # Few distinct values: many ties, which both break toward the lower index.
    values = jax.random.randint(jax.random.key(0), (64, 32), 0, 6).astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(top_k(values, 5)), np.asarray(jax.lax.top_k(values, 5)[1]))
    groups = values.reshape(64, 8, 4)
    scores = jnp.sum(jax.lax.top_k(groups, 2)[0], axis=-1)
    best = jax.lax.top_k(scores, 3)[1]
    want = jnp.repeat(jnp.any(best[:, :, None] == jnp.arange(8), axis=1), 4, axis=1)
    np.testing.assert_array_equal(np.asarray(best_groups(values, 8, 3)), np.asarray(want))


def test_counts_before_is_an_exclusive_cumulative_sum():
    counts = jax.random.randint(jax.random.key(1), (300, 7), 0, 3)  # 300 tokens: blocks of 256 and a partial one
    want = np.cumsum(np.asarray(counts), axis=0) - np.asarray(counts)
    np.testing.assert_array_equal(np.asarray(counts_before(counts)), want)
    np.testing.assert_array_equal(np.asarray(counts_before(counts[:5])), want[:5])


# --------------------------------------------------------------------------- dispatch


@pytest.mark.parametrize("group", [1, 16, 64])
def test_dispatch_puts_each_token_in_its_experts_rows(group):
    ids = jax.random.randint(jax.random.key(0), (40, 3), 0, 10).at[:, 0].set(9)  # expert 9 gets 40+ rows
    ids = jnp.where(ids == 4, 5, ids)  # expert 4 gets none
    rows = capacity(120, 10, group) if group > 1 else ragged_rows(120)
    d = dispatch(ids, 10, group, rows)
    src, pos = np.asarray(d.src), np.asarray(d.pos)
    starts, ends, counts = np.asarray(d.starts), np.asarray(d.ends), np.asarray(d.counts)
    assert src.shape == (rows,) and (src[pos] == np.arange(40)[:, None]).all()
    assert (counts == np.bincount(np.asarray(ids).reshape(-1), minlength=10)).all() and counts[4] == 0
    assert (starts % group == 0).all() and (ends - starts == -(-counts // group) * group).all()
    for t in range(40):
        for s in range(3):
            e = int(ids[t, s])
            assert starts[e] <= pos[t, s] < starts[e] + counts[e]
    assert (src[np.setdiff1d(np.arange(rows), pos)] == 40).all()  # every other row reads the zero row
    if group > 1:
        experts = np.asarray(tile_experts(d, group, rows))
        used = ends[-1] // group
        assert (experts[used:] == -1).all() and 4 not in experts
        for t, e in enumerate(experts[:used]):
            assert starts[e] <= t * group < ends[e]


def _routing(tokens, k, shares, seed):
    """Random routing, k distinct experts per token (as a router chooses), expert e with odds `shares[e]`."""
    rng = np.random.default_rng(seed)
    p = np.array(shares) / sum(shares)
    return jnp.array([rng.choice(len(shares), k, replace=False, p=p) for _ in range(tokens)], jnp.int32)


@pytest.mark.parametrize("group", [128, 256])
def test_block_dispatch_tiles_each_experts_rows(group):
    # 300 tokens x 3 of 7 experts: one without rows, the rest 0 to 3 tiles each, rows meeting mid-block.
    ids = _routing(300, 3, [6, 1, 0, 2, 3, 1, 1], 0)
    b = block_dispatch(ids, 7, group)
    r = group // BLOCKS
    src, pos = np.asarray(b.src), np.asarray(b.pos)
    experts, blocks = np.asarray(b.tile_expert), np.asarray(b.tile_blocks).reshape(-1, BLOCKS)
    counts = np.bincount(np.asarray(ids).reshape(-1), minlength=7)
    assert counts[2] == 0 and counts.max() > group
    assert experts.shape == (capacity(900, 7, group) // group,) and src.shape == (capacity(900, 7, r),)
    assert (np.bincount(src[src < 300], minlength=300) == 3).all() and (src[src >= 300] == 300).all()
    want = [e for e, n in enumerate(counts) for _ in range(-(-n // group))]
    assert list(experts[: len(want)]) == want and (experts[len(want) :] == -1).all()
    assert (blocks >= 0).all() and (blocks < src.size // r).all()
    # Each slot's output row is in a tile of its expert, where the tile's blocks hold the slot's token.
    assert len(set(pos.reshape(-1).tolist())) == 900
    for t in range(300):
        for s in range(3):
            tile, row = divmod(int(pos[t, s]), group)
            assert experts[tile] == ids[t, s] and src[blocks[tile, row // r] * r + row % r] == t


def test_capacity_is_the_most_rows_a_routing_can_need():
    # 8 slots of 8 experts in groups of 16: every expert can get one row, so each needs a tile.
    assert capacity(8, 8, 16) == 128 and capacity(64, 128, 16) == 1024 and capacity(4096 * 8, 128, 256) == 65536
    assert ragged_rows(64) == 64 and ragged_rows(600) == 1024 and ragged_rows(24) == 32
    assert default_group(8) == 16 and default_group(64) == 64 and default_group(4096) == 256
    qwen, kimi, v2 = CONFIGS["qwen3-30b-a3b"], CONFIGS["kimi-k2-small"], CONFIGS["deepseek-v2-lite"]
    assert group_candidates(8, qwen) == (16,) and group_candidates(64, qwen) == (16, 32, 64)
    assert group_candidates(4096, qwen) == (256, 384, 512) and group_candidates(4096, kimi) == (128, 256)
    assert group_candidates(4096, v2) == (384, 512, 768) and group_candidates(16384, qwen) == (512, 768, 1024)


# --------------------------------------------------------------------------- the grouped kernels


@pytest.mark.parametrize(
    ("dtype", "group", "block_f"),
    [(jnp.float32, 16, 128), (jnp.bfloat16, 32, 256)],
)
def test_expert_ffn_against_numpy(dtype, group, block_f):
    # Uneven experts: two without rows, one over several tiles, the layout's last tiles unused.
    sizes = [5, 0, 70, 16, 0, 1]
    d, f = 256, 256
    rows = capacity(sum(sizes), len(sizes), group)
    starts, ends = _groups(sizes, group)
    xs = np.zeros((rows, d), np.float32)
    for s, n in zip(np.asarray(starts), np.asarray(ends), strict=True):
        xs[s:n] = rand((n - s, d), int(s) + 1)
    xs = jnp.asarray(xs, dtype)
    w_gate, w_up = rand((6, d, f), 1, dtype) * 0.1, rand((6, d, f), 2, dtype) * 0.1
    w_down = rand((6, f, d), 3, dtype) * 0.1
    experts = [e for e, n in enumerate(sizes) for _ in range(-(-n // group))]
    experts = jnp.array(experts + [-1] * (rows // group - len(experts)), jnp.int32)
    tiles = ExpertTiles(block_f=block_f)
    args = (xs, w_gate, w_up, w_down, experts, tiles)
    out = run_checking_races(expert_ffn, *args)
    want = _grouped(xs, (w_gate, w_up, w_down), starts, ends, _swiglu, d)
    tol = 1e-4 if dtype == jnp.float32 else 3e-2 * np.abs(want).max()
    np.testing.assert_allclose(f64(out), want, rtol=tol, atol=tol)
    expert_ffn.verify(*args)
    # Only the experts with rows are read, each once per tile of its rows, every block of their weights.
    lowered = expert_ffn.lower(*args)
    tiles_run = _steps(lowered, experts)
    nf = f // block_f
    per_expert = {e: -(-n // group) for e, n in enumerate(sizes) if n}
    assert len(tiles_run) == nf * sum(per_expert.values())
    assert {int(experts[t]) for t, _ in tiles_run} == set(per_expert)


@pytest.mark.parametrize(("group", "block_f"), [(128, 256), (256, 128)])
def test_expert_ffn_blocks_against_numpy(group, block_f):
    # 300 tokens x 3 of 7 experts: one without rows, one over 3 tiles; tiles read past their expert's rows.
    ids = _routing(300, 3, [6, 1, 0, 2, 3, 1, 1], 0)
    d, f = 256, 256
    normed = rand((300, d), 4, jnp.bfloat16)
    w_gate, w_up = rand((7, d, f), 1, jnp.bfloat16) * 0.1, rand((7, d, f), 2, jnp.bfloat16) * 0.1
    w_down = rand((7, f, d), 3, jnp.bfloat16) * 0.1
    b = block_dispatch(ids, 7, group)
    xs = gather(normed, b.src).reshape(-1, group // BLOCKS, d)
    args = (xs, w_gate, w_up, w_down, b.tile_expert, b.tile_blocks, ExpertTiles(block_f=block_f))
    out = f64(run_checking_races(expert_ffn_blocks, *args))
    rows = out[np.asarray(b.pos)]  # (300, 3, d): each slot's row
    want = np.stack([_swiglu(f64(normed), w_gate[e], w_up[e], w_down[e]) for e in range(7)])  # (7, 300, d)
    want = want[np.asarray(ids), np.arange(300)[:, None]]
    np.testing.assert_allclose(rows, want, rtol=3e-2, atol=3e-2 * np.abs(want).max())
    assert (out[group * int(np.sum(np.asarray(b.tile_expert) >= 0)) :] == 0).all()  # the unused tiles
    expert_ffn_blocks.verify(*args)
    # Only the tiles with rows run, every block of hidden units.
    steps = _steps(expert_ffn_blocks.lower(*args), b.tile_expert, b.tile_blocks)
    counts = np.bincount(np.asarray(ids).reshape(-1), minlength=7)
    assert len(steps) == (f // block_f) * sum(-(-n // group) for n in counts)


def test_norm_router_against_numpy():
    x = rand((256, 512), 0, jnp.bfloat16)
    gamma = (1 + 0.1 * rand((1, 512), 1)).astype(jnp.bfloat16)
    w_router = rand((512, 128), 2) / 23
    args = (x, gamma, *split_router(w_router), RouterTiles(rows=128))
    normed, logits = run_checking_races(norm_router, *args)
    xf = f64(x)
    want = xf / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + 1e-6) * f64(gamma)
    np.testing.assert_allclose(f64(normed), want, rtol=1e-2, atol=1e-2)
    assert logits.dtype == jnp.float32  # the f32 router on the bf16 rows, to ~2^-17
    np.testing.assert_allclose(f64(logits), f64(normed) @ f64(w_router), rtol=1e-4, atol=1e-4)
    norm_router.verify(*args)


@pytest.mark.parametrize(("dtype", "tm", "tn"), [(jnp.float32, 64, 128), (jnp.bfloat16, 128, 256)])
def test_grouped_matmuls_against_numpy(dtype, tm, tn):
    # Unpadded: experts meet mid-tile; two have no rows; one spans several tiles; the last rows are no one's.
    sizes = [10, 0, 150, 30, 0, 40]
    rows, d, f = 256, 256, 256
    starts, ends = _groups(sizes)
    xs = rand((rows, d), 0, dtype)
    w_gate, w_up = rand((6, d, f), 1, dtype) * 0.1, rand((6, d, f), 2, dtype) * 0.1
    w_down = rand((6, f, d), 3, dtype) * 0.1
    tiles = GmmTiles(tm=tm, tn=tn)
    h = run_checking_races(gmm_swiglu, xs, w_gate, w_up, starts, ends, tiles)
    want_h = _grouped(xs, (w_gate, w_up), starts, ends, lambda x, g, u: _silu(x @ f64(g)) * (x @ f64(u)), f)
    tol = 1e-4 if dtype == jnp.float32 else 3e-2 * np.abs(want_h).max()
    np.testing.assert_allclose(f64(h), want_h, rtol=tol, atol=tol)
    y = run_checking_races(gmm_down, h, w_down, starts, ends, tiles)
    want_y = _grouped(h, (w_down,), starts, ends, lambda x, w: x @ f64(w), d)
    tol = 1e-4 if dtype == jnp.float32 else 3e-2 * np.abs(want_y).max()
    np.testing.assert_allclose(f64(y), want_y, rtol=tol, atol=tol)
    gmm_swiglu.verify(xs, w_gate, w_up, starts, ends, tiles)
    gmm_down.verify(h, w_down, starts, ends, tiles)
    # Each row tile's steps are the experts whose rows meet it, in order: never one without rows.
    steps = _steps(gmm_swiglu.lower(xs, w_gate, w_up, starts, ends, tiles), starts, ends)
    meets = [
        (t, e)
        for t in range(rows // tm)
        for e, n in enumerate(sizes)
        if n and starts[e] < (t + 1) * tm and ends[e] > t * tm
    ]
    assert steps == meets


def test_shared_experts_against_numpy():
    x, routed = rand((256, 256), 0, jnp.bfloat16), rand((256, 256), 1, jnp.bfloat16)
    gamma = (1 + 0.1 * rand((1, 256), 2)).astype(jnp.bfloat16)
    w_gate, w_up = rand((256, 512), 3, jnp.bfloat16) / 16, rand((256, 512), 4, jnp.bfloat16) / 16
    w_down = rand((512, 256), 5, jnp.bfloat16) / 23
    args = (x, gamma, w_gate, w_up, w_down, routed, SharedTiles(rows=256, block_f=256))
    out = run_checking_races(shared_experts, *args)
    xf = f64(x)
    normed = xf / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + 1e-6) * f64(gamma)
    want = xf + f64(routed) + _swiglu(normed, w_gate, w_up, w_down)
    np.testing.assert_allclose(f64(out), want, rtol=2e-2, atol=3e-2 * np.abs(want).max())
    shared_experts.verify(*args)


@pytest.mark.parametrize("shared", [False, True])
def test_combine_kernels_against_numpy(shared):
    k, t, d, f = 3, 256, 256, 512
    x, rows = rand((t, d), 0, jnp.bfloat16), rand((k, t, d), 1, jnp.bfloat16)
    weights = jax.random.uniform(jax.random.key(2), (k, t, 1))
    want = f64(x) + np.sum(f64(weights) * f64(rows), axis=0)
    if shared:
        gamma = (1 + 0.1 * rand((1, d), 2)).astype(jnp.bfloat16)
        w_gate, w_up = rand((d, f), 3, jnp.bfloat16) / 16, rand((d, f), 4, jnp.bfloat16) / 16
        w_down = rand((f, d), 5, jnp.bfloat16) / 23
        kernel, args = (
            shared_combine,
            (x, gamma, w_gate, w_up, w_down, rows, weights, SharedTiles(rows=128, block_f=256)),
        )
        xf = f64(x)
        want += _swiglu(xf / np.sqrt(np.mean(xf * xf, 1, keepdims=True) + 1e-6) * f64(gamma), w_gate, w_up, w_down)
    else:
        kernel, args = combine_rows, (x, rows, weights, CombineTiles(rows=128))
    out = run_checking_races(kernel, *args)
    np.testing.assert_allclose(f64(out), want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
    kernel.verify(*args)


@pytest.mark.parametrize(("config", "tokens"), [(SOFTMAX, 8), (SIGMOID, 64)])
def test_decode_kernels_run_each_chosen_expert_once_against_numpy(config, tokens):
    params = init_params(config)
    x = rand((tokens, config.d_model), 9, jnp.bfloat16)
    kernel, args = decode_call(x, params, config)
    assert kernel is (moe_decode_shared if config.shared else moe_decode)
    if config.shared:
        args = (*args, DecodeTiles(block_f=128))
    out = run_checking_races(kernel, *args)
    want = _moe(x, params, config)
    np.testing.assert_allclose(f64(out), want, rtol=2e-2, atol=3e-2 * np.abs(want).max())
    kernel.verify(*args)
    # A step per expert some token chose (its weights read once), and the shared experts' blocks.
    counts = args[5]
    shared = config.shared_ffn // 128 if config.shared else 0
    assert len(_steps(kernel.lower(*args), counts)) == int(np.sum(np.asarray(counts) > 0)) + shared


# --------------------------------------------------------------------------- whole layers


@pytest.mark.parametrize(
    ("config", "tokens", "group"),
    [(SIGMOID, 128, 32), (SOFTMAX, 64, None), (SIGMOID, 8, None), (SIGMOID, 128, None), (SOFTMAX, 512, 256)],
)
def test_moe_layer_against_numpy(config, tokens, group):
    params = init_params(config)
    x = rand((tokens, config.d_model), 7, jnp.bfloat16)
    want = _moe(x, params, config)
    tol = 3e-2 * np.abs(want).max()
    for out in (moe_layer(x, params, config, group), moe_layer_ragged(x, params, config)):
        assert out.dtype == jnp.bfloat16
        np.testing.assert_allclose(f64(out), want, rtol=2e-2, atol=tol)
    jitted = jax.jit(moe_layer, static_argnums=(2, 3))(x, params, config, group)
    np.testing.assert_allclose(f64(jitted), want, rtol=2e-2, atol=tol)


def test_decode_reads_only_the_experts_tokens_chose():
    # 4 tokens x 2 experts of 32: at most 8 experts have rows, and each is one tile (the group >= the tokens).
    config = MoeConfig(256, 32, 2, 256)
    params = init_params(config)
    x = rand((4, config.d_model), 5, jnp.bfloat16)
    group = choose_group(x, params, config)
    assert group == 16
    inputs = expert_inputs(x, params, config, group)[0]
    experts = np.asarray(inputs[-1])
    chosen = set(np.asarray(route(rmsnorm(x, params["norm"]), params["router"], params["bias"], config)[0]).flat)
    steps = _steps(expert_ffn.lower(*inputs), inputs[-1])
    assert len(steps) == len(chosen) and {int(experts[t]) for t, _ in steps} == chosen  # one step each
    assert expert_ffn.lower(*inputs).tiles == 8  # room for 8 experts of a row each; the unused take empty steps


@pytest.mark.parametrize("method", ["ragged_dot", "dense", "megablox"])
def test_baselines_against_numpy(method):
    params = init_params(SIGMOID)
    x = rand((128, SIGMOID.d_model), 7, jnp.bfloat16)
    want = _moe(x, params, SIGMOID)
    out = baseline(SIGMOID, method, megablox_tiling(128, 128))(x, baseline_params(params))
    np.testing.assert_allclose(f64(out), want, rtol=2e-2, atol=3e-2 * np.abs(want).max())


# --------------------------------------------------------------------------- Mosaic, at realistic sizes


@pytest.mark.parametrize(
    ("name", "tokens", "group", "block_f"),
    [
        ("qwen3-30b-a3b", 4096, 256, 256),
        ("deepseek-v2-lite", 16384, 512, 128),
        ("kimi-k2-small", 16384, 256, 512),
        ("deepseek-v3-small", 64, 64, 512),
    ],
)
def test_expert_ffn_lowers_for_v5e(name, tokens, group, block_f):
    config = CONFIGS[name]
    d, e, f = config.d_model, config.experts, config.ffn
    rows = capacity(tokens * config.top_k, e, group)
    specs = [
        spec(rows, d, dtype=jnp.bfloat16),
        spec(e, d, f, dtype=jnp.bfloat16),
        spec(e, d, f, dtype=jnp.bfloat16),
        spec(e, f, d, dtype=jnp.bfloat16),
        spec(rows // group, dtype=jnp.int32),
    ]
    lowered = expert_ffn.lower(*specs, ExpertTiles(block_f=block_f))
    assert lowered.schedule.runtime and lowered.vmem_bytes < 64 * 2**20
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(
    ("name", "tokens", "group", "block_f"),
    [
        ("qwen3-30b-a3b", 4096, 256, 1024),  # 768 hidden units: one block, the weights per tile
        ("qwen3-30b-a3b", 16384, 1024, 256),
        ("deepseek-v2-lite", 16384, 512, 2048),
        ("kimi-k2-small", 4096, 128, 512),
        ("deepseek-v3-small", 4096, 384, 128),
    ],
)
def test_expert_ffn_blocks_lowers_for_v5e(name, tokens, group, block_f):
    kernel, specs = experts_specs(CONFIGS[name], tokens, group)
    assert kernel is expert_ffn_blocks
    lowered = kernel.lower(*specs, ExpertTiles(block_f=block_f))
    assert lowered.schedule.runtime and lowered.vmem_bytes < 96 * 2**20
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(("tokens", "experts"), [(4096, 128), (16384, 64), (8, 384)])
def test_norm_router_lowers_for_v5e(tokens, experts):
    specs = [
        spec(tokens, 2048, dtype=jnp.bfloat16),
        spec(1, 2048, dtype=jnp.bfloat16),
        spec(2048, experts, dtype=jnp.bfloat16),
        spec(2048, experts, dtype=jnp.bfloat16),
    ]
    assert "tpu_custom_call" in lower_for_v5e(norm_router.lower(*specs), *specs)


@pytest.mark.parametrize(
    ("name", "tokens", "tm", "tn"),
    [("qwen3-30b-a3b", 4096, 256, 1024), ("deepseek-v2-lite", 16384, 512, 2048), ("kimi-k2-small", 4096, 256, 512)],
)
def test_grouped_matmuls_lower_for_v5e(name, tokens, tm, tn):
    config = CONFIGS[name]
    d, e, f = config.d_model, config.experts, config.ffn
    rows = ragged_rows(tokens * config.top_k)
    groups = [spec(e, dtype=jnp.int32), spec(e, dtype=jnp.int32)]
    up = [spec(rows, d, dtype=jnp.bfloat16), spec(e, d, f, dtype=jnp.bfloat16), spec(e, d, f, dtype=jnp.bfloat16)]
    down = [spec(rows, f, dtype=jnp.bfloat16), spec(e, f, d, dtype=jnp.bfloat16)]
    tiles = GmmTiles(tm=tm, tn=tn)  # 1408 hidden units: one block of all of them
    for kernel, specs in ((gmm_swiglu, up + groups), (gmm_down, down + groups)):
        lowered = kernel.lower(*specs, tiles)
        assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(("name", "tokens"), [("kimi-k2-small", 64), ("deepseek-v2-lite", 64), ("qwen3-30b-a3b", 8)])
def test_decode_kernels_lower_for_v5e(name, tokens):
    kernel, specs = experts_specs(CONFIGS[name], tokens, 16)
    lowered = kernel.lower(*specs)
    assert lowered.schedule.runtime and lowered.vmem_bytes < 96 * 2**20
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


@pytest.mark.parametrize(("tokens", "k", "shared"), [(4096, 8, 0), (16384, 6, 2816)])
def test_combine_kernels_lower_for_v5e(tokens, k, shared):
    rows = [spec(k, tokens, 2048, dtype=jnp.bfloat16), spec(k, tokens, 1)]
    x = spec(tokens, 2048, dtype=jnp.bfloat16)
    if shared:
        weights = [spec(1, 2048), spec(2048, shared), spec(2048, shared), spec(shared, 2048)]
        kernel, specs = shared_combine, [x, *(spec(*w.shape, dtype=jnp.bfloat16) for w in weights), *rows]
    else:
        kernel, specs = combine_rows, [x, *rows]
    assert "tpu_custom_call" in lower_for_v5e(kernel.lower(*specs), *specs)


def test_shared_experts_lower_for_v5e():
    specs = [
        spec(4096, 2048, dtype=jnp.bfloat16),
        spec(1, 2048, dtype=jnp.bfloat16),
        spec(2048, 2816, dtype=jnp.bfloat16),
        spec(2048, 2816, dtype=jnp.bfloat16),
        spec(2816, 2048, dtype=jnp.bfloat16),
        spec(4096, 2048, dtype=jnp.bfloat16),
    ]
    lowered = shared_experts.lower(*specs)  # 2816 = 2 x 1408 hidden units: the nearest block that divides
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


# --------------------------------------------------------------------------- on a TPU v5e


@pytest.mark.tpu
@pytest.mark.parametrize("name", ["qwen3-30b-a3b", "deepseek-v3-small", "kimi-k2-small"])
@pytest.mark.parametrize("tokens", [32, 2048])
def test_layers_match_ragged_dot_and_megablox_on_tpu(name, tokens):
    config = CONFIGS[name]
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (tokens, config.d_model), jnp.bfloat16)
    group = choose_group(x, params, config)  # tunes the experts' kernel on this routing
    moe_layer_ragged(x, params, config)  # tunes the grouped matmuls on it, eagerly
    ours = jax.jit(moe_layer, static_argnums=(2, 3))(x, params, config, group)
    ragged = jax.jit(moe_layer_ragged, static_argnums=2)(x, params, config)
    # `norm_router` and XLA round the logits differently: a token at a near tie may take another expert.
    ids = np.sort(np.asarray(route_tokens(x, params, config)[1]), axis=1)
    xla_ids = np.sort(np.asarray(route(rmsnorm(x, params["norm"]), params["router"], params["bias"], config)[0]), 1)
    same = (ids == xla_ids).all(axis=1)
    assert same.mean() >= 0.99
    bparams = baseline_params(params)
    for want in (baseline(config)(x, bparams), baseline(config, "megablox", megablox_tiling(128, 512))(x, bparams)):
        for out in (ours, ragged):
            diff = jnp.abs(out.astype(jnp.float32) - want.astype(jnp.float32))[same]
            assert float(diff.max()) <= 3e-2 * float(jnp.abs(want.astype(jnp.float32)).max())
