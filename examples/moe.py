"""Mixture-of-experts feed-forward layers (Qwen3-MoE, DeepSeek-V3, Kimi K2): dropless, SwiGLU experts, as
autokernel kernels composed in plain JAX.

    normed = rmsnorm(x)
    ids, weights = route(normed @ W_router)                        top-k experts per token, and their weights
    out = x + sum_k weights[t, k] * expert[ids[t, k]](normed[t]) + shared(normed[t])
    expert_e(v) = (silu(v @ W_gate[e]) * (v @ W_up[e])) @ W_down[e]

Routers: Qwen3-MoE takes the top-k of the softmax of the logits and renormalizes their weights; DeepSeek-V3
and Kimi K2 score experts with a sigmoid, choose by the scores plus a per-expert bias (load balancing
without an auxiliary loss), first keeping only the `topk_group` best of `n_group` expert groups (each
group scored by its two best experts), and scale the renormalized weights by `routed_scaling_factor`; they
add shared experts that every token goes through.

The layer, `moe_layer`, is four kernels and some XLA between them:

1. `norm_router`: RMSNorm and the router logits in one pass over x. The logits are f32 from the bf16
   normalized rows: the f32 router is split into two bf16 matrices (`split_router`, hi + lo), so two MXU
   passes give the f32 product to ~2^-17 (XLA's f32 `HIGHEST` dot takes six).
2. Routing and dispatch in XLA, without scatters or gathers of single elements (with them, the glue
   around the experts cost more than the experts at 4096 tokens on a v5e). The chosen experts are a
   one-hot (tokens, k, experts) mask: its sums are the rows per expert, a cumulative
   sum over tokens ranks each (token, slot) among its expert's, and each slot's row is its expert's first
   row plus its rank, all elementwise and reductions. The rows' tokens (the inverse map) come from one
   sort of an int32 key per slot, read as windows: each block of rows of one expert is a slice of the
   sorted tokens. The tokens' normalized rows are then gathered into the experts' rows: a row gather, which
   costs ~10-20 ns a row on a v5e (from the time padded layouts added to whole layers), so the layouts keep
   the rows gathered few.
3. The routed experts: one grouped SwiGLU kernel over the experts' rows (below).
4. The combine: each token's weighted sum of its k expert rows, a row gather and a sum that XLA fuses with
   the residual; with shared experts, `shared_experts` (one kernel: RMSNorm, the shared SwiGLU, the
   residual and the routed sum).

The grouped kernels. Every step reads a whole block of an expert's weights for the rows of a tile: G flops
per weight byte for G rows, so prefill wants G >= ~256 on a v5e (197 TFLOP/s, 819 GB/s), decode wants G
about the tokens (an expert gets at most a row per token), and fine-grained experts (DeepSeek-V3, Kimi K2:
~85-128 rows each at 4096 tokens) want G about their rows, or padding wastes up to half the MXU work.

- `expert_ffn` (decode: G < 128): each expert's rows padded to a multiple of G, so every tile of G rows
  belongs to one expert; `tile_expert` (an index array, in SMEM) says which, -1 past the used rows. A tile
  is a feed-forward layer of its own: its reduction loop walks blocks of the expert's hidden units,
  streaming `W_gate[e][:, f]`, `W_up[e][:, f]` and `W_down[e][f, :]` (runtime windows), with SwiGLU between
  the matmuls, so the hidden activations never go to HBM. Its mask (`tile_expert[i // G] < 0`) is a
  runtime mask: unused tiles take one empty step, and an expert without tokens has no tile, so its weights
  are never read. A step reads only the experts some token chose: decode stays bandwidth bound.
- `expert_ffn_blocks` (prefill: G >= 128): the same tiles, but their rows come from a layout padded only
  to G/8 rows per expert, as 8 runtime windows of G/8 rows (`tile_blocks`): the row gather writes the
  slots plus at most G/8 - 1 rows per expert instead of up to G - 1 (for Kimi K2 at 4096 tokens and
  G = 128, 39K rows instead of 82K). The last tile of an expert reads the next expert's rows into rows the
  combine never reads. When a block of hidden units covers the whole expert, its weights are per-tile
  windows placed by an index array: consecutive tiles of one expert read them once, so an expert's
  weights are read once whatever its rows, and G can follow the rows per expert, not the MXU.
  `choose_group` measures the group sizes about the rows per expert (`group_candidates`).
- `gmm_swiglu` and `gmm_down` are the megablox-style alternative, on unpadded rows: row tiles of any size
  (the tunable's), each a reduction over the experts whose rows meet the tile (a runtime mask on the group
  offsets), rows of other experts masked out of the result. A tile meets 1 + (experts starting inside it)
  experts, and each such step does a whole tile's matmul.

`baseline` is the same layer in plain JAX (routing and dispatch as above) with XLA's `jax.lax.ragged_dot`,
JAX's Pallas megablox `gmm`, or (for decode) every expert on every token, one-hot weighted.

Run with `uv run python examples/moe.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

import autokernel

EPS = 1e-6
SUBLANES = 16  # bf16 rows per (16, 128) VMEM tile: the smallest group size
GROUPS = (16, 32, 64)  # `expert_ffn`'s group sizes (decode)
BLOCK_GROUPS = (128, 256, 384, 512, 768, 1024)  # `expert_ffn_blocks`' group sizes (prefill)
BLOCKS = 8  # row windows per tile of `expert_ffn_blocks`


@dataclass(frozen=True)
class MoeConfig:
    """A mixture-of-experts layer: model width, routed experts (each `ffn` hidden units), experts per token,
    shared experts (always on, together one SwiGLU of `shared * ffn` hidden units) and the router."""

    d_model: int
    experts: int
    top_k: int
    ffn: int
    shared: int = 0
    scoring: str = "softmax"  # "softmax" (Qwen3, DeepSeek-V2) or "sigmoid" (DeepSeek-V3, Kimi K2)
    normalize: bool = True  # renormalize the top-k weights to sum to 1
    n_group: int = 1  # expert groups (group-limited routing: only the best `topk_group` groups are used)
    topk_group: int = 1
    scale: float = 1.0  # routed_scaling_factor

    @property
    def shared_ffn(self) -> int:
        return self.shared * self.ffn


# Small but realistically shaped layers (weights 1-2.5 GB in bf16).
CONFIGS = {
    "qwen3-30b-a3b": MoeConfig(2048, 128, 8, 768),
    "deepseek-v2-lite": MoeConfig(2048, 64, 6, 1408, shared=2, normalize=False),
    "deepseek-v3-small": MoeConfig(2048, 256, 8, 512, shared=1, scoring="sigmoid", n_group=8, topk_group=4, scale=2.5),
    "kimi-k2-small": MoeConfig(2048, 384, 8, 512, shared=1, scoring="sigmoid", scale=2.827),
}


# --------------------------------------------------------------------------- kernels


@autokernel.tunable
class RouterTiles:
    """Rows per tile."""

    rows: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


@autokernel.kernel
def norm_router(x, gamma, w_hi, w_lo, tiles=RouterTiles()):
    """rmsnorm(x) (x's dtype) and its router logits (f32), normed @ (w_hi + w_lo), on (tokens, d_model) rows:
    x is read once, the logits are two bf16 MXU passes accumulated in f32."""
    T, D = x.shape
    E = w_hi.shape[1]
    tm = min(tiles.rows, T)
    dtype = x.dtype
    normed = jnp.zeros((T, D), dtype)
    logits = jnp.zeros((T, E), jnp.float32)
    for i in range(0, T, tm):
        xf = x[i : i + tm, :].astype(jnp.float32)
        n = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=1, keepdims=True) + EPS) * gamma[0:1, :].astype(jnp.float32)
        n = n.astype(dtype)
        normed = normed.at[i : i + tm, :].set(n)
        logits = logits.at[i : i + tm, :].set(n @ w_hi[:, :] + n @ w_lo[:, :])
    return normed, logits


@autokernel.tunable
class ExpertTiles:
    """Hidden units of an expert per step (the rows per tile are the layout's group size)."""

    block_f: int = autokernel.choice(128, 256, 512, 1024, 2048, default=256)


@autokernel.kernel
def expert_ffn(xs, w_gate, w_up, w_down, tile_expert, tiles=ExpertTiles()):
    """Each tile of G rows of `xs` through its expert's SwiGLU feed-forward layer (`tile_expert`, or -1 for an
    unused tile, whose rows come out as zeros)."""
    M, D = xs.shape
    F = w_gate.shape[2]
    G = M // tile_expert.shape[0]
    bf = min(tiles.block_f, F)
    dtype = xs.dtype
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, G):
        acc = jnp.zeros((G, D), jnp.float32)
        for f in range(0, F, bf):  # the expert's hidden units, bf per step
            if tile_expert[i // G] < 0:  # past the experts' rows: decided when the kernel runs
                continue
            xt = xs[i : i + G, :]
            g = xt @ w_gate[tile_expert[i // G], :, f : f + bf]
            u = xt @ w_up[tile_expert[i // G], :, f : f + bf]
            acc += (jax.nn.silu(g) * u).astype(dtype) @ w_down[tile_expert[i // G], f : f + bf, :]
        out = out.at[i : i + G, :].set(acc.astype(dtype))
    return out


@autokernel.kernel
def expert_ffn_blocks(xs, w_gate, w_up, w_down, tile_expert, tile_blocks, tiles=ExpertTiles()):
    """Tiles of G = 8 R rows, each through its expert's SwiGLU feed-forward layer (`tile_expert`, -1 for an
    unused tile: zeros), its rows the blocks `tile_blocks[8 t : 8 t + 8]` of `xs`, (blocks, R, d_model)."""
    R, D = xs.shape[1], xs.shape[2]
    F = w_gate.shape[2]
    G = BLOCKS * R
    M = tile_expert.shape[0] * G
    bf = min(tiles.block_f, F)
    dtype = xs.dtype
    out = jnp.zeros((M, D), dtype)
    for i in range(0, M, G):
        acc = jnp.zeros((G, D), jnp.float32)
        for f in range(0, F, bf):  # the expert's hidden units, bf per step
            if tile_expert[i // G] < 0:  # past the experts' rows: decided when the kernel runs
                continue
            xt = jnp.concatenate(  # the tile's 8 windows of rows (i // R = 8 * its index)
                [
                    xs[tile_blocks[i // R], :, :],
                    xs[tile_blocks[i // R + 1], :, :],
                    xs[tile_blocks[i // R + 2], :, :],
                    xs[tile_blocks[i // R + 3], :, :],
                    xs[tile_blocks[i // R + 4], :, :],
                    xs[tile_blocks[i // R + 5], :, :],
                    xs[tile_blocks[i // R + 6], :, :],
                    xs[tile_blocks[i // R + 7], :, :],
                ],
                axis=0,
            )
            g = xt @ w_gate[tile_expert[i // G], :, f : f + bf]
            u = xt @ w_up[tile_expert[i // G], :, f : f + bf]
            acc += (jax.nn.silu(g) * u).astype(dtype) @ w_down[tile_expert[i // G], f : f + bf, :]
        out = out.at[i : i + G, :].set(acc.astype(dtype))
    return out


@autokernel.tunable
class GmmTiles:
    """Rows per tile, output columns per tile."""

    tm: int = autokernel.choice(64, 128, 256, 512, 1024, default=256)
    tn: int = autokernel.choice(256, 512, 1024, 2048, default=1024)


@autokernel.kernel
def gmm_swiglu(xs, w_gate, w_up, starts, ends, tiles=GmmTiles()):
    """silu(xs @ W_gate[e]) * (xs @ W_up[e]) for the rows starts[e]:ends[e] of each expert e (other rows: 0)."""
    M = xs.shape[0]
    E, _, F = w_gate.shape
    tm, tn = min(tiles.tm, M), min(tiles.tn, F)
    dtype = xs.dtype
    h = jnp.zeros((M, F), dtype)
    for j in range(0, F, tn):  # outermost, so the step tables cover one column of tiles
        for i in range(0, M, tm):
            g = jnp.zeros((tm, tn), jnp.float32)
            u = jnp.zeros((tm, tn), jnp.float32)
            for e in range(E):  # only the experts whose rows meet this tile's
                if ends[e] <= i or starts[e] >= i + tm or starts[e] >= ends[e]:
                    continue
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0)
                mine = (rows >= starts[e]) & (rows < ends[e])
                xt = xs[i : i + tm, :]
                g = jnp.where(mine, xt @ w_gate[e, :, j : j + tn], g)
                u = jnp.where(mine, xt @ w_up[e, :, j : j + tn], u)
            h = h.at[i : i + tm, j : j + tn].set((jax.nn.silu(g) * u).astype(dtype))
    return h


@autokernel.kernel
def gmm_down(h, w_down, starts, ends, tiles=GmmTiles()):
    """h @ W_down[e] for the rows starts[e]:ends[e] of each expert e (other rows: 0)."""
    M = h.shape[0]
    E, _, D = w_down.shape
    tm, tn = min(tiles.tm, M), min(tiles.tn, D)
    dtype = h.dtype
    out = jnp.zeros((M, D), dtype)
    for j in range(0, D, tn):
        for i in range(0, M, tm):
            acc = jnp.zeros((tm, tn), jnp.float32)
            for e in range(E):
                if ends[e] <= i or starts[e] >= i + tm or starts[e] >= ends[e]:
                    continue
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0)
                mine = (rows >= starts[e]) & (rows < ends[e])
                acc = jnp.where(mine, h[i : i + tm, :] @ w_down[e, :, j : j + tn], acc)
            out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
    return out


@autokernel.tunable
class SharedTiles:
    """Rows per output tile, hidden units per step."""

    rows: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=512)


@autokernel.kernel
def shared_experts(x, gamma, w_gate, w_up, w_down, routed, tiles=SharedTiles()):
    """The end of the layer: x + routed + the shared experts' SwiGLU of rmsnorm(x), on (tokens, d_model) rows."""
    T, D = x.shape
    F = w_gate.shape[1]
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        resid = x[i : i + tm, :].astype(jnp.float32)
        normed = resid * jax.lax.rsqrt(jnp.mean(resid * resid, axis=1, keepdims=True) + EPS) * gamma[0:1, :]
        normed = normed.astype(dtype)
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):  # the shared hidden units, bf per step, streaming the weights
            g = normed @ w_gate[:, f : f + bf]
            u = normed @ w_up[:, f : f + bf]
            y += (jax.nn.silu(g) * u).astype(dtype) @ w_down[f : f + bf, :]
        out = out.at[i : i + tm, :].set((resid + routed[i : i + tm, :].astype(jnp.float32) + y).astype(dtype))
    return out


# --------------------------------------------------------------------------- routing


def rmsnorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)
    return normed.astype(x.dtype)


def split_router(w_router: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An f32 matrix as two bf16 ones, hi + lo, equal to ~2^-17 relative: `norm_router`'s weights."""
    hi = w_router.astype(jnp.bfloat16)
    return hi, (w_router - hi.astype(jnp.float32)).astype(jnp.bfloat16)


def one_hot(ids: jax.Array, experts: int) -> jax.Array:
    """(T, k, experts) bool: slot (t, s) chose expert e."""
    return ids[:, :, None] == jnp.arange(experts, dtype=jnp.int32)


def route_logits(logits: jax.Array, bias: jax.Array, config: MoeConfig) -> tuple[jax.Array, jax.Array]:
    """Each token's experts (T, top_k) int32 and their weights (T, top_k) f32, from f32 logits (T, experts)."""
    if config.scoring == "softmax":
        scores = jax.nn.softmax(logits, axis=-1)
        choice = scores
    else:
        scores = jax.nn.sigmoid(logits)
        choice = scores + bias  # the bias steers the choice only, not the weights
    if config.n_group > 1:
        t, e = choice.shape
        groups = choice.reshape(t, config.n_group, e // config.n_group)
        group_scores = jnp.sum(jax.lax.top_k(groups, 2)[0], axis=-1)  # each group: its two best experts
        _, best = jax.lax.top_k(group_scores, config.topk_group)
        kept = jnp.any(best[:, :, None] == jnp.arange(config.n_group)[None, None, :], axis=1)
        choice = jnp.where(jnp.repeat(kept, e // config.n_group, axis=1), choice, -jnp.inf)
    _, ids = jax.lax.top_k(choice, config.top_k)
    ids = ids.astype(jnp.int32)
    # Each slot's score: a masked sum over the experts rather than a gather of T * k scalars.
    weights = jnp.sum(jnp.where(one_hot(ids, logits.shape[1]), scores[:, None, :], 0.0), axis=2)
    if config.normalize:
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-20)
    return ids, weights * config.scale


def route(normed: jax.Array, w_router: jax.Array, bias: jax.Array, config: MoeConfig) -> tuple[jax.Array, jax.Array]:
    """Each token's experts (T, top_k) int32 and their weights (T, top_k) f32. The logits are f32."""
    logits = jnp.dot(normed.astype(jnp.float32), w_router, precision=jax.lax.Precision.HIGHEST)
    return route_logits(logits, bias, config)


def route_tokens(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    """rmsnorm(x), each token's experts and their weights: `norm_router`, then `route_logits`. (Its logits
    and XLA's f32 ones differ by rounding, so a token at a near tie between two experts may take the other.)"""
    split = params.get("router_split") or split_router(params["router"])
    normed, logits = norm_router(x, params["norm"], *split)
    return normed, *route_logits(logits, params["bias"], config)


# --------------------------------------------------------------------------- dispatch and combine


def capacity(slots: int, experts: int, group: int) -> int:
    """Rows of the padded layout: the most any routing of `slots` (token, expert) pairs can need when each
    expert's rows are padded to a multiple of `group`."""
    return -(-(slots + min(experts, slots) * (group - 1)) // group) * group


def ragged_rows(slots: int) -> int:
    """Rows of the unpadded layout: the slots, rounded up so that row tiles of up to 512 divide them."""
    return -(-slots // 512) * 512 if slots > 512 else -(-slots // SUBLANES) * SUBLANES


@dataclass(frozen=True)
class Dispatch:
    """Where each (token, slot) goes in the experts' rows."""

    src: jax.Array  # (rows,) the token each row reads (`tokens` for an unused row: a row of zeros)
    pos: jax.Array  # (T, top_k) the row of each (token, slot)
    counts: jax.Array  # (E,) rows (tokens) per expert
    starts: jax.Array  # (E,) each expert's first row
    ends: jax.Array  # (E,) the end of each expert's rows, padding included


def _ranks(ids: jax.Array, hot: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each slot's rank among its expert's slots in (token, slot) order (T, k), and the slots per expert (E,):
    a cumulative sum over tokens of the one-hot mask, no sort."""
    per_token = jnp.sum(hot, axis=1, dtype=jnp.int32)  # (T, E): 0 or 1 from a router
    before = jnp.cumsum(per_token, axis=0, dtype=jnp.int32) - per_token  # the expert's slots of earlier tokens
    k = ids.shape[1]
    earlier = jnp.tril(jnp.ones((k, k), bool), -1)  # [s, s']: s' < s (a token choosing an expert twice)
    repeats = jnp.sum((ids[:, :, None] == ids[:, None, :]) & earlier, axis=2, dtype=jnp.int32)
    rank = jnp.sum(jnp.where(hot, before[:, None, :], 0), axis=2) + repeats
    return rank, jnp.sum(per_token, axis=0)


def _sources(ids: jax.Array, counts: jax.Array, starts: jax.Array, ends: jax.Array, group: int, rows: int):
    """The token of each row (`tokens` past an expert's rows). The (expert, slot) keys sorted are every
    expert's rows back to back; each block of `group` rows (one expert's) is a window of them."""
    tokens, k = ids.shape
    slots = tokens * k
    flat = ids.reshape(-1)
    if counts.shape[0] * slots < 2**31:  # one sort of an int32 key (expert, slot): cheaper than an argsort
        order = jnp.sort(flat * slots + jnp.arange(slots, dtype=jnp.int32)) % slots
    else:
        order = jnp.argsort(flat, stable=True).astype(jnp.int32)
    sorted_tokens = order // k
    if group == 1:  # no padding: the rows are the sorted slots
        return jnp.concatenate([sorted_tokens, jnp.full(rows - slots, tokens, jnp.int32)])
    first = jnp.cumsum(counts, dtype=jnp.int32) - counts  # each expert's first sorted slot
    block = jnp.arange(0, rows, group, dtype=jnp.int32)
    mine = (starts[None, :] <= block[:, None]) & (block[:, None] < ends[None, :])  # (blocks, E): one expert or none
    shift = jnp.sum(jnp.where(mine, first - starts, 0), axis=1)
    limit = jnp.sum(jnp.where(mine, starts + counts, 0), axis=1)  # the end of the expert's real rows
    padded = jnp.concatenate([sorted_tokens, jnp.full(group, tokens, jnp.int32)])
    windows = jax.vmap(lambda at: jax.lax.dynamic_slice(padded, (at,), (group,)))(block + shift)
    row = block[:, None] + jnp.arange(group, dtype=jnp.int32)
    return jnp.where(row < limit[:, None], windows, tokens).reshape(-1)


def dispatch(ids: jax.Array, experts: int, group: int, rows: int) -> Dispatch:
    """Lay the (token, slot) pairs out by expert, in token order, each expert's rows starting at a multiple of
    `group` (1: no padding), in a layout of `rows` rows."""
    hot = one_hot(ids, experts)
    rank, counts = _ranks(ids, hot)
    padded = -(-counts // group) * group
    ends = jnp.cumsum(padded, dtype=jnp.int32)
    starts = ends - padded
    pos = rank + jnp.sum(jnp.where(hot, starts, 0), axis=2)
    return Dispatch(_sources(ids, counts, starts, ends, group, rows), pos, counts, starts, ends)


def tile_experts(d: Dispatch, group: int, rows: int) -> jax.Array:
    """The expert of each tile of `group` rows of a padded layout, -1 past the last expert's rows."""
    tile_start = jnp.arange(0, rows, group, dtype=jnp.int32)
    expert = jnp.searchsorted(d.ends, tile_start, side="right", method="compare_all").astype(jnp.int32)
    return jnp.where(tile_start < d.ends[-1], expert, -1)


@dataclass(frozen=True)
class Blocks:
    """`expert_ffn_blocks`' layout: rows gathered with each expert's padded to G/8, tiles of G output rows."""

    src: jax.Array  # (rows,) the token each gathered row reads (`tokens`: a row of zeros)
    pos: jax.Array  # (T, top_k) the output row of each (token, slot)
    tile_expert: jax.Array  # (tiles,) each tile's expert, -1 past the used tiles
    tile_blocks: jax.Array  # (tiles * 8,) each tile's 8 blocks of G/8 gathered rows


def block_dispatch(ids: jax.Array, experts: int, group: int) -> Blocks:
    """`expert_ffn_blocks`' layout for tiles of `group` rows (a multiple of 128)."""
    tokens, k = ids.shape
    r = group // BLOCKS
    gathered = dispatch(ids, experts, r, capacity(tokens * k, experts, r))
    tiles = -(-gathered.counts // group)
    tile_end = jnp.cumsum(tiles, dtype=jnp.int32)
    tile_start = tile_end - tiles
    tile = jnp.arange(capacity(tokens * k, experts, group) // group, dtype=jnp.int32)
    mine = (tile_start[None, :] <= tile[:, None]) & (tile[:, None] < tile_end[None, :])  # (tiles, E)
    expert = jnp.where(jnp.any(mine, axis=1), jnp.sum(jnp.where(mine, jnp.arange(experts), 0), axis=1), -1)
    # A tile's blocks: its expert's first block, plus 8 per tile of the expert before it.
    first = jnp.sum(jnp.where(mine, gathered.starts // r - BLOCKS * tile_start, 0), axis=1) + BLOCKS * tile
    last = gathered.src.shape[0] // r - 1
    blocks = jnp.minimum(first[:, None] + jnp.arange(BLOCKS, dtype=jnp.int32), last)
    blocks = jnp.where(expert[:, None] < 0, 0, blocks).reshape(-1)
    # A slot's output row: its expert's first tile's first row plus its rank (its gathered row's offset).
    shift = jnp.sum(jnp.where(one_hot(ids, experts), group * tile_start - gathered.starts, 0), axis=2)
    return Blocks(gathered.src, gathered.pos + shift, expert, blocks)


def gather(normed: jax.Array, src: jax.Array) -> jax.Array:
    """The rows of `normed` the dispatch says, zeros where `src` is past the tokens."""
    return jnp.take(normed, src, axis=0, mode="fill", fill_value=0)


def combine(y: jax.Array, pos: jax.Array, weights: jax.Array) -> jax.Array:
    """Each token's weighted sum of its experts' rows, f32 (T, D): a gather and a sum, fused by XLA."""
    return jnp.sum(y[pos].astype(jnp.float32) * weights[:, :, None], axis=1)


def finish(x: jax.Array, routed: jax.Array, params: dict[str, Any], config: MoeConfig) -> jax.Array:
    """x + routed + the shared experts (a kernel), or just the residual."""
    if not config.shared:
        return (x.astype(jnp.float32) + routed).astype(x.dtype)
    s = params["shared"]
    return shared_experts(x, params["norm"], s["w_gate"], s["w_up"], s["w_down"], routed.astype(x.dtype))


# --------------------------------------------------------------------------- the layer


def default_group(tokens: int) -> int:
    """A group size without measuring: about the tokens (decode), at most 256 (prefill)."""
    return min(256, -(-tokens // SUBLANES) * SUBLANES)


def group_candidates(tokens: int, config: MoeConfig) -> tuple[int, ...]:
    """The group sizes worth measuring. Decode (under 64 rows per expert on average): the three biggest of
    `GROUPS` up to about the tokens. Prefill: `expert_ffn_blocks`' sizes about the rows per expert, the
    three biggest under twice the average, and the next one up."""
    per_expert = tokens * config.top_k / config.experts
    if per_expert < 64:
        return tuple(g for g in GROUPS if g < 2 * max(tokens, SUBLANES))[-3:]
    under = [g for g in BLOCK_GROUPS if g < 2 * per_expert]
    over = [g for g in BLOCK_GROUPS if g >= 2 * per_expert][:1]
    return tuple(under + over)[-3:]


def experts_call(normed: jax.Array, ids: jax.Array, params: dict[str, Any], config: MoeConfig, group: int):
    """The routed experts' kernel for tiles of `group` rows, its inputs, and the row of each (token, slot) in
    its output: `expert_ffn` on rows padded to `group` (decode), `expert_ffn_blocks` from 128 rows."""
    weights = (params["w_gate"], params["w_up"], params["w_down"])
    if group < BLOCK_GROUPS[0]:
        rows = capacity(ids.size, config.experts, group)
        d = dispatch(ids, config.experts, group, rows)
        return expert_ffn, (gather(normed, d.src), *weights, tile_experts(d, group, rows)), d.pos
    b = block_dispatch(ids, config.experts, group)
    xs = gather(normed, b.src).reshape(-1, group // BLOCKS, normed.shape[1])
    return expert_ffn_blocks, (xs, *weights, b.tile_expert, b.tile_blocks), b.pos


def expert_inputs(x: jax.Array, params: dict[str, Any], config: MoeConfig, group: int) -> tuple[Any, ...]:
    """Everything up to `expert_ffn` on rows padded to `group` (any size): its inputs, and the dispatch and
    weights the combine needs."""
    tokens = x.shape[0]
    normed = rmsnorm(x, params["norm"])
    ids, weights = route(normed, params["router"], params["bias"], config)
    rows = capacity(tokens * config.top_k, config.experts, group)
    d = dispatch(ids, config.experts, group, rows)
    inputs = (gather(normed, d.src), params["w_gate"], params["w_up"], params["w_down"], tile_experts(d, group, rows))
    return inputs, d, weights


def moe_layer(x: jax.Array, params: dict[str, Any], config: MoeConfig, group: int | None = None) -> jax.Array:
    """x + MoE(rmsnorm(x)) for x (tokens, d_model): `norm_router`, the dispatch, the experts' kernel for tiles
    of `group` rows (`default_group` unless given; `choose_group` measures) and the combine."""
    group = group or default_group(x.shape[0])
    normed, ids, weights = route_tokens(x, params, config)
    kernel, inputs, pos = experts_call(normed, ids, params, config, group)
    return finish(x, combine(kernel(*inputs), pos, weights), params, config)


def moe_layer_ragged(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> jax.Array:
    """The same with autokernel's megablox-style `gmm_swiglu` and `gmm_down` on unpadded rows."""
    normed, ids, weights = route_tokens(x, params, config)
    d = dispatch(ids, config.experts, 1, ragged_rows(ids.size))
    h = gmm_swiglu(gather(normed, d.src), params["w_gate"], params["w_up"], d.starts, d.ends)
    y = gmm_down(h, params["w_down"], d.starts, d.ends)
    return finish(x, combine(y, d.pos, weights), params, config)


def choose_group(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> int:
    """The group size whose tuned experts' kernel is fastest on this routing (call eagerly, with
    representative tokens: the kernel is tuned on them, and `moe_layer` with this group then reuses the
    tuning, under `jax.jit` too). Off a TPU nothing is measured: `default_group`."""
    tokens = x.shape[0]
    if jax.default_backend() != "tpu":
        return default_group(tokens)
    normed, ids, _ = route_tokens(x, params, config)
    seconds = {}
    for group in group_candidates(tokens, config):
        kernel, inputs, _ = experts_call(normed, ids, params, config, group)
        seconds[group] = kernel.tuning(*inputs).best.seconds
    return min(seconds, key=seconds.get)


def experts_specs(config: MoeConfig, tokens: int, group: int) -> tuple[autokernel.Kernel, tuple[Any, ...]]:
    """The experts' kernel `moe_layer` runs for `tokens` tokens and tiles of `group` rows, and its input specs
    (for lowering it, or timing it from a profile)."""
    d, e, f = config.d_model, config.experts, config.ffn

    def spec(*shape: int, dtype: Any = jnp.bfloat16) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape, dtype)

    weights = (spec(e, d, f), spec(e, d, f), spec(e, f, d))
    slots = tokens * config.top_k
    tiles = capacity(slots, e, group) // group
    if group < BLOCK_GROUPS[0]:
        return expert_ffn, (spec(tiles * group, d), *weights, spec(tiles, dtype=jnp.int32))
    r = group // BLOCKS
    xs = spec(capacity(slots, e, r) // r, r, d)
    return expert_ffn_blocks, (xs, *weights, spec(tiles, dtype=jnp.int32), spec(tiles * BLOCKS, dtype=jnp.int32))


def init_params(config: MoeConfig, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights: experts stacked (E, d_model, ffn) and (E, ffn, d_model); an f32 router (and its
    `split_router` halves) and bias."""
    d, e, f = config.d_model, config.experts, config.ffn
    keys = iter(jax.random.split(jax.random.key(seed), 9))

    def normal(shape, scale, dt=dtype):
        return (jax.random.normal(next(keys), shape, jnp.float32) * scale).astype(dt)

    params = {
        "norm": (1 + 0.1 * jax.random.normal(next(keys), (1, d))).astype(dtype),
        "router": normal((d, e), d**-0.5, jnp.float32),
        "bias": normal((e,), 0.1 if config.scoring == "sigmoid" else 0.0, jnp.float32),
        "w_gate": normal((e, d, f), d**-0.5),
        "w_up": normal((e, d, f), d**-0.5),
        "w_down": normal((e, f, d), f**-0.5),
    }
    params["router_split"] = split_router(params["router"])  # `norm_router`'s weights, made once
    if config.shared:
        fs = config.shared_ffn
        params["shared"] = {
            "w_gate": normal((d, fs), d**-0.5),
            "w_up": normal((d, fs), d**-0.5),
            "w_down": normal((fs, d), fs**-0.5),
        }
    return params


# --------------------------------------------------------------------------- baselines: plain JAX


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _swiglu(gate_up: jax.Array, ffn: int) -> jax.Array:
    return jax.nn.silu(gate_up[..., :ffn]) * gate_up[..., ffn:]


def baseline_params(params: dict[str, Any]) -> dict[str, Any]:
    """The baselines' weights: gate and up side by side, (E, d_model, 2 ffn), so one grouped matmul does both."""
    return {**params, "w_gate_up": jnp.concatenate([params["w_gate"], params["w_up"]], axis=2)}


def megablox_tiling(tm: int, cap: int):
    """A megablox tiling function: tm rows, and the biggest multiples of 128 up to `cap` dividing k and n."""

    def block(n: int) -> int:
        return next(b for b in range(min(cap, n), 0, -128) if n % b == 0) if n % 128 == 0 else n

    return lambda m, k, n: (min(tm, m), block(k), block(n))


def baseline_experts(xs, params, counts, config: MoeConfig, method: str = "ragged_dot", tiling: Any = None):
    """The routed experts on dispatched (unpadded) rows, in plain JAX: XLA's `jax.lax.ragged_dot` or JAX's
    Pallas megablox `gmm` (with `tiling`, see `megablox_tiling`), gate and up in one grouped matmul. Each
    accumulates in f32 and rounds to bf16, as the MXU does."""
    from jax.experimental.pallas.ops.tpu.megablox import gmm

    def grouped(lhs, rhs):
        if method == "megablox":
            return gmm(lhs, rhs, counts, lhs.dtype, tiling=tiling, interpret=jax.default_backend() != "tpu")
        return jax.lax.ragged_dot(lhs, rhs, counts, preferred_element_type=lhs.dtype)

    hidden = _swiglu(grouped(xs, params["w_gate_up"]).astype(jnp.float32), config.ffn).astype(xs.dtype)
    return grouped(hidden, params["w_down"])


def baseline(config: MoeConfig, method: str = "ragged_dot", tiling: Any = None):
    """The layer in plain JAX, jitted, taking (x, baseline_params(params)): the routed experts as
    `baseline_experts` does them, or "dense" (every expert on every token, one-hot weighted: for a few
    tokens); the shared experts in XLA."""

    def layer(x, params):
        tokens = x.shape[0]
        normed = rmsnorm(x, params["norm"])
        ids, weights = route(normed, params["router"], params["bias"], config)
        if method == "dense":
            gate_up = jnp.einsum("td,edf->etf", normed, params["w_gate_up"], preferred_element_type=jnp.float32)
            hidden = _swiglu(gate_up, config.ffn).astype(x.dtype)
            y = jnp.einsum("etf,efd->etd", hidden, params["w_down"], preferred_element_type=jnp.float32)
            dense = jnp.sum(jnp.where(one_hot(ids, config.experts), weights[:, :, None], 0.0), axis=1)
            routed = jnp.einsum("te,etd->td", dense, y.astype(x.dtype).astype(jnp.float32))
        else:
            d = dispatch(ids, config.experts, 1, ragged_rows(tokens * config.top_k))
            y = baseline_experts(gather(normed, d.src), params, d.counts, config, method, tiling)
            routed = combine(y, d.pos, weights)
        out = x.astype(jnp.float32) + routed.astype(x.dtype).astype(jnp.float32)
        if config.shared:
            s = params["shared"]
            hidden = (jax.nn.silu(_dot(normed, s["w_gate"])) * _dot(normed, s["w_up"])).astype(x.dtype)
            out = out + _dot(hidden, s["w_down"])
        return out.astype(x.dtype)

    return jax.jit(layer)


if __name__ == "__main__":
    config = MoeConfig(256, 8, 2, 256, shared=1, scoring="sigmoid", n_group=4, topk_group=2, scale=2.5)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (512, config.d_model), jnp.bfloat16)
    out = jax.jit(moe_layer, static_argnums=2)(x, params, config)  # the default tiles off a TPU
    normed, ids, _ = route_tokens(x, params, config)
    kernel, inputs, _ = experts_call(normed, ids, params, config, default_group(x.shape[0]))
    print(kernel.source(*inputs))
    print("experts' rows:", dispatch(ids, config.experts, 1, ids.size).counts.tolist(), "-> tiles' experts:")
    print(inputs[-2].tolist())
    want = baseline(config)(x, baseline_params(params))
    print("max |autokernel - ragged_dot|:", float(jnp.max(jnp.abs(out.astype(float) - want.astype(float)))))
