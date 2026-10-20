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

Dispatch is plain JAX: the (token, slot) pairs are sorted by expert (one sort of an int32 key), and each
token's normalized row is gathered into its expert's rows. The experts then run as one grouped feed-forward
kernel over those rows, and the combine (the weighted sum of each token's expert rows, plus the residual)
is a gather and a sum that XLA fuses.

The grouped kernel, `expert_ffn`: each expert's rows are padded to a multiple of `G` (the layout's group
size), so every tile of G rows belongs to one expert; `tile_expert` (an index array, in SMEM) says which, or
-1 for the tiles past the last expert's rows (the layout has room for the most rows any routing can need).
A tile is a feed-forward layer of its own, as `transformer.ffn_layer`: its reduction loop walks blocks of
the expert's hidden units, streaming `W_gate[e][:, f]`, `W_up[e][:, f]` and `W_down[e][f, :]` (runtime
windows: the expert is read from SMEM when the kernel runs), with SwiGLU between the matmuls, so the hidden
activations never go to HBM. Its mask (`tile_expert[i // G] < 0`) is a runtime mask: the tiles past the
used rows take one empty step, and an expert without tokens has no tile, so its weights are never read.
That is what makes decoding cheap: a step reads only the experts some token chose.

Every step reads a whole block of an expert's weights for G rows of work: G flops per weight byte, so
prefill wants G >= ~256 on a v5e (197 TFLOP/s, 819 GB/s), while padding wastes about G/2 rows per expert,
and decode wants G about the tokens (an expert gets at most one row per token: one tile, one read of its
weights). `choose_group` lets the tuner's measurements decide.

`gmm_swiglu` and `gmm_down` are the megablox-style alternative, on unpadded rows: row tiles of any size
(the tunable's), each a reduction over the experts whose rows meet the tile (a runtime mask on the group
offsets), rows of other experts masked out of the result. A tile meets 1 + (experts starting inside it)
experts, and each such step does a whole tile's matmul.

`baseline` is the same layer in plain JAX with XLA's `jax.lax.ragged_dot`, JAX's Pallas megablox `gmm`, or
(for decode) every expert on every token, one-hot weighted.

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
GROUPS = (16, 32, 64, 128, 256, 512)  # group sizes `choose_group` may pick


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


# --------------------------------------------------------------------------- routing, dispatch, combine


def rmsnorm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)
    return normed.astype(x.dtype)


def route(normed: jax.Array, w_router: jax.Array, bias: jax.Array, config: MoeConfig) -> tuple[jax.Array, jax.Array]:
    """Each token's experts (T, top_k) int32 and their weights (T, top_k) f32. The logits are f32."""
    logits = jnp.dot(normed.astype(jnp.float32), w_router, precision=jax.lax.Precision.HIGHEST)
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
    weights = jnp.take_along_axis(scores, ids, axis=1)
    if config.normalize:
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-20)
    return ids.astype(jnp.int32), weights * config.scale


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


def dispatch(ids: jax.Array, experts: int, group: int, rows: int) -> Dispatch:
    """Sort the (token, slot) pairs by expert, each expert's rows starting at a multiple of `group` (1: no
    padding) in a layout of `rows` rows."""
    tokens, k = ids.shape
    slots = tokens * k
    flat = ids.reshape(-1)
    if experts * slots < 2**31:  # one sort of an int32 key (expert, slot): cheaper than an argsort
        key = jnp.sort(flat * slots + jnp.arange(slots, dtype=jnp.int32))
        order, expert_of = key % slots, key // slots
    else:
        order = jnp.argsort(flat, stable=True).astype(jnp.int32)
        expert_of = flat[order]
    counts = jnp.zeros(experts, jnp.int32).at[flat].add(1)
    padded = -(-counts // group) * group
    ends = jnp.cumsum(padded, dtype=jnp.int32)
    starts = ends - padded
    first = jnp.cumsum(counts, dtype=jnp.int32) - counts  # each expert's first slot in sorted order
    row = starts[expert_of] + jnp.arange(slots, dtype=jnp.int32) - first[expert_of]
    pos = jnp.zeros(slots, jnp.int32).at[order].set(row).reshape(tokens, k)
    src = jnp.full(rows, tokens, jnp.int32).at[row].set(order // k)
    return Dispatch(src, pos, counts, starts, ends)


def tile_experts(d: Dispatch, group: int, rows: int) -> jax.Array:
    """The expert of each tile of `group` rows of a padded layout, -1 past the last expert's rows."""
    tile_start = jnp.arange(0, rows, group, dtype=jnp.int32)
    expert = jnp.searchsorted(d.ends, tile_start, side="right", method="compare_all").astype(jnp.int32)
    return jnp.where(tile_start < d.ends[-1], expert, -1)


def gather(normed: jax.Array, src: jax.Array) -> jax.Array:
    """The rows of `normed` the dispatch says, zeros where `src` is past the tokens."""
    return jnp.concatenate([normed, jnp.zeros((1, normed.shape[1]), normed.dtype)])[src]


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


def group_candidates(tokens: int) -> tuple[int, ...]:
    """The group sizes worth measuring: the three biggest up to the tokens (rounded up to 16 rows)."""
    return tuple(g for g in GROUPS if g < 2 * max(tokens, SUBLANES))[-3:]


def expert_inputs(x: jax.Array, params: dict[str, Any], config: MoeConfig, group: int) -> tuple[Any, ...]:
    """Everything up to the expert kernel: its inputs, and the dispatch and weights the combine needs."""
    tokens = x.shape[0]
    normed = rmsnorm(x, params["norm"])
    ids, weights = route(normed, params["router"], params["bias"], config)
    rows = capacity(tokens * config.top_k, config.experts, group)
    d = dispatch(ids, config.experts, group, rows)
    inputs = (gather(normed, d.src), params["w_gate"], params["w_up"], params["w_down"], tile_experts(d, group, rows))
    return inputs, d, weights


def moe_layer(x: jax.Array, params: dict[str, Any], config: MoeConfig, group: int | None = None) -> jax.Array:
    """x + MoE(rmsnorm(x)) for x (tokens, d_model), with autokernel's `expert_ffn` on rows padded to `group`
    per expert (`default_group` unless given; `choose_group` measures)."""
    group = group or default_group(x.shape[0])
    inputs, d, weights = expert_inputs(x, params, config, group)
    return finish(x, combine(expert_ffn(*inputs), d.pos, weights), params, config)


def moe_layer_ragged(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> jax.Array:
    """The same with autokernel's megablox-style `gmm_swiglu` and `gmm_down` on unpadded rows."""
    tokens = x.shape[0]
    normed = rmsnorm(x, params["norm"])
    ids, weights = route(normed, params["router"], params["bias"], config)
    rows = ragged_rows(tokens * config.top_k)
    d = dispatch(ids, config.experts, 1, rows)
    h = gmm_swiglu(gather(normed, d.src), params["w_gate"], params["w_up"], d.starts, d.ends)
    y = gmm_down(h, params["w_down"], d.starts, d.ends)
    return finish(x, combine(y, d.pos, weights), params, config)


def choose_group(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> int:
    """The group size whose tuned `expert_ffn` is fastest on this routing (call eagerly, with representative
    tokens: the kernel is tuned on them, and `moe_layer` with this group then reuses the tuning, under
    `jax.jit` too). Off a TPU nothing is measured: `default_group`."""
    tokens = x.shape[0]
    if jax.default_backend() != "tpu":
        return default_group(tokens)
    seconds = {}
    for group in group_candidates(tokens):
        seconds[group] = expert_ffn.tuning(*expert_inputs(x, params, config, group)[0]).best.seconds
    return min(seconds, key=seconds.get)


def init_params(config: MoeConfig, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights: experts stacked (E, d_model, ffn) and (E, ffn, d_model); an f32 router and bias."""
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
            dense = jnp.zeros((tokens, config.experts), jnp.float32)
            dense = dense.at[jnp.arange(tokens)[:, None], ids].add(weights)
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
    x = jax.random.normal(jax.random.key(1), (128, config.d_model), jnp.bfloat16)
    out = jax.jit(moe_layer, static_argnums=2)(x, params, config)  # the default tiles off a TPU
    inputs, d, _ = expert_inputs(x, params, config, default_group(128))
    print(expert_ffn.source(*inputs))
    print("experts' rows:", d.counts.tolist(), "-> tiles' experts:", inputs[-1].tolist())
    want = baseline(config)(x, baseline_params(params))
    print("max |autokernel - ragged_dot|:", float(jnp.max(jnp.abs(out.astype(float) - want.astype(float)))))
