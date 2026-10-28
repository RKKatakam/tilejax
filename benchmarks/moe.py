"""Mixture-of-experts layers (examples/moe.py) on a TPU v5e, bf16: autokernel's layer, tuned on the layer's own
routing, against the same layer in plain JAX with XLA's `jax.lax.ragged_dot`, with JAX's Pallas megablox
`gmm` (at its best of a grid of tilings), and, for decode, every expert on every token. The baselines route
and dispatch as moe.py does (no sort of the slots by expert, no scatters), so the layer comparison is of the
experts, the combine and the shared experts; the breakdown shows the dispatch against the sort-and-scatter
one moe.py used before.

For each layer and token count it prints:

- the group sizes `choose_group` measured (tuned device time of the experts' kernel for each);
- where autokernel's layer spends its time, each part jitted and timed on its own: `norm_router` + top-k
  ("router"; and XLA's RMSNorm + f32 router for reference), the dispatch's index arrays ("dispatch"; and
  the former sort + scatter dispatch for reference), the row gather into the experts' rows ("gather"),
  the experts' kernel, the combine (row gather + weighted sum, + the residual without shared experts), and
  the shared experts' kernel;
- the routed experts alone, on dispatched rows: autokernel's kernel at the chosen group (`expert_ffn` for
  decode, `expert_ffn_blocks` for prefill), autokernel's megablox-style `gmm_swiglu` + `gmm_down` on
  unpadded rows ("ragged"), ragged_dot and megablox (gate and up in one grouped matmul, SwiGLU in XLA, the
  down projection);
- the whole layer (RMSNorm, router, dispatch, experts, combine, shared experts, residual), each way.

Prefill reports TFLOP/s (the experts' useful flops, and the shared experts' in the layer); decode reports
GB/s of the weights a token batch has to read (the experts it chose, and the shared experts).

    uv run python benchmarks/moe.py
    uv run python benchmarks/moe.py --config qwen3-30b-a3b --tokens 4096 --tokens 8
    uv run python benchmarks/moe.py --small      # tiny shapes, anywhere (the Pallas interpreter off a TPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from moe import (
    CONFIGS,
    DECODE_TOKENS,
    MoeConfig,
    baseline,
    baseline_experts,
    baseline_params,
    combine,
    combine_rows,
    decode_call,
    dispatch,
    experts_call,
    gather,
    gmm_down,
    gmm_swiglu,
    group_candidates,
    init_params,
    megablox_tiling,
    moe_layer,
    moe_layer_ragged,
    one_hot,
    ragged_rows,
    rmsnorm,
    route,
    route_tokens,
    shared_combine,
    slot_ranks,
    slot_rows,
)
from timing import require_tpu, time_ms

import autokernel

RUNS = [  # config, token counts (decode: 8-64 tokens; prefill: 4096-16384)
    ("qwen3-30b-a3b", (8, 64, 4096, 16384)),
    ("deepseek-v2-lite", (8, 64, 4096, 16384)),
    ("deepseek-v3-small", (8, 64, 4096)),
    ("kimi-k2-small", (8, 64, 4096)),
]
LARGE = [  # one layer of real models at full width (`--large`)
    ("qwen3-235b-a22b", (8, 64, 4096, 16384)),
    ("qwen3-next-80b-a3b", (8, 64, 4096, 16384)),
    ("mixtral-8x7b", (8, 64, 4096, 16384)),
    ("deepseek-v3-64e", (8, 64, 4096, 16384)),
]
SMALL = MoeConfig(256, 8, 2, 256, shared=1, scoring="sigmoid", n_group=4, topk_group=2, scale=2.5)
DENSE_TOKENS = 64  # every expert on every token only for decode batches
MEGABLOX_TILINGS = [(tm, cap) for tm in (128, 256, 512, 1024) for cap in (512, 2048)]  # rows, k and n blocks up to cap
# XLA's scoped VMEM limit, which the baselines' kernels (megablox's `gmm` takes no limit of its own) and
# fusions compile under; autokernel's kernels set theirs (0.9 x VMEM). `--scoped-vmem-mib` raises it so
# megablox's bigger tilings fit: the fair comparison is each baseline at its best of both runs.
SCOPED_VMEM_FLAG = "--xla_tpu_scoped_vmem_limit_kib"
METHODS = ("autokernel", "autokernel_ragged", "ragged_dot", "megablox")
PARTS = ("router", "dispatch", "gather", "experts", "regather", "finish")


def _time_or_inf(fn: Any, *args: Any, runs: int) -> float:
    """`time_ms`, or infinity for a baseline configuration that doesn't compile (megablox tilings can run
    Mosaic out of VMEM)."""
    try:
        return time_ms(fn, *args, runs=runs)
    except (jax.errors.JaxRuntimeError, ValueError) as e:  # out of VMEM, or a tiling the shapes reject
        print(f"    (skipped a baseline configuration: {str(e).splitlines()[0][:100]})")
        return float("inf")


def megablox_tilings(config: MoeConfig, rows: int) -> list[tuple[str, Any]]:
    """Megablox tilings to try, each named by its (tm, tk, tn) for the gate-up and down matmuls, without
    repeats (small arrays clamp tilings to the same blocks)."""
    d, f = config.d_model, config.ffn
    seen, tilings = set(), []
    for tm, cap in MEGABLOX_TILINGS:
        tiling = megablox_tiling(tm, cap)
        blocks = (tiling(rows, d, 2 * f), tiling(rows, f, d))
        if blocks not in seen:
            seen.add(blocks)
            tilings.append((" ".join("x".join(map(str, b)) for b in blocks), tiling))
    return tilings


def max_diff(a: jax.Array, b: jax.Array, rows: np.ndarray) -> float:
    """The largest difference over these rows."""
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))[rows]))


def same_routing(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> np.ndarray:
    """The tokens `norm_router` and XLA's f32 router send to the same experts (they round the logits
    differently, so a token at a near tie may take another expert in one than in the other)."""
    ids = np.sort(np.asarray(route_tokens(x, params, config)[1]), axis=1)
    xla = np.sort(np.asarray(route(rmsnorm(x, params["norm"]), params["router"], params["bias"], config)[0]), axis=1)
    return (ids == xla).all(axis=1)


def sorted_dispatch(ids: jax.Array, experts: int, group: int, rows: int) -> tuple[jax.Array, ...]:
    """moe.py's former dispatch, for reference: one sort of the slots by expert, a scatter-add for the rows per
    expert, two scatters (each slot's row, each row's token) and gathers from per-expert tables."""
    tokens, k = ids.shape
    slots = tokens * k
    flat = ids.reshape(-1)
    key = jnp.sort(flat * slots + jnp.arange(slots, dtype=jnp.int32))
    order, expert_of = key % slots, key // slots
    counts = jnp.zeros(experts, jnp.int32).at[flat].add(1)
    padded = -(-counts // group) * group
    ends = jnp.cumsum(padded, dtype=jnp.int32)
    starts = ends - padded
    first = jnp.cumsum(counts, dtype=jnp.int32) - counts
    row = starts[expert_of] + jnp.arange(slots, dtype=jnp.int32) - first[expert_of]
    pos = jnp.zeros(slots, jnp.int32).at[order].set(row).reshape(tokens, k)
    src = jnp.full(rows, tokens, jnp.int32).at[row].set(order // k)
    return src, pos, starts, ends


def measure_groups(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> int:
    """`choose_group`, printing the tuned device time of the experts' kernel for each candidate group."""
    tokens = x.shape[0]
    normed, ids, _ = route_tokens(x, params, config)
    seconds = {}
    for group in group_candidates(tokens, config):
        kernel, inputs, _ = experts_call(normed, ids, params, config, group)
        tuning = kernel.tuning(*inputs)
        seconds[group] = tuning.best.seconds
        print(f"  group {group:5}: {kernel.nest.name:<18} {seconds[group] * 1e3:8.3f} ms  ({tuning.best.config})")
    return min(seconds, key=seconds.get)


def lax_router(x: jax.Array, params: dict[str, Any], config: MoeConfig) -> tuple[jax.Array, jax.Array]:
    """moe.py's former router, for reference: XLA's RMSNorm and f32 router, `jax.lax.top_k` (twice more for
    group-limited routing)."""
    logits = jnp.dot(rmsnorm(x, params["norm"]).astype(jnp.float32), params["router"], precision="highest")
    scores = jax.nn.softmax(logits, axis=-1) if config.scoring == "softmax" else jax.nn.sigmoid(logits)
    choice = scores if config.scoring == "softmax" else scores + params["bias"]
    if config.n_group > 1:
        t, e = choice.shape
        groups = choice.reshape(t, config.n_group, e // config.n_group)
        _, best = jax.lax.top_k(jnp.sum(jax.lax.top_k(groups, 2)[0], axis=-1), config.topk_group)
        kept = jnp.any(best[:, :, None] == jnp.arange(config.n_group), axis=1)
        choice = jnp.where(jnp.repeat(kept, e // config.n_group, axis=1), choice, -jnp.inf)
    _, ids = jax.lax.top_k(choice, config.top_k)
    return ids, jnp.take_along_axis(scores, ids, axis=1)


def breakdown(x: jax.Array, params: dict[str, Any], config: MoeConfig, group: int) -> dict[str, float]:
    """autokernel's layer part by part, each jitted and timed on its own inputs, ms: for decode, the routing
    (XLA), the per-expert counts and weights ("dispatch") and the one kernel ("experts"); for prefill, also
    the row gathers into the experts' rows ("gather") and back into slot-major rows ("regather"), and
    the end of the layer ("finish": `combine_rows` or `shared_combine`)."""
    e = config.experts
    parts = dict.fromkeys(PARTS, 0.0)
    xla_router = time_ms(jax.jit(lambda x: lax_router(x, params, config)), x, runs=10)
    if x.shape[0] <= DECODE_TOKENS:
        kernel, args = decode_call(x, params, config)
        bias_args = (params["router"], params["bias"])
        parts["router"] = time_ms(jax.jit(lambda x: route(rmsnorm(x, params["norm"]), *bias_args, config)), x, runs=10)
        parts["dispatch"] = time_ms(jax.jit(lambda x: decode_call(x, params, config)[1][5:]), x, runs=10)
        parts["dispatch"] = max(parts["dispatch"] - parts["router"], 0.0)  # (it routes too)
        parts["experts"] = time_ms(jax.jit(kernel), *args, runs=10)
        print("  autokernel's layer, part by part (ms): " + ", ".join(f"{p} {parts[p]:.3f}" for p in PARTS))
        print(f"    for reference: the former router (lax.top_k) {xla_router:.3f}")
        return parts
    normed, ids, weights = route_tokens(x, params, config)
    kernel, inputs, pos = experts_call(normed, ids, params, config, group)
    y = kernel(*inputs)

    def indices(ids):  # the dispatch's index arrays (XLA drops the row gather of nothing)
        _, inputs, pos = experts_call(jnp.zeros((ids.shape[0], 128), x.dtype), ids, params, config, group)
        return inputs[4:], pos

    def rows(normed, ids):
        return experts_call(normed, ids, params, config, group)[1][0]

    rows_back = slot_rows(y, pos)
    parts |= {
        "router": time_ms(jax.jit(lambda x: route_tokens(x, params, config)), x, runs=10),
        "dispatch": time_ms(jax.jit(indices), ids, runs=10),
        "gather": time_ms(jax.jit(rows), normed, ids, runs=10),
        "experts": time_ms(jax.jit(kernel), *inputs, runs=10),
        "regather": time_ms(jax.jit(slot_rows), y, pos, runs=10),
    }
    parts["gather"] = max(parts["gather"] - parts["dispatch"], 0.0)  # (`rows` computes the indices too)
    w = weights.T[:, :, None]
    if config.shared:
        s = params["shared"]
        args = (x, params["norm"], s["w_gate"], s["w_up"], s["w_down"], rows_back, w)
        parts["finish"] = time_ms(jax.jit(shared_combine), *args, runs=10)
    else:
        parts["finish"] = time_ms(jax.jit(combine_rows), x, rows_back, w, runs=10)

    def xla_combine(y, pos, weights, x):  # the former combine: one XLA fusion of the gather and the sum
        return (x.astype(jnp.float32) + combine(y, pos, weights)).astype(x.dtype)

    old_combine = time_ms(jax.jit(xla_combine), y, pos, weights, x, runs=10)
    old = time_ms(jax.jit(sorted_dispatch, static_argnums=(1, 2, 3)), ids, e, 1, ragged_rows(ids.size), runs=10)
    ranks = time_ms(jax.jit(lambda ids: slot_ranks(ids, one_hot(ids, e))), ids, runs=10)
    unpadded = jax.jit(lambda ids: vars(dispatch(ids, e, 1, ragged_rows(ids.size))))
    new = time_ms(unpadded, ids, runs=10)
    print("  autokernel's layer, part by part (ms): " + ", ".join(f"{p} {parts[p]:.3f}" for p in PARTS))
    print(
        f"    for reference: the former router (XLA, lax.top_k) {xla_router:.3f}; unpadded dispatch: sort + "
        f"scatters {old:.3f}, moe.py's {new:.3f} (of which ranks {ranks:.3f}); the former combine (XLA gather "
        f"+ sum, + residual) {old_combine:.3f}"
    )
    return parts


def run(config: MoeConfig, name: str, tokens: int, profile: bool = False) -> dict[str, Any]:
    """Time one layer at one token count; print it, and return the summary row. autokernel's ways run first,
    then the baselines', whose weights (gate and up side by side) replace the separate ones: a full-width
    layer's weights twice over and its rows at 16384 tokens don't fit a v5e's 16 GB."""
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (tokens, config.d_model), jnp.bfloat16)
    d, e, f, k = config.d_model, config.experts, config.ffn, config.top_k
    prefill = tokens > DENSE_TOKENS

    # The routing everything below shares (for the experts alone).
    normed, ids, _ = route_tokens(x, params, config)
    chosen = len(np.unique(np.asarray(ids)))
    rows = ragged_rows(tokens * k)
    disp = dispatch(ids, e, 1, rows)

    expert_flops = 6 * tokens * k * d * f
    layer_flops = expert_flops + 6 * tokens * d * config.shared_ffn + 2 * tokens * d * e
    # The fewest bytes from and to HBM: each chosen expert's weights once, the rows in and out (and for the
    # layer the shared experts' weights and the tokens in and out).
    expert_bytes = 3 * chosen * d * f * 2 + 2 * tokens * k * d * 2
    layer_bytes = expert_bytes + 3 * d * config.shared_ffn * 2 + 2 * tokens * d * 2

    def rate(ms: float, layer: bool) -> str:
        flops, moved = (layer_flops, layer_bytes) if layer else (expert_flops, expert_bytes)
        return f"{flops / ms / 1e9:6.1f} TFLOP/s, {moved / ms / 1e6:5.0f} GB/s"

    def show(label: str, ms: float, layer: bool) -> None:
        print(f"  {'layer:' if layer else 'experts alone:':<15}{label:<34} {ms:8.3f}  ({rate(ms, layer)})")

    print(f"{name}: {config}, {tokens} tokens ({chosen} of {e} experts chosen, {tokens * k / e:.0f} rows each): ms")

    # Tune autokernel's kernels on this routing, eagerly (under jit they would tune on random index arrays).
    decode = tokens <= DECODE_TOKENS  # (moe_layer: one kernel, no groups)
    group = 0 if decode else measure_groups(x, params, config)
    moe_layer(x, params, config, group)
    moe_layer_ragged(x, params, config)
    parts = breakdown(x, params, config, group)

    # autokernel: the routed experts alone (for decode, the whole layer's kernel, shared experts included),
    # then the layer.
    if decode:
        kernel, inputs = decode_call(x, params, config)
    else:
        kernel, inputs, _ = experts_call(normed, ids, params, config, group)
    ours_experts = time_ms(jax.jit(kernel), *inputs, runs=10)
    if profile:
        print(autokernel.profile(kernel, *inputs))
    print(f"  {kernel.nest.name}:", kernel.lower(*inputs).config)
    del inputs
    xs = gather(normed, disp.src)
    ragged = jax.jit(lambda xs, wg, wu, wd, s, n: gmm_down(gmm_swiglu(xs, wg, wu, s, n), wd, s, n))
    ragged_args = (xs, params["w_gate"], params["w_up"], params["w_down"], disp.starts, disp.ends)
    ours_ragged_experts = time_ms(ragged, *ragged_args, runs=10)
    if profile:
        print(autokernel.profile(gmm_swiglu, *ragged_args[:3], disp.starts, disp.ends))
    print("  gmm_swiglu:", gmm_swiglu.lower(*ragged_args[:3], disp.starts, disp.ends).config)
    h = jax.ShapeDtypeStruct((rows, f), jnp.bfloat16)
    print("  gmm_down:  ", gmm_down.lower(h, params["w_down"], disp.starts, disp.ends).config)
    del ragged_args
    ours_fn = jax.jit(moe_layer, static_argnums=(2, 3))
    ours = time_ms(ours_fn, x, params, config, group, runs=10)
    ours_ragged_fn = jax.jit(moe_layer_ragged, static_argnums=2)
    ours_ragged = time_ms(ours_ragged_fn, x, params, config, runs=10)
    same = np.ones(tokens, bool) if decode else same_routing(x, params, config)
    ours_out, ragged_out = ours_fn(x, params, config, group), ours_ragged_fn(x, params, config)
    if not decode:
        rows_back, w = jax.ShapeDtypeStruct((k, tokens, d), x.dtype), jax.ShapeDtypeStruct((k, tokens, 1), jnp.float32)
        if config.shared:
            s = params["shared"]
            end = shared_combine.lower(x, params["norm"], s["w_gate"], s["w_up"], s["w_down"], rows_back, w)
        else:
            end = combine_rows.lower(x, rows_back, w)
        print(f"  {end.plan.nest.name}:", end.config)

    # The baselines, on their weights (the separate gate and up weights freed).
    bparams = {name: value for name, value in baseline_params(params).items() if name not in ("w_gate", "w_up")}
    del params
    experts = jax.jit(baseline_experts, static_argnums=(3, 4, 5))
    ragged_dot_experts = time_ms(experts, xs, bparams, disp.counts, config, "ragged_dot", None, runs=10)
    tilings = megablox_tilings(config, rows)
    megablox_experts, megablox_name = min(
        (_time_or_inf(experts, xs, bparams, disp.counts, config, "megablox", tiling, runs=10), label)
        for label, tiling in tilings
    )
    del xs
    ragged_dot_fn = baseline(config)
    ragged_dot = time_ms(ragged_dot_fn, x, bparams, runs=10)
    megablox = min(_time_or_inf(baseline(config, "megablox", t), x, bparams, runs=10) for _, t in tilings)
    dense = time_ms(baseline(config, "dense"), x, bparams, runs=10) if not prefill else float("nan")
    want = ragged_dot_fn(x, bparams)
    diffs = (max_diff(ours_out, want, same), max_diff(ragged_out, want, same))

    show(f"autokernel {kernel.nest.name}" + ("" if decode else f" ({group})"), ours_experts, False)
    show("autokernel ragged", ours_ragged_experts, False)
    show("ragged_dot", ragged_dot_experts, False)
    show(f"megablox ({megablox_name})", megablox_experts, False)
    show(f"autokernel ({kernel.nest.name})", ours, True)
    show("autokernel ragged", ours_ragged, True)
    show("ragged_dot", ragged_dot, True)
    show("megablox (best tiling)", megablox, True)
    if not prefill:
        show("dense one-hot", dense, True)
    best_baseline = min(ragged_dot, megablox, dense if not prefill else float("inf"))
    best_ours = min(ours, ours_ragged)
    print(
        f"  speedup {best_baseline / best_ours:.2f}x over the best baseline; max |layer - ragged_dot layer|: "
        f"autokernel {diffs[0]:.4f}, ragged {diffs[1]:.4f} (outputs up to {float(jnp.max(jnp.abs(want))):.2f}; "
        f"over the {same.sum()} of {tokens} tokens routed as XLA's router does)"
    )
    row = {
        "name": name,
        "config": str(config),
        "tokens": tokens,
        "group": group,
        "experts_ms": dict(
            zip(METHODS, (ours_experts, ours_ragged_experts, ragged_dot_experts, megablox_experts), strict=True)
        ),
        "layer_ms": dict(zip((*METHODS, "dense"), (ours, ours_ragged, ragged_dot, megablox, dense), strict=True)),
        "expert_bytes": expert_bytes,
        "expert_flops": expert_flops,
        "parts_ms": parts,
        "scoped_vmem": SCOPED_VMEM_FLAG in os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    print(json.dumps({"moe": row}), flush=True)
    return {
        "name": name,
        "tokens": tokens,
        "group": group,
        "experts": (ours_experts, ours_ragged_experts, ragged_dot_experts, megablox_experts),
        "layer": (ours, ours_ragged, ragged_dot, megablox, dense),
        "parts": parts,
        "rate": rate(best_ours, True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS), help="a layer (repeatable)")
    parser.add_argument("--tokens", action="append", type=int, help="a token count (repeatable)")
    parser.add_argument("--small", action="store_true", help="a tiny layer, runs anywhere")
    parser.add_argument("--large", action="store_true", help="layers of real models at full width")
    parser.add_argument("--profile", action="store_true", help="print the experts' kernels' profiles")
    parser.add_argument("--scoped-vmem-mib", type=int, help="XLA's scoped VMEM limit for the baselines (MiB)")
    args = parser.parse_args()
    if args.small:
        runs = [("small", SMALL, tuple(args.tokens or (8, 512)))]
    else:
        if args.scoped_vmem_mib:  # (before anything starts JAX's TPU backend: libtpu reads its flags once)
            flag = f"{SCOPED_VMEM_FLAG}={args.scoped_vmem_mib * 1024}"
            os.environ["LIBTPU_INIT_ARGS"] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} {flag}".strip()
        require_tpu()
        table = LARGE if args.large else RUNS
        names = args.config or [name for name, _ in table]
        default = dict(RUNS + LARGE)
        runs = [(name, CONFIGS[name], tuple(args.tokens or default[name])) for name in names]
    summary = []
    for name, config, token_counts in runs:
        for tokens in token_counts:
            summary.append(run(config, name, tokens, args.profile))
            print()
    print(
        "layer              tokens group | experts: ours    ragged  ragged_dot megablox | "
        "layer: ours    ragged  ragged_dot megablox dense   | speedup  best rate"
    )
    for row in summary:
        ex, ly = row["experts"], row["layer"]
        best_base = min(ly[2], ly[3], ly[4] if not np.isnan(ly[4]) else float("inf"))
        print(
            f"{row['name']:<18} {row['tokens']:<6} {row['group'] or '-':<5} | "
            f"{ex[0]:13.3f} {ex[1]:7.3f} {ex[2]:10.3f} {ex[3]:8.3f} | "
            f"{ly[0]:11.3f} {ly[1]:7.3f} {ly[2]:10.3f} {ly[3]:8.3f} {ly[4]:7.3f} | "
            f"{best_base / min(ly[0], ly[1]):6.2f}x  {row['rate']}"
        )
    print()
    print("autokernel's layer, part by part (ms, each timed on its own):")
    print("layer              tokens group | " + " ".join(f"{p:>8}" for p in PARTS) + " |      sum    layer")
    for row in summary:
        parts = row["parts"]
        print(
            f"{row['name']:<18} {row['tokens']:<6} {row['group'] or '-':<5} | "
            + " ".join(f"{parts[p]:8.3f}" for p in PARTS)
            + f" | {sum(parts.values()):8.3f} {row['layer'][0]:8.3f}"
        )


if __name__ == "__main__":
    main()
