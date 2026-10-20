"""Mixture-of-experts layers (examples/moe.py) on a TPU v5e, bf16: autokernel's grouped expert kernels, tuned
on the layer's own routing, against the same layer in plain JAX with XLA's `jax.lax.ragged_dot`, with JAX's
Pallas megablox `gmm` (at its best of a grid of tilings), and, for decode, every expert on every token.

For each layer and token count it times the whole layer (RMSNorm, router, dispatch, experts, combine, shared
experts, residual) and the routed experts alone on the same dispatched rows:

- autokernel padded: `expert_ffn`, one kernel, each expert's rows padded to the group size `choose_group`
  measured best;
- autokernel ragged: `gmm_swiglu` then `gmm_down`, megablox-style, on unpadded rows;
- ragged_dot, megablox: gate and up in one grouped matmul, SwiGLU in XLA, the down projection.

Prefill reports TFLOP/s (the experts' useful flops, and the shared experts' in the layer); decode reports
GB/s of the weights a token batch has to read (the experts it chose, and the shared experts).

    uv run python benchmarks/moe.py
    uv run python benchmarks/moe.py --config qwen3-30b-a3b --tokens 4096 --tokens 8
    uv run python benchmarks/moe.py --small      # tiny shapes, anywhere (the Pallas interpreter off a TPU)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from moe import (
    CONFIGS,
    MoeConfig,
    baseline,
    baseline_experts,
    baseline_params,
    choose_group,
    dispatch,
    expert_ffn,
    expert_inputs,
    gather,
    gmm_down,
    gmm_swiglu,
    init_params,
    megablox_tiling,
    moe_layer,
    moe_layer_ragged,
    ragged_rows,
    rmsnorm,
    route,
    shared_experts,
)
from timing import require_tpu, time_ms

RUNS = [  # config, token counts (decode: 8-64 tokens; prefill: 4096-16384)
    ("qwen3-30b-a3b", (8, 64, 4096, 16384)),
    ("deepseek-v2-lite", (8, 64, 4096, 16384)),
    ("deepseek-v3-small", (8, 64, 4096)),
    ("kimi-k2-small", (8, 64, 4096)),
]
SMALL = MoeConfig(256, 8, 2, 256, shared=1, scoring="sigmoid", n_group=4, topk_group=2, scale=2.5)
DENSE_TOKENS = 64  # every expert on every token only for decode batches
MEGABLOX_TILINGS = [(tm, cap) for tm in (128, 256, 512) for cap in (512, 2048)]  # rows, k and n blocks up to cap


def _time_or_inf(fn: Any, *args: Any, runs: int) -> float:
    """`time_ms`, or infinity for a baseline configuration that doesn't compile (megablox tilings can run
    Mosaic out of VMEM)."""
    try:
        return time_ms(fn, *args, runs=runs)
    except jax.errors.JaxRuntimeError as e:
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


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def run(config: MoeConfig, name: str, tokens: int) -> dict[str, Any]:
    """Time one layer at one token count; print it, and return the summary row."""
    params = init_params(config)
    bparams = baseline_params(params)
    x = jax.random.normal(jax.random.key(1), (tokens, config.d_model), jnp.bfloat16)
    d, e, f, k = config.d_model, config.experts, config.ffn, config.top_k
    prefill = tokens > DENSE_TOKENS

    # The routing everything below shares (the layers route the same way; this is for the experts alone).
    normed = rmsnorm(x, params["norm"])
    ids, _ = route(normed, params["router"], params["bias"], config)
    chosen = len(np.unique(np.asarray(ids)))
    rows = ragged_rows(tokens * k)
    disp = dispatch(ids, e, 1, rows)
    xs = gather(normed, disp.src)

    expert_flops = 6 * tokens * k * d * f
    layer_flops = expert_flops + 6 * tokens * d * config.shared_ffn + 2 * tokens * d * e
    expert_bytes = 3 * chosen * d * f * 2
    layer_bytes = expert_bytes + 3 * d * config.shared_ffn * 2

    def rate(ms: float, layer: bool) -> str:
        if prefill:
            return f"{(layer_flops if layer else expert_flops) / ms / 1e9:6.1f} TFLOP/s"
        return f"{(layer_bytes if layer else expert_bytes) / ms / 1e6:6.0f} GB/s"

    print(f"{name}: {config}, {tokens} tokens ({chosen} of {e} experts chosen): ms")

    # Tune autokernel's kernels on this routing, eagerly (under jit they would tune on random index arrays).
    group = choose_group(x, params, config)
    moe_layer(x, params, config, group)
    moe_layer_ragged(x, params, config)
    padded_inputs = expert_inputs(x, params, config, group)[0]

    def show(label: str, ms: float, layer: bool) -> None:
        print(f"  {'layer:' if layer else 'experts alone:':<15}{label:<34} {ms:8.3f}  ({rate(ms, layer)})")

    # The routed experts alone.
    ours_experts = time_ms(jax.jit(expert_ffn), *padded_inputs, runs=10)
    ragged = jax.jit(lambda xs, wg, wu, wd, s, n: gmm_down(gmm_swiglu(xs, wg, wu, s, n), wd, s, n))
    ragged_args = (xs, params["w_gate"], params["w_up"], params["w_down"], disp.starts, disp.ends)
    ours_ragged_experts = time_ms(ragged, *ragged_args, runs=10)
    experts = jax.jit(baseline_experts, static_argnums=(3, 4, 5))
    ragged_dot_experts = time_ms(experts, xs, bparams, disp.counts, config, "ragged_dot", None, runs=10)
    tilings = megablox_tilings(config, rows)
    megablox_experts, megablox_name = min(
        (_time_or_inf(experts, xs, bparams, disp.counts, config, "megablox", tiling, runs=10), label)
        for label, tiling in tilings
    )

    def routing(x, params):  # what the padded layer does before its kernel: norm, router, sort, gather
        (xs, *_, tile_expert), d, weights = expert_inputs(x, params, config, group)
        return xs, tile_expert, d.pos, weights

    routing_ms = time_ms(jax.jit(routing), x, params, runs=10)
    print(f"  {'':<15}{'(norm + router + dispatch:':<34} {routing_ms:8.3f})")
    show(f"autokernel padded (group {group})", ours_experts, False)
    show("autokernel ragged", ours_ragged_experts, False)
    show("ragged_dot", ragged_dot_experts, False)
    show(f"megablox ({megablox_name})", megablox_experts, False)

    # The whole layer.
    ours_fn = jax.jit(moe_layer, static_argnums=(2, 3))
    ours = time_ms(ours_fn, x, params, config, group, runs=10)
    ours_ragged_fn = jax.jit(moe_layer_ragged, static_argnums=2)
    ours_ragged = time_ms(ours_ragged_fn, x, params, config, runs=10)
    ragged_dot_fn = baseline(config)
    ragged_dot = time_ms(ragged_dot_fn, x, bparams, runs=10)
    megablox = min(_time_or_inf(baseline(config, "megablox", t), x, bparams, runs=10) for _, t in tilings)
    dense = time_ms(baseline(config, "dense"), x, bparams, runs=10) if not prefill else float("nan")
    want = ragged_dot_fn(x, bparams)
    diffs = (max_diff(ours_fn(x, params, config, group), want), max_diff(ours_ragged_fn(x, params, config), want))
    show("autokernel padded", ours, True)
    show("autokernel ragged", ours_ragged, True)
    show("ragged_dot", ragged_dot, True)
    show("megablox (best tiling)", megablox, True)
    if not prefill:
        show("dense one-hot", dense, True)
    best_baseline = min(ragged_dot, megablox, dense if not prefill else float("inf"))
    best_ours = min(ours, ours_ragged)
    print(
        f"  speedup {best_baseline / best_ours:.2f}x over the best baseline; max |layer - ragged_dot layer|: "
        f"padded {diffs[0]:.4f}, ragged {diffs[1]:.4f} (outputs up to {float(jnp.max(jnp.abs(want))):.2f})"
    )
    print("  expert_ffn:", expert_ffn.lower(*padded_inputs).config)
    h = jax.ShapeDtypeStruct((rows, f), jnp.bfloat16)
    print("  gmm_swiglu:", gmm_swiglu.lower(*ragged_args[:3], disp.starts, disp.ends).config)
    print("  gmm_down:  ", gmm_down.lower(h, params["w_down"], disp.starts, disp.ends).config)
    if config.shared:
        s = params["shared"]
        print("  shared:    ", shared_experts.lower(x, params["norm"], s["w_gate"], s["w_up"], s["w_down"], x).config)
    return {
        "name": name,
        "tokens": tokens,
        "group": group,
        "experts": (ours_experts, ours_ragged_experts, ragged_dot_experts, megablox_experts),
        "layer": (ours, ours_ragged, ragged_dot, megablox, dense),
        "rate": rate(best_ours, True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS), help="a layer (repeatable)")
    parser.add_argument("--tokens", action="append", type=int, help="a token count (repeatable)")
    parser.add_argument("--small", action="store_true", help="a tiny layer, runs anywhere")
    args = parser.parse_args()
    if args.small:
        runs = [("small", SMALL, tuple(args.tokens or (8, 128)))]
    else:
        require_tpu()
        names = args.config or [name for name, _ in RUNS]
        default = dict(RUNS)
        runs = [(name, CONFIGS[name], tuple(args.tokens or default[name])) for name in names]
    summary = []
    for name, config, token_counts in runs:
        for tokens in token_counts:
            summary.append(run(config, name, tokens))
            print()
    print(
        "layer              tokens group | experts: padded  ragged  ragged_dot megablox | "
        "layer: padded  ragged  ragged_dot megablox dense   | speedup  best rate"
    )
    for row in summary:
        ex, ly = row["experts"], row["layer"]
        best_base = min(ly[2], ly[3], ly[4] if not np.isnan(ly[4]) else float("inf"))
        print(
            f"{row['name']:<18} {row['tokens']:<6} {row['group']:<5} | "
            f"{ex[0]:15.3f} {ex[1]:7.3f} {ex[2]:10.3f} {ex[3]:8.3f} | "
            f"{ly[0]:13.3f} {ly[1]:7.3f} {ly[2]:10.3f} {ly[3]:8.3f} {ly[4]:7.3f} | "
            f"{best_base / min(ly[0], ly[1]):6.2f}x  {row['rate']}"
        )


if __name__ == "__main__":
    main()
