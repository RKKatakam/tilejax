"""Linear attention with the delta rule on a TPU v5e: autokernel's chunked Gated DeltaNet (Qwen3-Next) and KDA
(Kimi Linear) kernels, tuned, against the same chunked algorithm in plain JAX (XLA; flash-linear-attention's
structure: intra-chunk work batched, a `lax.scan` over chunks for the state, outputs batched) and, for scale,
the token-by-token recurrence as a `lax.scan`; then Qwen3-Next's whole Gated DeltaNet layer (norm, projections,
short convolution, gates, delta rule, gated RMSNorm, output projection, residual) against the layer in plain
JAX. bf16 activations, head dim 128.

    uv run python benchmarks/linear_attention.py
    uv run python benchmarks/linear_attention.py --kernel gdn --shape 1,16,4096
    uv run python benchmarks/linear_attention.py --kernel layer --layer 2048,16,32,1,4096

(Tuning compiles each kernel's loop nest as plain JAX once per shape, every chunk of every head unrolled:
expect a minute or two of compilation per kernel and shape.)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from linear_attention import (
    baseline_chunked,
    baseline_recurrent,
    gated_delta_rule,
    kimi_delta_attention,
    make_inputs,
)
from linear_attention_layer import Config, baseline_layer, gated_deltanet_layer, init_params
from timing import require_tpu, time_ms

SHAPES = ["1,16,2048", "1,16,4096", "1,16,8192", "2,32,2048", "4,16,2048"]  # batch, heads, seq
LAYERS = [  # d_model, query/key heads, value heads, batch, seq
    "2048,16,32,1,2048",  # Qwen3-Next-80B-A3B's linear-attention layer
    "2048,16,32,1,4096",
    "2048,16,32,2,2048",
    "1024,8,16,4,2048",
]
SMALL_SHAPES = ["1,2,1024"]  # --small: a smoke test of this script, off a TPU (in the interpreter)
SMALL_LAYERS = ["256,1,2,1,512"]


def _max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _time(fn: Callable[..., Any], *args: Any, small: bool, slow: bool = False) -> float:
    if small:
        return time_ms(fn, *args, runs=1, repeats=1)
    return time_ms(fn, *args, runs=2, repeats=3) if slow else time_ms(fn, *args)


def bench_kernels(names: list[str], shapes: list[str], small: bool) -> list[tuple]:
    rows = []
    for name in names:
        kernel, per_channel = {"gdn": (gated_delta_rule, False), "kda": (kimi_delta_attention, True)}[name]
        ours = jax.jit(kernel)
        for b, h, s in [tuple(map(int, shape.split(","))) for shape in shapes]:
            inputs = make_inputs(b, h, s, per_channel)
            print(f"{kernel.nest.name}: B={b} H={h} S={s} head dim 128: ms")
            t_rec = _time(baseline_recurrent, *inputs, small=small, slow=True)
            t_chunked = _time(baseline_chunked, *inputs, small=small)
            t_ours = _time(ours, *inputs, small=small)  # tunes on the first call
            want = baseline_chunked(*inputs)
            diff, scale = _max_diff(ours(*inputs), want), float(jnp.max(jnp.abs(want.astype(jnp.float32))))
            print(f"  recurrence, lax.scan over tokens   {t_rec:9.3f}")
            print(f"  chunked, plain JAX                 {t_chunked:9.3f}")
            print(f"  autokernel, tuned                  {t_ours:9.3f}")
            print(
                f"  speedup {t_chunked / t_ours:.2f}x over chunked JAX, {t_rec / t_ours:.1f}x over the recurrence; "
                f"max |autokernel - chunked JAX| {diff:.4f} (outputs up to {scale:.3f})"
            )
            print("    " + str(kernel.tuning(*inputs)).replace("\n", "\n    "))
            rows.append((name, b, h, s, t_rec, t_chunked, t_ours, diff))
    return rows


def bench_layers(layers: list[str], small: bool) -> list[tuple]:
    rows = []
    ours = jax.jit(gated_deltanet_layer, static_argnames="config")
    for d, hk, hv, b, s in [tuple(map(int, layer.split(","))) for layer in layers]:
        config = Config(d_model=d, k_heads=hk, v_heads=hv)
        params = init_params(config)
        x = jax.random.normal(jax.random.key(1), (b, s, d), jnp.bfloat16)
        print(f"Gated DeltaNet layer: {config}, batch {b}, seq {s}: ms")
        t_base = _time(baseline_layer, x, params, config, small=small)
        t_ours = _time(ours, x, params, config, small=small)  # tunes each kernel on the first call
        diff = _max_diff(ours(x, params, config), baseline_layer(x, params, config))
        print(f"  plain JAX (chunked delta rule)     {t_base:9.3f}")
        print(f"  autokernel, tuned                  {t_ours:9.3f}")
        print(f"  speedup {t_base / t_ours:.2f}x; max |autokernel - plain JAX| {diff:.4f}")
        rows.append((d, hk, hv, b, s, t_base, t_ours, diff))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["gdn", "kda", "layer", "all"], default="all")
    parser.add_argument("--shape", action="append", help="B,H,S for the delta-rule kernels (repeatable)")
    parser.add_argument("--layer", action="append", help="d_model,k_heads,v_heads,B,S (repeatable)")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny shapes, any backend
    args = parser.parse_args()
    if not args.small:
        require_tpu()
    names = {"all": ["gdn", "kda"], "layer": []}.get(args.kernel, [args.kernel])
    kernels = bench_kernels(names, args.shape or (SMALL_SHAPES if args.small else SHAPES), args.small)
    layers = []
    if args.kernel in ("layer", "all"):
        layers = bench_layers(args.layer or (SMALL_LAYERS if args.small else LAYERS), args.small)
    if kernels:
        print("\nkernel  B  H   S     recurrence  chunked JAX  autokernel  vs chunked  vs recurrence  max diff")
        for name, b, h, s, t_rec, t_chunked, t_ours, diff in kernels:
            print(
                f"{name:<7} {b:<2} {h:<3} {s:<5} {t_rec:10.3f} {t_chunked:12.3f} {t_ours:11.3f}"
                f" {t_chunked / t_ours:10.2f}x {t_rec / t_ours:13.1f}x {diff:9.4f}"
            )
    if layers:
        print("\nlayer: d_model  k heads  v heads  B  S     plain JAX  autokernel  speedup  max diff")
        for d, hk, hv, b, s, t_base, t_ours, diff in layers:
            print(
                f"       {d:<7} {hk:<8} {hv:<8} {b:<2} {s:<5} {t_base:9.3f} {t_ours:11.3f}"
                f" {t_base / t_ours:7.2f}x {diff:9.4f}"
            )


if __name__ == "__main__":
    main()
