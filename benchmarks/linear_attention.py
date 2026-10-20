"""Linear attention with the delta rule on a TPU v5e: autokernel's chunked Gated DeltaNet (Qwen3-Next) and KDA
(Kimi Linear) kernels, tuned, against the same chunked algorithm in plain JAX (XLA; flash-linear-attention's
structure: intra-chunk work batched, a `lax.scan` over chunks for the state, outputs batched) and, for scale,
the token-by-token recurrence as a `lax.scan`; then Qwen3-Next's whole Gated DeltaNet layer (norm, projections,
short convolution, gates, delta rule, gated RMSNorm, output projection, residual) against the layer in plain
JAX. bf16 activations, head dim 128.

Two forms of each kernel are timed: the fused kernel (`gated_delta_rule`: per step, every chunk-head's
state-independent work, then the state chain) and the split form (`gated_delta_rule_split`: a parallel
kernel writing each chunk's operands to HBM, then a sequential scan kernel). Each line also gives the tuned
device time of each kernel (from tuning's profiles) and the rest of the wall time (launches, XLA glue).

    uv run python benchmarks/linear_attention.py
    uv run python benchmarks/linear_attention.py --kernel gdn --shape 1,16,4096 --form fused
    uv run python benchmarks/linear_attention.py --kernel layer --layer 2048,16,32,1,4096

Every kernel is first called eagerly, so it tunes on these inputs (the split form's scan on their chunks'
operands), then timed under `jax.jit`.
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
    delta_scan,
    gated_delta_rule,
    gated_delta_rule_gated,
    gated_delta_rule_split,
    kimi_delta_attention,
    kimi_delta_attention_split,
    make_inputs,
    split_inputs,
)
from linear_attention_layer import (
    GATES,
    Config,
    baseline_layer,
    gated_deltanet_layer,
    init_params,
    prepare,
    residual_matmul,
)
from rmsnorm_matmul import rmsnorm_matmul
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
FORMS = {  # name: (fused kernel, split form)
    "gdn": (gated_delta_rule, gated_delta_rule_split),
    "kda": (kimi_delta_attention, kimi_delta_attention_split),
}


def _max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _time(fn: Callable[..., Any], *args: Any, small: bool, slow: bool = False) -> float:
    if small:
        return time_ms(fn, *args, runs=1, repeats=1)
    return time_ms(fn, *args, runs=2, repeats=3) if slow else time_ms(fn, *args)


def _device_ms(kernel: Any, *args: Any) -> float:
    """A kernel's tuned device time for these inputs, ms (0 off a TPU: nothing is timed there)."""
    return kernel.tuning(*args).best.seconds * 1e3 if kernel.tuning(*args).trials else 0.0


def _tuning(kernel: Any, *args: Any) -> str:
    return "    " + str(kernel.tuning(*args)).replace("\n", "\n    ")


def bench_kernels(names: list[str], shapes: list[str], forms: list[str], small: bool) -> list[tuple]:
    rows = []
    for name in names:
        fused, split = FORMS[name]
        for b, h, s in [tuple(map(int, shape.split(","))) for shape in shapes]:
            inputs = make_inputs(b, h, s, per_channel=name == "kda")
            print(f"{fused.nest.name}: B={b} H={h} S={s} head dim 128: ms")
            want = baseline_chunked(*inputs)
            scale = float(jnp.max(jnp.abs(want.astype(jnp.float32))))
            t_rec = _time(baseline_recurrent, *inputs, small=small, slow=True)
            t_chunked = _time(baseline_chunked, *inputs, small=small)
            print(f"  recurrence, lax.scan over tokens   {t_rec:9.3f}")
            print(f"  chunked, plain JAX                 {t_chunked:9.3f}")
            results = {}
            if "fused" in forms:
                fused(*inputs)  # tunes on these inputs
                t = _time(jax.jit(fused), *inputs, small=small)
                diff = _max_diff(fused(*inputs), want)
                print(f"  autokernel fused, tuned            {t:9.3f}   (kernel {_device_ms(fused, *inputs):.3f})")
                print(_tuning(fused, *inputs))
                results["fused"] = (t, diff)
            if "split" in forms:
                split(*inputs)  # tunes each kernel on these inputs' operands
                t = _time(jax.jit(split), *inputs, small=small)
                diff = _max_diff(split(*inputs), want)
                chunks, args = split_inputs(*inputs)
                packed, decays = chunks(*args)
                t_chunks, t_scan = _device_ms(chunks, *args), _device_ms(delta_scan, packed, decays)
                print(
                    f"  autokernel split, tuned            {t:9.3f}   (chunks {t_chunks:.3f} + scan {t_scan:.3f}, "
                    f"other {t - t_chunks - t_scan:.3f})"
                )
                print(_tuning(chunks, *args))
                print(_tuning(delta_scan, packed, decays))
                results["split"] = (t, diff)
            for form, (t, diff) in results.items():
                print(
                    f"  {form}: speedup {t_chunked / t:.2f}x over chunked JAX, {t_rec / t:.1f}x over the recurrence; "
                    f"max |autokernel - chunked JAX| {diff:.4f} (outputs up to {scale:.3f})"
                )
                rows.append((name, form, b, h, s, t_rec, t_chunked, t, diff))
    return rows


def layer_kernels(x: jax.Array, params: dict[str, Any], config: Config) -> list[tuple[str, Any, tuple]]:
    """The layer's kernels (as `gated_deltanet_layer` calls them), with their inputs: (name, kernel, args)."""
    B, S, D = x.shape
    rows = x.reshape(B * S, D)
    proj = rmsnorm_matmul(rows, params["norm"], params["w_qkvz"]).reshape(B, S, -1)
    gates = rmsnorm_matmul(rows, params["norm"], params["w_ba"]).reshape(B, S, GATES)
    delta = (*prepare(proj, gates, params, config), proj, params["out_norm"])
    o = gated_delta_rule_gated(*delta).reshape(B * S, config.value_dim)
    return [
        ("qkvz projection", rmsnorm_matmul, (rows, params["norm"], params["w_qkvz"])),
        ("gate projection", rmsnorm_matmul, (rows, params["norm"], params["w_ba"])),
        ("delta rule", gated_delta_rule_gated, delta),
        ("output projection", residual_matmul, (rows, o, params["w_out"])),
    ]


def bench_layers(layers: list[str], small: bool) -> list[tuple]:
    rows = []
    ours = jax.jit(gated_deltanet_layer, static_argnames="config")
    for d, hk, hv, b, s in [tuple(map(int, layer.split(","))) for layer in layers]:
        config = Config(d_model=d, k_heads=hk, v_heads=hv)
        params = init_params(config)
        x = jax.random.normal(jax.random.key(1), (b, s, d), jnp.bfloat16)
        print(f"Gated DeltaNet layer: {config}, batch {b}, seq {s}: ms")
        gated_deltanet_layer(x, params, config)  # tunes each kernel on these inputs
        t_base = _time(baseline_layer, x, params, config, small=small)
        t_ours = _time(ours, x, params, config, small=small)
        diff = _max_diff(ours(x, params, config), baseline_layer(x, params, config))
        split = {name: _device_ms(kernel, *args) for name, kernel, args in layer_kernels(x, params, config)}
        print(f"  plain JAX (chunked delta rule)     {t_base:9.3f}")
        print(f"  autokernel, tuned                  {t_ours:9.3f}")
        kernels = ", ".join(f"{name} {ms:.3f}" for name, ms in split.items())
        print(f"    kernels: {kernels}; other {t_ours - sum(split.values()):.3f}")
        print(f"  speedup {t_base / t_ours:.2f}x; max |autokernel - plain JAX| {diff:.4f}")
        rows.append((d, hk, hv, b, s, t_base, t_ours, diff))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["gdn", "kda", "layer", "all"], default="all")
    parser.add_argument("--shape", action="append", help="B,H,S for the delta-rule kernels (repeatable)")
    parser.add_argument("--layer", action="append", help="d_model,k_heads,v_heads,B,S (repeatable)")
    parser.add_argument("--form", choices=["fused", "split", "both"], default="both")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny shapes, any backend
    args = parser.parse_args()
    if not args.small:
        require_tpu()
    names = {"all": ["gdn", "kda"], "layer": []}.get(args.kernel, [args.kernel])
    forms = ["fused", "split"] if args.form == "both" else [args.form]
    kernels = bench_kernels(names, args.shape or (SMALL_SHAPES if args.small else SHAPES), forms, args.small)
    layers = []
    if args.kernel in ("layer", "all"):
        layers = bench_layers(args.layer or (SMALL_LAYERS if args.small else LAYERS), args.small)
    if kernels:
        print("\nkernel  form   B  H   S     recurrence  chunked JAX  autokernel  vs chunked  vs recurrence  max diff")
        for name, form, b, h, s, t_rec, t_chunked, t_ours, diff in kernels:
            print(
                f"{name:<7} {form:<6} {b:<2} {h:<3} {s:<5} {t_rec:10.3f} {t_chunked:12.3f} {t_ours:11.3f}"
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
