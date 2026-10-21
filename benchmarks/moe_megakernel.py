"""Routing inside the kernel: a stack of MoE layers decoding as one autokernel kernel
(examples/moe_megakernel.py) against moe.py's decode layer per layer (XLA routing, then `moe_decode`) and
against plain JAX (XLA, every expert on every token, one-hot weighted), bf16, on a TPU v5e.

Each method's time is the median wall-clock time per call of the stack under jax.jit (the megakernel tuned
first, eagerly, on these tokens). GB/s: the chosen experts' weights and the routers each layer must read,
over that time.

    uv run python benchmarks/moe_megakernel.py
    uv run python benchmarks/moe_megakernel.py --shape 4,2048,128,768:8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
import moe
from moe_megakernel import TOP_K, Config, init_params, layer_params, megakernel_layers, moe_megakernel, per_layer
from moe_megakernel import xla_layers as xla_stack
from timing import require_tpu, time_ms

SHAPES = [  # layers, d_model, experts, ffn : tokens  (Qwen3-30B-A3B's layers; two: 2.4 GB, and again per layer)
    "2,2048,128,768:1",
    "2,2048,128,768:4",
    "2,2048,128,768:8",
    "2,2048,128,768:32",
    "2,2048,128,768:64",
]


def read_bytes(x, params, config: Config) -> int:
    """The bytes a stack must read for these tokens: each layer's router and the experts its tokens chose."""
    expert = 3 * config.d_model * config.ffn * 2
    total, h = 0, x
    for p in layer_params(params, config):
        normed = moe.rmsnorm(h, p["norm"])
        ids, _ = moe.route(normed, p["router"], p["bias"], config.moe)
        total += len(np.unique(np.asarray(ids))) * expert + p["router"].nbytes
        h = moe.moe_layer(h, p, config.moe)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="layers,d_model,experts,ffn:tokens (repeatable)")
    parser.add_argument("--xla", action="store_true", help="also time plain JAX (slow: every expert)")
    args = parser.parse_args()
    require_tpu()
    rows = []
    for shape in args.shape or SHAPES:
        model, tokens = shape.split(":")
        layers, d, e, f = map(int, model.split(","))
        config = Config(layers, d, e, TOP_K, f)
        params = init_params(config)
        x = jax.random.normal(jax.random.key(1), (int(tokens), d), jnp.bfloat16)
        start = time.perf_counter()
        megakernel_layers(x, params)  # tune eagerly, on these tokens
        print(f"{shape}: tuned in {time.perf_counter() - start:.0f} s", flush=True)
        layers_ = layer_params(params, config, gate_up=args.xla)
        per_layer(x, layers_, config)
        times = {
            "one kernel, routing inside": time_ms(jax.jit(megakernel_layers), x, params),
            "moe.py's layer per layer": time_ms(jax.jit(per_layer, static_argnums=2), x, layers_, config),
        }
        if args.xla:
            times["plain JAX (XLA)"] = time_ms(jax.jit(xla_stack, static_argnums=2), x, layers_, config)
        ours = megakernel_layers(x, params).astype(jnp.float32)
        theirs = per_layer(x, layers_, config).astype(jnp.float32)
        diff = float(jnp.max(jnp.abs(ours - theirs)))
        nbytes = read_bytes(x, params, config)
        print(f"{shape}: {nbytes / 1e6:.0f} MB to read; ms (GB/s)", flush=True)
        for name, ms in times.items():
            print(f"  {name:28} {ms:8.3f}  ({nbytes / ms / 1e6:6.1f})", flush=True)
        args_ = (x, params["gamma"], params["router"], params["w_gate"], params["w_up"], params["w_down"])
        print(f"  max |one kernel - per layer| {diff:.4f}; tuned {moe_megakernel.tuning(*args_).config}")
        rows.append((shape, *times.values()))
        del params, layers_
    print("\nshape (layers,d,experts,ffn:tokens)  one kernel  per layer  speedup")
    for shape, ours, theirs, *_ in rows:
        print(f"{shape:36} {ours:10.3f} {theirs:10.3f} {theirs / ours:7.2f}x")


if __name__ == "__main__":
    main()
