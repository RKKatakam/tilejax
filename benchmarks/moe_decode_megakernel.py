"""A whole MoE decode step as ONE kernel, the routing inside it (examples/moe_decode_megakernel.py), against
the same step with kernels per layer (RMSNorm + QKV, attention, output projection, then moe.py's decode
layer: XLA routing and `moe_decode`), on identical weights and caches, bf16, on a TPU v5e.

Each method runs `--steps` greedy decode steps chained under jax.jit, the caches donated; its first step
runs eagerly, so the kernels tune on the real lengths and routing. Reported: ms per step, tokens/s.

    uv run python benchmarks/moe_decode_megakernel.py
    uv run python benchmarks/moe_decode_megakernel.py --shape 2,2048,32,4,128,768,32768:8,1024
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
from decode_megakernel import layer_caches, random_caches
from moe_decode_megakernel import (
    Config,
    init_params,
    kernel_inputs,
    layer_weights,
    megakernel_step,
    moe_decode_megakernel,
    per_layer_step,
)
from timing import require_tpu

import autokernel

SHAPES = [  # layers, d_model, heads, kv_heads, experts, ffn, vocab : batch, mean context (Qwen3-30B-A3B's layers)
    "2,2048,32,4,128,768,32768:1,1024",
    "2,2048,32,4,128,768,32768:8,1024",
    "2,2048,32,4,128,768,32768:32,1024",
]


def time_steps(step, eager, tokens, state, extra, n):
    """ms per step over n chained greedy steps (state donated), after one eager step (tuning on real inputs)
    and two warm-up steps; and the first step's logits."""
    first, *state = jax.block_until_ready(eager(tokens, *state, *extra))
    tokens = jnp.argmax(first, axis=-1).astype(jnp.int32)
    for i in range(n + 2):
        if i == 2:
            jax.block_until_ready(tokens)
            start = time.perf_counter()
        logits, *state = step(tokens, *state, *extra)
        tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    jax.block_until_ready(tokens)
    return (time.perf_counter() - start) / n * 1e3, first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--profile", action="store_true", help="also print the one kernel's profile")
    args = parser.parse_args()
    require_tpu()
    rows = []
    for shape in args.shape or SHAPES:
        model, run = shape.split(":")
        config = Config(*map(int, model.split(",")))
        batch, mean = map(int, run.split(","))
        s_max = -(-(3 * mean // 2 + args.steps + 4) // 1024) * 1024
        lengths = jnp.asarray(np.random.default_rng(0).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)
        params = init_params(config)
        tokens = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
        print(f"{shape}: S_max {s_max}", flush=True)

        def mega(tok, k, v, lens, p):
            return megakernel_step(tok, k, v, lens, p)

        caches = random_caches(config, lengths, s_max)
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in kernel_inputs(tokens, *caches, lengths, params)]
        t_mega, want = time_steps(
            jax.jit(mega, donate_argnums=(1, 2)), mega, tokens, (*caches, lengths), (params,), args.steps
        )
        print(
            f"  one kernel                {t_mega:8.3f} ms   tuned {moe_decode_megakernel.tuning(*specs).config}",
            flush=True,
        )
        if args.profile:  # (the timed steps donated `caches`: fresh ones)
            fresh = kernel_inputs(tokens, *random_caches(config, lengths, s_max), lengths, params)
            print(autokernel.profile(moe_decode_megakernel, *fresh), flush=True)
        del caches
        layers = layer_weights(params)

        def per_layer(tok, caches, lens, lay, p, config=config):
            return per_layer_step(tok, caches, lens, lay, p, config)

        caches = layer_caches(*random_caches(config, lengths, s_max))
        t_layer, theirs = time_steps(
            jax.jit(per_layer, donate_argnums=1), per_layer, tokens, (caches, lengths), (layers, params), args.steps
        )
        diff = float(jnp.max(jnp.abs(want.astype(jnp.float32) - theirs.astype(jnp.float32))))
        print(f"  kernels per layer         {t_layer:8.3f} ms   max |logits - one kernel| {diff:.3f}", flush=True)
        rows.append((shape, t_mega, t_layer, batch))
        del caches, layers, params
    print("\nshape (layers,d,H,Hkv,E,F,vocab:B,mean)   one kernel  per layer  speedup  tokens/s (one kernel)")
    for shape, mega, layer, batch in rows:
        print(f"{shape:40} {mega:10.3f} {layer:10.3f} {layer / mega:7.2f}x {batch / mega * 1e3:10.0f}")


if __name__ == "__main__":
    main()
