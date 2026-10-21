"""A whole decode step of a dense GQA transformer (Qwen3-like, QK-norm) as ONE kernel with cross-layer prefetch
(examples/decode_megakernel.py), against the same step as decode_model.py's per-layer kernels (three per
layer, QK-norm / RoPE / cache append in XLA between them) and in plain JAX (XLA, dense masked attention), on
identical weights and caches, bf16, on a TPU v5e.

Each method runs `--steps` greedy decode steps chained under jax.jit, the caches donated (appended in
place); its first step runs eagerly, so its kernels tune on the real cache lengths (ragged, uniform in
[mean / 2, 3 mean / 2)). Reported per method: ms per step, tokens/s, and the HBM bandwidth the step's
bytes need (the weights read once; plus the valid K/V) against the v5e's ~819 GB/s; for the autokernel
methods also their kernels' tuned device time (the megakernel's one kernel; the per-layer kernels' sum),
which shows what the launches between kernels cost.

    uv run python benchmarks/decode_megakernel.py
    uv run python benchmarks/decode_megakernel.py --shape 16,2048,16,8,6144,32768:8,1024 --steps 32
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from decode import gqa_decode
from decode_megakernel import (
    HEAD_DIM,
    STACKED,
    Config,
    decode_megakernel,
    init_params,
    kernel_inputs,
    layer_weights,
    megakernel_step,
    per_layer_step,
    random_caches,
    xla_step,
)
from decode_model import decode_mlp
from rmsnorm_matmul import rmsnorm_matmul
from timing import require_tpu

PEAK_GBS = 819.0  # v5e HBM bandwidth
SHAPES = [  # layers, d_model, heads, kv_heads, ffn, vocab : batch, mean context
    "16,2048,16,8,6144,32768:1,1024",
    "16,2048,16,8,6144,32768:8,512",
    "16,2048,16,8,6144,32768:8,2048",
    "8,2048,16,8,6144,32768:8,4096",
    "8,2048,16,8,6144,32768:32,1024",
    "8,2048,16,8,6144,32768:64,512",
    "8,2048,16,8,6144,32768:64,32",
]
SMALL = ["2,256,4,2,512,1024:4,128"]  # --small: a quick run anywhere (the interpreter off a TPU)


def ragged_lengths(batch: int, mean: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)


def cache_len(mean: int, extra: int) -> int:
    """S_max: room for the longest sequence and `extra` more tokens, rounded up to a multiple of 1024 (of 256
    for short contexts: an input signature of their own, so the kernels tune on them, not on longer ones)."""
    unit = 1024 if mean >= 256 else 256
    return -(-(3 * mean // 2 + extra) // unit) * unit


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def time_steps(step: Callable, eager: Callable, tokens: jax.Array, state: tuple, extra: tuple, n: int):
    """ms per step over n chained greedy steps of the jitted `step(tokens, *state, *extra) -> (logits,
    *state)` (state donated), after one eager step (`eager`, same signature: tunes the kernels on the real
    lengths) and two warm-up steps; and the first step's logits (all methods start from the same state)."""
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


def best_seconds(kernel: Any, *args: Any) -> float:
    """The tuned device time of `kernel` on inputs like these (inf if nothing was timed: off a TPU)."""
    return kernel.tuning(*args).best.seconds


def bench(shape: str, steps: int, on_tpu: bool) -> tuple:
    model, run = shape.split(":")
    config = Config(*map(int, model.split(",")))
    batch, mean = map(int, run.split(","))
    s_max = cache_len(mean, steps + 4)
    lengths = ragged_lengths(batch, mean)
    tokens = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
    params = init_params(config)
    weights = sum(params[name].nbytes for name in (*STACKED, "final_norm", "w_vocab"))
    kv = config.layers * float(jnp.sum(lengths) + batch * (steps / 2 + 2)) * config.kv_heads * HEAD_DIM * 2 * 2
    print(f"decode step {config}, batch {batch}, mean context {mean} (S_max {s_max})")
    print(f"  per step: {weights / 1e6:.0f} MB of weights, ~{kv / 1e6:.0f} MB of valid K/V")
    results: dict[str, tuple[float, float, float]] = {}  # name -> (ms, max |logits - megakernel|, device ms)

    # The megakernel, on stacked weights and caches.
    def mega(tok, k, v, lens, p):
        return megakernel_step(tok, k, v, lens, p)

    caches = random_caches(config, lengths, s_max)
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in kernel_inputs(tokens, *caches, lengths, params)]
    t, want = time_steps(jax.jit(mega, donate_argnums=(1, 2)), mega, tokens, (*caches, lengths), (params,), steps)
    del caches
    results["megakernel (one kernel)"] = (t, 0.0, best_seconds(decode_megakernel, *specs) * 1e3)
    print("  megakernel tuned:", decode_megakernel.tuning(*specs).config)

    # The baselines, on per-layer weights and caches (the stacked per-layer weights dropped first).
    layers = layer_weights(params)
    params = {name: params[name] for name in ("embed", "final_norm", "w_vocab")}
    for name, fn in (("per-layer kernels (decode_model.py)", per_layer_step), ("plain JAX (XLA)", xla_step)):

        def method(tok, caches, lens, lay, p, fn=fn):
            return fn(tok, caches, lens, lay, p)

        caches = random_caches(config, lengths, s_max, stacked=False)
        t, logits = time_steps(
            jax.jit(method, donate_argnums=1), method, tokens, (caches, lengths), (layers, params), steps
        )
        device = float("nan")
        if fn is per_layer_step and on_tpu:
            device = per_layer_device_ms(config, layers, params, lengths, caches, batch)
        results[name] = (t, max_diff(logits, want), device)
        del caches

    for name, (t, diff, device) in results.items():
        rate = (weights + kv) / t / 1e6
        dev = f"  kernels {device:7.3f} ms" if device == device and device != float("inf") else ""
        print(
            f"  {name:38} {t:8.3f} ms {batch / t * 1e3:9.0f} tok/s  weights {weights / t / 1e6:4.0f} GB/s"
            f"  all {rate:4.0f} GB/s ({rate / PEAK_GBS:4.0%})  max |logits - megakernel| {diff:.3f}{dev}"
        )
    times = [r[0] for r in results.values()]
    diff = max(r[1] for r in results.values())
    return (model, batch, mean, *times, batch / times[0] * 1e3, weights / times[0] / 1e6, diff)


def per_layer_device_ms(config: Config, layers: list, params: dict, lengths: jax.Array, caches: list, batch: int):
    """The per-layer kernels' tuned device times, summed over a step (their tunings are cached by input
    signature, from the eager first step)."""
    x = jax.ShapeDtypeStruct((batch, config.d_model), jnp.bfloat16)
    q = jax.ShapeDtypeStruct((batch, 1, config.heads, HEAD_DIM), jnp.bfloat16)
    attn = jax.ShapeDtypeStruct((batch, config.heads * HEAD_DIM), jnp.bfloat16)
    layer = layers[0]
    per_layer = (
        best_seconds(rmsnorm_matmul, x, layer["attn_norm"], layer["w_qkv"])
        + best_seconds(gqa_decode, q, caches[0]["k"], caches[0]["v"], lengths + 1)
        + best_seconds(decode_mlp, x, attn, *(layer[n] for n in ("wo", "ffn_norm", "w_gate", "w_up", "w_down")))
    )
    return (config.layers * per_layer + best_seconds(rmsnorm_matmul, x, params["final_norm"], params["w_vocab"])) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="layers,d_model,heads,kv_heads,ffn,vocab:batch,mean_context")
    parser.add_argument("--steps", type=int, default=32, help="decode steps timed per method")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny shapes, runs anywhere
    args = parser.parse_args()
    on_tpu = jax.default_backend() == "tpu"
    if not args.small:
        require_tpu()
    shapes = args.shape or (SMALL if args.small else SHAPES)
    rows = [bench(shape, 2 if args.small else args.steps, on_tpu) for shape in shapes]
    print("\nDecode step, ms per step (bf16, v5e): megakernel vs per-layer kernels vs plain JAX (XLA)")
    print(
        "model (layers,d,H,Hkv,ffn,vocab)  B   mean  megakernel  per-layer  XLA      vs per-layer  vs XLA"
        "  tokens/s  weight GB/s  max diff"
    )
    for model, b, mean, mega, per_layer, xla, rate, gbs, diff in rows:
        print(
            f"{model:<33} {b:<3} {mean:<5} {mega:10.3f} {per_layer:10.3f} {xla:8.3f} {per_layer / mega:11.2f}x"
            f" {xla / mega:6.2f}x {rate:9.0f} {gbs:12.0f} {diff:9.3f}"
        )


if __name__ == "__main__":
    main()
