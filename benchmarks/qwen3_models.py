"""End-to-end Qwen3 inference on a TPU v5e (examples/qwen3_models.py): a dense Qwen3 (1.7B-like dims) and a
Qwen3-MoE (30B-A3B-like dims: 128 experts, top 8), 4 layers each, vocab 32k, bf16; every kernel tuned, against
the same models in plain JAX.

- Prefill (prompts (B, S) -> last-token logits + KV cache), ms and tokens/s: autokernel's kernels against
  plain JAX with JAX's splash attention (GQA as its MQA kernel per kv head or MHA with repeated kv; its
  default blocks, then its best over a grid) and XLA's FFN, or for MoE `jax.lax.ragged_dot` / JAX's Pallas
  megablox gmm (best of a few tilings).
- Decode (one token per sequence against ragged cache lengths, uniform in [mean / 2, 3 mean / 2)), ms per
  step and tokens/s over chained greedy steps with the caches donated: autokernel's kernels against plain
  JAX with XLA's attention over the whole cache or JAX's ragged_paged_attention (pages of 128), and XLA's
  FFN, or for MoE ragged_dot or every expert on every token (whichever is fastest).
- Per phase (attention / FFN or MoE experts / projections): the tuned kernels' device times, per model
  call (each layer's kernels times the layers); the rest of the wall time is XLA glue (embedding, routing
  and dispatch, cache writes) and launch gaps.

    uv run python benchmarks/qwen3_models.py
    uv run python benchmarks/qwen3_models.py --model moe --prefill 1,4096 --decode 8,2048 --decode 32,1024
    uv run python benchmarks/qwen3_models.py --model dense --layers 8 --steps 32
    uv run python benchmarks/qwen3_models.py --small      # tiny models, anywhere (the interpreter off a TPU)

The MoE model's experts are 4.8 GB; the baseline holds its gate and up weights side by side (one grouped
matmul), so the benchmark converts the model's weights layer by layer after timing autokernel, keeping the
experts in HBM about once (peak ~6 GB plus caches).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from kv_cache import page_tables, ragged_kv_pages, to_pages
from moe import megablox_tiling
from qwen3_models import (
    CONFIGS,
    ModelConfig,
    baseline_decode,
    baseline_params,
    baseline_prefill,
    decode_calls,
    decode_step,
    device_ms,
    init_params,
    jitted_decode_step,
    prefill,
    prefill_calls,
    prefill_group,
    random_cache,
    round_up,
)
from timing import require_tpu, time_ms

PREFILL = ["1,2048", "1,4096", "4,2048"]  # batch, prompt length
DECODE = ["1,1024", "1,4096", "8,1024", "8,4096", "32,1024", "32,2048"]  # batch, mean context
SPLASH_BLOCKS = (512, 1024)  # splash's block_q and block_kv grid
MEGABLOX_TILINGS = ((512, 2048), (256, 1024))  # rows, and k / n blocks up to
PAGE = 128  # ragged_paged_attention's page size
FUSED = "attention + Wo + FFN (fused layer)"
SMALL = {
    "dense": ModelConfig(layers=2, d_model=256, heads=4, kv_heads=2, head_dim=128, ffn=512, vocab=512),
    "moe": ModelConfig(
        layers=2, d_model=256, heads=4, kv_heads=1, head_dim=128, ffn=256, vocab=512, experts=8, top_k=2
    ),
}


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def rows_agreeing(a: jax.Array, b: jax.Array, tol: float = 3e-2) -> float:
    """The share of logits rows within `tol` relative RMS error of the baseline's (MoE: a token whose router
    is near a tie can take another expert in one model than in the other)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, -1) / np.mean(b**2, -1)) < tol))


def best_of(candidates: dict[str, Callable[[], float]]) -> tuple[float, str]:
    """The fastest of some named timings (those that fail to compile or run are skipped)."""
    best, name = float("inf"), "-"
    for label, timing in candidates.items():
        try:
            t = timing()
        except Exception as e:  # a block size or tiling the kernel rejects, out of memory, ...
            print(f"    ({label}: {type(e).__name__}: {str(e).splitlines()[0][:100]})")
            continue
        print(f"    {label:<56} {t:9.3f} ms")
        if t < best:
            best, name = t, label
    return best, name


def ragged_lengths(batch: int, mean: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)


def time_steps(step: Callable, token: jax.Array, cache: Any, lengths: jax.Array, params: Any, n: int, *extra: Any):
    """Wall-clock ms per greedy decode step over n chained steps (the caches donated), after 2 warm-up
    steps; and the first step's logits."""
    first = None
    for i in range(n + 2):
        if i == 2:
            jax.block_until_ready(token)
            start = time.perf_counter()
        logits, cache, lengths = step(token, cache, lengths, params, *extra)
        first = logits if first is None else first
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    jax.block_until_ready(token)
    return (time.perf_counter() - start) / n * 1e3, first


def show_phases(calls: list, layers: int, wall_ms: float) -> dict[str, float]:
    """Print each kernel's tuned schedule and device time, and the per-phase totals; return the totals."""
    for call in calls:
        tuning = call.kernel.tuning(*call.specs)
        print(f"    {call.name:<40} {tuning.best.seconds * 1e3:8.3f} ms  {tuning.config}")
    phases = device_ms(calls, layers)
    kernels = sum(phases.values())
    parts = ", ".join(f"{phase} {ms:.3f}" for phase, ms in sorted(phases.items()))
    print(f"    per call (x {layers} layers): {parts}; kernels {kernels:.3f} of {wall_ms:.3f} ms wall")
    return phases


# --------------------------------------------------------------------------- autokernel


def ours_prefill(params: dict, batch: int, seq: int, timed: Callable, on_tpu: bool) -> dict[str, Any]:
    config: ModelConfig = params["config"]
    tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, config.vocab)
    max_len = round_up(seq + 256, 512)
    group = prefill_group(tokens, params)  # MoE on a TPU: measured on the first layer's routing
    prefill(tokens, params, max_len, None, group)  # eagerly: every kernel tunes on real activations, routing
    fn = jax.jit(prefill, static_argnums=(2, 4))
    t = timed(fn, tokens, params, max_len, None, group)
    logits = fn(tokens, params, max_len, None, group)[0]
    print(f"  prefill B={batch} S={seq}: autokernel {t:.3f} ms, {batch * seq / t * 1e3:,.0f} tokens/s")
    phases = {}
    if on_tpu:
        calls = prefill_calls(config, batch, seq, group)
        if config.moe is None:  # the layer's form, as qwen3.fuse_ffn chose it: fused or attention + FFN
            seconds = {c.name: c.kernel.tuning(*c.specs).best.seconds for c in calls}
            split = sum(s for name, s in seconds.items() if name in ("attention + Wo + residual", "SwiGLU FFN"))
            calls = prefill_calls(config, batch, seq, group, fused=seconds[FUSED] <= split)
        phases = show_phases(calls, config.layers, t)
    return {"tokens": tokens, "max_len": max_len, "ms": t, "logits": logits, "phases": phases}


def ours_decode(params: dict, batch: int, mean: int, steps: int, on_tpu: bool, multiple: int) -> dict[str, Any]:
    config: ModelConfig = params["config"]
    lengths = ragged_lengths(batch, mean)
    max_len = round_up(3 * mean // 2 + steps + 4, multiple)
    token = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
    decode_step(token, random_cache(config, lengths, max_len), lengths, params)  # eagerly: tunes on real data
    t, logits = time_steps(jitted_decode_step, token, random_cache(config, lengths, max_len), lengths, params, steps)
    print(
        f"  decode B={batch} mean context {mean} (S_max {max_len}): autokernel {t:.3f} ms/step, "
        f"{batch / t * 1e3:,.0f} tokens/s"
    )
    phases = show_phases(decode_calls(config, batch, max_len), config.layers, t) if on_tpu else {}
    return {
        "lengths": lengths,
        "mean": mean,
        "max_len": max_len,
        "token": token,
        "ms": t,
        "logits": logits,
        "phases": phases,
    }


# --------------------------------------------------------------------------- baselines


def to_baseline(params: dict) -> dict:
    """`baseline_params(params)`, a layer at a time, CONSUMING params' layers: each layer's separate expert
    weights are dropped as its fused ones are made, so the experts are in HBM about once."""
    layers = params.pop("layers")
    converted = []
    while layers:
        converted.append(baseline_params(params | {"layers": [layers.pop(0)]})["layers"][0])
    return params | {"layers": converted}


def baseline_prefill_run(config: ModelConfig, base: dict, run: dict, timed: Callable, small: bool) -> tuple:
    """The baseline's prefill: splash's GQA forms at its default blocks, then the best form over a grid of
    blocks, then (MoE) megablox's tilings for the experts; the default-blocks time, the best time and
    what it was, and the best one's logits."""
    tokens, max_len = run["tokens"], run["max_len"]
    batch, seq = tokens.shape

    def timings(variants: dict[str, dict[str, Any]]) -> dict[str, Callable[[], float]]:
        return {
            label: lambda kw=kwargs: timed(baseline_prefill(config, seq, interpret=small, **kw), tokens, base, max_len)
            for label, kwargs in variants.items()
        }

    print(f"  prefill B={batch} S={seq}: plain JAX + splash")
    forms = {f"splash {form}, default blocks": {"form": form} for form in ("mqa", "mha")}
    t_default, label = best_of(timings(forms))
    best = forms[label]
    t_best = t_default

    def improve(variants: dict[str, dict[str, Any]]) -> None:
        nonlocal t_best, best, label
        t, name = best_of(timings(variants))
        if t < t_best:
            t_best, best, label = t, variants[name], name

    if not small:  # then splash's blocks, then (MoE) megablox's tilings, each from the best so far
        form = best["form"]
        grid = sorted({(min(bq, seq), min(bk, seq)) for bq in SPLASH_BLOCKS for bk in SPLASH_BLOCKS})
        improve({f"splash {form}, blocks {bq} x {bk}": best | {"block_q": bq, "block_kv": bk} for bq, bk in grid})
        if config.moe is not None:
            megablox = {"experts": "megablox"}
            improve(
                {f"{label}, megablox {t}": best | megablox | {"tiling": megablox_tiling(*t)} for t in MEGABLOX_TILINGS}
            )
    logits = baseline_prefill(config, seq, interpret=small, **best)(tokens, base, max_len)[0]
    return t_default, t_best, label, logits


def baseline_decode_run(config: ModelConfig, base: dict, run: dict, steps: int, on_tpu: bool) -> tuple:
    """The baseline's decode steps: XLA attention over the contiguous cache, or ragged_paged_attention over
    pages (on a TPU); MoE experts by ragged_dot or every expert. The best time, what it was, and its first
    step's logits."""
    lengths, max_len, token = run["lengths"], run["max_len"], run["token"]
    batch = len(lengths)
    tables = jnp.asarray(page_tables(batch, max_len, PAGE))
    first: dict[str, jax.Array] = {}

    def contiguous() -> list:
        return random_cache(config, lengths, max_len)

    def paged() -> list:
        pages = [{n: to_pages(c[n], tables) for n in c} for c in contiguous()]
        return [{"kv": ragged_kv_pages(c["k"], c["v"], config.kv_heads)} for c in pages]

    def timing(label: str, attention: str, experts: str) -> Callable[[], float]:
        def run_steps() -> float:
            step = baseline_decode(config, attention, experts)
            cache = contiguous() if attention == "xla" else paged()
            t, first[label] = time_steps(step, token, cache, lengths, base, steps, tables)
            return t

        return run_steps

    candidates = {}
    for experts in ("ragged_dot", "dense") if config.moe is not None else ("ragged_dot",):
        for attention in ("xla", "ragged") if on_tpu else ("xla",):
            label = f"{attention} attention" + (f", {experts} experts" if config.moe is not None else "")
            candidates[label] = timing(label, attention, experts)
    print(f"  decode B={batch}: plain JAX")
    t, label = best_of(candidates)
    return t, label, first.get(label)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=sorted(CONFIGS), help="dense or moe (repeatable)")
    parser.add_argument("--layers", type=int, help="layers per model (default 4)")
    parser.add_argument("--prefill", action="append", help="batch,prompt_length (repeatable)")
    parser.add_argument("--decode", action="append", help="batch,mean_context (repeatable)")
    parser.add_argument("--steps", type=int, default=16, help="decode steps timed per run")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny models, runs anywhere
    args = parser.parse_args()
    on_tpu = jax.default_backend() == "tpu"
    if not args.small:
        require_tpu()
    picked = args.prefill or args.decode
    prefills = args.prefill or ([] if picked else ["1,256"] if args.small else PREFILL)
    decodes = args.decode or ([] if picked else ["2,128"] if args.small else DECODE)
    steps = 2 if args.small else args.steps
    multiple = 128 if args.small else 512

    def timed(fn: Any, *inputs: Any) -> float:
        return time_ms(fn, *inputs, runs=1 if args.small else 5, repeats=1 if args.small else 5)

    summary = []
    for name in args.model or ["dense", "moe"]:
        config = (SMALL if args.small else CONFIGS)[name]
        if args.layers:
            config = dataclasses.replace(config, layers=args.layers)
        print(f"{name}: {config}")
        params = init_params(config)
        ours_p = [ours_prefill(params, *map(int, s.split(",")), timed, on_tpu) for s in prefills]
        ours_d = [ours_decode(params, *map(int, s.split(",")), steps, on_tpu, multiple) for s in decodes]
        base = to_baseline(params)
        del params
        for run in ours_p:
            t_default, t_best, label, logits = baseline_prefill_run(config, base, run, timed, args.small)
            b, s = run["tokens"].shape
            diff, agree = max_diff(run["logits"], logits), rows_agreeing(run["logits"], logits)
            print(
                f"  prefill B={b} S={s}: baseline {t_best:.3f} ms ({label}), autokernel {run['ms']:.3f} ms: "
                f"{t_best / run['ms']:.2f}x; max |logits diff| {diff:.4f}, rows agreeing {agree:.0%}"
            )
            summary.append((name, "prefill", b, s, t_default, t_best, run["ms"], b * s, diff, run["phases"]))
        for run in ours_d:
            t, label, logits = baseline_decode_run(config, base, run, steps, on_tpu)
            b, mean = len(run["lengths"]), run["mean"]
            diff = max_diff(run["logits"], logits) if logits is not None else float("nan")
            print(
                f"  decode B={b}: baseline {t:.3f} ms/step ({label}), autokernel {run['ms']:.3f}: {t / run['ms']:.2f}x"
            )
            summary.append((name, "decode", b, mean, t, t, run["ms"], b, diff, run["phases"]))
        del base
        print()

    print(
        "model  run      B   S/ctx | baseline ms (default)  best      autokernel | speedup | tokens/s: baseline  "
        "autokernel | max|diff| | kernels: attention  ffn/moe  projections"
    )
    for name, kind, b, s, t_default, t_best, ours, tokens, diff, phases in summary:
        ffn = phases.get("ffn", 0.0) + phases.get("moe", 0.0)
        kernels = (
            f"{phases.get('attention', 0.0):9.3f} {ffn:8.3f} {phases.get('projections', 0.0):12.3f}" if phases else ""
        )
        print(
            f"{name:<6} {kind:<8} {b:<3} {s:<5} | {t_default:21.3f} {t_best:9.3f} {ours:10.3f} | "
            f"{t_best / ours:6.2f}x | "
            f"{tokens / t_best * 1e3:18,.0f} {tokens / ours * 1e3:10,.0f} | {diff:9.4f} | {kernels}"
        )


if __name__ == "__main__":
    main()
