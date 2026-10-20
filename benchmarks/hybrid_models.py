"""Hybrid linear/full-attention models on a TPU v5e (examples/hybrid_models.py): a Qwen3-Next-like model (Gated
DeltaNet + gated GQA, 3:1, MoE with a gated shared expert) and a Kimi-Linear-like one (KDA + MLA, 3:1, MoE with
a sigmoid router), d_model 2048, 8 layers, 64 experts, ~4 GB of bf16 weights each. autokernel's kernels, tuned
(each on its first call), against the same models in plain JAX:

1. Prefill (tokens to the last token's logits and the decode state), batch 1-2, prompts of 4096-8192 tokens:
   the chunked delta rule in plain JAX (flash-linear-attention's structure) and JAX's splash attention,
   XLA's matmuls and `jax.lax.ragged_dot` experts. Tokens per second.
2. Decode (one greedy step per sequence, steps chained with the state donated), batch 1-32, contexts 1k-8k
   (each sequence's length uniform in [3/4, 1] of it; a random stand-in for the prefilled state): the
   recurrence in plain JAX either way (the per-token state update is bandwidth-bound elementwise work),
   XLA's attention over the whole caches vs `gqa_decode` / `mla_decode` (only the valid blocks), XLA vs
   autokernel's norm + projection and `expert_ffn` (only the chosen experts' weights) kernels. Tokens/s.

Each line reports the max |difference| of the logits from the baseline's (bf16: a few hundredths), and for
prefill the largest difference of the linear layers' states relative to their largest entry.

    uv run python benchmarks/hybrid_models.py
    uv run python benchmarks/hybrid_models.py --model qwen3-next-small --prefill 1,8192 --decode 32,4096

(Tuning compiles each kernel's loop nest as plain JAX once per shape: expect several minutes of tuning on the
first call of each model and shape.)
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
from hybrid_layers import HybridConfig
from hybrid_models import (
    CONFIGS,
    baseline_decode_step,
    baseline_prefill,
    decode_step,
    init_params,
    prefill,
    random_state,
    round_up,
    tiny_config,
)
from timing import require_tpu, time_ms

PREFILL = ["1,4096", "1,8192", "2,4096"]  # batch, prompt length
DECODE = ["1,1024", "1,8192", "8,4096", "32,1024", "32,4096"]  # batch, context
# --small: tiny models (top-2 of 4 experts), any backend: a smoke test of this script (the kernels in the
# interpreter off a TPU; routing near-ties between bf16 models can make their outputs differ).
SMALL_CONFIGS = {f"{kind}-small": tiny_config(kind) for kind in ("qwen3-next", "kimi-linear")}
SMALL_PREFILL, SMALL_DECODE = ["1,256"], ["2,256"]


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def state_diff(state: list[dict], want: list[dict]) -> float:
    """The largest |difference| of the linear layers' recurrent states, relative to their largest entry."""
    diffs = [
        max_diff(got["state"], base["state"]) / float(jnp.max(jnp.abs(base["state"])))
        for got, base in zip(state, want, strict=True)
        if "state" in got
    ]
    return max(diffs)


def bench_prefill(name: str, params: dict[str, Any], shape: str, small: bool) -> tuple:
    batch, seq = map(int, shape.split(","))
    config: HybridConfig = params["config"]
    tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, config.vocab)
    ours = jax.jit(prefill)
    base = baseline_prefill(attention="xla" if small else "splash", interpret=small)
    runs = {"runs": 1, "repeats": 1} if small else {"runs": 3, "repeats": 3}
    print(f"{name} prefill: batch {batch}, prompt {seq}")
    t_base = time_ms(base, tokens, params, **runs)
    t_ours = time_ms(ours, tokens, params, **runs)  # tunes each kernel on the first call
    logits, state = ours(tokens, params)
    want, want_state = base(tokens, params)
    diff, sdiff = max_diff(logits, want), state_diff(state, want_state)
    rate = lambda ms: batch * seq / ms * 1e3  # noqa: E731
    print(f"  plain JAX (chunked delta rule, splash)  {t_base:9.3f} ms  {rate(t_base):9.0f} tokens/s")
    print(f"  autokernel, tuned                       {t_ours:9.3f} ms  {rate(t_ours):9.0f} tokens/s")
    print(f"  speedup {t_base / t_ours:.2f}x; max |logits - plain JAX| {diff:.4f}, states {sdiff:.4f} (relative)")
    return ("prefill", name, batch, seq, t_base, t_ours, diff)


def time_steps(step: Callable, token: jax.Array, state: Any, lengths: jax.Array, params: Any, n: int):
    """Wall-clock ms per greedy decode step over n chained steps (state donated), after 2 warm-up steps;
    and the first step's logits."""
    first = None
    for i in range(n + 2):
        if i == 2:
            jax.block_until_ready(token)
            start = time.perf_counter()
        logits, state, lengths = step(token, state, lengths, params)
        first = logits if first is None else first
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    jax.block_until_ready(token)
    return (time.perf_counter() - start) / n * 1e3, first


def bench_decode(name: str, params: dict[str, Any], shape: str, steps: int, on_tpu: bool) -> tuple:
    batch, context = map(int, shape.split(","))
    config: HybridConfig = params["config"]
    lengths = np.random.default_rng(0).integers(3 * context // 4, context + 1, batch).astype(np.int32)
    max_len = round_up(context + steps + 2)
    token = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
    print(f"{name} decode: batch {batch}, context {context} (lengths {lengths.min()}-{lengths.max()})")

    if on_tpu:  # one step eagerly: the attention and expert kernels tune on real lengths and routing
        decode_step(token, random_state(config, lengths, max_len), jnp.asarray(lengths), params)
    ours = jax.jit(decode_step, donate_argnums=1)
    t_ours, first = time_steps(ours, token, random_state(config, lengths, max_len), jnp.asarray(lengths), params, steps)
    t_base, want = time_steps(
        baseline_decode_step(), token, random_state(config, lengths, max_len), jnp.asarray(lengths), params, steps
    )
    diff = max_diff(first, want)
    rate = lambda ms: batch / ms * 1e3  # noqa: E731
    print(f"  plain JAX (XLA attention, ragged_dot)   {t_base:9.3f} ms  {rate(t_base):9.0f} tokens/s")
    print(f"  autokernel, tuned                       {t_ours:9.3f} ms  {rate(t_ours):9.0f} tokens/s")
    print(f"  speedup {t_base / t_ours:.2f}x; max |logits - plain JAX| {diff:.4f}")
    return ("decode", name, batch, context, t_base, t_ours, diff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--prefill", action="append", help="batch,prompt length (repeatable)")
    parser.add_argument("--decode", action="append", help="batch,context (repeatable)")
    parser.add_argument("--steps", type=int, default=16, help="decode steps timed")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny models, any backend
    args = parser.parse_args()
    if not args.small:
        require_tpu()
    on_tpu = jax.default_backend() == "tpu"
    configs = SMALL_CONFIGS if args.small else CONFIGS
    names = list(configs) if args.model == "all" else [args.model]
    prefills = args.prefill or (SMALL_PREFILL if args.small else PREFILL)
    decodes = args.decode or (SMALL_DECODE if args.small else DECODE)
    steps = 2 if args.small else args.steps
    rows = []
    for name in names:
        params = init_params(configs[name])
        rows += [bench_prefill(name, params, shape, args.small) for shape in prefills]
        rows += [bench_decode(name, params, shape, steps, on_tpu) for shape in decodes]
        del params
    print("\n         model               batch  tokens  plain JAX ms  autokernel ms  speedup  ours tokens/s  max diff")
    for kind, name, batch, n, t_base, t_ours, diff in rows:
        tokens = batch * n if kind == "prefill" else batch
        print(
            f"{kind:<8} {name:<19} {batch:5} {n:7} {t_base:14.3f} {t_ours:14.3f} {t_base / t_ours:7.2f}x"
            f" {tokens / t_ours * 1e3:16.0f} {diff:9.4f}"
        )


if __name__ == "__main__":
    main()
