"""DeepSeek-V3-like and Kimi-K2-like inference on a TPU v5e (examples/deepseek.py), bf16: autokernel's prefill
and decode steps, every kernel tuned on the first (eager) call, against the same models in plain JAX.

- Prefill (batch 1-2, prompts 2048-4096): `prefill` (MLA in the expanded form, the latent cache written) against
  the plain-JAX model with XLA's attention (when its (B, H, S, S) scores fit) or JAX's splash attention (q/k
  head dim 192; default, 512 and 1024 blocks), MoE by XLA's ragged_dot: the best of them. ms and tokens/s.
- Decode (batch 1, 8, 32; every sequence at the same context, 1k-4k): greedy steps chained under jit with the
  latent cache donated, in both absorbed forms ("folded": W_uk and W_uv folded into the projections, one
  matmul each; "factored": applied head by head, fewer weight bytes), against the plain-JAX step (absorbed
  attention over the same latent cache by XLA; MoE by ragged_dot, or every expert on every token): the best
  of them. ms per step and tokens/s.
- The KV cache per token: the latent cache (as the math needs it and as stored) against multi-head attention's.

    uv run python benchmarks/deepseek.py
    uv run python benchmarks/deepseek.py --model kimi-k2 --prefill 1,4096 --decode 8,2048
    uv run python benchmarks/deepseek.py --small      # a tiny model, anywhere (the Pallas interpreter off a TPU)

The models' weights take 3-4.5 GB (the experts' 2.4-3.6); the baselines' layout of the experts' weights is made
after autokernel's runs, a layer at a time, replacing autokernel's, so a run stays under ~5.5 GB of HBM.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from deepseek import (
    CONFIGS,
    Config,
    baseline_decode_step,
    baseline_params,
    baseline_prefill,
    decode_step,
    init_params,
    kv_bytes_per_token,
    pack,
    prefill,
)
from timing import require_tpu, time_ms

PREFILL = ["1,4096", "2,2048", "2,4096"]  # batch, prompt length
DECODE = ["1,1024", "1,4096", "8,2048", "32,1024", "32,4096"]  # batch, context
FORMS = ("folded", "factored")
XLA_SCORES_LIMIT = 2**30  # bytes of f32 (B, H, S, S) scores the XLA attention baseline may materialize
SPLASH_BLOCKS = (None, 512, 1024)  # splash's default blocks, then square blocks of these sizes
SMALL = Config(2, 256, 4, 256, 512, 8, 2, 128, 512, 1024, n_group=2, topk_group=1)


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def weight_bytes(tree: Any) -> int:
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(tree))


def random_cache(config: Config, batch: int, context: int, max_len: int, seed: int = 0) -> list[dict[str, jax.Array]]:
    """A latent cache (as `prefill` writes it) of random latents and rotary keys, every sequence `context`
    positions long (zeros after): a stand-in for a prefilled cache."""
    valid = (jnp.arange(max_len) < context)[None, :, None]
    keys = jax.random.split(jax.random.key(seed), 2 * config.layers)
    cache = []
    for i in range(config.layers):
        c_kv = jax.random.normal(keys[2 * i], (batch, max_len, config.kv_rank), jnp.bfloat16)
        k_rope = jax.random.normal(keys[2 * i + 1], (batch, max_len, 64), jnp.bfloat16)
        cache.append({"c_kv": jnp.where(valid, c_kv, 0), "k_rope": jnp.where(valid, jnp.tile(k_rope, (1, 1, 2)), 0)})
    return cache


def to_baseline(params: dict[str, Any]) -> dict[str, Any]:
    """`baseline_params`, made a layer at a time, each MoE layer's separate gate and up weights dropped (from
    `params` too) once they are side by side: the copy never holds more than a layer's experts twice."""
    layers = []
    for layer in params["layers"]:
        plain = baseline_params(params | {"layers": [layer]})["layers"][0]
        if "moe" in layer:
            plain["moe"] = {k: v for k, v in plain["moe"].items() if k not in ("w_gate", "w_up")}
            del layer["moe"]["w_gate"], layer["moe"]["w_up"]
        layers.append(plain)
    return params | {"layers": layers}


def time_steps(step: Callable, tokens: jax.Array, cache: Any, lengths: jax.Array, params: Any, n: int):
    """Wall-clock ms per greedy decode step over n chained steps (the cache donated), after 2 warm-up steps;
    and the first step's logits."""
    first = None
    for i in range(n + 2):
        if i == 2:
            jax.block_until_ready(tokens)
            start = time.perf_counter()
        logits, cache, lengths = step(tokens, cache, lengths, params)
        first = logits if first is None else first
        tokens = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)[:, None]
    jax.block_until_ready(tokens)
    return (time.perf_counter() - start) / n * 1e3, first


def best_of(candidates: dict[str, Callable[[], tuple[float, Any]]]) -> tuple[float, str, Any]:
    """The fastest of named timings, each returning (ms, output); those that fail are reported and skipped."""
    best: tuple[float, str, Any] = (float("inf"), "-", None)
    for name, run in candidates.items():
        try:
            t, out = run()
        except Exception as error:  # a baseline variant that doesn't compile or fit
            print(f"    {name}: failed ({type(error).__name__}: {str(error).splitlines()[0][:100]})")
            continue
        print(f"    {name:<44} {t:9.3f} ms")
        if t < best[0]:
            best = (t, name, out)
    return best


def run_model(name: str, config: Config, prefills: list[str], decodes: list[str], steps: int, small: bool) -> list:
    timing = {"runs": 1, "repeats": 1} if small else {}
    interpret = jax.default_backend() != "tpu"
    params = init_params(config)
    packs = {form: pack(params, decode=form) for form in FORMS}
    print(f"\n=== {name}: {config}")
    print(f"  weights {weight_bytes(params) / 1e9:.2f} GB; KV cache per token (all layers):")
    for kind, n in kv_bytes_per_token(config).items():
        print(f"    {kind:<16} {n:7d} bytes")
    prefill_runs = [tuple(map(int, run.split(","))) for run in prefills]
    decode_runs = [tuple(map(int, run.split(","))) for run in decodes]
    ours: dict[tuple, dict[str, Any]] = {}

    # autokernel: a first eager call tunes every kernel on real routings and lengths, then jitted calls.
    for batch, seq in prefill_runs:
        tokens = jax.random.randint(jax.random.key(seq), (batch, seq), 0, config.vocab)
        prefill(tokens, packs["folded"], max_len=seq)
        fn = jax.jit(functools.partial(prefill, max_len=seq))
        t = time_ms(fn, tokens, packs["folded"], **timing)
        ours[("prefill", batch, seq)] = {"ms": t, "logits": fn(tokens, packs["folded"])[0]}
        print(f"  prefill B={batch} S={seq}: autokernel {t:.3f} ms ({batch * seq / t * 1e3:,.0f} tokens/s)")
    for batch, context in decode_runs:
        max_len = -(-(context + steps + 2) // 512) * 512
        lengths = jnp.full((batch,), context, jnp.int32)
        tokens = jax.random.randint(jax.random.key(context), (batch, 1), 0, config.vocab)
        result = {}
        for form in FORMS:
            decode_step(tokens, random_cache(config, batch, context, max_len), lengths, packs[form])  # tunes
            step = jax.jit(decode_step, donate_argnums=1)
            cache = random_cache(config, batch, context, max_len)
            result[form] = time_steps(step, tokens, cache, lengths, packs[form], steps)
            t = result[form][0]
            print(f"  decode B={batch} context {context} ({form}): {t:.3f} ms/step ({batch / t * 1e3:,.0f} tokens/s)")
        ours[("decode", batch, context)] = result
    del packs

    # The baselines, on their own layout of the weights: autokernel's copies are gone.
    plain = to_baseline(params)
    del params
    rows = []
    for batch, seq in prefill_runs:
        tokens = jax.random.randint(jax.random.key(seq), (batch, seq), 0, config.vocab)
        print(f"  prefill B={batch} S={seq}, baselines:")
        candidates = {}
        if batch * config.heads * seq * seq * 4 <= XLA_SCORES_LIMIT:
            candidates["plain JAX, XLA attention"] = baseline_prefill()
        for block in SPLASH_BLOCKS:
            if block is None or block <= seq:
                label = "default" if block is None else f"{block} x {block}"
                candidates[f"plain JAX, splash {label}"] = baseline_prefill("splash", block, block, interpret=interpret)

        def timed(fn: Callable, tokens=tokens, seq=seq) -> tuple[float, Any]:
            run = functools.partial(fn, max_len=seq)
            return time_ms(run, tokens, plain, **timing), run(tokens, plain)[0]

        t_base, base_name, logits = best_of({k: functools.partial(timed, fn) for k, fn in candidates.items()})
        mine = ours[("prefill", batch, seq)]
        error = max_diff(mine["logits"], logits)
        speedup = t_base / mine["ms"]
        print(f"  prefill B={batch} S={seq}: speedup {speedup:.2f}x over {base_name}; max |logits diff| {error:.4f}")
        rows.append((name, "prefill", batch, seq, t_base, base_name, mine["ms"], None, batch * seq, error))
    for batch, context in decode_runs:
        max_len = -(-(context + steps + 2) // 512) * 512
        lengths = jnp.full((batch,), context, jnp.int32)
        tokens = jax.random.randint(jax.random.key(context), (batch, 1), 0, config.vocab)
        print(f"  decode B={batch} context {context}, baselines:")

        def stepped(method: str, tokens=tokens, cache=(batch, context, max_len), lengths=lengths) -> tuple[float, Any]:
            return time_steps(baseline_decode_step(method), tokens, random_cache(config, *cache), lengths, plain, steps)

        methods = ("ragged_dot", "dense")
        candidates = {f"plain JAX, XLA attention, MoE {m}": functools.partial(stepped, m) for m in methods}
        t_base, base_name, logits = best_of(candidates)
        mine = ours[("decode", batch, context)]
        error = max(max_diff(mine[form][1], logits) for form in FORMS)
        t_folded, t_factored = mine["folded"][0], mine["factored"][0]
        speedup = t_base / min(t_folded, t_factored)
        print(f"  decode B={batch} context {context}: speedup {speedup:.2f}x over {base_name}; max diff {error:.4f}")
        rows.append((name, "decode", batch, context, t_base, base_name, t_folded, t_factored, batch, error))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=sorted(CONFIGS), help="(repeatable; default: all)")
    parser.add_argument("--prefill", action="append", help="batch,prompt_length (repeatable)")
    parser.add_argument("--decode", action="append", help="batch,context (repeatable)")
    parser.add_argument("--steps", type=int, default=16, help="decode steps timed per run")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # a tiny model, runs anywhere
    args = parser.parse_args()
    if not args.small:
        require_tpu()
    picked = args.prefill or args.decode
    prefills = args.prefill or ([] if picked else ["1,256"] if args.small else PREFILL)
    decodes = args.decode or ([] if picked else ["2,128"] if args.small else DECODE)
    models = {"small": SMALL} if args.small else {name: CONFIGS[name] for name in args.model or sorted(CONFIGS)}
    steps = 2 if args.small else args.steps
    rows = []
    for name, config in models.items():
        rows += run_model(name, config, prefills, decodes, steps, args.small)

    print("\nms (prefill: per call; decode: per step); tokens/s for autokernel's faster form")
    print(
        f"{'model':<12} {'phase':<8} {'B':>3} {'length':>6}  {'baseline':>9} {'(best of)':<44}"
        f" {'autokernel':>10} {'factored':>9} {'speedup':>8} {'tokens/s':>10} {'max diff':>9}"
    )
    for name, phase, b, n, t_base, base_name, t_ours, t_factored, tokens, error in rows:
        best = min(t_ours, t_factored or t_ours)
        factored = f"{t_factored:9.3f}" if t_factored is not None else f"{'':9}"
        print(
            f"{name:<12} {phase:<8} {b:>3} {n:>6}  {t_base:9.3f} {base_name:<44} {t_ours:10.3f} {factored}"
            f" {t_base / best:7.2f}x {tokens / best * 1e3:10,.0f} {error:9.4f}"
        )
    print("(decode: 'autokernel' is the folded form; prefill's length is the prompt's, decode's the context's)")


if __name__ == "__main__":
    main()
