"""Gemma 4 E2B (text) greedy generation with autokernel's kernels on a TPU v5e: output tokens/s on the workload
benchmarks/vllm_gemma4.py runs with vLLM (random 128-token prompts drawn from one numpy default_rng(0) stream,
batch after batch; 256 greedy tokens each, EOS ignored; wall time from the prompts on the host to every
token on the host, prefill included, after a warm-up run that compiles and tunes this batch's shapes).

    python benchmarks/gemma4.py --model /content/gemma-4-E2B            # the real weights
    python benchmarks/gemma4.py --batch 1 --batch 8 --no-tune           # default tiles, no tuning
    python benchmarks/gemma4.py --small                                  # a tiny random model (CPU smoke test)

Prints a JSON line per batch ("engine": "autokernel", batch, prompt_len, new_tokens, seconds,
output_tokens_per_s, plus prefill_ms and decode_tokens_per_s) and a summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gemma4 import TINY, Generator, Tiles, init_params, load
from gemma4_kernels import AttentionTiles, HeadTiles, PostTiles, QkvTiles

BATCHES = (1, 8, 32, 64, 128, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/content/gemma-4-E2B")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, action="append", help="batch sizes (repeatable)")
    parser.add_argument("--no-tune", action="store_true", help="the kernels' default tiles instead of tuning")
    parser.add_argument("--small", action="store_true", help="a tiny random model, few tokens (smoke test)")
    args = parser.parse_args()
    if args.small:
        params, vocab = init_params(TINY), TINY.vocab
        args.prompt_len, args.new_tokens = 64, 4
        batches = args.batch or [1, 2]
    else:
        start = time.perf_counter()
        params = load(args.model)
        vocab = params["config"].vocab
        print(f"loaded {args.model} in {time.perf_counter() - start:.1f} s", flush=True)
        batches = args.batch or list(BATCHES)
    tiles = Tiles()
    if args.no_tune:
        tiles = Tiles(qkv=QkvTiles(), attention=AttentionTiles(), post=PostTiles(), head=HeadTiles())
    rng = np.random.default_rng(0)
    rows = []
    for batch in batches:
        high = min(200_000, vocab)
        prompts = np.stack([rng.integers(min(1000, high - 1), high, args.prompt_len) for _ in range(batch)])
        prompts = prompts.astype(np.int32)
        generator = Generator(params, batch, args.prompt_len, args.new_tokens, tiles)
        start = time.perf_counter()
        generator(jnp.asarray(prompts))  # warm-up: tunes (on a TPU) and compiles this batch's shapes
        warmup = time.perf_counter() - start
        start = time.perf_counter()
        tokens, timings = generator(jnp.asarray(prompts))
        tokens = np.asarray(tokens)
        seconds = time.perf_counter() - start
        row = {
            "engine": "autokernel",
            "batch": batch,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "seconds": round(seconds, 4),
            "output_tokens_per_s": round(tokens.size / seconds, 1),
            "prefill_ms": round(timings.prefill_ms, 2),
            "decode_ms_per_step": round(timings.decode_ms / max(args.new_tokens - 1, 1), 3),
            "decode_tokens_per_s": round(timings.decode_tokens_per_s, 1),
            "warmup_s": round(warmup, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(f"\nautokernel, Gemma 4 E2B (text), {jax.devices()[0].device_kind}")
    print("  batch   seconds  output tok/s  prefill ms  decode ms/step")
    for row in rows:
        print(
            f"  {row['batch']:>5}  {row['seconds']:8.3f}  {row['output_tokens_per_s']:12.1f}"
            f"  {row['prefill_ms']:10.2f}  {row['decode_ms_per_step']:14.3f}"
        )


if __name__ == "__main__":
    main()
