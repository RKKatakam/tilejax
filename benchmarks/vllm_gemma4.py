"""vLLM (its TPU backend) on Gemma 4 E2B, text only, on a TPU v5e: output tokens/s for the workload
benchmarks/gemma4.py runs with autokernel's kernels (random 128-token prompts, 256 greedy tokens each,
EOS ignored, every prompt submitted at once), so the two compare directly.

Runs in vLLM's own environment (the `vllm-tpu` package pins its own JAX), e.g. on the VM:

    uv venv /content/vllm --python 3.12 && VIRTUAL_ENV=/content/vllm uv pip install vllm-tpu
    /content/vllm/bin/python benchmarks/vllm_gemma4.py --model /content/gemma-4-E2B
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

BATCHES = (1, 8, 32, 64, 128, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, action="append", help="batch sizes (repeatable)")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    batches = args.batch or list(BATCHES)
    llm = LLM(
        model=args.model,
        max_model_len=args.prompt_len + args.new_tokens + 16,
        max_num_seqs=max(batches),
        limit_mm_per_prompt={"image": 0, "audio": 0},  # the text model only
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.new_tokens, ignore_eos=True)
    rng = np.random.default_rng(0)
    rows = []
    for batch in batches:
        prompts = [
            TokensPrompt(prompt_token_ids=rng.integers(1000, 200_000, args.prompt_len).tolist()) for _ in range(batch)
        ]
        llm.generate(prompts, params, use_tqdm=False)  # warm-up: compiles this batch's shapes
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        row = {
            "engine": "vllm",
            "batch": batch,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "seconds": round(seconds, 4),
            "output_tokens_per_s": round(generated / seconds, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    print("\nvLLM, Gemma 4 E2B (text), TPU v5e: batch  seconds  output tokens/s")
    for row in rows:
        print(f"  {row['batch']:>5}  {row['seconds']:8.3f}  {row['output_tokens_per_s']:10.1f}")


if __name__ == "__main__":
    main()
