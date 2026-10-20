"""Check Gemma 4 E2B (text) on autokernel's kernels against the plain-JAX reference and Hugging Face transformers,
on the real checkpoint: a real prompt through our prefill and 32 greedy decode steps; the same tokens through
`reference_forward` (and its own greedy decoding); and, if torch and transformers import, through
transformers' Gemma4ForCausalLM in float32 on the CPU (the text weights only, read with our safetensors
reader). Prints the largest logit differences, whether the greedy tokens agree, and the decoded texts.

    python benchmarks/gemma4_verify.py                       # /content/gemma-4-E2B, downloaded if missing
    python benchmarks/gemma4_verify.py --model DIR --no-hf   # skip transformers (it needs ~20 GB of RAM)
    python benchmarks/gemma4_verify.py --model DIR --ids 2,818,5279   # token ids instead of a tokenizer
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

# benchmarks/profile.py would shadow the standard library's `profile`, which torch imports.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gemma4 import (
    Generator,
    decode_logits,
    load,
    prefill,
    read_safetensors,
    reference_forward,
)

PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]


def largest(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def ours_logits(params, ids: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Our logits (1 + len(tokens) - 1, V) predicting each generated token: prefill's at the prompt's last
    position, then decode steps fed our own tokens."""
    n = len(ids)
    max_len = -(-(n + len(tokens)) // 128) * 128
    logits, caches, lengths = jax.jit(prefill, static_argnums=3)(
        params, jnp.asarray(ids[None]), jnp.array([n]), max_len
    )
    out = [np.asarray(logits[0], np.float32)]
    step = jax.jit(decode_logits)
    for token in tokens[:-1]:
        logits, caches, lengths = step(params, jnp.array([token], jnp.int32), caches, lengths)
        out.append(np.asarray(logits[0], np.float32))
    return np.stack(out)


def reference_greedy(params, ids: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy tokens from `reference_forward` (on a fixed padded buffer: causal, so padding after the
    sequence changes nothing) and its logits at each step."""
    n = len(ids)
    total = -(-(n + steps) // 128) * 128
    buffer = np.zeros((1, total), np.int32)
    buffer[0, :n] = ids
    forward = jax.jit(reference_forward)
    tokens, logits = [], []
    for t in range(steps):
        row = np.asarray(forward(params, jnp.asarray(buffer))[0, n + t - 1])
        logits.append(row)
        tokens.append(int(np.argmax(row)))
        buffer[0, n + t] = tokens[-1]
    return np.array(tokens), np.stack(logits)


def hf_model(directory: Path):
    """transformers' Gemma4ForCausalLM for the checkpoint's text config, float32, our reader's weights."""
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    text = json.loads((directory / "config.json").read_text())["text_config"]
    config = Gemma4TextConfig(**text)
    config._attn_implementation = "eager"
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # older transformers
        from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        model = Gemma4ForCausalLM(config).float().eval()
    tensors = read_safetensors(directory / "model.safetensors", "model.language_model.")
    state = model.state_dict()
    for name in state:
        if name == "lm_head.weight":
            continue
        source = tensors[name[len("model.") :]]
        with torch.no_grad():
            state[name].copy_(torch.from_numpy(np.asarray(source, np.float32)))
    model.tie_weights()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/content/gemma-4-E2B")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--ids", help="comma-separated prompt token ids (instead of the tokenizer)")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--no-hf", action="store_true")
    args = parser.parse_args()
    directory = Path(args.model)
    if not (directory / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download("google/gemma-4-E2B", allow_patterns=FILES, local_dir=directory)
    tokenizer = None
    if args.ids:
        ids = np.array([int(i) for i in args.ids.split(",")], np.int32)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        ids = np.array(tokenizer.encode(args.prompt).ids, np.int32)
        if ids[0] != 2:  # Gemma's <bos>
            ids = np.concatenate([[2], ids]).astype(np.int32)
    print(f"prompt: {len(ids)} tokens {ids.tolist()}", flush=True)

    def text(tokens: np.ndarray) -> str:
        return tokenizer.decode(tokens.tolist()) if tokenizer else str(tokens.tolist())

    start = time.perf_counter()
    params = load(directory)
    print(f"loaded in {time.perf_counter() - start:.1f} s: {params['config']}", flush=True)
    tokens, timings = Generator(params, 1, len(ids), args.steps)(jnp.asarray(ids[None]))
    tokens = np.asarray(tokens[0])
    print(f"ours:      {tokens.tolist()}\n  {text(tokens)!r}\n  {timings}", flush=True)
    ours = ours_logits(params, ids, tokens)
    agree = np.argmax(ours, -1) == tokens
    print(f"our step logits' argmax = our greedy tokens: {agree.sum()}/{len(tokens)}", flush=True)

    sequence = np.concatenate([ids, tokens]).astype(np.int32)
    total = -(-len(sequence) // 128) * 128
    padded = np.zeros((1, total), np.int32)
    padded[0, : len(sequence)] = sequence
    reference = np.asarray(jax.jit(reference_forward)(params, jnp.asarray(padded))[0])
    reference = reference[len(ids) - 1 : len(sequence) - 1]
    print(
        f"reference (plain JAX, bf16), teacher-forced on our tokens: max |logits - ours| {largest(reference, ours):.3f}"
        f" (logits up to {np.abs(reference).max():.1f}); greedy agrees at"
        f" {(np.argmax(reference, -1) == tokens).sum()}/{len(tokens)} positions",
        flush=True,
    )
    ref_tokens, _ = reference_greedy(params, ids, args.steps)
    print(f"reference greedy: {ref_tokens.tolist()}\n  {text(ref_tokens)!r}", flush=True)
    print(f"  same tokens as ours: {bool((ref_tokens == tokens).all())}", flush=True)

    if args.no_hf:
        return
    try:
        import torch
    except ImportError:
        print("transformers / torch not importable: skipping Hugging Face", flush=True)
        return
    del params
    start = time.perf_counter()
    model = hf_model(directory)
    print(f"transformers model (f32, CPU) in {time.perf_counter() - start:.1f} s", flush=True)
    with torch.no_grad():
        hf = model(input_ids=torch.from_numpy(sequence[None].astype(np.int64))).logits[0].float().numpy()
    hf = hf[len(ids) - 1 : len(sequence) - 1]
    print(
        f"transformers f32, teacher-forced: max |logits - ours| {largest(hf, ours):.3f},"
        f" max |logits - reference| {largest(hf, reference):.3f} (logits up to {np.abs(hf).max():.1f});"
        f" greedy agrees with ours at {(np.argmax(hf, -1) == tokens).sum()}/{len(tokens)} positions",
        flush=True,
    )
    with torch.no_grad():
        out = model.generate(torch.from_numpy(ids[None].astype(np.int64)), max_new_tokens=args.steps, do_sample=False)
    hf_tokens = out[0, len(ids) :].numpy()
    print(f"transformers greedy: {hf_tokens.tolist()}\n  {text(hf_tokens)!r}", flush=True)
    same = min(len(hf_tokens), len(tokens))
    first_diff = next((i for i in range(same) if hf_tokens[i] != tokens[i]), None)
    print(f"  same tokens as ours: {first_diff is None} (first difference at {first_diff})", flush=True)


if __name__ == "__main__":
    main()
