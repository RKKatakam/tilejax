"""Check examples/gemma4.py against Hugging Face transformers on a tiny random Gemma 4 text model, on the CPU:
build transformers' Gemma4ForCausalLM (5 layers: sliding and full attention, two KV-shared layers with
double-wide MLPs, proportional RoPE, per-layer embeddings, a 128-key window), give it random bf16 weights,
write them as a checkpoint laid out like google/gemma-4-E2B's (config.json with a text_config,
model.safetensors with model.language_model.* names), read that with our loader, and compare logits on
200-token sequences (the window is exercised): our kernels in f32 and bf16 and the plain-JAX reference
against transformers in f32 (eager attention).

Needs torch, transformers (>= 5.5, with Gemma 4) and safetensors besides the repo's dependencies:

    python -P benchmarks/gemma4_hf_tiny.py      # -P: benchmarks/profile.py would shadow the stdlib's profile
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gemma4 import forward, generate, load, reference_forward

TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"]
TEXT_CONFIG = {
    "hidden_size": 256,
    "num_hidden_layers": 5,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "global_head_dim": 256,
    "intermediate_size": 512,
    "vocab_size": 512,
    "vocab_size_per_layer_input": 512,
    "hidden_size_per_layer_input": 128,
    "num_kv_shared_layers": 2,
    "layer_types": TYPES,
    "use_double_wide_mlp": True,
    "sliding_window": 128,
    "final_logit_softcapping": 30.0,
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
    },
    "tie_word_embeddings": True,
}


def hf_checkpoint(directory: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    """Write a random tiny checkpoint to `directory`; the transformers model, random ids and its f32 logits."""
    import torch
    from safetensors.torch import save_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(**TEXT_CONFIG)
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config).float().eval()
    generator = torch.Generator().manual_seed(0)
    state = {}
    for name, t in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        if "norm" in name or "layer_scalar" in name:
            value = 1 + 0.1 * torch.randn(t.shape, generator=generator)
            if "q_norm" in name or "k_norm" in name:  # unit-size attention scores (the scale is 1)
                value = value * t.shape[-1] ** -0.25
            if "layer_scalar" in name:
                value = 0.5 + 0.5 * torch.rand(t.shape, generator=generator)
        else:
            scale = t.shape[-1] ** -0.5 if t.ndim == 2 and "embed" not in name else 1.0
            value = torch.randn(t.shape, generator=generator) * scale
        state[name] = value.to(torch.bfloat16)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    model.tie_weights()
    ids = torch.randint(0, TEXT_CONFIG["vocab_size"], (2, 200), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float().numpy()
    renamed = {"model.language_model." + k[len("model.") :]: v.contiguous() for k, v in state.items()}
    save_file(renamed, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": TEXT_CONFIG}))
    return model, ids.numpy(), logits


def relative(a: np.ndarray, b: np.ndarray) -> str:
    rms = float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b**2)))
    return f"relative RMS {rms:.2e}, max abs {float(np.abs(a - b).max()):.2e}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model, ids, hf = hf_checkpoint(Path(tmp))
        params = load(tmp)
    ids = jnp.asarray(ids, jnp.int32)
    f32 = jax.tree.map(lambda a: a.astype(jnp.float32), params)
    rows = {
        "ours (kernels), f32": np.asarray(jax.jit(forward)(f32, ids), np.float32),
        "ours (kernels), bf16": np.asarray(jax.jit(forward)(params, ids).astype(jnp.float32)),
        "reference (plain JAX), bf16": np.asarray(jax.jit(reference_forward)(params, ids)),
    }
    print(f"transformers Gemma4ForCausalLM, f32, logits up to {np.abs(hf).max():.1f}:")
    for name, logits in rows.items():
        agree = float(np.mean(np.argmax(logits, -1) == np.argmax(hf, -1)))
        print(f"  {name:28s} {relative(logits, hf)}; argmax agrees at {agree:.1%} of positions")
    import torch

    prompt = np.asarray(ids[:, :150])
    with torch.no_grad():
        theirs = model.generate(torch.from_numpy(prompt.astype(np.int64)), max_new_tokens=8, do_sample=False)
    theirs = theirs[:, 150:].numpy()
    for mega in (True, False):
        ours, _ = generate(params, jnp.asarray(prompt), 8, mega=mega)
        path = "one-kernel decode" if mega else "per-layer decode"
        print(
            f"  greedy, 8 tokens, {path}: ours {np.asarray(ours).tolist()}, the same as transformers':"
            f" {bool((np.asarray(ours) == theirs).all())}"
        )


if __name__ == "__main__":
    main()
