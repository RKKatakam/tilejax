"""Where the Gemma 4 E2B decode megakernel's time goes on a TPU v5e, per batch size: its trace regions after
a 128-token prefill of random prompts (real weights), at the default tiles.

    python benchmarks/gemma4_profile.py --model /content/gemma-4-E2B --batch 1 --batch 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autokernel  # turns trace regions on: JAX's TPU backend hasn't started yet

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
import gemma4_megakernel as mk
from gemma4 import I32, load, prefill
from gemma4_kernels import MegaTiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/content/gemma-4-E2B")
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--prompt-len", type=int, default=128)
    args = parser.parse_args()
    params = load(args.model)
    rng = np.random.default_rng(0)
    for batch in args.batch or [1, 8, 32]:
        prompts = jnp.asarray(rng.integers(1000, 200_000, (batch, args.prompt_len)), I32)
        lengths = jnp.full((batch,), args.prompt_len, I32)
        logits, caches, lens = jax.jit(prefill, static_argnums=3)(params, prompts, lengths, 512)
        first = jnp.argmax(logits, axis=-1).astype(I32)
        inputs = mk.kernel_inputs(params, first, mk.stack_caches(params["config"], caches), lens)
        lowered = mk.gemma4_megakernel.lower(*inputs, MegaTiles())
        print(f"== batch {batch}: at most {lowered.steps} steps, {lowered.vmem_bytes / 2**20:.1f} MiB of VMEM")
        print(autokernel.profile(mk.gemma4_megakernel, *inputs, MegaTiles()), flush=True)


if __name__ == "__main__":
    main()
