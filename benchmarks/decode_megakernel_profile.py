"""Where the decode megakernel's time goes on a TPU v5e (examples/decode_megakernel.py): its trace regions at
its tuned schedule on each shape, against the step's HBM floor (its weights and valid K/V at 819 GB/s).

    python benchmarks/decode_megakernel_profile.py [--shape 16,2048,16,8,6144,32768:1,1024 ...]
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
from decode_megakernel import HEAD_DIM, STACKED, Config, decode_megakernel, init_params, kernel_inputs, random_caches

PEAK_GBS = 819.0
SHAPES = [  # layers, d_model, heads, kv_heads, ffn, vocab : batch, mean context (benchmarks/decode_megakernel.py's)
    "16,2048,16,8,6144,32768:1,1024",
    "16,2048,16,8,6144,32768:8,512",
    "16,2048,16,8,6144,32768:8,2048",
    "8,2048,16,8,6144,32768:8,4096",
    "8,2048,16,8,6144,32768:32,1024",
    "8,2048,16,8,6144,32768:64,512",
    "8,2048,16,8,6144,32768:64,32",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append")
    args = parser.parse_args()
    for shape in args.shape or SHAPES:
        model, run = shape.split(":")
        config = Config(*map(int, model.split(",")))
        batch, mean = map(int, run.split(","))
        lengths = jnp.asarray(np.random.default_rng(0).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)
        unit = 1024 if mean >= 256 else 256  # (benchmarks/decode_megakernel.py's cache_len)
        s_max = -(-(3 * mean // 2 + 36) // unit) * unit
        params = init_params(config)
        caches = random_caches(config, lengths, s_max)
        tokens = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
        inputs = kernel_inputs(tokens, *caches, lengths, params)
        weights = sum(params[name].nbytes for name in (*STACKED, "final_norm", "w_vocab"))
        kv = config.layers * float(jnp.sum(lengths)) * config.kv_heads * HEAD_DIM * 2 * 2
        lowered = decode_megakernel.lower(*inputs)  # (tunes)
        print(
            f"== {shape}: tuned {lowered.config}; {lowered.steps} steps at most, {lowered.vmem_bytes / 2**20:.1f} MiB"
        )
        print(f"   HBM floor {(weights + kv) / PEAK_GBS / 1e6:.3f} ms ({(weights + kv) / 1e6:.0f} MB)")
        print(autokernel.profile(decode_megakernel, *inputs), flush=True)


if __name__ == "__main__":
    main()
