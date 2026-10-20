"""Causal multi-head attention: autokernel against JAX's splash attention, bf16, head dim 128, q pre-scaled
for both, on a TPU v5e. At the same block sizes, and each at its best: splash over the same grid of block
sizes, autokernel as its tuner chooses.

    uv run python benchmarks/attention.py
    uv run python benchmarks/attention.py --shape 4,16,2048,128 --blocks 512 --blocks 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from causal_attention import AttentionBlocks, causal_attention, splash_attention
from timing import require_tpu, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="B,H,S,D (repeatable)")
    parser.add_argument("--blocks", type=int, action="append", help="block size for queries and keys (repeatable)")
    args = parser.parse_args()
    require_tpu()
    shapes = [tuple(map(int, s.split(","))) for s in args.shape or ["4,16,2048,128", "1,16,4096,128"]]
    ours = jax.jit(causal_attention)
    for b, h, s, d in shapes:
        keys = jax.random.split(jax.random.key(0), 3)
        q, k, v = (jax.random.normal(key, (b, h, s, d), jnp.bfloat16) for key in keys)
        q = (q * d**-0.5).astype(q.dtype)
        print(f"B={b} H={h} S={s} D={d}, ms")
        print(f"  {'block_q x block_kv':20} {'splash':>8} {'autokernel':>10} {'speedup':>8} {'max diff':>9}")
        best_splash = float("inf")
        for tq in args.blocks or [256, 512, 1024]:
            for tk in args.blocks or [256, 512, 1024]:
                splash = splash_attention(h, s, tq, tk)
                blocks = AttentionBlocks(block_q=tq, block_kv=tk)
                diff = jnp.max(jnp.abs(ours(q, k, v, blocks).astype(jnp.float32) - splash(q, k, v)))
                t_splash, t_ours = time_ms(splash, q, k, v), time_ms(ours, q, k, v, blocks)
                best_splash = min(best_splash, t_splash)
                speedup = t_splash / t_ours
                print(f"  {f'{tq} x {tk}':20} {t_splash:8.3f} {t_ours:10.3f} {speedup:7.2f}x {float(diff):9.4f}")
        t_tuned = time_ms(ours, q, k, v)  # tunes on the first call
        print(f"  splash at its best {best_splash:.3f}; autokernel tuned {t_tuned:.3f} ({best_splash / t_tuned:.2f}x)")
        print("    " + str(causal_attention.tuning(q, k, v)).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
