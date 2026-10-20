"""Causal multi-head attention: autokernel against JAX's splash attention at the same block sizes, bf16,
head dim 128, q pre-scaled for both, on a TPU v5e.

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
from causal_attention import make_causal_attention, make_splash_attention
from timing import require_tpu, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="B,H,S,D (repeatable)")
    parser.add_argument("--blocks", type=int, action="append", help="block size for queries and keys (repeatable)")
    args = parser.parse_args()
    require_tpu()
    shapes = [tuple(map(int, s.split(","))) for s in args.shape or ["4,16,2048,128", "1,16,4096,128"]]
    for b, h, s, d in shapes:
        keys = jax.random.split(jax.random.key(0), 3)
        q, k, v = (jax.random.normal(key, (b, h, s, d), jnp.bfloat16) for key in keys)
        q = (q * d**-0.5).astype(q.dtype)
        print(f"B={b} H={h} S={s} D={d}, ms")
        print(f"  {'block_q x block_kv':20} {'splash':>8} {'autokernel':>10} {'speedup':>8} {'max diff':>9}")
        for tq in args.blocks or [256, 512, 1024]:
            for tk in args.blocks or [256, 512, 1024]:
                splash = make_splash_attention(h, s, tq, tk)
                ours = jax.jit(make_causal_attention(tq, tk))
                diff = float(jnp.max(jnp.abs(ours(q, k, v).astype(jnp.float32) - splash(q, k, v).astype(jnp.float32))))
                t_splash, t_ours = time_ms(splash, q, k, v), time_ms(ours, q, k, v)
                print(f"  {f'{tq} x {tk}':20} {t_splash:8.3f} {t_ours:10.3f} {t_splash / t_ours:7.2f}x {diff:9.4f}")


if __name__ == "__main__":
    main()
