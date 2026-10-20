"""RMSNorm fused with the projection after it: autokernel (tuned, and at its defaults) against XLA
compiling `rmsnorm(x) @ w` in plain JAX, bf16, on a TPU v5e.

    uv run python benchmarks/rmsnorm_matmul.py
    uv run python benchmarks/rmsnorm_matmul.py --shape 8192,4096,4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from rmsnorm_matmul import NormMatmulTiles, rmsnorm_matmul, rmsnorm_then_matmul
from timing import require_tpu, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="M,D,N: x is (M, D), w is (D, N) (repeatable)")
    args = parser.parse_args()
    require_tpu()
    shapes = [
        tuple(map(int, s.split(","))) for s in args.shape or ["8192,4096,4096", "16384,2048,2048", "512,4096,4096"]
    ]
    xla = jax.jit(rmsnorm_then_matmul)
    ours = jax.jit(rmsnorm_matmul)
    for m, d, n in shapes:
        keys = jax.random.split(jax.random.key(0), 3)
        x = jax.random.normal(keys[0], (m, d), jnp.bfloat16)
        gamma = (1 + 0.1 * jax.random.normal(keys[1], (1, d))).astype(jnp.bfloat16)
        w = (jax.random.normal(keys[2], (d, n)) / d**0.5).astype(jnp.bfloat16)
        diff = jnp.max(jnp.abs(ours(x, gamma, w).astype(jnp.float32) - xla(x, gamma, w).astype(jnp.float32)))
        tflops = lambda ms: 2 * m * d * n / ms / 1e9  # noqa: E731, B023
        t_xla = time_ms(xla, x, gamma, w)
        t_default = time_ms(ours, x, gamma, w, NormMatmulTiles())
        t_tuned = time_ms(ours, x, gamma, w)
        print(f"M={m} D={d} N={n}, ms (TFLOP/s); max |autokernel - XLA| {float(diff):.4f}")
        print(f"  XLA, rmsnorm then matmul      {t_xla:8.3f}  ({tflops(t_xla):6.1f})")
        print(f"  autokernel fused, defaults    {t_default:8.3f}  ({tflops(t_default):6.1f})")
        print(f"  autokernel fused, tuned       {t_tuned:8.3f}  ({tflops(t_tuned):6.1f})  {t_xla / t_tuned:.2f}x XLA")
        print("    " + str(rmsnorm_matmul.tuning(x, gamma, w)).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
