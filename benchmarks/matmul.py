"""Matmul and causal matmul: autokernel against XLA, bf16, on a TPU v5e.

uv run python benchmarks/matmul.py                       # 8192^3, 512^3 and 1024x2048x1024 tiles
uv run python benchmarks/matmul.py --size 4096 --tiles 512,512,512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from causal_matmul import make_causal_matmul
from matmul import make_matmul
from timing import require_tpu, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--tiles", action="append", help="tm,tn,tk (repeatable)")
    args = parser.parse_args()
    require_tpu()
    n = args.size
    tiles = [tuple(map(int, t.split(","))) for t in args.tiles or ["512,512,512", "1024,2048,1024"]]
    x = jax.random.normal(jax.random.key(0), (n, n), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (n, n), jnp.bfloat16)
    flops = 2 * n**3

    xla = jax.jit(lambda x, y: jnp.dot(x, y, preferred_element_type=jnp.float32).astype(x.dtype))
    xla_causal = jax.jit(lambda x, y: jnp.tril(jnp.dot(x, y, preferred_element_type=jnp.float32)).astype(x.dtype))
    print(f"{n}^3 bf16, ms (TFLOP/s of the dense matmul's work)")
    t = time_ms(xla, x, y)
    print(f"  XLA dense                   {t:8.3f}  ({flops / t / 1e9:6.1f})")
    t = time_ms(xla_causal, x, y)
    print(f"  XLA causal (tril)           {t:8.3f}")
    for tm, tn, tk in tiles:
        t = time_ms(make_matmul(tm, tn, tk), x, y)
        print(f"  autokernel dense  {tm}x{tn}x{tk}  {t:8.3f}  ({flops / t / 1e9:6.1f})")
        t = time_ms(make_causal_matmul(tm, tn, tk), x, y)
        print(f"  autokernel causal {tm}x{tn}x{tk}  {t:8.3f}")


if __name__ == "__main__":
    main()
