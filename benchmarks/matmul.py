"""Matmul and causal matmul: autokernel (tuned, and at fixed tiles) against XLA, bf16, on a TPU v5e.

uv run python benchmarks/matmul.py
uv run python benchmarks/matmul.py --size 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from causal_matmul import causal_matmul
from matmul import MatmulTiles, matmul
from timing import require_tpu, time_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    args = parser.parse_args()
    require_tpu()
    n = args.size
    x = jax.random.normal(jax.random.key(0), (n, n), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (n, n), jnp.bfloat16)
    tflops = lambda ms: 2 * n**3 / ms / 1e9  # noqa: E731

    xla = jax.jit(lambda x, y: jnp.dot(x, y, preferred_element_type=jnp.float32).astype(x.dtype))
    xla_causal = jax.jit(lambda x, y: jnp.tril(jnp.dot(x, y, preferred_element_type=jnp.float32)).astype(x.dtype))
    print(f"{n}^3 bf16, ms (TFLOP/s of the dense matmul's work)")
    t = time_ms(xla, x, y)
    print(f"  {'XLA dense':48} {t:8.3f}  ({tflops(t):6.1f})")
    t = time_ms(xla_causal, x, y)
    print(f"  {'XLA causal (tril)':48} {t:8.3f}")
    for name, kernel in (("dense", matmul), ("causal", causal_matmul)):
        t = time_ms(jax.jit(kernel), x, y)  # tunes on the first call
        print(f"  {f'autokernel {name}, tuned':48} {t:8.3f}  ({tflops(t):6.1f})")
        print("    " + str(kernel.tuning(x, y)).replace("\n", "\n    "))
        t = time_ms(jax.jit(kernel), x, y, MatmulTiles())
        print(f"  {f'autokernel {name}, default {MatmulTiles()}':48} {t:8.3f}  ({tflops(t):6.1f})")


if __name__ == "__main__":
    main()
