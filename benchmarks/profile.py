"""Profile the examples on a TPU v5e from their trace regions, at their default schedules and tuned.

uv run python benchmarks/profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

import autokernel  # turns trace regions on: JAX's TPU backend hasn't started yet

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from causal_attention import AttentionBlocks, causal_attention
from causal_matmul import causal_matmul
from matmul import MatmulTiles, matmul
from timing import require_tpu


def main() -> None:
    require_tpu()
    x = jax.random.normal(jax.random.key(0), (8192, 8192), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (8192, 8192), jnp.bfloat16)
    keys = jax.random.split(jax.random.key(2), 3)
    q, k, v = (jax.random.normal(key, (4, 16, 2048, 128), jnp.bfloat16) for key in keys)
    cases = [
        ("matmul 8192^3", matmul, (x, y), MatmulTiles(tm=256, tn=256, tk=256)),
        ("causal matmul 8192^3", causal_matmul, (x, y), MatmulTiles()),
        ("attention 4x16x2048x128", causal_attention, (q, k, v), AttentionBlocks()),
    ]
    for title, kernel, inputs, tunable in cases:
        print(f"== {title}")
        print(autokernel.profile(kernel, *inputs, tunable))
        print(kernel.tuning(*inputs))
        print(autokernel.profile(kernel, *inputs))
        print()


if __name__ == "__main__":
    main()
