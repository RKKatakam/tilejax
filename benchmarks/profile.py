"""Profile the examples on a TPU v5e from their trace regions, and let `autokernel.optimize` choose their
buffering.

    uv run python benchmarks/profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import autokernel

autokernel.enable_profiling()  # before JAX starts its TPU backend

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from causal_attention import make_causal_attention
from causal_matmul import make_causal_matmul
from matmul import make_matmul
from timing import require_tpu


def main() -> None:
    require_tpu()
    x = jax.random.normal(jax.random.key(0), (8192, 8192), jnp.bfloat16)
    y = jax.random.normal(jax.random.key(1), (8192, 8192), jnp.bfloat16)
    keys = jax.random.split(jax.random.key(2), 3)
    q, k, v = (jax.random.normal(key, (4, 16, 2048, 128), jnp.bfloat16) for key in keys)
    cases = [
        ("matmul 8192^3, 256^3 tiles", make_matmul(256, 256, 256), (x, y)),
        ("matmul 8192^3, 512^3 tiles", make_matmul(512, 512, 512), (x, y)),
        ("causal matmul 8192^3, 512^3 tiles", make_causal_matmul(512, 512, 512), (x, y)),
        ("attention 4x16x2048x128, 512x512 blocks", make_causal_attention(512, 512), (q, k, v)),
    ]
    for title, kernel, inputs in cases:
        print(f"== {title}")
        print(autokernel.profile(kernel, *inputs))
        print(autokernel.optimize(kernel, *inputs))
        print()


if __name__ == "__main__":
    main()
