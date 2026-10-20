"""Bisect the decode megakernel against XLA at long caches, on a TPU v5e: which cache sizes and lengths make
it disagree.

    python benchmarks/megakernel_bisect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

import autokernel

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from decode_megakernel import (
    Config,
    MegaTiles,
    init_params,
    layer_caches,
    layer_weights,
    megakernel_step,
    random_caches,
    xla_step,
)


def largest(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


@autokernel.kernel
def last_rows(x):
    """Each (layer, block of 256 rows) of x (L, blocks, 256, 1024): its last 16 rows, through VMEM by DMA."""
    L, N, _, D = x.shape
    out = jnp.zeros((L, N, 16, D), x.dtype)
    for layer in range(L):
        for n in range(N):
            out = out.at[layer, n, :, :].set(x[layer, n, 240:256, :])
    return out


def dma_probe() -> None:
    """A plain kernel's DMAs from arrays just under and over 2**28 bytes."""
    for layers, rows in ((4, 32768), (4, 33024), (2, 65536), (2, 65792), (1, 139776)):
        x = jnp.arange(layers * rows * 1024, dtype=jnp.int32) % 251
        x = x.reshape(layers, rows // 256, 256, 1024).astype(jnp.bfloat16)
        bad = int(jnp.sum(last_rows(x) != x[:, :, 240:]))
        print(f"dma: bytes={x.nbytes} ({x.nbytes / 2**28:.3f} x 2**28): {bad} wrong elements", flush=True)


def main() -> None:
    dma_probe()
    tiles = MegaTiles(block_d=256, block_kv=256, block_a=512, block_f=512, block_v=1024)
    # Each cache (layers, batch * s_max, 8 heads * 128) bf16: around 2**28 bytes.
    for layers, batch, s_max in ((4, 8, 4096), (4, 8, 4352), (2, 8, 7168), (3, 8, 7168), (1, 8, 16384), (1, 8, 17408)):
        config = Config(layers=layers, d_model=1024, heads=8, kv_heads=8, ffn=1024, vocab=2048)
        params = init_params(config, seed=0)
        lens = [s_max * 3 // 4 - 7 * b for b in range(batch)]
        lengths = jnp.array(lens, jnp.int32)
        k, v = random_caches(config, lengths, s_max, seed=1)
        tokens = jnp.arange(batch, dtype=jnp.int32) + 3
        ours, *_ = megakernel_step(tokens, k, v, lengths, params, tiles)
        theirs, *_ = xla_step(tokens, layer_caches(k, v), lengths, layer_weights(params), params)
        print(
            f"bytes={k.nbytes} ({k.nbytes / 2**28:.3f} x 2**28) L={layers} B={batch} S={s_max}: "
            f"|kernel - xla| {largest(ours, theirs):.3f}  (logits up to {float(jnp.abs(theirs).max()):.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
