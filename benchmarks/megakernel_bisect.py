"""Bisect the decode megakernel against XLA at long caches, on a TPU v5e: which cache sizes and lengths make
it disagree.

    python benchmarks/megakernel_bisect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from decode_megakernel import (
    Config,
    MegaTiles,
    decode_megakernel,
    init_params,
    kernel_inputs,
    layer_caches,
    layer_weights,
    megakernel_step,
    random_caches,
    xla_step,
)


def largest(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def main() -> None:
    tiles = MegaTiles(block_d=256, block_kv=256, block_a=512, block_f=512, block_v=1024)
    config = Config(layers=2, d_model=256, heads=2, kv_heads=1, ffn=512, vocab=2048)
    params = init_params(config, seed=0)
    # The flattened cache has batch * s_max rows a layer: around 2**15 of them.
    for batch, s_max in ((8, 4096), (8, 4608), (16, 2048), (16, 2304), (4, 8192), (4, 8448), (2, 16384), (2, 16512)):
        lens = [s_max * 3 // 4 - 7 * b for b in range(batch)]
        lengths = jnp.array(lens, jnp.int32)
        k, v = random_caches(config, lengths, s_max, seed=1)
        tokens = jnp.arange(batch, dtype=jnp.int32) + 3
        ours, *_ = megakernel_step(tokens, k, v, lengths, params, tiles)
        theirs, *_ = xla_step(tokens, layer_caches(k, v), lengths, layer_weights(params), params)
        ref = decode_megakernel.reference(*kernel_inputs(tokens, k, v, lengths, params), tiles)[0]
        print(
            f"rows={batch * s_max} B={batch} S={s_max}: |kernel - xla| {largest(ours, theirs):.3f}  "
            f"|reference - xla| {largest(ref, theirs):.3f}  (logits up to {float(jnp.abs(theirs).max()):.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
