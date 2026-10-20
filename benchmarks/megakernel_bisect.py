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
    for layers, d, h, hkv, ffn in [(2, 256, 2, 1, 512), (2, 2048, 16, 8, 6144)]:
        config = Config(layers=layers, d_model=d, heads=h, kv_heads=hkv, ffn=ffn, vocab=2048)
        params = init_params(config, seed=0)
        for s_max in (4096, 5120, 6144, 7168, 8192):
            for lens in ([s_max * 3 // 4, 100], [4000, 100], [4200, 100], [100, 4200]):
                if max(lens) >= s_max:
                    continue
                lengths = jnp.array(lens, jnp.int32)
                k, v = random_caches(config, lengths, s_max, seed=1)
                tokens = jnp.array([3, 5], jnp.int32)
                ours, *_ = megakernel_step(tokens, k, v, lengths, params, tiles)
                theirs, *_ = xla_step(tokens, layer_caches(k, v), lengths, layer_weights(params), params)
                ref = decode_megakernel.reference(*kernel_inputs(tokens, k, v, lengths, params), tiles)[0]
                print(
                    f"d={d} S={s_max} lengths={lens}: |kernel - xla| {largest(ours, theirs):.3f}  "
                    f"|reference - xla| {largest(ref, theirs):.3f}  (logits up to {float(jnp.abs(theirs).max()):.2f})",
                    flush=True,
                )


if __name__ == "__main__":
    main()
