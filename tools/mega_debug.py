"""Bisect the megakernel mismatch at long caches (TPU)."""
import sys
sys.path.insert(0, "examples"); sys.path.insert(0, "tests")
import numpy as np, jax, jax.numpy as jnp
from decode_megakernel import Config, MegaTiles, megakernel_step, xla_step, layer_weights, layer_caches, init_params, random_caches, kernel_inputs, decode_megakernel

tiles = MegaTiles(block_d=256, block_kv=256, block_a=512, block_f=512, block_v=1024)
for layers, d, h, hkv, ffn in [(1, 256, 2, 1, 512), (1, 2048, 16, 8, 6144)]:
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
            d1 = float(jnp.max(jnp.abs(ours.astype(float) - theirs.astype(float))))
            d2 = float(jnp.max(jnp.abs(ref.astype(float) - theirs.astype(float))))
            print(f"d={d} S={s_max} lengths={lens}: |kernel - xla| {d1:.3f}  |reference - xla| {d2:.3f}  max {float(jnp.abs(theirs).max()):.2f}", flush=True)
