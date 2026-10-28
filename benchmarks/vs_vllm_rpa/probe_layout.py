"""Probe: how XLA lays out RPA v3's cache (pages, page_size, Hkv, 2, D) bf16 in a v5e's HBM, and which
views of it are free (a bitcast) rather than a copy.

    python benchmarks/vs_vllm_rpa/probe_layout.py
"""

import jax
import jax.numpy as jnp

cache = jnp.zeros((64, 128, 8, 2, 128), jnp.bfloat16)
print("cache format:", cache.format)
views = {
    "(pages, ps, Hkv * 2 * D)": lambda c: c.reshape(64, 128, 8 * 2 * 128),
    "(pages, ps, Hkv, 2 * D)": lambda c: c.reshape(64, 128, 8, 2 * 128),
    "(pages * ps, Hkv, 2, D)": lambda c: c.reshape(64 * 128, 8, 2, 128),
    "uint32 (pages, ps, Hkv, D)": lambda c: jax.lax.bitcast_convert_type(jnp.swapaxes(c, -1, -2), jnp.uint32),
}
for name, view in views.items():
    hlo = jax.jit(view).lower(cache).compile().as_text()
    body = [line.strip() for line in hlo.splitlines() if "ROOT" in line or " copy" in line or "fusion" in line]
    print(f"{name}: {'COPY' if any('copy' in b or 'fusion' in b for b in body) else 'free'}")
    for b in body[:4]:
        print("    ", b[:200])
