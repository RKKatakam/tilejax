"""Probe: the compute time of decode attention over one page of RPA v3's cache, (128, 8, 2, 128) bf16 (128
positions, 8 kv heads, each head's key and value side by side), in VMEM, for several ways of reading it.

Each kernel runs an online-softmax step of 32 query heads (4 per kv head) against the page N times; time
per page = (t(N) - t(1)) / (N - 1). The page's DMA at the v5e's 819 GB/s takes 0.64 us.

    python benchmarks/vs_vllm_rpa/probe_page.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PS, HKV, D, H = 128, 8, 128, 32
G = H // HKV
NEG = -1e30


def online(s, v, m_ref, l_ref, acc_ref, rows):
    """One online-softmax update of rows `rows` (a slice) with scores s and values v."""
    m_prev = m_ref[rows]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_ref[rows] = alpha * l_ref[rows] + jnp.sum(p, axis=1, keepdims=True)
    acc_ref[rows] = alpha * acc_ref[rows] + jnp.dot(p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)


def head_mask():
    """(H, PS * HKV): query head h sees the columns (position t, kv head g) with g == h // G."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (H, PS * HKV), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (H, PS * HKV), 1)
    return (cols % HKV) == (rows // G)


def all_heads(q, k, v, m_ref, l_ref, acc_ref):
    """Every query head against every kv head's keys (k, v: (PS * HKV, D), row t * HKV + g), others masked."""
    s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    s = jnp.where(head_mask(), s, NEG)
    online(s, v, m_ref, l_ref, acc_ref, slice(None))


def f_vllm(page_ref, q_ref, m_ref, l_ref, acc_ref):
    """vLLM's: the page as uint32 words (key | value << 16), each kv head's rows by a strided load."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)
    for g in range(HKV):
        kv = words[pl.ds(g, PS, stride=HKV)]
        k = pltpu.bitcast(kv.astype(jnp.uint16), jnp.bfloat16)
        v = pltpu.bitcast((kv >> 16).astype(jnp.uint16), jnp.bfloat16)
        q = q_ref[g * G : (g + 1) * G]
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        online(s, v, m_ref, l_ref, acc_ref, slice(g * G, (g + 1) * G))


def f_words_all(page_ref, q_ref, m_ref, l_ref, acc_ref):
    """The page as uint32 words, loaded whole; keys and values unpacked by shifts; all heads at once."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)[...]
    k = pltpu.bitcast(words << 16, jnp.float32).astype(jnp.bfloat16)
    v = pltpu.bitcast(words & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)
    all_heads(q_ref[...], k, v, m_ref, l_ref, acc_ref)


def f_value_slices(page_ref, q_ref, m_ref, l_ref, acc_ref):
    """The page loaded as a bf16 value, keys and values sliced out of it, all heads at once."""
    page = page_ref[...]
    k = page[:, :, 0, :].reshape(PS * HKV, D)
    v = page[:, :, 1, :].reshape(PS * HKV, D)
    all_heads(q_ref[...], k, v, m_ref, l_ref, acc_ref)


def f_ref_slices(page_ref, q_ref, m_ref, l_ref, acc_ref):
    """Keys and values loaded from the ref by index (page_ref[:, :, 0, :]), all heads at once."""
    k = page_ref[:, :, 0, :].reshape(PS * HKV, D)
    v = page_ref[:, :, 1, :].reshape(PS * HKV, D)
    all_heads(q_ref[...], k, v, m_ref, l_ref, acc_ref)


def f_f32_slices(page_ref, q_ref, m_ref, l_ref, acc_ref):
    """The page as f32, keys and values sliced out, all heads at once."""
    page = page_ref[...].astype(jnp.float32)
    k = page[:, :, 0, :].reshape(PS * HKV, D).astype(jnp.bfloat16)
    v = page[:, :, 1, :].reshape(PS * HKV, D).astype(jnp.bfloat16)
    all_heads(q_ref[...], k, v, m_ref, l_ref, acc_ref)


def f_classic(k_ref, v_ref, q_ref, m_ref, l_ref, acc_ref):
    """For scale: separate key and value pages (PS, HKV * D) (vLLM's classic layout), all heads at once
    through a block-diagonal query (examples/decode.py's)."""
    q = q_ref[...]
    group = jax.lax.broadcasted_iota(jnp.int32, (H, HKV * D), 0) // G
    block = jax.lax.broadcasted_iota(jnp.int32, (H, HKV * D), 1) // D
    qbd = jnp.where(group == block, jnp.tile(q, (1, HKV)), 0).astype(jnp.bfloat16)
    s = jax.lax.dot_general(qbd, k_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
    pv = jnp.dot(p.astype(jnp.bfloat16), v_ref[...], preferred_element_type=jnp.float32)  # (H, HKV * D)
    own = pv[:, :D]
    for g in range(1, HKV):
        own = jnp.where(jax.lax.broadcasted_iota(jnp.int32, (H, D), 0) // G == g, pv[:, g * D : (g + 1) * D], own)
    acc_ref[...] = alpha * acc_ref[...] + own


VARIANTS = {
    "vllm (u32 strided per head)": (f_vllm, False),
    "u32 whole, shifts, all heads": (f_words_all, False),
    "bf16 value slices, all heads": (f_value_slices, False),
    "ref slices, all heads": (f_ref_slices, False),
    "f32 value slices, all heads": (f_f32_slices, False),
    "classic K/V pages, block-diag": (f_classic, True),
}


def build(step, classic, n):
    def kernel(*refs):
        *ins, out_ref, m_ref, l_ref, acc_ref = refs
        m_ref[...] = jnp.full(m_ref.shape, NEG, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        @pl.loop(0, n)
        def _(i):
            step(*ins, m_ref, l_ref, acc_ref)

        out_ref[...] = acc_ref[...] / l_ref[...]

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((H, D), jnp.float32),
            in_specs=[vmem] * (3 if classic else 2),
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((H, 1), jnp.float32),
                pltpu.VMEM((H, 1), jnp.float32),
                pltpu.VMEM((H, D), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 << 20),
        )
    )


def main():
    rng = np.random.default_rng(0)
    page = jnp.asarray(rng.standard_normal((PS, HKV, 2, D)), jnp.bfloat16)
    q = jnp.asarray(rng.standard_normal((H, D)) * D**-0.5, jnp.bfloat16)
    k_cls = page[:, :, 0, :].reshape(PS, HKV * D)
    v_cls = page[:, :, 1, :].reshape(PS, HKV * D)
    # The answer: one page attended once.
    k64, v64 = np.asarray(page[:, :, 0, :], np.float64), np.asarray(page[:, :, 1, :], np.float64)
    q64 = np.asarray(q, np.float64).reshape(HKV, G, D)
    s = np.einsum("hgd,thd->hgt", q64, k64)
    p = np.exp(s - s.max(-1, keepdims=True))
    want = (np.einsum("hgt,thd->hgd", p, v64) / p.sum(-1)[..., None]).reshape(H, D)
    for name, (step, classic) in VARIANTS.items():
        args = (k_cls, v_cls, q) if classic else (page, q)
        try:
            times = {}
            for n in (1, 201):
                fn = build(step, classic, n)
                out = jax.block_until_ready(fn(*args))
                if n == 1:
                    err = float(np.abs(np.asarray(out, np.float64) - want).max())
                samples = []
                for _ in range(5):
                    t = time.perf_counter()
                    for _ in range(10):
                        r = fn(*args)
                    jax.block_until_ready(r)
                    samples.append((time.perf_counter() - t) / 10)
                times[n] = sorted(samples)[2]
            per = (times[201] - times[1]) / 200
            print(f"{name:34} {per * 1e6:7.3f} us per page   error vs f64 {err:.2e}", flush=True)
        except Exception as e:
            print(f"{name:34} failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)


if __name__ == "__main__":
    main()
