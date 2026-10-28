"""Probe: one prefill step of attention on a TPU v5e, in VMEM: BQ query tokens (32 heads, 4 per kv head)
against a page of RPA v3's cache (128 positions x 8 kv heads x key|value, bf16), flash-attention style,
for ways of taking each kv head's keys and values out of the page. The MXU's share at its peak: 2 x BQ x 32
x 128 x 128 MACs per page (BQ 256: 2.7 us).

    python benchmarks/vs_vllm_rpa/probe_prefill.py
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PS, HKV, D, H = 128, 8, 128, 32
G = H // HKV
NEG = -1e30


def unpack(kv):
    k = pltpu.bitcast(kv << 16, jnp.float32).astype(jnp.bfloat16)
    v = pltpu.bitcast(kv & jnp.uint32(0xFFFF0000), jnp.float32).astype(jnp.bfloat16)
    return k, v


def head_step(q, k, v, m_ref, l_ref, acc_ref, g, causal_cols):
    s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    s = jnp.where(causal_cols, s, NEG)
    m_prev = m_ref[g]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_ref[g] = alpha * l_ref[g] + jnp.sum(p, axis=1, keepdims=True)
    acc_ref[g] = alpha * acc_ref[g] + jnp.dot(p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)


def causal(bq):
    rows = jax.lax.broadcasted_iota(jnp.int32, (bq * G, PS), 0) // G
    cols = jax.lax.broadcasted_iota(jnp.int32, (bq * G, PS), 1)
    return cols <= rows + 64  # (half the page visible on the first rows: a mask to compute, as in RPA)


def f_strided(page_ref, qh_ref, m_ref, l_ref, acc_ref, bq):
    """RPA's: each kv head's words by a strided load from the page's VMEM buffer seen as words."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)
    mask = causal(bq)
    for g in range(HKV):
        k, v = unpack(words[pl.ds(g, PS, stride=HKV)])
        head_step(qh_ref[g], k, v, m_ref, l_ref, acc_ref, g, mask)


def f_slices(page_ref, qh_ref, m_ref, l_ref, acc_ref, bq):
    """Each kv head's words sliced out of the page loaded as words."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)[...].reshape(PS, HKV, D)
    mask = causal(bq)
    for g in range(HKV):
        k, v = unpack(words[:, g, :])
        head_step(qh_ref[g], k, v, m_ref, l_ref, acc_ref, g, mask)


def f_batched(page_ref, qh_ref, m_ref, l_ref, acc_ref, bq):
    """Strided loads per head, then every head's matmuls and softmax as batched (8, ...) arrays."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)
    kv = jnp.stack([words[pl.ds(g, PS, stride=HKV)] for g in range(HKV)])  # (8, PS, D)
    k, v = unpack(kv)
    s = jax.lax.dot_general(qh_ref[...], k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
    s = jnp.where(causal(bq)[None], s, NEG)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=2, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=2, keepdims=True)
    pv = jax.lax.dot_general(
        p.astype(jnp.bfloat16), v, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
    )
    acc_ref[...] = alpha * acc_ref[...] + pv


def f_wide(page_ref, qh_ref, m_ref, l_ref, acc_ref, bq, pages=1, mask=True):
    """Batched heads, the running max and sum lane-wide (8, R, 128), `pages` pages per step."""
    words = page_ref.bitcast(jnp.uint32).reshape(PS * HKV, D)
    kv = jnp.stack([words[pl.ds(g, PS, stride=HKV)] for g in range(HKV)])  # (8, PS, D)
    if pages > 1:
        kv = jnp.concatenate([kv] * pages, axis=1)  # (the same page again: the shapes of `pages` pages)
    k, v = unpack(kv)
    n = PS * pages
    s = jax.lax.dot_general(qh_ref[...], k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
    if mask:
        rows = jax.lax.broadcasted_iota(jnp.int32, s.shape, 1) // G
        cols = jax.lax.broadcasted_iota(jnp.int32, s.shape, 2)
        s = jnp.where(cols <= rows + 64, s, NEG)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=2, keepdims=True))
    p = jnp.exp(s - jnp.tile(m_new, (1, 1, n // 128)))
    alpha = jnp.exp(m_prev - m_new)
    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=2, keepdims=True)
    pv = jax.lax.dot_general(
        p.astype(jnp.bfloat16), v, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
    )
    acc_ref[...] = alpha * acc_ref[...] + pv


VARIANTS = {
    "strided, batched heads": (f_batched, 1),
    "wide stats, 1 page": (functools.partial(f_wide, pages=1), 1),
    "wide stats, 1 page, no mask": (functools.partial(f_wide, pages=1, mask=False), 1),
    "wide stats, 2 pages": (functools.partial(f_wide, pages=2), 2),
    "wide stats, 4 pages": (functools.partial(f_wide, pages=4), 4),
    "wide stats, 4 pages, no mask": (functools.partial(f_wide, pages=4, mask=False), 4),
}


def build(step, bq, n, wide):
    def kernel(page_ref, qh_ref, out_ref, m_ref, l_ref, acc_ref):
        m_ref[...] = jnp.full(m_ref.shape, NEG, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        @pl.loop(0, n)
        def _(i):
            step(page_ref, qh_ref, m_ref, l_ref, acc_ref, bq)

        out_ref[...] = acc_ref[...] / l_ref[...][..., :1]

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    rows = bq * G
    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((HKV, rows, D), jnp.float32),
            in_specs=[vmem, vmem],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((HKV, rows, 128 if wide else 1), jnp.float32),
                pltpu.VMEM((HKV, rows, 128 if wide else 1), jnp.float32),
                pltpu.VMEM((HKV, rows, D), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 << 20),
        )
    )


def main():
    rng = np.random.default_rng(0)
    page = jnp.asarray(rng.standard_normal((PS, HKV, 2, D)), jnp.bfloat16)
    for bq in (128, 256):
        qh = jnp.asarray(rng.standard_normal((HKV, bq * G, D)) * D**-0.5, jnp.bfloat16)
        peak = 2 * bq * H * PS * D / 98.5e12  # MACs at 197 TFLOP/s
        for name, (step, pages) in VARIANTS.items():
            try:
                times = {}
                for n in (1, 51):
                    fn = build(step, bq, n, "wide" in name)
                    jax.block_until_ready(fn(page, qh))
                    samples = []
                    for _ in range(5):
                        t = time.perf_counter()
                        for _ in range(5):
                            r = fn(page, qh)
                        jax.block_until_ready(r)
                        samples.append((time.perf_counter() - t) / 5)
                    times[n] = sorted(samples)[2]
                per = (times[51] - times[1]) / 50
                print(
                    f"BQ {bq:3} {name:30} {per / pages * 1e6:7.2f} us per page (MXU at peak: {peak * 1e6:.2f})",
                    flush=True,
                )
            except Exception as e:
                print(f"BQ {bq:3} {name:24} failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)


if __name__ == "__main__":
    main()
