"""Probe: the time of one KDA decode tile's math on a TPU v5e, in VMEM, for several formulations (no DMAs).

A tile is TH heads of one sequence: the state S (TH, d_k, d_v) f32 and the rows a (decay), k, q, v
(TH, 128), beta (TH, 1). Each kernel runs the step N times on VMEM-resident values; the time per step is
(t(N) - t(1)) / (N - 1).

    python benchmarks/vs_vllm_kda/probe_math.py
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

K = 128
DOT = (((2,), (1,)), ((0,), (0,)))


def ftb(x):  # (TH, K) -> (TH, K, 1): GDN v3's fused_transpose_broadcast
    x3 = x[:, None, :]
    shape = (x.shape[0], K, K)
    mask = jax.lax.broadcasted_iota(jnp.int32, shape, 2) == jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    return jnp.where(mask, x3, 0).sum(axis=2, keepdims=True)


def mxu(S, a, k, q, v, beta, precision=None):  # vLLM's step: matvecs on the MXU
    Sd = S * ftb(a)
    vu = jax.lax.dot(k[:, None, :], Sd, dimension_numbers=DOT, preferred_element_type=jnp.float32, precision=precision)
    vn = beta[:, :, None] * (v[:, None, :] - vu)
    S2 = Sd + ftb(k) * vn
    out = jax.lax.dot(q[:, None, :], S2, dimension_numbers=DOT, preferred_element_type=jnp.float32, precision=precision)
    return out[:, 0], S2


def vpu_ftb(S, a, k, q, v, beta):  # matvecs as sublane reductions, columns by fused_transpose_broadcast
    ac, kc, qc = ftb(a), ftb(k), ftb(q)
    Sd = S * ac
    vn = beta * (v - jnp.sum(kc * Sd, axis=1))
    S2 = Sd + kc * vn[:, None, :]
    return jnp.sum(qc * S2, axis=1), S2


def vpu_ftb_once(S, a, k, q, v, beta):  # one read of S for both reductions: S^T(a k), S^T(a q)
    ac, kc = ftb(a), ftb(k)
    akc, aqc = ftb(a * k), ftb(a * q)
    rk = jnp.sum(akc * S, axis=1)
    rq = jnp.sum(aqc * S, axis=1)
    u = beta * (v - rk)
    out = rq + jnp.sum(k * q, axis=1, keepdims=True) * u
    return out, ac * S + kc * u[:, None, :]


def _columns(rows):  # (n, TH, K) rows -> (K, n * TH) by one 2-D transpose (rows padded to 128)
    n, th, _ = rows.shape
    r = rows.reshape(n * th, K)
    if n * th < 128:
        r = jnp.concatenate([r, jnp.zeros((128 - n * th, K), r.dtype)])
    return r.T


def vpu_transpose_heads(S, a, k, q, v, beta):  # one transpose for every column, then head by head
    th = S.shape[0]
    cols = _columns(jnp.stack([a, k, q]))
    outs, states = [], []
    for h in range(th):
        ac, kc, qc = cols[:, h : h + 1], cols[:, th + h : th + h + 1], cols[:, 2 * th + h : 2 * th + h + 1]
        Sd = S[h] * ac
        vn = beta[h : h + 1] * (v[h : h + 1] - jnp.sum(kc * Sd, axis=0, keepdims=True))
        S2 = Sd + kc * vn
        outs.append(jnp.sum(qc * S2, axis=0, keepdims=True))
        states.append(S2)
    return jnp.concatenate(outs), jnp.stack(states)


def vpu_reshape(S, a, k, q, v, beta):  # columns by a plain reshape (K from lanes to sublanes)
    ac, kc, qc = a[:, :, None], k[:, :, None], q[:, :, None]
    Sd = S * ac
    vn = beta * (v - jnp.sum(kc * Sd, axis=1))
    S2 = Sd + kc * vn[:, None, :]
    return jnp.sum(qc * S2, axis=1), S2


def transposed(S, a, k, q, v, beta):  # the state as (TH, d_v, d_k): matvecs as lane reductions
    Sd = S * a[:, None, :]
    vu = jnp.sum(Sd * k[:, None, :], axis=2, keepdims=True)  # (TH, V, 1)
    vc = ftb(v)
    vn = beta[:, :, None] * (vc - vu)
    S2 = Sd + vn * k[:, None, :]
    out = jnp.sum(S2 * q[:, None, :], axis=2, keepdims=True)  # (TH, V, 1)
    return out[:, :, 0], S2


VARIANTS = {
    "mxu_default": mxu,
    "mxu_highest": functools.partial(mxu, precision=jax.lax.Precision.HIGHEST),
    "vpu_ftb": vpu_ftb,
    "vpu_ftb_once": vpu_ftb_once,
    "vpu_transpose_heads": vpu_transpose_heads,
    "vpu_reshape": vpu_reshape,
    "transposed": transposed,
}


def kernel_for(step, n):
    def kernel(s_ref, a_ref, k_ref, q_ref, v_ref, b_ref, out_ref, s_out):
        s_out[...] = s_ref[...]
        out_ref[...] = jnp.zeros_like(out_ref)

        @pl.loop(0, n)
        def _(i):
            o, s = step(s_out[...], a_ref[...], k_ref[...], q_ref[...], v_ref[...], b_ref[...])
            s_out[...] = s
            out_ref[...] += o

    return kernel


def run(name, step, th, n):
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel_for(step, n),
        out_shape=(jax.ShapeDtypeStruct((th, K), jnp.float32), jax.ShapeDtypeStruct((th, K, K), jnp.float32)),
        in_specs=[vmem] * 6,
        out_specs=(vmem, vmem),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 << 20),
        name=name,
    )
    return jax.jit(call)


def main():
    rng = np.random.default_rng(0)
    for th in (16, 32):
        S = jnp.asarray(rng.standard_normal((th, K, K)) * 0.05, jnp.float32)
        rows = [jnp.asarray(rng.random((th, K)), jnp.float32) * 0.1 for _ in range(4)]
        rows[0] = rows[0] + 0.9
        beta = jnp.asarray(rng.random((th, 1)), jnp.float32)
        args = (S, *rows, beta)
        want = None
        for name, step in VARIANTS.items():
            try:
                times = {}
                for n in (1, 101):
                    fn = run(name, step, th, n)
                    jax.block_until_ready(fn(*args))
                    samples = []
                    for _ in range(5):
                        t = time.perf_counter()
                        for _ in range(10):
                            r = fn(*args)
                        jax.block_until_ready(r)
                        samples.append((time.perf_counter() - t) / 10)
                    times[n] = sorted(samples)[2]
                per = (times[101] - times[1]) / 100
                o1 = run(name, step, th, 1)(*args)
                if name == "transposed":
                    o1 = o1  # (state layout differs; outputs comparable)
                want = o1[0] if want is None else want
                err = float(jnp.max(jnp.abs(o1[0] - want)))
                print(
                    f"TH {th:2} {name:20} {per * 1e6:8.3f} us per tile ({per * 1e6 * 32 / th:7.3f} us per sequence)"
                    f"  out diff vs first {err:.2e}",
                    flush=True,
                )
            except Exception as e:
                print(f"TH {th:2} {name:20} failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)


if __name__ == "__main__":
    main()
