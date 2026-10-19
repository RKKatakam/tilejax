"""A stack of transformer layers written as autokernel tile loops: fused vs unfused kernels vs plain XLA.

Run on a TPU VM:
    uv run python benchmarks/transformer.py                    # prefill: bf16, S=2048, D=2048, F=8192, L=4
    uv run python benchmarks/transformer.py --batch 4 --breakdown   # where the time goes, per op
    uv run python benchmarks/transformer.py --case decode      # few rows: launch overhead shows

Each layer is pre-norm causal attention and a pre-norm ReLU MLP: five tile loop nests with no JAX code
between them, written so that no transposes are needed (a head is a block of DH columns of q, k and v).
autokernel runs the program two ways, and XLA a third:

- fused:   the nests of a layer fused into kernels (at most three nests each), the layer loop a
           `jax.lax.fori_loop` (`layer` is a runtime scalar, so one kernel serves every layer);
- unfused: one kernel per nest (`fuse=False`), the same layer loop;
- XLA + splash: the same model in plain JAX with JAX's Pallas splash attention kernel for the attention;
- XLA:     the same model in plain JAX (`jax.nn.dot_product_attention`), the layers a `jax.lax.scan`.

The attention block sizes are the same for autokernel and splash: every (block_q, block_kv) in
`--attention-blocks` is run by both (splash with `block_kv_compute = block_kv`). `--splash-extra` adds
splash on its own best configuration (on a v5e 1024x1024 blocks computed 512 keys at a time, a knob the
autokernel nest has no equivalent of), so splash isn't handicapped by the matching. The matmul block sizes
can't be made equal: XLA picks its own tiling for `jnp.einsum`, autokernel uses `--tiles` / `--*-tiles`.

For prefill it also times the attention alone (the autokernel nest against JAX splash, same blocks).
`--breakdown` profiles one call of each model and prints the device time per op (from the profiler's
"XLA Ops" line), and times each projection as a plain matmul: XLA's `jnp.dot` against the autokernel
matmul nest at the transformer's tiles and at 512x512x512. `--hlo DIR` writes the compiled HLO of the
XLA + splash model there.

`--case decode` runs only the MLP half of the layers on a handful of rows: short, weight-streaming kernels
where what each launch and pipeline fill costs shows. Off a TPU, `--interpret` runs a tiny configuration
in the Pallas interpreter to check the script.

The model's sizes are module globals, set by `configure` before the program is built.
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

LANES = 128
EPS = 1e-6
MASK_VALUE = -1e30
# Set by `configure`. Each matmul nest has its own block sizes (rows, columns, reduction), so each can be
# tuned on its own: QKV projection TQ*, output projection TO*, MLP up TU*, MLP down TD*; TQ x TKV attention.
B = S = D = H = F = L = DH = 0
TQM = TQN = TQK = TOM = TON = TOK = TUM = TUN = TUK = TDM = TDN = TDK = 0
TQ = TKV = 0
PM = PN = PK = 0  # the plain matmul nest's block sizes (`--breakdown`)
SCALE = 1.0
dtype = jnp.bfloat16
f32 = jnp.float32
TILE_NAMES = ("TQM", "TQN", "TQK", "TOM", "TON", "TOK", "TUM", "TUN", "TUK", "TDM", "TDN", "TDK", "TQ", "TKV")


def configure(**sizes) -> None:
    globals().update(sizes)
    globals()["DH"] = D // H if H else D
    globals()["SCALE"] = 1.0 / math.sqrt(globals()["DH"])


def transformer(x, norm1, wq, wk, wv, wo, norm2, w_up, w_down):
    for layer in range(L):
        # RMSNorm folded into the Q, K, V projections: rmsnorm(x) @ w = inv_rms(x) * ((x * g) @ w) row by
        # row, the sum of squares accumulated (lane-wide) over the same tiles of x the matmuls read.
        q = jnp.zeros((B, S, D), dtype)
        k = jnp.zeros((B, S, D), dtype)
        v = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, TQM):
                for j in range(0, D, TQN):
                    aq = jnp.zeros((TQM, TQN))
                    ak = jnp.zeros((TQM, TQN))
                    av = jnp.zeros((TQM, TQN))
                    ss = jnp.zeros((TQM, LANES))
                    for kk in range(0, D, TQK):
                        xf = x[b, i : i + TQM, kk : kk + TQK].astype(f32)
                        ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                        xt = (xf * norm1[layer, :, kk : kk + TQK]).astype(dtype)
                        aq += xt @ wq[layer, kk : kk + TQK, j : j + TQN]
                        ak += xt @ wk[layer, kk : kk + TQK, j : j + TQN]
                        av += xt @ wv[layer, kk : kk + TQK, j : j + TQN]
                    inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TQN // LANES))
                    q = q.at[b, i : i + TQM, j : j + TQN].set((aq * inv * SCALE).astype(dtype))
                    k = k.at[b, i : i + TQM, j : j + TQN].set((ak * inv).astype(dtype))
                    v = v.at[b, i : i + TQM, j : j + TQN].set((av * inv).astype(dtype))
        # Causal flash attention; a head is a block of DH columns (q is pre-scaled).
        attn = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for hc in range(0, D, DH):
                for i in range(0, S, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)
                    row_sum = jnp.zeros((TQ, LANES))
                    acc = jnp.zeros((TQ, DH))
                    for j in range(0, S, TKV):
                        if j > i + TQ - 1:
                            continue
                        s = q[b, i : i + TQ, hc : hc + DH] @ k[b, j : j + TKV, hc : hc + DH].T
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TKV), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TKV), 1)
                        s = jnp.where(cols <= rows, s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TKV // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        pv = p.astype(dtype) @ v[b, j : j + TKV, hc : hc + DH]
                        acc = jnp.tile(alpha, (1, DH // LANES)) * acc + pv
                        row_max = m_new
                    out = acc / jnp.tile(row_sum, (1, DH // LANES))
                    attn = attn.at[b, i : i + TQ, hc : hc + DH].set(out.astype(dtype))
        # Output projection and residual.
        x1 = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, TOM):
                for j in range(0, D, TON):
                    acc = jnp.zeros((TOM, TON))
                    for kk in range(0, D, TOK):
                        acc += attn[b, i : i + TOM, kk : kk + TOK] @ wo[layer, kk : kk + TOK, j : j + TON]
                    x1 = x1.at[b, i : i + TOM, j : j + TON].set((x[b, i : i + TOM, j : j + TON] + acc).astype(dtype))
        # MLP: RMSNorm folded into the up projection, ReLU, down projection and residual.
        up = jnp.zeros((B, S, F), dtype)
        for b in range(B):
            for i in range(0, S, TUM):
                for j in range(0, F, TUN):
                    acc = jnp.zeros((TUM, TUN))
                    ss = jnp.zeros((TUM, LANES))
                    for kk in range(0, D, TUK):
                        xf = x1[b, i : i + TUM, kk : kk + TUK].astype(f32)
                        ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                        acc += (xf * norm2[layer, :, kk : kk + TUK]).astype(dtype) @ w_up[
                            layer, kk : kk + TUK, j : j + TUN
                        ]
                    inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TUN // LANES))
                    up = up.at[b, i : i + TUM, j : j + TUN].set(jnp.maximum(acc * inv, 0).astype(dtype))
        x2 = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, TDM):
                for j in range(0, D, TDN):
                    acc = jnp.zeros((TDM, TDN))
                    for kk in range(0, F, TDK):
                        acc += up[b, i : i + TDM, kk : kk + TDK] @ w_down[layer, kk : kk + TDK, j : j + TDN]
                    x2 = x2.at[b, i : i + TDM, j : j + TDN].set((x1[b, i : i + TDM, j : j + TDN] + acc).astype(dtype))
        x = x2
    return x


def attention(q, k, v):
    """The transformer's attention nest on its own (q pre-scaled), to time it alone."""
    attn = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for hc in range(0, D, DH):
            for i in range(0, S, TQ):
                row_max = jnp.full((TQ, LANES), MASK_VALUE)
                row_sum = jnp.zeros((TQ, LANES))
                acc = jnp.zeros((TQ, DH))
                for j in range(0, S, TKV):
                    if j > i + TQ - 1:
                        continue
                    s = q[b, i : i + TQ, hc : hc + DH] @ k[b, j : j + TKV, hc : hc + DH].T
                    rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TKV), 0)
                    cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TKV), 1)
                    s = jnp.where(cols <= rows, s, MASK_VALUE)
                    m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                    p = jnp.exp(s - jnp.tile(m_new, (1, TKV // LANES)))
                    alpha = jnp.exp(row_max - m_new)
                    row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                    pv = p.astype(dtype) @ v[b, j : j + TKV, hc : hc + DH]
                    acc = jnp.tile(alpha, (1, DH // LANES)) * acc + pv
                    row_max = m_new
                out = acc / jnp.tile(row_sum, (1, DH // LANES))
                attn = attn.at[b, i : i + TQ, hc : hc + DH].set(out.astype(dtype))
    return attn


def mlp_stack(x, norm2, w_up, w_down):
    """The MLP half of the layers only (the decode-like case)."""
    for layer in range(L):
        up = jnp.zeros((B, S, F), dtype)
        for b in range(B):
            for i in range(0, S, TUM):
                for j in range(0, F, TUN):
                    acc = jnp.zeros((TUM, TUN))
                    ss = jnp.zeros((TUM, LANES))
                    for kk in range(0, D, TUK):
                        xf = x[b, i : i + TUM, kk : kk + TUK].astype(f32)
                        ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                        acc += (xf * norm2[layer, :, kk : kk + TUK]).astype(dtype) @ w_up[
                            layer, kk : kk + TUK, j : j + TUN
                        ]
                    inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TUN // LANES))
                    up = up.at[b, i : i + TUM, j : j + TUN].set(jnp.maximum(acc * inv, 0).astype(dtype))
        x2 = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, TDM):
                for j in range(0, D, TDN):
                    acc = jnp.zeros((TDM, TDN))
                    for kk in range(0, F, TDK):
                        acc += up[b, i : i + TDM, kk : kk + TDK] @ w_down[layer, kk : kk + TDK, j : j + TDN]
                    x2 = x2.at[b, i : i + TDM, j : j + TDN].set((x[b, i : i + TDM, j : j + TDN] + acc).astype(dtype))
        x = x2
    return x


def matmul(a, w):
    """A plain (M, K) @ (K, N) matmul nest, bf16 out with f32 accumulation (`--breakdown`)."""
    M, K = a.shape
    N = w.shape[1]
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, PM):
        for j in range(0, N, PN):
            acc = jnp.zeros((PM, PN))
            for kk in range(0, K, PK):
                acc += a[i : i + PM, kk : kk + PK] @ w[kk : kk + PK, j : j + PN]
            out = out.at[i : i + PM, j : j + PN].set(acc.astype(dtype))
    return out


# --------------------------------------------------------------------------- the same model in plain JAX


def _dot(a, w):
    return jnp.einsum("bsd,de->bse", a, w, preferred_element_type=jnp.float32)


def _rmsnorm(x, g):
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * g).astype(x.dtype)


def _mlp(x, g, wu, wd):
    up = jnp.maximum(_dot(_rmsnorm(x, g), wu), 0).astype(x.dtype)
    return (x + _dot(up, wd)).astype(x.dtype)


def xla_transformer(x, norm1, wq, wk, wv, wo, norm2, w_up, w_down):
    def layer(x, w):
        n1, wq, wk, wv, wo, n2, wu, wd = w
        xn = _rmsnorm(x, n1)
        q, k, v = ((_dot(xn, w)).astype(x.dtype).reshape(B, S, H, DH) for w in (wq, wk, wv))
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=True).reshape(B, S, D)
        x1 = (x + _dot(attn, wo)).astype(x.dtype)
        return _mlp(x1, n2, wu, wd), None

    return jax.lax.scan(layer, x, (norm1, wq, wk, wv, wo, norm2, w_up, w_down))[0]


def make_splash(block_q: int, block_kv: int, block_kv_compute: int | None = None):
    """JAX's splash attention (causal) on (batch, heads, seq, head dim), vmapped over the batch; q pre-scaled."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask

    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((S, S)) for _ in range(H)])
    blocks = splash_kernel.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv_compute or block_kv)
    return jax.vmap(splash_kernel.make_splash_mha_single_device(mask, block_sizes=blocks))


def make_splash_transformer(block_q: int, block_kv: int, block_kv_compute: int | None = None):
    """The model in plain JAX with JAX's splash attention kernel for the attention."""
    splash = make_splash(block_q, block_kv, block_kv_compute)

    def heads(t):
        return t.reshape(B, S, H, DH).transpose(0, 2, 1, 3)

    def splash_transformer(x, norm1, wq, wk, wv, wo, norm2, w_up, w_down):
        def layer(x, w):
            n1, wq, wk, wv, wo, n2, wu, wd = w
            xn = _rmsnorm(x, n1)
            q = heads((_dot(xn, wq) * SCALE).astype(x.dtype))
            k, v = (heads(_dot(xn, w).astype(x.dtype)) for w in (wk, wv))
            attn = splash(q, k, v).transpose(0, 2, 1, 3).reshape(B, S, D)
            x1 = (x + _dot(attn, wo)).astype(x.dtype)
            return _mlp(x1, n2, wu, wd), None

        return jax.lax.scan(layer, x, (norm1, wq, wk, wv, wo, norm2, w_up, w_down))[0]

    return splash_transformer


def xla_mlp_stack(x, norm2, w_up, w_down):
    return jax.lax.scan(lambda x, w: (_mlp(x, *w), None), x, (norm2, w_up, w_down))[0]


# --------------------------------------------------------------------------- timing


def inputs(case: str, seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 9)

    def w(key, *shape):
        return (jax.random.normal(key, (L, *shape), jnp.float32) / math.sqrt(shape[0])).astype(dtype)

    x = jax.random.normal(keys[0], (B, S, D), jnp.float32).astype(dtype)
    norm = 1 + 0.1 * jax.random.normal(keys[1], (L, 1, D), jnp.float32)
    mlp = (norm, w(keys[2], D, F), w(keys[3], F, D))
    if case == "decode":
        return (x, *mlp)
    return (x, norm, w(keys[4], D, D), w(keys[5], D, D), w(keys[6], D, D), w(keys[7], D, D), *mlp)


def time_ms(fn, *args, runs: int = 10, repeats: int = 3) -> float:
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs * 1e3)
    return float(np.median(samples))


def first_call_s(fn, *args) -> float:
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def flops(case: str) -> float:
    mlp = 2 * 2 * B * S * D * F
    if case == "decode":
        return L * mlp
    proj = 4 * 2 * B * S * D * D
    attention = 2 * B * H * S * S * DH  # the causal half of q @ k.T and p @ v
    return L * (mlp + proj + attention)


def run(
    case: str,
    runs: int,
    interpret: bool,
    attention_blocks: list[tuple[int, int]],
    skip_xla: bool,
    tune: bool,
    with_unfused: bool,
    splash_extra: list[tuple[int, int, int]],
) -> None:
    """The whole model: for each attention block size, autokernel (fused, unfused) and XLA + splash with it."""
    args = inputs(case)
    fn, plain = (mlp_stack, xla_mlp_stack) if case == "decode" else (transformer, xla_transformer)
    options = dict(interpret=True) if interpret else dict(interpret=False)
    if tune:  # each matmul nest's block sizes, ranked by the cost model, the best timed on the TPU; the attention
        # keeps --attention-blocks, the block sizes splash runs with too
        names = [name for name in TILE_NAMES if name not in ("TQ", "TKV")]
        options.update(tune={name: "auto" for name in names}, autotune="model" if interpret else "measure")
    with_splash = case == "prefill" and not interpret
    if case == "decode" or interpret:
        attention_blocks = [(TQ, TKV)]
    work = flops(case)
    results, kernels, tuned = {}, {}, []

    def measure(name, f):
        try:
            compile_s = first_call_s(f, *args)
            ms = time_ms(f, *args, runs=runs, repeats=1 if interpret else 3)
            results[name] = (compile_s, ms, f(*args))
        except Exception as e:  # e.g. XLA's attention running out of HBM at long sequences
            results[name] = (None, None, f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")

    for bq, bkv in attention_blocks:
        # The kernels read TQ and TKV when they are first lowered, so each block size is built and run in turn.
        configure(TQ=bq, TKV=bkv)
        blocks = "" if case == "decode" else f" {bq}x{bkv}"
        programs = [(f"autokernel fused{blocks}", autokernel.tpu(fn, **options))]
        if with_unfused:
            programs.append((f"autokernel unfused{blocks}", autokernel.tpu(fn, fuse=False, **options)))
        for name, program in programs:
            if tune and not interpret:  # autotune="measure" times its candidates on concrete arrays: call eagerly first
                jax.block_until_ready(program(*args))
            measure(name, jax.jit(program))  # timed under jit, as the XLA models are: no per-call Python work
            if results[name][1] is not None:
                kernels[name] = len(program.lower(*args))
                if tune and "unfused" in name:
                    tuned = [(n, k.overrides, k.buffers) for n, k in program.lower(*args)]
        if with_splash:
            measure(f"XLA + splash{blocks}", jax.jit(make_splash_transformer(bq, bkv)))
    if with_splash:  # splash on its own best block sizes, which autokernel's attention has no equivalent of
        for bq, bkv, bkvc in splash_extra:
            measure(f"XLA + splash {bq}x{bkv}/{bkvc}", jax.jit(make_splash_transformer(bq, bkv, bkvc)))
    if not skip_xla:
        measure("XLA", jax.jit(plain))
    ok = {n: r for n, r in results.items() if r[1] is not None}
    ref_name = "XLA" if "XLA" in ok else next((n for n in ok if n.startswith("XLA + splash")), next(iter(ok)))
    want = ok[ref_name][2].astype(jnp.float32)
    print(f"  {'':28} {'ms':>9} {'TFLOP/s':>8} {'first call s':>13} {'kernels':>8}  max |error| vs {ref_name}")
    for name, (compile_s, ms, out) in results.items():
        if ms is None:
            print(f"  {name:28} failed: {out}", flush=True)
            continue
        err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - want)))
        k = kernels.get(name, "")
        print(f"  {name:28} {ms:9.3f} {work / ms / 1e9:8.1f} {compile_s:13.1f} {k!s:>8}  {err:.3g}", flush=True)
    for name, overrides, buffers in tuned:
        print(f"  tuned {name}: {overrides}, {buffers} buffers")
    if with_splash:
        for bq, bkv in attention_blocks:
            ours, theirs = ok.get(f"autokernel fused {bq}x{bkv}"), ok.get(f"XLA + splash {bq}x{bkv}")
            if ours and theirs:
                print(f"  attention blocks {bq}x{bkv} for both: autokernel fused {ours[1]:.3f} ms, XLA + splash "
                      f"{theirs[1]:.3f} ms: {theirs[1] / ours[1]:.3f}x, {theirs[1] - ours[1]:+.3f} ms")  # fmt: skip
        attention_alone(runs, attention_blocks, splash_extra)


def attention_alone(runs: int, attention_blocks: list[tuple[int, int]], splash_extra=()) -> None:
    """The attention on its own at the transformer's shapes: the autokernel nest vs JAX splash, same blocks."""
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    q, k, v = (jax.random.normal(key, (B, S, D), jnp.float32).astype(dtype) for key in keys)
    q = (q * SCALE).astype(dtype)
    qh, kh, vh = (t.reshape(B, S, H, DH).transpose(0, 2, 1, 3) for t in (q, k, v))
    work = 2 * B * H * S * S * DH  # q @ k.T and p @ v over the causal half
    print(f"  attention alone (B={B} H={H} S={S} DH={DH}); autokernel on (B, S, D), splash on (B, H, S, DH):")
    for bq, bkv in attention_blocks:
        configure(TQ=bq, TKV=bkv)
        ours = jax.jit(autokernel.tpu(attention, interpret=False))
        want = ours(q, k, v).astype(jnp.float32)
        t_ours = time_ms(ours, q, k, v, runs=runs)
        splash = jax.jit(make_splash(bq, bkv))
        t_splash = time_ms(splash, qh, kh, vh, runs=runs)
        err = float(jnp.max(jnp.abs(splash(qh, kh, vh).transpose(0, 2, 1, 3).reshape(B, S, D) - want)))
        print(f"    {bq}x{bkv}: autokernel {t_ours:7.3f} ms ({work / t_ours / 1e9:5.1f} TFLOP/s), JAX splash "
              f"{t_splash:7.3f} ms ({work / t_splash / 1e9:5.1f} TFLOP/s): {t_splash / t_ours:.2f}x, "
              f"{L} layers: {L * (t_splash - t_ours):+.3f} ms; max |diff| {err:.3g}", flush=True)  # fmt: skip
    for bq, bkv, bkvc in splash_extra:
        t_splash = time_ms(jax.jit(make_splash(bq, bkv, bkvc)), qh, kh, vh, runs=runs)
        print(f"    JAX splash {bq}x{bkv}, block_kv_compute {bkvc}: {t_splash:7.3f} ms "
              f"({work / t_splash / 1e9:5.1f} TFLOP/s)", flush=True)  # fmt: skip


# --------------------------------------------------------------------------- where the time goes


def op_category(text: str) -> str:
    """What an HLO instruction (as the profiler names it: its HLO text) does, for grouping device time."""
    name = text.split(" = ")[0].lstrip("%")
    if " while(" in text:
        return ""  # the layer loop: a container of the other ops, not counted
    if "tpu_custom_call" in text:
        if name.startswith("splash"):
            return "attention (JAX splash kernel)"
        parts = {"_q_attn_x1": "qkv+attention+out (one fused autokernel kernel)", "_attn": "attention (autokernel)",
                 "_q": "norm1+qkv (autokernel)", "_x1": "out projection+residual (autokernel)",
                 "_up": "norm2+up+relu (autokernel)", "_x2": "down projection+residual (autokernel)"}  # fmt: skip
        return next((v for k, v in parts.items() if name.startswith(f"transformer{k}")), f"autokernel {name}")
    if "kind=kOutput" in text or "convolution" in name:
        return "matmul fusion (XLA)"
    if " copy(" in text or " transpose(" in text:
        return "layout copy/transpose (XLA)"
    if " fusion(" in text:
        return "elementwise/reduction fusion (XLA)"
    return "other"


def device_ops(fn, *args, runs: int = 3) -> tuple[dict[str, list], float]:
    """Device time per op of `fn(*args)`, from a profile: {op: [ms per call, count per call, category, shape]}."""
    from jax.profiler import ProfileData

    jax.block_until_ready(fn(*args))
    directory = tempfile.mkdtemp()
    with jax.profiler.trace(directory):
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
    path = glob.glob(os.path.join(directory, "**", "*.xplane.pb"), recursive=True)[0]
    ops: dict[str, list] = {}
    module_ns = 0
    for plane in ProfileData.from_file(path).planes:
        if not plane.name.startswith("/device:TPU:0"):
            continue
        for line in plane.lines:
            if line.name == "XLA Modules":
                module_ns += sum(event.duration_ns for event in line.events)
            if line.name != "XLA Ops":
                continue
            for event in line.events:  # the event name is the instruction's HLO text
                name, _, rest = event.name.partition(" = ")
                category = op_category(event.name)
                if not category:
                    continue
                entry = ops.setdefault(name.lstrip("%"), [0.0, 0, category, rest.split(" ")[0][:48]])
                entry[0] += event.duration_ns / 1e6 / runs
                entry[1] += 1 / runs
    return ops, module_ns / 1e6 / runs


def print_ops(title: str, ops: dict[str, list], module_ms: float, top: int = 16) -> None:
    total = sum(entry[0] for entry in ops.values())
    print(f"  {title}: {module_ms:.3f} ms on the device ({total:.3f} ms in ops)")
    categories = collections.defaultdict(float)
    for ms, _, category, _ in ops.values():
        categories[category] += ms
    for category, ms in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {ms:8.3f} ms  {category}")
    print("    largest ops:")
    for name, (ms, count, category, shape) in sorted(ops.items(), key=lambda x: -x[1][0])[:top]:
        if ms >= 0.005:
            print(f"      {ms:8.3f} ms  x{count:<3.0f} {name:40} {shape:48} {category}")


def plain_matmuls(runs: int, tiles: dict[str, int]) -> None:
    """Each projection as a plain bf16 matmul (f32 accumulation): XLA vs the autokernel nest."""
    global PM, PN, PK
    shapes = {"qkv": (D, 3 * D, "TQ"), "out": (D, D, "TO"), "up": (D, F, "TU"), "down": (F, D, "TD")}
    print(f"  plain matmuls, M = B*S = {B * S} rows (ms; per layer):")
    xla = jax.jit(lambda a, w: jnp.dot(a, w, preferred_element_type=jnp.float32).astype(dtype))
    for name, (K, N, prefix) in shapes.items():
        keys = jax.random.split(jax.random.PRNGKey(2), 2)
        a = jax.random.normal(keys[0], (B * S, K), jnp.float32).astype(dtype)
        w = (jax.random.normal(keys[1], (K, N), jnp.float32) / math.sqrt(K)).astype(dtype)
        want = xla(a, w).astype(jnp.float32)
        line = [f"XLA {time_ms(xla, a, w, runs=runs):7.3f}"]
        ours = tuple(tiles[prefix + d] for d in "MNK")
        if name == "qkv":  # the nest's three (TQM, TQN) accumulators: one (TQM, 3 * TQN) tile of x @ [wq|wk|wv]
            ours = (ours[0], 3 * ours[1], ours[2])
        for pm, pn, pk in (ours, (512, 512, 512)):
            PM, PN, PK = min(pm, B * S), min(pn, N), min(pk, K)
            try:
                kernel = jax.jit(autokernel.tpu(matmul, interpret=False))
                err = float(jnp.max(jnp.abs(kernel(a, w).astype(jnp.float32) - want)))
                line.append(f"autokernel {PM}x{PN}x{PK} {time_ms(kernel, a, w, runs=runs):7.3f} (err {err:.2g})")
            except Exception as e:
                line.append(f"autokernel {PM}x{PN}x{PK} failed: {type(e).__name__}")
        work = 2 * B * S * K * N
        print(f"    {name:5} {B * S}x{K}x{N} ({work / 1e9:.0f} GFLOP): " + ", ".join(line), flush=True)


def breakdown(runs: int, attention_blocks: list[tuple[int, int]], tiles: dict[str, int], hlo_dir: str | None) -> None:
    args = inputs("prefill")
    print(f"\n  breakdown at B={B} S={S} (device time per call of the whole {L}-layer model, per op):")
    for bq, bkv in attention_blocks:
        configure(TQ=bq, TKV=bkv)
        splash = jax.jit(make_splash_transformer(bq, bkv))
        for title, fn in (
            (f"autokernel unfused, attention {bq}x{bkv}", jax.jit(autokernel.tpu(transformer, fuse=False))),
            (f"autokernel fused, attention {bq}x{bkv}", jax.jit(autokernel.tpu(transformer))),
            (f"XLA + splash, attention {bq}x{bkv}", splash),
        ):
            print_ops(title, *device_ops(fn, *args))
        if hlo_dir:
            os.makedirs(hlo_dir, exist_ok=True)
            path = os.path.join(hlo_dir, f"xla_splash_B{B}_{bq}x{bkv}.hlo.txt")
            with open(path, "w") as f:
                f.write(splash.lower(*args).compile().as_text())
            print(f"  wrote {path}")
    plain_matmuls(runs, tiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--batch", default="1,4", help="batch sizes (prefill) or rows (decode), comma-separated")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048, help="model width D")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=8192, help="MLP width F")
    parser.add_argument("--layers", type=int, default=4)
    # Defaults: the best of a sweep on a v5e (bf16, S=2048, D=2048, F=8192).
    parser.add_argument("--tiles", default="1024,1024,1024", help="every matmul's block sizes: rows,columns,reduction")
    defaults = {"qkv": "1024,1024,512", "up": "1024,2048,2048"}
    for nest in ("qkv", "out", "up", "down"):
        parser.add_argument(f"--{nest}-tiles", default=defaults.get(nest), help=f"the {nest} projection's block sizes")
    parser.add_argument("--tune", action="store_true", help="tune every nest's block sizes (timed on the TPU)")
    parser.add_argument(
        "--attention-blocks",
        default="512,512;1024,1024",
        help="block_q,block_kv pairs, ;-separated: each is run by both the autokernel attention and JAX splash",
    )
    parser.add_argument(
        "--splash-extra",
        default="1024,1024,512",
        help="splash-only block_q,block_kv,block_kv_compute configs, ;-separated (splash's best on a v5e); '' for none",
    )
    parser.add_argument("--no-unfused", action="store_true", help="skip the unfused autokernel program")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--skip-xla", action="store_true", help="skip plain XLA (its attention is O(S^2) memory)")
    parser.add_argument("--breakdown", action="store_true", help="device time per op, and the matmuls alone")
    parser.add_argument("--hlo", metavar="DIR", help="with --breakdown: write the XLA + splash model's compiled HLO")
    parser.add_argument("--interpret", action="store_true", help="tiny sizes in the Pallas interpreter")
    args = parser.parse_args()
    interpret = args.interpret
    if not interpret and jax.default_backend() != "tpu":
        raise SystemExit(f"benchmarks need a TPU (the default backend is {jax.default_backend()}); try --interpret")
    tiles = {}
    for nest, prefix in (("qkv", "TQ"), ("out", "TO"), ("up", "TU"), ("down", "TD")):
        values = getattr(args, f"{nest}_tiles") or args.tiles
        tiles.update(zip((f"{prefix}M", f"{prefix}N", f"{prefix}K"), map(int, values.split(",")), strict=True))
    attention_blocks = [tuple(map(int, b.split(","))) for b in args.attention_blocks.split(";")]
    splash_extra = [tuple(map(int, b.split(","))) for b in args.splash_extra.split(";") if b]
    tq, tkv = attention_blocks[0]
    device = jax.devices()[0].device_kind
    for n in map(int, args.batch.split(",")):
        if interpret:
            sizes = dict(B=1, S=256, D=256, H=2, F=512, L=2, TQ=128, TKV=128)
            sizes.update({k: 128 for k in tiles})
        elif args.case == "decode":  # n rows of one sequence
            sizes = dict(B=1, S=n, D=args.width, H=1, F=args.hidden, L=args.layers, TQ=tq, TKV=tkv)
            sizes.update({k: min(v, n) if k.endswith("M") else v for k, v in tiles.items()})
        else:
            sizes = dict(B=n, S=args.seq, D=args.width, H=args.heads, F=args.hidden, L=args.layers, TQ=tq, TKV=tkv)
            sizes.update(tiles)
        configure(**sizes)
        shown = {k: v for k, v in sizes.items() if k not in ("TQ", "TKV")}
        print(f"\n{device} {args.case}: " + " ".join(f"{k}={v}" for k, v in shown.items()), flush=True)
        run(
            args.case,
            args.runs,
            interpret,
            attention_blocks,
            args.skip_xla,
            args.tune,
            not args.no_unfused,
            splash_extra,
        )
        if args.breakdown and args.case == "prefill" and not interpret:
            breakdown(args.runs, attention_blocks, tiles, args.hlo)


if __name__ == "__main__":
    main()
