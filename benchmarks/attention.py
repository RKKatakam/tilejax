"""Causal multi-head attention: autokernel vs JAX's splash attention kernel at identical block sizes.

Run on a TPU VM:
    uv run python benchmarks/attention.py                                   # the README grid
    uv run python benchmarks/attention.py --shape 4,16,2048,128 --sizes 512,1024
    uv run python benchmarks/attention.py --splash-variants                 # also try to beat our splash setup

For every shape and every (block_q, block_kv) in `--sizes` x `--sizes` it times, on the same inputs:

- `splash`: `jax.experimental.pallas.ops.tpu.splash_attention` with a causal mask and
  `BlockSizes(block_q=TQ, block_kv=TK, block_kv_compute=TK)` (the backward block sizes don't affect the
  forward pass), on (batch, heads, seq, head_dim), vmapped over the batch;
- `ak heads`: the autokernel nest (as in `examples/splash_attention.py`) on the same (B, H, S, D) arrays;
- `ak seq`: the transformer's autokernel nest on (B, S, H*D) arrays, a head being a block of D columns
  (no transposes between the projections and the attention), the layout `benchmarks/transformer.py` uses.

Everything gets q already scaled by 1/sqrt(D), so no kernel pays for the scaling and no XLA op is timed
with a kernel. `--splash-variants` also times splash with its own default block sizes (128s), with
`block_kv_compute` < `block_kv`, with the batch folded into the heads instead of vmapped, and with k in
sequence-minor layout. Off a TPU the timings are meaningless, so the script refuses to run there.
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask

import autokernel

LANES = 128
MASK_VALUE = -1e30


def make_autokernel_heads(B: int, H: int, S: int, D: int, TQ: int, TK: int, vmem_limit_bytes: int | None, **options):
    """Causal flash attention on (B, H, S, D) as autokernel tile loops (q pre-scaled); future blocks are skipped."""

    def causal_attention(q, k, v):
        dtype = q.dtype
        out = jnp.zeros((B, H, S, D), dtype)
        for b in range(0, B, 1):
            for h in range(0, H, 1):
                for i in range(0, S, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)  # lane-wide running row max
                    row_sum = jnp.zeros((TQ, LANES))  # lane-wide running row sum
                    acc = jnp.zeros((TQ, D))
                    for j in range(0, S, TK):
                        if j > i + TQ - 1:  # the whole key/value block is in the future: skip it
                            continue
                        s = q[b, h, i : i + TQ, :] @ k[b, h, j : j + TK, :].T
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                        s = jnp.where(cols <= rows, s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + TK, :]
                        row_max = m_new
                    out = out.at[b, h, i : i + TQ, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
        return out

    return autokernel.tpu(causal_attention, interpret=False, vmem_limit_bytes=vmem_limit_bytes, **options)


def make_autokernel_seq(B: int, H: int, S: int, D: int, TQ: int, TK: int, vmem_limit_bytes: int | None, **options):
    """The transformer's nest: (B, S, H*D) arrays, a head a block of D columns (q pre-scaled)."""
    W = H * D

    def causal_attention(q, k, v):
        dtype = q.dtype
        out = jnp.zeros((B, S, W), dtype)
        for b in range(B):
            for hc in range(0, W, D):
                for i in range(0, S, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)
                    row_sum = jnp.zeros((TQ, LANES))
                    acc = jnp.zeros((TQ, D))
                    for j in range(0, S, TK):
                        if j > i + TQ - 1:
                            continue
                        s = q[b, i : i + TQ, hc : hc + D] @ k[b, j : j + TK, hc : hc + D].T
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                        s = jnp.where(cols <= rows, s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, j : j + TK, hc : hc + D]
                        row_max = m_new
                    out = out.at[b, i : i + TQ, hc : hc + D].set(
                        (acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype)
                    )
        return out

    return autokernel.tpu(causal_attention, interpret=False, vmem_limit_bytes=vmem_limit_bytes, **options)


def make_splash(B: int, H: int, S: int, TQ: int, TK: int, TKC: int | None = None, fold_batch=False, k_seq_minor=False):
    """JAX splash attention (causal) on (B, H, S, D), q pre-scaled: vmapped over B, or B folded into the heads."""
    heads = B * H if fold_batch else H
    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((S, S)) for _ in range(heads)])
    layout = splash_kernel.QKVLayout.SEQ_MINOR if k_seq_minor else splash_kernel.QKVLayout.HEAD_DIM_MINOR
    blocks = splash_kernel.BlockSizes(block_q=TQ, block_kv=TK, block_kv_compute=TKC or TK, k_layout=layout)
    kernel = splash_kernel.make_splash_mha_single_device(mask, block_sizes=blocks)
    if fold_batch:
        return jax.jit(lambda q, k, v: kernel(*(t.reshape(heads, S, -1) for t in (q, k, v))).reshape(q.shape))
    return jax.jit(jax.vmap(kernel))


def make_splash_default(B: int, H: int, S: int):
    """Splash with its own default block sizes (`BlockSizes.get_default()`, all 128)."""
    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((S, S)) for _ in range(H)])
    return jax.jit(jax.vmap(splash_kernel.make_splash_mha_single_device(mask)))


def make_xla():
    """XLA's attention on (B, S, H, D); q is pre-scaled, so the scale is 1."""
    return jax.jit(
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=1.0, implementation="xla")
    )


def time_ms(fn, *args, runs: int = 10, repeats: int = 3) -> float:
    """Median over `repeats` of the mean time per call over `runs` calls, after a warm-up call."""
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs * 1e3)
    return float(np.median(samples))


def measure(fn, args, reference, to_heads=lambda x: x, runs: int = 10):
    """(ms, max |error| vs reference), or (None, message) if it doesn't compile or run."""
    try:
        out = to_heads(fn(*args)).astype(jnp.float32)
        err = float(jnp.max(jnp.abs(out - reference))) if reference is not None else float("nan")
        return time_ms(fn, *args, runs=runs), err
    except Exception as e:  # e.g. a block size that doesn't fit in VMEM
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"


def print_matrix(title: str, sizes: list[int], cell) -> None:
    print(f"\n{title}\n")
    print("| block_q \\ block_kv | " + " | ".join(str(tk) for tk in sizes) + " |")
    print("| --- | " + " | ".join("---" for _ in sizes) + " |")
    for tq in sizes:
        print(f"| {tq} | " + " | ".join(cell(tq, tk) for tk in sizes) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shape",
        action="append",
        help="batch,heads,seq,head_dim (repeatable); default 4,16,2048,128 and 1,16,4096,128",
    )
    parser.add_argument("--sizes", default="256,512,1024", help="block sizes; every (block_q, block_kv) pair is run")
    parser.add_argument("--kernels", default="splash,ak_heads,ak_seq", help="which of splash, ak_heads, ak_seq")
    parser.add_argument("--splash-variants", action="store_true", help="also try other splash configurations")
    parser.add_argument("--skip-xla", action="store_true")
    parser.add_argument("--vmem-limit-mib", type=int, default=0, help="autokernel's VMEM limit (0: its own estimate)")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        raise SystemExit(f"benchmarks need a TPU (the default backend is {jax.default_backend()})")
    vmem = args.vmem_limit_mib * 2**20 or None
    sizes = [int(s) for s in args.sizes.split(",")]
    kernels = args.kernels.split(",")

    shapes = [tuple(map(int, s.split(","))) for s in (args.shape or ["4,16,2048,128", "1,16,4096,128"])]
    print(f"{jax.devices()[0].device_kind}, bf16, causal, q pre-scaled; ms (median of 3 x {args.runs} runs)")
    for B, H, S, D in shapes:
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        q, k, v = (jax.random.normal(key, (B, H, S, D), jnp.float32) for key in keys)
        q = q / math.sqrt(D)
        q, k, v = (t.astype(jnp.bfloat16) for t in (q, k, v))
        seq = tuple(t.transpose(0, 2, 1, 3).reshape(B, S, H * D) for t in (q, k, v))  # (B, S, H*D)
        bshd = tuple(t.transpose(0, 2, 1, 3) for t in (q, k, v))  # XLA's (B, S, H, D)

        def from_seq(out):
            return out.reshape(B, S, H, D).transpose(0, 2, 1, 3)

        flops = 2 * B * H * S * S * D  # two matmuls, each 2*S*S*D per head, over the causal half
        print(f"\n## batch={B} heads={H} seq={S} head_dim={D}  (TFLOP/s = {flops / 1e9:.0f} / ms)")
        reference = None
        if not args.skip_xla:
            ms, err = measure(make_xla(), bshd, None, runs=args.runs)
            if ms is not None:
                reference = make_xla()(*bshd).transpose(0, 2, 1, 3).astype(jnp.float32)
                print(f"XLA dot_product_attention: {ms:.3f} ms")
            else:
                print(f"XLA dot_product_attention: {err}")

        results: dict[tuple[str, int, int], tuple] = {}
        for tq in sizes:
            for tk in sizes:
                if "splash" in kernels:
                    results["splash", tq, tk] = measure(
                        make_splash(B, H, S, tq, tk), (q, k, v), reference, runs=args.runs
                    )
                if reference is None and ("splash", tq, tk) in results and results["splash", tq, tk][0] is not None:
                    reference = make_splash(B, H, S, tq, tk)(q, k, v).astype(jnp.float32)
                if "ak_heads" in kernels:
                    fn = jax.jit(make_autokernel_heads(B, H, S, D, tq, tk, vmem))  # no per-call Python work
                    results["ak_heads", tq, tk] = measure(fn, (q, k, v), reference, runs=args.runs)
                if "ak_seq" in kernels:
                    fn = jax.jit(make_autokernel_seq(B, H, S, D, tq, tk, vmem))
                    results["ak_seq", tq, tk] = measure(fn, seq, reference, from_seq, runs=args.runs)
                line = ", ".join(
                    f"{name} {r[0]:.3f} ms (err {r[1]:.3g})" if r[0] is not None else f"{name} FAILED {r[1]}"
                    for (name, a, b), r in results.items()
                    if (a, b) == (tq, tk)
                )
                print(f"  {tq}x{tk}: {line}", flush=True)

        def ms_cell(name):
            def cell(tq, tk):
                r = results.get((name, tq, tk))
                return "-" if r is None else ("fail" if r[0] is None else f"{r[0]:.3f}")

            return cell

        def ratio_cell(name):
            def cell(tq, tk):
                a, b = results.get(("splash", tq, tk)), results.get((name, tq, tk))
                if a is None or b is None or a[0] is None or b[0] is None:
                    return "-"
                return f"{a[0] / b[0]:.2f}x"

            return cell

        titles = {"splash": "JAX splash", "ak_heads": "autokernel, (B, H, S, D)", "ak_seq": "autokernel, (B, S, H*D)"}
        for name in kernels:
            print_matrix(f"{titles[name]}, ms, B={B} H={H} S={S} D={D}", sizes, ms_cell(name))
        for name in kernels:
            if name != "splash" and "splash" in kernels:
                print_matrix(f"splash ms / {titles[name]} ms (>1: autokernel faster)", sizes, ratio_cell(name))
        best = {}
        for name in kernels:
            ok = [(r[0], tq, tk) for (n, tq, tk), r in results.items() if n == name and r[0] is not None]
            if ok:
                best[name] = min(ok)
                ms, tq, tk = best[name]
                print(f"best {titles[name]}: {ms:.3f} ms at {tq}x{tk} ({flops / ms / 1e9:.0f} TFLOP/s)")

        if args.splash_variants:
            print("\nsplash variants (ms; err vs reference):")
            variants = [("defaults (128x128, compute 128)", make_splash_default(B, H, S))]
            for tq in sizes:
                for tk in sizes:
                    for tkc in (128, 256, 512):
                        if tkc < tk:
                            variants.append((f"{tq}x{tk}, block_kv_compute {tkc}", make_splash(B, H, S, tq, tk, tkc)))
            for tq, tk in ((512, 512), (1024, 512), (1024, 1024)):
                if tq in sizes and tk in sizes:
                    variants.append(
                        (f"{tq}x{tk}, batch folded into heads", make_splash(B, H, S, tq, tk, fold_batch=True))
                    )
                    variants.append((f"{tq}x{tk}, k sequence-minor", make_splash(B, H, S, tq, tk, k_seq_minor=True)))
            for name, fn in variants:
                ms, err = measure(fn, (q, k, v), reference, runs=args.runs)
                print(
                    f"  {name:44} " + (f"{ms:8.3f} ms  err {err:.3g}" if ms is not None else f"FAILED {err}"),
                    flush=True,
                )


if __name__ == "__main__":
    main()
