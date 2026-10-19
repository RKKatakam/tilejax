"""Causal multi-head attention: autokernel's splash attention vs JAX's splash attention kernel vs XLA.

Run on a TPU VM:
    uv run python benchmarks/attention.py
    uv run python benchmarks/attention.py --shape 2,16,8192,128 --blocks 512x512,1024x512

For every shape it times XLA (`jax.nn.dot_product_attention`, skipped if it runs out of memory) and, for
every block size, `jax.experimental.pallas.ops.tpu.splash_attention` and the autokernel kernel, checking
each output against XLA (or against JAX splash when XLA can't run). Timings off-TPU are meaningless, so
the script refuses to run there.
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


def make_autokernel_attention(B: int, H: int, S: int, D: int, TQ: int, TK: int, vmem_limit_bytes: int):
    """Causal flash attention as autokernel tile loops; future key/value blocks are skipped."""
    scale = 1.0 / math.sqrt(D)

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
                        s = (q[b, h, i : i + TQ, :] @ k[b, h, j : j + TK, :].T) * scale
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

    return jax.jit(autokernel.tpu(causal_attention, interpret=False, vmem_limit_bytes=vmem_limit_bytes))


def make_jax_splash(H: int, S: int, D: int, TQ: int, TK: int):
    """JAX's splash attention with a causal mask, vmapped over the batch. It doesn't scale q itself."""
    mask = splash_mask.MultiHeadMask([splash_mask.CausalMask((S, S)) for _ in range(H)])
    blocks = splash_kernel.BlockSizes(block_q=TQ, block_kv=TK, block_kv_compute=TK)
    kernel = splash_kernel.make_splash_mha_single_device(mask, block_sizes=blocks)
    scale = 1.0 / math.sqrt(D)
    return jax.jit(jax.vmap(lambda q, k, v: kernel(q * scale, k, v)))


def make_xla():
    """XLA's attention; takes (batch, seq, heads, dim), so the inputs are transposed outside the timing."""
    return jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla"))


def time_ms(fn, *args, runs: int = 10, repeats: int = 3) -> float:
    """Median over `repeats` of the mean time per call over `runs` calls, after a warm-up call."""
    fn(*args).block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        out.block_until_ready()
        samples.append((time.perf_counter() - start) / runs * 1e3)
    return float(np.median(samples))


def parse_blocks(text: str) -> list[tuple[int, int]]:
    return [tuple(map(int, item.split("x"))) for item in text.split(",")]  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shape",
        action="append",
        help="batch,heads,seq,head_dim (repeatable); default 4,16,4096,128 and 2,16,8192,128",
    )
    parser.add_argument(
        "--blocks", default="256x256,512x256,512x512,1024x512", help="q_block x kv_block, comma-separated"
    )
    parser.add_argument("--vmem-limit-mib", type=int, default=64)
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        raise SystemExit(f"benchmarks need a TPU (the default backend is {jax.default_backend()})")

    shapes = [tuple(map(int, s.split(","))) for s in (args.shape or ["4,16,4096,128", "2,16,8192,128"])]
    print(f"{jax.devices()[0].device_kind}, bf16, causal; ms (median of 3 x 10 runs), TFLOP/s of the causal half")
    for B, H, S, D in shapes:
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        q, k, v = (jax.random.normal(key, (B, H, S, D), jnp.bfloat16) for key in keys)
        flops = 2 * B * H * S * S * D  # two matmuls, each 2*S*S*D per head, over the causal half

        print(f"\nbatch={B} heads={H} seq={S} head_dim={D}")
        qt, kt, vt = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
        xla = make_xla()
        try:
            reference = xla(qt, kt, vt).transpose(0, 2, 1, 3).astype(jnp.float32)
            ms = time_ms(xla, qt, kt, vt)
            print(f"  {'XLA dot_product_attention':<28} {ms:9.3f} ms  {flops / ms / 1e9:6.1f} TFLOP/s")
        except jax.errors.JaxRuntimeError as e:
            reference = None
            print(f"  {'XLA dot_product_attention':<28} skipped: {str(e).splitlines()[0][:80]}")

        print(f"  {'blocks':<10} {'JAX splash':>22} {'autokernel':>22} {'speedup':>8}  max error vs reference")
        for TQ, TK in parse_blocks(args.blocks):
            ours = make_autokernel_attention(B, H, S, D, TQ, TK, args.vmem_limit_mib * 2**20)
            theirs = make_jax_splash(H, S, D, TQ, TK)
            splash_out = theirs(q, k, v).astype(jnp.float32)
            ref = reference if reference is not None else splash_out
            errors = [float(jnp.max(jnp.abs(out - ref))) for out in (splash_out, ours(q, k, v).astype(jnp.float32))]
            t_theirs, t_ours = time_ms(theirs, q, k, v), time_ms(ours, q, k, v)
            print(
                f"  {f'{TQ}x{TK}':<10} {t_theirs:9.3f} ms {flops / t_theirs / 1e9:6.1f} TF"
                f" {t_ours:9.3f} ms {flops / t_ours / 1e9:6.1f} TF {t_theirs / t_ours:7.2f}x"
                f"  splash {errors[0]:.3g}, autokernel {errors[1]:.3g}"
            )


if __name__ == "__main__":
    main()
