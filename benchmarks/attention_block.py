"""A whole attention block (causal multi-head attention, post-attention RMSNorm, feed-forward layer) as one
autokernel kernel, tuned, against JAX's splash attention followed by RMSNorm and the feed-forward layer in
XLA. bf16, q pre-scaled for both, on a TPU v5e.

    uv run python benchmarks/attention_block.py
    uv run python benchmarks/attention_block.py --shape 4,8,2048,128,4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from attention_block import attention_block, make_inputs, splash_then_norm_and_ffn
from timing import require_tpu, time_ms

SHAPES = [  # B, H, S, head dim, F
    "4,8,2048,128,4096",
    "8,8,1024,128,4096",
    "2,16,2048,128,8192",
    "1,16,4096,128,8192",
    "4,16,2048,64,4096",
    "1,32,2048,128,16384",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="B,H,S,Dh,F (repeatable)")
    args = parser.parse_args()
    require_tpu()
    fused = jax.jit(attention_block)
    summary = []
    for b, h, s, dh, f in [tuple(map(int, shape.split(","))) for shape in args.shape or SHAPES]:
        inputs = make_inputs(b, h, s, dh, f)
        d = h * dh
        flops = 2 * b * h * s * s * dh + 4 * b * s * d * f  # causal attention (half the scores) + the FFN
        rate = lambda ms: flops / ms / 1e9  # noqa: E731, B023
        print(f"B={b} H={h} S={s} Dh={dh} F={f} (D={d}): ms (TFLOP/s)")
        default_splash = splash_then_norm_and_ffn(h, s)
        t_default = time_ms(default_splash, *inputs)
        best, best_blocks = t_default, "default"
        for bq in (256, 512, 1024):
            for bk in (256, 512, 1024):
                t = time_ms(splash_then_norm_and_ffn(h, s, min(bq, s), min(bk, s)), *inputs)
                if t < best:
                    best, best_blocks = t, f"{bq} x {bk}"
        t_ours = time_ms(fused, *inputs)  # tunes on the first call
        diff = jnp.max(jnp.abs(fused(*inputs).astype(jnp.float32) - default_splash(*inputs).astype(jnp.float32)))
        scale = jnp.max(jnp.abs(default_splash(*inputs).astype(jnp.float32)))
        print(f"  splash (default blocks) + RMSNorm + FFN     {t_default:8.3f}  ({rate(t_default):6.1f})")
        print(f"  splash ({best_blocks:>11}) + RMSNorm + FFN     {best:8.3f}  ({rate(best):6.1f})")
        print(f"  autokernel fused block, tuned               {t_ours:8.3f}  ({rate(t_ours):6.1f})")
        print(
            f"  speedup {t_default / t_ours:.2f}x over default splash, {best / t_ours:.2f}x over the best splash; "
            f"max |fused - unfused| {float(diff):.4f} (outputs up to {float(scale):.2f})"
        )
        print("    " + str(attention_block.tuning(*inputs)).replace("\n", "\n    "))
        summary.append((b, h, s, dh, f, t_default, best, t_ours))
    print("\nB  H   S     Dh  F      default splash  best splash  autokernel  vs default  vs best")
    for b, h, s, dh, f, t_default, best, t_ours in summary:
        print(
            f"{b:<2} {h:<3} {s:<5} {dh:<3} {f:<6} {t_default:14.3f} {best:12.3f} {t_ours:11.3f}"
            f" {t_default / t_ours:10.2f}x {best / t_ours:7.2f}x"
        )


if __name__ == "__main__":
    main()
