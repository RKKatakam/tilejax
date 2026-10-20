"""A whole decoder-only transformer (tokens -> logits): autokernel's kernels, each tuned, against the same
model in plain JAX with JAX's splash attention (at its default block sizes, and at its best over a grid),
bf16, on a TPU v5e.

    uv run python benchmarks/transformer.py
    uv run python benchmarks/transformer.py --model 4,1024,8,4096,32768 --batch 4 --seq 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from rmsnorm_matmul import rmsnorm_matmul
from timing import require_tpu, time_ms
from transformer import (
    Config,
    attention_layer,
    baseline,
    ffn_layer,
    fuse_ffn,
    init_params,
    transformer,
    transformer_layer,
)

RUNS = [  # (layers, d_model, heads, ffn, vocab), batch, seq
    ((4, 1024, 8, 4096, 32768), 4, 2048),
    ((4, 2048, 16, 8192, 32768), 2, 2048),
    ((2, 2048, 16, 8192, 32768), 1, 4096),
    ((2, 4096, 32, 16384, 32768), 1, 2048),
]


def flops(config: Config, batch: int, seq: int) -> float:
    """Model FLOPs of a forward pass: the matmuls, with causal attention computing half the scores."""
    d, f, t = config.d_model, config.ffn, batch * seq
    per_layer = 2 * t * d * 3 * d + 2 * batch * seq * seq * d + 2 * t * d * d + 4 * t * d * f
    return config.layers * per_layer + 2 * t * d * config.vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="layers,d_model,heads,ffn,vocab")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    args = parser.parse_args()
    require_tpu()
    runs = [(tuple(map(int, args.model.split(","))), args.batch, args.seq)] if args.model else RUNS
    ours = jax.jit(transformer)
    summary = []
    for model, batch, seq in runs:
        config = Config(*model)
        params = init_params(config)
        tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, config.vocab)
        rate = lambda ms: flops(config, batch, seq) / ms / 1e9  # noqa: E731, B023
        print(f"{config}, batch {batch}, seq {seq}: ms (TFLOP/s)")
        default = baseline(config, seq)
        t_default = time_ms(default, tokens, params, runs=5)
        best, best_blocks = t_default, "default"
        for bq in (512, 1024):
            for bk in (512, 1024):
                t = time_ms(baseline(config, seq, min(bq, seq), min(bk, seq)), tokens, params, runs=5)
                if t < best:
                    best, best_blocks = t, f"{bq} x {bk}"
        t_ours = time_ms(ours, tokens, params, runs=5)  # tunes each kernel on the first call
        logits, want = ours(tokens, params), default(tokens, params)
        diff = float(jnp.max(jnp.abs(logits.astype(jnp.float32) - want.astype(jnp.float32))))
        print(f"  plain JAX + splash (default blocks)  {t_default:8.3f}  ({rate(t_default):6.1f})")
        print(f"  plain JAX + splash ({best_blocks:>11})  {best:8.3f}  ({rate(best):6.1f})")
        print(f"  autokernel, tuned                    {t_ours:8.3f}  ({rate(t_ours):6.1f})")
        print(f"  speedup {t_default / t_ours:.2f}x / {best / t_ours:.2f}x; max |logits - baseline| {diff:.4f}")
        d, s, b = config.d_model, seq, batch
        x = jax.ShapeDtypeStruct((b, s, d), jnp.bfloat16)
        qkv = jax.ShapeDtypeStruct((b, s, 3 * d), jnp.bfloat16)
        layer = params["layers"][0]
        layer_args = (x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])
        rows = jax.ShapeDtypeStruct((b * s, d), jnp.bfloat16)
        if fuse_ffn(x, qkv, layer):
            print("  layer (fused):", transformer_layer.lower(*layer_args).config)
        else:
            print("  layer (split): attention", attention_layer.lower(x, qkv, layer["wo"]).config)
            print("                 ffn      ", ffn_layer.lower(rows, *layer_args[3:]).config)
        fused_ms = transformer_layer.tuning(*layer_args).best.seconds * 1e3
        attention_ms = attention_layer.tuning(x, qkv, layer["wo"]).best.seconds * 1e3
        ffn_ms = ffn_layer.tuning(rows, *layer_args[3:]).best.seconds * 1e3
        print(f"    tuned device times: fused {fused_ms:.3f} ms; split {attention_ms:.3f} + {ffn_ms:.3f} ms")
        print("  qkv:  ", rmsnorm_matmul.lower(rows, layer["attn_norm"], layer["w_qkv"]).config)
        print("  head: ", rmsnorm_matmul.lower(rows, params["final_norm"], params["w_vocab"]).config)
        summary.append((model, batch, seq, t_default, best, t_ours))
    print("\nlayers d_model heads ffn   batch seq   splash default  splash best  autokernel  vs default  vs best")
    for (layers, d, h, f, _), batch, seq, t_default, best, t_ours in summary:
        print(
            f"{layers:<6} {d:<7} {h:<5} {f:<5} {batch:<5} {seq:<5} {t_default:14.3f} {best:12.3f} {t_ours:11.3f}"
            f" {t_default / t_ours:10.2f}x {best / t_ours:7.2f}x"
        )


if __name__ == "__main__":
    main()
