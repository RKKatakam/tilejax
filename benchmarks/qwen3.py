"""Qwen3-like dense decoders (prefill: tokens -> logits): autokernel's kernels, each tuned, against the same
model in plain JAX with JAX's splash attention for GQA (its MQA kernel per kv head or MHA with repeated kv,
whichever is faster; at splash's default block sizes, and at its best over a grid), bf16, on a TPU v5e.
Per layer (the QKV kernel + attention + feed-forward layer, against one plain-JAX + splash layer) and the
whole model.

    uv run python benchmarks/qwen3.py
    uv run python benchmarks/qwen3.py --model 4,1024,16,8,3072 --batch 4 --seq 2048
    uv run python benchmarks/qwen3.py --model 4,2048,16,8,6144 --batch 1 --seq 4096 --window 1024
    uv run python benchmarks/qwen3.py --model 4,2048,16,4,6144 --batch 2 --seq 2048 --next   # gated, partial RoPE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gqa_attention import make_attention
from qwen3 import (
    Config,
    baseline,
    baseline_layer,
    decoder_layer,
    ffn_weights,
    forward,
    fuse_ffn,
    init_params,
    make_qkv,
    qk_weights,
    rope_tables,
    with_fused_projections,
)
from swiglu import swiglu_ffn
from timing import require_tpu, time_ms

VOCAB = 32768
RUNS = [  # (layers, d_model, heads, kv_heads, ffn), batch, seq, window, Qwen3-Next-style attention
    ((4, 1024, 16, 8, 3072), 4, 2048, None, False),
    ((4, 2048, 16, 8, 6144), 2, 2048, None, False),
    ((2, 2560, 32, 8, 9728), 2, 2048, None, False),
    ((2, 4096, 32, 8, 12288), 1, 4096, None, False),
    ((4, 2048, 16, 8, 6144), 1, 4096, 1024, False),  # sliding-window attention
    ((4, 2048, 16, 4, 6144), 2, 2048, None, True),  # gated attention, partial RoPE
]
BLOCKS = (512, 1024)  # splash's block_q and block_kv grid


def layer_flops(config: Config, batch: int, seq: int) -> float:
    """Model FLOPs of one layer: the matmuls, attention counting only the visible (causal, windowed) scores."""
    d, f, t = config.d_model, config.ffn, batch * seq
    hd, kd = config.heads * config.head_dim, config.kv_heads * config.head_dim
    window = config.window or seq
    visible = sum(min(i + 1, window) for i in range(seq))  # keys per query, summed
    projections = 2 * t * d * (hd + 2 * kd + (hd if config.gated else 0)) + 2 * t * hd * d
    return projections + 4 * batch * visible * hd + 6 * t * d * f


def model_flops(config: Config, batch: int, seq: int) -> float:
    return config.layers * layer_flops(config, batch, seq) + 2 * batch * seq * config.d_model * config.vocab


def describe_kernels(config: Config, params: dict[str, Any], batch: int, seq: int) -> None:
    """The schedules tuning chose for a layer's kernels, and their tuned device times."""
    b16 = jnp.bfloat16
    x = jax.ShapeDtypeStruct((batch, seq, config.d_model), b16)
    layer = params["layers"][0]
    cos, sin = rope_tables(jnp.arange(seq), config.head_dim, config.rope_theta, config.rotary_dim)
    hd, kd = config.heads * config.head_dim, config.kv_heads * config.head_dim
    qkv = tuple(jax.ShapeDtypeStruct((batch, seq, n), b16) for n in (hd, kd, kd) + ((hd,) if config.gated else ()))
    kernels = make_qkv(config.qk_norm, config.rotary_dim)
    projection = kernels.gated_projection if config.gated else kernels.projection
    weights = (layer["w_q"], layer["w_k"], layer["w_v"]) + ((layer["w_attn_gate"],) if config.gated else ())
    tuning = projection.tuning(x, layer["attn_norm"], *weights, *qk_weights(layer, config), cos, sin)
    print(f"    qkv:       {tuning.config}  {tuning.best.seconds * 1e3:.3f} ms")
    attention = make_attention(config.head_dim, config.window)
    fused_kernel, attention_kernel = (
        (attention.gated_layer, attention.gated_attention) if config.gated else (attention.layer, attention.attention)
    )
    fused = fused_kernel.tuning(x, *qkv, layer["wo"], *ffn_weights(layer))
    split_attention = attention_kernel.tuning(x, *qkv, layer["wo"])
    rows = jax.ShapeDtypeStruct((batch * seq, config.d_model), b16)
    split_ffn = swiglu_ffn.tuning(rows, *ffn_weights(layer))
    form = "fused" if fuse_ffn(x, qkv, layer, config) else "split"
    print(f"    layer ({form}): fused {fused.config}  {fused.best.seconds * 1e3:.3f} ms")
    print(f"      split: attention {split_attention.config}  {split_attention.best.seconds * 1e3:.3f} ms")
    print(f"             ffn       {split_ffn.config}  {split_ffn.best.seconds * 1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="layers,d_model,heads,kv_heads,ffn (head_dim 128, vocab 32768)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--window", type=int, help="sliding-window attention over this many keys")
    parser.add_argument("--next", action="store_true", help="Qwen3-Next's attention: gated, a quarter of RoPE")
    parser.add_argument("--interpret", action="store_true", help=argparse.SUPPRESS)  # a tiny smoke run on CPU
    args = parser.parse_args()
    if args.interpret:
        runs, vocab, runs_per_timing = [((1, 256, 4, 2, 512), 1, 256, args.window, args.next)], 512, 1
    else:
        require_tpu()
        runs = (
            [(tuple(map(int, args.model.split(","))), args.batch, args.seq, args.window, args.next)]
            if args.model
            else RUNS
        )
        vocab, runs_per_timing = VOCAB, 5

    def timed(fn: Any, *inputs: Any) -> float:
        return time_ms(fn, *inputs, runs=runs_per_timing, repeats=1 if args.interpret else 5)

    ours = jax.jit(forward)
    summary = []
    for (layers, d, heads, kv_heads, ffn), batch, seq, window, next_style in runs:
        rotary = 0.25 if next_style else 1.0
        config = Config(
            layers, d, heads, kv_heads, 128, ffn, vocab, window=window, gated=next_style, rotary_fraction=rotary
        )
        params = init_params(config)
        tokens = jax.random.randint(jax.random.key(1), (batch, seq), 0, vocab)
        rate = lambda ms: model_flops(config, batch, seq) / ms / 1e9  # noqa: E731, B023
        print(f"{config}, batch {batch}, seq {seq}: ms (TFLOP/s)")

        # The baseline: splash's GQA forms, with separate or side-by-side q, k, v weights, at splash's default
        # blocks; then the fastest of those over a grid of blocks.
        fused_params = with_fused_projections(params)
        t_default, form, base_params = min(
            (timed(baseline(config, seq, form=f, interpret=args.interpret), tokens, p), f, p)
            for f in ("mqa", "mha")
            for p in (params, fused_params)
        )
        weights = "fused" if base_params is fused_params else "separate"
        best, best_blocks = t_default, (None, None)
        for bq in BLOCKS:
            for bk in BLOCKS:
                blocks = (min(bq, seq), min(bk, seq))
                t = timed(baseline(config, seq, *blocks, form=form, interpret=args.interpret), tokens, base_params)
                if t < best:
                    best, best_blocks = t, blocks
        t_ours = timed(ours, tokens, params)  # tunes each kernel on the first call
        logits = ours(tokens, params)
        want = baseline(config, seq, form=form, interpret=args.interpret)(tokens, base_params)
        diff = float(jnp.max(jnp.abs(logits.astype(jnp.float32) - want.astype(jnp.float32))))
        scale = float(jnp.max(jnp.abs(want.astype(jnp.float32))))
        blocks_name = "default" if best_blocks[0] is None else f"{best_blocks[0]} x {best_blocks[1]}"
        print(
            f"  plain JAX ({weights} qkv) + splash {form} (default blocks)  {t_default:8.3f}  ({rate(t_default):6.1f})"
        )
        print(f"  plain JAX ({weights} qkv) + splash {form} ({blocks_name:>11})  {best:8.3f}  ({rate(best):6.1f})")
        print(f"  autokernel, tuned                                         {t_ours:8.3f}  ({rate(t_ours):6.1f})")
        print(
            f"  speedup {t_default / t_ours:.2f}x / {best / t_ours:.2f}x; max |logits - baseline| {diff:.4f} "
            f"(logits up to {scale:.2f})"
        )

        # One layer: the QKV kernel and the rest of the layer, against one baseline layer at its best blocks.
        x = params["embed"][tokens]
        cos, sin = rope_tables(jnp.arange(seq), config.head_dim, config.rope_theta, config.rotary_dim)
        layer, base_layer = params["layers"][0], base_params["layers"][0]
        our_layer = jax.jit(lambda x, layer, cos, sin: decoder_layer(x, layer, config, cos, sin))  # noqa: B023
        their_layer = jax.jit(baseline_layer(config, seq, *best_blocks, form=form, interpret=args.interpret))
        l_ours, l_base = timed(our_layer, x, layer, cos, sin), timed(their_layer, x, base_layer, cos, sin)
        l_rate = layer_flops(config, batch, seq)
        print(
            f"  one layer: baseline {l_base:.3f} ms ({l_rate / l_base / 1e9:.1f}), autokernel {l_ours:.3f} ms "
            f"({l_rate / l_ours / 1e9:.1f}): {l_base / l_ours:.2f}x"
        )
        if not args.interpret:
            describe_kernels(config, params, batch, seq)
        summary.append((config, batch, seq, t_default, best, t_ours, l_base, l_ours, diff))

    print(
        "\nlayers d_model heads kv ffn   window gated batch seq   splash default  splash best  autokernel  vs default"
        "  vs best  layer: splash  autokernel  speedup  max|diff|"
    )
    for config, batch, seq, t_default, best, t_ours, l_base, l_ours, diff in summary:
        window = config.window or "-"
        print(
            f"{config.layers:<6} {config.d_model:<7} {config.heads:<5} {config.kv_heads:<2} {config.ffn:<5} "
            f"{window!s:<6} {'yes' if config.gated else 'no':<5} {batch:<5} {seq:<5} "
            f"{t_default:14.3f} {best:12.3f} {t_ours:11.3f} "
            f"{t_default / t_ours:10.2f}x {best / t_ours:7.2f}x {l_base:14.3f} {l_ours:11.3f} {l_base / l_ours:7.2f}x"
            f" {diff:10.4f}"
        )


if __name__ == "__main__":
    main()
