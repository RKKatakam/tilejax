"""Cross-layer prefetch: a stack of RMSNorm + SwiGLU feed-forward layers as one autokernel kernel (a repeat
loop over the layers) against one kernel per layer and against plain JAX (XLA), bf16, on a TPU v5e.

    uv run python benchmarks/layer_stack.py
    uv run python benchmarks/layer_stack.py --shape 64,2048,8192,8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from layer_stack import ffn_layer, ffn_stack, layer_by_layer, make_inputs, plain_jax, split_layers
from timing import require_tpu, time_ms

SHAPES = [  # tokens, d_model, ffn, layers
    "16,2048,8192,8",
    "64,2048,8192,8",
    "256,2048,8192,8",
    "64,4096,12288,4",
    "4096,2048,8192,4",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="tokens,d_model,ffn,layers (repeatable)")
    args = parser.parse_args()
    require_tpu()
    summary = []
    for t, d, f, layers in [tuple(map(int, shape.split(","))) for shape in args.shape or SHAPES]:
        inputs = make_inputs(t, d, f, layers)
        weight_bytes = layers * 3 * d * f * 2
        x, gamma, w_gate, w_up, w_down = inputs
        ffn_stack(*inputs)  # tune eagerly, on these inputs
        ffn_layer(x, gamma[0], w_gate[0], w_up[0], w_down[0])
        times = {
            "plain JAX (XLA)": time_ms(jax.jit(plain_jax), *inputs),
            "a kernel per layer": time_ms(jax.jit(layer_by_layer), x, split_layers(*inputs[1:])),
            "one kernel, a repeat loop": time_ms(jax.jit(ffn_stack), *inputs),
        }
        want = plain_jax(*inputs).astype(jnp.float32)
        diff = float(jnp.max(jnp.abs(ffn_stack(*inputs).astype(jnp.float32) - want)))
        print(f"tokens={t} d={d} ffn={f} layers={layers}: ms (weights GB/s)")
        for name, ms in times.items():
            print(f"  {name:28} {ms:8.3f}  ({weight_bytes / ms / 1e6:7.1f})")
        print(f"  max |stack - XLA| {diff:.4f}")
        print("  stack:", ffn_stack.tuning(*inputs).config)
        print("  layer:", ffn_layer.tuning(x, gamma[0], w_gate[0], w_up[0], w_down[0]).config)
        summary.append((t, d, f, layers, *times.values()))
    print("\ntokens d     ffn   layers  XLA       per layer  stack     vs XLA  vs per layer")
    for t, d, f, layers, xla, per_layer, stack in summary:
        print(
            f"{t:<6} {d:<5} {f:<5} {layers:<7} {xla:8.3f} {per_layer:9.3f} {stack:9.3f}"
            f" {xla / stack:6.2f}x {per_layer / stack:9.2f}x"
        )


if __name__ == "__main__":
    main()
