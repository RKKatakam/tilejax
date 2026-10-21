"""Probe: is the decode megakernel's DMA queue drained at its phase boundaries? Its profile at deeper input
pipelines, and with the per-layer code between phases in named scopes (attention_start, attention_rows).

    python benchmarks/probe_megakernel_depth.py
"""

from __future__ import annotations

import linecache
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
import decode_megakernel as mk

SCOPES = {  # the first line of a phase's body (or of code between phases) -> (its scope, its lines)
    "            q_bd, row_max, row_sum, acc = attention_start(new, H, Hkv)\n": ("attention_start", 1),
    "            attn_blocks = lane_blocks(attention_rows(acc, row_sum, H, Hkv).astype(dtype), ba)\n": (
        "attention_rows",
        1,
    ),
    "                qkv += n_blocks[d // bd] @ w_qkv[layer, d : d + bd, :]\n": ("qkv", 1),
    "                kt = k_cache[layer, autokernel.rows(j, j - j % S + lengths[j // S], bk), :]\n": ("attention", 9),
    "                proj += attn_blocks[a // ba] @ wo[layer, a : a + ba, :]\n": ("wo", 1),
    "                hidden = jax.nn.silu(normed @ w_gate[layer, :, f : f + bf]) * (normed @ w_up[layer, :, f : f + bf])\n": (  # noqa: E501
        "ffn",
        2,
    ),
    "            logit_blocks = logit_blocks.at[c // bv].set((final @ w_vocab[:, c : c + bv]).astype(dtype))\n": (
        "vocab",
        1,
    ),
}


def scoped_kernel() -> types.ModuleType:
    """decode_megakernel's module with the code between its phases in named scopes."""
    source = (EXAMPLES / "decode_megakernel.py").read_text()
    lines = source.splitlines(keepends=True)
    for first, (scope, count) in SCOPES.items():
        at = lines.index(first)
        indent = first[: len(first) - len(first.lstrip())]
        lines[at : at + count] = [f'{indent}with jax.named_scope("{scope}"):\n'] + [
            "    " + x for x in lines[at : at + count]
        ]
    source = "".join(lines)
    filename = "<decode_megakernel, scoped>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType("decode_megakernel_scoped")
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def main() -> None:
    autokernel.pipeline.MAX_INPUT_BUFFERS = 8  # (for the probe: deeper than the tuner goes)
    scoped = scoped_kernel()
    for shape in ("8,2048,16,8,6144,32768:64,512", "8,2048,16,8,6144,32768:64,16", "8,2048,16,8,6144,32768:1,16"):
        model, run = shape.split(":")
        config = mk.Config(*map(int, model.split(",")))
        batch, mean = map(int, run.split(","))
        lengths = jnp.asarray(np.random.default_rng(0).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)
        params = mk.init_params(config)
        caches = mk.random_caches(config, lengths, -(-(3 * mean // 2 + 36) // 1024) * 1024)
        tokens = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
        inputs = mk.kernel_inputs(tokens, *caches, lengths, params)
        tiles = mk.MegaTiles()
        print(f"== {shape}", flush=True)
        print(autokernel.profile(scoped.decode_megakernel, *inputs, scoped.MegaTiles()), flush=True)
        for depth in ():
            config_ = autokernel.Config(tiles, autokernel.Pipeline(depth, 1))
            lowered = mk.decode_megakernel.lowered(mk.decode_megakernel.signature(*inputs), config_, trace=True)
            print(f"  {depth} input buffers, {lowered.vmem_bytes / 2**20:.0f} MiB of VMEM:", flush=True)
            print(autokernel.profiling.measure(lowered, inputs, 5), flush=True)


if __name__ == "__main__":
    main()
