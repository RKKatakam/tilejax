"""Probe: the decode megakernel's attention steps over groups of sequences (MegaTiles.group), profiled on a
TPU v5e at a few batches and contexts.

    python benchmarks/probe_megakernel_groups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
import decode_megakernel as mk

SHAPES = ["8,2048,16,8,6144,32768:64,16", "8,2048,16,8,6144,32768:64,512", "8,2048,16,8,6144,32768:32,1024"]


def main() -> None:
    for shape in sys.argv[1:] or SHAPES:
        model, run = shape.split(":")
        config = mk.Config(*map(int, model.split(",")))
        batch, mean = map(int, run.split(","))
        lengths = jnp.asarray(np.random.default_rng(0).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)
        params = mk.init_params(config)
        caches = mk.random_caches(config, lengths, -(-(3 * mean // 2 + 36) // 1024) * 1024)
        tokens = jax.random.randint(jax.random.key(2), (batch,), 0, config.vocab)
        inputs = mk.kernel_inputs(tokens, *caches, lengths, params)
        print(f"== {shape}", flush=True)
        for group in mk.GROUPS:
            p = autokernel.profile(mk.decode_megakernel, *inputs, mk.MegaTiles(group=group))
            print(
                f"  group {group}: {p.device_s * 1e6:8.1f} us, {p.steps:5.0f} steps, compute {p.compute_s * 1e6:7.1f}, "
                f"waits {p.wait_s * 1e6:7.1f}, other {p.other_s * 1e6:6.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
