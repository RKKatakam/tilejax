"""Sweep autokernel's KDA decode schedules on a TPU v5e: each kernel's device time (from a profiler trace)
in a donated decode loop (the pools updated in place) and called without donation (XLA copies the pools
first; only the kernel's own op is counted), against vLLM's kernel the same two ways.

    python benchmarks/vs_vllm_kda/sweep.py [--batch 64,256]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / "examples")]

import jax
import jax.numpy as jnp
from kda_decode import ARGS, KdaTiles, args_of, kda_decode_step, make_inputs, step_args
from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda

from autokernel import Config, Pipeline, profiling

DONATE = (ARGS.index("conv_state"), ARGS.index("state"))
SCHEDULES = [
    (16, Pipeline(3, 2)),
    (32, Pipeline(3, 2)),
    (16, Pipeline.bursts(2)),
    (16, Pipeline.bursts(4)),
    (16, Pipeline.bursts(8)),
    (16, Pipeline.bursts(16)),
    (32, Pipeline.bursts(1)),
    (32, Pipeline.bursts(2)),
    (32, Pipeline.bursts(4)),
    (32, Pipeline.bursts(8)),
    (32, Pipeline.bursts(16)),
]


def kernel_us(fn, args, donated, name, calls=5):
    """Device time of the op named `name` per call."""
    step = jax.jit(fn, donate_argnums=DONATE if donated else ())
    args = list(args)
    if donated:
        args[DONATE[0]], args[DONATE[1]] = jnp.copy(args[DONATE[0]]), jnp.copy(args[DONATE[1]])
    for _ in range(3):
        out, conv, state = step(*args)
        if donated:
            args[DONATE[0]], args[DONATE[1]] = conv, state
    jax.block_until_ready(out)
    logdir = tempfile.mkdtemp(prefix="sweep-")
    jax.profiler.start_trace(logdir)
    for _ in range(calls):
        out, conv, state = step(*args)
        if donated:
            args[DONATE[0]], args[DONATE[1]] = conv, state
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    ops = [e for e in profiling.device_events(logdir) if e.line == profiling.OPS_LINE and e.name.startswith(name)]
    total = [e for e in profiling.device_events(logdir) if e.line == profiling.OPS_LINE]
    return sum(e.seconds for e in ops) / calls * 1e6, sum(e.seconds for e in total) / calls * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="64,256")
    opts = parser.parse_args()
    for batch in map(int, opts.batch.split(",")):
        inputs = make_inputs(batch)
        args = args_of(inputs)
        for donated in (True, False):
            k, t = kernel_us(decode_kda, args, donated, "kda_decode_fused_conv_recurrent")
            print(f"B {batch:3} vllm {'donated' if donated else 'copied '}: kernel {k:8.1f} us, all ops {t:8.1f} us")
        sig = kda_decode_step.signature(*step_args(*args))
        for heads, pipeline in SCHEDULES:
            config = Config(KdaTiles(heads=heads), pipeline)
            try:
                lowered = kda_decode_step.lowered(sig, config)
            except Exception as e:
                print(f"B {batch:3} {heads} {pipeline}: {e}")
                continue
            fn = lambda *a, lowered=lowered: lowered.entry(*step_args(*a))  # noqa: E731
            for donated in (True, False):
                k, t = kernel_us(fn, args, donated, "kda_decode_step")
                print(
                    f"B {batch:3} heads {heads} {pipeline!s:28} {'donated' if donated else 'copied '}: "
                    f"kernel {k:8.1f} us, all ops {t:8.1f} us",
                    flush=True,
                )


if __name__ == "__main__":
    main()
