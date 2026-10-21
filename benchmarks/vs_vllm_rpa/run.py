"""Ragged paged attention on a TPU v5e: vLLM TorchTPU's RPA v3 (vendored, unmodified, from vllm-torchtpu
v0.29.0) against autokernel's (examples/ragged_paged_attention.py).

Workloads (Llama 3 8B's attention: 32 query heads, 8 kv heads of 128; bf16; pages of 128 positions,
8192-position sequences at most): decode batches (one new token per sequence, ragged cache lengths),
a mixed batch (decodes and chunked-prefill chunks) and prefills. Each method is jitted with the cache
donated, as a model step runs it. Device: the time per call the profiler shows ops on the TPU (the
kernels and their XLA glue). GB/s: the cache bytes a step must read (every sequence's keys and values,
new tokens included) over the device time.

    python benchmarks/vs_vllm_rpa/run.py [--workloads decode8_1k,...] [--methods vllm,autokernel]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / "examples"), str(HERE.parent)]

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from ragged_paged_attention import HEAD_DIM, args_of, make_inputs, reference
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa

from autokernel import profiling

SCALE = HEAD_DIM**-0.5


def ragged(batch: int, mean: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(mean // 2, 3 * mean // 2 + 1, batch)]


def workloads() -> dict[str, tuple[list[int], list[int]]]:
    """name -> (q_lens, kv_lens)."""
    found = {}
    for batch in (8, 32, 128):
        for mean in (1024, 4096):
            found[f"decode{batch}_{mean // 1024}k"] = ([1] * batch, ragged(batch, mean, batch + mean))
    decodes = ragged(32, 2048, 7)
    found["mixed32+2x512"] = ([1] * 32 + [512, 512], [*decodes, 1024 + 512, 3072 + 512])
    found["prefill4x1024"] = ([1024] * 4, [1024] * 4)
    found["prefill1x4096"] = ([4096], [4096])
    return found


def vllm(*args):
    return rpa.ragged_paged_attention(*args, sm_scale=SCALE)


def methods() -> dict:
    found = {"vllm": vllm}
    try:
        from ragged_paged_attention import autokernel_rpa
    except ImportError:
        return found
    found["autokernel"] = lambda *args: autokernel_rpa(*args, sm_scale=SCALE)
    return found


def cache_bytes(q_lens: list[int], kv_lens: list[int]) -> int:
    per_position = 2 * 8 * 128 * 2  # keys and values of 8 kv heads, bf16
    return sum(kv_lens) * per_position


def timed(fn, inputs: dict, runs: int = 10) -> tuple[float, float]:
    """(wall seconds, device seconds) per call, the cache donated and fed back."""
    step = jax.jit(fn, donate_argnums=(3,))
    args = list(args_of(inputs))
    args[3] = jnp.copy(args[3])
    fresh = [tuple(jnp.copy(a) for a in args[:3]) for _ in range(3 + 5 * runs + runs)]  # (RPA donates q, k, v)

    def call():
        q, k, v = fresh.pop()
        out, args[3] = step(q, k, v, *args[3:])
        return out

    for _ in range(3):
        jax.block_until_ready(call())
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(runs):
            out = call()
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs)
    logdir = tempfile.mkdtemp(prefix="rpa-")
    jax.profiler.start_trace(logdir)
    for _ in range(runs):
        out = call()
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    ops = [e for e in profiling.device_events(logdir) if e.line == profiling.OPS_LINE]
    by_name: dict[str, float] = {}
    for e in ops:
        by_name[e.name.split(" ")[0][:60]] = by_name.get(e.name.split(" ")[0][:60], 0.0) + e.seconds / runs
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:5]
    print("    ops: " + ", ".join(f"{n} {t * 1e6:.1f}" for n, t in top), flush=True)
    return sorted(samples)[2], sum(e.seconds for e in ops) / runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", default="")
    parser.add_argument("--methods", default="vllm,autokernel")
    parser.add_argument("--check", action="store_true", help="compare with float64 NumPy (slow for big batches)")
    opts = parser.parse_args()
    on_tpu = jax.default_backend() == "tpu"
    if not on_tpu:  # (a smoke test: their kernel in the interpreter, at explicit block sizes)
        interpreting = pltpu.force_tpu_interpret_mode()
        interpreting.__enter__()
        pltpu.get_tpu_info = lambda: types.SimpleNamespace(vmem_capacity_bytes=128 << 20)
    available = methods()
    names = [m for m in opts.methods.split(",") if m in available]
    chosen = workloads()
    if opts.workloads:
        chosen = {k: chosen[k] for k in opts.workloads.split(",")}
    print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}; methods {names}", flush=True)
    for name, (q_lens, kv_lens) in chosen.items():
        inputs = make_inputs(q_lens, kv_lens, max_tokens=-(-sum(q_lens) // 16) * 16)
        nbytes = cache_bytes(q_lens, kv_lens)
        print(f"{name}: {len(q_lens)} sequences, {sum(q_lens)} tokens, {nbytes / 1e6:.1f} MB of cache", flush=True)
        want = reference(inputs, SCALE) if opts.check else None
        for method in names:
            fn = available[method]
            if method == "autokernel" and any(q > 1 for q in q_lens):
                continue  # (decode batches only, so far)
            if method == "autokernel" and on_tpu:
                from ragged_paged_attention import tune

                print("  " + str(tune(inputs, SCALE)).replace("\n", "\n  "), flush=True)
            if want is not None:
                out, cache = jax.jit(fn)(*[jnp.copy(a) for a in args_of(inputs)])
                t = int(np.asarray(inputs["cu_q_lens"])[len(q_lens)])
                err = float(np.abs(np.asarray(out[:t], np.float64) - want[0]).max())
                cerr = float(np.abs(np.asarray(cache, np.float64) - want[1]).max())
                print(f"  {method:10} out vs f64 {err:.3g}  cache vs f64 {cerr:.3g}", flush=True)
            wall, device = timed(fn, inputs) if on_tpu else (float("nan"), float("nan"))
            print(
                f"  {method:10} wall {wall * 1e6:9.1f} us  device {device * 1e6:9.1f} us  "
                f"{nbytes / device / 1e9:6.1f} GB/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
