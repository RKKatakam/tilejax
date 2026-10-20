"""KDA decode (Kimi Linear's linear attention, one token per sequence, paged states updated in place) on a
TPU v5e: vLLM TorchTPU's fused `decode_kda` kernel (vendored, unmodified, from vllm-torchtpu v0.29.0)
against autokernel's (examples/kda_decode.py) and the plain-JAX step (XLA).

Every method gets the same inputs: Kimi Linear's KDA layer (32 heads of 128, a 4-tap conv, bf16
activations, f32 state), a pool of 2 * batch + 1 slots (slot 0 the null block), each sequence in a random
real slot, a tenth of them on their first step. Each is timed as a decode loop runs it: jitted with both
state pools donated, each call taking the previous call's pools. Wall: the median over 5 repeats of the
mean of 20 back-to-back calls with one block_until_ready after them (after compiling and 3 untimed calls).
Device: the time per call the profiler's trace shows ops on the TPU (sum of the op events of 5 calls / 5).
GB/s: the bytes a step must move (read and write each sequence's state and conv-state slot; read its
inputs, write its output) over the device time.

    python benchmarks/vs_vllm_kda/run.py [--batch 1,8,32] [--methods vllm,autokernel,xla] [--json out.jsonl]
"""

from __future__ import annotations

import argparse
import json
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
from kda_decode import ARGS, args_of, make_inputs, reference, xla_decode
from vllm_torchtpu.kernels.kimi_k3.decode_kda import decode_kda

from autokernel import profiling

BATCHES = [1, 8, 32, 64, 128, 256]
HBM_GBS = 819.0
DONATE = (ARGS.index("conv_state"), ARGS.index("state"))


def methods() -> dict:
    found = {"vllm": decode_kda, "xla": xla_decode}
    try:
        from kda_decode import kda_decode
    except ImportError:
        return found
    found["autokernel"] = kda_decode
    from kda_decode import KdaTiles, kda_decode_step, step_args

    for heads in (16, 32):  # a fixed tunable (not tuned)
        found[f"autokernel{heads}"] = lambda *a, t=KdaTiles(heads=heads): kda_decode_step(*step_args(*a), t)
    return found


def step_bytes(inputs: dict) -> int:
    """What one step must move: per sequence its state and conv-state slot read and written, its inputs read
    and its output written; the weights once."""
    b = inputs["state_indices"].shape[0]
    per_seq = 2 * (inputs["state"][0].nbytes + inputs["conv_state"][0].nbytes)
    per_seq += (inputs["mixed_qkv"][0].nbytes + inputs["g"][0].nbytes + inputs["beta"][0].nbytes) + 32 * 128 * 2
    weights = sum(inputs[k].nbytes for k in ("conv_weight", "a_log", "dt_bias"))
    return b * per_seq + weights


def bench(step, args: list, runs: int = 20, repeats: int = 5) -> tuple[float, list]:
    """Mean seconds per call (median over repeats) of a donated decode loop; the final arguments."""
    conv_at, state_at = DONATE

    def call(args):
        out, conv, state = step(*args)
        args[conv_at], args[state_at] = conv, state
        return out

    for _ in range(4):  # compile + 3 untimed
        jax.block_until_ready(call(args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = call(args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs)
    return sorted(samples)[len(samples) // 2], args


def device_time(step, args: list, calls: int = 5) -> float:
    """Seconds per call the device spends in ops, from a profiler trace."""
    conv_at, state_at = DONATE
    logdir = tempfile.mkdtemp(prefix="kda-")
    jax.profiler.start_trace(logdir)
    try:
        for _ in range(calls):
            out, args[conv_at], args[state_at] = step(*args)
        jax.block_until_ready(out)
    finally:
        jax.profiler.stop_trace()
    events = profiling.device_events(logdir)
    ops = [e for e in events if e.line == profiling.OPS_LINE]
    by_name: dict[str, float] = {}
    for e in ops:
        by_name[e.name] = by_name.get(e.name, 0.0) + e.seconds / calls
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:6]
    print("    ops: " + ", ".join(f"{name} {t * 1e6:.1f} us" for name, t in top), flush=True)
    return sum(e.seconds for e in ops) / calls


def check(name: str, result, inputs: dict, want, theirs) -> dict:
    """Max abs differences against float64 (sampled sequences) and against vLLM's kernel (everything).

    Against float64 also the median over heads of each head's largest state error: a conv output that
    rounds to bf16 the other way than in the reference (f32 SiLUs differ by an ulp) moves a whole head's
    state by ~1e-4, so the largest error says little about the arithmetic; the median head does."""
    idx = np.asarray(inputs["state_indices"])
    seqs, (o64, c64, s64) = want
    f = lambda x: np.asarray(x, np.float64)  # noqa: E731
    out, conv, state = result
    state_err = np.abs(f(state)[idx[seqs]] - s64).max(axis=(2, 3))  # (sequences, heads)
    row = {
        "out_vs_f64": float(np.abs(f(out)[seqs] - o64).max()),
        "conv_vs_f64": float(np.abs(f(conv)[idx[seqs]] - c64).max()),
        "state_vs_f64": float(state_err.max()),
        "state_vs_f64_median_head": float(np.median(state_err)),
    }
    if theirs is not None:
        row |= {
            "out_vs_vllm": float(jnp.max(jnp.abs(out.astype(jnp.float32) - theirs[0].astype(jnp.float32)))),
            "conv_vs_vllm": float(jnp.max(jnp.abs(conv.astype(jnp.float32) - theirs[1].astype(jnp.float32)))),
            "state_vs_vllm": float(jnp.max(jnp.abs(state - theirs[2]))),
        }
    print(f"  {name:10} " + "  ".join(f"{k} {v:.3g}" for k, v in row.items()), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=",".join(map(str, BATCHES)))
    parser.add_argument("--methods", default="vllm,autokernel,xla")
    parser.add_argument("--json", default=str(HERE / "results.jsonl"))
    opts = parser.parse_args()
    if jax.default_backend() != "tpu":  # (a smoke test: their kernel in the interpreter)
        interpreting = pltpu.force_tpu_interpret_mode()
        interpreting.__enter__()
        pltpu.get_tpu_info = lambda: types.SimpleNamespace(vmem_capacity_bytes=128 << 20)  # (their vmem limit)
    available = methods()
    names = [m for m in opts.methods.split(",") if m in available]
    print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}; methods {names}", flush=True)
    for batch in map(int, opts.batch.split(",")):
        inputs = make_inputs(batch)
        nbytes = step_bytes(inputs)
        seqs = np.unique(np.linspace(0, batch - 1, min(batch, 16)).astype(int))
        want = (seqs, reference(inputs, seqs))
        print(f"batch {batch}: {nbytes / 1e6:.1f} MB per step, pool of {inputs['state'].shape[0]} slots", flush=True)
        theirs = None
        rows = []
        for name in names:
            fn = available[name]
            if name == "autokernel":  # tune on these inputs (under jax.jit it would tune on random ones)
                from kda_decode import kda_decode_step, step_args

                print(
                    "  " + str(kda_decode_step.tuning(*step_args(*args_of(inputs)))).replace("\n", "\n  "), flush=True
                )
            first = jax.jit(fn)(*args_of(inputs))  # undonated: the inputs stay for the other methods
            jax.block_until_ready(first)
            row = {"batch": batch, "method": name, "bytes": nbytes}
            row |= check(name, first, inputs, want, theirs if name != "vllm" else None)
            if name == "vllm":
                theirs = first
            else:
                del first
            step = jax.jit(fn, donate_argnums=DONATE)
            args = [jnp.copy(a) for a in args_of(inputs)]
            wall, args = bench(step, args)
            device = device_time(step, args) if jax.default_backend() == "tpu" else wall
            del args
            if name == "autokernel" and jax.default_backend() == "tpu":  # where its time goes: trace regions
                from kda_decode import kda_decode_step, step_args

                import autokernel

                report = autokernel.profile(kda_decode_step, *step_args(*args_of(inputs)))
                print("  " + str(report).replace("\n", "\n  "), flush=True)
            row |= {"wall_us": wall * 1e6, "device_us": device * 1e6, "gbs": nbytes / device / 1e9}
            rows.append(row)
            print(
                f"  {name:10} wall {wall * 1e6:9.1f} us  device {device * 1e6:9.1f} us  "
                f"{row['gbs']:6.1f} GB/s ({row['gbs'] / HBM_GBS:.0%} of {HBM_GBS:.0f})",
                flush=True,
            )
        with open(opts.json, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        del theirs


if __name__ == "__main__":
    main()
