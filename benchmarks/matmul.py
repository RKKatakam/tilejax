"""Time autokernel against XLA for dense and causal matmuls.

Run on a TPU VM:  uv run python benchmarks/matmul.py [--size 8192] [--tiles auto | tm,tn,tk] [--dtypes bfloat16,float32]

With `--tiles auto` (the default) autokernel picks its own block sizes the way XLA does: the cost model
ranks a grid of candidates and the best few are timed on the TPU. Timings off-TPU are meaningless (the
kernels run in an interpreter), so the script refuses to run there.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

TILE_GRID = {"tm": (512, 1024, 2048), "tn": (512, 1024, 2048), "tk": (512, 1024)}


def time_it(fn, *args, runs: int = 20) -> float:
    """Average seconds per call, after a warm-up call that compiles."""
    fn(*args).block_until_ready()
    start = time.perf_counter()
    for _ in range(runs):
        out = fn(*args)
    out.block_until_ready()
    return (time.perf_counter() - start) / runs


def make_kernels(size: int, tiles: tuple[int, int, int] | None, dtype) -> dict[str, autokernel.TpuKernel]:
    tm, tn, tk = tiles or (512, 512, 512)  # with `tiles=None` these are the tuner's starting values

    def dense(x, y):
        out = jnp.zeros((size, size), dtype)
        for i in range(0, size, tm):
            for j in range(0, size, tn):
                acc = jnp.zeros((tm, tn), jnp.float32)
                for k in range(0, size, tk):
                    acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
                out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
        return out

    def causal(x, y):
        out = jnp.zeros((size, size), dtype)
        for i in range(0, size, tm):
            for j in range(0, size, tn):
                if j > i + tm - 1:  # whole tile above the diagonal
                    continue
                acc = jnp.zeros((tm, tn), jnp.float32)
                for k in range(0, size, tk):
                    acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0)
                cols = j + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 1)
                out = out.at[i : i + tm, j : j + tn].set(jnp.where(cols <= rows, acc, 0.0).astype(dtype))
        return out

    opts = dict(interpret=False)
    if tiles is None:
        opts.update(tune=TILE_GRID, autotune="measure")
    return {
        "dense": autokernel.tpu(dense, **opts),
        "causal (background fill)": autokernel.tpu(causal, **opts),
        "causal (prefill)": autokernel.tpu(causal, background_fill=False, **opts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--tiles", default="auto", help="'auto' (tuned per kernel) or tile_m,tile_n,tile_k")
    parser.add_argument("--dtypes", default="bfloat16,float32")
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        raise SystemExit("benchmarks need a TPU (the default backend is " + jax.default_backend() + ")")

    tiles = None if args.tiles == "auto" else tuple(map(int, args.tiles.split(",")))
    size = args.size
    dense_flops, causal_flops = 2 * size**3, size * size * (size + 1)
    rng = np.random.default_rng(0)
    print(f"{jax.devices()[0].device_kind}, {size}^3, tiles {'tuned' if tiles is None else 'x'.join(map(str, tiles))}")
    for name in args.dtypes.split(","):
        dtype = jnp.dtype(name)
        x = jnp.asarray(rng.standard_normal((size, size), dtype=np.float32), dtype)
        y = jnp.asarray(rng.standard_normal((size, size), dtype=np.float32), dtype)
        kernels = make_kernels(size, tiles, dtype)
        xla_dense = jax.jit(lambda a, b: jnp.dot(a, b, precision=jax.lax.Precision.HIGHEST))
        xla_causal = jax.jit(lambda a, b: jnp.tril(jnp.dot(a, b, precision=jax.lax.Precision.HIGHEST)))
        print(f"\n{name}  (causal TFLOP/s count only the lower triangle)")
        rows = [
            ("XLA dot", xla_dense, dense_flops, None),
            ("autokernel dense", jax.jit(kernels["dense"]), dense_flops, kernels["dense"]),
            ("XLA dot + tril", xla_causal, causal_flops, None),
            (
                "autokernel causal (background fill)",
                jax.jit(kernels["causal (background fill)"]),
                causal_flops,
                kernels["causal (background fill)"],
            ),
            (
                "autokernel causal (prefill)",
                jax.jit(kernels["causal (prefill)"]),
                causal_flops,
                kernels["causal (prefill)"],
            ),
        ]
        for label, fn, flops, kernel in rows:
            chosen = ""
            if kernel is not None:
                lowered = kernel.lower(x, y)  # with arrays: tunes (and times candidates) before the jit traces
                t = lowered.overrides
                chosen = f"  tiles {t['tm']}x{t['tn']}x{t['tk']}, {lowered.buffers} buffers" if t else ""
            seconds = time_it(fn, x, y)
            print(f"  {label:<38} {seconds * 1e3:8.2f} ms  {flops / seconds / 1e12:6.1f} TFLOP/s{chosen}")


if __name__ == "__main__":
    main()
