"""Decode-like MLP layers: a few rows through x @ up -> ReLU -> @ down, where every layer streams its weights.

Run on a TPU VM:
    uv run python benchmarks/decode.py                         # rows 64 and 256, L=4, D=2048, F=8192, bf16
    uv run python benchmarks/decode.py --rows 64 --kernels     # also time each kernel alone in a fori_loop

Each layer reads 2 * D * F bf16 weights (64 MiB at D=2048, F=8192): about 82 us at a v5e's 819 GB/s, the
floor for any implementation. The script times, per layer:

- autokernel with the block sizes measured before (`--old-tiles`, 512x512: 103 us a layer at 64 rows,
  against XLA's 95), fused (both nests in one kernel) and not;
- autokernel with `tune="auto"`: the cost model's blocks (stream each weight once, activations fetched again
  for as few column tiles as fit: TN = 2048 on both projections);
- the same tuned blocks with 2, 3 and 4 input buffers (`--buffers`);
- XLA.

Next to each autokernel variant it prints what the cost model predicts (the kernels' `in_program_s`, the
time inside a jitted loop), so a run checks the model as well as the kernels. `--kernels` also times each
nest's kernel on its own inside a `fori_loop` over the layers, the setting the model's weight-streaming
calibration (tests/test_cost.py) comes from. Off a TPU, `--interpret` runs a tiny configuration in the
Pallas interpreter to check the script.
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

LANES = 128
EPS = 1e-6
dtype = jnp.bfloat16
f32 = jnp.float32
# Set by `configure`: rows, model width, MLP width, layers, and each projection's blocks (rows, columns,
# reduction): up TU*, down TD*.
R = D = F = L = 0
TUM = TUN = TUK = TDM = TDN = TDK = 0
TILE_NAMES = ("TUM", "TUN", "TUK", "TDM", "TDN", "TDK")


def configure(**sizes) -> None:
    globals().update(sizes)


def mlp_stack(x, norm, w_up, w_down):
    """L layers of x + relu(rmsnorm(x) @ w_up) @ w_down; RMSNorm folded into the up projection."""
    for layer in range(L):
        up = jnp.zeros((R, F), dtype)
        for i in range(0, R, TUM):
            for j in range(0, F, TUN):
                acc = jnp.zeros((TUM, TUN))
                ss = jnp.zeros((TUM, LANES))
                for kk in range(0, D, TUK):
                    xf = x[i : i + TUM, kk : kk + TUK].astype(f32)
                    ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                    acc += (xf * norm[layer, :, kk : kk + TUK]).astype(dtype) @ w_up[layer, kk : kk + TUK, j : j + TUN]
                inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TUN // LANES))
                up = up.at[i : i + TUM, j : j + TUN].set(jnp.maximum(acc * inv, 0).astype(dtype))
        y = jnp.zeros((R, D), dtype)
        for i in range(0, R, TDM):
            for j in range(0, D, TDN):
                acc = jnp.zeros((TDM, TDN))
                for kk in range(0, F, TDK):
                    acc += up[i : i + TDM, kk : kk + TDK] @ w_down[layer, kk : kk + TDK, j : j + TDN]
                y = y.at[i : i + TDM, j : j + TDN].set((x[i : i + TDM, j : j + TDN] + acc).astype(dtype))
        x = y
    return x


def xla_mlp_stack(x, norm, w_up, w_down):
    def layer(x, w):
        g, wu, wd = w
        xf = x.astype(f32)
        xn = (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * g).astype(dtype)
        up = jnp.maximum(jnp.dot(xn, wu, preferred_element_type=f32), 0).astype(dtype)
        return (x + jnp.dot(up, wd, preferred_element_type=f32)).astype(dtype), None

    return jax.lax.scan(layer, x, (norm, w_up, w_down))[0]


def inputs(seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(keys[0], (R, D), f32).astype(dtype)
    norm = 1 + 0.1 * jax.random.normal(keys[1], (L, 1, D), f32)
    w_up = (jax.random.normal(keys[2], (L, D, F), f32) / math.sqrt(D)).astype(dtype)
    w_down = (jax.random.normal(keys[3], (L, F, D), f32) / math.sqrt(F)).astype(dtype)
    return x, norm, w_up, w_down


def time_ms(fn, *args, runs: int = 20, repeats: int = 5) -> float:
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs * 1e3)
    return float(np.median(samples))


def kernels(program, args) -> list[tuple[str, autokernel.Lowered]]:
    """Every nest kernel the program runs, fused ones split into their parts."""
    out = []
    for name, k in program.lower(*args):
        out += list(getattr(k, "parts", ())) or [(name, k)]
    return out


def predicted_us(program, args) -> float:
    """The model's time for one layer: the nests' kernels inside a jitted loop (unfused: one launch each)."""
    return sum(k.estimate.in_program_s for _, k in kernels(program, args)) * 1e6


def kernel_alone_us(lowered, specs, runs: int = 40) -> float:
    """One kernel, `L` times inside a fori_loop (its layer index the loop variable), per call."""
    xs = [jnp.zeros(s.shape, s.dtype) if s.shape == () else jnp.ones(s.shape, s.dtype) for s in specs]
    scalar = next((n for n, s in enumerate(specs) if s.shape == ()), None)

    def loop(*xs):
        def body(i, carry):
            args = list(xs)
            if scalar is not None:
                args[scalar] = i % L
            out = lowered.entry(*args, interpret=False)
            return carry + jnp.sum(out[0, :LANES].astype(f32))

        return jax.lax.fori_loop(0, runs, body, jnp.float32(0))

    fn = jax.jit(loop)
    return time_ms(fn, *xs, runs=1, repeats=5) / runs * 1e3


def run(args, rows: int) -> None:
    interpret = args.interpret
    options = dict(interpret=True) if interpret else dict(interpret=False)
    old = tuple(map(int, args.old_tiles.split(",")))
    base = dict(R=rows, D=args.width, F=args.hidden, L=args.layers)
    if interpret:
        base = dict(R=16, D=256, F=512, L=2)
        old = (128, 128)
    configure(**base, TUM=base["R"], TUN=old[0], TUK=old[1], TDM=base["R"], TDN=old[0], TDK=old[1])
    xs = inputs()
    weights_mb = 2 * D * F * 2 / 1e6
    print(f"\n{jax.devices()[0].device_kind}: rows {R}, D={D}, F={F}, {L} layers; weights {weights_mb:.1f} MB a layer, "
          f"{weights_mb / 819e3 * 1e3:.1f} us at 819 GB/s")  # fmt: skip
    print(f"  {'variant':34} {'ms':>8} {'us/layer':>9} {'GB/s':>6} {'predicted us/layer':>19}  blocks")
    want = None
    tuned_blocks = None
    variants = [
        (f"autokernel {old[0]}x{old[1]} fused", dict(), None),
        (f"autokernel {old[0]}x{old[1]} unfused", dict(fuse=False), None),
        ("autokernel tune='auto' fused", dict(tune={n: "auto" for n in TILE_NAMES}), None),
        ("autokernel tune='auto' unfused", dict(tune={n: "auto" for n in TILE_NAMES}, fuse=False), None),
    ]
    variants += [(f"autokernel tuned, {b} buffers", dict(fuse=False, buffers=b), "tuned") for b in args.buffers]
    for label, extra, blocks in variants:
        if blocks == "tuned":
            if tuned_blocks is None:
                continue
            configure(**tuned_blocks)
        program = autokernel.tpu(mlp_stack, **options, **extra)
        try:
            ms = time_ms(program, *xs) if not interpret else float("nan")
            out = program(*xs)
        except Exception as e:  # e.g. a block size Mosaic rejects
            print(f"  {label:34} failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True)
            continue
        if want is None:
            want = out.astype(f32)
        lowered = kernels(program, xs)
        blocks_used = {n: globals()[n] for n in TILE_NAMES}
        blocks_used.update({k: v for _, lw in lowered for k, v in lw.overrides.items()})
        if "tune" in extra and tuned_blocks is None:
            tuned_blocks = {n: blocks_used[n] for n in TILE_NAMES}
        err = float(jnp.max(jnp.abs(out.astype(f32) - want)))
        per_layer = ms / L * 1e3
        gbs = weights_mb / per_layer * 1e3 if per_layer == per_layer else float("nan")
        print(f"  {label:34} {ms:8.3f} {per_layer:9.1f} {gbs:6.0f} {predicted_us(program, xs):19.1f}  "
              f"{' '.join(f'{n}={blocks_used[n]}' for n in TILE_NAMES)}  (err {err:.3g})", flush=True)  # fmt: skip
        if args.kernels and not interpret and "fuse" in extra:
            for (name, lw), (_, specs) in zip(lowered, _specs(program, xs), strict=False):
                us, model = kernel_alone_us(lw, specs), lw.estimate.in_program_s * 1e6
                print(f"      {name:30} alone in a fori_loop {us:7.1f} us, predicted {model:7.1f} "
                      f"({lw.buffers} buffers, {lw.steps} steps)", flush=True)  # fmt: skip
        configure(**base, TUM=base["R"], TUN=old[0], TUK=old[1], TDM=base["R"], TDN=old[0], TDK=old[1])
    if not interpret:
        xla = jax.jit(xla_mlp_stack)
        ms = time_ms(xla, *xs)
        err = float(jnp.max(jnp.abs(xla(*xs).astype(f32) - want)))
        per_layer = ms / L * 1e3
        print(f"  {'XLA':34} {ms:8.3f} {per_layer:9.1f} {weights_mb / per_layer * 1e3:6.0f} {'':19}  (err {err:.3g})")


def _specs(program, xs):
    low = program.lower(*xs)
    return [(name, specs) for (name, _), specs in zip(low, low.inputs, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default="64,256", help="rows (tokens) per layer, comma-separated")
    parser.add_argument("--width", type=int, default=2048, help="model width D")
    parser.add_argument("--hidden", type=int, default=8192, help="MLP width F")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--old-tiles", default="512,512", help="hand-picked columns,reduction blocks for both nests")
    parser.add_argument("--buffers", default="2,3,4", help="input buffer depths to time with the tuned blocks")
    parser.add_argument("--kernels", action="store_true", help="also time each unfused kernel alone")
    parser.add_argument("--interpret", action="store_true", help="tiny sizes in the Pallas interpreter")
    args = parser.parse_args()
    args.buffers = [int(b) for b in args.buffers.split(",") if b]
    if not args.interpret and jax.default_backend() != "tpu":
        raise SystemExit(f"benchmarks need a TPU (the default backend is {jax.default_backend()}); try --interpret")
    for rows in [0] if args.interpret else map(int, args.rows.split(",")):
        run(args, rows)


if __name__ == "__main__":
    main()
