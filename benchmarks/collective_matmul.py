"""All-gather matmul over a TPU v5e-8's chips: autokernel's ring kernel against XLA, bf16.

Each chip holds m rows of x (K wide) and n columns of w; every chip computes all chips' rows times its own
columns. Three ways, ms per call:

- XLA: `jax.lax.all_gather(x)` then the matmul, under jax.shard_map;
- the floor: the same matmul with the rows already gathered (no communication at all);
- autokernel: `examples/collective_matmul.py`, at its default tiles and a few others.

uv run python benchmarks/collective_matmul.py                       # m 1024, K 8192, n 3584 per chip
uv run python benchmarks/collective_matmul.py --m 2048 --k 4096 --n 1792
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from collective_matmul import CollectiveTiles, all_gather_matmul, inputs
from timing import require_tpu, time_ms

TILES = [(256, 512, 512), (512, 512, 512), (256, 1024, 512), (512, 1024, 1024), (1024, 1024, 512), (512, 2048, 512)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024, help="rows of x per chip")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=3584, help="columns of w per chip")
    parser.add_argument("--chips", type=int, default=0, help="chips on the ring (0: all)")
    args = parser.parse_args()
    require_tpu()
    chips = args.chips or len(jax.devices())
    mesh = jax.make_mesh((chips,), ("x",))
    m, K, n = args.m, args.k, args.n
    x, w = inputs(mesh, chips * m, K, chips * n)
    flops = 2 * chips * m * K * n  # per chip
    tflops = lambda ms: flops / ms / 1e9  # noqa: E731
    shard = dict(mesh=mesh, check_vma=False)

    def xla(x: jax.Array, w: jax.Array) -> jax.Array:
        xs = jax.lax.all_gather(x, "x", axis=0, tiled=True)
        return jnp.dot(xs, w, preferred_element_type=jnp.float32).astype(x.dtype)

    def floor(xs: jax.Array, w: jax.Array) -> jax.Array:
        return jnp.dot(xs, w, preferred_element_type=jnp.float32).astype(xs.dtype)

    xla_fn = jax.jit(jax.shard_map(xla, in_specs=(P("x", None), P(None, "x")), out_specs=P(None, "x"), **shard))
    floor_fn = jax.jit(jax.shard_map(floor, in_specs=(P(None, None), P(None, "x")), out_specs=P(None, "x"), **shard))
    xs = jax.device_put(jax.device_get(x), NamedSharding(mesh, P(None, None)))  # every chip has every row

    print(f"{chips} chips, per chip: x {m}x{K} (all chips' {chips * m} rows gathered), w {K}x{n}, bf16")
    print("max |kernel - loop nest| at the default tiles:", all_gather_matmul.verify(x, w))
    print("ms per call (TFLOP/s per chip)")
    floor_ms = time_ms(floor_fn, xs, w)
    print(f"  {'floor: the matmul alone, rows already gathered':52} {floor_ms:8.3f}  ({tflops(floor_ms):6.1f})")
    t = time_ms(xla_fn, x, w)
    print(f"  {'XLA: all_gather, then the matmul':52} {t:8.3f}  ({tflops(t):6.1f})")
    for tiles in [CollectiveTiles(), *(CollectiveTiles(tm=tm, tn=tn, tk=tk) for tm, tn, tk in TILES)]:
        if m % min(tiles.tm, m) or n % min(tiles.tn, n) or K % min(tiles.tk, K):
            continue
        run = jax.jit(
            jax.shard_map(
                lambda x, w, tiles=tiles: all_gather_matmul(x, w, tiles),
                in_specs=(P("x", None), P(None, "x")),
                out_specs=P(None, "x"),
                **shard,
            )
        )
        label = f"autokernel ring {tiles}" + (" (default)" if tiles == CollectiveTiles() else "")
        try:
            t = time_ms(run, x, w)
        except Exception as e:  # doesn't build, or Mosaic can't fit it in VMEM
            print(f"  {label:52} {type(e).__name__}: {str(e).splitlines()[0][:60]}")
            continue
        print(f"  {label:52} {t:8.3f}  ({tflops(t):6.1f})  {floor_ms / t:5.1%} of the floor")


if __name__ == "__main__":
    main()
