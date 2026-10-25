"""A collective matmul: `all_gather(x) @ w` over the chips of a mesh axis, the gather inside the kernel.

In a tensor-parallel layer each chip holds some of the rows of the activations `x` and some of the columns
of the weights `w`, and needs every chip's rows times its own columns. Gathering `x` first (an all-gather
over the ICI), then multiplying, leaves the MXU idle while the rows travel. Here the gather is one
prologue statement,

    xs = autokernel.all_gather(x, "x")        # every chip's rows of x, in the order of mesh axis "x"

and the kernel overlaps the two: its tiles walk the rows a chip's shard at a time, its own first, while
each shard travels on around a ring of the chips, arriving before the tiles that read it.

The function is plain JAX inside `jax.shard_map` (`autokernel.all_gather` is `jax.lax.all_gather`), so it
is its own reference, as every kernel's is.

Run with `uv run python examples/collective_matmul.py`. Without TPUs it simulates 8 chips on the CPU, in the
Pallas TPU interpreter with its race detector; on a TPU v5e-8 it runs on the chips
(`benchmarks/collective_matmul.py` times it).
"""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):  # (before JAX starts)
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=8".strip()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import autokernel


@autokernel.tunable
class CollectiveTiles:
    """(tm, tn) output tiles summing tk-deep products; tm rows at a time of one chip's shard of x."""

    tm: int = autokernel.choice(128, 256, 512, 1024, default=256)
    tn: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tk: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


@autokernel.kernel
def all_gather_matmul(x, w, tiles=CollectiveTiles()):
    """x: this chip's rows (m, K); w: this chip's columns (K, n). Every chip's rows times these columns."""
    xs = autokernel.all_gather(x, "x")
    M, K = xs.shape
    N = w.shape[1]
    tm, tn, tk = min(tiles.tm, x.shape[0]), min(tiles.tn, N), min(tiles.tk, K)
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            acc = jnp.zeros((tm, tn), jnp.float32)
            for k in range(0, K, tk):
                acc += xs[i : i + tm, k : k + tk] @ w[k : k + tk, j : j + tn]
            out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
    return out


def sharded(mesh: jax.sharding.Mesh) -> jax.stages.Wrapped:
    """The kernel over the mesh: x's rows and w's columns split along axis "x"; the output's columns too."""
    specs = dict(in_specs=(P("x", None), P(None, "x")), out_specs=P(None, "x"), check_vma=False)
    return jax.jit(jax.shard_map(all_gather_matmul, mesh=mesh, **specs))


def inputs(mesh: jax.sharding.Mesh, M: int, K: int, N: int, dtype: jnp.dtype = jnp.bfloat16) -> tuple[jax.Array, ...]:
    """x (M, K) and w (K, N), sharded the way the kernel's chips see them."""
    x = jax.random.normal(jax.random.key(0), (M, K), dtype)
    w = jax.random.normal(jax.random.key(1), (K, N), dtype)
    return jax.device_put(x, NamedSharding(mesh, P("x", None))), jax.device_put(w, NamedSharding(mesh, P(None, "x")))


if __name__ == "__main__":
    chips = min(8, len(jax.devices()))
    mesh = jax.make_mesh((chips,), ("x",))
    M, K, N = chips * 256, 512, chips * 512
    x, w = inputs(mesh, M, K, N)
    with jax.set_mesh(mesh):  # (the source is built for the mesh's ring: each chip's shard)
        print(all_gather_matmul.source(jax.ShapeDtypeStruct((M // chips, K), x.dtype), w.addressable_shards[0].data))
    out = sharded(mesh)(x, w)
    want = np.asarray(x, np.float64) @ np.asarray(w, np.float64)
    print(f"{chips} chips: max |kernel - float64| = {np.max(np.abs(np.asarray(out, np.float64) - want)):.3g}")
    print("max |kernel - loop nest|:", all_gather_matmul.verify(x, w))
