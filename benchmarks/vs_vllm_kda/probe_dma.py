"""Probe: how fast a TPU v5e moves a decode step's traffic (read a slot, write a slot) with DMAs.

Each kernel streams N chunks of a pool (N, C, 128, 128) f32 from HBM into VMEM slots and back out to
another pool, a ring of `reads` read slots fetched ahead and `writes` writes in flight; the time per call is
device-bound. Against XLA's own elementwise pass over the same array and read-only / write-only streams.

    python benchmarks/vs_vllm_kda/probe_dma.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MIB = 1 << 20


def timed(fn, *args, runs=10):
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t) / runs)
    return sorted(samples)[2]


def stream(n, chunk, reads, writes, split, mode):
    """mode: "copy" (read then write each chunk), "read", "write"."""
    piece = chunk[0] // split

    def kernel(src, dst, bufs, rsem, wsem):
        def read(i, slot):
            return [
                pltpu.make_async_copy(
                    src.at[i, pl.ds(p * piece, piece)], bufs.at[slot, pl.ds(p * piece, piece)], rsem.at[slot, p]
                )
                for p in range(split)
            ]

        def write(i, slot):
            return [
                pltpu.make_async_copy(
                    bufs.at[slot, pl.ds(p * piece, piece)], dst.at[i, pl.ds(p * piece, piece)], wsem.at[slot, p]
                )
                for p in range(split)
            ]

        slots = reads + writes  # a slot is free once its write is done
        if mode != "write":
            for i in range(min(reads - 1, n)):
                for c in read(i, i % slots):
                    c.start()

        @pl.loop(0, n)
        def _(i):
            slot = jax.lax.rem(i, slots)
            if mode != "write":

                @pl.when(i + reads - 1 < n)
                def _():
                    ahead = i + reads - 1

                    @pl.when(ahead >= slots)  # its slot's previous write
                    def _():
                        if mode == "copy":
                            for c in write(0, jax.lax.rem(ahead, slots)):
                                c.wait()

                    for c in read(ahead, jax.lax.rem(ahead, slots)):
                        c.start()

                for c in read(i, slot):
                    c.wait()
            if mode != "read":
                if mode == "write":

                    @pl.when(i >= writes)
                    def _():
                        for c in write(0, jax.lax.rem(i - writes, slots)):
                            c.wait()

                for c in write(i, slot):
                    c.start()

        if mode != "read":
            keep = writes if mode == "write" else min(slots, n)
            for k in range(max(0, n - keep), n):
                for c in write(0, k % slots):
                    c.wait()

    shape = (n, *chunk)
    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM((reads + writes, *chunk), jnp.float32),
                pltpu.SemaphoreType.DMA((reads + writes, split)),
                pltpu.SemaphoreType.DMA((reads + writes, split)),
            ],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * MIB),
        )
    )


def main():
    total = 512 * MIB
    print("XLA elementwise x * 2 (read + write):", end=" ")
    for heads in (16,):
        n = total // (heads * 64 * 1024)
        x = jnp.ones((n, heads, 128, 128), jnp.float32)
        t = timed(jax.jit(lambda x: x * 2), x)
        print(f"{2 * total / t / 1e9:.0f} GB/s ({t * 1e6:.0f} us)", flush=True)
    for heads, reads, writes, split, mode in [
        (16, 3, 2, 1, "read"),
        (16, 3, 2, 1, "write"),
        (16, 3, 2, 1, "copy"),
        (16, 4, 4, 1, "copy"),
        (16, 3, 2, 2, "copy"),
        (16, 3, 2, 4, "copy"),
        (16, 4, 4, 4, "copy"),
        (8, 4, 4, 1, "copy"),
        (8, 6, 6, 1, "copy"),
        (32, 3, 2, 1, "copy"),
        (32, 3, 3, 2, "copy"),
        (4, 8, 8, 1, "copy"),
    ]:
        n = total // (heads * 64 * 1024)
        x = jnp.ones((n, heads, 128, 128), jnp.float32)
        try:
            fn = stream(n, (heads, 128, 128), reads, writes, split, mode)
            t = timed(fn, x)
            moved = total * (1 if mode != "copy" else 2)
            print(
                f"{mode:5} chunk {heads * 64:5} KiB, {reads} reads / {writes} writes in flight, split {split}: "
                f"{moved / t / 1e9:6.0f} GB/s ({t * 1e6:7.0f} us)",
                flush=True,
            )
        except Exception as e:
            print(f"{mode} {heads} {reads} {writes} {split} failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}")


if __name__ == "__main__":
    main()
