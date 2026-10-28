"""Probe: does a TPU v5e copy faster when reads and writes don't overlap (HBM read/write turnarounds)?

A pool (N, 16, 128, 128) f32 (1 MiB chunks) is copied through VMEM in groups of G chunks with two group
buffers. "overlap": the next group's reads start while this group's writes run (a usual pipeline).
"burst": a group's writes start only once the next group's reads are done, and the reads after them only
once the writes are done, so at any time the DMAs in flight are all reads or all writes.

    python benchmarks/vs_vllm_kda/probe_burst.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MIB = 1 << 20
CHUNK = (16, 128, 128)


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


def copier(n, group, burst, work):
    groups = n // group

    def kernel(src, dst, bufs, rsem, wsem):
        def reads(g, slot):
            return [
                pltpu.make_async_copy(src.at[g * group + j], bufs.at[slot, j], rsem.at[slot, j]) for j in range(group)
            ]

        def writes(g, slot):
            return [
                pltpu.make_async_copy(bufs.at[slot, j], dst.at[g * group + j], wsem.at[slot, j]) for j in range(group)
            ]

        for c in reads(0, 0):
            c.start()
        for c in reads(0, 0):
            c.wait()

        @pl.loop(0, groups)
        def _(g):
            slot = jax.lax.rem(g, 2)
            nxt = 1 - slot
            if not burst:

                @pl.when(g >= 1)  # the other buffer's writes (group g - 1) before reading into it
                def _():
                    for c in writes(0, nxt):
                        c.wait()

            @pl.when(g + 1 < groups)
            def _():
                for c in reads(g + 1, nxt):
                    c.start()

            if work:  # some compute on this group while the reads run
                for j in range(group):
                    bufs[slot, j] = bufs[slot, j] * 1.0001 + 1e-3

            @pl.when(g + 1 < groups)
            def _():
                if burst:
                    for c in reads(g + 1, nxt):
                        c.wait()

            for c in writes(g, slot):
                c.start()
            if burst:  # the writes alone, then the next reads
                for c in writes(g, slot):
                    c.wait()
            else:

                @pl.when(g + 1 < groups)
                def _():
                    for c in reads(g + 1, nxt):
                        c.wait()

        if not burst:  # (group groups - 2's writes were waited in the last iteration)
            for c in writes(0, (groups - 1) % 2):
                c.wait()

    shape = (n, *CHUNK)
    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM((2, group, *CHUNK), jnp.float32),
                pltpu.SemaphoreType.DMA((2, group)),
                pltpu.SemaphoreType.DMA((2, group)),
            ],
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=110 * MIB),
        )
    )


def main():
    n = 512
    x = jnp.ones((n, *CHUNK), jnp.float32)
    t = timed(jax.jit(lambda x: x * 2), x)
    print(f"XLA x * 2: {2 * n * MIB / t / 1e9:.0f} GB/s ({t * 1e6:.0f} us)", flush=True)
    for work in (False, True):
        for group in (1, 2, 4, 8, 16):
            for burst in (False, True):
                try:
                    t = timed(copier(n, group, burst, work), x)
                    print(
                        f"{'burst  ' if burst else 'overlap'} groups of {group:2} MiB{' + compute' if work else ''}: "
                        f"{2 * n * MIB / t / 1e9:5.0f} GB/s ({t * 1e6:6.0f} us)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"group {group} burst {burst} failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    for n_small in (2, 8):  # a small batch: read it all, then write it all
        xs = jnp.ones((n_small, *CHUNK), jnp.float32)
        for group, burst in ((1, False), (1, True), (n_small, True)):
            t = timed(copier(n_small, group, burst, False), xs, runs=50)
            print(f"{n_small} MiB, groups of {group}, {'burst' if burst else 'overlap'}: {t * 1e6:.1f} us", flush=True)


if __name__ == "__main__":
    main()
