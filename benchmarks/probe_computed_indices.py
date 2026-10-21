"""Probe: can a Pallas TPU kernel turn a vector it computed (int32 ids, in vregs) into SMEM scalars, and use
them as DMA addresses in the same kernel? Ways tried: a VMEM -> SMEM DMA of a staging buffer, and
extracting each element as a scalar.

    python benchmarks/probe_computed_indices.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

N, ROWS, D = 8, 128, 128


def ids_of(scores):
    """(1, 128) f32 scores -> the indices of the N largest (1, 128) int32 (first N lanes), by repeated max."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    ids = jnp.zeros(scores.shape, jnp.int32)
    for k in range(N):
        best = jnp.max(scores, axis=1, keepdims=True)
        pick = jnp.min(jnp.where(scores == best, lanes, 128), axis=1, keepdims=True)
        ids = jnp.where(lanes == k, pick, ids)
        scores = jnp.where(lanes == pick, -jnp.inf, scores)
    return ids


def kernel_dma(scores_ref, table_hbm, out_ref, stage, ids_smem, rows, sems):
    """Stage the ids in VMEM, DMA them to SMEM, then copy the chosen rows of the table."""
    stage[...] = ids_of(scores_ref[...])
    copy = pltpu.make_async_copy(stage, ids_smem, sems.at[N])
    copy.start()
    copy.wait()
    for k in range(N):
        pltpu.make_async_copy(table_hbm.at[pl.ds(ids_smem[0, k], 1)], rows.at[pl.ds(k, 1)], sems.at[k]).start()
    for k in range(N):
        pltpu.make_async_copy(table_hbm.at[pl.ds(ids_smem[0, k], 1)], rows.at[pl.ds(k, 1)], sems.at[k]).wait()
    out_ref[...] = rows[...]


def kernel_extract(scores_ref, table_hbm, out_ref, stage, ids_smem, rows, sems):
    """Extract each id from the vector as a scalar into SMEM."""
    ids = ids_of(scores_ref[...])
    for k in range(N):
        ids_smem[0, k] = ids[0, k]
    for k in range(N):
        pltpu.make_async_copy(table_hbm.at[pl.ds(ids_smem[0, k], 1)], rows.at[pl.ds(k, 1)], sems.at[k]).start()
    for k in range(N):
        pltpu.make_async_copy(table_hbm.at[pl.ds(ids_smem[0, k], 1)], rows.at[pl.ds(k, 1)], sems.at[k]).wait()
    out_ref[...] = rows[...]


def run(body, scores, table, interpret):
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((N, D), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[
            pltpu.VMEM((1, 128), jnp.int32),
            pltpu.SMEM((1, 128), jnp.int32),
            pltpu.VMEM((N, D), jnp.float32),
            pltpu.SemaphoreType.DMA((N + 1,)),
        ],
        interpret=interpret,
    )(scores, table)


def main() -> None:
    interpret = jax.default_backend() != "tpu"
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.permutation(128)[None, :].astype(np.float32))
    table = jnp.asarray(rng.standard_normal((ROWS, D)), jnp.float32)
    want = np.asarray(table)[np.argsort(-np.asarray(scores)[0])[:N]]  # the rows of the N best lanes
    for name, body in (("vmem -> smem dma", kernel_dma), ("scalar extract", kernel_extract)):
        try:
            out = jax.jit(functools.partial(run, body, interpret=interpret))(scores, table)
            print(f"{name}: ok, matches: {np.allclose(np.asarray(out), want)}", flush=True)
        except Exception as e:
            print(f"{name}: failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)


if __name__ == "__main__":
    main()
