"""Probe: variants of the prefill kernel's tile code (examples/ragged_paged_attention.py), each profiled on
prefill4x1024 on a TPU v5e at a few page counts per step, with its error against float64 NumPy.

    python benchmarks/vs_vllm_rpa/probe_prefill_variants.py [variant,...]
"""

from __future__ import annotations

import linecache
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent.parent / "examples")]

import jax
import jax.numpy as jnp
import numpy as np
import ragged_paged_attention as rpa
from run import workloads

import autokernel

SCALE = rpa.HEAD_DIM**-0.5

TEMPLATE = """
def rpa_prefill(q, kv_cache, page_indices, tile_page, tile_rows, tile_pos, tile_stop, n_tiles, tiles=PrefillTiles()):
    _, H, D = q.shape
    ps, Hkv = kv_cache.shape[1], kv_cache.shape[2]
    G, NP, BQ = H // Hkv, tiles.pages, BLOCK
    dtype = q.dtype
    out = q
    for k in range(n_tiles[0]):
        qh = QUERIES
        row_max = jnp.full((Hkv, BQ * G, LANES), MASK_VALUE, jnp.float32)
        row_sum = jnp.zeros((Hkv, BQ * G, LANES), jnp.float32)
        acc = jnp.zeros((Hkv, BQ * G, D), jnp.float32)
        for j in range(0, tile_stop[k], NP * ps):
            words = autokernel.pairs(
                kv_cache[page_indices[tile_page[k] + j // ps : tile_page[k] + j // ps + NP], :, :, :, :]
            )
STEP
        o = acc * jnp.tile(1.0 / jnp.maximum(row_sum, 1e-30), (1, 1, D // LANES))
        out = out.at[autokernel.rows(tile_rows[k], tile_rows[k + 1], BQ), :, :].set(token_heads(o, H).astype(dtype))
    return out
"""

QUERIES = "head_rows(q[autokernel.rows(tile_rows[k], tile_rows[k + 1], BQ), :, :], Hkv)"
SCALED_QUERIES = f"({QUERIES}.astype(jnp.float32) * sm_scale).astype(dtype)"

BATCHED = """
            kv = [keys_values(words[:, :, g, :]) for g in range(Hkv)]
            k_heads = jnp.stack([k_ for k_, _ in kv])
            v_heads = jnp.stack([v_ for _, v_ in kv])
            s = jax.lax.dot_general(qh, k_heads, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            SCORES
            row_max, row_sum, acc = online_softmax_heads(s, v_heads, row_max, row_sum, acc)
"""

PER_HEAD = """
            row_max, row_sum, acc = heads_one_by_one(qh, words, row_max, row_sum, acc, G, tile_pos[k], j, tile_stop[k])
"""
PER_HEAD_PRESCALED = PER_HEAD.replace("tile_stop[k])", "tile_stop[k], prescaled=True)")

MASKED = "s = jnp.where(visible(G, BQ * G, NP * ps, tile_pos[k], j, tile_stop[k]), s * sm_scale, MASK_VALUE)"
MASKED_PRESCALED = "s = jnp.where(visible(G, BQ * G, NP * ps, tile_pos[k], j, tile_stop[k]), s, MASK_VALUE)"
UNMASKED = "s = s * sm_scale"
COND = "s = masked_if_needed(s * sm_scale, G, tile_pos[k], j, tile_stop[k])"
COND_PRESCALED = "s = masked_if_needed(s, G, tile_pos[k], j, tile_stop[k])"


def masked_if_needed(s, g, first_row, first_col, stop):
    """s masked where `visible` says, if any column of the step is later than the tile's first query or
    past its stop (else every pair attends)."""
    _, rows, cols = s.shape
    needed = (first_col + cols - 1 > first_row) | (first_col + cols > stop)
    return jax.lax.cond(
        needed,
        lambda: jnp.where(rpa.visible(g, rows, cols, first_row, first_col, stop), s, rpa.MASK_VALUE),
        lambda: s,
    )


def heads_one_by_one(qh, words, row_max, row_sum, acc, g, first_row, first_col, stop, prescaled=False):
    """The step for one kv head after another (2-D arrays, as RPA's), the mask only where needed."""
    ms, ls, accs = [], [], []
    for h in range(qh.shape[0]):
        k, v = rpa.keys_values(words[:, :, h, :])
        s = jax.lax.dot_general(qh[h], k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        s = masked_if_needed((s if prescaled else s * SCALE)[None], g, first_row, first_col, stop)[0]
        m, total, a = rpa.online_softmax(s, v, row_max[h], row_sum[h], acc[h])
        ms.append(m)
        ls.append(total)
        accs.append(a)
    return jnp.stack(ms), jnp.stack(ls), jnp.stack(accs)


VARIANTS = {
    "base": (QUERIES, BATCHED, MASKED, 256),
    "prescaled": (SCALED_QUERIES, BATCHED, MASKED_PRESCALED, 256),
    "unmasked (wrong)": (QUERIES, BATCHED, UNMASKED, 256),
    "cond mask": (QUERIES, BATCHED, COND, 256),
    "cond mask prescaled": (SCALED_QUERIES, BATCHED, COND_PRESCALED, 256),
    "per head cond": (QUERIES, PER_HEAD, "", 256),
    "per head cond prescaled": (SCALED_QUERIES, PER_HEAD_PRESCALED, "", 256),
    "BQ 128": (QUERIES, BATCHED, MASKED, 128),
    "BQ 128 cond prescaled": (SCALED_QUERIES, BATCHED, COND_PRESCALED, 128),
}


def build(name: str, queries: str, step: str, scores: str, block: int) -> autokernel.Kernel:
    source = TEMPLATE.replace("QUERIES", queries).replace("STEP", step.strip("\n").replace("SCORES", scores))
    source = source.replace("BLOCK", str(block))
    filename = f"<variant {name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = {
        **vars(rpa),
        "sm_scale": SCALE,
        "masked_if_needed": masked_if_needed,
        "heads_one_by_one": heads_one_by_one,
    }
    exec(compile(source, filename, "exec"), namespace)
    return autokernel.kernel(namespace["rpa_prefill"])


def main() -> None:
    chosen = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    on_tpu = jax.default_backend() == "tpu"
    q_lens, kv_lens = workloads()["prefill4x1024"] if on_tpu else ([300, 200], [300, 600])
    inputs = rpa.make_inputs(q_lens, kv_lens, max_len=8192 if on_tpu else 1024)
    want, _ = rpa.reference(inputs, SCALE)
    q, k, v, cache, kv, pages, cu, dist = rpa.args_of(inputs)
    cache = rpa.write_new_tokens(cache, k, v, kv, pages, cu, dist)
    for name in chosen:
        queries, step, scores, block = VARIANTS[name]
        try:
            kernel = build(name, queries, step, scores, block)
            tiles = rpa.prefill_tiles(kv, pages, cu, dist, block_q=block, max_tokens=q.shape[0])
            for pages_per_step in (2, 4, 8):
                args = (q, cache, pages, *tiles.values(), rpa.PrefillTiles(pages=pages_per_step))
                out = kernel(*args)
                err = float(np.abs(np.asarray(out, np.float64) - want).max())
                if not on_tpu:
                    print(f"{name:28} pages {pages_per_step}: error vs f64 {err:.4f}", flush=True)
                    continue
                p = autokernel.profile(kernel, *args)
                print(
                    f"{name:28} pages {pages_per_step}: {p.device_s * 1e6:7.1f} us, {p.steps:3} steps, "
                    f"compute {p.compute_s * 1e6:7.1f} us; error vs f64 {err:.4f}",
                    flush=True,
                )
        except Exception as e:
            print(f"{name:28} failed: {type(e).__name__}: {str(e).splitlines()[0][:400]}", flush=True)


if __name__ == "__main__":
    main()
