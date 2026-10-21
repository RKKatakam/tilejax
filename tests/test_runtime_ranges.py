"""Runtime ranges (loop bounds that read index arrays: the kernel finds its steps, and with a runtime count
of tiles its tiles, as it runs), ragged row windows (`autokernel.rows`), windows gathered through index
array slices, and `autokernel.pairs`."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, spec

import autokernel
from autokernel import Config, KernelError, Pipeline


def i32(*values):
    return jnp.array(values, jnp.int32)


def check(kernel, *args, pipeline=None):
    """Run in the interpreter checking races (with `pipeline` if given), verify, cross-lower for a v5e."""
    if pipeline is None:
        out = run_checking_races(kernel, *args)
        lowered = kernel.lower(*args)
    else:
        lowered = kernel.lowered(kernel.signature(*args), Config(None, pipeline))
        out = lowered.run(*args)
    kernel.verify(*args)
    specs = [spec(*x.shape, dtype=x.dtype) for x in args]
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)
    return out, lowered.source


SUMS = """
def f(cache, lengths):
    B, S, D = cache.shape
    out = jnp.zeros((B, 8, D))
    for b in range(B):
        acc = jnp.zeros((8, D))
        for j in range(0, lengths[b], 128):  # the cache's blocks up to its length: counted as the kernel runs
            rows = j + jax.lax.broadcasted_iota(jnp.int32, (128, D), 0)
            acc += jnp.sum(jnp.where(rows < lengths[b], cache[b, j : j + 128, :], 0.0).reshape(16, 8, D), axis=0)
        out = out.at[b, :, :].set(acc)
    return out
"""


def _sums(cache, lengths):
    out = np.zeros((cache.shape[0], 8, cache.shape[2]))
    for b, n in enumerate(np.asarray(lengths)):
        rows = np.concatenate([f64(cache[b, :n]), np.zeros((-n % 8, cache.shape[2]))])
        out[b] = rows.reshape(-1, 8, cache.shape[2]).sum(0)
    return out


@pytest.mark.parametrize("pipeline", [None, Pipeline(4, 3), Pipeline(1, 1)])
def test_a_runtime_range_steps_through_each_tiles_blocks(pipeline):
    kernel = make_kernel(SUMS)
    cache, lengths = rand((4, 512, 128)), i32(0, 1, 300, 512)  # an empty range: one empty step
    out, src = check(kernel, cache, lengths, pipeline=pipeline)
    np.testing.assert_allclose(f64(out), _sums(cache, lengths), rtol=1e-5, atol=1e-4)
    assert "jax.lax.while_loop(lambda _: cursor[0, 0] < 4" in src and "def advance(row):" in src
    assert "step_tile" not in src  # no step tables
    assert kernel.lower(cache, lengths).schedule.cursor


def test_runtime_ranges_in_two_phases_with_a_start_and_a_static_phase():
    kernel = make_kernel(
        """
        def f(x, w, lo, hi):
            B, S, D = x.shape
            out = jnp.zeros((B, 8, D))
            for b in range(B):
                acc = jnp.zeros((8, D))
                for j in range(lo[b], hi[b], 8):  # from lo[b] (a multiple of 8) to hi[b]
                    acc += x[b, j : j + 8, :]
                for k in range(0, 256, 128):  # a static phase after it
                    acc += jnp.sum(w[k : k + 128, :].reshape(16, 8, D), axis=0)
                out = out.at[b, :, :].set(acc)
            return out
        """
    )
    x, w = rand((3, 64, 128)), rand((256, 128), 1)
    lo, hi = i32(0, 8, 16), i32(16, 8, 64)  # (an empty range in the middle)
    out, src = check(kernel, x, w, lo, hi)
    want = np.stack(
        [f64(x[b, int(a) : int(z)]).reshape(-1, 8, 128).sum(0) for b, (a, z) in enumerate(zip(lo, hi, strict=True))]
    )
    want = want + f64(w).reshape(-1, 8, 128).sum(0)
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    assert "later = ph + 1 < 2" in src


def test_a_runtime_range_indexing_with_its_value_and_a_tile_mask():
    kernel = make_kernel(
        """
        def f(x, n):
            B, S, _, D = x.shape
            out = jnp.zeros((B, 8, D))
            for b in range(B):
                if b == 1:
                    continue
                acc = jnp.zeros((8, D))
                for j in range(n[b]):
                    acc += x[b, j, :, :]
                out = out.at[b, :, :].set(acc)
            return out
        """
    )
    x, n = rand((3, 4, 8, 128)), i32(2, 4, 9)  # (9: past the dim, clamped as JAX clamps)
    out, _ = check(kernel, x, n)
    want = np.stack([f64(x[b, : min(int(c), 4)]).sum(0) for b, c in enumerate(n)])
    want[1] = 0
    want[2] += 5 * f64(x[2, 3])  # the clamped steps read the last block
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)


RAGGED = """
def f(q, rows, n):
    T, D, dtype = q.shape[0], q.shape[2], q.dtype
    out = q
    for k in range(n[0]):  # the tiles, counted as the kernel runs
        x = q[autokernel.rows(rows[k], rows[k + 1], 16), :, :]
        acc = jnp.zeros((16, 8, D))
        for j in range(0, 2 * D, D):
            acc += x.astype(jnp.float32) * 2
        out = out.at[autokernel.rows(rows[k], rows[k + 1], 16), :, :].set(acc.astype(dtype))
    return out
"""


@pytest.mark.parametrize("pipeline", [None, Pipeline(2, 1), Pipeline(3, 3)])
def test_ragged_rows_and_a_runtime_count_of_tiles(pipeline):
    kernel = make_kernel(RAGGED, autokernel=autokernel)
    q = rand((64, 8, 128), dtype=jnp.bfloat16)
    rows, n = i32(0, 5, 21, 37, 40, 40, 40), i32(4)  # tiles of 5, 16, 16 and 3 rows; rows 40.. untouched
    out, src = check(kernel, q, rows, n, pipeline=pipeline)
    want = f64(q).copy()
    want[:40] *= 4
    np.testing.assert_allclose(f64(out), want, rtol=1e-2)
    assert "n_tiles = jnp.maximum(" in src and "def _last_out0():" in src
    assert "pl.ds(jnp.clip(rows_smem[jnp.clip(k_blk, 0, 6)], 0, 64)" in src  # (a runtime tile: clamped)
    none, _ = check(kernel, q, rows, i32(0))  # no tiles at all
    np.testing.assert_array_equal(np.asarray(none), np.asarray(q))


def test_ragged_stores_of_static_tiles_drain_at_their_own_sizes():
    kernel = make_kernel(
        """
        def f(q, rows):
            out = q
            for k in range(3):
                acc = jnp.zeros((16, 8, 128))
                for j in range(0, rows[k + 1] - rows[k], 16):
                    acc += q[autokernel.rows(rows[k], rows[k + 1], 16), :, :] * 3
                out = out.at[autokernel.rows(rows[k], rows[k + 1], 16), :, :].set(acc)
            return out
        """,
        autokernel=autokernel,
    )
    q, rows = rand((48, 8, 128)), i32(0, 16, 20, 36)
    out, src = check(kernel, q, rows)
    want = f64(q).copy()
    want[:36] *= 3
    np.testing.assert_allclose(f64(out), want, rtol=1e-5)
    assert "out_copy((2,), 0).wait()" in src  # the last tiles' own copies


def test_ragged_reads_along_tiled_dims_copy_whole_tiles():
    kernel = make_kernel(
        """
        def f(x, y, n):
            out = jnp.zeros((8, 256))
            for i in range(0, 8, 8):
                a = x[autokernel.rows(n[0], n[1], 8), :]  # (20, 256): sublanes, 20 isn't whole tiles: as is
                b = y[0:8, autokernel.rows(n[2], n[3], 256)]  # (8, 512): lanes, rounded up to 128
                rows = jax.lax.broadcasted_iota(jnp.int32, (8, 256), 0)
                cols = jax.lax.broadcasted_iota(jnp.int32, (8, 256), 1)
                ok = (rows < n[1] - n[0]) & (cols < n[3] - n[2])  # (past the stops: unspecified)
                out = out.at[i : i + 8, :].set(jnp.where(ok, a + b, 0.0))
            return out
        """,
        autokernel=autokernel,
    )
    x, y, n = rand((20, 256)), rand((8, 512)), i32(4, 9, 128, 300)
    out, src = check(kernel, x, y, n)
    want = np.zeros((8, 256))
    want[:5, :172] = f64(x[4:9, :172]) + f64(y[:5, 128:300])
    np.testing.assert_allclose(f64(out), want, rtol=1e-6)
    assert "+ 127) // 128 * 128" in src and "+ 7) // 8 * 8" not in src


GATHER = """
def f(pages, table, w):
    B = table.shape[0] // 4
    out = jnp.zeros((B, 8, 128))
    for b in range(B):
        acc = jnp.sum(w[table[b * 4 : b * 4 + 2], :, :], axis=0) * 0  # (a tile's own gathered window)
        for p in range(0, 4, 2):  # two pages a step, gathered through the table
            acc += jnp.sum(pages[table[b * 4 + p : b * 4 + p + 2], :, :], axis=0) * w[table[0 : 0 + 2], :, :][0]
        out = out.at[b, :, :].set(acc)
    return out
"""


def test_windows_gathered_through_index_array_slices():
    kernel = make_kernel(GATHER)
    pages, w = rand((10, 8, 128)), rand((10, 8, 128), 1)
    table = i32(3, 1, 4, 1, 5, 9, 2, 6)
    out, src = check(kernel, pages, table, w)
    t = np.asarray(table).reshape(2, 4)
    want = np.stack([f64(pages)[t[b]].sum(0) * f64(w)[3] for b in range(2)])
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    assert "class _Copies:" in src and "pages_bufs.at[slot].at[1, :, :]" in src and "w2_buf.at[1, :, :]" in src


def test_nested_elements_in_windows_are_clamped():
    kernel = make_kernel(
        """
        def f(pages, table, first, n):
            out = jnp.zeros((2, 8, 128))
            for b in range(2):
                acc = jnp.zeros((8, 128))
                for p in range(n[b]):
                    acc += pages[table[first[b] + p], :, :]
                out = out.at[b, :, :].set(acc)
            return out
        """
    )
    pages, table, first, n = rand((6, 8, 128)), i32(5, 4, 3, 2), i32(0, 3), i32(2, 2)  # 3 + 1: past the table
    out, src = check(kernel, pages, table, first, n)
    want = np.stack([f64(pages[5]) + f64(pages[4]), f64(pages[2]) + f64(pages[2])])
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    assert "table_smem[jnp.clip(first_smem[b_blk] + kb, 0, 3)]" in src


PAIRS = """
def f(kv, table):
    out = jnp.zeros((2, 16, 128), jnp.float32)
    for b in range(2):
        acc = jnp.zeros((16, 128), jnp.float32)
        for p in range(2):
            words = autokernel.pairs(kv[table[b * 2 + p], :, :, :, :])  # (16, 4, 128)
            lows = jnp.stack([words[:, g, :] for g in range(4)])  # one kv head's words at a time: strided
            whole = autokernel.pairs(kv[table[b * 2 + p], :, :, :, :])[0]  # (indexed otherwise: loaded)
            k = jax.lax.bitcast_convert_type(lows << 16, jnp.float32)
            high = autokernel.pairs(kv[table[b * 2 + p], :, :, :, :])[:, 1, :] & jnp.uint32(0xFFFF0000)
            v = jax.lax.bitcast_convert_type(high, jnp.float32)
            acc += jnp.sum(k, axis=0) + v + jnp.sum(jax.lax.bitcast_convert_type(whole << 16, jnp.float32)) * 0
        out = out.at[b, :, :].set(acc)
    return out
"""


def test_pairs_reads_a_tiles_row_pairs_as_words():
    kernel = make_kernel(PAIRS, autokernel=autokernel)
    kv = rand((5, 16, 4, 2, 128), dtype=jnp.bfloat16)
    table = i32(4, 0, 2, 3)
    out, src = check(kernel, kv, table)
    x = f64(kv)
    t = np.asarray(table).reshape(2, 2)
    want = np.stack([sum(x[t[b, p], :, :, 0, :].sum(1) + x[t[b, p], :, 1, 1, :] for p in range(2)) for b in range(2)])
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-4)
    assert "kv_bufs.bitcast(jnp.uint32)" in src and "_strided(kv_words, g, 16, 4, interpret)" in src
    assert "import autokernel" not in src


WRITE = """
def f(cache, keys, values, table, page, first, stop, src, n):
    ps, dtype = cache.shape[1], cache.dtype
    out = cache
    for w in range(n[0]):  # a step per run of rows, counted as the kernel runs
        k = keys[autokernel.rows(src[w], src[w] + stop[w] - first[w], ps), :, :]
        v = values[autokernel.rows(src[w], src[w] + stop[w] - first[w], ps), :, :]
        bits = jax.lax.bitcast_convert_type(k.astype(jnp.float32), jnp.uint32) >> 16
        words = bits | (jax.lax.bitcast_convert_type(v.astype(jnp.float32), jnp.uint32) & jnp.uint32(0xFFFF0000))
        out = out.at[table[page[w]], autokernel.rows(first[w], stop[w], ps), :, :, :].set(
            autokernel.unpairs(words, dtype)
        )
    return out
"""


@pytest.mark.parametrize("pipeline", [None, Pipeline(2, 1), Pipeline(4, 3)])
def test_single_steps_counted_as_the_kernel_runs_store_words(pipeline):
    kernel = make_kernel(WRITE, autokernel=autokernel)
    cache = rand((6, 16, 4, 2, 128), dtype=jnp.bfloat16)
    keys, values = rand((32, 4, 128), dtype=jnp.bfloat16), rand((32, 4, 128), dtype=jnp.bfloat16)
    table = i32(5, 2, 0, 4)
    page, first, stop, src = i32(1, 3, 0, 0), i32(3, 0, 10, 0), i32(16, 7, 11, 0), i32(0, 13, 20, 0)
    out, src_ = check(kernel, cache, keys, values, table, page, first, stop, src, i32(3), pipeline=pipeline)
    want = f64(cache).copy()
    for p, a, b, r in [(2, 3, 16, 0), (4, 0, 7, 13), (5, 10, 11, 20)]:
        want[p, a:b, :, 0, :], want[p, a:b, :, 1, :] = f64(keys[r : r + b - a]), f64(values[r : r + b - a])
    np.testing.assert_array_equal(f64(out), want)
    assert "@pl.loop(0, n_tiles)" in src_ and "cursor" not in src_
    assert "_store_pairs(out_bufs, " in src_
    none, _ = check(kernel, cache, keys, values, table, page, first, stop, src, i32(0))
    np.testing.assert_array_equal(np.asarray(none), np.asarray(cache))


def test_single_steps_counted_as_the_kernel_runs_take_no_bursts():
    kernel = make_kernel(WRITE, autokernel=autokernel)
    cache = rand((6, 16, 4, 2, 128), dtype=jnp.bfloat16)
    keys = rand((32, 4, 128), dtype=jnp.bfloat16)
    args = (cache, keys, keys, i32(5, 2, 0, 4), *[i32(0, 0, 0, 0)] * 4, i32(1))
    with pytest.raises(KernelError, match="only in kernels whose tiles take one step, counted when built"):
        kernel.lowered(kernel.signature(*args), Config(None, Pipeline.bursts(2)))


@pytest.mark.parametrize(
    ("store", "message"),
    [
        ("autokernel.unpairs(x[i, :, :], jnp.bfloat16) * 2", "autokernel.unpairs(words, dtype) is a stored value"),
        ("autokernel.unpairs(x[i, :, :])", "autokernel.unpairs(words, dtype) is a stored value"),
        ("autokernel.unpairs(x[i, :, :], jnp.float32)", "stores a 16-bit tile of pairs of rows"),
    ],
)
def test_unpairs_errors(store, message):
    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((2, 8, 2, 128), {"jnp.float32" if "float32" in store else "jnp.bfloat16"})
            for i in range(2):
                out = out.at[i, :, :, :].set({store})
            return out
        """,
        autokernel=autokernel,
    )
    with pytest.raises(KernelError, match=re.escape(message)):
        kernel.lower(jax.random.bits(jax.random.key(0), (2, 8, 128), jnp.uint32))


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("a = autokernel.pairs(x[i : i + 16, :] + 1)", "autokernel.pairs takes a tile read of an input"),
        ("a = autokernel.pairs(x[i : i + 16, :])", "takes a 16-bit tile of pairs of rows (..., 2, d), not float32"),
    ],
)
def test_pairs_errors(code, message):
    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((32, 128))
            for i in range(0, 32, 16):
                {code}
                out = out.at[i : i + 16, :].set(x[i : i + 16, :])
            return out
        """,
        autokernel=autokernel,
    )
    with pytest.raises(KernelError, match=message.replace("(", r"\(").replace(")", r"\)")):
        kernel.lower(rand((32, 128)))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (  # a repeat loop's bounds
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for l in range(n[0]):\n            for j in range(0, 256, 128):\n"
            "                acc += x[i : i + 8, j : j + 128]\n        out = out.at[i : i + 8, :].set(acc)",
            "only reduction loops' bounds, or a kernel's one parallel loop's, may read index arrays",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(0, n[0], 128):\n            if j > 0:\n                continue\n"
            "            acc += x[i : i + 8, j : j + 128]\n        out = out.at[i : i + 8, :].set(acc)",
            "a loop with runtime bounds can't be masked",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(0, n[0], 0.5):\n            acc += x[i : i + 8, j : j + 128]\n"
            "        out = out.at[i : i + 8, :].set(acc)",
            "a loop with runtime bounds steps by a positive int, not 0.5",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(0, n[0], 128):\n            acc += x[i : i + 8, j : j + 128]\n"
            "        for j2 in range(0, 256, 128):\n            if j2 > 0:\n                continue\n"
            "            acc += x[i : i + 8, j2 : j2 + 128]\n        out = out.at[i : i + 8, :].set(acc)",
            "a kernel with runtime loop bounds can't mask its reduction loops",
        ),
        (
            "out = jnp.zeros((8, 256))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(0, n[0], 128):\n            acc += x[i : i + 8, j : j + 128]\n"
            "            out = out.at[i : i + 8, j : j + 128].set(acc)",
            "a kernel with runtime loop bounds stores its outputs per tile",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(n[0]):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(0, 256, 128):\n            acc += x[0:8, j : j + 128]\n"
            "        out = out.at[i, :, :].set(acc)",
            "stores only outputs updated in place",
        ),
        (
            "out = x\n    for i in range(n[0]):\n        if i > 0:\n            continue\n"
            "        acc = jnp.zeros((8, 128))\n        for j in range(0, 256, 128):\n"
            "            acc += x[0:8, j : j + 128]\n        out = out.at[0:8, 0:128].set(acc)",
            "a loop with runtime bounds can't be masked",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(n[0]):\n            acc += x[i : i + 8, 0:128]\n"
            "        for l in range(2):\n            for k in range(0, 256, 128):\n"
            "                acc += x[i : i + 8, k : k + 128]\n        out = out.at[i : i + 8, :].set(acc)",
            "a kernel with runtime loop bounds can't have repeat loops",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        acc = jnp.zeros((8, 128))\n"
            "        for j in range(2):\n            acc += x[n[j : j + 2], 0:128].sum(axis=0)\n"
            "        out = out.at[i : i + 8, :].set(acc)",
            "breaks the TPU layout",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n        b = jnp.zeros((), jnp.int32)\n"
            "        out = out.at[i : i + 8, :].set(x[autokernel.rows(b, 8, 8), 0:128])",
            "a window's index can't read `b`, which the tile code computes",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n"
            "        out = out.at[i : i + 8, :].set(x[autokernel.rows(n[0], 8, i + 8), 0:128])",
            "autokernel.rows's size is an int when planning",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n"
            "        out = out.at[i : i + 8, :].set(x[autokernel.rows(n[0], 8), 0:128])",
            "autokernel.rows takes (start, stop, size)",
        ),
        (
            "out = jnp.zeros((8, 128))\n    for i in range(0, 8, 8):\n"
            "        out = out.at[i : i + 8, :].set(x[n[0:2:1], 0:128].sum(axis=0))",
            "a window gathered through an index array slices one of its dims",
        ),
        (
            "out = jnp.zeros((16, 128))\n    for i in range(0, 8, 8):\n"
            "        out = out.at[autokernel.rows(n[0], n[1], 8), :].set(x[i : i + 8, 0:128])",
            "an index array places this tile, so no tile might store the rest of `out`",
        ),
        (
            "out = x\n    for i in range(0, 8, 8):\n        out = out.at[n[0 : 0 + 8], 0:128].set(x[i : i + 8, 0:128])",
            "an output tile isn't gathered through an index array",
        ),
        (
            "out = x\n    for i in range(0, 8, 8):\n"
            "        out = out.at[autokernel.rows(n[0], n[1], 8), 0:128].set(x[i : i + 8, 0:128])",
            "ragged rows of an output go in a dim before the last two",
        ),
        (
            "out = x\n    for i in range(0, 8, 8):\n        out = out.at[i : i + 8, :].set(x[nothing(i), 0:128])",
            "each index is `v:v",
        ),
        (
            "out = x\n    for i in range(0, 16, 8):\n        if i >= 0:\n            continue\n"
            "        acc = jnp.zeros((8, 256))\n        for j in range(n[0]):\n            acc += x[i : i + 8, :]\n"
            "        out = out.at[i : i + 8, :].set(acc)",
            "the masks skip every output tile",
        ),
    ],
)
def test_runtime_errors(source, message):
    kernel = make_kernel(f"def f(x, n):\n    {source}\n    return out\n", autokernel=autokernel)
    with pytest.raises(KernelError, match=message.replace("(", r"\(").replace(")", r"\)")):
        kernel.lower(rand((16, 256)), i32(0, 8, 1, 2, 3, 4, 5, 6, 7))


def test_the_reference_runs_runtime_ranges_eagerly():
    kernel = make_kernel(SUMS)
    cache, lengths = rand((2, 256, 128)), i32(130, 256)
    np.testing.assert_allclose(f64(kernel.reference(cache, lengths)), _sums(cache, lengths), rtol=1e-5, atol=1e-4)
    assert kernel.nest.runtime_masks
