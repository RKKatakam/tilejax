"""Tile masks: skipped tiles, compacted grids and how skipped tiles get their value."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel


def make_causal(m, k, n, tm, tn, tk, dtype, pipeline="manual"):
    def causal(x, y):
        out = jnp.zeros((m, n), dtype)
        for i in range(0, m, tm):
            for j in range(0, n, tn):
                if j > i + tm - 1:  # whole tile above the diagonal
                    continue
                acc = jnp.zeros((tm, tn), jnp.float32)
                for kk in range(0, k, tk):
                    acc += x[i : i + tm, kk : kk + tk] @ y[kk : kk + tk, j : j + tn]
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0)
                cols = j + jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 1)
                out = out.at[i : i + tm, j : j + tn].set(jnp.where(cols <= rows, acc, 0.0).astype(dtype))
        return out

    return autokernel.tpu(causal, pipeline=pipeline)


def make_masked_matmul(m, k, n, tm, tn, tk, alloc, **options):
    """Lower-triangular tiles of x @ y; skipped tiles keep whatever `alloc(shape)` holds."""

    def masked(x, y):
        out = alloc((m, n))
        for i in range(0, m, tm):
            for j in range(0, n, tn):
                if j > i + tm - 1:
                    continue
                acc = jnp.zeros((tm, tn))
                for kk in range(0, k, tk):
                    acc += x[i : i + tm, kk : kk + tk] @ y[kk : kk + tk, j : j + tn]
                out = out.at[i : i + tm, j : j + tn].set(acc)
        return out

    return autokernel.tpu(masked, **options)


@pytest.mark.parametrize(
    "m, k, n, tm, tn, tk",
    [
        (512, 256, 512, 128, 128, 128),  # square tiles
        (512, 256, 512, 256, 128, 128),  # tiles taller than wide
        (512, 256, 512, 128, 256, 128),  # tiles wider than tall
        (256, 256, 1024, 128, 256, 256),  # wide matrix, most tiles skipped
        (1024, 256, 256, 256, 256, 256),  # tall matrix, lower rows fully computed
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_causal_mask_matches_tril(m, k, n, tm, tn, tk, dtype, pipeline):
    kern = make_causal(m, k, n, tm, tn, tk, dtype, pipeline)
    x, y = rand((m, k), dtype=dtype), rand((k, n), 1, dtype=dtype)
    kern.verify(x, y)
    want = jnp.tril(jnp.dot(x, y, precision="highest", preferred_element_type=jnp.float32)).astype(dtype)
    tol = dict(atol=1e-3, rtol=1e-3) if dtype == jnp.float32 else dict(atol=0.5, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(kern(x, y), np.float32), np.asarray(want, np.float32), **tol)


@pytest.mark.parametrize("m, n, tm, tn", [(512, 512, 128, 128), (256, 1024, 128, 256), (1024, 256, 256, 256)])
def test_causal_visits_only_active_tiles(m, n, tm, tn, pipeline):
    lw = make_causal(m, 256, n, tm, tn, 128, jnp.float32, pipeline).lower(rand((m, 256)), rand((256, n), 1))
    active = sum(j * tn <= i * tm + tm - 1 for i in range(m // tm) for j in range(n // tn))
    assert (lw.active_tiles, lw.total_tiles, lw.steps) == (active, (m // tm) * (n // tn), active * 2)
    if pipeline == "grid":
        assert lw.grid == (active, 2)
        assert "PrefetchScalarGridSpec" in lw.source and "input_output_aliases" in lw.source
    else:
        assert lw.grid == () and "input_output_aliases" not in lw.source  # the kernel writes every tile


def test_if_and_continue_masks_without_reduction(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def tril_copy(x):
        out = jnp.zeros((512, 512))
        for i in range(0, 512, 128):
            for j in range(0, 512, 128):
                if j <= i:
                    rows = i + jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
                    cols = j + jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
                    out = out.at[i : i + 128, j : j + 128].set(
                        jnp.where(cols <= rows, x[i : i + 128, j : j + 128], 0.0)
                    )
        return out

    x = rand((512, 512))
    tril_copy.verify(x)
    np.testing.assert_array_equal(tril_copy(x), jnp.tril(x))
    assert tril_copy.lower(x).steps == 10


@pytest.mark.parametrize(
    "alloc, fill",
    [
        (lambda s: jnp.full(s, -1.0), -1.0),  # the interpreter's empty buffer (zeros) can't fake this
        (lambda s: jnp.ones(s), 1.0),
        (lambda s: jnp.full(s, fill_value=-jnp.inf), -np.inf),
        (lambda s: jnp.zeros(s) - 2.5, -2.5),  # found by following the jaxpr, not the call's name
    ],
)
def test_background_fill_writes_masked_tiles(alloc, fill, pipeline):
    # 256x1024 with 128 tiles: 3 computed tiles and 13 constant ones, so the in-flight limit kicks in.
    kern = make_masked_matmul(256, 256, 1024, 128, 128, 128, alloc, pipeline=pipeline)
    x, y = rand((256, 256)), rand((256, 1024), 1)
    lw = kern.lower(x, y)
    assert (lw.fill, lw.active_tiles, lw.total_tiles) == ("background", 3, 16)
    assert "_fill_copy(c).start()" in lw.source
    if pipeline == "grid":
        assert "jax.lax.empty" in lw.source
    out = np.asarray(kern(x, y))
    assert np.all(out[:128, 128:] == fill) and np.all(out[128:, 256:] == fill)
    kern.verify(x, y)


def test_background_fill_has_no_races(pipeline):
    from jax.experimental.pallas import tpu as pltpu

    kern = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.full(s, -1.0), pipeline=pipeline)
    x, y = rand((512, 256)), rand((256, 512), 1)
    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    out = kern.lower(x, y).entry(x, y, interpret=params).block_until_ready()
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    assert interpreter.races is not None and not interpreter.races.races_found
    np.testing.assert_allclose(out, kern.reference(x, y), atol=1e-4, rtol=1e-4)


def test_prefill_fallbacks(pipeline):
    x, y = rand((256, 256)), rand((256, 1024), 1)
    # Opted out: the alias keeps the allocation, filled before the kernel.
    off = make_masked_matmul(
        256, 256, 1024, 128, 128, 128, lambda s: jnp.full(s, 2.0), background_fill=False, pipeline=pipeline
    )
    assert off.lower(x, y).fill == "prefill" and "input_output_aliases" in off.source(x, y)
    off.verify(x, y)
    # Not a constant: the compiler can't know the skipped tiles' value, so it keeps the prefill.
    ramp = make_masked_matmul(
        256,
        256,
        1024,
        128,
        128,
        128,
        lambda s: jnp.arange(s[0] * s[1], dtype=jnp.float32).reshape(s),
        pipeline=pipeline,
    )
    assert ramp.lower(x, y).fill == "prefill"
    ramp.verify(x, y)


def test_background_fill_without_reduction_and_full_dims(pipeline):
    @autokernel.tpu(pipeline=pipeline)
    def first_rows(x):
        out = jnp.full((256, 256), 7.0)
        for i in range(0, 256, 32):
            if i < 96:
                out = out.at[i : i + 32, :].set(x[i : i + 32, :] * 2)
        return out

    x = rand((256, 256))
    lw = first_rows.lower(x)
    assert (lw.fill, lw.steps) == ("background", 3)
    out = np.asarray(first_rows(x))
    np.testing.assert_allclose(out[:96], np.asarray(x[:96]) * 2)
    assert np.all(out[96:] == 7.0)
