"""Tile masks: skipped tiles, and how skipped tiles get their value."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races

import autokernel


def make_causal(m, k, n, tm, tn, tk, dtype, **options):
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

    return autokernel.tpu(causal, **options)


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
def test_causal_mask_matches_tril(m, k, n, tm, tn, tk, dtype):
    kern = make_causal(m, k, n, tm, tn, tk, dtype)
    x, y = rand((m, k), dtype=dtype), rand((k, n), 1, dtype=dtype)
    kern.verify(x, y)
    want = jnp.tril(jnp.dot(x, y, precision="highest", preferred_element_type=jnp.float32)).astype(dtype)
    tol = dict(atol=1e-3, rtol=1e-3) if dtype == jnp.float32 else dict(atol=0.5, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(kern(x, y), np.float32), np.asarray(want, np.float32), **tol)


@pytest.mark.parametrize("m, n, tm, tn", [(512, 512, 128, 128), (256, 1024, 128, 256), (1024, 256, 256, 256)])
def test_causal_visits_only_active_tiles(m, n, tm, tn):
    lw = make_causal(m, 256, n, tm, tn, 128, jnp.float32).lower(rand((m, 256)), rand((256, n), 1))
    active = sum(j * tn <= i * tm + tm - 1 for i in range(m // tm) for j in range(n // tn))
    assert (lw.tiles, lw.total_tiles, lw.steps) == (active, (m // tm) * (n // tn), active * 2)
    assert lw.fill == ("background" if active < lw.total_tiles else None)  # skipped tiles get fill DMAs
    assert "input_output_aliases" not in lw.source


def test_if_and_continue_masks_without_reduction():
    @autokernel.tpu
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
        (lambda s: jnp.full(s, -1.0), -1.0),
        (lambda s: jnp.ones(s), 1.0),
        (lambda s: jnp.full(s, fill_value=-jnp.inf), -np.inf),
        (lambda s: jnp.zeros(s) - 2.5, -2.5),  # found by following the jaxpr, not the call's name
    ],
)
def test_background_fill_writes_masked_tiles(alloc, fill):
    # 256x1024 with 128 tiles: 3 computed tiles and 13 constant ones, so the in-flight limit kicks in.
    kern = make_masked_matmul(256, 256, 1024, 128, 128, 128, alloc)
    x, y = rand((256, 256)), rand((256, 1024), 1)
    lw = kern.lower(x, y)
    assert (lw.fill, lw.tiles, lw.total_tiles) == ("background", 3, 16)
    out = np.asarray(kern(x, y))
    assert np.all(out[:128, 128:] == fill) and np.all(out[128:, 256:] == fill)
    kern.verify(x, y)


@pytest.mark.parametrize("cores", [1, 2])
def test_background_fill_has_no_races(cores):
    kern = make_masked_matmul(512, 256, 512, 128, 128, 128, lambda s: jnp.full(s, -1.0), cores=cores)
    x, y = rand((512, 256)), rand((256, 512), 1)
    np.testing.assert_allclose(run_checking_races(kern, x, y), kern.reference(x, y), atol=1e-4, rtol=1e-4)


def test_prefill_fallbacks():
    x, y = rand((256, 256)), rand((256, 1024), 1)
    # Opted out: the alias keeps the allocation, filled before the kernel.
    off = make_masked_matmul(256, 256, 1024, 128, 128, 128, lambda s: jnp.full(s, 2.0), background_fill=False)
    assert off.lower(x, y).fill == "prefill" and "input_output_aliases" in off.source(x, y)
    off.verify(x, y)
    # Not a constant: the compiler can't know the skipped tiles' value, so it keeps the prefill.
    ramp = make_masked_matmul(
        256, 256, 1024, 128, 128, 128, lambda s: jnp.arange(s[0] * s[1], dtype=jnp.float32).reshape(s)
    )
    assert ramp.lower(x, y).fill == "prefill"
    ramp.verify(x, y)


def test_background_fill_without_reduction_and_full_dims():
    @autokernel.tpu
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


# --------------------------------------------------------------------------- what a mask means


def _masked_row_sums(tmp_path, cond, extra_param=""):
    """A kernel summing x's column blocks for each row block, skipping the steps where `cond` holds."""
    path = tmp_path / "masked.py"
    path.write_text(f"""
import jax.numpy as jnp
import numpy as np

PATTERN = np.array([[1, 0], [0, 0]])  # a module constant a static mask reads
W = 0


def masked(x{extra_param}):
    out = jnp.zeros((256, 128))
    for i in range(0, 256, 128):
        acc = jnp.zeros((128, 128))
        for k in range(0, 256, 128):
            if {cond}:
                continue
            acc += x[i : i + 128, k : k + 128]
        out = out.at[i : i + 128, :].set(acc)
    return out
""")
    spec = importlib.util.spec_from_file_location("masked", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return autokernel.tpu(module.masked)


def _ints(shape):
    return jnp.asarray(np.random.default_rng(0).integers(-4, 5, shape), jnp.float32)


@pytest.mark.parametrize(
    "cond",
    [
        "~(k // 128) == -2",  # `~` inverts ints (on bools too: `~True` is -2, not False)
        "k < i - (W or 512)",  # `or` used for its value
        "k > max(0, i - 128)",  # builtins on loop variables
        "abs(k - i) > 0 and min(i, k) >= 0",
        "k in (128,)",
        "np.sum(PATTERN[i // 128, :]) == 0",  # an aggregate per step, not over every step at once
        "k >= x.shape[0] // 2",  # an input's shape is known at trace time
    ],
)
def test_static_masks_mean_what_python_says(tmp_path, cond):
    kern, x = _masked_row_sums(tmp_path, cond), _ints((256, 256))
    assert kern.lower(x).schedule != "runtime"
    assert kern.verify(x) == 0.0


def test_runtime_masks_are_evaluated_per_step(tmp_path):
    """`jnp.any(m[k // 128, :])` reduces one row of the mask per step, not the whole mask at once."""
    kern = _masked_row_sums(tmp_path, "not jnp.any(m[k // 128, :]) or not bool(m[i // 128, k // 128])", ", m")
    x, m = _ints((256, 256)), jnp.asarray([[True, False], [False, False]])  # row 1 all False: k = 128 skipped
    assert kern.lower(x, m).schedule == "runtime"
    assert kern.verify(x, m) == 0.0
    np.testing.assert_array_equal(kern(x, m), np.concatenate([x[:128, :128], np.zeros((128, 128))]))


def test_runtime_mask_python_control_flow_is_rejected_at_lower_time(tmp_path):
    kern = _masked_row_sums(tmp_path, "int(m[i // 128, k // 128]) == 0", ", m")
    with pytest.raises(autokernel.AutoKernelError, match=r"line 14\).*reads inputs"):
        kern.lower(rand((256, 256)), jnp.ones((2, 2), bool))


def test_mask_reads_an_earlier_reduction_variable():
    """After its loop a reduction variable keeps its last value, in a later phase's mask too."""

    @autokernel.tpu
    def two_phases(x):
        out = jnp.zeros((256, 128))
        for i in range(0, 256, 128):
            acc = jnp.zeros((128, 128))
            for k in range(0, 256, 128):
                acc += x[i : i + 128, k : k + 128]
            acc2 = jnp.zeros((128, 128))
            for k2 in range(0, 256, 128):
                if k2 < k:  # k == 128 here
                    continue
                acc2 += x[i : i + 128, k2 : k2 + 128]
            out = out.at[i : i + 128, :].set(acc + 2 * acc2)
        return out

    assert two_phases.verify(_ints((256, 256))) == 0.0
