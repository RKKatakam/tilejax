"""`autokernel.tile`: block sizes declared in the prologue, tunable like module constants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, tiles
from autokernel import tune as tune_lib


def matmul(x, y):
    M, K = x.shape
    N = y.shape[1]
    TM = autokernel.tile(M, 256)
    TN = autokernel.tile(N, 256)
    TK = autokernel.tile(K, 256)
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc)
    return out


def test_tile_is_the_block_or_the_whole_extent():
    assert autokernel.tile(1000, 256) == 256
    assert autokernel.tile(100, 256) == 100
    assert autokernel.tile(np.int64(512)) == 512
    for bad in ((0, 4), (8, 0), (8.0, 4), (True, 4), (8, "4")):
        with pytest.raises(AutoKernelError, match="must be a positive int"):
            autokernel.tile(*bad)


def test_untuned_declarations_are_their_value():
    kern = autokernel.tpu(matmul)
    x, y = rand((512, 384)), rand((384, 256), 1)
    lw = kern.lower(x, y)
    assert lw.plan.spec.consts["TM"] == 256 and lw.plan.spec.consts["TK"] == 256 and lw.plan.spec.consts["TN"] == 256
    kern.verify(x, y)


def test_auto_tunes_declared_block_sizes():
    kern = autokernel.tpu(matmul, tune="auto")
    x, y = rand((512, 1024)), rand((1024, 512), 1)
    lw = kern.lower(x, y)
    assert set(lw.overrides) == {"TM", "TN", "TK"}
    assert {c.overrides["TK"] for c in kern.tuning(x, y)} >= {128, 256, 512, 1024}
    kern.verify(x, y)


def test_explicit_values_for_a_declared_block_size():
    kern = autokernel.tpu(matmul, tune={"TK": (128, 512)})
    x, y = rand((256, 512)), rand((512, 256), 1)
    assert {c.overrides["TK"] for c in kern.tuning(x, y)} == {128, 512}
    kern.verify(x, y)


def test_the_reference_uses_the_tuned_value():
    def scaled(x):
        (M,) = x.shape[:1]
        T = autokernel.tile(M, 128)
        out = jnp.zeros((M, 128))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * T)  # the output depends on the block size
        return out

    kern = autokernel.tpu(scaled, tune={"T": (256,)})
    x = rand((256, 128))
    np.testing.assert_allclose(kern(x), 256 * np.asarray(x), rtol=1e-6)
    np.testing.assert_allclose(kern.reference(x), 256 * np.asarray(x), rtol=1e-6)
    kern.verify(x)


def test_other_prologue_names_still_cant_be_tuned():
    def computed(x):
        M = x.shape[0]
        T = min(M, 128)
        out = jnp.zeros((M, 128))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :])
        return out

    with pytest.raises(AutoKernelError, match=r"declare it as a block size to tune it: `T = autokernel.tile"):
        autokernel.tpu(computed, tune={"T": (64,)}).lower(rand((256, 128)))
    with pytest.raises(AutoKernelError, match=r"autokernel.tile\(extent, block\)"):
        autokernel.tpu(computed, tune={"T": "auto"}).lower(rand((256, 128)))


def test_declarations_are_found_however_tile_is_spelled():
    import ast

    from autokernel import tile as imported

    src = """
A = autokernel.tile(M, 8)
B = tile(M, 8)
C = other.tile(M, 8)
D = missing.tile(M, 8)
E = autokernel.tile
F, G = autokernel.tile(M), 2
H = f(M)
"""
    stmts = ast.parse(src).body
    other = type("Other", (), {"tile": staticmethod(lambda *a: 1)})()
    found = tiles.declarations(stmts, {"autokernel": autokernel, "tile": imported, "other": other})
    assert sorted(found) == ["A", "B"]


def test_a_sharded_kernel_keeps_its_tuned_declared_block(monkeypatch):
    """The per-shard kernel gets the tuned values as constants: its declarations mustn't recompute them."""
    kern = autokernel.tpu(matmul, tune={"TM": (128,)})
    x, y = rand((256, 256)), rand((256, 256), 1)
    lowered = kern.lower(x, y)
    ctx, _ = kern._bind((x, y), {})
    plan = type("Plan", (), {"loops": (("i", ("d",), 2),)})()
    monkeypatch.setattr(
        "autokernel.sharding.local_nest", lambda spec, shard_plan: (ctx.nest, {})
    )  # the nest as is: only its environment matters here
    local = kern._local_kernel(ctx, lowered, plan)
    assert local.lower(x, y).plan.spec.consts["TM"] == 128
    local.verify(x, y)


def test_nearest_current_reads_declared_values():
    nest = autokernel.tpu(matmul).nest
    specs = [jax.ShapeDtypeStruct((512, 512), jnp.float32)] * 2
    from autokernel import lowering

    options = lowering.Options(tune=lowering.Options.make_tune({"TM": (128, 256, 512)}))
    space = tune_lib.Space(nest, matmul, specs, options, tune_lib.grid(nest, matmul, specs, options))
    assert space.overrides(space._nearest_current()) == {"TM": 256}
    failing = tune_lib._prologue_env(nest, matmul, [jax.ShapeDtypeStruct((5,), jnp.float32)] * 2)  # `M, K = x.shape`
    assert "TM" not in failing and "x" not in failing  # just the constants
