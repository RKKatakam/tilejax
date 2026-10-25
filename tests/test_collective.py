"""Kernels gathering an input across chips (`autokernel.all_gather`): a collective matmul on simulated meshes
of CPU devices, run in the Pallas TPU interpreter with its race detector, cross-lowered to Mosaic for a v5e-8,
and every rule a ring's tiles must follow."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from collective_matmul import CollectiveTiles, all_gather_matmul, inputs, sharded
from helpers import V5E, f64, make_kernel, spec
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import autokernel
from autokernel import tune as tune_module
from autokernel.pipeline import Pipeline
from autokernel.tune import Config

pytestmark = pytest.mark.cpu  # (simulated chips)

ROWS = P("x", None)  # x: a shard of the rows per chip
COLS = P(None, "x")  # w, and the output: a shard of the columns per chip


def mesh_of(chips: int, rows: int = 1) -> Mesh:
    devices = np.array(jax.devices()[: chips * rows])
    return Mesh(devices.reshape(rows, chips), ("y", "x")) if rows > 1 else Mesh(devices, ("x",))


def run_checking_races(kernel: autokernel.Kernel, mesh: Mesh, specs: tuple[P, ...], out: P, *args: Any) -> Any:
    """Run a kernel on every chip of a simulated mesh, in the interpreter with its race detector and
    uninitialized memory as NaN (a shard read before it arrived would be NaN)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    arrays, extra = args[: len(specs)], args[len(specs) :]

    def body(*xs: Any) -> Any:
        return kernel.lower(*xs, *extra).entry(*xs, interpret=params)

    run = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=out, check_vma=False))
    got = jax.block_until_ready(run(*arrays))
    assert interpreter.races is not None and not interpreter.races.races_found, "the kernel has data races"
    return got


def matmul_inputs(chips: int, m: int, K: int, n: int, dtype: Any = jnp.float32) -> tuple[jax.Array, jax.Array]:
    x = jax.random.normal(jax.random.key(0), (chips * m, K), dtype)
    w = jax.random.normal(jax.random.key(1), (K, chips * n), dtype)
    return x, w


# ---------------------------------------------------------------------------------------------- the ring


@pytest.mark.parametrize("chips", [8, 4, 2, 1])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_all_gather_matmul_on_a_ring(chips, dtype):
    x, w = matmul_inputs(chips, 32, 256, 256, dtype)
    tiles = CollectiveTiles(tm=16 if dtype == jnp.bfloat16 else 8, tn=128, tk=128)
    got = run_checking_races(all_gather_matmul, mesh_of(chips), (ROWS, COLS), COLS, x, w, tiles)
    assert got.shape == (chips * 32, chips * 256) and got.dtype == dtype
    np.testing.assert_allclose(f64(got), f64(x) @ f64(w), rtol=2e-2 if dtype == jnp.bfloat16 else 1e-4, atol=0.5)


def test_rings_along_one_axis_of_a_2d_mesh():
    # A 2 x 4 mesh: each row of 4 chips is a ring of its own (x is replicated across "y").
    mesh = mesh_of(4, rows=2)
    x, w = matmul_inputs(4, 32, 256, 128)
    got = run_checking_races(all_gather_matmul, mesh, (ROWS, COLS), COLS, x, w, CollectiveTiles(tm=8, tn=128, tk=128))
    np.testing.assert_allclose(f64(got), f64(x) @ f64(w), rtol=1e-4, atol=1e-3)


def test_verify_compares_every_chips_outputs():
    mesh = mesh_of(4)
    x, w = inputs(mesh, 4 * 64, 256, 4 * 128, jnp.float32)
    assert all_gather_matmul.verify(x, w) < 1e-3  # the default tiles (clamped to the shard)
    assert all_gather_matmul.verify(x, w, CollectiveTiles(tm=16, tn=128, tk=128)) < 1e-3
    out = sharded(mesh)(x, w)
    np.testing.assert_allclose(f64(out), f64(x) @ f64(w), rtol=1e-4, atol=1e-3)
    with pytest.raises(autokernel.KernelError, match="global arrays sharded over one mesh"):
        all_gather_matmul.verify(np.asarray(x), w)


def test_the_function_is_plain_jax():
    mesh = mesh_of(4)
    x, w = inputs(mesh, 4 * 32, 128, 4 * 128, jnp.float32)
    fn = jax.shard_map(all_gather_matmul.fn, mesh=mesh, in_specs=(ROWS, COLS), out_specs=COLS, check_vma=False)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(f64(jax.jit(fn)(x, w)), f64(x) @ f64(w), rtol=1e-4, atol=1e-3)


def test_the_source_shows_the_ring():
    with jax.set_mesh(mesh_of(8)):
        lowered = all_gather_matmul.lower(spec(32, 256), spec(256, 128), CollectiveTiles(tm=16, tn=128, tk=128))
    (ring,) = lowered.plan.rings
    assert (ring.name, ring.source, ring.axis, ring.size, ring.rows, ring.loop, ring.shard_blocks) == (
        "xs",
        "x",
        "x",
        8,
        32,
        "i",
        2,
    )
    source = lowered.source
    for line in (
        "me = jax.lax.axis_index('x')",
        "pltpu.get_barrier_semaphore()",
        "collective_id=0",
        "xs_send_own().start()",
        "jax.lax.fori_loop(xs_arrived[0] + 1, r + 1, receive, 0)",
        "i_blk = jax.lax.rem(me + 8 - (i_blk >> 1), 8) * 2 + (i_blk & 1)",
        "pltpu.SemaphoreType.DMA((8,)),  # xs_recv_sems",
        "xs_arrive(7)",
        "xs_send(6).wait_send()",
    ):
        assert line in source, line
    assert lowered.steps == 8 * 2 * 2  # 8 shards of 2 row blocks, 2 k-blocks each (one column block)


def test_cross_lowers_to_mosaic_for_a_v5e_8():
    for shape, names, dtype in (((8,), ("x",), jnp.bfloat16), ((2, 4), ("y", "x"), jnp.float32)):
        auto = (jax.sharding.AxisType.Auto,) * len(shape)
        mesh = jax.sharding.AbstractMesh(shape, names, axis_types=auto, abstract_device=V5E)
        chips = dict(zip(names, shape, strict=True))["x"]

        def native(x, w):
            return all_gather_matmul.lower(x, w).entry(x, w, interpret=False)

        with jax.sharding.use_abstract_mesh(mesh):
            run = jax.jit(jax.shard_map(native, mesh=mesh, in_specs=(ROWS, COLS), out_specs=COLS, check_vma=False))
            traced = run.trace(spec(chips * 512, 2048, dtype=dtype), spec(2048, chips * 1024, dtype=dtype))
            text = traced.lower(lowering_platforms=("tpu",)).as_text()
        assert "tpu_custom_call" in text


def test_one_window_per_tile_and_one_step_tiles():
    # No reduction loop: a tile takes one step and reads its rows of xs once, shared by the next tiles of the
    # same rows (the column loop doesn't move it).
    kernel = make_kernel(
        """
        def gathered_rows_matmul(x, w):
            xs = autokernel.all_gather(x, "x")
            M, K = xs.shape
            N = w.shape[1]
            out = jnp.zeros((M, N), jnp.float32)
            for i in range(0, M, 8):
                for j in range(0, N, 128):
                    out = out.at[i : i + 8, j : j + 128].set(xs[i : i + 8, :] @ w[:, j : j + 128])
            return out
        """,
        autokernel=autokernel,
    )
    x, w = matmul_inputs(4, 16, 128, 256)
    got = run_checking_races(kernel, mesh_of(4), (ROWS, COLS), COLS, x, w)
    np.testing.assert_allclose(f64(got), f64(x) @ f64(w), rtol=1e-4, atol=1e-3)


def test_copies_in_bursts():
    kernel = make_kernel(
        """
        def gathered_scale(x):
            xs = autokernel.all_gather(x, "x")
            M, N = xs.shape
            out = jnp.zeros((M, N), jnp.float32)
            for i in range(0, M, 8):
                out = out.at[i : i + 8, :].set(xs[i : i + 8, :] * 2.0)
            return out
        """,
        autokernel=autokernel,
    )
    mesh = mesh_of(4)
    x = jax.random.normal(jax.random.key(0), (4 * 16, 128), jnp.float32)
    with jax.set_mesh(mesh):
        signature = kernel.signature(jax.ShapeDtypeStruct((16, 128), jnp.float32))
        lowered = kernel.lowered(signature, Config(None, Pipeline.bursts(2)))
    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    run = jax.shard_map(
        lambda x: lowered.entry(x, interpret=params), mesh=mesh, in_specs=ROWS, out_specs=P(), check_vma=False
    )
    got = jax.jit(run)(x)
    np.testing.assert_array_equal(np.asarray(got), 2 * np.asarray(x))


def test_tuning_runs_the_default_schedule(monkeypatch):
    # On a TPU, tuning would time candidates outside jax.shard_map: a kernel gathering across chips keeps its
    # default schedule.
    monkeypatch.setattr(tune_module, "_can_time", lambda: True)
    with jax.set_mesh(mesh_of(2)):
        tuning = all_gather_matmul.tuning(spec(256, 512), spec(512, 512))
    assert tuning.config.tunable == CollectiveTiles() and not tuning.trials


# ---------------------------------------------------------------------------------------------- the rules


GATHER = 'xs = autokernel.all_gather(x, "x")'


def kernel_with(prologue: str = GATHER, tile: str = "xs[i : i + 8, :]", **more: str) -> Any:
    loops = more.get("loops", "for i in range(0, M, 8):")
    mask = more.get("mask", "")
    allocate = more.get("allocate", "jnp.zeros((M, 128), jnp.float32)")
    return make_kernel(
        f"""
        def k(x, lengths):
            {prologue}
            M = 4 * x.shape[0]
            out = {allocate}
            {loops}
                {mask}
                out = out.at[i : i + 8, :].set({tile} * 2.0)
            return out
        """,
        autokernel=autokernel,
    )


@pytest.mark.parametrize(
    ("prologue", "message"),
    [
        ('xs, ys = autokernel.all_gather(x, "x")', "as a statement of its own"),
        ("xs = autokernel.all_gather(x, 0)", "as a statement of its own"),
        ('xs = autokernel.all_gather(x, axis_name="x")', "as a statement of its own"),
        ('xs = autokernel.all_gather(y, "x")', "as a statement of its own"),
        ('x = autokernel.all_gather(x, "x")', "gather into a name of its own"),
        ('xs = autokernel.all_gather(x, "x"); ys = autokernel.all_gather(x, "x")', "gathers one input"),
        ('xs = autokernel.all_gather(x, "x"); xs = xs', "don't assign it again"),
    ],
)
def test_the_statement(prologue, message):
    with pytest.raises(autokernel.KernelError, match=message):
        kernel_with(prologue)


def plan_error(message: str, chips: int = 4, x: Any = (8, 128), **kernel: str) -> None:
    k = kernel_with(**kernel)
    with jax.set_mesh(mesh_of(chips)), pytest.raises(autokernel.KernelError, match=message):
        k.lower(spec(*x), jax.ShapeDtypeStruct((4,), jnp.int32))


def test_only_autokernels_all_gather():
    plan_error("with autokernel.all_gather", prologue='xs = jax.lax.all_gather(x, "x")')


def test_no_mask_skips_a_tile():
    plan_error("can't be masked", mask="if i > 8:\n                    continue")


def test_tiles_take_the_outer_loops_rows():
    plan_error("take its rows `i : i \\+ 8`", tile="xs[0:8, :]")  # rows no loop moves
    k = make_kernel(  # rows computed from the loop
        """
        def k(x):
            xs = autokernel.all_gather(x, "x")
            M = xs.shape[0]
            out = jnp.zeros((M // 2, 128), jnp.float32)
            for i in range(0, M // 2, 8):
                out = out.at[i : i + 8, :].set(xs[2 * i : 2 * i + 8, :])
            return out
        """,
        autokernel=autokernel,
    )
    with jax.set_mesh(mesh_of(4)), pytest.raises(autokernel.KernelError, match="take its rows"):
        k.lower(spec(8, 128))


def test_the_outer_loop_walks_every_row():
    plan_error("walks all of `xs`'s 32 rows from 0", loops="for i in range(0, M - 8, 8):")


def test_the_outer_loop_steps_within_a_shard():
    k = make_kernel(
        """
        def k(x):
            xs = autokernel.all_gather(x, "x")
            M = xs.shape[0]
            out = jnp.zeros((M, 128), jnp.float32)
            for i in range(0, M, 16):
                out = out.at[i : i + 16, :].set(xs[i : i + 16, :])
            return out
        """,
        autokernel=autokernel,
    )
    with jax.set_mesh(mesh_of(2)), pytest.raises(autokernel.KernelError, match="must divide one chip's 24 rows"):
        k.lower(spec(24, 128))


def test_something_must_read_it():
    plan_error("no tile reads it", tile="x[0:8, :]")


def test_the_gathered_input_is_read_as_tiles():
    plan_error("kernel code reads tiles of it", tile="xs[i, 0]")


def test_the_source_is_an_input_read_as_tiles():
    k = make_kernel(
        """
        def k(lengths):
            ls = autokernel.all_gather(lengths, "x")
            M = ls.shape[0]
            out = jnp.zeros((M,), jnp.int32)
            for i in range(0, M, 128):
                out = out.at[i : i + 128].set(ls[i : i + 128] + lengths[0])
            return out
        """,
        autokernel=autokernel,
    )
    with jax.set_mesh(mesh_of(4)), pytest.raises(autokernel.KernelError, match="is an index array"):
        k.lower(jax.ShapeDtypeStruct((128,), jnp.int32))


def test_nothing_updates_them_in_place():
    plan_error("can't be updated in place too", allocate="x", x=(32, 128))
    plan_error("an output can't update it in place", allocate="xs", x=(8, 128))


def test_all_gather_is_a_statement_of_its_own():
    plan_error("a prologue statement of its own", prologue='xs = jnp.asarray(autokernel.all_gather(x, "x"))')


def test_the_mesh_names_the_axis():
    k = kernel_with()
    with pytest.raises(autokernel.KernelError, match=r"call it inside jax.shard_map.*there is none"):
        k.lower(spec(8, 128), jax.ShapeDtypeStruct((4,), jnp.int32))
    other = Mesh(np.array(jax.devices()[:4]), ("model",))
    with jax.set_mesh(other), pytest.raises(autokernel.KernelError, match=r"its axes are \['model'\]"):
        k.lower(spec(8, 128), jax.ShapeDtypeStruct((4,), jnp.int32))
