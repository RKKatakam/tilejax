"""Copies in bursts (pipeline.py): a single-step kernel fetches a burst of steps' inputs while the burst
before it computes, and copies a burst's outputs out once the next burst's inputs are in."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, spec
from jax.experimental.pallas import tpu as pltpu
from test_tune import Tiles, X, profile

import autokernel
from autokernel import Config, KernelError, Pipeline
from autokernel.tune import _bursts, _keep, _Stats, proposals

# Rows of a pool updated in place, through an index array; a weight read once; the rows scaled.
SCATTER = """
def f(x, w, pool, slots):
    B, H, D = x.shape
    new = pool
    out = jnp.zeros((B, H, D))
    for b in range(B):
        for h in range(0, H, 8):
            old = pool[slots[b], h : h + 8, :]
            out = out.at[b, h : h + 8, :].set(old * w[0:8, :])
            new = new.at[slots[b], h : h + 8, :].set(old + x[b, h : h + 8, :])
    return out, new
"""


def run(kernel, config, *args, trace=False):
    """The kernel with this schedule, in the interpreter with its race detector."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    lowered = kernel.lowered(kernel.signature(*args), config, trace=trace)
    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    out = jax.block_until_ready(lowered.entry(*args, interpret=params))
    assert interpreter.races is not None and not interpreter.races.races_found, "the kernel has data races"
    return lowered, out


@pytest.mark.parametrize("burst", [1, 2, 3, 4, 16])
def test_bursts_of_steps(burst):
    kernel = make_kernel(SCATTER)
    x, w, pool = rand((5, 16, 128)), rand((8, 128), 2), rand((11, 16, 128), 1)
    slots = jnp.array([5, 0, 2, 9, 7], jnp.int32)
    lowered, (out, new) = run(kernel, Config(None, Pipeline.bursts(burst)), x, w, pool, slots, trace=burst == 3)
    want_out, want_new = np.zeros(x.shape), f64(pool).copy()
    for b, s in enumerate(np.asarray(slots)):
        want_out[b], want_new[s] = f64(pool[s]) * np.tile(f64(w), (2, 1)), f64(pool[s]) + f64(x[b])
    np.testing.assert_allclose(f64(out), want_out, rtol=1e-6)
    np.testing.assert_allclose(f64(new), want_new, rtol=1e-6)
    src = lowered.source
    assert "def _next_burst():" in src and "def _end_burst():" in src and "_free_output_slot" not in src
    assert f"pltpu.VMEM(({2 * burst}, 8, 128), jnp.float32),  # pool_bufs" in src
    assert f"pltpu.VMEM(({burst}, 8, 128), jnp.float32),  # out_bufs" in src
    assert str(lowered.pipeline) == f"bursts of {burst} steps"
    specs = [spec(5, 16, 128), spec(8, 128), spec(11, 16, 128), spec(5, dtype=jnp.int32)]
    assert "tpu_custom_call" in lower_for_v5e(lowered, *specs)


def test_bursts_with_skipped_tiles():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((512, 128))
            for i in range(0, 512, 128):
                if i == 128:
                    continue
                out = out.at[i : i + 128, :].set(x[i : i + 128, :] + 1)
            return out
        """
    )
    x = rand((512, 128))
    _, out = run(kernel, Config(None, Pipeline.bursts(2)), x)
    want = f64(x) + 1
    want[128:256] = 0
    np.testing.assert_allclose(f64(out), want, rtol=1e-6)


def test_bursts_of_a_kernel_that_fetches_everything_once():
    kernel = make_kernel(ROWS, Rows=Rows)
    x = rand((256, 128))
    for t, steps in ((Rows(tm=128), 2), (Rows(tm=256), 1)):
        lowered, out = run(kernel, Config(t, Pipeline.bursts(2)), x, trace=True)
        assert lowered.steps == steps
        np.testing.assert_allclose(f64(out), 2 * f64(x), rtol=1e-6)


def test_bursts_need_single_step_tiles():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    x = rand((128, 256))
    with pytest.raises(KernelError, match="copies move in bursts only in kernels whose tiles take one step"):
        kernel.lowered(kernel.signature(x), Config(None, Pipeline.bursts(2)))


def test_burst_pipelines_are_checked():
    assert Pipeline.bursts(4) == Pipeline(8, 4, 4)
    with pytest.raises(KernelError, match="2 input buffers and 1 output buffer per step of a burst"):
        Pipeline(3, 2, 2)
    with pytest.raises(KernelError, match="bursts of 1 to 16 steps"):
        Pipeline.bursts(17)
    assert _bursts(0) == [16, 8, 4] and _bursts(8) == [16, 4] and _bursts(16) == [8] and _bursts(2) == [4]


@autokernel.tunable
class Rows:
    tm: int = autokernel.choice(64, 128, 256, default=64)


ROWS = """
def f(x, t=Rows()):
    out = jnp.zeros((256, 128))
    for i in range(0, 256, t.tm):
        out = out.at[i : i + t.tm, :].set(x[i : i + t.tm, :] * 2)
    return out
"""


def test_copy_bound_single_step_kernels_try_bursts():
    kernel = make_kernel(ROWS, Rows=Rows)
    sig = kernel.signature(rand((256, 128)))
    start = Config(Rows(), Pipeline())
    dma = autokernel.Trial(start, profile(start, seconds=10e-6, wait=5e-6))
    configs = proposals(kernel, sig, dma)
    bursts = [c.pipeline for c in configs if c.pipeline.burst]
    assert bursts == [Pipeline.bursts(16), Pipeline.bursts(8), Pipeline.bursts(4)]
    assert configs[0].pipeline == Pipeline(4, 2)  # more input buffers first
    # From a burst: twice and half as many steps, no more input buffers; the tiles keep the bursts.
    burst = Config(Rows(), Pipeline.bursts(8))
    configs = proposals(kernel, sig, autokernel.Trial(burst, profile(burst, seconds=10e-6, wait=5e-6)))
    assert [c.pipeline for c in configs[:2]] == [Pipeline.bursts(16), Pipeline.bursts(4)]
    assert configs[2:] == [Config(Rows(tm=128), Pipeline.bursts(4))]  # as many bytes in fewer steps
    configs = proposals(kernel, sig, autokernel.Trial(burst, profile(burst, seconds=10e-6)))
    assert configs == [Config(Rows(tm=128), Pipeline.bursts(4))]  # tiles twice as big: as many bytes a burst
    # Compute-bound: no bursts.
    compute = autokernel.Trial(start, profile(start, seconds=10e-6))
    assert not any(c.pipeline.burst for c in proposals(kernel, sig, compute))


def test_neighbours_whose_tiles_take_steps_drop_the_bursts():
    from test_tune import kernel as matmul

    k = matmul()
    sig = k.signature(X, rand((256, 128), 1))
    single = Config(Tiles(tm=128, tk=256), Pipeline.bursts(2))  # one k block: a step per tile
    configs = proposals(k, sig, autokernel.Trial(single, profile(single, seconds=10e-6)))
    stepped = next(c for c in configs if c.tunable == Tiles(tm=128, tk=128))
    assert stepped.pipeline == Pipeline(3, 2)
    assert next(c for c in configs if c.tunable == Tiles(tm=64, tk=256)).pipeline == Pipeline.bursts(4)


@autokernel.tunable
class BigRows:
    tm: int = autokernel.choice(1024, 4096, default=1024)


def test_neighbours_keep_the_longest_bursts_that_fit():
    kernel = make_kernel(
        """
        def f(x, t=BigRows()):
            out = jnp.zeros((16384, 512))
            for i in range(0, 16384, t.tm):
                out = out.at[i : i + t.tm, :].set(x[i : i + t.tm, :] * 2)
            return out
        """,
        BigRows=BigRows,
    )
    sig = kernel.signature(spec(16384, 512))
    start = Config(BigRows(), Pipeline.bursts(16))  # 2 MiB tiles: 48 slots, 96 MiB
    configs = proposals(kernel, sig, autokernel.Trial(start, profile(start, seconds=10e-6)))
    assert configs == [Config(BigRows(tm=4096), Pipeline.bursts(4))]  # 8 MiB tiles: as many bytes a burst
    plan, schedule = kernel.build(sig, BigRows(tm=4096))
    tiny = _Stats(steps=16, dma_bytes=16 * 2**26)  # steps of 64 MiB before: 16 MiB ones would need 64
    assert _keep(plan, schedule, Pipeline.bursts(16), tiny) == Pipeline.bursts(4)  # (16 is the most, 4 fits)
    assert _keep(plan, schedule, Pipeline(3, 2), tiny) == Pipeline(3, 2)
