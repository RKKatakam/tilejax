"""Step order and tables: the steps a masked kernel runs are exactly the loop nest's, in its order."""

import itertools

import numpy as np
import pytest
from helpers import make_kernel, spec

from autokernel import KernelError


def schedule_of(source: str, *inputs, **namespace):
    kernel = make_kernel(source, **namespace)
    return kernel.build(kernel.signature(*inputs), None)[1]


BANDED = """
def f(x):
    out = jnp.zeros((B, 512, 512))
    for b in range(B):
        for i in range(0, 512, 128):
            for j in range(0, 512, 128):
                if j > i:
                    continue
                acc = jnp.zeros((128, 128))
                for k in range(0, 512, 128):
                    if k < i - 128:
                        continue
                    acc += x[b, i : i + 128, k : k + 128]
                out = out.at[b, i : i + 128, j : j + 128].set(acc)
    return out
"""


def test_tables_walk_exactly_the_nests_steps():
    sch = schedule_of(BANDED, spec(3, 512, 512), B=3)
    want = [
        ((b, i, j), k)
        for b, i, j in itertools.product(range(3), range(4), range(4))
        if j <= i
        for k in range(4)
        if k >= i - 1
    ]
    got = []
    for step in range(sch.steps):
        tile, phase, k = sch.step_at(step)
        assert phase == 0
        got.append((sch.blocks_at(tile), k))
    assert got == want
    assert sch.tiles == 30 and sch.total_tiles == 48 and sch.fills == 18 and sch.repeats == 3
    assert [r.var for r in sch.outer] == ["b"] and [r.var for r in sch.inner] == ["i", "j"]


def test_without_masks_steps_are_arithmetic():
    sch = schedule_of(
        BANDED.replace("if j > i:", "if False:").replace("if k < i - 128:", "if False:"), spec(2, 512, 512), B=2
    )
    assert sch.tables == {} and sch.steps_per_tile == 4  # masks that skip nothing make no tables
    sch = schedule_of(
        """
        def f(x):
            out = jnp.zeros((256, 256))
            for i in range(0, 256, 128):
                for j in range(0, 256, 128):
                    out = out.at[i : i + 128, j : j + 128].set(x[i : i + 128, j : j + 128])
            return out
        """,
        spec(256, 256),
    )
    assert (sch.steps, sch.steps_per_tile, sch.tables) == (4, 1, {})
    assert [sch.blocks_at(t) for t in range(4)] == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize(
    ("tile_mask", "step_mask", "message"),
    [
        ("True", "False", "the masks skip every output tile"),
        ("False", "undefined", "can't evaluate `undefined`"),
        ("False", "jnp.ones(2) > 0", "masks are Python conditions on loop variables"),
    ],
)
def test_mask_errors(tile_mask, step_mask, message):
    source = f"""
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                if {tile_mask}:
                    continue
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    if {step_mask}:
                        continue
                    acc += x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    with pytest.raises(KernelError, match=message):
        schedule_of(source, spec(256, 256))


@pytest.mark.parametrize("skip", ["True", "i == 0", "i == 128 and k == 0"])
def test_a_phase_whose_mask_skips_all_of_a_tiles_blocks_takes_one_empty_step(skip):
    from helpers import f64, rand, run_checking_races

    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                acc = jnp.full((128, 128), 2.0)
                for k in range(0, 256, 128):
                    if {skip}:
                        continue
                    acc += x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(acc + 1)
            return out
        """
    )
    x = rand((256, 256))
    sch = kernel.lower(x).schedule
    runs = {(i, k) for i in (0, 128) for k in (0, 128) if not eval(skip, {"i": i, "k": k})}
    assert sch.empty == (len({i for i, _ in runs}) < 2)
    steps = [(sch.step_at(s), sch.empty_at(s)) for s in range(sch.steps)]
    for i in (0, 128):
        mine = [k for (tile, _, k), empty in steps if tile == i // 128 and not empty]
        assert mine == [k // 128 for ii, k in sorted(runs) if ii == i]
    out = run_checking_races(kernel, x)
    want = np.full((256, 128), 3.0)
    for i, k in runs:
        want[i : i + 128] += f64(x[i : i + 128, k : k + 128])
    np.testing.assert_allclose(f64(out), want, rtol=1e-5, atol=1e-5)
    kernel.verify(x)


def test_values_a_masked_phase_makes_for_the_code_after_it_are_carried():
    # The phase may take an empty step, whose body doesn't run: the code after it reads them from VMEM.
    source = """
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    if k > i:
                        continue
                    acc += x[i : i + 128, k : k + 128]
                    twice = acc * 2
                out = out.at[i : i + 128, :].set(twice)
            return out
        """
    from helpers import f64, rand, run_checking_races

    kernel = make_kernel(source)
    x = rand((256, 256))
    assert set(kernel.lower(x).plan.carried) == {"acc", "twice"}
    want = 2 * np.concatenate([f64(x[:128, :128]), f64(x[128:, :128]) + f64(x[128:, 128:])])
    np.testing.assert_allclose(f64(run_checking_races(kernel, x)), want, rtol=1e-5, atol=1e-5)


def test_tables_must_fit_in_smem():
    source = """
        def f(x):
            out = jnp.zeros((4096, 65536))
            for i in range(0, 4096, 8):
                for j in range(0, 65536, 128):
                    if (i // 8 + j // 128) % 2:
                        continue
                    out = out.at[i : i + 8, j : j + 128].set(1.0)
            return out
        """
    with pytest.raises(KernelError, match="KiB of SMEM"):
        schedule_of(source, spec(8, 128))


def test_step_arithmetic_stays_in_int32():
    source = """
        def f(x):
            out = jnp.zeros((2**20, 256, 128))
            for b in range(2**20):
                for i in range(0, 256, 128):
                    if i:
                        continue
                    out = out.at[b, i : i + 128, :].set(1.0)
            return out
        """
    with pytest.raises(KernelError, match="int32 step arithmetic"):
        schedule_of(source, spec(8, 128))


def test_a_causal_matmul_of_one_tile():
    from causal_matmul import causal_matmul
    from helpers import rand, run_checking_races
    from matmul import MatmulTiles

    tiles = MatmulTiles(tm=256, tn=256, tk=128)
    x, y = rand((256, 256)), rand((256, 256), 1)
    run_checking_races(causal_matmul, x, y, tiles)
    causal_matmul.verify(x, y, tiles)
    assert causal_matmul.lower(x, y, tiles).schedule.tables == {}


def test_phases_decode_arithmetically():
    from test_codegen import TWO_PASS

    sch = schedule_of(TWO_PASS, spec(256, 384), spec(384, 256))
    assert not sch.step_tables and sch.phase_counts == (3, 3) and sch.phase_steps == (6, 6)
    want = [(tile, phase, k) for tile in range(2) for phase in range(2) for k in range(3)]
    assert [sch.step_at(s) for s in range(sch.steps)] == want
