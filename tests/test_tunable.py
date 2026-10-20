"""Tunables: frozen pytrees of choices, the one object a kernel gives autokernel to optimize over."""

import dataclasses

import jax
import pytest

import autokernel
from autokernel.tunable import choices, is_tunable, neighbours


@autokernel.tunable
class Tiles:
    tm: int = autokernel.choice(128, 256, 512, default=256)
    tk: int = autokernel.choice(128, 256)


def test_a_tunable_is_a_frozen_dataclass_and_a_pytree_without_leaves():
    t = Tiles()
    assert (t.tm, t.tk) == (256, 128) and is_tunable(t) and not is_tunable(3) and not is_tunable(Tiles)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.tm = 512  # type: ignore[misc]
    assert jax.tree_util.tree_leaves(t) == [] and hash(t) == hash(Tiles())
    assert jax.jit(lambda x, tiles: x * tiles.tm)(2, Tiles(tm=512)) == 1024  # static under jit
    assert choices(t) == {"tm": (128, 256, 512), "tk": (128, 256)}


def test_neighbours_move_one_field_one_step():
    assert list(neighbours(Tiles())) == [Tiles(tm=512), Tiles(tm=128), Tiles(tk=256)]
    assert list(neighbours(Tiles(tm=384))) == [Tiles(tm=384, tk=256)]  # a value off the choices stays put


def test_fields_must_be_choices():
    with pytest.raises(TypeError, match=r"declare tunable fields with autokernel\.choice"):

        @autokernel.tunable
        class Bad:
            tm: int = 128

    with pytest.raises(ValueError, match="at least one value"):
        autokernel.choice()
    with pytest.raises(ValueError, match="isn't one of the choices"):
        autokernel.choice(1, 2, default=3)


def test_a_kernels_tunable_must_be_a_tunable():
    def f(x, tiles=(128, 128)):
        out = jax.numpy.zeros((8, 128))
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :])
        return out

    with pytest.raises(autokernel.KernelError, match=r"default must be an @autokernel\.tunable"):
        autokernel.kernel(f)
