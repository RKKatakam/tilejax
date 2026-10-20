"""Tunables: the one object a kernel gives autokernel to optimize over.

    @autokernel.tunable
    class MatmulTiles:
        tm: int = autokernel.choice(128, 256, 512, 1024, default=512)
        tn: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
        tk: int = autokernel.choice(128, 256, 512, 1024, default=512)

    @autokernel.kernel
    def matmul(x, y, tiles=MatmulTiles()):
        for i in range(0, M, tiles.tm): ...

A tunable is a frozen dataclass and a JAX pytree with no leaves (its fields are static), so a kernel stays
a plain JAX function: `matmul(x, y, MatmulTiles(tm=256))` runs exactly as written, under `jax.jit` too.
Called without it, autokernel tunes the fields over their choices (tune.py).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, dataclass_transform

import jax

__all__ = ["choice", "choices", "is_tunable", "neighbours", "tunable"]

_CHOICES = "autokernel.choices"


def choice(*values: Any, default: Any = None) -> Any:
    """A tunable field's candidate values, in order (neighbours are tried one step apart); its default is
    `default`, else the first value."""
    if not values:
        raise ValueError("a tunable field needs at least one value to choose from")
    default = values[0] if default is None else default
    if default not in values:
        raise ValueError(f"the default {default!r} isn't one of the choices {values}")
    return dataclasses.field(default=default, metadata={_CHOICES: tuple(values)})


@dataclass_transform(frozen_default=True, field_specifiers=(choice,))
def tunable[T: type](cls: T) -> T:
    """Make `cls` a tunable: a frozen dataclass whose fields autokernel may choose, registered as a JAX
    pytree with no leaves. Every field is declared with `autokernel.choice(...)`."""
    cls = dataclasses.dataclass(frozen=True)(cls)
    fields = dataclasses.fields(cls)
    for f in fields:
        if _CHOICES not in f.metadata:
            raise TypeError(f"{cls.__name__}.{f.name}: declare tunable fields with autokernel.choice(...)")
    cls.__autokernel_tunable__ = True  # type: ignore[attr-defined]
    jax.tree_util.register_dataclass(cls, data_fields=[], meta_fields=[f.name for f in fields])
    return cls


def is_tunable(value: Any) -> bool:
    """Is `value` an instance of an `@autokernel.tunable` class?"""
    return getattr(type(value), "__autokernel_tunable__", False)


def choices(value: Any) -> dict[str, tuple[Any, ...]]:
    """Each field of a tunable and the values it may take."""
    return {f.name: f.metadata[_CHOICES] for f in dataclasses.fields(value)}


def neighbours(value: Any) -> Iterator[Any]:
    """The tunables one step away from `value`: one field moved to the next or previous of its choices."""
    for name, options in choices(value).items():
        current = getattr(value, name)
        at = options.index(current) if current in options else None
        for step in (1, -1):
            if at is not None and 0 <= at + step < len(options):
                yield dataclasses.replace(value, **{name: options[at + step]})
