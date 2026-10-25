"""`autokernel.all_gather`: an input gathered across the chips of a mesh axis, as kernel code reads it.

In a kernel's prologue, `xs = autokernel.all_gather(x, "x")` is every chip's `x`, stacked along its first
dim in the order of the mesh axis `"x"` (`jax.lax.all_gather(x, "x", tiled=True)`): the kernel function,
run as plain JAX inside `jax.shard_map`, means exactly that. The kernel reads tiles of `xs` like those of
any input, but no chip ever holds it before the kernel starts: its rows travel a ring of the axis's chips
while the kernel computes on the rows that have arrived (codegen.py), a collective matmul.
"""

from __future__ import annotations

from typing import Any

import jax

from .errors import KernelError

__all__ = ["all_gather", "axis_size"]


def all_gather(x: Any, axis_name: str) -> Any:
    """Every chip's `x` along the mesh axis `axis_name`, stacked along the first dim in the axis's order:
    in a kernel's prologue, `xs = autokernel.all_gather(x, "x")`."""
    if isinstance(x, jax.ShapeDtypeStruct):
        raise KernelError(
            "autokernel.all_gather is a prologue statement of its own, `xs = autokernel.all_gather(x, axis)`, "
            "on an input of the kernel"
        )
    return jax.lax.all_gather(x, axis_name, axis=0, tiled=True)


def axis_size(axis_name: str, kernel: str) -> int:
    """How many chips the mesh axis `axis_name` has, from the mesh the kernel is called under (inside
    `jax.shard_map`, or `jax.set_mesh(mesh)` to build it outside)."""
    mesh = jax.sharding.get_abstract_mesh()
    sizes = dict(mesh.shape)
    if axis_name not in sizes:
        known = f"its axes are {sorted(sizes)}" if sizes else "there is none"
        raise KernelError(
            f"{kernel} gathers over the mesh axis {axis_name!r}: call it inside jax.shard_map over a mesh with that "
            f"axis (or build it under jax.set_mesh(mesh)); {known}"
        )
    return int(sizes[axis_name])
