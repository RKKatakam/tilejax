"""Differentiating and batching through kernels.

A Pallas kernel with hand-written DMAs has no JAX derivative or batching rules, so each kernel is wrapped:

- By default in a `jax.custom_jvp` whose rule runs the kernel for the primal output and takes the
  tangent from the JVP of the kernel's Python loop nest, run as plain JAX. The loop nest is the kernel's
  specification, so its derivative is the right one; it is just computed by XLA (with the loops unrolled),
  not by a generated kernel. The tangent is linear in the input tangents, so JAX transposes it for
  reverse mode: `jax.jvp`, `jax.jacfwd`, `jax.grad`, `jax.jacrev` and `jax.linearize` all work, and the
  kernel runs once per primal evaluation in each of them.
- With a user `backward=` (typically other autokernel kernels, or a program), in a `jax.custom_vjp`
  whose backward pass is that function. Only reverse mode is then defined.

Inside either, a `jax.custom_batching.custom_vmap` gives the kernel its batching rule (see `batching`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

__all__ = ["VmapRule", "differentiable", "vmap_by"]

# custom_vmap rule: (axis_size, in_batched, *args) -> (outputs, outputs_batched)
VmapRule = Callable[..., tuple[Any, Any]]


def differentiable(
    forward: Callable[..., Any],
    reference: Callable[..., Any],
    backward: Callable | None,
    *,
    vmap_rule: VmapRule | None = None,
    fallback: Callable[..., Any] | None = None,
) -> Callable:
    """`forward` (the kernel) with derivatives, and with `vmap_rule` as its batching rule if given.

    `backward(*inputs, cotangent)` returns one gradient per input (None for inputs that get none); when
    it is None the derivatives (forward and reverse mode) are those of `reference`, which computes the
    same function as `forward`.

    `fallback` computes the same function as `forward` some other way that JAX can batch and shard (a
    fused kernel's nests, each run as its own kernel): it is used under `jax.vmap` (unless `vmap_rule`
    is given) and for inputs sharded over a mesh.
    """
    if fallback is not None:
        vmap_rule = vmap_rule or vmap_by(fallback)
        forward = _unless_sharded(forward, fallback)
    base = forward
    if vmap_rule is not None:
        base = jax.custom_batching.custom_vmap(forward)
        base.def_vmap(vmap_rule)
    if backward is None:
        return _jvp_of_reference(base, reference)

    @jax.custom_vjp
    def call(*args: Any) -> Any:
        return base(*args)

    def call_fwd(*args: Any) -> tuple[Any, tuple[Any, ...]]:
        return base(*args), args

    def call_bwd(args: tuple[Any, ...], cotangent: Any) -> tuple[Any, ...]:
        grads = backward(*args, cotangent)
        grads = tuple(grads) if isinstance(grads, (tuple, list)) else (grads,)
        if len(grads) != len(args):
            raise ValueError(f"backward returned {len(grads)} gradients for {len(args)} inputs")
        return tuple(_zero_or(g, a) for g, a in zip(grads, args, strict=True))

    call.defvjp(call_fwd, call_bwd)
    return call


def _jvp_of_reference(base: Callable[..., Any], reference: Callable[..., Any]) -> Callable:
    @jax.custom_jvp
    def call(*args: Any) -> Any:
        return base(*args)

    def call_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
        out = call(*primals)  # the kernel computes the primal output (differentiable again: higher orders)
        # Some tangent is nonzero: JAX doesn't call the rule when they all are symbolic zeros.
        live = [not isinstance(t, SymbolicZero) for t in tangents]

        def along_live(*xs: Any) -> Any:  # the reference as a function of the inputs with nonzero tangents
            it = iter(xs)
            return reference(*[next(it) if lv else p for p, lv in zip(primals, live, strict=True)])

        pick = [p for p, lv in zip(primals, live, strict=True) if lv]
        _, tangent = jax.jvp(along_live, pick, [t for t, lv in zip(tangents, live, strict=True) if lv])
        return out, jax.tree.map(_like, tangent, out)

    call.defjvp(call_jvp, symbolic_zeros=True)
    return call


def vmap_by(fn: Callable[..., Any]) -> VmapRule:
    """A batching rule that computes a batch with `jax.vmap(fn)`: for a kernel whose function `fn` also
    computes some other way (a fused kernel's nests run as kernels of their own, each with its own rule)."""

    def rule(axis_size: int, in_batched: Sequence[bool], *args: Any) -> tuple[Any, Any]:
        in_axes = [0 if b else None for b in in_batched]
        out = jax.vmap(fn, in_axes=in_axes, axis_size=axis_size)(*args)
        return out, jax.tree.map(lambda _: True, out)

    return rule


def _unless_sharded(forward: Callable[..., Any], fallback: Callable[..., Any]) -> Callable[..., Any]:
    from .sharding import find_mesh

    def call(*args: Any) -> Any:
        return (fallback if find_mesh(args) is not None else forward)(*args)

    return call


def _like(tangent: Any, primal: Any) -> Any:
    """`tangent` with the tangent type of `primal` (the reference may differ in weak typing)."""
    if jnp.issubdtype(primal.dtype, jnp.inexact):
        return jnp.asarray(tangent).astype(primal.dtype)
    return np.zeros(primal.shape, jax.dtypes.float0)


def _zero_or(grad: Any, primal: Any) -> Any:
    """A cotangent for `primal`: `grad` cast to its type, or zeros (float0 for integer inputs) for None."""
    if grad is None:
        return jax.tree.map(_zeros_like_primal, primal)
    return jnp.asarray(grad).astype(primal.dtype)


def _zeros_like_primal(x: Any) -> Any:
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros(x.shape, x.dtype)
    return np.zeros(x.shape, jax.dtypes.float0)
