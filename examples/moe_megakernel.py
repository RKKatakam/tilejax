"""A stack of mixture-of-experts layers decoding in ONE kernel, the routing inside it.

    for each layer l:  n = rmsnorm(h) * gamma[l]
                       experts, weights = top_k(softmax(n @ W_router[l]))        each token's top-k, renormalized
                       h = h + sum_e weights[:, e] * (silu(n @ W_gate[l, e]) * (n @ W_up[l, e])) @ W_down[l, e]

For a few tokens a layer is bound by reading the chosen experts' weights. moe.py's decode layer routes in
XLA, then runs one kernel over the experts some token chose (`moe_decode`): every layer is XLA ops, a
launch, a pipeline that starts empty. `moe_megakernel` runs every layer in one kernel, and computes each
layer's routing itself, from that layer's activations:

    one tile: all T tokens
      h = x                                     the residual stream, f32, in VMEM throughout
      REPEAT LOOP over the layers  (weights stacked by layer, (L, ...)):
        phase 1, d: the router, a block of its f32 rows per step (logits += n_blocks[d // bd] @ W_router)
           then, at its last step, the routing: `chosen` (the experts some token chose, in order), how many
           (`n`), and each token's weight for each: a COMPUTED INDEX ARRAY, copied into SMEM
        phase 2, c: the chosen experts, a block of hidden units per step, n[0] * F / be steps: the loop's
           bound and its windows (`w_gate[layer, chosen[c // F], :, c % F : c % F + be]`) read the routing

The kernel learns a layer's experts at its router's last step; their first blocks are fetched then (the
steps after wait for them), and from there the pipeline runs ahead as usual. The next layer's norm and
router stream while this layer's last experts run: cross-layer prefetch everywhere the routing allows it.

The routing is plain JAX that Mosaic compiles: the top-k is k max-reductions; an expert's slot is how many
chosen experts come before it (a (1, E) @ (E, E) triangular matmul, exact in f32); `chosen` and the weights
come from the one-hot (slots, E) of that rank, so no gathers or scatters. It routes as moe.py's
`route_logits` does (softmax, top-k, renormalized; f32 logits from the f32 router at full precision).

`megakernel_layers` runs the stack; `per_layer` is the same stack with moe.py's decode layer (XLA routing
and `moe_decode`) per layer; `xla_layers` with moe.py's plain-JAX "dense" layer.

Run with `uv run python examples/moe_megakernel.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import moe

import autokernel

EPS = 1e-6
TOP_K = 8  # experts per token (the kernel's constant: Qwen3-MoE's)


@dataclass(frozen=True)
class Config:
    """A stack of `layers` MoE layers (no shared experts, softmax routing renormalized: Qwen3-MoE's)."""

    layers: int
    d_model: int
    experts: int
    top_k: int
    ffn: int

    @property
    def moe(self) -> moe.MoeConfig:
        return moe.MoeConfig(self.d_model, self.experts, self.top_k, self.ffn)


@autokernel.tunable
class MoeTiles:
    """Rows of the router per step, hidden units of an expert per step."""

    block_d: int = autokernel.choice(256, 512, 1024, 2048, default=512)
    block_e: int = autokernel.choice(128, 256, 384, 512, 768, 1024, default=256)


def rmsnorm(x, gamma):
    """RMSNorm of f32 rows x with weights gamma (1, D)."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def lane_blocks(x, size):
    """(rows, n) as (n / size, rows, size): column blocks on a leading axis (read as `blocks[k]`)."""
    return jnp.stack([x[:, c : c + size] for c in range(0, x.shape[1], size)])


def route(logits, top_k, slots):
    """The layer's routing from its f32 logits (T, E): `chosen` (slots,) int32, the experts some token chose
    in increasing order (then zeros); `n` (1,) int32, how many; and each token's weight for each slot (T,
    slots) f32 (zero where it didn't choose it). Each token's top-k of the softmax, renormalized."""
    t, e = logits.shape
    scores = jax.nn.softmax(logits, axis=-1)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (t, e), 1)
    choice, weights = scores, jnp.zeros((t, e), jnp.float32)
    for _ in range(top_k):  # (ties to the lower expert, as moe.top_k)
        best = jnp.max(choice, axis=1, keepdims=True)
        pick = jnp.min(jnp.where(choice == best, lanes, e), axis=1, keepdims=True)
        weights = jnp.where(lanes == pick, scores, weights)
        choice = jnp.where(lanes == pick, -jnp.inf, choice)
    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-20)
    used = jnp.max((weights > 0).astype(jnp.float32), axis=0, keepdims=True)  # (1, E)
    before = jax.lax.broadcasted_iota(jnp.int32, (e, e), 0) < jax.lax.broadcasted_iota(jnp.int32, (e, e), 1)
    rank = (used @ before.astype(jnp.float32)).astype(jnp.int32)  # chosen experts before each one
    slot = jax.lax.broadcasted_iota(jnp.int32, (slots, e), 0)
    hot = (slot == rank) & (used > 0)  # (slots, E): slot s holds the s-th chosen expert
    chosen = jnp.sum(jnp.where(hot, jax.lax.broadcasted_iota(jnp.int32, (slots, e), 1), 0), axis=1)
    by_slot = jax.lax.dot_general(  # (T, slots); (in a helper, f32 at full precision only if asked)
        weights, hot.astype(jnp.float32), (((1,), (1,)), ((), ())), precision=jax.lax.Precision.HIGHEST
    )
    return chosen.astype(jnp.int32), jnp.sum(used).astype(jnp.int32).reshape(1), by_slot


@autokernel.kernel
def moe_megakernel(x, gamma, w_router, w_gate, w_up, w_down, tiles=MoeTiles()):
    """Every layer of a MoE stack for the T tokens x (T, D): gamma (L, 1, D), the f32 router (L, D, E),
    experts (L, E, D, F) and (L, E, F, D); TOP_K experts per token."""
    T, D = x.shape
    L, E, _, F = w_gate.shape
    C = min(E, T * TOP_K)  # the most experts the tokens can choose
    bd, be = min(tiles.block_d, D), min(tiles.block_e, F)
    assert F % be == 0  # a step's block of hidden units is one expert's
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, T):
        h = x[i : i + T, :].astype(jnp.float32)  # the residual stream, in VMEM from the first layer to the last
        for layer in range(L):  # a repeat loop
            normed = rmsnorm(h, gamma[layer, :, :]).astype(dtype)
            blocks = lane_blocks(normed.astype(jnp.float32), bd)
            logits = jnp.zeros((T, E), jnp.float32)
            for d in range(0, D, bd):  # phase 1: the router, bd rows a step
                logits += blocks[d // bd] @ w_router[layer, d : d + bd, :]
            chosen, n, weights = route(logits, TOP_K, C)  # computed index arrays: the kernel learns them here
            acc = jnp.zeros((T, D), jnp.float32)
            for c in range(0, n[0] * F, be):  # phase 2: the chosen experts' hidden units, be a step
                g = normed @ w_gate[layer, chosen[c // F], :, c % F : c % F + be]
                u = normed @ w_up[layer, chosen[c // F], :, c % F : c % F + be]
                slot = jax.lax.broadcasted_iota(jnp.int32, (T, C), 1) == c // F
                weight = jnp.sum(jnp.where(slot, weights, 0.0), axis=1, keepdims=True)  # (T, 1)
                acc += weight * (
                    (jax.nn.silu(g) * u).astype(dtype) @ w_down[layer, chosen[c // F], c % F : c % F + be, :]
                )
            h = h + acc
        out = out.at[i : i + T, :].set(h.astype(dtype))
    return out


# --------------------------------------------------------------------------- the stack, three ways


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, jax.Array]:
    """Random weights stacked by layer: gamma (L, 1, D), the f32 router (L, D, E), experts (L, E, D, F) and
    (L, E, F, D)."""
    c = config
    keys = iter(jax.random.split(jax.random.key(seed), 8))

    def normal(shape, scale, dt=dtype):
        return (jax.random.normal(next(keys), shape, jnp.float32) * scale).astype(dt)

    return {
        "gamma": (1 + 0.1 * jax.random.normal(next(keys), (c.layers, 1, c.d_model))).astype(dtype),
        "router": normal((c.layers, c.d_model, c.experts), c.d_model**-0.5, jnp.float32),
        "w_gate": normal((c.layers, c.experts, c.d_model, c.ffn), c.d_model**-0.5),
        "w_up": normal((c.layers, c.experts, c.d_model, c.ffn), c.d_model**-0.5),
        "w_down": normal((c.layers, c.experts, c.ffn, c.d_model), c.ffn**-0.5),
    }


def megakernel_layers(x, params, tiles=None):
    """The stack as one kernel: (T, D) -> (T, D)."""
    args = (x, params["gamma"], params["router"], params["w_gate"], params["w_up"], params["w_down"])
    return moe_megakernel(*args, *(() if tiles is None else (tiles,)))


def layer_params(params: dict[str, jax.Array], config: Config, gate_up: bool = False) -> list[dict[str, Any]]:
    """moe.py's per-layer params (separate arrays, no bias) for each layer (and with `gate_up` the experts'
    gate and up side by side, which its plain-JAX baseline reads)."""
    zero = jnp.zeros((config.experts,), jnp.float32)
    layers = []
    for layer in range(config.layers):
        p = {
            "norm": jnp.copy(params["gamma"][layer]),
            "router": jnp.copy(params["router"][layer]),
            "bias": zero,
            **{name: jnp.copy(params[name][layer]) for name in ("w_gate", "w_up", "w_down")},
        }
        layers.append(moe.baseline_params(p) if gate_up else p)
    return layers


def per_layer(x, layers, config: Config):
    """The stack with moe.py's decode layer per layer (XLA routing, then `moe_decode`)."""
    for p in layers:
        x = moe.moe_layer(x, p, config.moe)
    return x


def xla_layers(x, layers, config: Config):
    """The stack in plain JAX: moe.py's dense decode baseline (every expert on every token, one-hot weighted)
    per layer."""
    layer = moe.baseline(config.moe, method="dense")
    for p in layers:
        x = layer(x, p)
    return x


if __name__ == "__main__":
    config = Config(layers=2, d_model=512, experts=16, top_k=TOP_K, ffn=256)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (4, config.d_model), jnp.bfloat16)
    tiles = MoeTiles(block_d=256, block_e=128)
    args = (x, params["gamma"], params["router"], params["w_gate"], params["w_up"], params["w_down"])
    lowered = moe_megakernel.lower(*args, tiles)
    print(lowered.source)
    print("max |kernel - loop nest|:", moe_megakernel.verify(*args, tiles))
    ours = megakernel_layers(x, params, tiles)
    theirs = per_layer(x, layer_params(params, config), config)
    print("max |megakernel - per-layer kernels|:", float(jnp.max(jnp.abs(ours.astype(float) - theirs.astype(float)))))
