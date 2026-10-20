"""A stack of feed-forward layers as one kernel: a repeat loop over the layers, with cross-layer prefetch.

    for each layer l:  x = x + (silu(n @ W_gate[l]) * (n @ W_up[l])) @ W_down[l],   n = rmsnorm(x) * gamma[l]

`ffn_stack` runs every layer of a tile of rows in one kernel: the loop over layers is a repeat loop around
the loop over blocks of hidden units, so the layer's weights stream block by block, and the pipeline
fetches the next layer's first blocks while this layer's last ones are computed. The rows' activations
stay in VMEM from the first layer to the last. `ffn_layer` is one layer as a kernel of its own, called once
per layer: each call starts with its pipeline empty and ends by writing the rows to HBM.

With few rows (decoding) the layers are bound by streaming their weights from HBM, and the kernel
boundaries are what a stack in one kernel saves; with many rows they are bound by the MXU.

Run with `uv run python examples/layer_stack.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

import autokernel

EPS = 1e-6


@autokernel.tunable
class StackTiles:
    """Rows per tile, hidden units per step."""

    rows: int = autokernel.choice(64, 128, 256, 512, 1024, 2048, default=512)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, default=512)


def rmsnorm(x, gamma):
    """RMSNorm of f32 rows x with weights gamma (1, D)."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


@autokernel.kernel
def ffn_stack(x, gamma, w_gate, w_up, w_down, tiles=StackTiles()):
    """Every layer of a SwiGLU feed-forward stack, for (tokens, d_model) rows; weights stacked by layer."""
    T, D = x.shape
    L, _, F = w_up.shape
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        h = x[i : i + tm, :].astype(jnp.float32)
        for layer in range(L):  # the layers, one after another: a repeat loop
            n = rmsnorm(h, gamma[layer, :, :]).astype(dtype)
            y = jnp.zeros((tm, D), jnp.float32)
            for f in range(0, F, bf):
                up = n @ w_up[layer, :, f : f + bf]
                hidden = (jax.nn.silu(n @ w_gate[layer, :, f : f + bf]) * up).astype(dtype)
                y += hidden @ w_down[layer, f : f + bf, :]
            h = h + y
        out = out.at[i : i + tm, :].set(h.astype(dtype))
    return out


@autokernel.kernel
def ffn_layer(x, gamma, w_gate, w_up, w_down, tiles=StackTiles()):
    """One layer of the stack (gamma (1, D), weights of one layer), for (tokens, d_model) rows."""
    T, D = x.shape
    F = w_up.shape[1]
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        h = x[i : i + tm, :].astype(jnp.float32)
        n = rmsnorm(h, gamma[0:1, :]).astype(dtype)
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):
            up = n @ w_up[:, f : f + bf]
            hidden = (jax.nn.silu(n @ w_gate[:, f : f + bf]) * up).astype(dtype)
            y += hidden @ w_down[f : f + bf, :]
        out = out.at[i : i + tm, :].set((h + y).astype(dtype))
    return out


def split_layers(gamma, w_gate, w_up, w_down) -> list[tuple]:
    """The stacked weights as one tuple of arrays per layer (slicing a layer out of the stack under jit
    would copy its weights before every call)."""
    return [(gamma[i], w_gate[i], w_up[i], w_down[i]) for i in range(w_up.shape[0])]


def layer_by_layer(x, layers):
    """The stack as one `ffn_layer` call per layer, on `split_layers`' weights."""
    for weights in layers:
        x = ffn_layer(x, *weights)
    return x


def plain_jax(x, gamma, w_gate, w_up, w_down):
    """The stack in plain JAX, for XLA."""
    h = x.astype(jnp.float32)

    def dot(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    for layer in range(w_up.shape[0]):
        n = rmsnorm(h, gamma[layer]).astype(x.dtype)
        hidden = (jax.nn.silu(dot(n, w_gate[layer])) * dot(n, w_up[layer])).astype(x.dtype)
        h = h + dot(hidden, w_down[layer])
    return h.astype(x.dtype)


def make_inputs(tokens: int, d: int, f: int, layers: int, dtype: Any = jnp.bfloat16, seed: int = 0) -> tuple:
    keys = jax.random.split(jax.random.key(seed), 5)
    x = jax.random.normal(keys[0], (tokens, d), dtype)
    gamma = (1 + 0.1 * jax.random.normal(keys[1], (layers, 1, d))).astype(dtype)
    w_gate = (jax.random.normal(keys[2], (layers, d, f)) / d**0.5).astype(dtype)
    w_up = (jax.random.normal(keys[3], (layers, d, f)) / d**0.5).astype(dtype)
    w_down = (jax.random.normal(keys[4], (layers, f, d)) / f**0.5).astype(dtype)
    return x, gamma, w_gate, w_up, w_down


if __name__ == "__main__":
    inputs = make_inputs(128, 256, 512, 3)
    tiles = StackTiles(rows=128, block_f=256)
    print(ffn_stack.source(*inputs, tiles))
    print("max |kernel - loop nest|:", ffn_stack.verify(*inputs, tiles))
    diff = jnp.abs(ffn_stack(*inputs, tiles).astype(float) - plain_jax(*inputs).astype(float))
    print("max |stack - plain JAX|:", float(diff.max()))
