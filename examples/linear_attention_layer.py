"""Qwen3-Next's Gated DeltaNet layer (its linear-attention token mixer), prefill, as autokernel kernels composed
in plain JAX on (batch, seq, d_model) activations:

    h            = rmsnorm(x)                                         (the decoder layer's input norm)
    z, q, k, v   = h @ w_qkvz                                         (heads side by side, 128 columns each)
    b, a         = h @ w_ba                                           (a beta and a decay per value head)
    q, k, v      = silu(causal_conv1d(q, k, v))                       (depthwise, 4 taps)
    beta, g      = sigmoid(b), -exp(A_log) * softplus(a + dt_bias)
    o            = gated_delta_rule(q, k, v, g, beta)                 (q and k L2-normalized, q scaled)
    out          = x + (rmsnorm_per_head(o) * norm_weight * silu(z)) @ w_out

Qwen3-Next has fewer query/key heads than value heads (16 and 32 in Qwen3-Next-80B-A3B); each query/key head
serves `v_heads // k_heads` consecutive value heads (`repeat_interleave`).

Kernels: the norm and the projections are `rmsnorm_matmul` (examples/rmsnorm_matmul.py), the norm fused
into each matmul; the chunked delta rule with the gated RMSNorm fused at the end of each block of tokens is
`gated_delta_rule_gated` (examples/linear_attention.py); the output projection and the residual are
`residual_matmul`. The short convolution, the gates and the query/key head repeat stay in plain JAX, one
fused XLA pass between the projection and the delta rule: a convolution over tokens can't join the
projection kernel (its row tiles would need the rows before them), and the delta rule kernel reads its
inputs a block of tokens at a time, without the three tokens before each block.

The projection's columns are laid out [z | q | k | v] (the checkpoint's are interleaved per key head; a
fixed permutation of w_qkvz's columns), so the delta-rule kernel reads z straight from the projection's
output: its first columns.

`baseline_layer` is the same layer in plain JAX, compiled by XLA, with `baseline_chunked` for the delta rule.

Run with `uv run python examples/linear_attention_layer.py`; without a TPU it runs in the Pallas TPU
interpreter.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from linear_attention import HEAD_DIM, baseline_chunked, gated_delta_rule_gated, gated_rmsnorm
from matmul import MatmulTiles
from rmsnorm_matmul import rmsnorm_matmul, rmsnorm_then_matmul

import autokernel

CONV = 4  # taps of the short convolution
GATES = 128  # columns of the gate projection: b and a for each value head, padded to a lane tile


@dataclass(frozen=True)
class Config:
    d_model: int
    k_heads: int  # query/key heads
    v_heads: int  # value heads, a multiple of k_heads, at most GATES // 2

    @property
    def key_dim(self) -> int:
        return self.k_heads * HEAD_DIM

    @property
    def value_dim(self) -> int:
        return self.v_heads * HEAD_DIM


@autokernel.kernel
def residual_matmul(x, a, w, tiles=MatmulTiles()):
    """x + a @ w: an output projection and the residual after it."""
    M, K = a.shape
    N = w.shape[1]
    tm, tn, tk = min(tiles.tm, M), min(tiles.tn, N), min(tiles.tk, K)
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            acc = jnp.zeros((tm, tn), jnp.float32)
            for k in range(0, K, tk):
                acc += a[i : i + tm, k : k + tk] @ w[k : k + tk, j : j + tn]
            out = out.at[i : i + tm, j : j + tn].set(
                (x[i : i + tm, j : j + tn].astype(jnp.float32) + acc).astype(dtype)
            )
    return out


def prepare(proj: jax.Array, gates: jax.Array, params: dict[str, Any], config: Config) -> tuple[jax.Array, ...]:
    """From the projections, (B, S, value_dim + 2 key_dim + value_dim) [z | q | k | v] and (B, S, GATES)
    [b | a | 0]: the delta rule's q, k, v (B, S, value_dim), after the convolution and SiLU, q and k repeated
    for each value head; its log decays g and betas (B, v_heads, S) f32."""
    B, S, _ = proj.shape
    hv, r = config.v_heads, config.v_heads // config.k_heads
    qkv = jnp.pad(proj[..., config.value_dim :].astype(jnp.float32), ((0, 0), (CONV - 1, 0), (0, 0)))
    w = params["conv"].astype(jnp.float32)  # (CONV, 2 key_dim + value_dim): tap j weighs token t - CONV + 1 + j
    conv = jax.nn.silu(sum(w[j] * qkv[:, j : j + S] for j in range(CONV))).astype(proj.dtype)
    q, k, v = jnp.split(conv, [config.key_dim, 2 * config.key_dim], axis=-1)

    def repeat(x):  # each query/key head for the r value heads after it
        return jnp.repeat(x.reshape(B, S, config.k_heads, HEAD_DIM), r, axis=2).reshape(B, S, config.value_dim)

    gates = gates.astype(jnp.float32)
    beta = jax.nn.sigmoid(gates[..., :hv])
    g = -jnp.exp(params["A_log"]) * jax.nn.softplus(gates[..., hv : 2 * hv] + params["dt_bias"])
    return repeat(q), repeat(k), v, g.transpose(0, 2, 1), beta.transpose(0, 2, 1)


def gated_deltanet_layer(x: jax.Array, params: dict[str, Any], config: Config) -> jax.Array:
    """The layer with autokernel kernels: x (B, S, d_model) -> x + the token mixer's output."""
    B, S, D = x.shape
    rows = x.reshape(B * S, D)
    proj = rmsnorm_matmul(rows, params["norm"], params["w_qkvz"]).reshape(B, S, -1)
    gates = rmsnorm_matmul(rows, params["norm"], params["w_ba"]).reshape(B, S, GATES)
    q, k, v, g, beta = prepare(proj, gates, params, config)
    o = gated_delta_rule_gated(q, k, v, g, beta, proj, params["out_norm"])  # z: proj's first value_dim columns
    return residual_matmul(rows, o.reshape(B * S, config.value_dim), params["w_out"]).reshape(B, S, D)


@functools.partial(jax.jit, static_argnames="config")
def _baseline_mixer(x, params, config):
    B, S, D = x.shape
    rows = x.reshape(B * S, D)
    proj = rmsnorm_then_matmul(rows, params["norm"], params["w_qkvz"]).reshape(B, S, -1)
    gates = rmsnorm_then_matmul(rows, params["norm"], params["w_ba"]).reshape(B, S, GATES)
    q, k, v, g, beta = prepare(proj, gates, params, config)
    o = baseline_chunked(q, k, v, g, beta).reshape(B * S, config.value_dim)
    gated = gated_rmsnorm(o, proj[..., : config.value_dim].reshape(B * S, -1), params["out_norm"]).astype(x.dtype)
    out = rows.astype(jnp.float32) + jnp.dot(gated, params["w_out"], preferred_element_type=jnp.float32)
    return out.astype(x.dtype).reshape(B, S, D)


def baseline_layer(x: jax.Array, params: dict[str, Any], config: Config) -> jax.Array:
    """The same layer in plain JAX (XLA), with the chunked delta rule of `baseline_chunked`."""
    return _baseline_mixer(x, params, config)


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights at the scales Qwen3-Next initializes them (A in [1, 16], dt_bias 1)."""
    d, kd, vd, hv = config.d_model, config.key_dim, config.value_dim, config.v_heads
    keys = jax.random.split(jax.random.key(seed), 7)
    w_ba = jnp.zeros((d, GATES)).at[:, : 2 * hv].set(jax.random.normal(keys[1], (d, 2 * hv)) / d**0.5)
    return {
        "norm": (1 + 0.1 * jax.random.normal(keys[0], (1, d))).astype(dtype),
        "w_qkvz": (jax.random.normal(keys[2], (d, 2 * vd + 2 * kd)) / d**0.5).astype(dtype),
        "w_ba": w_ba.astype(dtype),
        "conv": (jax.random.normal(keys[3], (CONV, 2 * kd + vd)) / CONV**0.5).astype(dtype),
        "A_log": jnp.log(jax.random.uniform(keys[4], (hv,), minval=1.0, maxval=16.0)),
        "dt_bias": jnp.ones((hv,)),
        "out_norm": (1 + 0.1 * jax.random.normal(keys[5], (1, HEAD_DIM))).astype(dtype),
        "w_out": (jax.random.normal(keys[6], (vd, d)) / vd**0.5).astype(dtype),
    }


if __name__ == "__main__":
    config = Config(d_model=256, k_heads=1, v_heads=2)
    params = init_params(config)
    x = jax.random.normal(jax.random.key(1), (1, 512, config.d_model), jnp.bfloat16)
    out = gated_deltanet_layer(x, params, config)  # each kernel tuned on a TPU, the default tiles elsewhere
    want = baseline_layer(x, params, config)
    diff = jnp.max(jnp.abs(out.astype(jnp.float32) - want.astype(jnp.float32)))
    print(f"out {out.shape}; max |autokernel - plain JAX| {float(diff):.4f}")
