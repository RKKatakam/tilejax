"""A whole decoder-only transformer: autokernel kernels composed in plain JAX.

    tokens -> embedding -> L x [ x += Wo(attention(qkv(rmsnorm(x))));  x += FFN(rmsnorm(x)) ] -> rmsnorm -> logits

Pre-norm layers with causal multi-head attention and a GELU feed-forward layer, all on (batch, seq, d_model)
activations (a head is a block of head_dim columns, so nothing is ever transposed). Each layer is two
kernels, each tuned for its shapes on the first call:

1. `rmsnorm_matmul` (examples/rmsnorm_matmul.py): the attention RMSNorm fused with the QKV projection,
   writing q, k and v side by side, (B, S, 3D). q comes out pre-scaled by 1/sqrt(head_dim) (folded into its
   weights).
2. `transformer_layer`: the rest of the layer, one tile per (batch, query block), in three phases: causal
   attention over the key blocks for every head at once (skipping future blocks); the output projection, a
   head's rows of `wo` per step; then, after the residual and the feed-forward RMSNorm, the feed-forward
   layer over blocks of hidden units, streaming its weights; and the second residual. Nothing between the
   QKV projection and the layer's output goes to HBM.

   With big feed-forward weights, the attention's query blocks are too few rows to reuse them well, and a
   feed-forward kernel of its own (`ffn_layer`, after `attention_layer`) is faster: each layer runs the
   form its kernels' tuning measured faster (`fuse_ffn`).

The QKV projection can't join the layer before it: attention for a query block needs the keys and values
of every earlier token, so they must all exist first. The final norm and the logits are `rmsnorm_matmul`
again. `baseline` is the same model in plain JAX, with JAX's splash attention.

Run with `uv run python examples/transformer.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from rmsnorm_matmul import rmsnorm_matmul

import autokernel

EPS = 1e-6  # as rmsnorm_matmul's
LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@dataclass(frozen=True)
class Config:
    layers: int
    d_model: int
    heads: int
    ffn: int
    vocab: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads


@autokernel.tunable
class LayerTiles:
    """Queries per output tile, keys per attention step, hidden units per feed-forward step."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, 16384, default=1024)


@autokernel.tunable
class FfnTiles:
    """Rows per output tile, hidden units per step."""

    rows: int = autokernel.choice(256, 512, 1024, 2048, 4096, default=1024)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, 16384, default=512)


def widen(stat, n):
    """A lane-wide (..., 128) row statistic as (..., n): repeated across lanes, or its first n lanes."""
    return jnp.tile(stat, (1,) * (stat.ndim - 1) + (n // LANES,)) if n >= LANES else stat[..., :n]


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _dot_nt(a, b):
    """a @ b.T, the MXU reading b transposed."""
    return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def attention_step(qt, kt, vt, i, j, row_max, row_sum, acc):
    """One online-softmax step of causal attention, every head at once: the query block at row i of `qt`
    against the key block at row j of `kt` and `vt` ((rows, heads * head_dim) each, q pre-scaled), updating
    the per-head running row max and sum (lane-wide) and the output accumulator (heads, rows, head_dim).
    (A helper is plain JAX: its matmuls ask for f32 accumulation, as `@` in kernel code means.)"""
    H, bq, Dh = acc.shape
    bk = kt.shape[0]
    rows = i + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    cols = j + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
    s = jnp.stack(
        [
            jnp.where(cols <= rows, _dot_nt(qt[:, h * Dh : (h + 1) * Dh], kt[:, h * Dh : (h + 1) * Dh]), MASK_VALUE)
            for h in range(H)
        ]
    )
    max_new = jnp.maximum(row_max, jnp.max(s, axis=2, keepdims=True))
    p = jnp.exp(s - widen(max_new, bk))
    alpha = jnp.exp(row_max - max_new)
    row_sum = alpha * row_sum + jnp.sum(p, axis=2, keepdims=True)
    pv = jnp.stack([_dot(p[h].astype(qt.dtype), vt[:, h * Dh : (h + 1) * Dh]) for h in range(H)])
    return max_new, row_sum, widen(alpha, Dh) * acc + pv


@autokernel.kernel
def transformer_layer(x, qkv, wo, gamma, w_up, w_down, tiles=LayerTiles()):
    """A layer after its QKV projection, fused: attention, output projection, residual, RMSNorm, feed-forward
    layer, residual."""
    B, S, D = x.shape
    H, Dh = wo.shape[0], wo.shape[1]
    F = w_up.shape[1]
    bq, bk, bf = min(tiles.block_q, S), min(tiles.block_kv, S), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for i in range(0, S, bq):
            row_max = jnp.full((H, bq, LANES), MASK_VALUE, jnp.float32)
            row_sum = jnp.zeros((H, bq, LANES), jnp.float32)
            acc = jnp.zeros((H, bq, Dh), jnp.float32)
            for j in range(0, S, bk):  # phase 1: causal attention, every head at once
                if j > i + bq - 1:  # every key is in the future of every query
                    continue
                qt, kt, vt = qkv[b, i : i + bq, 0:D], qkv[b, j : j + bk, D : 2 * D], qkv[b, j : j + bk, 2 * D : 3 * D]
                row_max, row_sum, acc = attention_step(qt, kt, vt, i, j, row_max, row_sum, acc)
            heads = (acc / widen(row_sum, Dh)).astype(dtype)
            attn = jnp.zeros((bq, D), jnp.float32)
            for hh in range(H):  # phase 2: the output projection, a head's rows of wo per step
                attn += heads[hh] @ wo[hh, :, :]
            resid = x[b, i : i + bq, :].astype(jnp.float32) + attn
            normed = resid * jax.lax.rsqrt(jnp.mean(resid * resid, axis=1, keepdims=True) + EPS) * gamma[0:1, :]
            normed = normed.astype(dtype)
            y = jnp.zeros((bq, D), jnp.float32)
            for f in range(0, F, bf):  # phase 3: the feed-forward layer, bf hidden units per step
                y += jax.nn.gelu(normed @ w_up[:, f : f + bf]).astype(dtype) @ w_down[f : f + bf, :]
            out = out.at[b, i : i + bq, :].set((resid + y).astype(dtype))
    return out


@autokernel.kernel
def attention_layer(x, qkv, wo, tiles=LayerTiles()):
    """The attention half of a layer after its QKV projection: attention, output projection, residual."""
    B, S, D = x.shape
    H, Dh = wo.shape[0], wo.shape[1]
    bq, bk = min(tiles.block_q, S), min(tiles.block_kv, S)
    dtype = x.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for i in range(0, S, bq):
            row_max = jnp.full((H, bq, LANES), MASK_VALUE, jnp.float32)
            row_sum = jnp.zeros((H, bq, LANES), jnp.float32)
            acc = jnp.zeros((H, bq, Dh), jnp.float32)
            for j in range(0, S, bk):
                if j > i + bq - 1:
                    continue
                qt, kt, vt = qkv[b, i : i + bq, 0:D], qkv[b, j : j + bk, D : 2 * D], qkv[b, j : j + bk, 2 * D : 3 * D]
                row_max, row_sum, acc = attention_step(qt, kt, vt, i, j, row_max, row_sum, acc)
            heads = (acc / widen(row_sum, Dh)).astype(dtype)
            attn = jnp.zeros((bq, D), jnp.float32)
            for hh in range(H):
                attn += heads[hh] @ wo[hh, :, :]
            out = out.at[b, i : i + bq, :].set((x[b, i : i + bq, :].astype(jnp.float32) + attn).astype(dtype))
    return out


@autokernel.kernel
def ffn_layer(x, gamma, w_up, w_down, tiles=FfnTiles()):
    """The feed-forward half of a layer, on (tokens, d_model) rows: RMSNorm, feed-forward layer, residual."""
    T, D = x.shape
    F = w_up.shape[1]
    tm, bf = min(tiles.rows, T), min(tiles.block_f, F)
    dtype = x.dtype
    out = jnp.zeros((T, D), dtype)
    for i in range(0, T, tm):
        resid = x[i : i + tm, :].astype(jnp.float32)
        normed = resid * jax.lax.rsqrt(jnp.mean(resid * resid, axis=1, keepdims=True) + EPS) * gamma[0:1, :]
        normed = normed.astype(dtype)
        y = jnp.zeros((tm, D), jnp.float32)
        for f in range(0, F, bf):
            y += jax.nn.gelu(normed @ w_up[:, f : f + bf]).astype(dtype) @ w_down[f : f + bf, :]
        out = out.at[i : i + tm, :].set((resid + y).astype(dtype))
    return out


def fused_layer(x, qkv, layer):
    return transformer_layer(x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])


def split_layer(x, qkv, layer):
    b, s, d = x.shape
    resid = attention_layer(x, qkv, layer["wo"]).reshape(b * s, d)
    return ffn_layer(resid, layer["ffn_norm"], layer["w_up"], layer["w_down"]).reshape(b, s, d)


def fuse_ffn(x, qkv, layer) -> bool:
    """Should the feed-forward layer run in the attention kernel (`fused_layer`) rather than a kernel of its
    own (`split_layer`)? Fusing saves a round trip of the activations through HBM, but its row tiles are the
    attention's query blocks, and with big weights a kernel of its own reuses them over more rows: tuning
    measures both (off a TPU nothing is measured: fused)."""
    b, s, d = x.shape
    fused = transformer_layer.tuning(x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])
    if not fused.trials:
        return True
    rows = jax.ShapeDtypeStruct((b * s, d), x.dtype)
    attention = attention_layer.tuning(x, qkv, layer["wo"])
    ffn = ffn_layer.tuning(rows, layer["ffn_norm"], layer["w_up"], layer["w_down"])
    # (The split form's extra launch and HBM round trip are small: on a v5e, whole models ran faster split
    # whenever the two kernels' device times said so, by 1.7-2.1%.)
    return fused.best.seconds <= attention.best.seconds + ffn.best.seconds


def transformer(tokens, params):
    """The model with autokernel kernels: logits (B, S, vocab) for tokens (B, S)."""
    x = params["embed"][tokens]
    B, S, D = x.shape
    for layer in params["layers"]:
        qkv = rmsnorm_matmul(x.reshape(B * S, D), layer["attn_norm"], layer["w_qkv"]).reshape(B, S, 3 * D)
        x = fused_layer(x, qkv, layer) if fuse_ffn(x, qkv, layer) else split_layer(x, qkv, layer)
    return rmsnorm_matmul(x.reshape(B * S, D), params["final_norm"], params["w_vocab"]).reshape(B, S, -1)


def init_params(config: Config, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, Any]:
    """Random weights; q's are pre-scaled by 1/sqrt(head_dim), so attention needs no scaling."""
    d, f, h, dh = config.d_model, config.ffn, config.heads, config.head_dim
    keys = iter(jax.random.split(jax.random.key(seed), 4 + 6 * config.layers))

    def normal(shape, scale):
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm_weight(n):
        return (1 + 0.1 * jax.random.normal(next(keys), (1, n))).astype(dtype)

    layers = []
    for _ in range(config.layers):
        w_qkv = jax.random.normal(next(keys), (d, 3 * d)) / d**0.5
        w_qkv = w_qkv.at[:, :d].multiply(dh**-0.5).astype(dtype)
        layers.append(
            {
                "attn_norm": norm_weight(d),
                "w_qkv": w_qkv,
                "wo": normal((h, dh, d), d**-0.5),
                "ffn_norm": norm_weight(d),
                "w_up": normal((d, f), d**-0.5),
                "w_down": normal((f, d), f**-0.5),
            }
        )
    return {
        "embed": normal((config.vocab, d), 1.0),
        "layers": layers,
        "final_norm": norm_weight(d),
        "w_vocab": normal((d, config.vocab), d**-0.5),
    }


# --------------------------------------------------------------------------- the baseline: plain JAX + splash


def _rmsnorm(x, gamma):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def baseline(config: Config, seq: int, block_q: int | None = None, block_kv: int | None = None):
    """The same model in plain JAX, compiled by XLA, with JAX's splash attention (at its default block sizes
    unless given) on (B, H, S, Dh)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

    h, dh = config.heads, config.head_dim
    mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq, seq)) for _ in range(h)])
    sizes = None
    if block_q is not None:
        sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    attend = jax.vmap(splash.make_splash_mha_single_device(mask, block_sizes=sizes))

    def model(tokens, params):
        x = params["embed"][tokens]
        b, s, _ = x.shape
        for layer in params["layers"]:
            qkv = _dot(_rmsnorm(x, layer["attn_norm"]).astype(x.dtype), layer["w_qkv"]).astype(x.dtype)
            q, k, v = (t.reshape(b, s, h, dh).transpose(0, 2, 1, 3) for t in jnp.split(qkv, 3, axis=-1))
            a = attend(q, k, v)  # (B, H, S, Dh)
            resid = x.astype(jnp.float32) + jnp.einsum(
                "bhsd,hde->bse", a, layer["wo"], preferred_element_type=jnp.float32
            )
            normed = _rmsnorm(resid, layer["ffn_norm"]).astype(x.dtype)
            hidden = jax.nn.gelu(_dot(normed, layer["w_up"])).astype(x.dtype)
            x = (resid + _dot(hidden, layer["w_down"])).astype(x.dtype)
        return _dot(_rmsnorm(x, params["final_norm"]).astype(x.dtype), params["w_vocab"]).astype(x.dtype)

    return jax.jit(model)


if __name__ == "__main__":
    config = Config(layers=2, d_model=256, heads=2, ffn=512, vocab=512)
    params = init_params(config)
    tokens = jax.random.randint(jax.random.key(1), (1, 256), 0, config.vocab)
    logits = jax.jit(transformer)(tokens, params)  # tuned on a TPU, the default tiles elsewhere
    x = params["embed"][tokens]
    qkv = jnp.zeros((1, 256, 3 * config.d_model), x.dtype)
    layer = params["layers"][0]
    lowered = transformer_layer.lower(x, qkv, layer["wo"], layer["ffn_norm"], layer["w_up"], layer["w_down"])
    print(lowered.source)
    print(
        f"logits {logits.shape}; a layer's kernel runs {lowered.steps} steps in phases",
        [p.range.var for p in lowered.plan.phases],
    )
    if jax.default_backend() == "tpu":
        want = baseline(config, 256)(tokens, params)
        print("max |autokernel - splash + XLA|:", float(jnp.max(jnp.abs(logits.astype(float) - want.astype(float)))))
