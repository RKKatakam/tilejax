"""Grouped-query causal attention fused with the output projection and the residual (Qwen3, Llama 3, ...).

    o_h = softmax(q_h @ k_g.T + mask) @ v_g           g = h // G: H query heads share Hkv = H / G kv heads
    out = x + concat_h(o_h * gate_h) @ wo             (gate_h only for gated attention, Qwen3-Next)

Activations are (B, S, heads * head_dim): a head is a block of head_dim columns, nothing is transposed. q is
pre-scaled by 1/sqrt(head_dim) (the QKV kernel does that, qwen3.py), as JAX's splash attention expects too.

One tile per (batch, query block), every head at once, in phases:

1. Flash attention over the key blocks, skipping blocks entirely in the future of the query block (and,
   with a sliding window of W keys, entirely before the window of its first query): masks evaluated when
   the kernel is built, so the kernel never visits, fetches or computes them. A step reads one key and
   one value block of every kv head, (block_kv, Hkv * head_dim), and for each kv head stacks its group's
   G query heads along rows: one (G * block_q, head_dim) @ (head_dim, block_kv) matmul per kv head instead
   of G small ones, so each key block is pushed through the MXU once per group, and the online-softmax
   statistics are one (Hkv, G * block_q, 128) array (lane-wide row max and sum, as in transformer.py).
   Stacking is free: the heads are 128-lane-aligned column blocks and block_q a multiple of 16 rows, so
   concatenating them only renames vector registers.
2. The output projection, `block_o` rows of wo per step (a step multiplies the matching block of the
   heads' columns). A block of all of wo's rows keeps wo in VMEM (fetched once per kernel); smaller ones
   stream it. Then the residual: the query block's rows of x.
3. Optionally (`layer` kernels) the rest of the layer: RMSNorm and the SwiGLU feed-forward layer over
   blocks of hidden units (swiglu.py), and the second residual: a whole Qwen3 layer after its QKV
   projection in one kernel. With big feed-forward weights the query blocks are too few rows to reuse them
   and the split form (this kernel, then `swiglu_ffn` over bigger row tiles) is faster: qwen3.py measures both.

Variants are closure constants of `make_attention(head_dim, window)` (masks are Python on loop variables
and constants, evaluated when a kernel is built): `window=None` is causal attention, `window=W` sliding-window
causal attention (a query sees itself and the W - 1 keys before it). Gated attention (Qwen3-Next) multiplies
each head's output by its gate, sigmoid(x_normed @ w_gate) from the QKV kernel, before the projection.

Run with `uv run python examples/gqa_attention.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from swiglu import rms_rows, swiglu
from transformer import widen

import autokernel

LANES = 128  # TPU vector lanes
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@autokernel.tunable
class AttentionTiles:
    """Queries per output tile, keys per attention step, rows of wo per output-projection step."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=512)
    block_o: int = autokernel.choice(128, 256, 512, 1024, 2048, 4096, 8192, default=512)


@autokernel.tunable
class LayerTiles:
    """AttentionTiles, and hidden units per feed-forward step."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=512)
    block_o: int = autokernel.choice(128, 256, 512, 1024, 2048, 4096, 8192, default=512)
    block_f: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, 16384, default=512)


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _dot_nt(a, b):
    """a @ b.T, the MXU reading b transposed."""
    return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def score_bias(i, j, bq, bk, groups, window):
    """What to add to the scores of the queries from row i against the keys from row j: 0 where the key is
    visible (not in the future; with a window, fewer than `window` keys back), MASK_VALUE elsewhere. The
    (bq, bk) pattern repeated for each of a group's query heads, which are stacked along rows."""
    rows = i + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    cols = j + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
    visible = cols <= rows
    if window is not None:
        visible = visible & (rows - cols < window)
    bias = jnp.where(visible, jnp.float32(0), jnp.float32(MASK_VALUE))
    return jnp.concatenate([bias] * groups, axis=0) if groups > 1 else bias


def gqa_step(qt, kt, vt, bias, row_max, row_sum, acc):
    """One online-softmax step for every kv head: the query block `qt` (bq, H * Dh) against the key and value
    block `kt`, `vt` (bk, Hkv * Dh). Kv head g's query heads g*G .. g*G + G-1 are stacked along rows, so
    `row_max`, `row_sum` (lane-wide) and `acc` are (Hkv, G * bq, 128 or Dh). (A helper is plain JAX: its
    matmuls ask for f32 accumulation, as `@` in kernel code means.)"""
    kv_heads, rows, dh = acc.shape
    bq, bk = qt.shape[0], kt.shape[0]
    groups = rows // bq
    maxes, sums, accs = [], [], []
    for g in range(kv_heads):
        heads = [qt[:, h * dh : (h + 1) * dh] for h in range(g * groups, (g + 1) * groups)]
        qg = jnp.concatenate(heads, axis=0) if groups > 1 else heads[0]
        s = _dot_nt(qg, kt[:, g * dh : (g + 1) * dh]) + bias
        max_new = jnp.maximum(row_max[g], jnp.max(s, axis=1, keepdims=True))
        p = jnp.exp(s - widen(max_new, bk))
        alpha = jnp.exp(row_max[g] - max_new)
        sums.append(alpha * row_sum[g] + jnp.sum(p, axis=1, keepdims=True))
        accs.append(widen(alpha, dh) * acc[g] + _dot(p.astype(vt.dtype), vt[:, g * dh : (g + 1) * dh]))
        maxes.append(max_new)
    return jnp.stack(maxes), jnp.stack(sums), jnp.stack(accs)


def head_blocks(acc, row_sum, bq, block, gate=None):
    """The attention output, normalized, the heads side by side in their order (head g*G + r is row block r
    of kv head g's stack), times the gate if given, as (H * Dh // block, bq, block) blocks of columns: one
    per output-projection step. f32."""
    kv_heads, rows, dh = acc.shape
    heads = []
    for g in range(kv_heads):
        out = acc[g] * widen(1.0 / row_sum[g], dh)
        heads += [out[r * bq : (r + 1) * bq] for r in range(rows // bq)]
    cols = jnp.concatenate(heads, axis=1) if len(heads) > 1 else heads[0]
    if gate is not None:
        cols = cols * gate.astype(jnp.float32)
    return jnp.stack([cols[:, c : c + block] for c in range(0, cols.shape[1], block)])


class AttentionKernels(NamedTuple):
    """The attention kernels for one head size and mask (`make_attention`)."""

    attention: autokernel.Kernel  # (x, q, k, v, wo) -> x + attention @ wo
    gated_attention: autokernel.Kernel  # (x, q, k, v, gate, wo)
    layer: autokernel.Kernel  # (x, q, k, v, wo, ffn_norm, w_gate, w_up, w_down): the whole layer
    gated_layer: autokernel.Kernel  # (x, q, k, v, gate, wo, ffn_norm, w_gate, w_up, w_down)


@functools.cache
def make_attention(head_dim: int = 128, window: int | None = None) -> AttentionKernels:
    """The GQA attention kernels for heads of `head_dim` and causal attention (`window=None`) or
    sliding-window causal attention over `window` keys (a query and the window - 1 keys before it)."""

    @autokernel.kernel
    def gqa_attention(x, q, k, v, wo, tiles=AttentionTiles()):
        """x + GQA attention of q, k, v (B, S, heads * head_dim) projected by wo (H * head_dim, D)."""
        B, S, D = x.shape
        HD, KD = q.shape[2], k.shape[2]
        groups = HD // KD  # query heads per kv head
        bq, bk = min(tiles.block_q, S), min(tiles.block_kv, S)
        bo = max(min(tiles.block_o, HD), head_dim)
        dtype = x.dtype
        out = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, bq):
                row_max = jnp.full((KD // head_dim, groups * bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((KD // head_dim, groups * bq, LANES), jnp.float32)
                acc = jnp.zeros((KD // head_dim, groups * bq, head_dim), jnp.float32)
                for j in range(0, S, bk):  # phase 1: attention, every head at once
                    if j > i + bq - 1 or (window is not None and j + bk + window <= i + 1):
                        continue  # every key is in the future of (or too far back for) every query
                    bias = score_bias(i, j, bq, bk, groups, window)
                    qt, kt, vt = q[b, i : i + bq, :], k[b, j : j + bk, :], v[b, j : j + bk, :]
                    row_max, row_sum, acc = gqa_step(qt, kt, vt, bias, row_max, row_sum, acc)
                heads = head_blocks(acc, row_sum, bq, bo).astype(dtype)
                attn = jnp.zeros((bq, D), jnp.float32)
                for c in range(0, HD, bo):  # phase 2: the output projection, bo rows of wo per step
                    attn += heads[c // bo] @ wo[c : c + bo, :]
                out = out.at[b, i : i + bq, :].set((x[b, i : i + bq, :].astype(jnp.float32) + attn).astype(dtype))
        return out

    @autokernel.kernel
    def gated_gqa_attention(x, q, k, v, gate, wo, tiles=AttentionTiles()):
        """gqa_attention with each head's output times its gate (B, S, heads * head_dim) first."""
        B, S, D = x.shape
        HD, KD = q.shape[2], k.shape[2]
        groups = HD // KD
        bq, bk = min(tiles.block_q, S), min(tiles.block_kv, S)
        bo = max(min(tiles.block_o, HD), head_dim)
        dtype = x.dtype
        out = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, bq):
                row_max = jnp.full((KD // head_dim, groups * bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((KD // head_dim, groups * bq, LANES), jnp.float32)
                acc = jnp.zeros((KD // head_dim, groups * bq, head_dim), jnp.float32)
                for j in range(0, S, bk):
                    if j > i + bq - 1 or (window is not None and j + bk + window <= i + 1):
                        continue
                    bias = score_bias(i, j, bq, bk, groups, window)
                    qt, kt, vt = q[b, i : i + bq, :], k[b, j : j + bk, :], v[b, j : j + bk, :]
                    row_max, row_sum, acc = gqa_step(qt, kt, vt, bias, row_max, row_sum, acc)
                heads = head_blocks(acc, row_sum, bq, bo, gate[b, i : i + bq, :]).astype(dtype)
                attn = jnp.zeros((bq, D), jnp.float32)
                for c in range(0, HD, bo):
                    attn += heads[c // bo] @ wo[c : c + bo, :]
                out = out.at[b, i : i + bq, :].set((x[b, i : i + bq, :].astype(jnp.float32) + attn).astype(dtype))
        return out

    @autokernel.kernel
    def gqa_layer(x, q, k, v, wo, ffn_norm, w_gate, w_up, w_down, tiles=LayerTiles()):
        """A whole layer after its QKV projection: gqa_attention, then x + SwiGLU(rmsnorm(x))."""
        B, S, D = x.shape
        HD, KD = q.shape[2], k.shape[2]
        F = w_up.shape[1]
        groups = HD // KD
        bq, bk, bf = min(tiles.block_q, S), min(tiles.block_kv, S), min(tiles.block_f, F)
        bo = max(min(tiles.block_o, HD), head_dim)
        dtype = x.dtype
        out = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, bq):
                row_max = jnp.full((KD // head_dim, groups * bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((KD // head_dim, groups * bq, LANES), jnp.float32)
                acc = jnp.zeros((KD // head_dim, groups * bq, head_dim), jnp.float32)
                for j in range(0, S, bk):  # phase 1: attention
                    if j > i + bq - 1 or (window is not None and j + bk + window <= i + 1):
                        continue
                    bias = score_bias(i, j, bq, bk, groups, window)
                    qt, kt, vt = q[b, i : i + bq, :], k[b, j : j + bk, :], v[b, j : j + bk, :]
                    row_max, row_sum, acc = gqa_step(qt, kt, vt, bias, row_max, row_sum, acc)
                heads = head_blocks(acc, row_sum, bq, bo).astype(dtype)
                attn = jnp.zeros((bq, D), jnp.float32)
                for c in range(0, HD, bo):  # phase 2: the output projection
                    attn += heads[c // bo] @ wo[c : c + bo, :]
                normed = rms_rows(x[b, i : i + bq, :].astype(jnp.float32) + attn, ffn_norm[0:1, :]).astype(dtype)
                y = jnp.zeros((bq, D), jnp.float32)
                for f in range(0, F, bf):  # phase 3: the SwiGLU feed-forward layer, bf hidden units per step
                    hidden = swiglu(normed @ w_gate[:, f : f + bf], normed @ w_up[:, f : f + bf])
                    y += hidden.astype(dtype) @ w_down[f : f + bf, :]
                resid = x[b, i : i + bq, :].astype(jnp.float32) + attn
                out = out.at[b, i : i + bq, :].set((resid + y).astype(dtype))
        return out

    @autokernel.kernel
    def gated_gqa_layer(x, q, k, v, gate, wo, ffn_norm, w_gate, w_up, w_down, tiles=LayerTiles()):
        """gqa_layer with gated attention."""
        B, S, D = x.shape
        HD, KD = q.shape[2], k.shape[2]
        F = w_up.shape[1]
        groups = HD // KD
        bq, bk, bf = min(tiles.block_q, S), min(tiles.block_kv, S), min(tiles.block_f, F)
        bo = max(min(tiles.block_o, HD), head_dim)
        dtype = x.dtype
        out = jnp.zeros((B, S, D), dtype)
        for b in range(B):
            for i in range(0, S, bq):
                row_max = jnp.full((KD // head_dim, groups * bq, LANES), MASK_VALUE, jnp.float32)
                row_sum = jnp.zeros((KD // head_dim, groups * bq, LANES), jnp.float32)
                acc = jnp.zeros((KD // head_dim, groups * bq, head_dim), jnp.float32)
                for j in range(0, S, bk):
                    if j > i + bq - 1 or (window is not None and j + bk + window <= i + 1):
                        continue
                    bias = score_bias(i, j, bq, bk, groups, window)
                    qt, kt, vt = q[b, i : i + bq, :], k[b, j : j + bk, :], v[b, j : j + bk, :]
                    row_max, row_sum, acc = gqa_step(qt, kt, vt, bias, row_max, row_sum, acc)
                heads = head_blocks(acc, row_sum, bq, bo, gate[b, i : i + bq, :]).astype(dtype)
                attn = jnp.zeros((bq, D), jnp.float32)
                for c in range(0, HD, bo):
                    attn += heads[c // bo] @ wo[c : c + bo, :]
                normed = rms_rows(x[b, i : i + bq, :].astype(jnp.float32) + attn, ffn_norm[0:1, :]).astype(dtype)
                y = jnp.zeros((bq, D), jnp.float32)
                for f in range(0, F, bf):
                    hidden = swiglu(normed @ w_gate[:, f : f + bf], normed @ w_up[:, f : f + bf])
                    y += hidden.astype(dtype) @ w_down[f : f + bf, :]
                resid = x[b, i : i + bq, :].astype(jnp.float32) + attn
                out = out.at[b, i : i + bq, :].set((resid + y).astype(dtype))
        return out

    return AttentionKernels(gqa_attention, gated_gqa_attention, gqa_layer, gated_gqa_layer)


# --------------------------------------------------------------------------- the baseline: JAX's splash attention


def splash_gqa(
    heads: int,
    kv_heads: int,
    seq: int,
    window: int | None = None,
    block_q: int | None = None,
    block_kv: int | None = None,
    form: str = "mqa",
    interpret: bool = False,
) -> Any:
    """JAX's splash attention for GQA on (B, S, heads * head_dim) activations -> (B, S, heads * head_dim):
    `form="mqa"` runs splash's MQA kernel for each kv head (vmapped) with its group's query heads; `"mha"`
    repeats k and v for every query head. Causal, or sliding-window over `window` keys (LocalMask); splash's
    default block sizes unless given."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

    groups = heads // kv_heads
    if window is None:
        mask = mask_lib.CausalMask((seq, seq))
    else:
        mask = mask_lib.LocalMask((seq, seq), window_size=(window - 1, 0), offset=0)
    sizes = None
    if block_q is not None:
        sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    if form == "mqa":
        multi = mask_lib.MultiHeadMask([mask] * groups)
        kernel = splash.make_splash_mqa_single_device(multi, block_sizes=sizes, interpret=interpret)
        attend = jax.vmap(jax.vmap(kernel))  # over batch, then kv heads

        def run(q, k, v):
            b, s, hd = q.shape
            dh = hd // heads
            qh = q.reshape(b, s, kv_heads, groups, dh).transpose(0, 2, 3, 1, 4)  # (B, Hkv, G, S, Dh)
            kh, vh = (t.reshape(b, s, kv_heads, dh).transpose(0, 2, 1, 3) for t in (k, v))  # (B, Hkv, S, Dh)
            return attend(qh, kh, vh).transpose(0, 3, 1, 2, 4).reshape(b, s, hd)

        return run
    multi = mask_lib.MultiHeadMask([mask] * heads)
    attend = jax.vmap(splash.make_splash_mha_single_device(multi, block_sizes=sizes, interpret=interpret))

    def run_mha(q, k, v):
        b, s, hd = q.shape
        dh = hd // heads
        qh = q.reshape(b, s, heads, dh).transpose(0, 2, 1, 3)
        kh, vh = (jnp.repeat(t.reshape(b, s, kv_heads, dh).transpose(0, 2, 1, 3), groups, axis=1) for t in (k, v))
        return attend(qh, kh, vh).transpose(0, 2, 1, 3).reshape(b, s, hd)

    return run_mha


def make_inputs(
    batch: int, seq: int, d_model: int, heads: int, kv_heads: int, head_dim: int = 128, dtype=jnp.bfloat16, seed=0
) -> tuple[jax.Array, ...]:
    """Random x, q (pre-scaled), k, v, gate (in (0, 1)) and wo for the attention kernels."""
    keys = jax.random.split(jax.random.key(seed), 6)
    hd, kd = heads * head_dim, kv_heads * head_dim
    x = jax.random.normal(keys[0], (batch, seq, d_model), dtype)
    q = (jax.random.normal(keys[1], (batch, seq, hd)) * head_dim**-0.5).astype(dtype)
    k = jax.random.normal(keys[2], (batch, seq, kd), dtype)
    v = jax.random.normal(keys[3], (batch, seq, kd), dtype)
    gate = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq, hd))).astype(dtype)
    wo = (jax.random.normal(keys[5], (hd, d_model)) / hd**0.5).astype(dtype)
    return x, q, k, v, gate, wo


if __name__ == "__main__":
    x, q, k, v, gate, wo = make_inputs(1, 512, 256, heads=4, kv_heads=2)
    kernels = make_attention(128, window=256)
    tiles = AttentionTiles(block_q=128, block_kv=128, block_o=256)
    lowered = kernels.attention.lower(x, q, k, v, wo, tiles)
    print(lowered.source)
    print(
        f"{lowered.steps} steps over {lowered.tiles} query blocks; phases", [p.range.var for p in lowered.plan.phases]
    )
    print("max |kernel - loop nest|:", kernels.attention.verify(x, q, k, v, wo, tiles))
    print("gated: max |kernel - loop nest|:", kernels.gated_attention.verify(x, q, k, v, gate, wo, tiles))
