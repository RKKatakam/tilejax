"""Multi-head latent attention (MLA: DeepSeek-V2/V3, Kimi K2), prefill: autokernel kernels composed in plain JAX.

A pre-norm attention block over (batch, seq, d_model) activations x, with H heads:

    h = rmsnorm(x)
    c_q  = rmsnorm(h @ W_dq)                     (q_rank: the query latent)
    c_kv = rmsnorm(h @ W_dkv)                    (kv_rank, 512: the key/value latent, all a decode cache keeps)
    k_rope = rope(h @ W_kr)                      (64: ONE rotary key, shared by every head)
    [q_nope_h | q_rope_h] = c_q @ W_uq_h;  q_rope_h = rope(q_rope_h)        (128 + 64 per head)
    k_nope_h, v_h = c_kv @ W_uk_h, c_kv @ W_uv_h                             (128 each per head)
    s_h = (q_nope_h . k_nope_h + q_rope_h . k_rope) / sqrt(192);  o_h = causal_softmax(s_h) @ v_h
    out = x + concat_h(o_h) @ W_o

(DeepSeek-V3: d_model 7168, 128 heads, q_rank 1536; Kimi K2: 64 heads, the same latent and head dims.) RoPE
is the rotate-half form, `t * cos + rotate_half(t) * sin` on 64 dims; DeepSeek's checkpoints rotate
interleaved pairs, which is the same thing after permuting W_kr's and W_uq's rope columns. (YaRN's frequency
scaling would only change the cos/sin tables.)

This is the "expanded" form, the fast one for prefill: k_nope and v are materialized once per token and every
query block reuses them. `mla_absorbed` is the same attention computed over the latent c_kv (q_nope absorbed
into W_uk, W_uv applied after attention), what decoding wants: a cache of 512 + 64 numbers per token.

Four kernel calls (three kernels), each tuned for its shapes on the first call:

1. `mla_down`: the block's RMSNorm fused with the three down projections, one pass over x. The projections
   are one matmul against W_down = [W_dq | W_dkv | W_kr | W_kr], whose output row (q_rank + kv_rank + 128)
   is a whole tile, so the epilogue can normalize c_q and c_kv by their own RMS (the q_a and kv_a norms) and
   rotate k_rope; nothing but the three results goes to HBM. The rope key gets a 128-lane slot holding it
   twice, [k_rope | k_rope] (W_kr's columns twice: the MXU computes 128 columns either way).
2. Up projections: `matmul` (examples/matmul.py), twice. q = c_q @ W_uq with columns
   [q_nope of every head (H * 128) | q_rope of every head (H * 64)], pre-scaled by 1/sqrt(192) (folded into
   W_uq; RoPE is linear, so it commutes); k_nope and v = c_kv @ [W_uk | W_uv], (H * 128) each. They are
   materialized: with S in the thousands, recomputing them from c_kv for every query block would multiply
   their FLOPs by the number of query blocks.
3. `mla_attention`: causal flash attention for every head of a (batch, query block) tile at once (as
   examples/transformer.py; the online softmax head by head, so one head's scores are live at a time), then
   the output projection (a phase over heads) and the residual. Lane layout of the rope part: q's rope
   columns hold two heads per 128-lane slot, [q_rope_2p | q_rope_2p+1] (the natural head-major order). Once
   per query tile, before the key loop, they are rotated and split by parity into [q_rope_2p | 0] and
   [0 | q_rope_2p+1]; against the shared key slot [k_rope | k_rope], head h's score is then two 128-deep MXU
   matmuls, `q_nope_h @ k_nope_h.T + q_slot(h) @ k_slot.T`: the MXU passes of one 192-deep matmul (the MXU
   is 128 deep), no lane work in a key step, no padding in HBM, and one k_rope slot per key for all heads
   (splash attention needs it copied into every head's keys).

The fusion boundaries: attention for a query block needs the keys and values of every earlier token, so the
projections can't join the attention kernel; the up projections are plain matmuls of their own because
their outputs are H * 192 or H * 256 columns wide (a kernel storing them whole per row tile would need
those rows of f32 in VMEM, and resident weights of up to tens of MiB at 64-128 heads).

`baseline` is the same block in plain JAX for XLA; `splash_baseline` uses JAX's splash attention (which
takes a 192-wide q/k head dim and a 128-wide v head dim; `pad=True` pads q and k to 256).

Run with `uv run python examples/mla.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from matmul import matmul
from transformer import widen

import autokernel

EPS = 1e-6
LANES = 128  # TPU vector lanes
NOPE = 128  # per-head dims: the position-free part of q and k, and v
ROPE = 64  # the rotary part of q and k (half a lane slot)
QK = NOPE + ROPE
THETA = 10000.0
MASK_VALUE = -1e30  # finite, so exp(old max - new max) never computes -inf + inf


@dataclass(frozen=True)
class MlaConfig:
    d_model: int
    heads: int  # even: q's rope part holds two heads per 128-lane slot
    q_rank: int  # q_lora_rank
    kv_rank: int = 512  # kv_lora_rank


# --------------------------------------------------------------------------- helpers called from tile code


def rms(t, gamma):
    """RMSNorm of f32 rows t (rows, n) with weights gamma (1, n)."""
    return t * jax.lax.rsqrt(jnp.mean(t * t, axis=1, keepdims=True) + EPS) * gamma.astype(jnp.float32)


def rope(t, cos, sin):
    """RoPE, rotate-half, on every 64-lane group of t (rows, n), n a multiple of 128; cos and sin (rows, 128)
    hold the angles of both groups of a lane slot. Lane rolls and an iota select, no 64-lane slicing."""
    t = t.astype(jnp.float32)
    lane = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    half = ROPE // 2
    rotated = jnp.where(lane % ROPE < half, -jnp.roll(t, -half, axis=1), jnp.roll(t, half, axis=1))
    reps = (1, t.shape[1] // LANES)
    return t * jnp.tile(cos, reps) + rotated * jnp.tile(sin, reps)


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _dot_nt(a, b):
    """a @ b.T, the MXU reading b transposed."""
    return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def rope_by_parity(t, cos, sin):
    """q's rope part t (rows, H * 64), two heads per 128-lane slot, rotated and split by head parity: the even
    heads' lanes with the odd heads' zeroed, and the odd heads' with the even heads' zeroed. Against the key
    slot [k_rope | k_rope], a slot of either is then one head's rope score. (Masked once per query tile, in
    f32, so a key step does no lane work.)"""
    r = rope(t, cos, sin)
    even = jax.lax.broadcasted_iota(jnp.int32, r.shape, 1) % LANES < ROPE
    return jnp.where(even, r, 0.0).astype(t.dtype), jnp.where(even, 0.0, r).astype(t.dtype)


def mla_attention_step(q_nope, q_even, q_odd, k_nope, v, k_rope, i, j, row_max, row_sum, acc):
    """One online-softmax step of causal MLA for every head: the query block at row i against the key block
    at row j. q_nope, k_nope and v are (rows, H * 128); q_even and q_odd (rows, H * 64) from `rope_by_parity`;
    k_rope (rows, 128) is [k_rope | k_rope]. Updates each head's running row max and sum (lane-wide,
    (H, rows, 128)) and output accumulator (H, rows, 128). Head by head, so only one head's scores are live
    at a time."""
    H, bq, _ = acc.shape
    bk = k_nope.shape[0]
    rows = i + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    causal = j + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1) <= rows
    maxes, sums, accs = [], [], []
    for h in range(H):
        cols = slice(h * NOPE, (h + 1) * NOPE)
        q_slot = (q_odd if h % 2 else q_even)[:, h // 2 * LANES : (h // 2 + 1) * LANES]
        s = _dot_nt(q_nope[:, cols], k_nope[:, cols]) + _dot_nt(q_slot, k_rope)
        s = jnp.where(causal, s, MASK_VALUE)
        max_new = jnp.maximum(row_max[h], jnp.max(s, axis=1, keepdims=True))
        p = jnp.exp(s - widen(max_new, bk))
        alpha = jnp.exp(row_max[h] - max_new)
        maxes.append(max_new)
        sums.append(alpha * row_sum[h] + jnp.sum(p, axis=1, keepdims=True))
        accs.append(widen(alpha, NOPE) * acc[h] + _dot(p.astype(v.dtype), v[:, cols]))
    return jnp.stack(maxes), jnp.stack(sums), jnp.stack(accs)


# --------------------------------------------------------------------------- the kernels


@autokernel.tunable
class DownTiles:
    """Tokens per output tile, model dims per step (a block of all of them keeps W_down in VMEM)."""

    rows: int = autokernel.choice(128, 256, 512, 1024, 2048, default=256)
    block_d: int = autokernel.choice(256, 512, 1024, 2048, 4096, 8192, default=512)


@autokernel.kernel
def mla_down(x, attn_norm, w_down, q_norm, kv_norm, cos, sin, tiles=DownTiles()):
    """rmsnorm(x) @ [W_dq | W_dkv | W_kr | W_kr], then the q and kv latents' RMSNorms and RoPE on the rope
    key: c_q (B, S, q_rank), c_kv (B, S, kv_rank) and [k_rope | k_rope] (B, S, 128)."""
    B, S, D = x.shape
    Rq, Rkv = q_norm.shape[1], kv_norm.shape[1]
    N = w_down.shape[1]
    tm, tk = min(tiles.rows, S), min(tiles.block_d, D)
    dtype = x.dtype
    c_q = jnp.zeros((B, S, Rq), dtype)
    c_kv = jnp.zeros((B, S, Rkv), dtype)
    k_rope = jnp.zeros((B, S, LANES), dtype)
    for b in range(B):
        for i in range(0, S, tm):
            acc = jnp.zeros((tm, N), jnp.float32)  # (x * gamma) @ W_down
            sum_sq = jnp.zeros((tm, LANES), jnp.float32)  # each row's sum of squares, lane-wide
            for k in range(0, D, tk):
                xk = x[b, i : i + tm, k : k + tk].astype(jnp.float32)
                sum_sq += jnp.sum(xk * xk, axis=1, keepdims=True)
                acc += (xk * attn_norm[0:1, k : k + tk].astype(jnp.float32)).astype(dtype) @ w_down[k : k + tk, :]
            h = acc * widen(jax.lax.rsqrt(sum_sq / D + EPS), N)  # rmsnorm(x) @ W_down: the row scale commutes
            c_q = c_q.at[b, i : i + tm, :].set(rms(h[:, :Rq], q_norm[0:1, :]).astype(dtype))
            c_kv = c_kv.at[b, i : i + tm, :].set(rms(h[:, Rq : Rq + Rkv], kv_norm[0:1, :]).astype(dtype))
            k_rope = k_rope.at[b, i : i + tm, :].set(
                rope(h[:, Rq + Rkv :], cos[i : i + tm, :], sin[i : i + tm, :]).astype(dtype)
            )
    return c_q, c_kv, k_rope


@autokernel.tunable
class AttentionTiles:
    """Queries per output tile, keys per attention step."""

    block_q: int = autokernel.choice(128, 256, 512, 1024, default=256)
    block_kv: int = autokernel.choice(128, 256, 512, 1024, default=256)


@autokernel.kernel
def mla_attention(x, q, kv, k_rope, cos, sin, w_o, tiles=AttentionTiles()):
    """x + concat_h(causal_softmax(q_h . k_h) @ v_h) @ W_o, from q (B, S, H * 192) = [q_nope | q_rope] (q_rope
    not yet rotated, pre-scaled), kv (B, S, H * 256) = [k_nope | v], [k_rope | k_rope] (B, S, 128)."""
    B, S, D = x.shape
    H = w_o.shape[0]
    HN = H * NOPE  # the columns of q_nope, k_nope and v
    bq, bk = min(tiles.block_q, S), min(tiles.block_kv, S)
    dtype = x.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for i in range(0, S, bq):
            q_even, q_odd = rope_by_parity(q[b, i : i + bq, HN : HN + H * ROPE], cos[i : i + bq, :], sin[i : i + bq, :])
            row_max = jnp.full((H, bq, LANES), MASK_VALUE, jnp.float32)
            row_sum = jnp.zeros((H, bq, LANES), jnp.float32)
            acc = jnp.zeros((H, bq, NOPE), jnp.float32)
            for j in range(0, S, bk):  # phase 1: causal attention, every head at once
                if j > i + bq - 1:  # every key is in the future of every query
                    continue
                q_nope, k_nope, v = q[b, i : i + bq, 0:HN], kv[b, j : j + bk, 0:HN], kv[b, j : j + bk, HN : 2 * HN]
                row_max, row_sum, acc = mla_attention_step(
                    q_nope, q_even, q_odd, k_nope, v, k_rope[b, j : j + bk, :], i, j, row_max, row_sum, acc
                )
            heads = (acc / widen(row_sum, NOPE)).astype(dtype)
            attn = jnp.zeros((bq, D), jnp.float32)
            for hh in range(H):  # phase 2: the output projection, a head's rows of W_o per step
                attn += heads[hh] @ w_o[hh, :, :]
            out = out.at[b, i : i + bq, :].set((x[b, i : i + bq, :].astype(jnp.float32) + attn).astype(dtype))
    return out


def rope_tables(seq: int, dtype: Any = jnp.float32) -> tuple[jax.Array, jax.Array]:
    """cos and sin of the rotary angles, (seq, 128): lane l of position p holds p * theta**(-2f / 64) for
    f = l % 32, the rotate-half order of a 64-dim group, twice (two groups per lane slot)."""
    inv_freq = THETA ** (-jnp.arange(0, ROPE, 2, dtype=dtype) / ROPE)
    angles = jnp.arange(seq, dtype=dtype)[:, None] * jnp.tile(inv_freq, LANES // (ROPE // 2))[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def mla_block(x: jax.Array, packed: dict[str, jax.Array]) -> jax.Array:
    """The MLA block (prefill) with autokernel kernels: x (B, S, d_model) -> x + attention, on `pack`ed
    weights."""
    B, S, _ = x.shape
    cos, sin = rope_tables(S)
    c_q, c_kv, k_rope = mla_down(
        x, packed["attn_norm"], packed["w_down"], packed["q_norm"], packed["kv_norm"], cos, sin
    )
    q = matmul(c_q.reshape(B * S, -1), packed["w_uq"]).reshape(B, S, -1)
    kv = matmul(c_kv.reshape(B * S, -1), packed["w_ukv"]).reshape(B, S, -1)
    return mla_attention(x, q, kv, k_rope, cos, sin, packed["w_o"])


# --------------------------------------------------------------------------- weights


def init_params(config: MlaConfig, dtype: Any = jnp.bfloat16, seed: int = 0) -> dict[str, jax.Array]:
    """Random weights in the model's own layout: w_uq (q_rank, H, 192) = [nope | rope] per head, w_uk and
    w_uv (kv_rank, H, 128), w_o (H, 128, d_model); norm weights (n,)."""
    d, h, rq, rkv = config.d_model, config.heads, config.q_rank, config.kv_rank
    keys = iter(jax.random.split(jax.random.key(seed), 10))

    def normal(shape: tuple[int, ...], fan_in: int) -> jax.Array:
        return (jax.random.normal(next(keys), shape) * fan_in**-0.5).astype(dtype)

    def norm_weight(n: int) -> jax.Array:
        return (1 + 0.1 * jax.random.normal(next(keys), (n,))).astype(dtype)

    return {
        "attn_norm": norm_weight(d),
        "w_dq": normal((d, rq), d),
        "q_norm": norm_weight(rq),
        "w_uq": normal((rq, h, QK), rq),
        "w_dkv": normal((d, rkv), d),
        "w_kr": normal((d, ROPE), d),
        "kv_norm": norm_weight(rkv),
        "w_uk": normal((rkv, h, NOPE), rkv),
        "w_uv": normal((rkv, h, NOPE), rkv),
        "w_o": normal((h, NOPE, d), h * NOPE),
    }


def pack(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """The kernels' layout of the weights (done once, when they are loaded): W_down = [W_dq | W_dkv | W_kr |
    W_kr]; W_uq = [q_nope columns of every head | q_rope columns of every head], scaled by 1/sqrt(192);
    W_ukv = [W_uk | W_uv]; norm weights as (1, n) rows."""
    rq, h, _ = params["w_uq"].shape
    w_uq = params["w_uq"].astype(jnp.float32) * QK**-0.5
    return {
        "attn_norm": params["attn_norm"][None, :],
        "w_down": jnp.concatenate([params["w_dq"], params["w_dkv"], params["w_kr"], params["w_kr"]], axis=1),
        "q_norm": params["q_norm"][None, :],
        "kv_norm": params["kv_norm"][None, :],
        "w_uq": jnp.concatenate(
            [w_uq[:, :, :NOPE].reshape(rq, h * NOPE), w_uq[:, :, NOPE:].reshape(rq, h * ROPE)], axis=1
        ).astype(params["w_uq"].dtype),
        "w_ukv": jnp.concatenate([params["w_uk"].reshape(-1, h * NOPE), params["w_uv"].reshape(-1, h * NOPE)], 1),
        "w_o": params["w_o"],
    }


# --------------------------------------------------------------------------- plain JAX: references and baselines


def _f32(t: jax.Array) -> jax.Array:
    """t at least f32 (f64 stays f64)."""
    return t.astype(jnp.promote_types(t.dtype, jnp.float32))


def _rmsnorm(t: jax.Array, gamma: jax.Array) -> jax.Array:
    t = _f32(t)
    return t * jax.lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + EPS) * _f32(gamma)


def _rope(t: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half RoPE on the last dim (64) of t (B, S, ..., 64); cos and sin (S, 64)."""
    t = _f32(t)
    first, second = t[..., : ROPE // 2], t[..., ROPE // 2 :]
    shape = (cos.shape[0],) + (1,) * (t.ndim - 3) + (ROPE,)
    return t * cos.reshape(shape) + jnp.concatenate([-second, first], axis=-1) * sin.reshape(shape)


def _einsum(spec: str, a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.einsum(spec, a, b, preferred_element_type=jnp.promote_types(a.dtype, jnp.float32))


def latents(x: jax.Array, params: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The per-token projections, in plain JAX: q_nope (B, S, H, 128), q_rope (B, S, H, 64), rotated, both
    unscaled; c_kv (B, S, kv_rank); k_rope (B, S, 64), rotated. Activations are kept in x's dtype between
    layers, as a model does."""
    dtype = x.dtype
    cos, sin = (t[:, :ROPE] for t in rope_tables(x.shape[1], _f32(x).dtype))
    h = _rmsnorm(x, params["attn_norm"]).astype(dtype)
    c_q = _rmsnorm(_einsum("bsd,dr->bsr", h, params["w_dq"]), params["q_norm"]).astype(dtype)
    q = _einsum("bsr,rhe->bshe", c_q, params["w_uq"])
    q_rope = _rope(q[..., NOPE:].astype(dtype), cos, sin).astype(dtype)
    c_kv = _rmsnorm(_einsum("bsd,dr->bsr", h, params["w_dkv"]), params["kv_norm"]).astype(dtype)
    k_rope = _rope(_einsum("bsd,de->bse", h, params["w_kr"]).astype(dtype), cos, sin).astype(dtype)
    return q[..., :NOPE].astype(dtype), q_rope, c_kv, k_rope


def _causal_softmax(s: jax.Array) -> jax.Array:
    """Softmax over the last axis of scores (B, H, S, S) under a causal mask."""
    causal = jnp.tril(jnp.ones(s.shape[-2:], bool))
    return jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)


def baseline(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """The MLA block, expanded form, in plain JAX (what XLA compiles): k_nope and v per head, then causal
    attention over the whole (S, S) score matrix, the output projection and the residual."""
    dtype = x.dtype
    q_nope, q_rope, c_kv, k_rope = latents(x, params)
    k_nope = _einsum("bsr,rhe->bshe", c_kv, params["w_uk"]).astype(dtype)
    v = _einsum("bsr,rhe->bshe", c_kv, params["w_uv"]).astype(dtype)
    s = _einsum("bqhe,bkhe->bhqk", q_nope, k_nope) + _einsum("bqhe,bke->bhqk", q_rope, k_rope)
    p = _causal_softmax(s * QK**-0.5)
    o = _einsum("bhqk,bkhe->bqhe", p.astype(dtype), v).astype(dtype)
    return (_f32(x) + _einsum("bqhe,hed->bqd", o, params["w_o"])).astype(dtype)


def mla_absorbed(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """The same block, absorbed form (what decoding computes): attention runs over the latent c_kv. q_nope_h
    absorbed into W_uk_h gives a query in latent space, `(q_nope_h @ W_uk_h.T) . c_kv`; the attention output
    stays latent, `p @ c_kv`, until W_uv_h maps it to the head's values. Equal to `baseline` in exact
    arithmetic (tested in float64); no k_nope or v is ever formed."""
    dtype = x.dtype
    q_nope, q_rope, c_kv, k_rope = latents(x, params)
    q_latent = _einsum("bqhe,rhe->bqhr", q_nope, params["w_uk"]).astype(dtype)
    s = _einsum("bqhr,bkr->bhqk", q_latent, c_kv) + _einsum("bqhe,bke->bhqk", q_rope, k_rope)
    p = _causal_softmax(s * QK**-0.5)
    o_latent = _einsum("bhqk,bkr->bqhr", p.astype(dtype), c_kv).astype(dtype)
    o = _einsum("bqhr,rhe->bqhe", o_latent, params["w_uv"]).astype(dtype)
    return (_f32(x) + _einsum("bqhe,hed->bqd", o, params["w_o"])).astype(dtype)


def splash_baseline(
    heads: int, seq: int, block_q: int | None = None, block_kv: int | None = None, pad: bool = False, interpret=False
):
    """The block in plain JAX with JAX's splash attention (default block sizes unless given) on (B, H, S, d):
    q and k as [nope | rope] (192 wide, or zero-padded to 256 with `pad`; the shared k_rope copied to every
    head), v 128 wide. q is pre-scaled. Jitted; takes (x, params)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

    mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq, seq)) for _ in range(heads)])
    sizes = None
    if block_q is not None:
        sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
    attend = jax.vmap(splash.make_splash_mha_single_device(mask, block_sizes=sizes, interpret=interpret))

    def block(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
        dtype = x.dtype
        b, s, _ = x.shape
        q_nope, q_rope, c_kv, k_rope = latents(x, params)
        k_nope = _einsum("bsr,rhe->bshe", c_kv, params["w_uk"]).astype(dtype)
        v = _einsum("bsr,rhe->bshe", c_kv, params["w_uv"]).astype(dtype)
        widths = [(0, 0)] * 3 + [(0, 2 * LANES - QK if pad else 0)]
        q = jnp.pad((jnp.concatenate([q_nope, q_rope], -1) * QK**-0.5).astype(dtype), widths)
        k = jnp.pad(jnp.concatenate([k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, s, heads, ROPE))], -1), widths)
        o = attend(*(t.transpose(0, 2, 1, 3) for t in (q, k, v)))  # (B, H, S, 128)
        return (_f32(x) + _einsum("bhqe,hed->bqd", o, params["w_o"])).astype(dtype)

    return jax.jit(block)


if __name__ == "__main__":
    config = MlaConfig(d_model=256, heads=4, q_rank=256, kv_rank=256)
    params = init_params(config)
    packed = pack(params)
    x = jax.random.normal(jax.random.key(1), (1, 256, config.d_model), jnp.bfloat16)
    out = jax.jit(mla_block)(x, packed)  # tuned on a TPU, the default tiles elsewhere
    cos, sin = rope_tables(256)
    q = jnp.zeros((1, 256, config.heads * QK), x.dtype)
    kv = jnp.zeros((1, 256, 2 * config.heads * NOPE), x.dtype)
    k_rope = jnp.zeros((1, 256, LANES), x.dtype)
    lowered = mla_attention.lower(x, q, kv, k_rope, cos, sin, packed["w_o"], AttentionTiles(block_q=128, block_kv=128))
    print(lowered.source)
    print(f"{lowered.steps} steps in phases", [p.range.var for p in lowered.plan.phases])
    diff = jnp.max(jnp.abs(out.astype(jnp.float32) - baseline(x, params).astype(jnp.float32)))
    print("max |autokernel - plain JAX|:", float(diff))
