"""Linear attention with the delta rule: Gated DeltaNet (Qwen3-Next) and Kimi Delta Attention (Kimi Linear),
chunked, for prefill.

Per head, a state S (d_k x d_v) is updated token by token (q and k L2-normalized, q scaled by d_k**-0.5):

    S_t = (I - beta_t k_t k_t^T) Diag(a_t) S_{t-1} + beta_t k_t v_t^T        o_t = S_t^T q_t

with beta_t in [0, 1] and a decay a_t = exp(g_t), g_t <= 0: a scalar per head and token in Gated DeltaNet
(`gated_delta_rule`), one per key channel in KDA (`kimi_delta_attention`). (Equivalently, as
flash-linear-attention's recurrent kernels compute it: S *= a_t; u = beta_t (v_t - S^T k_t); S += k_t u^T.)

**Chunks.** Over a chunk of C tokens starting from state S, with G_r the cumulative log decay up to token r
within the chunk and Gamma_r = exp(G_r) (per channel in KDA), the WY/UT form of the delta rule is

    A     = strictly_lower(Diag(beta) (K Gamma) (K / Gamma)^T)    (entries beta_r k_r.k_s exp(G_r - G_s))
    T     = (I + A)^-1
    W     = T Diag(beta) (Gamma K)        U~ = T Diag(beta) V
    U     = U~ - W S                      (the chunk's "new values")
    O     = (Gamma Q) S + lower(Q K^T decayed) U
    S'    = Gamma_C S + (K exp(G_C - G))^T U

C is 128: every matmul is a full 128 x 128 MXU tile with bf16 operands and f32 accumulation (as
flash-linear-attention's kernels round), and a chunk's per-token values fit lane-wide (C, 128) matrices, row r
holding token r's value in every lane, which scale the rows of (C, HEAD_DIM) tiles as they are. T comes from
doubling: the inverse of the s x s diagonal blocks of I + A gives that of the 2s x 2s ones, T_2s = T_s -
T_s A_off T_s, in log2(C) levels; unlike a Neumann product (I - A)(I + A^2)..., it never forms large powers
of A (keys pointing the same way make those blow up). The cumulative decays are MXU matmuls with triangular
0/1 matrices, the decays split in three bf16 parts: f32-accurate. Pairwise decays exp(G_r - G_s) are exact
for a scalar decay; for KDA's channel-wise decay they factor through a reference point in each 32-row block
of the chunk, exp(G_r - G_ref) exp(G_ref - G_s), which stays inside f32's range while per-token log decays
are >= KDA_GATE_MIN (-5, Kimi Linear's "safe gate" lower bound): they are clamped to it.

**Kernels.** One tile per (batch, group of heads), with a reduction loop over the sequence carrying every
head's state in VMEM, f32 (transposed, d_v x d_k, so KDA's channel decay scales its columns). Each step
streams a block of tokens of the group's columns of q, k, v (B, S, H * HEAD_DIM) (and of KDA's decays), and
every head's gates, (H, block) of a (B, H, S) array; it normalizes q and k, then runs the block's chunks one
after the other for every head of the group, in one unrolled stretch of code (independent MXU work for the
scheduler to interleave). The tunable picks the heads per tile and the tokens per step. Out-of-range gates
are clamped (log decays to <= 0, betas to [0, 1]), so the random inputs tuning runs on are valid too.

A reduction step can't store an output tile (outputs are stored once per tile, at its end), so each step
appends its outputs to a carried buffer, and the tile stores the whole sequence at the end. A carried value
is loaded and stored whole by every step that assigns it, so the sequence is split into PARTS phases, each
with a buffer of its own: a step moves a quarter of the tile's outputs, not all of them.

`reference_recurrent` is the recurrence above in float64 NumPy; `baseline_chunked` the chunked algorithm
in plain JAX (intra-chunk work batched over every chunk, a `lax.scan` over chunks for the state, outputs
batched after it, as flash-linear-attention structures it); `baseline_recurrent` the recurrence as a
`lax.scan` over tokens.

Run with `uv run python examples/linear_attention.py`; without a TPU it runs in the Pallas TPU interpreter.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

HEAD_DIM = 128  # d_k = d_v: Qwen3-Next's and Kimi Linear's
CHUNK = 128  # tokens per chunk: C x C matrices are full MXU tiles, and C = HEAD_DIM = lanes (see above)
PARTS = 4  # phases per tile: each step appends its outputs to a quarter of the sequence
SUB = 32  # KDA: rows per block of a chunk sharing a decay reference point
KDA_GATE_MIN = -5.0  # KDA: per-token log decays are clamped to [KDA_GATE_MIN, 0]
NORM_EPS = 1e-6  # q and k L2 normalization, as flash-linear-attention's


@autokernel.tunable
class DeltaTiles:
    """Heads per tile, and tokens per step (a whole number of chunks)."""

    heads: int = autokernel.choice(1, 2, 4, default=2)
    block: int = autokernel.choice(128, 256, 512, default=256)


# --------------------------------------------------------------------------- the chunk math (plain JAX)
# Shared by the kernels (2-D, one head at a time) and by the baselines (batched over leading dims).


def _bf16(x):
    return x.astype(jnp.bfloat16)


def _dot(a, b):
    return jnp.matmul(a, b, preferred_element_type=jnp.float32)


def _dot_nt(a, b):
    """a @ b^T over the last two dims (any leading dims are batch dims, the same in both): the MXU reads b
    transposed, no transpose is computed."""
    batch = tuple(range(a.ndim - 2))
    dims = (((a.ndim - 1,), (b.ndim - 1,)), (batch, batch))
    return jax.lax.dot_general(a, b, dims, preferred_element_type=jnp.float32)


def _iotas(n):
    return jax.lax.broadcasted_iota(jnp.int32, (n, n), 0), jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)


def _split3(x):
    """f32 `x` as three bf16 parts summing to it to ~24 bits: a matmul of each with a 0/1 matrix is exact."""
    hi = _bf16(x)
    rest = x - hi.astype(jnp.float32)
    mid = _bf16(rest)
    return hi, mid, _bf16(rest - mid.astype(jnp.float32))


def cumsum_lanes(x):
    """Inclusive cumulative sums along the last dim (C), f32-accurate, on the MXU: x @ upper-triangular 1s."""
    rows, cols = _iotas(x.shape[-1])
    upper = _bf16(rows <= cols)
    return sum(_dot(part, upper) for part in _split3(x))


def cumsum_rows(x):
    """Inclusive cumulative sums along the second-to-last dim (C), f32-accurate: lower-triangular 1s @ x."""
    rows, cols = _iotas(x.shape[-2])
    lower = _bf16(rows >= cols)
    return sum(_dot(lower, part) for part in _split3(x))


def unit_lower_inverse(a):
    """(I + a)^-1 for strictly lower-triangular (..., C, C) `a`, C a power of two, by doubling the size of the
    inverted diagonal blocks: T_2s = T_s - T_s A_off T_s, A_off the blocks of `a` below T_s's within each
    2s x 2s block."""
    n = a.shape[-1]
    rows, cols = _iotas(n)
    apart = rows ^ cols  # < s iff in the same s x s block (s a power of two): no vector division
    t = jnp.where(rows == cols, 1.0, 0.0) - jnp.where(apart < 2, a, 0.0)  # T_2
    s = 2
    while s < n:
        off = jnp.where((apart >= s) & (apart < 2 * s), a, 0.0)
        tb = _bf16(t)
        t = t - _dot(tb, _bf16(_dot(_bf16(off), tb)))
        s *= 2
    return t


def _row_to_lanes(row, n):
    """A (..., 1, n) row as (..., n, n) lane-wide columns: element [r, s] is row[r]."""
    return jnp.swapaxes(jnp.broadcast_to(row, (*row.shape[:-2], n, n)), -1, -2)


def _intra(q, k, v, beta_r, gamma, k_hat, decay_last, a, qk):
    """T = (I + A)^-1, then W and U~; with the chunk's other operands, as `chunk_update` and `chunk_output`
    take them (bf16 for the MXU)."""
    d = q.shape[-1]
    t = _bf16(unit_lower_inverse(a))
    wu = _dot(t, jnp.concatenate([_bf16(beta_r * gamma * k), _bf16(beta_r * v)], axis=-1))
    return _bf16(wu[..., :d]), _bf16(wu[..., d:]), _bf16(q * gamma), _bf16(qk), _bf16(k_hat), decay_last


def gdn_intra(q, k, v, g_row, beta_row):
    """Gated DeltaNet's work on a chunk that doesn't need the state. q, k, v (..., C, d) bf16 (q and k
    normalized, q scaled); g_row and beta_row (..., 1, C) f32, its tokens' log decays and betas. Returns
    (w, u~, q_gamma, qk, k_hat, decay_last)."""
    n = q.shape[-2]
    lead = g_row.shape[:-2]
    rows, cols = _iotas(n)
    g_cum = cumsum_lanes(jnp.broadcast_to(g_row, (*lead, 8, n)))[..., :1, :]  # (a matmul of 8 rows, not 1)
    g_s = jnp.broadcast_to(g_cum, (*lead, n, n))  # [r, s] = G_s
    g_r = jnp.swapaxes(g_s, -1, -2)  # [r, s] = G_r: row r's log decay, lane-wide
    decay = jnp.exp(jnp.where(rows >= cols, g_r - g_s, -jnp.inf))  # exp(G_r - G_s) on and below the diagonal
    beta_r = _row_to_lanes(beta_row, n)
    a = jnp.where(rows > cols, beta_r * _dot_nt(k, k) * decay, 0.0)
    g_last = jnp.min(g_r, axis=-2, keepdims=True)  # G_C in every lane (log decays are <= 0)
    k_hat = k * jnp.exp(g_last - g_r)
    return _intra(q, k, v, beta_r, jnp.exp(g_r), k_hat, jnp.exp(g_last), a, _dot_nt(q, k) * decay)


def kda_intra(q, k, v, g, beta_row):
    """KDA's work on a chunk that doesn't need the state: as `gdn_intra`, with g (..., C, d_k) f32 log
    decays per key channel, each in [KDA_GATE_MIN, 0]."""
    n = q.shape[-2]
    rows, cols = _iotas(n)
    gc = cumsum_rows(g)  # G (..., C, d_k)
    row = jax.lax.broadcasted_iota(jnp.int32, gc.shape, gc.ndim - 2)
    akk, aqk = [], []
    for lo in range(0, n, SUB):  # exp(G_r - G_s) = exp(G_r - G_ref) exp(G_ref - G_s), a reference per block
        ref = jnp.sum(jnp.where(row == lo + SUB // 2 - 1, gc, 0.0), axis=-2, keepdims=True)
        rel = jnp.exp(gc[..., lo : lo + SUB, :] - ref)  # within exp(+-80): log decays are >= -5
        kc = _bf16(k * jnp.exp(jnp.where(row < lo + SUB, ref - gc, 0.0)))  # (later keys are masked out below)
        akk.append(_dot_nt(_bf16(k[..., lo : lo + SUB, :] * rel), kc))
        aqk.append(_dot_nt(_bf16(q[..., lo : lo + SUB, :] * rel), kc))
    beta_r = _row_to_lanes(beta_row, n)
    a = jnp.where(rows > cols, beta_r * jnp.concatenate(akk, axis=-2), 0.0)
    qk = jnp.where(rows >= cols, jnp.concatenate(aqk, axis=-2), 0.0)
    g_last = jnp.min(gc, axis=-2, keepdims=True)  # G_C per channel
    k_hat = k * jnp.exp(g_last - gc)
    return _intra(q, k, v, beta_r, jnp.exp(gc), k_hat, jnp.exp(g_last), a, qk)


def chunk_update(w, u, k_hat, decay_last, state):
    """The chunk's new values U = U~ - W S (bf16) and the state after it; `state` (..., d_v, d_k) f32 is
    the transposed state S^T at the chunk's start."""
    u = u - _dot_nt(w, _bf16(state))
    return _bf16(u), decay_last * state + _dot(_bf16(jnp.swapaxes(u, -1, -2)), k_hat)


def chunk_output(q_gamma, qk, u, state):
    """The chunk's outputs (..., C, d_v) f32, from the state at its start and its new values."""
    return _dot_nt(q_gamma, _bf16(state)) + _dot(qk, u)


def l2_normalize(x, scale=1.0):
    """Each row of `x` (..., HEAD_DIM) over its L2 norm, times `scale`, f32."""
    x = x.astype(jnp.float32)
    return x * (jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + NORM_EPS) * scale)


def gated_rmsnorm(o, z, weight, eps=NORM_EPS):
    """Qwen3-Next's output gate on (rows, heads * HEAD_DIM): each head of `o` RMS-normalized, times `weight`
    (1, HEAD_DIM), times silu of the gate `z`; f32."""
    heads = []
    for lanes in range(0, o.shape[1], HEAD_DIM):
        oh = o[:, lanes : lanes + HEAD_DIM].astype(jnp.float32)
        zh = z[:, lanes : lanes + HEAD_DIM].astype(jnp.float32)
        normed = oh * jax.lax.rsqrt(jnp.mean(oh * oh, axis=1, keepdims=True) + eps)
        heads.append(normed * weight.astype(jnp.float32) * jax.nn.silu(zh))
    return jnp.concatenate(heads, axis=1)


# --------------------------------------------------------------------------- the kernels


def _head_row(x, head):
    """Row `head` (a traced int: it comes from a loop variable) of a (H, n) tile, as (1, n)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    return jnp.sum(jnp.where(rows == head, x, 0.0), axis=0, keepdims=True)


def delta_block(intra, q, k, v, g, beta, col, state):
    """A block of tokens for a group of heads, chunk after chunk: their outputs (block, heads * HEAD_DIM) f32,
    and their states (heads, d_v, d_k) after it.

    q, k, v (block, heads * HEAD_DIM) are the group's columns, from column `col` of the arrays; beta
    (H, block) holds every head's betas; g is either that too (a decay per head and token, Gated DeltaNet:
    `intra` is `gdn_intra`) or like k (a decay per key channel, KDA: `kda_intra`)."""
    first, per_channel = col // HEAD_DIM, g.shape == k.shape
    outs, states = [], []
    for h in range(state.shape[0]):
        lanes = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        qh = _bf16(l2_normalize(q[:, lanes], HEAD_DIM**-0.5))
        kh = _bf16(l2_normalize(k[:, lanes]))
        vh = _bf16(v[:, lanes])
        bh = jnp.clip(_head_row(beta, first + h), 0.0, 1.0)
        if per_channel:
            gh = jnp.clip(g[:, lanes].astype(jnp.float32), KDA_GATE_MIN, 0.0)
        else:
            gh = jnp.minimum(_head_row(g, first + h), 0.0)
        st, chunks = state[h], []
        for c in range(0, q.shape[0], CHUNK):
            rows = slice(c, c + CHUNK)
            gc = gh[rows] if per_channel else gh[:, rows]
            w, u, q_gamma, qk, k_hat, decay_last = intra(qh[rows], kh[rows], vh[rows], gc, bh[:, rows])
            u, st_next = chunk_update(w, u, k_hat, decay_last, st)
            chunks.append(chunk_output(q_gamma, qk, u, st))
            st = st_next
        outs.append(jnp.concatenate(chunks, axis=0))
        states.append(st)
    return jnp.concatenate(outs, axis=1), jnp.stack(states)


def append(buffer, rows):
    """`buffer` shifted up by the rows of `rows`, which go at its end: after a phase's steps, the buffer holds
    their outputs in order."""
    n = rows.shape[0]
    rows = rows.astype(buffer.dtype)
    return jnp.concatenate([buffer[n:], rows], axis=0) if n < buffer.shape[0] else rows


def gdn_block(q, k, v, g, beta, col, state, outputs):
    """A step of `gated_delta_rule`: `delta_block`, its outputs appended to `outputs`."""
    o, state = delta_block(gdn_intra, q, k, v, g, beta, col, state)
    return state, append(outputs, o)


def gdn_gated_block(q, k, v, g, beta, z, weight, col, state, outputs):
    """A step of `gated_delta_rule_gated`: `gdn_block`, with Qwen3-Next's gated RMSNorm on its outputs."""
    o, state = delta_block(gdn_intra, q, k, v, g, beta, col, state)
    return state, append(outputs, gated_rmsnorm(o, z, weight))


def kda_block(q, k, v, g, beta, col, state, outputs):
    """A step of `kimi_delta_attention`: `delta_block`, its outputs appended to `outputs`."""
    o, state = delta_block(kda_intra, q, k, v, g, beta, col, state)
    return state, append(outputs, o)


@autokernel.kernel
def gated_delta_rule(q, k, v, g, beta, tiles=DeltaTiles()):
    """Gated DeltaNet, chunked. q, k, v (B, S, H * HEAD_DIM), q and k not yet normalized; g and beta
    (B, H, S) f32: log decays (clamped to <= 0) and betas (clamped to [0, 1]). The outputs (B, S, H * HEAD_DIM)."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    part = S // PARTS
    T = min(tiles.block, part)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)  # per head, transposed: (d_v, d_k)
            o0 = jnp.zeros((part, W), dtype)  # the tile's outputs, a quarter of the sequence each
            o1 = jnp.zeros((part, W), dtype)
            o2 = jnp.zeros((part, W), dtype)
            o3 = jnp.zeros((part, W), dtype)
            for c0 in range(0, part, T):
                qt, kt, vt = q[b, c0 : c0 + T, j : j + W], k[b, c0 : c0 + T, j : j + W], v[b, c0 : c0 + T, j : j + W]
                state, o0 = gdn_block(qt, kt, vt, g[b, :, c0 : c0 + T], beta[b, :, c0 : c0 + T], j, state, o0)
            for c1 in range(part, 2 * part, T):
                qt, kt, vt = q[b, c1 : c1 + T, j : j + W], k[b, c1 : c1 + T, j : j + W], v[b, c1 : c1 + T, j : j + W]
                state, o1 = gdn_block(qt, kt, vt, g[b, :, c1 : c1 + T], beta[b, :, c1 : c1 + T], j, state, o1)
            for c2 in range(2 * part, 3 * part, T):
                qt, kt, vt = q[b, c2 : c2 + T, j : j + W], k[b, c2 : c2 + T, j : j + W], v[b, c2 : c2 + T, j : j + W]
                state, o2 = gdn_block(qt, kt, vt, g[b, :, c2 : c2 + T], beta[b, :, c2 : c2 + T], j, state, o2)
            for c3 in range(3 * part, S, T):
                qt, kt, vt = q[b, c3 : c3 + T, j : j + W], k[b, c3 : c3 + T, j : j + W], v[b, c3 : c3 + T, j : j + W]
                state, o3 = gdn_block(qt, kt, vt, g[b, :, c3 : c3 + T], beta[b, :, c3 : c3 + T], j, state, o3)
            out = out.at[b, :, j : j + W].set(jnp.concatenate([o0, o1, o2, o3], axis=0))
    return out


@autokernel.kernel
def gated_delta_rule_gated(q, k, v, g, beta, z, weight, tiles=DeltaTiles()):
    """`gated_delta_rule` followed by Qwen3-Next's gated RMSNorm: each head's outputs RMS-normalized, times
    `weight` (1, HEAD_DIM), times silu(z), z like v."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    part = S // PARTS
    T = min(tiles.block, part)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            o0 = jnp.zeros((part, W), dtype)
            o1 = jnp.zeros((part, W), dtype)
            o2 = jnp.zeros((part, W), dtype)
            o3 = jnp.zeros((part, W), dtype)
            wt = weight[:, :]
            for c0 in range(0, part, T):
                qt, kt, vt = q[b, c0 : c0 + T, j : j + W], k[b, c0 : c0 + T, j : j + W], v[b, c0 : c0 + T, j : j + W]
                gt, bt, zt = g[b, :, c0 : c0 + T], beta[b, :, c0 : c0 + T], z[b, c0 : c0 + T, j : j + W]
                state, o0 = gdn_gated_block(qt, kt, vt, gt, bt, zt, wt, j, state, o0)
            for c1 in range(part, 2 * part, T):
                qt, kt, vt = q[b, c1 : c1 + T, j : j + W], k[b, c1 : c1 + T, j : j + W], v[b, c1 : c1 + T, j : j + W]
                gt, bt, zt = g[b, :, c1 : c1 + T], beta[b, :, c1 : c1 + T], z[b, c1 : c1 + T, j : j + W]
                state, o1 = gdn_gated_block(qt, kt, vt, gt, bt, zt, wt, j, state, o1)
            for c2 in range(2 * part, 3 * part, T):
                qt, kt, vt = q[b, c2 : c2 + T, j : j + W], k[b, c2 : c2 + T, j : j + W], v[b, c2 : c2 + T, j : j + W]
                gt, bt, zt = g[b, :, c2 : c2 + T], beta[b, :, c2 : c2 + T], z[b, c2 : c2 + T, j : j + W]
                state, o2 = gdn_gated_block(qt, kt, vt, gt, bt, zt, wt, j, state, o2)
            for c3 in range(3 * part, S, T):
                qt, kt, vt = q[b, c3 : c3 + T, j : j + W], k[b, c3 : c3 + T, j : j + W], v[b, c3 : c3 + T, j : j + W]
                gt, bt, zt = g[b, :, c3 : c3 + T], beta[b, :, c3 : c3 + T], z[b, c3 : c3 + T, j : j + W]
                state, o3 = gdn_gated_block(qt, kt, vt, gt, bt, zt, wt, j, state, o3)
            out = out.at[b, :, j : j + W].set(jnp.concatenate([o0, o1, o2, o3], axis=0))
    return out


@autokernel.kernel
def kimi_delta_attention(q, k, v, g, beta, tiles=DeltaTiles()):
    """KDA, chunked: as `gated_delta_rule`, with g (B, S, H * HEAD_DIM) f32, log decays per key channel
    (clamped to [KDA_GATE_MIN, 0])."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    part = S // PARTS
    T = min(tiles.block, part)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            o0 = jnp.zeros((part, W), dtype)
            o1 = jnp.zeros((part, W), dtype)
            o2 = jnp.zeros((part, W), dtype)
            o3 = jnp.zeros((part, W), dtype)
            for c0 in range(0, part, T):
                qt, kt, vt = q[b, c0 : c0 + T, j : j + W], k[b, c0 : c0 + T, j : j + W], v[b, c0 : c0 + T, j : j + W]
                gt, bt = g[b, c0 : c0 + T, j : j + W], beta[b, :, c0 : c0 + T]
                state, o0 = kda_block(qt, kt, vt, gt, bt, j, state, o0)
            for c1 in range(part, 2 * part, T):
                qt, kt, vt = q[b, c1 : c1 + T, j : j + W], k[b, c1 : c1 + T, j : j + W], v[b, c1 : c1 + T, j : j + W]
                gt, bt = g[b, c1 : c1 + T, j : j + W], beta[b, :, c1 : c1 + T]
                state, o1 = kda_block(qt, kt, vt, gt, bt, j, state, o1)
            for c2 in range(2 * part, 3 * part, T):
                qt, kt, vt = q[b, c2 : c2 + T, j : j + W], k[b, c2 : c2 + T, j : j + W], v[b, c2 : c2 + T, j : j + W]
                gt, bt = g[b, c2 : c2 + T, j : j + W], beta[b, :, c2 : c2 + T]
                state, o2 = kda_block(qt, kt, vt, gt, bt, j, state, o2)
            for c3 in range(3 * part, S, T):
                qt, kt, vt = q[b, c3 : c3 + T, j : j + W], k[b, c3 : c3 + T, j : j + W], v[b, c3 : c3 + T, j : j + W]
                gt, bt = g[b, c3 : c3 + T, j : j + W], beta[b, :, c3 : c3 + T]
                state, o3 = kda_block(qt, kt, vt, gt, bt, j, state, o3)
            out = out.at[b, :, j : j + W].set(jnp.concatenate([o0, o1, o2, o3], axis=0))
    return out


# --------------------------------------------------------------------------- references and baselines


def _per_channel(g: Any, v: Any) -> bool:
    """Is `g` KDA's (B, S, H * HEAD_DIM) log decays per key channel, rather than Gated DeltaNet's (B, H, S)?"""
    return tuple(g.shape) == tuple(v.shape)


def reference_recurrent(q: Any, k: Any, v: Any, g: Any, beta: Any) -> np.ndarray:
    """The recurrence, token by token, in float64 NumPy (inputs clamped as the kernels clamp them): Gated
    DeltaNet for g (B, H, S), KDA for g (B, S, H * HEAD_DIM). The outputs (B, S, H * HEAD_DIM)."""
    q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
    B, S, D = v.shape
    H = D // HEAD_DIM
    if _per_channel(g, v):
        decay = np.exp(np.clip(g, KDA_GATE_MIN, 0.0)).reshape(B, S, H, HEAD_DIM)
    else:
        decay = np.broadcast_to(np.exp(np.minimum(g, 0.0)).transpose(0, 2, 1)[..., None], (B, S, H, HEAD_DIM))

    def normalized(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        x = x.reshape(B, S, H, HEAD_DIM)
        return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + NORM_EPS) * scale

    q, k, v = normalized(q, HEAD_DIM**-0.5), normalized(k), v.reshape(B, S, H, HEAD_DIM)
    beta = np.clip(beta, 0.0, 1.0).transpose(0, 2, 1)  # (B, S, H)
    state = np.zeros((B, H, HEAD_DIM, HEAD_DIM))  # (d_k, d_v) per head
    out = np.empty((B, S, H, HEAD_DIM))
    for t in range(S):
        state = decay[:, t, :, :, None] * state
        u = beta[:, t, :, None] * (v[:, t] - np.einsum("bhkv,bhk->bhv", state, k[:, t]))
        state = state + k[:, t, :, :, None] * u[:, :, None, :]
        out[:, t] = np.einsum("bhkv,bhk->bhv", state, q[:, t])
    return out.reshape(B, S, D)


@jax.jit
def baseline_chunked(q, k, v, g, beta):
    """The chunked algorithm in plain JAX, as flash-linear-attention structures it: the work of every chunk
    that doesn't need the state, batched; a `lax.scan` over chunks for the state; the outputs, batched. The
    kernels' math and rounding, and their layouts in and out (Gated DeltaNet for g (B, H, S), KDA for g like
    v)."""
    B, S, D = v.shape
    H, n = D // HEAD_DIM, S // CHUNK

    def chunks(x):  # (B, S, H * d) -> (n, B, H, C, d)
        return x.reshape(B, n, CHUNK, H, HEAD_DIM).transpose(1, 0, 3, 2, 4)

    def rows(x):  # (B, H, S) -> (n, B, H, 1, C)
        return x.reshape(B, H, n, 1, CHUNK).transpose(2, 0, 1, 3, 4)

    def normalized(x, scale=1.0):
        return _bf16(l2_normalize(x.reshape(B, S, H, HEAD_DIM), scale).reshape(B, S, D))

    qc, kc, vc = chunks(normalized(q, HEAD_DIM**-0.5)), chunks(normalized(k)), chunks(_bf16(v))
    beta_rows = rows(jnp.clip(beta, 0.0, 1.0))
    if _per_channel(g, v):
        intra = kda_intra(qc, kc, vc, chunks(jnp.clip(g.astype(jnp.float32), KDA_GATE_MIN, 0.0)), beta_rows)
    else:
        intra = gdn_intra(qc, kc, vc, rows(jnp.minimum(g, 0.0)), beta_rows)
    w, u, q_gamma, qk, k_hat, decay_last = intra

    def step(state, chunk):
        u, after = chunk_update(*chunk, state)
        return after, (u, _bf16(state))

    state = jnp.zeros((B, H, HEAD_DIM, HEAD_DIM), jnp.float32)
    _, (u, states) = jax.lax.scan(step, state, (w, u, k_hat, decay_last))
    out = chunk_output(q_gamma, qk, u, states)  # (n, B, H, C, d)
    return out.transpose(1, 0, 3, 2, 4).reshape(B, S, D).astype(v.dtype)


@jax.jit
def baseline_recurrent(q, k, v, g, beta):
    """The recurrence as a `lax.scan` over tokens in plain JAX, f32: one token per iteration, for scale."""
    B, S, D = v.shape
    H = D // HEAD_DIM
    if _per_channel(g, v):
        decay = jnp.exp(jnp.clip(g, KDA_GATE_MIN, 0.0)).reshape(B, S, H, HEAD_DIM)
    else:
        decay = jnp.exp(jnp.minimum(g, 0.0)).transpose(0, 2, 1)[..., None]  # (B, S, H, 1)

    def tokens(x):  # (B, S, ...) -> (S, B, ...)
        return jnp.moveaxis(x, 1, 0)

    q = l2_normalize(q.reshape(B, S, H, HEAD_DIM), HEAD_DIM**-0.5)
    k = l2_normalize(k.reshape(B, S, H, HEAD_DIM))
    v = v.reshape(B, S, H, HEAD_DIM).astype(jnp.float32)
    beta = jnp.clip(beta, 0.0, 1.0).transpose(0, 2, 1)
    highest = jax.lax.Precision.HIGHEST

    def step(state, token):  # state (B, H, d_k, d_v)
        q_t, k_t, v_t, a_t, b_t = token
        state = a_t[..., :, None] * state
        u = b_t[..., None] * (v_t - jnp.einsum("bhkv,bhk->bhv", state, k_t, precision=highest))
        state = state + k_t[..., :, None] * u[..., None, :]
        return state, jnp.einsum("bhkv,bhk->bhv", state, q_t, precision=highest)

    state = jnp.zeros((B, H, HEAD_DIM, HEAD_DIM), jnp.float32)
    _, out = jax.lax.scan(step, state, tuple(tokens(x) for x in (q, k, v, decay, beta)))
    return tokens(out).reshape(B, S, D).astype(jnp.bfloat16)


def make_inputs(batch: int, heads: int, seq: int, per_channel: bool = False, seed: int = 0) -> tuple[jax.Array, ...]:
    """Random kernel inputs: bf16 q, k, v (B, S, H * HEAD_DIM); log decays as Qwen3-Next's gate makes them,
    -exp(A_log) * softplus(a + dt_bias) per head (B, H, S), or as Kimi Linear's safe gate does, in [-5, 0]
    per key channel (like v); betas sigmoid(b), (B, H, S)."""
    keys = jax.random.split(jax.random.key(seed), 5)
    d = heads * HEAD_DIM
    q, k, v = (jax.random.normal(keys[i], (batch, seq, d), jnp.bfloat16) for i in range(3))
    if per_channel:
        g = KDA_GATE_MIN * jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq, d)) - 3.0)
    else:
        a = jnp.linspace(1.0, 16.0, heads)[None, :, None]  # Qwen3-Next draws A from [1, 16]
        g = -a * jax.nn.softplus(jax.random.normal(keys[3], (batch, heads, seq)) - 4.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, heads, seq)))
    return q, k, v, g.astype(jnp.float32), beta


if __name__ == "__main__":
    for kernel, per_channel in ((gated_delta_rule, False), (kimi_delta_attention, True)):
        inputs = make_inputs(1, 2, 1024, per_channel)
        out = kernel(*inputs)  # tuned on a TPU, the default tiles elsewhere
        print(kernel.tuning(*inputs))
        want = reference_recurrent(*inputs)
        diff = np.abs(np.asarray(out, np.float64) - want).max()
        print(
            f"{kernel.nest.name}: max |kernel - float64 recurrence| {diff:.4f} (outputs up to {np.abs(want).max():.3f})"
        )
        print("max |kernel - loop nest|:", kernel.verify(*inputs))
    print(gated_delta_rule.source(*make_inputs(1, 2, 1024)))
