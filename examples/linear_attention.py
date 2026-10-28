"""Linear attention with the delta rule: Gated DeltaNet (Qwen3-Next) and Kimi Delta Attention (Kimi Linear),
chunked, for prefill.

Per head, a state S (d_k x d_v) is updated token by token (q and k L2-normalized, q scaled by d_k**-0.5):

    S_t = (I - beta_t k_t k_t^T) Diag(a_t) S_{t-1} + beta_t k_t v_t^T        o_t = S_t^T q_t

with beta_t in [0, 1] and a decay a_t = exp(g_t), g_t <= 0: a scalar per head and token in Gated DeltaNet
(`gated_delta_rule`), one per key channel in KDA (`kimi_delta_attention`). (Equivalently, as
flash-linear-attention's recurrent kernels compute it: S *= a_t; u = beta_t (v_t - S^T k_t); S += k_t u^T.)

**Chunks.** Over a chunk of C tokens starting from state S, with G_r the cumulative log decay up to token r
within the chunk, Gamma = Diag(exp(G_r)) (exp(G) elementwise per channel in KDA), D = Diag(beta) and
Kt = strictly_lower(K K^T decayed) (entries k_r.k_s exp(G_r - G_s)), the WY/UT form of the delta rule is

    T  = (I + Kt D)^-1                 W = T (Gamma K)            U = T V          (no state needed)
    U' = U - W S                       (the chunk's "new values", before the betas)
    O  = (Gamma Q) S + (lower(Q K^T decayed) D) U'
    S' = Gamma_C S + (K exp(G_C - G))^T D U'

(The usual form, T0 = (I + D Kt)^-1 and U0 = T0 D (V - Gamma K S), is the same: T0 D = D T, so U0 = D U'. The
betas move to the columns of Kt, of the intra-chunk attention and of the key update, where they are rows
of a (1, C) vector: a token's values never have to be laid out down the rows for them.) T comes from
doubling: the inverse of the s x s diagonal blocks of I + A gives that of the 2s x 2s ones, T_2s = T_s -
T_s A_off T_s, in log2(C) levels, and T_s A_off T_s is T_s (A T_s) masked to the new blocks, so a level is
two matmuls and one select; unlike a Neumann product it never forms large powers of A (keys pointing the
same way make those blow up). Pairwise decays exp(G_r - G_s) are exact for a scalar decay; for KDA's
channel-wise decay they factor through a reference point in each 32-row block of the chunk, exp(G_r - G_ref)
exp(G_ref - G_s), which stays inside f32's range while per-token log decays are >= KDA_GATE_MIN (-5,
Kimi Linear's "safe gate" lower bound): they are clamped to it. Cumulative sums are MXU matmuls with
triangular 0/1 matrices, the values split in three bf16 parts: f32-accurate.

C is 128: every matmul is a full 128 x 128 MXU tile with bf16 operands and f32 accumulation (as
flash-linear-attention's kernels round). Everything but four matmuls per chunk (W S and Gamma Q S, the
attention times U', the keys' update; two of them in the chain from one state to the next) is independent
of the state.

**Kernels.** `gated_delta_rule`, `gated_delta_rule_gated` (with Qwen3-Next's gated RMSNorm fused) and
`kimi_delta_attention` are one kernel each: a tile per (batch, group of heads), a reduction loop over blocks
of tokens carrying every head's state in VMEM (f32, d_k x d_v), each step storing its block's outputs.
A step first computes, for every chunk of the block and every head of the group, the state-independent
operands above; then runs the chunks' state updates in order, the group's heads side by side. The code is
written stage by stage across those chunk-heads (all their normalizations, then all their K K^T, then each
doubling level for all of them, ...): consecutive matmuls are independent, so the MXUs can pipeline them
instead of waiting for each result, and only the short state chain is sequential. (Written chunk after
chunk, the same work took ~23 ns per head and token whatever the tiles: it ran one dependent matmul after
another, 14 deep per chunk.)
Nothing but q, k, v, the gates and the outputs crosses HBM. Out-of-range gates are clamped (log decays to
<= 0, betas to [0, 1]), so the random inputs tuning runs on are valid too.

**The split form** (`gated_delta_rule_split`, `kimi_delta_attention_split`) is flash-linear-attention's
structure: a fully parallel kernel over (batch, block of chunks, head) tiles (`gdn_chunks`, `kda_chunks`)
writes each chunk's operands to HBM (bf16: [W; Gamma Q], U, the decayed attention and the decayed keys,
transposed: 640 x 128 per chunk and head, and the chunk's decay), and a thin sequential kernel
(`delta_scan`) runs the state chain over them, a tile per (batch, group of heads). It moves 3.5x the
fused kernel's bytes. (Under `jax.jit` the scan tunes on random operands, whose recurrence would grow
without bound: it clamps its states to STATE_MAX every 512 tokens, which real ones never reach.)

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
SUB = 32  # KDA: rows per block of a chunk sharing a decay reference point
KDA_GATE_MIN = -5.0  # KDA: per-token log decays are clamped to [KDA_GATE_MIN, 0]
NORM_EPS = 1e-6  # q and k L2 normalization, as flash-linear-attention's
PACKED = 5 * CHUNK  # the split form's rows per chunk and head: [W; Gamma Q], U, attention, keys^T
STATE_MAX = 1e8  # the split form's scan clamps its states to this, far beyond what real inputs reach
STATE_CLAMP_CHUNKS = 4  # ... every this many chunks (512 tokens), whatever its block


@autokernel.tunable
class DeltaTiles:
    """Heads per tile, and tokens per step (a whole number of chunks)."""

    heads: int = autokernel.choice(1, 2, 4, 8, 16, default=4)
    block: int = autokernel.choice(128, 256, 512, default=256)


@autokernel.tunable
class ChunkTiles:
    """The split form's parallel kernels: chunks per tile (of one head)."""

    chunks: int = autokernel.choice(1, 2, 4, 8, 16, default=8)


SCAN_TILES = DeltaTiles(heads=8, block=256)  # `delta_scan`'s defaults: a thin step, many heads side by side


# --------------------------------------------------------------------------- small helpers (plain JAX)
# (Helpers the kernels call use list comprehensions, not generator expressions: when this module runs as a
# script, autokernel copies them into the kernel's source with the globals they read, and doesn't see the
# ones read only inside a generator expression.)


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
    return sum([_dot(part, upper) for part in _split3(x)])


def cumsum_rows(x):
    """Inclusive cumulative sums along the second-to-last dim (C), f32-accurate: lower-triangular 1s @ x."""
    rows, cols = _iotas(x.shape[-2])
    lower = _bf16(rows >= cols)
    return sum([_dot(lower, part) for part in _split3(x)])


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


def _head_row(x, head):
    """Row `head` (a traced int: it comes from a loop variable) of a (H, n) tile, as (1, n)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    return jnp.sum(jnp.where(rows == head, x, 0.0), axis=0, keepdims=True)


def _column(row):
    """A (1, n) row as an (n, n) matrix of columns: element [r, s] is row[r] (a transpose, on the XLU)."""
    n = row.shape[-1]
    return jnp.broadcast_to(row, (n, n)).T


def _zero_rows(x, n):
    """`x` (rows, m) with zero rows appended up to `n` rows."""
    return jnp.concatenate([x, jnp.zeros((n - x.shape[0], x.shape[1]), x.dtype)]) if x.shape[0] < n else x


def _whole_rows(x):
    """`x` (rows, n) with zero rows appended up to a multiple of 8: an MXU operand of whole sublane tiles."""
    return _zero_rows(x, x.shape[0] + -x.shape[0] % 8)


# --------------------------------------------------------------------------- the chunk math
# On lists of chunk-heads (each 2-D: (C, HEAD_DIM) tiles, (1, C) rows), stage by stage: each line does one
# thing for every chunk-head before the next line starts, so consecutive matmuls are independent. The
# kernels pass the chunk-heads of a step; the plain-JAX baseline vmaps lists of one over every chunk.


def inverse_by_doubling(a):
    """T = (I + A)^-1 for each strictly lower-triangular (C, C) f32 A, by doubling (see the module
    docstring); bf16, as the matmuls take it."""
    rows, cols = _iotas(CHUNK)
    apart = rows ^ cols  # < s iff in the same s x s block (s a power of two)
    minus_a = [_bf16(-x) for x in a]
    t = [jnp.where(apart < 2, jnp.where(rows == cols, 1.0, -x), 0.0) for x in a]  # T_2
    s = 2
    while s < CHUNK:
        new = (apart >= s) & (apart < 2 * s)  # the blocks T_2s adds: below T_s's, within 2s x 2s blocks
        tb = [_bf16(x) for x in t]
        z = [_bf16(_dot(na, x)) for na, x in zip(minus_a, tb, strict=True)]  # -A T_s
        t = [jnp.where(new, _dot(x, zz), old) for x, zz, old in zip(tb, z, t, strict=True)]  # T_s (-A T_s) T_s
        s *= 2
    return [_bf16(x) for x in t]


def _packed(w_u, q_gamma, qk, k_dec, decay):
    """A chunk-head's operands for its state update: [W; Gamma Q] (2C, d_k), U (C, d_v), the attention
    (C, C) and the decayed keys transposed (d_k, C), bf16 (the split form stores them as they are, one
    above the other); and its decay over the chunk (1, d_k) f32."""
    return (
        jnp.concatenate([_bf16(w_u[:, :HEAD_DIM]), _bf16(q_gamma)]),
        _bf16(w_u[:, HEAD_DIM:]),
        qk,
        k_dec,
        decay,
    )


def gdn_operands(q, k, v, g, beta):
    """Gated DeltaNet's state-independent work on chunks: lists of q, k, v (C, HEAD_DIM) bf16 (not yet
    normalized) and of rows (1, C) f32: cumulative log decays G (clamped to <= 0) and betas (clamped to
    [0, 1]). A list of `_packed` operands."""
    rows, cols = _iotas(CHUNK)
    causal = jnp.where(rows >= cols, 0.0, -jnp.inf)  # a bound on the decay's exponent: 0 above the diagonal
    qn = [l2_normalize(x, HEAD_DIM**-0.5) for x in q]
    kn = [l2_normalize(x) for x in k]
    kb = [_bf16(x) for x in kn]
    g = [jnp.minimum(x, 0.0) for x in g]
    beta = [jnp.clip(x, 0.0, 1.0) for x in beta]
    g_col = [_column(x) for x in g]  # [r, s] = G_r
    # exp(G_r - G_s) beta_s on and below the diagonal: the decays and betas of both K K^T and Q K^T
    scale = [jnp.exp(jnp.minimum(gc - gr, causal)) * b for gc, gr, b in zip(g_col, g, beta, strict=True)]
    kq = [_dot_nt(jnp.concatenate([x, _bf16(y)]), x) for x, y in zip(kb, qn, strict=True)]  # [K; Q] K^T
    a = [jnp.where(rows > cols, x[:CHUNK] * d, 0.0) for x, d in zip(kq, scale, strict=True)]
    qk = [_bf16(x[CHUNK:] * d) for x, d in zip(kq, scale, strict=True)]
    t = inverse_by_doubling(a)
    gamma = [jnp.exp(x) for x in g_col]  # exp(G_r), row r in every lane: it scales rows of (C, HEAD_DIM) tiles
    w_u = [_dot(x, jnp.concatenate([_bf16(y * gm), vv], axis=1)) for x, y, gm, vv in zip(t, kn, gamma, v, strict=True)]
    g_last = [jnp.min(x, axis=1, keepdims=True) for x in g]  # G_C (log decays are <= 0)
    k_dec = [_bf16(x.T * (b * jnp.exp(gl - y))) for x, b, gl, y in zip(kn, beta, g_last, g, strict=True)]
    return [
        _packed(wu, x * gm, a_qk, kd, jnp.broadcast_to(jnp.exp(gl), (1, HEAD_DIM)))
        for wu, x, gm, a_qk, kd, gl in zip(w_u, qn, gamma, qk, k_dec, g_last, strict=True)
    ]


def kda_operands(q, k, v, g, beta):
    """KDA's state-independent work on chunks: as `gdn_operands`, with g a list of (C, HEAD_DIM) f32
    cumulative log decays per key channel (of log decays in [KDA_GATE_MIN, 0])."""
    rows, cols = _iotas(CHUNK)  # (also rows of (C, HEAD_DIM) tiles: C = HEAD_DIM)
    qn = [l2_normalize(x, HEAD_DIM**-0.5) for x in q]
    kn = [l2_normalize(x) for x in k]
    beta = [jnp.clip(x, 0.0, 1.0) for x in beta]
    qk_rows = [jnp.concatenate([x, y]) for x, y in zip(kn, qn, strict=True)]  # [K; Q] (2C, d_k)
    akk, aqk = [[] for _ in q], [[] for _ in q]
    for lo in range(0, CHUNK, SUB):  # exp(G_r - G_s) = exp(G_r - G_ref) exp(G_ref - G_s), a reference per block
        ref = [x[lo + SUB // 2 : lo + SUB // 2 + 1] for x in g]
        rel = [jnp.exp(x[lo : lo + SUB] - r) for x, r in zip(g, ref, strict=True)]  # within exp(+-80)
        # Only keys up to the block's last row meet its rows (the rest are zero rows of the matmul).
        kc = [
            _zero_rows(_bf16(x[: lo + SUB] * jnp.exp(r - y[: lo + SUB])), CHUNK)
            for x, r, y in zip(kn, ref, g, strict=True)
        ]
        lhs = [
            _bf16(jnp.concatenate([x[lo : lo + SUB], x[CHUNK + lo : CHUNK + lo + SUB]]) * jnp.concatenate([rl, rl]))
            for x, rl in zip(qk_rows, rel, strict=True)
        ]
        for i, block in enumerate([_dot_nt(x, y) for x, y in zip(lhs, kc, strict=True)]):
            akk[i].append(block[:SUB])
            aqk[i].append(block[SUB:])
    a = [jnp.where(rows > cols, jnp.concatenate(x) * b, 0.0) for x, b in zip(akk, beta, strict=True)]
    qk = [_bf16(jnp.where(rows >= cols, jnp.concatenate(x), 0.0) * b) for x, b in zip(aqk, beta, strict=True)]
    t = inverse_by_doubling(a)
    gamma = [jnp.exp(x) for x in g]
    w_u = [_dot(x, jnp.concatenate([_bf16(y * gm), vv], axis=1)) for x, y, gm, vv in zip(t, kn, gamma, v, strict=True)]
    g_last = [jnp.min(x, axis=0, keepdims=True) for x in g]  # G_C per channel
    k_dec = [_bf16((x * jnp.exp(gl - y)).T * b) for x, gl, y, b in zip(kn, g_last, g, beta, strict=True)]
    return [
        _packed(wu, x * gm, a_qk, kd, jnp.exp(gl))
        for wu, x, gm, a_qk, kd, gl in zip(w_u, qn, gamma, qk, k_dec, g_last, strict=True)
    ]


def scan_chunk(ops, state):
    """One chunk's state update for each head of a group: lists of `_packed` operands and of states S
    (d_k, d_v) f32. The chunk's outputs (C, d_v) f32, and the states after it."""
    # The matmuls in the chain from one state to the next come first (W S, then the keys' update); those
    # for the outputs, after: the MXU runs them while the vector units finish the new states.
    sb = [_bf16(s) for s in state]
    u = [_bf16(op[1].astype(jnp.float32) - _dot(op[0][:CHUNK], s)) for op, s in zip(ops, sb, strict=True)]
    update = [_dot(op[3], uu) for op, uu in zip(ops, u, strict=True)]
    # Decays are <= 1 (also on random operands, which tuning runs on).
    decay = [_column(jnp.clip(op[4], 0.0, 1.0)) for op in ops]
    new_state = [d * s + x for d, s, x in zip(decay, state, update, strict=True)]
    out = [_dot(op[0][CHUNK:], s) + _dot(op[2], uu) for op, s, uu in zip(ops, sb, u, strict=True)]
    return out, new_state


def _chunk_heads(x):
    """The (C, HEAD_DIM) pieces of a (chunks * C, heads * HEAD_DIM) tile, chunk-major."""
    n, width = x.shape
    return [x[c : c + CHUNK, h : h + HEAD_DIM] for c in range(0, n, CHUNK) for h in range(0, width, HEAD_DIM)]


def _scan_block(ops, state):
    """A step's chunks in order, the group's heads side by side: each chunk's outputs (C, heads * d_v) f32,
    and the states (heads, d_k, d_v) after them."""
    heads = state.shape[0]
    states, outs = [state[h] for h in range(heads)], []
    for c in range(0, len(ops), heads):
        out, states = scan_chunk(ops[c : c + heads], states)
        outs.append(jnp.concatenate(out, axis=1))
    return outs, jnp.stack(states)


def gdn_block(q, k, v, g, beta, col, state):
    """A block of tokens for a group of heads (a step of the Gated DeltaNet kernels): its outputs (block,
    heads * HEAD_DIM) f32, and the heads' states (heads, d_k, d_v) after it.

    q, k, v (block, heads * HEAD_DIM) are the group's columns, from column `col` of the arrays; g and beta
    (H, block) every head's log decays and betas."""
    n = q.shape[0]
    first = col // HEAD_DIM
    # Every head's cumulative log decays per chunk, one set of (H, C) matmuls for all of them.
    g_cum = [cumsum_lanes(_whole_rows(jnp.minimum(g[:, c : c + CHUNK], 0.0))) for c in range(0, n, CHUNK)]
    pieces = [(c, h) for c in range(0, n, CHUNK) for h in range(state.shape[0])]
    ops = gdn_operands(
        _chunk_heads(q),
        _chunk_heads(k),
        _chunk_heads(v),
        [_head_row(g_cum[c // CHUNK], first + h) for c, h in pieces],
        [_head_row(beta[:, c : c + CHUNK], first + h) for c, h in pieces],
    )
    outs, state = _scan_block(ops, state)
    return jnp.concatenate(outs), state


def kda_block(q, k, v, g, beta, col, state):
    """A block of tokens for a group of heads in KDA: as `gdn_block`, with g like q (log decays per key
    channel)."""
    n = q.shape[0]
    first = col // HEAD_DIM
    # Every channel's cumulative log decays per chunk: (C, C) @ (C, heads * HEAD_DIM) matmuls.
    g_cum = [cumsum_rows(jnp.clip(g[c : c + CHUNK].astype(jnp.float32), KDA_GATE_MIN, 0.0)) for c in range(0, n, CHUNK)]
    pieces = [(c, h) for c in range(0, n, CHUNK) for h in range(state.shape[0])]
    ops = kda_operands(
        _chunk_heads(q),
        _chunk_heads(k),
        _chunk_heads(v),
        [g_cum[c // CHUNK][:, h * HEAD_DIM : (h + 1) * HEAD_DIM] for c, h in pieces],
        [_head_row(beta[:, c : c + CHUNK], first + h) for c, h in pieces],
    )
    outs, state = _scan_block(ops, state)
    return jnp.concatenate(outs), state


# --------------------------------------------------------------------------- the kernels


@autokernel.kernel
def gated_delta_rule(q, k, v, g, beta, tiles=DeltaTiles()):
    """Gated DeltaNet, chunked. q, k, v (B, S, H * HEAD_DIM), q and k not yet normalized; g and beta
    (B, H, S) f32: log decays (clamped to <= 0) and betas (clamped to [0, 1]). The outputs (B, S, H * HEAD_DIM)."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    T = min(tiles.block, S)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)  # per head (d_k, d_v)
            for c in range(0, S, T):  # a block of tokens per step, its outputs stored by the step
                qt, kt, vt = q[b, c : c + T, j : j + W], k[b, c : c + T, j : j + W], v[b, c : c + T, j : j + W]
                o, state = gdn_block(qt, kt, vt, g[b, :, c : c + T], beta[b, :, c : c + T], j, state)
                out = out.at[b, c : c + T, j : j + W].set(o.astype(dtype))
    return out


@autokernel.kernel
def gated_delta_rule_gated(q, k, v, g, beta, z, weight, tiles=DeltaTiles()):
    """`gated_delta_rule` followed by Qwen3-Next's gated RMSNorm: each head's outputs RMS-normalized, times
    `weight` (1, HEAD_DIM), times silu(z), z (B, S, >= H * HEAD_DIM): its first columns."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    T = min(tiles.block, S)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            for c in range(0, S, T):
                qt, kt, vt = q[b, c : c + T, j : j + W], k[b, c : c + T, j : j + W], v[b, c : c + T, j : j + W]
                o, state = gdn_block(qt, kt, vt, g[b, :, c : c + T], beta[b, :, c : c + T], j, state)
                normed = gated_rmsnorm(o, z[b, c : c + T, j : j + W], weight[:, :])
                out = out.at[b, c : c + T, j : j + W].set(normed.astype(dtype))
    return out


@autokernel.kernel
def kimi_delta_attention(q, k, v, g, beta, tiles=DeltaTiles()):
    """KDA, chunked: as `gated_delta_rule`, with g (B, S, H * HEAD_DIM) f32, log decays per key channel
    (clamped to [KDA_GATE_MIN, 0])."""
    B, S, D = v.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    T = min(tiles.block, S)
    dtype = v.dtype
    out = jnp.zeros((B, S, D), dtype)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            for c in range(0, S, T):
                qt, kt, vt = q[b, c : c + T, j : j + W], k[b, c : c + T, j : j + W], v[b, c : c + T, j : j + W]
                gt, bt = g[b, c : c + T, j : j + W], beta[b, :, c : c + T]
                o, state = kda_block(qt, kt, vt, gt, bt, j, state)
                out = out.at[b, c : c + T, j : j + W].set(o.astype(dtype))
    return out


# --------------------------------------------------------------------------- the split form
# Inputs by chunk: (B, S, D) arrays reshaped (for free) to (B, NC, C, D), and per-token rows of each head
# laid out as (B, NC, rows, 1, D): row i of chunk n of head h is element [b, n, i, 0, h * C : (h + 1) * C].


def _pack(op):
    """`_packed` operands as the split form stores them: (PACKED, HEAD_DIM) bf16, and the decay."""
    return jnp.concatenate(op[:4]), op[4]


def _store_chunks(ops):
    """A tile's chunks' operands as its two output tiles: (n, PACKED, HEAD_DIM) and (n, 1, HEAD_DIM)."""
    packed = [_pack(op) for op in ops]
    return jnp.stack([p for p, _ in packed]), jnp.stack([d for _, d in packed])


@autokernel.kernel
def gdn_chunks(q, k, v, rows, tiles=ChunkTiles()):
    """The split form's parallel kernel for Gated DeltaNet: q, k, v (B, NC, C, D) bf16, rows (B, NC, 2, 1, D)
    f32 (cumulative log decays G, betas). Each chunk's operands (B, NC, PACKED, D) bf16, and its decay
    (B, NC, 1, D) f32."""
    B, NC, _, D = v.shape
    TN = min(tiles.chunks, NC)
    packed = jnp.zeros((B, NC, PACKED, D), jnp.bfloat16)
    decays = jnp.zeros((B, NC, 1, D), jnp.float32)
    for b in range(B):
        for n in range(0, NC, TN):
            for j in range(0, D, HEAD_DIM):
                qt, kt = q[b, n : n + TN, :, j : j + HEAD_DIM], k[b, n : n + TN, :, j : j + HEAD_DIM]
                vt, rt = v[b, n : n + TN, :, j : j + HEAD_DIM], rows[b, n : n + TN, :, :, j : j + HEAD_DIM]
                ops = gdn_operands(
                    [qt[i] for i in range(TN)],
                    [kt[i] for i in range(TN)],
                    [vt[i] for i in range(TN)],
                    [rt[i, 0] for i in range(TN)],
                    [rt[i, 1] for i in range(TN)],
                )
                pt, dt = _store_chunks(ops)
                packed = packed.at[b, n : n + TN, :, j : j + HEAD_DIM].set(pt)
                decays = decays.at[b, n : n + TN, :, j : j + HEAD_DIM].set(dt)
    return packed, decays


@autokernel.kernel
def kda_chunks(q, k, v, g, rows, tiles=ChunkTiles()):
    """The split form's parallel kernel for KDA: q, k, v (B, NC, C, D) bf16, g (B, NC, C, D) f32 log decays
    per key channel, rows (B, NC, 1, 1, D) f32 (betas). As `gdn_chunks`."""
    B, NC, _, D = v.shape
    TN = min(tiles.chunks, NC)
    packed = jnp.zeros((B, NC, PACKED, D), jnp.bfloat16)
    decays = jnp.zeros((B, NC, 1, D), jnp.float32)
    for b in range(B):
        for n in range(0, NC, TN):
            for j in range(0, D, HEAD_DIM):
                qt, kt = q[b, n : n + TN, :, j : j + HEAD_DIM], k[b, n : n + TN, :, j : j + HEAD_DIM]
                vt, gt = v[b, n : n + TN, :, j : j + HEAD_DIM], g[b, n : n + TN, :, j : j + HEAD_DIM]
                rt = rows[b, n : n + TN, :, :, j : j + HEAD_DIM]
                ops = kda_operands(
                    [qt[i] for i in range(TN)],
                    [kt[i] for i in range(TN)],
                    [vt[i] for i in range(TN)],
                    [cumsum_rows(jnp.clip(gt[i], KDA_GATE_MIN, 0.0)) for i in range(TN)],
                    [rt[i, 0] for i in range(TN)],
                )
                pt, dt = _store_chunks(ops)
                packed = packed.at[b, n : n + TN, :, j : j + HEAD_DIM].set(pt)
                decays = decays.at[b, n : n + TN, :, j : j + HEAD_DIM].set(dt)
    return packed, decays


def scan_step(packed, decays, state, first):
    """A step of `delta_scan`: packed (chunks, PACKED, heads * HEAD_DIM) operands and decays (chunks, 1,
    heads * HEAD_DIM), from chunk `first` on; the chunks' outputs (chunks, C, heads * d_v) f32 and the states
    after them."""
    # The delta rule's states stay near the size of its values (each token's update is a contraction plus
    # beta k v^T); on random operands, which tuning runs on, they would grow ~100x per chunk and overflow.
    # Clamped every STATE_CLAMP_CHUNKS chunks of the sequence, not every step: a step's chunks are the
    # tunable's, and the result mustn't be.
    limit = jnp.where(first % STATE_CLAMP_CHUNKS == 0, STATE_MAX, jnp.inf)
    state = jnp.clip(state, -limit, limit)
    ops = []
    for c in range(packed.shape[0]):
        for h in range(0, packed.shape[2], HEAD_DIM):
            x = packed[c, :, h : h + HEAD_DIM]
            rows = [x[i : i + CHUNK] for i in range(2 * CHUNK, PACKED, CHUNK)]
            ops.append((x[: 2 * CHUNK], *rows, decays[c, :, h : h + HEAD_DIM]))
    outs, state = _scan_block(ops, state)
    return jnp.stack(outs), state


@autokernel.kernel
def delta_scan(packed, decays, tiles=SCAN_TILES):
    """The split form's sequential kernel: from `gdn_chunks`' or `kda_chunks`' outputs, the delta rule's
    outputs (B, NC, C, D), chunk by chunk, every head's state carried in VMEM."""
    B, NC, _, D = packed.shape
    G = min(tiles.heads, D // HEAD_DIM)
    W = G * HEAD_DIM
    N = min(tiles.block // CHUNK, NC)
    assert N == NC or STATE_CLAMP_CHUNKS % N == 0  # (every step starts at the chunks the states are clamped at)
    out = jnp.zeros((B, NC, CHUNK, D), jnp.bfloat16)
    for b in range(B):
        for j in range(0, D, W):
            state = jnp.zeros((G, HEAD_DIM, HEAD_DIM), jnp.float32)
            for c in range(0, NC, N):
                o, state = scan_step(packed[b, c : c + N, :, j : j + W], decays[b, c : c + N, :, j : j + W], state, c)
                out = out.at[b, c : c + N, :, j : j + W].set(o.astype(jnp.bfloat16))
    return out


def by_chunk(x: jax.Array) -> jax.Array:
    """(B, S, D) as (B, S / C, C, D): free."""
    B, S, D = x.shape
    return x.reshape(B, S // CHUNK, CHUNK, D)


def head_rows(*xs: jax.Array) -> jax.Array:
    """Per-token rows (B, H, S) of each head, as (B, NC, len(xs), 1, H * C) f32: chunk n's rows of head h at
    lanes h * C to (h + 1) * C."""
    B, H, S = xs[0].shape
    stacked = jnp.stack(xs).astype(jnp.float32).reshape(len(xs), B, H, S // CHUNK, CHUNK)
    return stacked.transpose(1, 3, 0, 2, 4).reshape(B, S // CHUNK, len(xs), 1, H * CHUNK)


def split_inputs(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array) -> tuple[Any, tuple]:
    """The split form's parallel kernel for these inputs (`gdn_chunks` for g (B, H, S), `kda_chunks` for g
    like v), and its inputs: q, k, v (and KDA's g) by chunk; the per-token rows of each head (Gated
    DeltaNet's cumulative log decays, from XLA: a tiny array; the betas)."""
    S = v.shape[1]
    if _per_channel(g, v):
        return kda_chunks, (by_chunk(q), by_chunk(k), by_chunk(v), by_chunk(g), head_rows(beta))
    g = jnp.minimum(g.astype(jnp.float32), 0.0)
    g_cum = jnp.cumsum(g.reshape(*g.shape[:2], S // CHUNK, CHUNK), axis=-1).reshape(g.shape)
    return gdn_chunks, (by_chunk(q), by_chunk(k), by_chunk(v), head_rows(g_cum, beta))


def _split_form(q, k, v, g, beta, tiles):
    chunks, args = split_inputs(q, k, v, g, beta)
    packed, decays = chunks(*args) if tiles is None else chunks(*args, tiles[0])
    out = delta_scan(packed, decays) if tiles is None else delta_scan(packed, decays, tiles[1])
    return out.reshape(v.shape).astype(v.dtype)


def gated_delta_rule_split(q, k, v, g, beta, tiles: tuple[ChunkTiles, DeltaTiles] | None = None):
    """`gated_delta_rule` as flash-linear-attention structures it: `gdn_chunks`, then `delta_scan` (each
    tuned on its first call, or with `tiles`)."""
    return _split_form(q, k, v, g, beta, tiles)


def kimi_delta_attention_split(q, k, v, g, beta, tiles: tuple[ChunkTiles, DeltaTiles] | None = None):
    """`kimi_delta_attention` as flash-linear-attention structures it: `kda_chunks`, then `delta_scan`."""
    return _split_form(q, k, v, g, beta, tiles)


# --------------------------------------------------------------------------- the previous one-pass chunk math
# Each chunk's work in one stretch, chunk after chunk, with the state transposed (d_v, d_k): kept for
# hybrid_layers.py's prefill kernels (`delta_block`), which also store the final states.


def unit_lower_inverse(a):
    """(I + a)^-1 for strictly lower-triangular (..., C, C) `a`, C a power of two, by doubling the size of the
    inverted diagonal blocks: T_2s = T_s - T_s A_off T_s, A_off the blocks of `a` below T_s's within each
    2s x 2s block."""
    n = a.shape[-1]
    rows, cols = _iotas(n)
    apart = rows ^ cols
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
    """Gated DeltaNet's work on a chunk that doesn't need the state (one-pass form). q, k, v (..., C, d) bf16
    (q and k normalized, q scaled); g_row and beta_row (..., 1, C) f32. Returns (w, u~, q_gamma, qk, k_hat,
    decay_last)."""
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
    """KDA's work on a chunk that doesn't need the state (one-pass form): as `gdn_intra`, with g (..., C, d_k)
    f32 log decays per key channel, each in [KDA_GATE_MIN, 0]."""
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
    """The chunk's new values U = U~ - W S (bf16) and the state after it (one-pass form); `state` (..., d_v,
    d_k) f32 is the transposed state S^T at the chunk's start."""
    u = u - _dot_nt(w, _bf16(state))
    return _bf16(u), decay_last * state + _dot(_bf16(jnp.swapaxes(u, -1, -2)), k_hat)


def chunk_output(q_gamma, qk, u, state):
    """The chunk's outputs (..., C, d_v) f32, from the state at its start and its new values (one-pass form)."""
    return _dot_nt(q_gamma, _bf16(state)) + _dot(qk, u)


def delta_block(intra, q, k, v, g, beta, col, state):
    """A block of tokens for a group of heads, chunk after chunk (one-pass form): their outputs (block,
    heads * HEAD_DIM) f32, and their transposed states (heads, d_v, d_k) after it.

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

    def chunks(x):  # (B, S, H * d) -> (n * B * H, C, d)
        return x.reshape(B, n, CHUNK, H, HEAD_DIM).transpose(1, 0, 3, 2, 4).reshape(-1, CHUNK, HEAD_DIM)

    def rows(x):  # (B, H, S) -> (n * B * H, 1, C)
        return x.astype(jnp.float32).reshape(B, H, n, 1, CHUNK).transpose(2, 0, 1, 3, 4).reshape(-1, 1, CHUNK)

    qc, kc, vc = chunks(q), chunks(k), chunks(_bf16(v))
    if _per_channel(g, v):
        g_cum = jnp.cumsum(chunks(jnp.clip(g.astype(jnp.float32), KDA_GATE_MIN, 0.0)), axis=-2)
        math = kda_operands
    else:
        g_cum = jnp.cumsum(rows(jnp.minimum(g, 0.0)), axis=-1)
        math = gdn_operands
    ops = jax.vmap(lambda *xs: math(*([x] for x in xs))[0])(qc, kc, vc, g_cum, rows(beta))
    wq, u, qk, k_dec, decay = (x.reshape(n, B * H, *x.shape[1:]) for x in ops)

    def step(state, chunk):  # state (B * H, d_k, d_v) f32
        w, u, k_dec, decay = chunk
        u = _bf16(u.astype(jnp.float32) - _dot(w, _bf16(state)))
        return jnp.swapaxes(decay, -1, -2) * state + _dot(k_dec, u), (u, _bf16(state))

    state = jnp.zeros((B * H, HEAD_DIM, HEAD_DIM), jnp.float32)
    _, (u, states) = jax.lax.scan(step, state, (wq[:, :, :CHUNK], u, k_dec, decay))
    out = _dot(wq[:, :, CHUNK:], states) + _dot(qk, u)  # (n, B * H, C, d)
    return out.reshape(n, B, H, CHUNK, HEAD_DIM).transpose(1, 0, 3, 2, 4).reshape(B, S, D).astype(v.dtype)


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
