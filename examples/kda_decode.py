"""Kimi Delta Attention (Kimi Linear's linear attention), one decode step: one token per sequence, the
recurrent and short-convolution states in paged pools, updated in place.

WORK IN PROGRESS: inputs, the float64 reference and the plain-JAX path.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import autokernel

# Kimi Linear 48B-A3B's KDA layers (HF config.json, linear_attn_config): 32 heads of 128, a 4-tap conv.
HEADS = 32
HEAD_DIM = 128
CONV = 4
L2_EPS = 1e-6


def make_inputs(batch: int, seed: int = 0, heads: int = HEADS, slots: int | None = None, fresh: float = 0.1):
    """One decode step's inputs in vLLM's decode_kda ABI (the native rank-5 conv state): a pool of
    `slots` (default 2 * batch + 1) state slots, slot 0 the null block; each sequence a distinct random real
    slot; a share `fresh` of them on their first step (has_initial_state 0)."""
    rng = np.random.default_rng(seed)
    slots = 2 * batch + 1 if slots is None else slots
    d = heads * HEAD_DIM

    def normal(shape, scale=1.0, dtype=jnp.bfloat16):
        return jnp.asarray(rng.standard_normal(shape) * scale, dtype)

    beta_raw = normal((batch, heads))
    return {
        "mixed_qkv": normal((batch, 3 * d)),
        "g": normal((batch, heads, HEAD_DIM)),
        "beta": jax.nn.sigmoid(beta_raw.astype(jnp.float32)).astype(jnp.bfloat16),
        "conv_state": normal((slots, CONV - 1, 3, heads, HEAD_DIM), 0.1),
        "conv_weight": normal((CONV, 3, heads, HEAD_DIM), 0.3),
        "state": normal((slots, heads, HEAD_DIM, HEAD_DIM), 0.05, jnp.float32),
        "a_log": normal((heads,), 0.3, jnp.float32),
        "dt_bias": normal((heads, HEAD_DIM), 0.2, jnp.float32),
        "state_indices": jnp.asarray(1 + rng.permutation(slots - 1)[:batch], jnp.int32),
        "has_initial_state": jnp.asarray(rng.random(batch) >= fresh, jnp.int32),
    }


ARGS = (
    "mixed_qkv",
    "g",
    "beta",
    "conv_state",
    "conv_weight",
    "state",
    "a_log",
    "dt_bias",
    "state_indices",
    "has_initial_state",
)


def args_of(inputs: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(inputs[k] for k in ARGS)


# --------------------------------------------------------------------------- the kernel


@autokernel.tunable
class KdaTiles:
    heads: int = autokernel.choice(16, 32, default=16)  # heads per tile (bf16 tiles: 16 rows at least)


def short_conv(x, prev, w):
    """The causal depthwise convolution and SiLU of one token: x (3, TH, d_k) f32, the CONV - 1 inputs before
    it prev (CONV - 1, 3, TH, d_k), the taps w (CONV, 3, TH, d_k); f32 (the newest tap first, as vLLM's)."""
    pre = x * w[CONV - 1]
    for tap in range(CONV - 1):
        pre = pre + prev[tap] * w[tap]
    return jax.nn.silu(pre)


def l2_normalize(x, scale=1.0):
    """Each row of x (TH, d) over its L2 norm, times `scale`."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + L2_EPS) * scale


def decays(g, dt_bias, a_log, lower_bound):
    """The decays exp(gate) per key channel (TH, d_k): gate = -exp(A_log) softplus(g + dt_bias), or with a
    lower bound, lower_bound * sigmoid(exp(A_log) (g + dt_bias)) (vLLM's two forms)."""
    x = g + dt_bias
    if lower_bound is None:
        return jnp.exp(-jnp.exp(a_log) * jax.nn.softplus(x))
    return jnp.exp(lower_bound * jax.nn.sigmoid(jnp.exp(a_log) * x))


def columns(x):
    """Rows x (TH, d) as columns (TH, d, 1): the transpose a select and a lane sum make (exact)."""
    th, d = x.shape
    eye = jax.lax.broadcasted_iota(jnp.int32, (th, d, d), 1) == jax.lax.broadcasted_iota(jnp.int32, (th, d, d), 2)
    return jnp.sum(jnp.where(eye, x[:, None, :], 0.0), axis=2, keepdims=True)


def delta_rule(q, k, v, a, beta, s):
    """One token of KDA for TH heads, f32, in vLLM's order: rows q, k (normalized, q scaled), v, the decays a
    (TH, d_k), betas (TH, 1), states s (TH, d_k, d_v):

        S' = Diag(a) S;  u = beta (v - S'^T k);  S'' = S' + k u^T;  o = S''^T q

    The two matrix-vector products are MXU matmuls with S' and S'' as the weights, at the MXU's default
    precision for f32 (as vLLM's kernel), so the results are the same."""
    batched = (((2,), (1,)), ((0,), (0,)))
    decayed = s * columns(a)
    read = jax.lax.dot_general(k[:, None, :], decayed, batched, preferred_element_type=jnp.float32)
    u = beta[:, :, None] * (v[:, None, :] - read)  # (TH, 1, d_v)
    new = decayed + columns(k) * u
    out = jax.lax.dot_general(q[:, None, :], new, batched, preferred_element_type=jnp.float32)
    return out[:, 0, :], new


def make_kda_decode_step(lower_bound: float | None = None, state_transposed: bool = False) -> autokernel.Kernel:
    """The step as one loop nest, a tile per (sequence, block of heads), for vLLM's options: the gate's lower
    bound (None: the softplus form), and pools whose slots hold each head's state as (d_v, d_k)."""

    @autokernel.kernel
    def kda_decode_step(
        mixed_qkv,
        g,
        beta,
        conv_state,
        conv_weight,
        state,
        a_log,
        dt_bias,
        state_indices,
        has_initial_state,
        tiles=KdaTiles(),
    ):
        """mixed_qkv (B, 3, H, d_k), g (B, H, d_k), beta (B, H, 1), the pools conv_state
        (slots, CONV - 1, 3, H, d_k) and state (slots, H, d_k, d_v), conv_weight (CONV, 3, H, d_k), a_log
        (H, 1), dt_bias (H, d_k) f32, int32 state_indices and has_initial_state (B,). The pools are updated in
        place: each sequence's slot is read and written back where the index array says, nothing else."""
        B, _, H, K = mixed_qkv.shape
        TH = min(tiles.heads, H)
        scale = K**-0.5
        dtype, conv_dtype = mixed_qkv.dtype, conv_state.dtype
        out = jnp.zeros((B, H, K), dtype)
        new_conv = conv_state
        new_state = state
        for b in range(B):
            for h in range(0, H, TH):
                valid = state_indices[b] > 0  # slot 0 is the null block (padding): write it back as found
                carry = valid & (has_initial_state[b] != 0)  # else the sequence starts from zero states
                stored_conv = conv_state[state_indices[b], :, :, h : h + TH, :]
                prev = jnp.where(carry, stored_conv.astype(jnp.float32), 0.0)
                x = mixed_qkv[b, :, h : h + TH, :].astype(jnp.float32)
                conv = short_conv(x, prev, conv_weight[:, :, h : h + TH, :].astype(jnp.float32))
                conv = conv.astype(dtype).astype(jnp.float32)  # (rounded as vLLM's kernel rounds it)
                q = l2_normalize(conv[0], scale)
                k = l2_normalize(conv[1])
                a = decays(
                    g[b, h : h + TH, :].astype(jnp.float32), dt_bias[h : h + TH, :], a_log[h : h + TH, :], lower_bound
                )
                stored = state[state_indices[b], h : h + TH, :, :]
                s = jnp.swapaxes(stored, 1, 2) if state_transposed else stored
                o, s = delta_rule(
                    q, k, conv[2], a, beta[b, h : h + TH, :].astype(jnp.float32), jnp.where(carry, s, 0.0)
                )
                s = jnp.swapaxes(s, 1, 2) if state_transposed else s
                tail = jnp.concatenate([prev[1:], x[None]]).astype(conv_dtype)
                out = out.at[b, h : h + TH, :].set(jnp.where(valid, o, 0.0).astype(dtype))
                new_conv = new_conv.at[state_indices[b], :, :, h : h + TH, :].set(jnp.where(valid, tail, stored_conv))
                new_state = new_state.at[state_indices[b], h : h + TH, :, :].set(jnp.where(valid, s, stored))
        return out, new_conv, new_state

    return kda_decode_step


kda_decode_step = make_kda_decode_step()
_STEPS = {(None, False): kda_decode_step}


def step_kernel(lower_bound: float | None = None, state_transposed: bool = False) -> autokernel.Kernel:
    """The kernel for these options (made once)."""
    key = (lower_bound, state_transposed)
    if key not in _STEPS:
        _STEPS[key] = make_kda_decode_step(lower_bound, state_transposed)
    return _STEPS[key]


def step_args(mixed_qkv, g, beta, conv_state, conv_weight, state, a_log, dt_bias, state_indices, has_initial_state):
    """`kda_decode`'s arguments as `kda_decode_step` takes them (the reshapes vLLM's kernel makes too)."""
    B, H, K = g.shape
    return (
        mixed_qkv.reshape(B, 3, H, K),
        g,
        beta.reshape(B, H, 1),
        conv_state,
        conv_weight,
        state,
        a_log.reshape(H, 1).astype(jnp.float32),
        dt_bias.astype(jnp.float32),
        state_indices.astype(jnp.int32),
        has_initial_state.astype(jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("lower_bound", "state_transposed"))
def kda_decode(
    mixed_qkv,
    g,
    beta,
    conv_state,
    conv_weight,
    state,
    a_log,
    dt_bias,
    state_indices,
    has_initial_state,
    *,
    lower_bound=None,
    state_transposed=False,
):
    """vLLM decode_kda's ABI (with the rank-5 conv state): mixed_qkv (B, 3 * H * d_k), g (B, H, d_k) raw,
    beta (B, H) activated, the pools, conv_weight (CONV, 3, H, d_k), a_log (H,), dt_bias (H, d_k),
    state_indices (B,) (0: a null entry), has_initial_state (B,). Returns (out (B, H, d_v), conv_state,
    state); donate the pools to update them in place. (Tune it on real inputs first,
    `step_kernel(...).tuning(*step_args(...))`: under jax.jit it would tune on random ones.)"""
    args = (mixed_qkv, g, beta, conv_state, conv_weight, state, a_log, dt_bias, state_indices, has_initial_state)
    return step_kernel(lower_bound, state_transposed)(*step_args(*args))


# --------------------------------------------------------------------------- float64 reference


def _conv(window: np.ndarray, conv_weight: jax.Array) -> jax.Array:
    """The short convolution and SiLU of one sequence's window (CONV, 3, H, d_k), as the kernels compute it:
    f32, the newest tap first, rounded to bf16."""
    x, w = jnp.asarray(window, jnp.float32), conv_weight.astype(jnp.float32)
    pre = x[CONV - 1] * w[CONV - 1]
    for tap in range(CONV - 1):
        pre = pre + x[tap] * w[tap]
    return jax.nn.silu(pre).astype(jnp.bfloat16)


def reference(
    inputs: dict[str, Any], seqs: Any = None, lower_bound: float | None = None, state_transposed: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The step in float64 NumPy for sequences `seqs` (default all): (out (n, H, d_v), the new conv states
    (n, CONV - 1, 3, H, d_k), the new states (n, H, d_k, d_v), or (n, H, d_v, d_k) as the pool keeps them if
    `state_transposed`), per sequence (a null slot's: zeros, and its slot's contents unchanged). The
    convolution is computed as the kernels compute it (f32, rounded to bf16 after the SiLU): in f64 a value
    near a bf16 rounding boundary would round the other way, which changes the state by ~1e-4 (one bf16 step
    of a key)."""
    f = lambda x: np.asarray(x, np.float64)  # noqa: E731
    idx = np.asarray(inputs["state_indices"])
    has = np.asarray(inputs["has_initial_state"])
    seqs = range(len(idx)) if seqs is None else seqs
    w = f(inputs["conv_weight"])
    a = np.exp(f(inputs["a_log"]))[:, None]
    dt = f(inputs["dt_bias"])
    scale = HEAD_DIM**-0.5
    outs, convs, states = [], [], []
    for b in seqs:
        slot, carry = int(idx[b]), bool(has[b]) and idx[b] > 0
        stored_conv, stored = f(inputs["conv_state"][slot]), f(inputs["state"][slot])
        if state_transposed:
            stored = np.swapaxes(stored, 1, 2)
        prev = stored_conv if carry else np.zeros_like(stored_conv)
        x = f(inputs["mixed_qkv"][b]).reshape(3, *w.shape[2:])
        window = np.concatenate([prev, x[None]])
        conv = f(_conv(window, inputs["conv_weight"]))
        q, k, v = conv
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + L2_EPS) * scale
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + L2_EPS)
        gate = f(inputs["g"][b]) + dt
        if lower_bound is None:
            decay = np.exp(-a * np.logaddexp(0.0, gate))
        else:
            decay = np.exp(lower_bound / (1 + np.exp(-a * gate)))
        beta = f(inputs["beta"][b])
        s = (stored if carry else np.zeros_like(stored)) * decay[:, :, None]
        u = beta[:, None] * (v - np.einsum("hk,hkv->hv", k, s))
        s = s + k[:, :, None] * u[:, None, :]
        out = np.einsum("hk,hkv->hv", q, s)
        if state_transposed:
            s, stored = np.swapaxes(s, 1, 2), np.swapaxes(stored, 1, 2)
        if slot > 0:
            outs.append(out), convs.append(window[1:]), states.append(s)
        else:
            outs.append(np.zeros_like(out)), convs.append(stored_conv), states.append(stored)
    return np.stack(outs), np.stack(convs), np.stack(states)


# --------------------------------------------------------------------------- plain JAX (XLA)


@jax.jit
def xla_decode(mixed_qkv, g, beta, conv_state, conv_weight, state, a_log, dt_bias, state_indices, has_initial_state):
    """The step in plain JAX, as hybrid_layers.py's decode does it (gather the slots, conv, gates,
    `delta_step`, scatter them back): (out (B, H, d_v) bf16, conv_state, state)."""
    from hybrid_layers import delta_step

    B, H, K = g.shape
    valid = state_indices > 0
    carry = (valid & (has_initial_state != 0))[:, None, None, None]
    stored_conv = conv_state[state_indices]  # (B, CONV - 1, 3, H, K)
    stored = state[state_indices]
    prev = jnp.where(carry[..., None], stored_conv, 0).astype(jnp.float32)
    x = mixed_qkv.reshape(B, 3, H, K)
    w = conv_weight.astype(jnp.float32)
    pre = x.astype(jnp.float32) * w[CONV - 1]
    for tap in range(CONV - 1):
        pre = pre + prev[:, tap] * w[tap]
    conv = jax.nn.silu(pre).astype(mixed_qkv.dtype).astype(jnp.float32)
    q, k, v = conv[:, 0], conv[:, 1], conv[:, 2]
    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + L2_EPS) * K**-0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + L2_EPS)
    decay = jnp.exp(-jnp.exp(a_log)[:, None] * jax.nn.softplus(g.astype(jnp.float32) + dt_bias))
    s = jnp.where(carry, stored, 0.0)
    out, new = delta_step(q, k, v, decay, beta.astype(jnp.float32), s)
    out = jnp.where(valid[:, None, None], out, 0).astype(mixed_qkv.dtype)
    new_conv = jnp.concatenate([prev[:, 1:].astype(conv_state.dtype), x[:, None]], axis=1)
    new_conv = jnp.where(valid[:, None, None, None, None], new_conv, stored_conv)
    new = jnp.where(valid[:, None, None, None], new, stored)
    return out, conv_state.at[state_indices].set(new_conv), state.at[state_indices].set(new)
