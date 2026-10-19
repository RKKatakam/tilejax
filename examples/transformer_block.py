"""A stack of transformer blocks written entirely as tile loops.

Each layer is pre-norm attention followed by a pre-norm MLP:

    x = x + attention(rmsnorm(x))          # causal multi-head attention
    x = x + down(relu(up(rmsnorm(x))))

The function below is one `@autokernel.tpu` *program*. Every `for` nest that ends in
`out = out.at[...].set(...)` (with `out` allocated just before it) is compiled to a gridless Pallas
kernel, and nests with no other code between them are fused into one kernel (at most three to a kernel).
The reshapes and transposes between them (`qkv.reshape(...).transpose(...)`) are *views*: the attention
reads each head of q, k and v as a window of the QKV projection's output, and the output projection reads
the attention's output the same way, so XLA copies nothing and the views don't separate the nests. Each
layer's weights are read straight out of the stacked weight arrays with an integer index
(`wqkv[layer, ...]`): `layer` is a runtime scalar of the kernels, so one kernel serves every layer, and
the `for layer in range(L)` loop runs as a `jax.lax.fori_loop`. The seven nests become three kernels,
compiled once whatever L is.

Run with `uv run python examples/transformer_block.py`. Without a TPU it runs in the Pallas TPU interpreter.
"""

import math

import jax
import jax.numpy as jnp

import autokernel

L = 2  # layers
B, S = 2, 256  # batch, sequence length
D, H = 256, 2  # model width, heads
DH = D // H  # head dim
F = 4 * D  # MLP width
T = 128  # tile size of the matmuls and norms
TQ, TK = 128, 128  # query / key-value blocks of the attention
LANES = 128  # TPU vector lanes: per-row softmax statistics are kept lane-wide
EPS = 1e-6
MASK_VALUE = -1e30


@autokernel.tpu
def transformer(x, norm1, wqkv, wo, norm2, w_up, w_down):
    rows = B * S
    x = x.reshape(rows, D)
    for layer in range(L):
        # ---- RMSNorm: a tile is whole rows, so the mean is over the full width.
        xn = jnp.zeros((rows, D))
        for i in range(0, rows, T):
            row = x[i : i + T, :]
            inv_rms = jax.lax.rsqrt(jnp.mean(row * row, axis=1, keepdims=True) + EPS)
            xn = xn.at[i : i + T, :].set(row * inv_rms * norm1[layer, :, :])

        # ---- Fused Q, K, V projection.
        qkv = jnp.zeros((rows, 3 * D))
        for i in range(0, rows, T):
            for j in range(0, 3 * D, T):
                acc = jnp.zeros((T, T))
                for k in range(0, D, T):
                    acc += xn[i : i + T, k : k + T] @ wqkv[layer, k : k + T, j : j + T]
                qkv = qkv.at[i : i + T, j : j + T].set(acc)
        # (q/k/v, batch, head, seq, head dim): a view, read in place by the attention below
        qkv = qkv.reshape(B, S, 3, H, DH).transpose(2, 0, 3, 1, 4)

        # ---- Causal flash attention: per (batch, head, query block), stream over the key blocks
        # at or before it, keeping a running row max, row sum and output (the online softmax).
        attn = jnp.zeros((B, H, S, DH))
        for b in range(B):
            for h in range(H):
                for i in range(0, S, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)
                    row_sum = jnp.zeros((TQ, LANES))
                    acc = jnp.zeros((TQ, DH))
                    for j in range(0, S, TK):
                        if j > i + TQ - 1:  # every key in the block is in the future: skip it entirely
                            continue
                        s = (qkv[0, b, h, i : i + TQ, :] @ qkv[1, b, h, j : j + TK, :].T) * (1 / math.sqrt(DH))
                        q_pos = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                        k_pos = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                        s = jnp.where(k_pos <= q_pos, s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, DH // LANES)) * acc + p @ qkv[2, b, h, j : j + TK, :]
                        row_max = m_new
                    attn = attn.at[b, h, i : i + TQ, :].set(acc / jnp.tile(row_sum, (1, DH // LANES)))
        attn = attn.transpose(0, 2, 1, 3).reshape(rows, D)  # heads back side by side: a view too

        # ---- Output projection and residual.
        x1 = jnp.zeros((rows, D))
        for i in range(0, rows, T):
            for j in range(0, D, T):
                acc = jnp.zeros((T, T))
                for k in range(0, D, T):
                    acc += attn[i : i + T, k : k + T] @ wo[layer, k : k + T, j : j + T]
                x1 = x1.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + acc)

        # ---- MLP: RMSNorm, up projection with ReLU, down projection and residual.
        xn = jnp.zeros((rows, D))
        for i in range(0, rows, T):
            row = x1[i : i + T, :]
            inv_rms = jax.lax.rsqrt(jnp.mean(row * row, axis=1, keepdims=True) + EPS)
            xn = xn.at[i : i + T, :].set(row * inv_rms * norm2[layer, :, :])
        up = jnp.zeros((rows, F))
        for i in range(0, rows, T):
            for j in range(0, F, T):
                acc = jnp.zeros((T, T))
                for k in range(0, D, T):
                    acc += xn[i : i + T, k : k + T] @ w_up[layer, k : k + T, j : j + T]
                up = up.at[i : i + T, j : j + T].set(jnp.maximum(acc, 0))
        x2 = jnp.zeros((rows, D))
        for i in range(0, rows, T):
            for j in range(0, D, T):
                acc = jnp.zeros((T, T))
                for k in range(0, F, T):
                    acc += up[i : i + T, k : k + T] @ w_down[layer, k : k + T, j : j + T]
                x2 = x2.at[i : i + T, j : j + T].set(x1[i : i + T, j : j + T] + acc)
        x = x2
    return x.reshape(B, S, D)


def plain_transformer(x, norm1, wqkv, wo, norm2, w_up, w_down):
    """The same model in plain JAX, at full f32 precision."""
    hp = "highest"

    def rmsnorm(x, g):
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * g

    causal = jnp.tril(jnp.ones((S, S), bool))
    for layer in range(L):
        q, k, v = (
            jnp.einsum("bsd,de->bse", rmsnorm(x, norm1[layer]), wqkv[layer], precision=hp)
            .reshape(B, S, 3, H, DH)
            .transpose(2, 0, 3, 1, 4)
        )
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=hp) / math.sqrt(DH)
        p = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
        attn = jnp.einsum("bhqk,bhkd->bqhd", p, v, precision=hp).reshape(B, S, D)
        x = x + jnp.einsum("bsd,de->bse", attn, wo[layer], precision=hp)
        up = jnp.maximum(jnp.einsum("bsd,df->bsf", rmsnorm(x, norm2[layer]), w_up[layer], precision=hp), 0)
        x = x + jnp.einsum("bsf,fd->bsd", up, w_down[layer], precision=hp)
    return x


if __name__ == "__main__":
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    x = jax.random.normal(keys[0], (B, S, D))
    norm1 = 1 + 0.1 * jax.random.normal(keys[1], (L, 1, D))
    wqkv = jax.random.normal(keys[2], (L, D, 3 * D)) / math.sqrt(D)
    wo = jax.random.normal(keys[3], (L, D, D)) / math.sqrt(D)
    norm2 = 1 + 0.1 * jax.random.normal(keys[4], (L, 1, D))
    w_up = jax.random.normal(keys[5], (L, D, F)) / math.sqrt(D)
    w_down = jax.random.normal(keys[6], (L, F, D)) / math.sqrt(F)
    args = (x, norm1, wqkv, wo, norm2, w_up, w_down)

    lowered = transformer.lower(*args)
    unfused = autokernel.tpu(transformer.fn, fuse=False).lower(*args)
    print(f"{len(transformer.kernels)} tile loop nests: {len(unfused)} kernels, {len(lowered)} fused, for {L} layers:")
    for name, kernel in lowered:
        print(f"  {name:64} {kernel.steps:4} steps, {kernel.tiles:3} tiles")
    for cut in lowered.cuts:
        print(f"  {cut}")
    windows = [line.strip() for line in lowered.source.splitlines() if line.strip().startswith("src = qkv_hbm")]
    print("the attention reads q, k and v as windows of the QKV projection's output:", *windows, sep="\n  ")
    assert "transpose" not in jax.jit(transformer).lower(*args).as_text()  # XLA copies nothing

    out = transformer(*args)
    err = jnp.max(jnp.abs(out - plain_transformer(*args)))
    print("max abs error vs plain JAX:", err)
    print("max abs error vs the Python loops:", transformer.verify(*args))
