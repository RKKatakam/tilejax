"""Let autokernel pick the block sizes of a matmul.

`tune` names the module constants the loops use and the values to try. autokernel builds every
combination, drops the ones that break TPU layout rules or don't fit in VMEM, and ranks the rest with a
cost model built from the TPU's specs (`pltpu.get_tpu_info`) and JAX's cost analysis. With
`autotune="measure"` it also times the best few on the TPU.

Run with `uv run python examples/tuned_matmul.py`. Without a TPU it plans for a v5e and runs the chosen
kernel in the Pallas TPU interpreter.
"""

import jax
import jax.numpy as jnp

import autokernel

M = K = N = 2048
TM, TN, TK = 128, 128, 128  # defaults; tuning overrides them


@autokernel.tpu(tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (128, 256, 512)}, autotune="measure")
def matmul(x, y):
    out = jnp.zeros((M, N), jnp.bfloat16)
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN), jnp.float32)
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc.astype(jnp.bfloat16))
    return out


if __name__ == "__main__":
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), jnp.bfloat16)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), jnp.bfloat16)

    lowered = matmul.lower(x, y)
    print("chosen:", lowered.overrides, f"buffers={lowered.buffers}")
    print(f"{'TM':>5} {'TN':>5} {'TK':>5} {'buffers':>7} {'predicted':>12} {'measured':>12}")
    for c in matmul.tuning(x, y)[:10]:
        measured = f"{c.measured_s * 1e3:9.3f} ms" if c.measured_s else "-"
        predicted = f"{c.predicted_s * 1e3:9.3f} ms" if c.error is None else c.error
        tiles = " ".join(f"{c.overrides[k]:>5}" for k in ("TM", "TN", "TK"))
        print(f"{tiles} {c.buffers:>7} {predicted:>12} {measured:>12}")
    print("max diff vs Python loops:", matmul.verify(x, y))
