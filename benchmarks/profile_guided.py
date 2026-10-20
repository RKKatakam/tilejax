"""Profiling and profile-guided tuning on a TPU.

    uv run python benchmarks/profile_guided.py report   # where each kernel's time goes (a kernel, a program)
    uv run python benchmarks/profile_guided.py tune     # autotune="profile"; run it twice: the second run
                                                        # continues from the first's measurements (cache)

The trace of `report` is kept (its directory is printed): `tensorboard --logdir <it>` shows the timeline, with
each kernel's steps split into `<kernel>/wait`, `<kernel>/compute` and `<kernel>/out_wait`.
"""

import sys
import time

import autokernel

autokernel.enable_profiling()  # before JAX starts its TPU backend

import jax
import jax.numpy as jnp

S = 4096
TM, TN, TK = 256, 256, 256


def matmul(x, y):
    out = jnp.zeros((S, S), jnp.bfloat16)
    for i in range(0, S, TM):
        for j in range(0, S, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, S, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc.astype(jnp.bfloat16))
    return out


def mlp(x, w1, w2):
    h = jnp.zeros((S, S), jnp.bfloat16)
    for i in range(0, S, 512):
        for j in range(0, S, 512):
            acc = jnp.zeros((512, 512))
            for k in range(0, S, 512):
                acc += x[i : i + 512, k : k + 512] @ w1[k : k + 512, j : j + 512]
            h = h.at[i : i + 512, j : j + 512].set(jnp.maximum(acc, 0).astype(jnp.bfloat16))
    out = jnp.zeros((S, S), jnp.bfloat16)
    for i in range(0, S, 512):
        for j in range(0, S, 512):
            acc = jnp.zeros((512, 512))
            for k in range(0, S, 512):
                acc += h[i : i + 512, k : k + 512] @ w2[k : k + 512, j : j + 512]
            out = out.at[i : i + 512, j : j + 512].set(acc.astype(jnp.bfloat16))
    return out


def arrays(n):
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return [jax.random.normal(k, (S, S), jnp.bfloat16) for k in keys]


def report():
    x, y = arrays(2)
    print("== a kernel (256^3 blocks: small steps)")
    print(autokernel.profile(autokernel.tpu(matmul), x, y))
    print("\n== a program (two nests, fused)")
    print(autokernel.profile(autokernel.tpu(mlp), *arrays(3)))


def tune():
    x, y = arrays(2)
    kern = autokernel.tpu(matmul, tune="auto", autotune="profile")
    start = time.perf_counter()
    best = kern.lower(x, y)
    took = time.perf_counter() - start
    timed = [c for c in kern.tuning(x, y) if c.measured_s is not None]
    print(f"picked {best.overrides} buffers={best.buffers} in {took:.1f}s; {len(timed)} measured (this run + cache)")
    for c in sorted(timed, key=lambda c: c.measured_s)[:10]:
        print(f"  {c.measured_s * 1e3:8.3f} ms  {c.overrides} b={c.buffers}  {c.bottleneck or ''}")
    kern.verify(x, y)
    print(autokernel.profile(kern, x, y))


if __name__ == "__main__":
    {"report": report, "tune": tune}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
