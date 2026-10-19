"""Code autokernel must reject, with a message pointing at the problem."""

import importlib.util

import jax
import jax.numpy as jnp
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError


def _load(tmp_path, src):
    path = tmp_path / "mod.py"
    path.write_text("import jax.numpy as jnp\n" + src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.f


@pytest.mark.parametrize(
    "src, match",
    [
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 8, :].set(a[i:i + 8, :] + a.sum())
    return out
""",
            "may only be read through a tile slice",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        for j in range(0, 128, 128):
            out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "parallel loop `j` must tile",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 32, 8):
        out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "don't write all of `out`'s dim 0",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 16, :].set(a[i:i + 16, :])
    return out
""",
            "must equal the step",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        print(i)
        out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "may only contain",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        if True:
            pass
        out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "tile mask must be",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        acc = jnp.zeros((8, 128))
        for k in range(0, 128, 128):
            if acc[0, 0] > 0:
                continue
            acc += a[i:i + 8, k:k + 128]
        out = out.at[i:i + 8, :].set(acc)
    return out
""",
            "only depend on loop variables, constants and inputs, not `acc`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        if i > 1000:
            out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "skips every tile",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 8, :].set(out[i:i + 8, :])
    return out
""",
            "can't be read inside the loop nest",
        ),
        (  # Python carries `n` from one j to the next; the kernel recomputes it per tile
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        n = 0.0
        for j in range(0, 128, 128):
            n = n + 1.0
            out = out.at[i:i + 8, j:j + 128].set(a[i:i + 8, j:j + 128] * n)
    return out
""",
            r"line 8\): `n` is assigned between the parallel loops \(line 6\) and again further in",
        ),
        (  # `acc` would carry over from one tile to the next
            """
def f(a):
    out = jnp.zeros((64, 128))
    acc = jnp.zeros((8, 128))
    for i in range(0, 64, 8):
        for k in range(0, 128, 128):
            acc += a[i:i + 8, k:k + 128]
        out = out.at[i:i + 8, :].set(acc)
    return out
""",
            r"line 8\): `acc` is read before it is assigned in the tile",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        last = k
        acc = jnp.zeros((8, 128))
        for k in range(0, 128, 128):
            acc += a[i:i + 8, k:k + 128]
        out = out.at[i:i + 8, :].set(acc + last)
    return out
""",
            r"line 6\): loop variable `k` is read before its loop starts",
        ),
        (  # a triangular loop: its trip count would change from tile to tile
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        acc = jnp.zeros((8, 128))
        for k in range(0, i + 8, 8):
            acc += a[i:i + 8, :] * k
        out = out.at[i:i + 8, :].set(acc)
    return out
""",
            r"can't use `i`, which the loop nest assigns; .*use a mask",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        c = 1
        out = out.at[i:i + 8, :].set(a[i:i + 8, c * 128:c * 128 + 128])
    return out
""",
            "can't use `c`, which the loop nest assigns",
        ),
        (  # Python would read x[:, -128:0], an empty slice
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        acc = jnp.zeros((8, 128))
        for k in range(-128, 128, 128):
            acc += a[i:i + 8, k:k + 128]
        out = out.at[i:i + 8, :].set(acc)
    return out
""",
            "loop `k` starts at -128",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        w = 0.5 if i == 0 else 1.0
        out = out.at[i:i + 8, :].set(a[i:i + 8, :] * w)
    return out
""",
            r"line 6\): Python control flow .* can't depend on loop variables",
        ),
        (
            """
def f(a):
    M, N, K = a.shape
    out = jnp.zeros((M, N))
    for i in range(0, M, 8):
        out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            r"line 4\): the prologue failed \(ValueError",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        acc = jnp.zeros((8, 128))
        for k in range(0, 128, 128):
            b = a[i:i + 8, k:k + 128]
            acc += b @ jnp.ones((128, 3))
        out = out.at[i:i + 8, :].set(acc)
    return out
""",
            r"line 9\): couldn't trace the tile body",
        ),
        (
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 8, :].set(a[i:i + 8, :64])
    return out
""",
            r"value stored into `out` has shape \(8, 64\), which doesn't fit its tile \(8, 128\)",
        ),
    ],
)
def test_rejects_unsupported_code(tmp_path, src, match):
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(_load(tmp_path, src)).lower(rand((64, 128)))


def test_error_names_the_source_line(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 8, :].set(a[i:i + 8, :] + a.sum())
    return out
"""
    with pytest.raises(AutoKernelError, match=r"f \(line 6\)"):
        autokernel.tpu(_load(tmp_path, src)).lower(rand((64, 128)))


def test_rejects_tiles_that_break_tpu_layout():
    def bad(a, b):
        c = jnp.zeros((256, 192))
        for i in range(0, 256, 64):
            for j in range(0, 192, 32):
                acc = jnp.zeros((64, 32))
                for k in range(0, 384, 128):
                    acc += a[i : i + 64, k : k + 128] @ b[k : k + 128, j : j + 32]
                c = c.at[i : i + 64, j : j + 32].set(acc)
        return c

    a, b = rand((256, 384)), rand((384, 192), 1)
    with pytest.raises(AutoKernelError, match="multiple of 128"):
        autokernel.tpu(bad)(a, b)
    unchecked = autokernel.tpu(bad, check_tiling=False)
    if jax.default_backend() == "tpu":
        # Mosaic itself rejects the same tiles (at the DMA slice), so the pre-check matches real hardware.
        with pytest.raises(Exception, match=r"divisible by 8 and 128|aligned to tiles"):
            unchecked(a, b)
    else:
        # The interpreter doesn't enforce layout rules; the kernel is still semantically correct.
        unchecked.verify(a, b)


def test_bf16_sublane_rule():
    @autokernel.tpu
    def copy(x):
        out = jnp.zeros((64, 128), jnp.bfloat16)
        for i in range(0, 64, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :])
        return out

    with pytest.raises(AutoKernelError, match="multiple of 16 for bfloat16"):
        copy.lower(rand((64, 128), dtype=jnp.bfloat16))


def test_mask_on_reduction_loop_is_rejected():
    @autokernel.tpu
    def bad(x, y):
        out = jnp.zeros((256, 256))
        for i in range(0, 256, 128):
            for j in range(0, 256, 128):
                if j > i + k:  # noqa: F821 - deliberately reads the reduction loop's variable
                    continue
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += x[i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
                out = out.at[i : i + 128, j : j + 128].set(acc)
        return out

    with pytest.raises(AutoKernelError, match="can't depend on the reduction loop"):
        bad.lower(rand((256, 256)), rand((256, 256), 1))


def test_bad_options_fail_at_decoration_time():
    with pytest.raises(AutoKernelError, match="matmul_precision"):
        autokernel.tpu(matmul_precision="fastest")


def test_wrong_number_of_inputs():
    @autokernel.tpu
    def double(x):
        out = jnp.zeros((64, 128))
        for i in range(0, 64, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * 2)
        return out

    with pytest.raises(AutoKernelError, match="takes 1 arrays, got 2"):
        double.lower(rand((64, 128)), rand((64, 128)))


def test_requires_tpu_when_not_interpreting():
    if jax.default_backend() == "tpu":
        pytest.skip("running on a real TPU")

    @autokernel.tpu(interpret=False)
    def double(x):
        out = jnp.zeros((64, 128))
        for i in range(0, 64, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * 2)
        return out

    with pytest.raises(AutoKernelError, match="only targets TPU"):
        double(rand((64, 128)))
