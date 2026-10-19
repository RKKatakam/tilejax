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
            "doesn't write all of output dim 0",
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
        if a[0, 0] > 0:
            out = out.at[i:i + 8, :].set(a[i:i + 8, :])
    return out
""",
            "only depend on loop variables",
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


def test_rejects_tiles_that_break_tpu_layout(pipeline):
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
        autokernel.tpu(bad, pipeline=pipeline)(a, b)
    unchecked = autokernel.tpu(bad, check_tiling=False, pipeline=pipeline)
    if jax.default_backend() == "tpu":
        # Mosaic itself rejects the same tiles (grid: Pallas's BlockSpec check; manual: the DMA slice),
        # so the pre-check matches real hardware.
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
