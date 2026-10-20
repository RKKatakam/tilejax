"""The generated kernels: every pipelining and decoding form runs right (in the TPU interpreter, with its
race detector) and lowers to Mosaic for a v5e; the source defines what the tile code reads."""

import functools
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, lower_for_v5e, make_kernel, rand, run_checking_races, source_function, spec

import autokernel
from autokernel import Config, KernelError, Pipeline
from autokernel.codegen import _literal


def check(kernel, *inputs, lower=True):
    """Run `kernel` checking races, verify it against its loop nest, and cross-lower it for a v5e."""
    out = run_checking_races(kernel, *inputs)
    kernel.verify(*inputs)
    if lower:
        specs = [spec(*x.shape, dtype=x.dtype) for x in inputs]
        assert "tpu_custom_call" in lower_for_v5e(kernel.lower(*inputs), *specs)
    return out


MASKED = """
def f(x, y):
    out = jnp.full((4, 512, 256), -jnp.inf)
    other = jnp.zeros((4, 512, 256), jnp.bfloat16)
    for b in range(4):
        for i in range(0, 512, 128):
            for j in range(0, 256, 128):
                if j > i:
                    continue
                acc = jnp.zeros((128, 128))
                for k in range(0, 384, 128):
                    if k > i + 128 or k < i - 128:
                        continue
                    acc += x[b, i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
                out = out.at[b, i : i + 128, j : j + 128].set(acc)
                other = other.at[b, i : i + 128, j : j + 128].set((acc * 2).astype(jnp.bfloat16))
    return out, other
"""


def test_tile_and_step_masks_with_an_outer_loop_and_two_outputs():
    kernel = make_kernel(MASKED)
    x, y = rand((4, 512, 384)), rand((384, 256), 1)
    out, other = check(kernel, x, y)
    lowered = kernel.lower(x, y)
    sch = lowered.schedule
    assert sch.repeats == 4 and sch.pattern_tiles == 7 and sch.pattern_fills == 1 and sch.fills == 4
    assert lowered.steps == 4 * sum(
        min(3, i // 128 + 2) - max(0, i // 128 - 1) for i in range(0, 512, 128) for j in range(0, 256, 128) if j <= i
    )
    assert float(out[0, 0, 128]) == -np.inf and float(other[0, 0, 128]) == 0.0  # a skipped tile keeps the fill
    src = lowered.source
    assert "jax.lax.div(s, " in src and "fill_blocks" in src and "STEP_CODE = np.array" in src


def test_a_step_mask_alone():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 512, 128):
                    if k % 256:
                        continue
                    acc += x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    x = rand((256, 512))
    out = check(kernel, x)
    np.testing.assert_allclose(f64(out), f64(x[:, :128] + x[:, 256:384]), rtol=1e-6)
    sch = kernel.lower(x).schedule
    assert sch.repeats == 2 and sch.pattern_steps == 2 and not sch.tile_tables  # one tile's pattern, twice


def test_a_tile_mask_on_the_only_loop():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((512, 128))
            for i in range(0, 512, 128):
                if i == 256:
                    continue
                out = out.at[i : i + 128, :].set(x[i : i + 128, :] * 2)
            return out
        """
    )
    x = rand((512, 128))
    out = check(kernel, x)
    np.testing.assert_allclose(f64(out[256:384]), 0)
    np.testing.assert_allclose(f64(out[:256]), 2 * f64(x[:256]))


def test_single_step_tiles_have_no_scratch():
    kernel = make_kernel(
        """
        def f(x, w, bias, r):
            out = jnp.zeros((256, 256))
            for i in range(0, 256, 128):
                for j in range(0, 256, 128):
                    for k in range(0, 256, 256):
                        h = x[i : i + 128, k : k + 256] @ w[k : k + 256, j : j + 128]
                    h = gelu(h + bias[0:1, :128])
                    out = out.at[i : i + 128, j : j + 128].set(h + r[i : i + 128, j : j + 128] + j)
            return out
        """,
        gelu=jax.nn.gelu,
    )
    inputs = rand((256, 256)), rand((256, 256), 1), rand((1, 128), 2), rand((256, 256), 3)
    check(kernel, *inputs)
    lowered = kernel.lower(*inputs)
    assert lowered.plan.single_step and not lowered.plan.carried
    assert "def step(" not in lowered.source and "bias_buf" in lowered.source and "_weak(j_blk * 128)" in lowered.source


@pytest.mark.cpu  # on a TPU, XLA may skip the reference's f32 -> bf16 -> f32 round trip (excess precision)
def test_per_tile_inputs_carried_casts_and_renaming_after_the_reduction():
    kernel = make_kernel(
        """
        def f(x, y, r):
            out = jnp.zeros((256, 256))
            for i in range(0, 256, 128):
                for j in range(0, 256, 128):
                    acc = 0.0
                    first = r[i : i + 128, j : j + 128]
                    t = 0.0
                    for k in range(0, 384, 128):
                        with jax.named_scope("dot"):
                            acc += x[i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
                        t = acc
                    acc = acc * 2
                    t = t + first
                    out = out.at[i : i + 128, j : j + 128].set((acc + t).astype(jnp.bfloat16))
            return out
        """
    )
    inputs = rand((256, 384), 0, jnp.bfloat16), rand((384, 256), 1, jnp.bfloat16), rand((256, 256), 2)
    check(kernel, *inputs)
    src = kernel.source(*inputs)
    assert "acc_ref[...] = jnp.broadcast_to(0.0, (128, 128))\n" in src
    assert "acc2 = acc_ref[...] * 2" in src and "t2 = t + first_ref[...]" in src
    assert "out_bufs[tile & 1] = (acc2 + t2).astype(jnp.bfloat16).astype(out_bufs.dtype)" in src
    assert "jax.lax.rem(s, 3)" in src and "with jax.named_scope('dot'):" in src


def test_stores_broadcast_and_one_block_loops_use_constant_windows():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((2, 128, 128), jnp.int32)
            for h in range(2):
                for i in range(0, 128, 128):
                    out = out.at[h, :, i : i + 128].set(1 + i + x[h, 8:16, :].sum().astype(jnp.int32))
            return out
        """
    )
    x = rand((2, 128, 128))
    check(kernel, x)
    src = kernel.source(x)
    assert "x_hbm.at[h_blk, pl.ds(8, 8), :]" in src and "out_hbm.at[h_blk, :, pl.ds(0, 128)]" in src
    assert "i = 0" in src and "jnp.broadcast_to(" in src


def test_offset_windows():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                for k in range(128, 384, 128):
                    t = x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(t)
            return out
        """
    )
    x = rand((256, 384))
    out = check(kernel, x)
    np.testing.assert_allclose(f64(out), f64(x[:, 256:]))
    assert "pl.ds(kb * 128 + 128, 128)" in kernel.source(x)


def test_three_output_buffers_and_four_input_buffers():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((1024, 128))
            for i in range(0, 1024, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    acc += x[i : i + 128, k : k + 128]
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    x = rand((1024, 256))
    lowered = kernel.lowered(kernel.signature(x), Config(None, Pipeline(4, 3)))
    np.testing.assert_allclose(f64(lowered.run(x)), f64(x[:, :128] + x[:, 128:]), rtol=1e-6)
    assert "tpu_custom_call" in lower_for_v5e(lowered, spec(1024, 256))
    src = lowered.source
    assert "jax.lax.rem(tile, 3)" in src and "(s & 3)" in src and "pltpu.VMEM((4, 128, 128)" in src


def test_trace_regions():
    kernel = make_kernel(MASKED)
    x, y = rand((4, 512, 384)), rand((384, 256), 1)
    signature = kernel.signature(x, y)
    traced = kernel.lowered(signature, kernel.lower(x, y).config, trace=True)
    for region in ("wait", "out_wait", "compute"):
        assert f"with jax.named_scope('f/{region}'):" in traced.source
    out = traced.run(x, y)
    np.testing.assert_allclose(f64(out[0]), f64(kernel(x, y)[0]))
    assert "tpu_custom_call" in lower_for_v5e(traced, spec(4, 512, 384), spec(384, 256))


def test_names_the_kernel_uses_itself_are_renamed():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                acc = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    s = x[i : i + 128, k : k + 128]
                    tile = s * at + step + f_kernel
                    acc += tile
                out = out.at[i : i + 128, :].set(acc)
            return out
        """,
        at=2.0,
        step=1.0,
        f_kernel=3.0,
    )
    x = rand((256, 256))
    check(kernel, x)
    src = kernel.source(x)
    assert "def f_kernel2(" in src and "def step2(s2):" in src and "tile2, ph, kb, first, last = step2(s2)" in src


def helper_scale(v):
    return v * HELPER_FACTOR


HELPER_FACTOR = 3.0
OFFSET = 1.0


def test_constants_the_tile_code_reads():
    def local_helper(v):
        return v + OFFSET

    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                v = x[i : i + 8, :] * SCALE + len(NAMES) + I8(1) + INF * 0 + num(COMPLEX.real)
                v = scale(local(v)) + os.sep.count("/") + exp(v).astype(DTYPE) * jnp.zeros((), TYPE)
                out = out.at[i : i + 8, :].set(v + sum(PAIR) + FLAGS[0])
            return out
        """,
        SCALE=2,
        NAMES=("a", None),
        I8=np.int8,
        INF=float("inf"),
        COMPLEX=1j,
        num=float,
        scale=helper_scale,
        local=local_helper,
        os=os,
        exp=jnp.exp,
        DTYPE=jnp.dtype("float32"),
        TYPE=jnp.float32,
        PAIR=[np.float32(1.5), 2],
        FLAGS=(True,),
    )
    x = rand((8, 128))
    check(kernel, x)
    src = kernel.source(x)
    for line in [
        "SCALE = 2",
        "NAMES = ('a', None)",
        "I8 = np.int8",
        "INF = float('inf')",
        "COMPLEX = 1j",
        "from test_codegen import helper_scale as scale",
        "import os",
        "exp = jnp.exp",
        "DTYPE = jnp.dtype('float32')",
        "TYPE = jnp.float32",
        "PAIR = [np.dtype('float32').type(1.5), 2]",
        "FLAGS = (True,)",
        "def local_helper(v):",
        "local = local_helper",
        "OFFSET = 1.0",
        "from builtins import float as num",
    ]:
        assert line in src, line


def test_functions_defined_in_main_are_copied():
    fn = source_function("def double(v):\n    return v * 2\n")
    fn.__module__ = "__main__"
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(double(x[i : i + 8, :]))
            return out
        """,
        double=fn,
    )
    check(kernel, rand((8, 128)))
    assert "def double(v):" in kernel.source(rand((8, 128)))


@pytest.mark.parametrize(
    ("expression", "value", "message"),
    [
        ("v * C", np.ones(128, np.float32), "which is an array: pass it to the kernel as an input"),
        ("v * C.scale", types.SimpleNamespace(scale=2.0), "which is SimpleNamespace"),
        ("C(v)", functools.partial(jnp.multiply, 2), "can't be imported or written into the kernel"),
        ("C(v)", lambda v: v, "can't be imported or written into the kernel"),
    ],
)
def test_values_the_kernel_cant_define(expression, value, message):
    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                v = x[i : i + 8, :]
                out = out.at[i : i + 8, :].set({expression})
            return out
        """,
        C=value,
    )
    with pytest.raises(KernelError, match=message):
        kernel.lower(spec(8, 128))


@pytest.mark.parametrize(("name", "message"), [("np", "uses `np` for numpy"), ("_weak", "defines itself")])
def test_names_the_generated_code_owns(name, message):
    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set({name}.exp(x[i : i + 8, :]))
            return out
        """,
        **{name: jnp},
    )
    with pytest.raises(KernelError, match=message):
        kernel.lower(spec(8, 128))


def test_decorated_and_conflicting_helpers():
    decorated = source_function("def same(fn):\n    return fn\n\n\n@same\ndef helper(v):\n    return v\n")
    decorated.__module__ = "__main__"
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(helper(x[i : i + 8, :]))
            return out
        """,
        helper=decorated,
    )
    with pytest.raises(KernelError, match="a decorated function"):
        kernel.lower(spec(8, 128))

    conflicting = source_function("def helper(v):\n    return v * K\n", K=2.0)
    conflicting.__module__ = "__main__"
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(helper(x[i : i + 8, :]) * K)
            return out
        """,
        helper=conflicting,
        K=3.0,
    )
    with pytest.raises(KernelError, match="two different values named `K`"):
        kernel.lower(spec(8, 128))


def test_literals():
    assert _literal(None) == "None" and _literal(b"x") == "b'x'" and _literal(float("nan")) == "float('nan')"
    assert _literal(jnp.bfloat16) == "jnp.bfloat16" and _literal(np.float16) == "np.float16"
    assert _literal(jax.lax.exp) == "jax.lax.exp" and _literal(np.dtype("int8")) == "jnp.dtype('int8')"
    assert _literal((1, (2.5, "a"))) == "(1, (2.5, 'a'))" and _literal([object()]) is None
    assert _literal(object()) is None


def test_kernels_without_a_tpu_raise_on_other_tpus(monkeypatch):
    from autokernel import api as kernel_module

    class Device:
        device_kind = "TPU v6 lite"

    monkeypatch.setattr(kernel_module.v5e, "on_tpu", lambda: True)
    monkeypatch.setattr(kernel_module.jax, "devices", lambda: [Device()])
    with pytest.raises(KernelError, match="targets TPU v5e"):
        kernel_module.interpret_mode()
    Device.device_kind = "TPU v5 lite"
    assert kernel_module.interpret_mode() is False


def test_jit_and_eager_agree():
    kernel = make_kernel(MASKED)
    x, y = rand((4, 512, 384)), rand((384, 256), 1)
    eager = kernel(x, y)
    jitted = jax.jit(kernel)(x, y)
    for a, b in zip(eager, jitted, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert isinstance(autokernel.kernel(source_function(MASKED)), autokernel.Kernel)


def test_inputs_no_loop_moves_and_no_streamed_inputs():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((256, 128))
            for i in range(0, 256, 128):
                acc = x[1, :, :]
                for k in range(0, 2):
                    acc = acc * 2
                acc = acc + 1
                acc = acc + 1
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    x = rand((2, 128, 128))
    out = check(kernel, x)
    np.testing.assert_allclose(f64(out[:128]), 4 * f64(x[1]) + 2, rtol=1e-6)
    src = kernel.source(x)
    assert "x_hbm.at[1, :, :]" in src and "fetch_ahead" not in src and "acc2 = acc2 + 1" in src


def test_functions_whose_module_cant_be_imported_are_copied():
    fn = source_function("def triple(v):\n    return v * 3\n")
    fn.__module__ = "no_such_module_anywhere"
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(triple(x[i : i + 8, :]))
            return out
        """,
        triple=fn,
    )
    check(kernel, rand((8, 128)))
    assert "def triple(v):" in kernel.source(rand((8, 128)))


def test_a_helper_may_share_a_constant_with_the_tile_code():
    shared = source_function("def helper(v):\n    return v * K\n", K=3.0)
    shared.__module__ = "__main__"
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(helper(x[i : i + 8, :]) * K)
            return out
        """,
        helper=shared,
        K=3.0,
    )
    check(kernel, rand((8, 128)))
    assert kernel.source(rand((8, 128))).count("K = 3.0") == 1


def test_review_regressions():
    # Names: a parameter named like a later window of another input, and parameters named like the
    # generated module's own names.
    kernel = make_kernel(
        """
        def f(x, x2, interpret, pl):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                v = x[i : i + 8, 0:128] + x[i : i + 8, 128:256] + x2[i : i + 8, :] + interpret[i : i + 8, :]
                out = out.at[i : i + 8, :].set(v + pl[i : i + 8, :])
            return out
        """
    )
    inputs = rand((8, 256)), rand((8, 128), 1), rand((8, 128), 2), rand((8, 128), 3)
    check(kernel, *inputs)
    assert [op.name for op in kernel.lower(*inputs).plan.operands] == ["x", "x3", "x2", "interpret", "pl"]
    assert "def f(x, x2, interpret2, pl2, *, interpret=False):" in kernel.source(*inputs)

    # Trace regions with nothing to wait for; `a @= b.T` on the MXU's transposed read, as the reference.
    kernel = make_kernel(
        """
        def f(w):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, 128):
                acc = w[0:128, :]
                acc @= w[0:128, :].T
                out = out.at[i : i + 128, :].set(acc)
            return out
        """
    )
    w = rand((128, 128)) / 8
    lowered = kernel.lowered(kernel.signature(w), Config(None, Pipeline()), trace=True)
    assert "f/wait" not in lowered.source and "_mxu_dot_nt(acc, w_ref[...])" in lowered.source
    np.testing.assert_allclose(f64(lowered.run(w)), f64(w) @ f64(w).T, rtol=1e-5, atol=1e-5)
    assert "tpu_custom_call" in lower_for_v5e(lowered, spec(128, 128))


def test_helpers_defaults_and_strong_constants():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                out = out.at[i : i + 8, :].set(scale(x[i : i + 8, :]) + jnp.minimum(BIG, 0))
            return out
        """,
        scale=source_function("def scale(v, s=FACTOR):\n    return v * s\n", FACTOR=2.0),
        BIG=np.float64("inf"),
    )
    kernel.fn.__globals__["scale"].__module__ = "__main__"
    x = rand((8, 128))
    check(kernel, x)
    src = kernel.source(x)
    assert "FACTOR = 2.0" in src and "BIG = np.dtype('float64').type(float('inf'))" in src


@pytest.mark.parametrize("name", ["pl", "jnp", "_weak"])
def test_tile_code_cant_bind_the_kernels_own_names(name):
    kernel = make_kernel(
        f"""
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                {name} = x[i : i + 8, :]
                out = out.at[i : i + 8, :].set({name})
            return out
        """
    )
    with pytest.raises(KernelError, match=f"uses the name `{name}` itself"):
        kernel.lower(spec(8, 128))


def test_a_number_starting_an_integer_carried_value_is_cast():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((8, 128))
            for i in range(0, 8, 8):
                acc = 0
                for k in range(0, 256, 128):
                    acc = acc + x[i : i + 8, k : k + 128]
                out = out.at[i : i + 8, :].set(acc)
            return out
        """
    )
    x = rand((8, 256))
    check(kernel, x)
    assert "acc_ref[...] = jnp.broadcast_to(0, (8, 128)).astype(acc_ref.dtype)" in kernel.source(x)


TWO_PASS = """
def f(x, w):
    M, D = x.shape
    out = jnp.zeros((M, 256))
    for i in range(0, M, 128):
        sum_sq = jnp.zeros((128, 128))
        for k in range(0, D, 128):
            xk = x[i : i + 128, k : k + 128]
            sum_sq += xk * xk
        scale = jax.lax.rsqrt(jnp.sum(sum_sq, axis=1, keepdims=True) / D + 1e-6)
        acc = jnp.zeros((128, 256))
        for k2 in range(0, D, 128):
            acc += (x[i : i + 128, k2 : k2 + 128] * scale) @ w[k2 : k2 + 128, :]
        out = out.at[i : i + 128, :].set(acc)
    return out
"""


def test_two_phases_one_after_another():
    kernel = make_kernel(TWO_PASS)
    x, w = rand((256, 384)), rand((384, 256), 1)
    out = check(kernel, x, w)
    xf = f64(x)
    want = xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-6) @ f64(w)
    np.testing.assert_allclose(f64(out), want, rtol=1e-4, atol=1e-4)
    lowered = kernel.lower(x, w)
    assert [p.range.var for p in lowered.plan.phases] == ["k", "k2"] and lowered.steps == 2 * 6
    assert {op.name: op.phase for op in lowered.plan.operands} == {"x": 0, "x2": 1, "w": 1}
    assert set(lowered.plan.carried) == {"sum_sq", "scale", "acc"}
    src = lowered.source
    assert "(within >= 3).astype(jnp.int32)" in src and "def _phase1():" in src and "def _after_phase0():" in src
    assert "x2_copy(blocks(ahead_tile), ahead_kb" in src  # the next phase's inputs are fetched ahead too


MASKED_PHASES = """
def f(x, w):
    out = jnp.zeros((256, 128))
    for i in range(0, 256, 128):
        a = jnp.zeros((128, 128))
        for k in range(0, 384, 128):
            if k > i:
                continue
            y = x[i : i + 128, k : k + 128]
            a += y
        y = y * 2
        a = a + y
        t = a * 0
        for c in range(0, 128, 128):
            t = t + w[c : c + 128, :]
        b = jnp.zeros((128, 128))
        for k2 in range(0, 256, 128):
            if k2 > i:
                continue
            b += x[i : i + 128, k2 : k2 + 128] @ w[0:128, :]
        b = b * 3
        out = out.at[i : i + 128, :].set(a + b + t)
    return out
"""


def test_masked_phases_and_a_loop_of_one_iteration():
    kernel = make_kernel(MASKED_PHASES)
    x, w = rand((256, 384)), rand((128, 128), 1) / 8
    out = check(kernel, x, w)
    xf, wf = f64(x), f64(w)
    for i in (0, 1):
        rows = xf[i * 128 : (i + 1) * 128]
        a = sum(rows[:, k * 128 : (k + 1) * 128] for k in range(i + 1))
        a = a + 2 * rows[:, i * 128 : (i + 1) * 128]
        b = 3 * sum(rows[:, k * 128 : (k + 1) * 128] @ wf for k in range(i + 1))
        np.testing.assert_allclose(f64(out[i * 128 : (i + 1) * 128]), a + b + (a * 0 + wf), rtol=1e-4, atol=1e-4)
    lowered = kernel.lower(x, w)
    sch = lowered.schedule
    assert [p.range.var for p in lowered.plan.phases] == ["k", "k2"] and sch.step_tables
    assert sch.steps == 2 + 4 and sch.phase_steps == (3, 3)
    assert [sch.step_at(s) for s in range(sch.steps)] == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    src = lowered.source
    assert "y2 = y * 2" in src and "a_ref[...] = a_ref[...] + y2" in src  # the phase's `y`, renamed after it
    assert "c = 0" in src and "w_buf" in src  # the loop of one iteration runs inline; its window is fetched once


def test_phases_back_to_back():
    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, 128):
                a = jnp.zeros((128, 128))
                for k in range(0, 256, 128):
                    a += x[i : i + 128, k : k + 128]
                for k2 in range(0, 256, 128):
                    a = jnp.maximum(a, x[i : i + 128, k2 : k2 + 128])
                out = out.at[i : i + 128, :].set(a)
            return out
        """
    )
    x = rand((128, 256))
    out = check(kernel, x)
    xf = f64(x)
    total = xf[:, :128] + xf[:, 128:]
    np.testing.assert_allclose(f64(out), np.maximum(np.maximum(total, xf[:, :128]), xf[:, 128:]), rtol=1e-6)
    assert "_after_phase0" not in kernel.source(x)


def test_unpacking_into_carried_values():
    def step(acc, total, tile):
        return acc + tile, total + tile.sum(axis=1, keepdims=True)

    kernel = make_kernel(
        """
        def f(x):
            out = jnp.zeros((128, 128))
            for i in range(0, 128, 128):
                acc, total = jnp.zeros((128, 128)), jnp.zeros((128, 1))
                for k in range(0, 384, 128):
                    acc, total = step(acc, total, x[i : i + 128, k : k + 128])
                    acc = acc * 0.5
                out = out.at[i : i + 128, :].set(acc + total)
            return out
        """,
        step=step,
    )
    x = rand((128, 384))
    out = check(kernel, x)
    acc, total = np.zeros((128, 128)), np.zeros((128, 1))
    for k in range(3):
        tile = f64(x[:, k * 128 : (k + 1) * 128])
        acc, total = (acc + tile) * 0.5, total + tile.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(f64(out), acc + total, rtol=1e-5, atol=1e-5)
    src = kernel.source(x)
    assert "acc, total = step(acc_ref[...], total_ref[...], x_ref[...])" in src
    assert "total_ref[...] = total" in src and "acc_ref[...] = acc * 0.5" in src
