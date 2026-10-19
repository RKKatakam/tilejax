"""Regressions for the frontend and analysis, second round: narrow integer scalars held as 32-bit values,
x64, scopes the kernel renames, outputs named like parameters, masked assignments and allocations reading
earlier outputs."""

import importlib.util
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError

M = N = K = 256
T = 128
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(tmp_path, src, **globals_):
    """`f` from a module holding `src` (after `import jax`, `import jax.numpy as jnp` and `globals_`)."""
    path = tmp_path / "mod.py"
    header = "import jax\nimport jax.numpy as jnp\n" + "".join(f"{k} = {v!r}\n" for k, v in globals_.items())
    path.write_text(header + src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.f


def _kernel(tmp_path, src, **globals_):
    return autokernel.tpu(_load(tmp_path, src, M=M, N=N, K=K, T=T, **globals_))


def _elementwise(value):
    """A kernel storing `y = value` (of the tile `t = x[i:i + T, j:j + T]` and the scalar `n`) into every tile."""
    return f"""
def f(x, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            t = x[i:i + T, j:j + T]
            y = {value}
            out = out.at[i:i + T, j:j + T].set(y)
    return out
"""


# --------------------------------------------------------------------------- narrow scalars: exact operations


@pytest.mark.parametrize(
    "value, n, want",
    [
        ("jnp.where(n > 0, t, 0.0)", jnp.int8(-3), lambda t, n: np.zeros_like(t)),
        ("jnp.where(n > 0, t, 0.0)", jnp.int8(3), lambda t, n: t),
        ("jnp.where(n == 1, t, -t)", jnp.uint8(1), lambda t, n: t),
        ("jnp.where(n > 0, t, -t)", jnp.int16(-200), lambda t, n: -t),
        ("t + jnp.maximum(n, 0)", jnp.int8(-100), lambda t, n: t),
        ("t + jnp.minimum(n, 0)", jnp.int8(-100), lambda t, n: t - 100),
        ("t * (n & 3) + (n | 1) + (n ^ 2)", jnp.int8(-1), lambda t, n: 3 * t + -1 + -3),
        ("t * ~n", jnp.int8(5), lambda t, n: -6 * t),
        ("t * n + n / 4", jnp.uint16(600), lambda t, n: 600 * t + 150),
    ],
)
def test_narrow_scalar_operations_exact_on_32_bits_work(tmp_path, value, n, want):
    kernel = _kernel(tmp_path, _elementwise(value))
    x = rand((M, N))
    expected = want(np.asarray(x), n)
    np.testing.assert_allclose(kernel(x, n), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(kernel.reference(x, n), expected, rtol=1e-4, atol=1e-4)


def test_a_narrow_scalar_mixing_with_a_wider_integer_tile_works(tmp_path):
    src = """
def f(x, n):
    out = jnp.zeros((M, N), jnp.int16)
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] + n)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x = jnp.arange(M * N, dtype=jnp.int16).reshape(M, N)
    n = jnp.int8(-7)  # Python adds in int16, as the kernel does
    np.testing.assert_array_equal(kernel(x, n), np.asarray(x) - 7)


# --------------------------------------------------------------------------- narrow scalars: wrapping operations


@pytest.mark.parametrize(
    "value, op",
    [
        ("t * (n + 1)", "add"),
        ("t * (n - i)", "sub"),
        ("t * -n", "neg"),
        ("t * abs(n)", "abs"),
        ("t * (n // 2)", "div"),
        ("t * (n % 3)", "rem"),
        ("t * (n << 1)", "shift_left"),
        ("t * n**2", "integer_pow"),
        ("t * (jnp.maximum(n, 0) * 2)", "mul"),  # computed from `n` exactly, then arithmetic
        ("jnp.full((T, T), n) * 2 + t", "mul"),  # broadcast, then arithmetic in Python's int8
    ],
)
def test_narrow_scalar_arithmetic_that_can_wrap_is_rejected(tmp_path, value, op):
    kernel = _kernel(tmp_path, _elementwise(value))
    match = rf"\(line 13\): `y = .*` does `{op}` on `n` \(an int8 runtime scalar\).*computes in 32 bits"
    with pytest.raises(AutoKernelError, match=match):
        kernel(rand((M, N)), jnp.int8(100))


def test_bitwise_not_of_an_unsigned_narrow_scalar_is_rejected(tmp_path):
    kernel = _kernel(tmp_path, _elementwise("t * ~n"))  # ~uint8(5) is 250 in Python, -6 on 32 bits
    with pytest.raises(AutoKernelError, match=r"does `not` on `n` \(a uint8 runtime scalar\)"):
        kernel(rand((M, N)), jnp.uint8(5))


def test_a_narrow_scalar_promoting_an_integer_tile_is_rejected(tmp_path):
    src = """
def f(x, n):
    out = jnp.zeros((M, N), jnp.int32)
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] + n)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x = jnp.ones((M, N), jnp.int16)
    # Python adds a uint16 and an int16 in int32; the kernel would convert n's 32-bit value to int16.
    match = r"mixes `n` \(a uint16 runtime scalar\) .* with int16 values: Python computes that in int32"
    with pytest.raises(AutoKernelError, match=match):
        kernel(x, jnp.uint16(40000))


def test_narrow_scalar_arithmetic_on_a_later_step_is_rejected(tmp_path):
    # Python's `a` is a Python int on the first step and an int8 after it, so `a * 2` wraps from the second.
    src = """
def f(x, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            a = 0
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i:i + T, k:k + T] * (a * 2)
                a = jnp.maximum(a, n)
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    with pytest.raises(AutoKernelError, match=r"\(line 15\): `acc \+= .*` does `mul` on `n`"):
        kernel(rand((M, K)), jnp.int8(100))


def test_a_program_compares_a_traced_narrow_scalar(tmp_path):
    src = """
def f(x, flags):
    n = flags[0]
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            h = h.at[i:i + T, j:j + T].set(jnp.where(n > 0, x[i:i + T, j:j + T] * 2, x[i:i + T, j:j + T]))
    return h + 1
"""
    prog = _kernel(tmp_path, src)
    x = rand((M, N))
    for flags, scale in [(jnp.array([1, 0], jnp.int8), 2), (jnp.array([-1, 0], jnp.int8), 1)]:
        np.testing.assert_allclose(prog(x, flags), np.asarray(x) * scale + 1, rtol=1e-6)


# --------------------------------------------------------------------------- narrow scalars in runtime indices


def _indexed(index):
    return f"""
def f(w, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(w[{index}, i:i + T, j:j + T])
    return out
"""


@pytest.mark.parametrize("index, op", [("n + 1", "add"), ("n * 2", "mul"), ("n - 1", "sub")])
def test_a_runtime_index_wrapping_a_narrow_scalar_is_rejected(tmp_path, index, op):
    # n = int8(127): Python's `n + 1` is -128, which reads w[2]; on 32 bits it would be 128.
    kernel = _kernel(tmp_path, _indexed(index))
    match = rf"`w\[{index.replace('+', '[+]').replace('*', '[*]')}, .*\]`: the runtime index .* does `{op}` on `n`"
    with pytest.raises(AutoKernelError, match=match):
        kernel(rand((130, M, N)), jnp.int8(127))


@pytest.mark.parametrize("index, n, row", [("n", jnp.int8(-1), 2), ("jnp.maximum(n, 0)", jnp.int8(-5), 0)])
def test_a_runtime_index_exact_on_a_narrow_scalar_works(tmp_path, index, n, row):
    kernel = _kernel(tmp_path, _indexed(index))
    w = rand((3, M, N))
    np.testing.assert_allclose(kernel(w, n), np.asarray(w)[row])


# --------------------------------------------------------------------------- x64


_X64 = """
import jax, jax.numpy as jnp, numpy as np
import autokernel

assert jax.config.jax_enable_x64
M, N, T = 16, 128, 8


@autokernel.tpu(interpret=True)
def offsets(x, s):
    out = jnp.zeros((M, N), jnp.int32)
    for i in range(0, M, T):
        r = i * 2 + 1
        out = out.at[i:i + T, :].set(x[i:i + T, :] + jax.lax.broadcasted_iota(jnp.int32, (T, N), 0) + r + s)
    return out


x = jnp.ones((M, N), jnp.int32)
got = np.asarray(offsets(x, jnp.int32(5)))
rows = np.arange(M)
want = 1 + rows % T + (rows // T * T) * 2 + 1 + 5
np.testing.assert_array_equal(got, np.broadcast_to(want[:, None], (M, N)))
print("ok")
"""


def test_int_scalars_under_x64_are_planned(tmp_path):
    # Under x64 the loop variables' and scalars' arithmetic traces as int64 scalars: that isn't narrow.
    script = tmp_path / "x64.py"
    script.write_text(_X64)
    src = os.path.join(os.path.dirname(HERE), "src")
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    run = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert run.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------- comprehension and lambda scopes


@pytest.mark.parametrize(
    "body, want",
    [
        (["a = x[i:i + T, j:j + T]", "s = sum([a * c for c in range(3)])", "c = s + 1"], lambda x: 3 * x + 1),
        (["a = x[i:i + T, j:j + T]", "b = jax.tree.map(lambda a: a * 2, a)"], lambda x: 2 * x),
        (["b = x[i:i + T, j:j + T] * sum([out for out in (1.0, 2.0)])"], lambda x: 3 * x),
    ],
)
def test_a_scope_may_bind_a_temporary_or_output_name(tmp_path, body, want):
    lines = "\n".join(" " * 12 + s for s in body)
    result = body[-1].split(" = ")[0]
    src = f"""
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
{lines}
            out = out.at[i:i + T, j:j + T].set({result})
    return out
"""
    kernel = _kernel(tmp_path, src)
    x = rand((M, N))
    np.testing.assert_allclose(kernel(x), want(np.asarray(x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "update", ["acc = acc + jax.tree.map(lambda acc: acc * 2, p)", "acc = acc + sum([p * acc for acc in (1.0, 2.0)])"]
)
def test_a_scope_binding_a_state_variable_is_rejected(tmp_path, update):
    src = f"""
def f(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                p = x[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
                {update}
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    match = r"\(line 15\): a comprehension or lambda binds `acc`, which the kernel keeps in a scratch buffer"
    with pytest.raises(AutoKernelError, match=match):
        kernel(rand((M, K)), rand((K, N), 1))


# --------------------------------------------------------------------------- outputs named like parameters


@pytest.mark.parametrize("alloc", ["jnp.zeros((M, N))", "jnp.zeros_like(x)"])
def test_an_output_rebinding_a_parameter_the_nest_doesnt_read_works(tmp_path, alloc):
    src = f"""
def f(x, y):
    x = {alloc}
    for i in range(0, M, T):
        for j in range(0, N, T):
            x = x.at[i:i + T, j:j + T].set(y[i:i + T, j:j + T] * 2)
    return x
"""
    kernel = _kernel(tmp_path, src)
    x, y = rand((M, N)), rand((M, N), 1)
    np.testing.assert_allclose(kernel(x, y), 2 * np.asarray(y))


@pytest.mark.parametrize(
    "prologue, body, line, match",
    [
        ("", "t = x[i:i + T, j:j + T]", 13, r"output `x` rebinds parameter `x` \(line 9\), so `x` here is the output"),
        ("M2 = x.shape[0]", "t = y[i:i + T, j:j + T]", 10, r"the prologue reads output `x` after its allocation"),
    ],
)
def test_reading_an_output_that_rebinds_a_parameter_is_rejected(tmp_path, prologue, body, line, match):
    src = f"""
def f(x, y):
    x = jnp.zeros((M, N))
    {prologue}
    for i in range(0, M, T):
        for j in range(0, N, T):
            {body}
            x = x.at[i:i + T, j:j + T].set(t * 2)
    return x
"""
    with pytest.raises(AutoKernelError, match=rf"\(line {line}\): {match}"):
        _kernel(tmp_path, src)


def test_a_nest_reading_the_shape_of_an_output_rebinding_a_parameter_is_a_program(tmp_path):
    src = """
def f(x, y):
    x = jnp.zeros((2 * M, N))
    for i in range(0, x.shape[0], T):
        for j in range(0, N, T):
            x = x.at[i:i + T, j:j + T].set(y[i:i + T, j:j + T] * 2)
    return x
"""
    prog = _kernel(tmp_path, src)
    x, y = rand((M, N)), rand((2 * M, N), 1)
    np.testing.assert_allclose(prog(x, y), 2 * np.asarray(y))


# --------------------------------------------------------------------------- values only a masked loop assigns


def test_a_value_a_static_mask_assigns_in_every_tile_works(tmp_path):
    # `k = 0` runs in every tile, so `last` is always assigned: as in Python, it holds the last step's value.
    src = """
def f(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            for k in range(0, K, T):
                if k > i:
                    continue
                last = x[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set(last)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, y = rand((M, K)), rand((K, N), 1)
    xn, yn = np.asarray(x), np.asarray(y)
    expected = np.concatenate([xn[:T, :T] @ yn[:T], xn[T:, T:] @ yn[T:]])
    np.testing.assert_allclose(kernel(x, y), expected, rtol=1e-4, atol=1e-3)


def test_a_value_two_static_masks_assign_between_them_works(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            for k in range(0, K, T):
                if i > 0:
                    continue
                t = x[i:i + T, k:k + T]
            for k2 in range(0, K, T):
                if i == 0:
                    continue
                t = x[i:i + T, k2:k2 + T] * 2
            out = out.at[i:i + T, j:j + T].set(t)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x = rand((M, K))
    xn = np.asarray(x)
    expected = np.concatenate([np.tile(xn[:T, T:], 2), np.tile(xn[T:, T:] * 2, 2)])
    np.testing.assert_allclose(kernel(x), expected)


def test_a_static_mask_skipping_a_whole_tile_names_the_tile(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            for k in range(0, K, T):
                if k > i or j > 0:
                    continue
                t = x[i:i + T, k:k + T]
            out = out.at[i:i + T, j:j + T].set(t)
    return out
"""
    match = r"`t` is only assigned under the reduction mask \(line 13\), which skips every step of the tile at "
    match += r"i = 0, j = 128"
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, src)(rand((M, K)))


# --------------------------------------------------------------------------- allocations and earlier outputs


def test_an_allocation_may_read_an_earlier_output(tmp_path):
    src = """
def f(x, y):
    out = jnp.zeros((M, N))
    out2 = jnp.zeros(out.shape, jnp.bfloat16)
    out3 = jnp.full_like(out, 2.0)
    for i in range(0, M, T):
        for j in range(0, N, T):
            if i > j:
                continue
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T])
            out2 = out2.at[i:i + T, j:j + T].set(y[i:i + T, j:j + T].astype(jnp.bfloat16))
            out3 = out3.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] + 1)
    return out, out2, out3
"""
    kernel = _kernel(tmp_path, src)
    x, y = rand((M, N)), rand((M, N), 1)
    out, out2, out3 = kernel(x, y)
    upper = np.kron(np.triu(np.ones((2, 2))), np.ones((T, T))).astype(bool)
    assert out2.dtype == jnp.bfloat16
    np.testing.assert_allclose(out, np.where(upper, x, 0))
    np.testing.assert_allclose(out2.astype(jnp.float32), np.where(upper, y.astype(jnp.bfloat16).astype(np.float32), 0))
    np.testing.assert_allclose(out3, np.where(upper, np.asarray(x) + 1, 2.0))


def test_a_program_allocation_kept_where_no_tile_writes_may_read_an_earlier_output(tmp_path):
    # `g`'s allocation is passed to the kernel (aliased to `g`): it reads `h`'s allocation, as in Python.
    src = """
def f(x):
    h = x * 2
    g = h + 1
    for i in range(0, M, T):
        for j in range(T, N, T):
            h = h.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T])
            g = g.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T])
    return h * 1, g * 1
"""
    prog = _kernel(tmp_path, src)
    x = rand((M, N))
    xn, right = np.asarray(x), np.arange(N) >= T
    h, g = prog(x)
    np.testing.assert_allclose(h, np.where(right, xn, 2 * xn), rtol=1e-6)
    np.testing.assert_allclose(g, np.where(right, xn, 2 * xn + 1), rtol=1e-6)


def test_jit_of_a_kernel_with_an_allocation_reading_an_output(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    out2 = out + 3.0
    for i in range(0, M, T):
        for j in range(0, N, T):
            if j > 0:
                continue
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T])
            out2 = out2.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * 2)
    return out, out2
"""
    kernel = _kernel(tmp_path, src)
    x = rand((M, N))
    out, out2 = jax.jit(kernel)(x)
    left = np.arange(N) < T
    np.testing.assert_allclose(out, np.where(left, x, 0))
    np.testing.assert_allclose(out2, np.where(left, 2 * np.asarray(x), 3.0))


# --------------------------------------------------------------------------- `@=` in a program's nests


def test_matmul_augassign_in_a_program_nest_is_an_mxu_matmul(tmp_path):
    src = """
def f(x, w):
    x = x * 1
    h = jnp.zeros((M, T), jnp.float32)
    for i in range(0, M, T):
        a = x[i:i + T, 0:T]
        a @= w[:, :]
        h = h.at[i:i + T, :].set(a)
    return h + 0
"""
    prog = _kernel(tmp_path, src)
    x, w = rand((M, K), dtype=jnp.bfloat16), rand((T, T), 1, dtype=jnp.bfloat16)
    expected = np.asarray(x, np.float64)[:, :T] @ np.asarray(w, np.float64)
    np.testing.assert_allclose(prog(x, w), expected, atol=1e-4)  # accumulated in f32, not rounded to bf16
    np.testing.assert_allclose(prog.reference(x, w), expected, atol=1e-4)
