"""Regressions for the frontend and analysis: Python scoping and ordering rules the kernel must follow (or
reject with a clear error), and runtime values typed the way Python types them."""

import ast
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, analysis, frontend

M = N = K = 256
T = 128


def _load(tmp_path, src, **globals_):
    """`f` from a module holding `src` (after `import jax.numpy as jnp` and `globals_`)."""
    path = tmp_path / "mod.py"
    header = "import jax.numpy as jnp\n" + "".join(f"{k} = {v!r}\n" for k, v in globals_.items())
    path.write_text(header + src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.f


def _kernel(tmp_path, src, **globals_):
    return autokernel.tpu(_load(tmp_path, src, M=M, N=N, K=K, T=T, **globals_))


# --------------------------------------------------------------------------- an output named like an input


def test_an_output_rebinding_its_parameter_is_rejected(tmp_path):
    src = """
def f(x):
    x = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            x = x.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] + 1)
    return x
"""
    with pytest.raises(AutoKernelError, match=r"output `x` rebinds parameter `x`.*rename the output"):
        _kernel(tmp_path, src)


def test_an_output_aliasing_an_input_still_keeps_its_values(tmp_path):
    src = """
def f(x):
    out = x
    for i in range(T, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * 2)
    return out
"""
    x = rand((M, N))
    expected = np.asarray(x).copy()
    expected[T:] *= 2
    np.testing.assert_allclose(_kernel(tmp_path, src)(x), expected)


# --------------------------------------------------------------------------- values only a masked loop assigns


@pytest.mark.parametrize("mask", ["k != i - T", "keep[i // T, k // T] == 0"])
def test_a_value_only_a_masked_reduction_assigns_is_rejected(tmp_path, mask):
    src = f"""
def f(x, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            for k in range(0, K, T):
                if {mask}:
                    continue
                t = x[i:i + T, k:k + T] + j
            out = out.at[i:i + T, j:j + T].set(t)
    return out
"""
    # The static mask skips every step of the tiles with i = 0; the runtime one may skip any tile's steps.
    match = r"\(line 15\): `t` is only assigned under the reduction mask \(line 12\), which (skips|may skip)"
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, src)(rand((M, K)), jnp.zeros((2, 2), jnp.int32))


def test_a_value_a_masked_reduction_updates_after_assigning_it_before_works(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            t = jnp.zeros((T, T))
            for k in range(0, K, T):
                if k != i - T:
                    continue
                t = x[i:i + T, k:k + T] + j
            out = out.at[i:i + T, j:j + T].set(t)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x = rand((M, K))
    expected = np.zeros((M, N), np.float32)
    expected[T:, :T] = np.asarray(x)[T:, :T]
    expected[T:, T:] = np.asarray(x)[T:, :T] + T
    np.testing.assert_allclose(kernel(x), expected)


# --------------------------------------------------------------------------- `@=`


def test_matmul_augassign_accumulates_like_matmul(tmp_path):
    src = """
def f(x, w):
    out = jnp.zeros((M, T), jnp.float32)
    for i in range(0, M, T):
        h = x[i:i + T, 0:T]
        for k in range(0, K, T):
            h @= w[:, :]
        out = out.at[i:i + T, :].set(h)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, w = rand((M, K), dtype=jnp.bfloat16), (rand((T, T), 1) * 0.05).astype(jnp.bfloat16)
    xn, wn = np.asarray(x, np.float64), np.asarray(w, np.float64)
    expected = np.concatenate([xn[i : i + T, :T] @ wn @ wn for i in range(0, M, T)])
    np.testing.assert_allclose(kernel(x, w), expected, atol=1e-5)
    np.testing.assert_allclose(kernel.reference(x, w), expected, atol=1e-5)


def test_matmul_augassign_outside_a_reduction_is_an_mxu_matmul(tmp_path):
    src = """
def f(x, w):
    out = jnp.zeros((M, T), jnp.float32)
    for i in range(0, M, T):
        h = x[i:i + T, 0:T]
        h @= w[:, :]
        out = out.at[i:i + T, :].set(h)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, w = rand((M, K), dtype=jnp.bfloat16), rand((T, T), 1, dtype=jnp.bfloat16)
    assert "h = _mxu_dot(h, " in kernel.source(x, w)
    expected = np.asarray(x, np.float64)[:, :T] @ np.asarray(w, np.float64)
    np.testing.assert_allclose(kernel(x, w), expected, atol=1e-4)  # accumulated in f32, not rounded to bf16


# --------------------------------------------------------------------------- comprehension and lambda scopes


@pytest.mark.parametrize(
    "value", ["acc * sum([k for k in range(3)])", "(lambda k: acc * k)(3.0)", "acc * (lambda k=k: 3.0)()"]
)
def test_a_comprehension_or_lambda_variable_named_like_a_finished_loop(tmp_path, value):
    src = f"""
def f(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set({value})
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, y = rand((M, K)), rand((K, N), 1)
    np.testing.assert_allclose(kernel(x, y), 3 * (np.asarray(x) @ np.asarray(y)), rtol=1e-4, atol=1e-3)


def test_reads_sees_a_name_read_outside_the_comprehension_binding_it():
    node = ast.parse("acc = t + sum([t for t in range(2)]) + (lambda u: u + v)(1)").body[0]
    assert frontend.reads(node) == {"t", "sum", "range", "v"}


@pytest.mark.parametrize(
    "stmt, match",
    [
        ("acc += t + sum([t for t in range(2)])", "a comprehension or lambda binds `t`, which the kernel keeps"),
        ("acc += (lambda x: x)(t)", "a comprehension or lambda binds `x`"),
        ("acc += (b := t * 2)", "`:=` isn't supported"),
    ],
)
def test_names_the_kernel_renames_cant_be_rebound_in_a_scope(tmp_path, stmt, match):
    src = f"""
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            t = x[i:i + T, 0:T] * 2
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                {stmt}
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    with pytest.raises(AutoKernelError, match=match):  # `t` is a state variable: known once specialised
        _kernel(tmp_path, src)(rand((M, K)))


def test_a_mask_comprehension_variable_named_like_an_input_is_no_input_read(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                if sum(x for x in (i, k)) != T:
                    continue
                acc += x[i:i + T, k:k + T]
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    assert not analysis.reads_inputs(kernel.nest, kernel.nest.reductions[0].guard)
    x = rand((M, K))
    expected = np.zeros((M, N), np.float32)
    expected[:T] = np.tile(np.asarray(x)[:T, T:], 2)
    expected[T:] = np.tile(np.asarray(x)[T:, :T], 2)
    np.testing.assert_allclose(kernel(x), expected)


# --------------------------------------------------------------------------- runtime values typed as in Python


def test_runtime_mask_loop_coordinates_are_python_ints(tmp_path):
    src = """
def f(x, y, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                if keep[i // T, k // T] + k < 200:
                    continue
                acc += x[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, y = rand((M, K)), rand((K, N), 1)
    keep = jnp.full((2, 2), 200, jnp.uint8)  # uint8 200 + 128 wraps to 72 in Python: step k = 128 is skipped
    expected = np.asarray(x)[:, :T] @ np.asarray(y)[:T, :]
    np.testing.assert_allclose(kernel(x, y, keep), expected, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(kernel.reference(x, y, keep), expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [jnp.int8(100), jnp.uint8(200), jnp.int16(20000)])
def test_narrow_integer_scalar_arithmetic_is_rejected(tmp_path, n):
    # `n * 2` wraps in Python's narrow type, but the TPU's scalar unit only computes in 32 bits, so the kernel
    # holds `n` as its 32-bit value and wouldn't wrap: rejected.
    src = """
def f(x, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * (n * 2))
    return out
"""
    kernel = _kernel(tmp_path, src)
    match = r"does `mul` on `n` \(an? u?int(8|16) runtime scalar\).*computes in 32 bits.*`n.astype\(jnp.int32\)`"
    with pytest.raises(AutoKernelError, match=match):
        kernel(rand((M, N)), n)


@pytest.mark.parametrize("n", [jnp.int8(1), jnp.uint8(1), jnp.int16(-1)])
def test_narrow_integer_scalars_index_and_mix_with_tiles(tmp_path, n):
    src = """
def f(x, w, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(w[n, i:i + T, j:j + T] * n + x[i:i + T, j:j + T])
    return out
"""
    kernel = _kernel(tmp_path, src)
    x, w = rand((M, N)), rand((2, M, N), 1)
    want = np.asarray(w)[int(n)] * np.float32(n) + np.asarray(x)
    np.testing.assert_allclose(kernel(x, w, n), want, rtol=1e-6)


def test_a_32_bit_integer_scalar_is_still_a_python_int():
    assert analysis.scalar_aval(jax.ShapeDtypeStruct((), jnp.int32)).weak_type


# --------------------------------------------------------------------------- the prologue and allocations


def test_an_allocation_reading_a_name_the_prologue_reassigns_is_rejected(tmp_path):
    src = """
def f(x):
    s = 1.0
    out = jnp.full((M, N), s)
    s = 2.0
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * s)
    return out
"""
    match = r"\(line 10\): `s` is reassigned after the output allocation `out = .*` \(line 9\)"
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, src)


@pytest.mark.parametrize("prologue", ["U = T2 * 2", "out = jnp.zeros((M, N), dt)"])
def test_the_prologue_reading_a_name_the_function_assigns_later_is_rejected(tmp_path, prologue):
    src = f"""
def f(x):
    {prologue}
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            T2 = 3
            dt = 4
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * T2 * dt)
    return out
"""
    with pytest.raises(AutoKernelError, match="Python would raise UnboundLocalError"):
        _kernel(tmp_path, src, T2=1, dt="float32")


# --------------------------------------------------------------------------- names the tile assigns


def test_the_zeros_check_of_an_accumulator_sees_tile_values(tmp_path):
    src = """
def f(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            init = 1.0
            acc = jnp.full((T, T), init)
            for k in range(0, T, T):
                acc += x[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src, init=0.0)  # a global of the same name, which is zero
    x, y = rand((M, T)), rand((T, N), 1)
    np.testing.assert_allclose(kernel(x, y), np.asarray(x) @ np.asarray(y) + 1, rtol=1e-4, atol=1e-3)


def test_a_runtime_index_reading_a_tile_value_is_rejected(tmp_path):
    src = """
def f(x, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            off = 1
            out = out.at[i:i + T, j:j + T].set(x[n + off, i:i + T, j:j + T])
    return out
"""
    kernel = _kernel(tmp_path, src, off=0)
    with pytest.raises(AutoKernelError, match="the index `n \\+ off` can't read `off`, which the loop nest assigns"):
        kernel(rand((4, M, N)), jnp.int32(1))


# --------------------------------------------------------------------------- error messages


def test_an_empty_closure_cell_names_the_variable():
    def make():
        @autokernel.tpu
        def g(x):
            out = jnp.zeros((M, N))
            for i in range(0, M, TT):
                out = out.at[i : i + TT, :].set(x[i : i + TT, :])
            return out

        with pytest.raises(AutoKernelError, match="closure variable `TT` has no value yet"):
            g(rand((M, N)))
        TT = T  # assigned after the call

    make()


def test_an_ellipsis_index_is_rejected(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(x[..., i:i + T, :])
    return out
"""
    with pytest.raises(AutoKernelError, match=r"`\.\.\.` isn't supported in a tile index"):
        _kernel(tmp_path, src)(rand((M, N)))


def test_a_user_global_named_xp_is_seen_by_a_runtime_mask(tmp_path):
    src = """
def f(x, keep):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if keep[i // T, j // T] + xp > 3 and i >= 0:
                out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T])
    return out
"""
    kernel = _kernel(tmp_path, src, xp=3)
    x, keep = rand((M, N)), jnp.array([[1, 0], [0, 1]], jnp.int32)
    expected = np.asarray(x) * np.kron(np.asarray(keep), np.ones((T, T)))
    np.testing.assert_allclose(kernel(x, keep), expected)
    np.testing.assert_allclose(jax.jit(kernel.reference)(x, keep), expected)


def test_a_loop_beyond_int32_is_rejected(tmp_path):
    src = """
def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for b in range(2**31, 2**31 + 2):
            out = out.at[i:i + T, :].set(x[i:i + T, :])
    return out
"""
    with pytest.raises(AutoKernelError, match="doesn't fit the kernel's int32"):
        _kernel(tmp_path, src)(rand((M, N)))
