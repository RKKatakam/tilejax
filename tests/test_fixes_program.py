"""Regression tests for programs: loop rolling, the reference, allocations, runtime scalars and caching."""

import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError

M = N = 256
T = 128
L = 3


def layer_inputs():
    return rand((M, N), 1), rand((L, M, N), 2)


def top_level(prog, *args):
    """The primitives of a program's top-level jaxpr (a rolled loop is one `scan`)."""
    return [e.primitive.name for e in jax.make_jaxpr(prog._program_fn())(*args).jaxpr.eqns]


# --------------------------------------------------------------------------- side effects in rolled loops


class Box:
    pass


def attr_counter(x, w):
    b = Box()
    b.n = 0
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[layer, i : i + T, j : j + T])
        x = h
        b.n += 1  # a fori_loop would run this once
    return x * b.n


def test_attribute_store_in_a_loop_unrolls_it():
    prog = autokernel.tpu(attr_counter)
    x, w = layer_inputs()
    want = np.asarray(x) * np.prod(np.asarray(w), axis=0) * L
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-4, atol=1e-4)
    assert any("`b.n` is assigned" in why for why in prog.unrolled)


def appender(x, w):
    outs = []
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[layer, i : i + T, j : j + T])
        outs.append(h)
    return jnp.stack(outs)


def test_list_append_in_a_loop_unrolls_it():
    prog = autokernel.tpu(appender)
    x, w = layer_inputs()
    np.testing.assert_allclose(prog(x, w), x[None] * w, rtol=1e-6)
    assert any("`outs.append(...)` on a Python list" in why for why in prog.unrolled)


def counter_needed_at_trace_time(x, w):
    n = 0
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[layer, i : i + T, j : j + T])
        x = h
        n = n + 1
    return jnp.zeros((n, 1)) + x[:n]  # `n` must still be a Python int


def test_carried_python_number_unrolls_the_loop():
    prog = autokernel.tpu(counter_needed_at_trace_time)
    x, w = layer_inputs()
    want = (np.asarray(x) * np.prod(np.asarray(w), axis=0))[:L]
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-4, atol=1e-4)
    assert any("`n`, a Python int" in why for why in prog.unrolled)


def test_print_in_a_loop_unrolls_it(capsys):
    def f(x, w):
        for layer in range(L):
            print("layer")
            h = jnp.zeros((M, N))
            for i in range(0, M, T):
                for j in range(0, N, T):
                    h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + w[layer, i : i + T, j : j + T])
            x = h
        return x

    prog = autokernel.tpu(f)
    x, w = layer_inputs()
    np.testing.assert_allclose(prog(x, w), x + w.sum(0), rtol=1e-5, atol=1e-5)
    assert capsys.readouterr().out.count("layer") == L
    assert any("`print(...)`" in why for why in prog.unrolled)


# --------------------------------------------------------------------------- closures and the reference


def stale_lambda(x, w):
    get = lambda: x  # noqa: E731
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + w[layer, i : i + T, j : j + T])
        x = h * 0.5 + get() * 0.5  # `get()` sees the current `x`
    return x


def test_closure_reading_a_carried_value_keeps_the_loop_unrolled():
    prog = autokernel.tpu(stale_lambda)
    x, w = layer_inputs()
    want = np.asarray(x)
    for layer in range(L):
        want = (want + np.asarray(w[layer])) * 0.5 + want * 0.5
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prog.reference(x, w), want, rtol=1e-5, atol=1e-5)
    assert any("a lambda defined outside the loop reads `x`" in why for why in prog.unrolled)


def rolled(x, w):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + w[layer, i : i + T, j : j + T])
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                y = y.at[i : i + T, j : j + T].set(h[i : i + T, j : j + T] * 0.5)
        x = y
    return x


def test_reference_runs_the_original_control_flow():
    prog = autokernel.tpu(rolled)
    x, w = layer_inputs()
    assert top_level(prog, x, w) == ["scan"]  # the program: a fori_loop around a fused kernel
    reference = ast.unparse(prog._reference_ast)
    assert "for layer in range(L)" in reference  # the reference: the loop as written, nests as Python loops
    assert "__autokernel_loop" not in reference and "__autokernel_groups" not in reference
    want = np.asarray(x)
    for layer in range(L):
        want = (want + np.asarray(w[layer])) * 0.5
    np.testing.assert_allclose(prog.reference(x, w), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jax.jit(prog.reference)(x, w), want, rtol=1e-5, atol=1e-5)
    assert prog.verify(x, w) < 1e-5


# --------------------------------------------------------------------------- what a loop carries


def first_bound_in_an_inner_loop(x, w):
    acc = jnp.zeros((M, N))
    for _ in range(2):
        for layer in range(L):  # `h` has no value before this loop on the first pass
            h = jnp.zeros((M, N))
            for i in range(0, M, T):
                for j in range(0, N, T):
                    h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[layer, i : i + T, j : j + T])
        acc = acc + h
    return acc


def test_value_first_bound_inside_a_nested_loop():
    prog = autokernel.tpu(first_bound_in_an_inner_loop)
    x, w = layer_inputs()
    want = 2 * np.asarray(x) * np.asarray(w[L - 1])
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prog.reference(x, w), want, rtol=1e-5, atol=1e-5)


def augmented(x, w):
    acc = jnp.zeros((M, N))
    y = jnp.zeros((M, N))
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[layer, i : i + T, j : j + T])
        acc += h  # reads `acc`: it is carried
        y = acc * 2
    return y


def test_augmented_assignment_is_a_carried_read():
    prog = autokernel.tpu(augmented)
    x, w = layer_inputs()
    want = 2 * (np.asarray(x)[None] * np.asarray(w)).sum(0)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and "scan" in top_level(prog, x, w)


def with_comprehension(x, w):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + w[layer, i : i + T, j : j + T])
        parts = [h * c for c in (0.5, 0.5)]  # `c` is the comprehension's own
        x = parts[0] + parts[1]
    return x


def test_comprehension_in_a_loop_body_still_rolls():
    prog = autokernel.tpu(with_comprehension)
    x, w = layer_inputs()
    np.testing.assert_allclose(prog(x, w), x + w.sum(0), rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and top_level(prog, x, w) == ["scan"]


# --------------------------------------------------------------------------- glue and allocations


def tuple_glue(x, w):
    tiles = (128, 128)
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    tm = tiles[0]  # indexing a Python tuple: not a view of an array
    out = jnp.zeros((M, N))
    for i in range(0, M, tm):
        for j in range(0, N, T):
            out = out.at[i : i + tm, j : j + T].set(h[i : i + tm, j : j + T] + w[i : i + tm, j : j + T])
    return out


def test_python_glue_between_nests_runs_as_python():
    prog = autokernel.tpu(tuple_glue)
    x, w = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, w), 2 * x + w, rtol=1e-6)


def computed_allocation(x, w):
    y = x * 1  # computed, not a constant fill: rows no tile writes keep it
    for i in range(0, M // 2, T):
        for j in range(0, N, T):
            y = y.at[i : i + T, j : j + T].set(w[i : i + T, j : j + T])
    return y * 1


def reshaped_allocation(x, w):
    y = x.reshape(2, M // 2, N)  # a view of `x`, updated in place
    for i in range(0, M // 2, T):
        for j in range(0, N, T):
            y = y.at[1, i : i + T, j : j + T].set(w[i : i + T, j : j + T])
    return y * 1


def test_partial_writes_keep_a_computed_allocation():
    x, w = rand((M, N), 1), rand((M, N), 2)
    want = np.asarray(x).copy()
    want[: M // 2] = np.asarray(w)[: M // 2]
    prog = autokernel.tpu(computed_allocation)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-6)
    assert prog.verify(x, w) < 1e-6
    want = np.asarray(x).reshape(2, M // 2, N).copy()
    want[1] = np.asarray(w)[: M // 2]
    prog = autokernel.tpu(reshaped_allocation)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-6)
    assert prog.verify(x, w) < 1e-6


def test_a_constant_fill_must_still_be_covered():
    def f(x):
        y = jnp.zeros((M, N))
        for i in range(0, M // 2, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :])
        return y * 1

    with pytest.raises(AutoKernelError, match="the tiles don't write all of `y`'s dim 0"):
        autokernel.tpu(f)(rand((M, N)))


def shape_bound(x, w):
    h = jnp.zeros((M, N))
    for i in range(0, h.shape[0], T):  # the output's shape, not its values
        for j in range(0, N, T):
            h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i : i + T, j : j + T].set(h[i : i + T, j : j + T] + w[i : i + T, j : j + T])
    return out


def test_reading_an_outputs_shape_keeps_the_nest_whole():
    prog = autokernel.tpu(shape_bound)
    x, w = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, w), 2 * x + w, rtol=1e-6)
    assert prog.unrolled == []
    lowered = prog.lower(x, w)
    assert [lw.plan.spec.parallel[0].var for _, lw in lowered[0][1].parts] == ["i", "i"]  # `i` tiles the kernel


# --------------------------------------------------------------------------- runtime scalars and constants


def squared(x, w, p):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * (p * p * 1.0))
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i : i + T, j : j + T].set(h[i : i + T, j : j + T] + w[i : i + T, j : j + T])
    return out


def test_python_int_arithmetic_outside_int32_is_compiled_in():
    prog = autokernel.tpu(squared)
    x, w = rand((M, N), 1), rand((M, N), 2)
    for p in (3, 4):  # `p * p` fits in int32: a runtime scalar, one kernel for both
        np.testing.assert_allclose(prog(x, w, p), x * p * p + w, rtol=1e-5, atol=1e-5)
    assert len(prog.kernels[0].kernels) == 1
    big = 100_000  # `p * p` is 1e10: in int32 it would wrap around
    np.testing.assert_allclose(prog(x, w, big), np.asarray(x) * 1e10 + np.asarray(w), rtol=1e-5)
    assert len(prog.kernels[0].kernels) == 2


SCALE = 2.0


def global_scale(x, w):
    h = jnp.zeros(x.shape)
    for i in range(0, x.shape[0], T):
        for j in range(0, N, T):
            h = h.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * SCALE)
    out = jnp.zeros(x.shape)
    for i in range(0, x.shape[0], T):
        for j in range(0, N, T):
            out = out.at[i : i + T, j : j + T].set(h[i : i + T, j : j + T] + w[i : i + T, j : j + T] * SCALE)
    return out + SCALE


def test_kernels_see_changed_globals_like_the_code_around_them():
    global SCALE
    prog = autokernel.tpu(global_scale)
    x, w = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, w), 2 * x + 2 * w + 2, rtol=1e-5, atol=1e-5)
    try:
        SCALE = 3.0
        x2, w2 = rand((2 * M, N), 1), rand((2 * M, N), 2)  # new shapes: the program is traced again
        np.testing.assert_allclose(prog(x2, w2), 3 * x2 + 3 * w2 + 3, rtol=1e-5, atol=1e-5)
    finally:
        SCALE = 2.0


def options(x, w, cfg=None, *, scale=2.0):
    s = cfg["s"] if cfg else 1.0
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * s * scale + w[i : i + T, j : j + T])
    return out + 0


def test_dict_static_arguments():
    prog = autokernel.tpu(options)
    x, w = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, w, {"s": 2.0}), x * 4 + w, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prog(x, w, {"s": 3.0}), x * 6 + w, rtol=1e-5, atol=1e-5)
    assert prog.verify(x, w, {"s": 2.0}) < 1e-5
    with pytest.raises(AutoKernelError, match=r"a static argument .* must be hashable"):
        prog(x, w, {"s": {1.0}})


# --------------------------------------------------------------------------- bookkeeping


def test_unroll_reasons_are_recorded_once():
    def f(x):
        for n in range(1, 3):
            y = jnp.zeros((M, n * T))  # a shape: `n` must be a Python int
            for i in range(0, M, T):
                for j in range(0, n * T, T):
                    y = y.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * n)
            x = jnp.concatenate([y, x[:, n * T :]], axis=1)
        return x

    prog = autokernel.tpu(f)
    x = rand((M, N))
    for _ in range(3):
        prog.lower(x)
    assert len(prog.unrolled) == 1 and prog.unrolled[0].startswith("line ")
