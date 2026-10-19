"""Regression tests for programs: which Python loops around nests roll into fori_loops (and which must not),
integer arithmetic of runtime scalars, and the global values a program reads."""

import heapq
import random
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel

M, N, T, L = 16, 128, 8, 3


def layer_inputs():
    return rand((M, N), 1), rand((L, M, N), 2) * 0.5


def rolled(prog, *args):
    """Does the program's top-level jaxpr run a loop (a rolled `for layer in range(L)` is a `scan`)?"""
    prims = [e.primitive.name for e in jax.make_jaxpr(prog._program_fn())(*args).jaxpr.eqns]
    return any(p in ("scan", "while") for p in prims)


def layers(x, w, step):
    """`x` after the layers, each `x = step(layer, x, x * w[layer])`, in NumPy."""
    x, w = np.asarray(x), np.asarray(w)
    for layer in range(L):
        x = step(layer, x, x * w[layer])
    return x


# --------------------------------------------------------------------------- method calls on what a chain reaches


def kv_cache(x, w, kv):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        kv = {"k": kv["k"].at[layer].set(y), "v": kv["v"].at[layer].set(x)}  # `.set` of an array, not of `kv`
        x = y
    return x, kv


def dict_and_tuple_methods(x, params):
    state = (x, x)
    for layer in range(L):
        wl = params["w"].astype(jnp.float32)  # `.astype` of the array `params["w"]`, not of the dict
        h = state[0].astype(jnp.float32)  # ... of `state[0]`, not of the tuple
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(h[i : i + T, :] * wl[layer, i : i + T, :])
        state = (y + params["b"].reshape(M, N), state[1])
    return state[0]


def test_methods_of_arrays_reached_through_containers_roll():
    x, w = layer_inputs()
    kv = {"k": jnp.zeros((L, M, N)), "v": jnp.zeros((L, M, N))}
    prog = autokernel.tpu(kv_cache)
    out, cache = prog(x, w, kv)
    assert prog.unrolled == [] and rolled(prog, x, w, kv)
    np.testing.assert_allclose(out, layers(x, w, lambda _, x, y: y), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cache["k"][L - 1], out, rtol=1e-6)
    params = {"w": w, "b": jnp.ones((M * N,))}
    prog = autokernel.tpu(dict_and_tuple_methods)
    np.testing.assert_allclose(prog(x, params), layers(x, w, lambda _, x, y: y + 1), rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and rolled(prog, x, params)


def appends_through_a_dict(x, w, bufs):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        bufs["outs"].append(y)  # changes the list the dict holds
        x = y
    return x


def test_a_method_of_a_list_reached_through_a_dict_unrolls():
    x, w = layer_inputs()
    bufs = {"outs": []}
    prog = autokernel.tpu(appends_through_a_dict)
    prog(x, w, bufs)
    assert len(bufs["outs"]) == L
    assert any("""`bufs['outs'].append(...)` on a Python list""" in why for why in prog.unrolled), prog.unrolled


# --------------------------------------------------------------------------- calls with hidden effects


def np_random_noise(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = y + np.random.normal(size=(M, N)).astype(np.float32)  # a new draw on every iteration
    return x


def python_random(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = y + random.random()
    return x


def test_module_level_random_numbers_unroll_the_loop():
    x, w = layer_inputs()
    prog = autokernel.tpu(np_random_noise)
    np.random.seed(0)
    got = prog(x, w)
    np.random.seed(0)
    want = prog.reference(x, w)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert any("`np.random.normal(...)`" in why for why in prog.unrolled), prog.unrolled
    prog = autokernel.tpu(python_random)
    random.seed(0)
    got = prog(x, w)
    random.seed(0)
    np.testing.assert_allclose(got, prog.reference(x, w), rtol=1e-5, atol=1e-5)
    assert any("`random.random(...)`" in why for why in prog.unrolled), prog.unrolled


def pushes_on_a_heap(x, w, heap):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        heapq.heappush(heap, layer)
        x = y
    return x


def aliased_append(x, w):
    outs = []
    add = outs.append
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        add(y)  # `outs.append(y)` by another name
        x = y
    return jnp.stack(outs)


def aliased_in_the_body(x, w):
    outs = []
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        push = outs.append
        push(y)
        x = y
    return jnp.stack(outs)


STORE = {"outs": []}


def store():
    return STORE["outs"]


def appends_to_a_call_result(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        store().append(y)
        x = y
    return x


def test_stateful_functions_aliases_and_call_results_unroll_the_loop():
    x, w = layer_inputs()
    heap = []
    prog = autokernel.tpu(pushes_on_a_heap, unroll=False)
    prog(x, w, heap)
    assert sorted(heap) == list(range(L))
    assert any("`heapq.heappush(...)`" in why for why in prog.unrolled), prog.unrolled
    for fn in (aliased_append, aliased_in_the_body):
        prog = autokernel.tpu(fn)
        out = prog(x, w)
        assert out.shape == (L, M, N)
        np.testing.assert_allclose(out[-1], layers(x, w, lambda _, x, y: y), rtol=1e-5, atol=1e-5)
        assert prog.unrolled and "a Python list" in prog.unrolled[0], prog.unrolled
    STORE["outs"].clear()
    prog = autokernel.tpu(appends_to_a_call_result)
    prog(x, w)
    assert len(STORE["outs"]) == L
    assert any("`store().append(...)` may change a Python object" in why for why in prog.unrolled)


# --------------------------------------------------------------------------- closures


def checkpointed(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = jax.checkpoint(lambda a: a * 2.0 + x)(y)  # a lambda defined in the loop, reading its values
    return x


def tree_mapped(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = jax.tree.map(lambda a: a + x, y)
    return x


def head_after(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = x + y

    def head(v):  # defined after the loop: it can't run while the loop does
        return v + x

    return head(x) + sum(t * 2 for t in (x, x))  # so is the generator, whose iterable is evaluated here


def test_functions_that_cant_see_stale_values_roll():
    x, w = layer_inputs()
    cases = [
        (checkpointed, lambda _, x, y: y * 2 + x),
        (tree_mapped, lambda _, x, y: y + x),
        (head_after, lambda _, x, y: x + y),
    ]
    for fn, step in cases:
        prog = autokernel.tpu(fn)
        want = layers(x, w, step)
        want = want * 6 if fn is head_after else want
        np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
        assert prog.unrolled == [] and rolled(prog, x, w), (fn.__name__, prog.unrolled)


def defined_later_in_an_outer_loop(x, w):
    get = None
    for _ in range(2):
        for layer in range(L):
            y = jnp.zeros((M, N))
            for i in range(0, M, T):
                y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
            x = y + (get() if get is not None else 0.0)  # on the second pass: the current `x`
        get = lambda: x  # noqa: E731
    return x


def test_function_defined_after_the_loop_in_a_loop_around_it_keeps_it_unrolled():
    x, w = layer_inputs()
    prog = autokernel.tpu(defined_later_in_an_outer_loop)
    want = np.asarray(x)
    for p in range(2):
        for layer in range(L):
            want = want * np.asarray(w[layer]) + (want if p else 0.0)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-4, atol=1e-4)
    assert any("a lambda defined outside the loop reads `x`" in why for why in prog.unrolled), prog.unrolled


def late_binding(x, w):
    fs = []
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        fs.append(lambda: y)  # each sees the `y` of the last iteration, as Python closures do
        x = y
    return sum(f() for f in fs)


def test_an_unrolled_loop_runs_as_written():
    x, w = layer_inputs()
    prog = autokernel.tpu(late_binding)
    want = L * layers(x, w, lambda _, x, y: y)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    assert any("`fs.append(...)` on a Python list" in why for why in prog.unrolled), prog.unrolled


# --------------------------------------------------------------------------- what a loop carries

USE_BIAS = True


def bias_in_one_branch(x, w):
    for layer in range(L):
        if USE_BIAS:
            b = x * 0.5  # bound in one branch only, and not read after the loop
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        if USE_BIAS:
            y = y + b
        x = y
    return x


def bound_by_an_inner_python_loop(x, w):
    for layer in range(L):
        for k in range(2):  # a plain Python loop: `g` has no value before it
            g = x * (k + 1)
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(g[i : i + T, :] * w[layer, i : i + T, :])
        x = y
    return x


def read_before_bound(x, w):
    for layer in range(L):
        if USE_BIAS:
            x = x + prev  # noqa: F821 (no value on the first iteration: Python raises)
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        prev = y  # noqa: F841 (read on the next iteration)
        x = y
    return x


def test_names_bound_only_inside_the_loop_dont_unroll_it():
    x, w = layer_inputs()
    prog = autokernel.tpu(bias_in_one_branch)
    np.testing.assert_allclose(prog(x, w), layers(x, w, lambda _, x, y: y + x * 0.5), rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and rolled(prog, x, w)
    prog = autokernel.tpu(bound_by_an_inner_python_loop)
    np.testing.assert_allclose(prog(x, w), layers(x, w, lambda _, x, y: y * 2), rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and rolled(prog, x, w)
    with pytest.raises(UnboundLocalError):  # as in Python
        autokernel.tpu(read_before_bound)(x, w)


def float_schedule(x, w):
    scale = 1.0
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :] * scale)
        x = y
        scale = scale * 0.5
    return x


def float_read_after(x, w):
    scale = 1.0
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :] * scale)
        x = y
        scale = scale * 0.5
    return x[: int(scale * 64)]  # `scale` must still be a Python number here


def test_carried_python_float_rolls_unless_read_after_the_loop():
    x, w = layer_inputs()
    prog = autokernel.tpu(float_schedule)
    want = layers(x, w, lambda layer, x, y: y * 0.5**layer)
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    assert prog.unrolled == [] and rolled(prog, x, w)
    assert len(prog.lower(x, w)) == 1  # the scale is a runtime scalar: one kernel for every layer
    assert prog.verify(x, w) < 1e-5
    prog = autokernel.tpu(float_read_after)
    np.testing.assert_allclose(prog(x, w), want[: int(0.5**L * 64)], rtol=1e-5, atol=1e-5)
    assert any("`scale`, a Python float" in why for why in prog.unrolled), prog.unrolled


# --------------------------------------------------------------------------- integer arithmetic with loop variables


def shifted_mask(x, w, pos):
    y = jnp.zeros((M, N))
    for i in range(0, M, T):
        q = i + pos + jax.lax.broadcasted_iota(jnp.int32, (T, N), 0)  # `i + pos`: `i` is at most M - T
        y = y.at[i : i + T, :].set(jnp.where(q >= 4, x[i : i + T, :], 0.0))
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(y[i : i + T, :] + w[i : i + T, :])
    return out


def test_int_arithmetic_with_loop_variables_stays_a_runtime_scalar():
    x, w = rand((M, N), 1), rand((M, N), 2)
    prog = autokernel.tpu(shifted_mask)
    rows = np.arange(M)[:, None]
    for pos in range(4):
        want = np.where(rows + pos >= 4, np.asarray(x), 0) + np.asarray(w)
        np.testing.assert_allclose(prog(x, w, pos), want, rtol=1e-6)
    assert len(prog.kernels[0].kernels) == 1
    big = int(np.iinfo(np.int32).max) - 4  # `i + pos` leaves int32 for the second tile: compiled in
    prog.lower(x, w, big)
    assert len(prog.kernels[0].kernels) == 2


def layer_offsets(x, w):
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            q = i + layer + jax.lax.broadcasted_iota(jnp.int32, (T, N), 0)
            y = y.at[i : i + T, :].set(jnp.where(q >= 3, x[i : i + T, :] * w[layer, i : i + T, :], 0.0))
        x = y
    return x


def test_unrolled_layers_and_verify_share_one_kernel():
    x, w = layer_inputs()
    rows = np.arange(M)[:, None]
    want = layers(x, w, lambda layer, x, y: np.where(rows + layer >= 3, y, 0))
    prog = autokernel.tpu(layer_offsets, unroll=True)  # `layer` is a Python int: a runtime scalar all the same
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    assert prog.lower(x, w).names == ["layer_offsets_y"]
    prog = autokernel.tpu(layer_offsets)
    assert prog.verify(x, w) < 1e-5  # the reference runs the Python loop: one kernel for every layer too
    assert len(prog.kernels[0].kernels) == 1


# --------------------------------------------------------------------------- globals: read once per signature

GAIN = 2.0


def gained(x, w):
    y = jnp.zeros(x.shape)
    for i in range(0, x.shape[0], T):
        y = y.at[i : i + T, :].set(x[i : i + T, :] * GAIN)
    out = jnp.zeros(x.shape)
    for i in range(0, x.shape[0], T):
        out = out.at[i : i + T, :].set(y[i : i + T, :] + w[i : i + T, :])
    return out + GAIN


def test_a_changed_global_is_read_once_per_signature():
    global GAIN
    x, w = rand((M, N), 1), rand((M, N), 2)
    prog = autokernel.tpu(gained)
    np.testing.assert_allclose(prog(x, w), 2 * x + w + 2, rtol=1e-5, atol=1e-5)
    try:
        GAIN = 3.0
        # The same signature: the program, its reference and verify all still see 2, as under jax.jit.
        np.testing.assert_allclose(prog(x, w), 2 * x + w + 2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(prog.reference(x, w), 2 * x + w + 2, rtol=1e-5, atol=1e-5)
        assert prog.verify(x, w) < 1e-5
        assert prog.verify_nests(x, w)
        # A new signature reads the new value, everywhere.
        x2, w2 = rand((2 * M, N), 1), rand((2 * M, N), 2)
        np.testing.assert_allclose(prog(x2, w2), 3 * x2 + w2 + 3, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(prog.reference(x2, w2), 3 * x2 + w2 + 3, rtol=1e-5, atol=1e-5)
        assert prog.verify(x2, w2) < 1e-5
    finally:
        GAIN = 2.0


def test_a_changed_closure_value_is_read_once_per_signature():
    gain = 2.0

    def f(x, w):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * gain)
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(y[i : i + T, :] + w[i : i + T, :])
        return out

    x, w = rand((M, N), 1), rand((M, N), 2)
    prog = autokernel.tpu(f)
    np.testing.assert_allclose(prog(x, w), 2 * x + w, rtol=1e-5, atol=1e-5)
    gain = 3.0  # read by `f`
    np.testing.assert_allclose(prog.reference(x, w), 2 * x + w, rtol=1e-5, atol=1e-5)
    assert prog.verify(x, w) < 1e-5


# --------------------------------------------------------------------------- bookkeeping


class Box:
    pass


def attribute_store(x, w):
    b = Box()
    b.n = 0
    for layer in range(L):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = y
        b.n += 1
    return x * b.n


def test_unroll_reasons_name_one_loop_line():
    x, w = layer_inputs()
    prog = autokernel.tpu(attribute_store)
    prog(x, w)
    (why,) = prog.unrolled
    assert re.match(r"line \d+: at line \d+, `b.n` is assigned", why), why
