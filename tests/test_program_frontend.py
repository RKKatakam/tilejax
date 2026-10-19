"""Which code of a program is a tile loop nest, and what a nest may read and write: outputs updated in place,
ordinary Python loops, names bound by a nest, global arrays, keyword arguments and per-nest verification."""

import ast
import importlib.util
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, run_checking_races

import autokernel
from autokernel import AutoKernelError, program

M, N, T, L = 64, 128, 8, 3


def _module(tmp_path, src, name="mod"):
    path = tmp_path / f"{name}.py"
    path.write_text("import jax\nimport jax.numpy as jnp\n" + textwrap.dedent(src))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop(name, None)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- outputs allocated elsewhere


def allocated_before_an_if(x, flag):
    out = jnp.zeros((M, N))
    if flag:  # a static argument
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * 2)
    return out + 1


def test_output_allocated_outside_the_nests_block():
    prog = autokernel.tpu(allocated_before_an_if)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x, True), x * 2 + 1)
    np.testing.assert_allclose(prog(x, False), jnp.ones((M, N)))
    assert prog.verify(x, True) == 0
    nest = prog.kernels[0]
    assert nest.region.existing == {"out"} and nest.inputs == ["out", "x"]
    # The tiles cover `out`: its current value isn't needed, so it isn't aliased.
    assert prog.lower(x, True)[0][1].plan.aliased == []


def test_nest_updating_an_argument_keeps_the_tiles_it_skips():
    def f(x, y):
        for i in range(0, M, T):  # updates the argument `y`: any array value, not only an allocation
            if i == 2 * T:  # a static tile mask: this tile keeps y's value
                continue
            y = y.at[i : i + T, :].set(x[i : i + T, :])
        return y * 2

    prog = autokernel.tpu(f)
    x, y = rand((M, N), 1), rand((M, N), 2)
    want = np.asarray(x).copy()
    want[2 * T : 3 * T] = np.asarray(y[2 * T : 3 * T])
    np.testing.assert_allclose(prog(x, y), want * 2)
    assert prog.verify(x, y) == 0
    assert prog.lower(x, y)[0][1].plan.aliased == ["y"]


def test_a_lone_nest_updating_an_argument_is_a_program():
    def f(x, y):
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] + 1)
        return y

    prog = autokernel.tpu(f)
    assert isinstance(prog, autokernel.TpuProgram)
    x, y = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, y), x + 1)


def test_allocation_used_before_its_nest_and_a_second_unallocated_output():
    def f(a, c):
        out = jnp.zeros((M, N))
        b = a + out  # `out` is used between its allocation and the nest: the nest starts from its value
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(b[i : i + T, :] * 2)
            c = c.at[i : i + T, :].set(b[i : i + T, :] + 1)  # `c` is an argument
        return out, c

    prog = autokernel.tpu(f)
    a, c = rand((M, N), 1), rand((M, N), 2)
    out, c2 = prog(a, c)
    np.testing.assert_allclose(out, a * 2)
    np.testing.assert_allclose(c2, a + 1)
    assert prog.verify(a, c) == 0
    assert prog.kernels[0].region.existing == {"out", "c"}


def preallocated_per_layer(x, w):
    outs = jnp.zeros((L, M, N))
    for layer in range(L):
        for i in range(0, M, T):  # writes one layer of `outs`: the others keep their values
            outs = outs.at[layer, i : i + T, :].set(x[i : i + T, :] * w[layer, :, :])
        x = x + 1
    return outs


def test_nest_in_a_loop_fills_an_array_allocated_before_the_loop():
    prog = autokernel.tpu(preallocated_per_layer)
    x, w = rand((M, N), 1), rand((L, 1, N), 2)
    want = jnp.stack([(x + layer) * w[layer] for layer in range(L)])
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-6)
    assert prog.verify(x, w) < 1e-5
    lowered = prog.lower(x, w)
    # The store's layer index is needed at trace time, so the loop is unrolled: a kernel per layer.
    assert len(lowered) == L and all(k.plan.aliased == ["outs"] for _, k in lowered)


def test_single_kernel_output_from_an_input_needs_not_cover_it():
    def f(v, x):
        out = v
        for i in range(0, M, T):
            out = out.at[1, i : i + T, :].set(x[i : i + T, :] * 2)
        return out

    kernel = autokernel.tpu(f)
    assert isinstance(kernel, autokernel.TpuKernel)
    v, x = rand((3, M, N), 1), rand((M, N), 2)
    out = run_checking_races(kernel, v, x)
    np.testing.assert_allclose(out, v.at[1].set(x * 2))
    assert kernel.verify(v, x) == 0


def test_partial_output_still_needs_a_known_allocation_to_be_covered():
    def f(x):
        out = jnp.zeros((2, M, N))
        for i in range(0, M, T):
            out = out.at[0, i : i + T, :].set(x[i : i + T, :])
        return out

    with pytest.raises(AutoKernelError, match="the tiles don't write all of `out`'s dim 0"):
        autokernel.tpu(f)(rand((M, N)))


@pytest.mark.parametrize("fuse", [False, True])
def test_consecutive_nests_writing_the_same_array(fuse):
    def f(x):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] + 1)
        for i in range(0, M, T):  # updates `out` in place: the tiles it skips keep the first nest's values
            if i >= M // 2:
                continue
            out = out.at[i : i + T, :].set(x[i : i + T, :] * 3)
        return out

    prog = autokernel.tpu(f, fuse=fuse)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), jnp.where(jnp.arange(M)[:, None] < M // 2, x * 3, x + 1))
    assert prog.verify(x) == 0


# --------------------------------------------------------------------------- ordinary Python loops


def _is_nest(src):
    loop = ast.parse(textwrap.dedent(src)).body[0]
    return bool(program._nest_stores(loop))


@pytest.mark.parametrize(
    "src, nest",
    [
        ("for i in range(0, M, T):\n    out = out.at[i:i + T, :].set(x[i:i + T, :])", True),
        ("for h in range(H):\n    for i in range(0, S, T):\n        o = o.at[h, i:i + T, :].set(x[h, i:i + T])", True),
        ("for i in range(2):\n    on = on.at[i, :].set(flags[i, :] * 2)", True),  # a tile of one row
        ("for i in range(0, M, T):\n    if i > 8:\n        continue\n    out = out.at[i:i + T].set(x[i:i + T])", True),
        # element and row updates: no slice in the store
        ("for t in range(1, n):\n    h = h.at[t].set(h[t - 1] * a + x[t])", False),
        ("for layer in range(L):\n    outs = outs.at[layer].set(h * layer)", False),
        # the store doesn't place tiles by the loop variable
        ("for t in range(n):\n    h = h.at[t + 1, :].set(x[t, :])", False),
        ("for i in range(0, M, T):\n    for j in range(0, N, T):\n        o = o.at[i:i + T].set(x[i:i + T])", False),
        ("for i in range(0, M, T):\n    o = o.at[0:T, :].set(x[i:i + T, :])", False),
        # reads what it stores: a scan over tiles
        ("for i in range(T, M, T):\n    c = c.at[i:i + T].set(c[i - T:i] + x[i:i + T])", False),
        # not a range loop, or stores not at the end
        ("for i in idx:\n    out = out.at[i:i + T].set(x[i:i + T])", False),
        ("for i in range(0, M, T):\n    out = out.at[i:i + T].set(x[i:i + T])\n    n += 1", False),
        # a Python loop around a nest that also stores
        (
            "for l in range(L):\n    h = jnp.zeros((M, N))\n    for i in range(0, M, T):\n"
            "        h = h.at[i:i + T, :].set(x[i:i + T, :])\n    outs = outs.at[l, :, :].set(h)",
            False,
        ),
    ],
)
def test_which_loops_are_nests(src, nest):
    assert _is_nest(src) == nest


def ordinary_loops(x):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):  # the one nest
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 3)
    c = jnp.zeros((M, N))
    for t in range(1, 4):  # a scan: plain JAX
        c = c.at[t].set(c[t - 1] + h[t])
    outs = jnp.zeros((2, M, N))
    for layer in range(2):  # collecting results: plain JAX
        outs = outs.at[layer].set(h * layer)
    for i in range(0, M, T):  # reads the array it updates: plain JAX
        c = c.at[i : i + T, :].set(c[i : i + T, :] * 2)
    return c, outs


def test_ordinary_loops_run_as_jax():
    prog = autokernel.tpu(ordinary_loops)
    assert [n.name for n in prog.kernels] == ["ordinary_loops_h"]
    x = rand((M, N))
    h = np.asarray(x) * 3
    c = np.zeros((M, N), np.float32)
    for t in range(1, 4):
        c[t] = c[t - 1] + h[t]
    c, outs = np.asarray(c) * 2, np.stack([h * 0, h * 1])
    got = prog(x)
    np.testing.assert_allclose(got[0], c, rtol=1e-6)
    np.testing.assert_allclose(got[1], outs, rtol=1e-6)
    assert prog.verify(x) < 1e-5


def test_python_loop_around_a_nest_collecting_its_results():
    def f(x):
        outs = jnp.zeros((L, M, N))
        for layer in range(L):
            h = jnp.zeros((M, N))
            for i in range(0, M, T):
                h = h.at[i : i + T, :].set(jnp.tanh(x[i : i + T, :]))
            x = x + h
            outs = outs.at[layer, :, :].set(x)
        return outs

    prog = autokernel.tpu(f)
    x = rand((M, N))
    want, cur = [], x
    for _ in range(L):
        cur = cur + jnp.tanh(cur)
        want.append(cur)
    np.testing.assert_allclose(prog(x), jnp.stack(want), rtol=1e-5, atol=1e-5)
    assert prog.verify(x) < 1e-5
    assert prog.unrolled == []  # a fori_loop: the collecting store is ordinary JAX inside it


# --------------------------------------------------------------------------- names a nest binds


@pytest.mark.parametrize(
    "src, match",
    [
        (  # a loop variable read after the nest: Python would see its last value
            """
            def f(x):
                i = 0
                out = jnp.zeros((64, 128))
                for i in range(0, 64, 8):
                    out = out.at[i:i + 8, :].set(x[i:i + 8, :])
                return out * i
            """,
            r"f \(line 9\): `i` is read here, but its value may come from the tile loop nest at line 7, where it is "
            "a loop variable",
        ),
        (  # read on the next iteration of a Python loop around the nest
            """
            def f(x):
                acc = 1.0
                for layer in range(2):
                    x = x * acc
                    out = jnp.zeros((64, 128))
                    for i in range(0, 64, 8):
                        acc = x[i:i + 8, :] * 2
                        out = out.at[i:i + 8, :].set(acc)
                    x = out
                return x
            """,
            r"f \(line 7\): `acc` is read here, .* nest at line 9, where it is a temporary",
        ),
        (  # read by another nest
            """
            def f(x):
                out = jnp.zeros((64, 128))
                for i in range(0, 64, 8):
                    s = x[i:i + 8, :] * 2
                    out = out.at[i:i + 8, :].set(s)
                y = jnp.zeros((64, 128))
                for i in range(0, 64, 8):
                    y = y.at[i:i + 8, :].set(out[i:i + 8, :] * s)
                return y
            """,
            r"f \(line 11\): `s` is read here",
        ),
        (  # read by a closure defined after the nest
            """
            def f(x):
                out = jnp.zeros((64, 128))
                for i in range(0, 64, 8):
                    s = x[i:i + 8, :] * 2
                    out = out.at[i:i + 8, :].set(s)
                g = lambda: s
                return out + g()
            """,
            r"f \(line 9\): `s` is read here",
        ),
    ],
)
def test_reading_a_nests_names_after_it_is_an_error(tmp_path, src, match):
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(_module(tmp_path, src).f)


def test_names_rebound_after_a_nest_are_fine():
    def f(x, flag):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            t = x[i : i + T, :] * 2
            out = out.at[i : i + T, :].set(t)
        y = jnp.zeros((M, N))
        for i in range(0, M, T):  # the same names in the next nest
            t = out[i : i + T, :] + 1
            y = y.at[i : i + T, :].set(t)
        if flag:
            t = 3
        else:
            t = 4
        for i in range(2):  # an ordinary loop rebinding `i`
            y = y + i
        g = [t for t in range(3)]  # a comprehension's own `t`
        return y * t + len(g)

    prog = autokernel.tpu(f)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x, True), ((x * 2 + 1) + 1) * 3 + 3, rtol=1e-6)
    assert prog.verify(x, False) == 0


# --------------------------------------------------------------------------- global arrays


GLOBALS = """
M, N, T = 64, 128, 8
G = jnp.ones((M, N))
SIZES = [2, 3]  # a list: a constant (it has no shape)

def f(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(x[i:i + T, :] * G[i:i + T, :] * SIZES[0])
    return out + 1
"""


def test_global_arrays_are_kernel_inputs(tmp_path):
    mod = _module(tmp_path, GLOBALS, "globals_mod")
    prog = autokernel.tpu(mod.f)
    x = rand((M, N))
    assert prog.kernels[0].globals == ["G", "SIZES"]  # in source order
    assert [op.param for op in prog.lower(x)[0][1].operands] == ["x", "G"]
    np.testing.assert_allclose(prog(x), x * 2 + 1, rtol=1e-6)
    # Each call reads the global's current value (not the one seen when the program was first traced).
    mod.G = jnp.full((M, N), 3.0)
    np.testing.assert_allclose(prog(x), x * 6 + 1, rtol=1e-6)
    assert prog.verify(x) < 1e-5


def test_closure_arrays_are_kernel_inputs():
    scale = rand((M, N), 7)

    def f(x):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * scale[i : i + T, :])
        return out - 1  # code after the nest: a program

    prog = autokernel.tpu(f)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), x * scale - 1, rtol=1e-6)
    assert "scale" in [op.param for op in prog.lower(x)[0][1].operands]
    scale = scale * 2  # the closure cell changes
    np.testing.assert_allclose(prog(x), x * f.__closure__[0].cell_contents - 1, rtol=1e-6)


# --------------------------------------------------------------------------- keyword arguments


def with_keywords(x, y=None, *, scale=2.0, relu=False):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * scale)
    if y is not None:
        out = out + y
    return jnp.maximum(out, 0) if relu else out


def test_keyword_arguments_and_defaults():
    prog = autokernel.tpu(with_keywords)
    x, y = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x), x * 2)
    np.testing.assert_allclose(prog(x, y=y, scale=0.5), x * 0.5 + y)
    np.testing.assert_allclose(prog(x, y, relu=True, scale=-1.0), jnp.maximum(y - x, 0))
    np.testing.assert_allclose(prog.reference(x, scale=3.0), x * 3)
    assert prog.verify(x, y=y, relu=True) == 0
    assert prog.lower(x, scale=3.0).names == ["with_keywords_out"]
    with pytest.raises(TypeError, match=r"with_keywords\(\): .*unexpected keyword argument 'bias'"):
        prog(x, bias=1)


# --------------------------------------------------------------------------- verifying nest by nest


def two_nests(x):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 2)
    h = h + 1
    y = jnp.zeros((M, N))
    for i in range(0, M, T):
        y = y.at[i : i + T, :].set(h[i : i + T, :] * 3)
    return y


def test_verify_nests_checks_each_nest_on_its_actual_inputs():
    prog = autokernel.tpu(two_nests)
    x = rand((M, N))
    assert prog.verify_nests(x) == {"two_nests_h": 0.0, "two_nests_y": 0.0}
    assert prog.verify(x, per_nest=True) == 0


def test_verify_names_the_nest_that_disagrees(monkeypatch):
    prog = autokernel.tpu(two_nests, fuse=False)
    x = rand((M, N))
    prog.lower(x)  # builds each nest's kernel
    (bad,) = prog.kernels[1].kernels.values()
    real = autokernel.TpuKernel.__call__
    monkeypatch.setattr(
        autokernel.TpuKernel, "__call__", lambda self, *a: real(self, *a) + (1.0 if self is bad else 0.0)
    )
    line = two_nests.__code__.co_firstlineno + 6
    match = rf"nest `two_nests_y` \(line {line}\) disagrees with its Python loops"
    with pytest.raises(AssertionError, match=match):
        prog.verify_nests(x)
    with pytest.raises(AssertionError, match=match):
        prog.verify(x, per_nest=True)
    with pytest.raises(AssertionError, match=r"(?s)two_nests: .*the first nest that disagrees: .*" + match):
        prog.verify(x)


def test_verify_nests_counts_the_runs_of_a_nest_in_a_loop(monkeypatch):
    def f(x, w):
        for layer in range(L):
            h = jnp.zeros((M, N))
            for i in range(0, M, T):
                h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, :, :])
            x = h
        return x

    prog = autokernel.tpu(f)
    x, w = rand((M, N), 1), rand((L, 1, N), 2)
    assert set(prog.verify_nests(x, w)) == {"f_h"}
    real, calls = autokernel.TpuKernel.__call__, []

    def call(self, *args):  # the kernel goes wrong on its second run
        calls.append(1)
        return real(self, *args) + (1.0 if len(calls) == 2 else 0.0)

    monkeypatch.setattr(autokernel.TpuKernel, "__call__", call)
    with pytest.raises(AssertionError, match=r"nest `f_h` \(line \d+\), run 2 of 3 disagrees"):
        prog.verify_nests(x, w)
