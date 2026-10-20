"""Corner cases of the frontend and analysis: malformed loop nests, unusual index and mask spellings,
views, narrow integer scalars, and the package's metadata fallback."""

import ast
import importlib
import importlib.metadata
import importlib.util
import pathlib
import tomllib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, analysis, frontend, scopes
from autokernel import views as views_mod

M = N = K = 256
T = 128


def _load(tmp_path, src, name="f", **globals_):
    """`name` from a module holding `src` (after `import jax`, `import jax.numpy as jnp` and `globals_`)."""
    path = tmp_path / "mod.py"
    header = "import jax\nimport jax.numpy as jnp\n" + "".join(f"{k} = {v!r}\n" for k, v in globals_.items())
    path.write_text(header + src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _fn(tmp_path, src, **globals_):
    return _load(tmp_path, src, M=M, N=N, K=K, T=T, **globals_)


def _kernel(tmp_path, src, **globals_):
    return autokernel.tpu(_fn(tmp_path, src, **globals_))


# --------------------------------------------------------------------------- the frontend's parse errors


@pytest.mark.parametrize(
    "src, match",
    [
        (
            """
def f(a, *rest):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"f \(line \d+\): only plain positional array parameters",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            "expected one top-level tile loop nest before the return",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
""",
            r"must end with `return <output>`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out + 1
""",
            "return the output arrays by name",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out, out
""",
            "each output may be returned only once",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    print(out.shape)
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            "only simple constant assignments / asserts may precede the loop nest",
        ),
        (
            """
def f(a):
    a = 1
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            "the prologue may not rebind input `a`",
        ),
        (
            """
def f(a):
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"missing output allocation `out = jnp.zeros\(...\)`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    else:
        pass
    return out
""",
            r"tile loops must be `for v in range\(start, stop, step\)`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in [0, 128]:
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"tile loops must be `for v in range\(start, stop, step\)`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        t = a[i:i + T, :]
    return out
""",
            "the innermost parallel loop must end with the output stores",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        if i > 0:
            continue
        out = out.at[i:i + T, :].set(a[i:i + T, :])
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"must end with exactly one `out = out.at\[...\].set\(...\)`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        acc = jnp.zeros((T, N))
        for k in range(0, K, T):
            acc += a[i:i + T, :]
            for k2 in range(0, K, T):
                acc += a[i:i + T, :]
        out = out.at[i:i + T, :].set(acc)
    return out
""",
            "reduction loops may only nest perfectly",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        acc = jnp.zeros((T, N))
        for i in range(0, K, T):
            acc += a[i:i + T, :]
        out = out.at[i:i + T, :].set(acc)
    return out
""",
            "loop variable `i` is used by two loops",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = a[i:i + T, :]
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"`out` may only be written by its final `.at\[...\].set\(...\)`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        a = 2.0
        out = out.at[i:i + T, :].set(a)
    return out
""",
            "can't assign to input / loop variable `a` inside the loop nest",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        i = 2
        out = out.at[0:T, :].set(a[0:T, :])
    return out
""",
            "can't assign to input / loop variable `i` inside the loop nest",
        ),
    ],
)
def test_malformed_nests_are_rejected_by_the_parser(tmp_path, src, match):
    with pytest.raises(AutoKernelError, match=match):
        frontend.parse(_fn(tmp_path, src))


def test_an_empty_masked_body_is_rejected(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        acc = jnp.zeros((T, N))
        for k in range(0, K, T):
            if k > i:
                continue
        out = out.at[i:i + T, :].set(acc)
    return out
"""
    with pytest.raises(AutoKernelError, match="the body under the reduction mask is empty"):
        frontend.parse(_fn(tmp_path, src))


def test_a_function_without_source_is_rejected():
    namespace = {}
    exec("def f(a):\n    return a\n", namespace)  # no file to read the source from
    with pytest.raises(AutoKernelError, match="can't read the source of"):
        autokernel.tpu(namespace["f"])


def test_a_lambda_is_rejected(tmp_path):
    f = _load(tmp_path, "f = lambda a: a\n")
    with pytest.raises(AutoKernelError, match="<lambda>: expected a plain `def` function"):
        autokernel.tpu(f)


def test_an_error_without_a_node_names_only_the_function(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
"""
    nest = frontend.parse(_fn(tmp_path, src))
    assert str(nest.error(None, "boom")) == "f: boom"
    assert str(nest.error(nest.parallel[0].node, "boom")).startswith("f (line ")


def test_a_comprehension_with_several_clauses_in_a_tile_body(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        t = a[i:i + T, :]
        y = sum([t * u * v for u in (1, 2) for v in (1, 3) if u + v > 2])
        out = out.at[i:i + T, :].set(y)
    return out
"""
    a = rand((M, N))
    # (u, v) in (1, 3), (2, 1), (2, 3): 3 + 2 + 6 = 11
    np.testing.assert_allclose(_kernel(tmp_path, src)(a), 11 * np.asarray(a), rtol=1e-5)


# --------------------------------------------------------------------------- analysis: loops, indices, outputs


@pytest.mark.parametrize(
    "src, match",
    [
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, UNDEFINED):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"can't evaluate .*`UNDEFINED` at trace time \(NameError",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, 128.0):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"`128.0` must be a Python int, got 128.0",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :, 0])
    return out
""",
            r"`a\[i:i \+ T, :, 0\]`: 3 indices for a rank-2 array",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, 300][:, None])
    return out
""",
            "index 300 in dim 1 is outside the array",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, ::2])
    return out
""",
            "dim 1 must be a tile slice, not a strided one",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i * 2, :])
    return out
""",
            r"dim 0 must look like `i:i \+ T`",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, 3:131])
    return out
""",
            "constant slice 3:131 in dim 1 must be a whole aligned block",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        acc = jnp.zeros((T, N))
        for k in range(0, K, T):
            acc += a[k:k + T, :]
        out = out.at[i:i + T, :].set(acc + a[k:k + T, :])
    return out
""",
            r"`a\[k:k \+ T, :\]`: can't be tiled by loop `k` here",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N), UNDEFINED)
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            r"couldn't evaluate the output allocation \(NameError",
        ),
        (
            """
def f(a):
    out = [jnp.zeros((M, N))]
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
""",
            "an output allocation must produce a single array",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :] + jnp.zeros((T, 3)))
    return out
""",
            "couldn't trace the tile body",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(jnp.zeros((T, 3)))
    return out
""",
            r"the value stored into `out` has shape \(128, 3\), which doesn't fit its tile \(128, 256\)",
        ),
        (
            """
def f(a):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        acc = jnp.zeros((T, N))
        for k in range(0, K, T):
            acc = acc[:, :128] + a[i:i + T, k:k + T]
        out = out.at[i:i + T, :].set(acc)
    return out
""",
            r"`acc` changes shape \(\(128, 256\) -> \(128, 128\)\) between steps",
        ),
    ],
)
def test_analysis_rejects(tmp_path, src, match):
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, src).lower(rand((M, N)))


def test_a_tile_size_may_be_written_before_the_loop_variable(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:T + i, :].set(a[i:T + i, :] * 2)
    return out
"""
    a = rand((M, N))
    np.testing.assert_allclose(_kernel(tmp_path, src)(a), 2 * np.asarray(a), rtol=1e-6)


def test_a_stored_row_is_broadcast_to_the_tile(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :].sum(0, keepdims=True))
    return out
"""
    a = rand((M, N))
    want = np.repeat(np.asarray(a).reshape(2, T, N).sum(1), T, axis=0)
    np.testing.assert_allclose(_kernel(tmp_path, src)(a), want, rtol=1e-4, atol=1e-4)


def test_an_empty_allocation_is_a_constant_fill(tmp_path):
    src = """
def f(a):
    out = jnp.empty((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :] + 1)
    return out
"""
    kernel = _kernel(tmp_path, src)
    a = rand((M, N))
    assert kernel.lower(a).plan.spec.outputs[0].fill == 0
    np.testing.assert_allclose(kernel(a), np.asarray(a) + 1, rtol=1e-6)


def test_an_output_keeping_an_input_may_be_tiled_along_its_diagonal(tmp_path):
    src = """
def f(a):
    out = a
    for i in range(0, M, T):
        out = out.at[i:i + T, i:i + T].set(a[i:i + T, i:i + T] * 2)
    return out
"""
    a = np.asarray(rand((M, M)))
    want = a.copy()
    for i in range(0, M, T):
        want[i : i + T, i : i + T] *= 2
    np.testing.assert_allclose(_kernel(tmp_path, src)(jnp.asarray(a)), want, rtol=1e-6)


def test_the_wrong_number_of_specs_is_rejected(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
"""
    fn = _fn(tmp_path, src)
    spec = jax.ShapeDtypeStruct((M, N), jnp.float32)
    with pytest.raises(AutoKernelError, match="f takes 1 arrays, got 2"):
        analysis.specialize(frontend.parse(fn), fn, [spec, spec])


def test_globals_of_an_environment_are_its_names():
    env = analysis.Environment({"T": 128}, "<test>")
    assert analysis.user_globals(env) == {"T": 128}
    assert analysis.user_globals(env) is not env.names  # a copy
    assert analysis.environment(env) is env


def test_an_unassigned_closure_variable_is_rejected():
    def make():
        @autokernel.tpu
        def g(a):
            out = jnp.zeros((M, N))
            for i in range(0, M, late):
                out = out.at[i : i + late, :].set(a[i : i + late, :])
            return out

        with pytest.raises(AutoKernelError, match="closure variable `late` has no value yet"):
            g.lower(rand((M, N)))
        late = T
        return g

    g = make()
    a = rand((M, N))
    np.testing.assert_allclose(g(a), a)


# --------------------------------------------------------------------------- accumulators


@pytest.mark.parametrize("update", ["acc = acc + a[i:i + T, k:k + T]", "acc = a[i:i + T, k:k + T] + acc"])
def test_an_accumulator_may_be_spelled_as_an_addition(tmp_path, update):
    src = f"""
def f(a):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        acc = jnp.zeros((T, T))
        for k in range(0, T, T):
            {update}
        out = out.at[i:i + T, :].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    a = rand((M, N))
    assert [acc.var for acc in kernel.lower(a).plan.spec.accumulators] == ["acc"]
    np.testing.assert_allclose(kernel(a), np.asarray(a)[:, :T], rtol=1e-6)


def test_other_updates_are_not_accumulators(tmp_path):
    src = """
def f(a):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        acc = jnp.zeros((T, T))
        for k in range(0, T, T):
            acc = acc * 2 + a[i:i + T, k:k + T]
        out = out.at[i:i + T, :].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    a = rand((M, N))
    assert kernel.lower(a).plan.spec.accumulators == []
    np.testing.assert_allclose(kernel(a), np.asarray(a)[:, :T], rtol=1e-6)


# --------------------------------------------------------------------------- runtime scalars


def _layer_kernel(tmp_path, index, store="out.at[i:i + T, :]"):
    src = f"""
def f(w, s):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = {store}.set(w[{index}, i:i + T, :] * 2)
    return out
"""
    return _kernel(tmp_path, src)


@pytest.mark.parametrize(
    "index, store, match",
    [
        ("s.missing", "out.at[i:i + T, :]", r"can't evaluate the runtime index `s.missing` \(AttributeError"),
        ("s * 1.5", "out.at[i:i + T, :]", r"the runtime index `s \* 1.5` must be one integer"),
        ("s[0]", "out.at[i:i + T, :]", r"`s` is a scalar input \(0-d\); it can't be indexed"),
    ],
)
def test_bad_runtime_indices_are_rejected(tmp_path, index, store, match):
    with pytest.raises(AutoKernelError, match=match):
        _layer_kernel(tmp_path, index, store).lower(rand((3, M, N)), jnp.int32(1))


def test_an_output_tile_cant_be_placed_by_a_runtime_index(tmp_path):
    src = """
def f(w, s):
    out = jnp.zeros((3, M, N))
    for i in range(0, M, T):
        out = out.at[s, i:i + T, :].set(w[0, i:i + T, :])
    return out
"""
    with pytest.raises(AutoKernelError, match=r"`out`'s tiles can't be placed by a runtime index"):
        _kernel(tmp_path, src).lower(rand((3, M, N)), jnp.int32(1))


# --------------------------------------------------------------------------- masks that read inputs


def _masked(cond):
    """A kernel copying the row tiles of `a` its tile mask keeps (`cond`, of `i` and the scalar `n`)."""
    return f"""
def f(a, n):
    out = jnp.zeros((8 * T, T))
    for i in range(0, 8 * T, T):
        if not ({cond}):
            continue
        out = out.at[i:i + T, :].set(a[i:i + T, :])
    return out
"""


@pytest.mark.parametrize(
    "cond, keep",
    [
        ("~(i // T) < -n", lambda b, n: ~b < -n),
        ("abs(i // T - n) <= 1", lambda b, n: abs(b - n) <= 1),
        ("max(i // T, n, 2) == i // T", lambda b, n: max(b, n, 2) == b),
        ("min(i // T, n) == n", lambda b, n: min(b, n) == n),
        ("i // T in (n, n + 2)", lambda b, n: b in (n, n + 2)),
        ("i // T not in [n, 0]", lambda b, n: b not in [n, 0]),
        ("1 < i // T < n", lambda b, n: 1 < b < n),
        ("(i // T if n > 0 else 0) > 4", lambda b, n: (b if n > 0 else 0) > 4),
    ],
)
def test_runtime_masks_keep_pythons_meaning(tmp_path, cond, keep):
    kernel = _kernel(tmp_path, _masked(cond))
    a, n = rand((8 * T, T)), 3
    want = np.zeros((8 * T, T), np.float32)
    for b in range(8):
        if keep(b, n):
            want[b * T : (b + 1) * T] = a[b * T : (b + 1) * T]
    np.testing.assert_allclose(kernel(a, jnp.int32(n)), want)
    np.testing.assert_allclose(kernel.reference(a, jnp.int32(n)), want)


@pytest.mark.parametrize(
    "cond, match",
    [
        ("jnp.zeros(3) + n > i", r"can't evaluate the tile mask .* on device \(ValueError: the condition is an array"),
        ("max((i, n)) > 0", "Python control flow on the values it reads"),
    ],
)
def test_bad_runtime_masks_are_rejected(tmp_path, cond, match):
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, _masked(cond)).lower(rand((8 * T, T)), jnp.int32(3))


def test_a_masked_step_may_assign_a_value_it_reads(tmp_path):
    # `t` has no value before the first step the mask runs; the reference keeps it as computed
    src = """
def f(a, n):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        acc = jnp.zeros((T, T))
        for k in range(0, K, T):
            if k // T < n:
                t = a[i:i + T, k:k + T] * 2
                acc += t
        out = out.at[i:i + T, :].set(acc)
    return out
"""
    kernel = _kernel(tmp_path, src)
    a = rand((M, K))
    want = 2 * np.asarray(a)[:, :T]
    np.testing.assert_allclose(kernel.reference(a, jnp.int32(1)), want, rtol=1e-6)
    np.testing.assert_allclose(kernel(a, jnp.int32(1)), want, rtol=1e-6)


# --------------------------------------------------------------------------- views


def _view_spec(tmp_path, src, view_shape, base_shape):
    """Specialise `f` for a view input `v` of shape `view_shape`, a reshape of an array of `base_shape`."""
    fn = _fn(tmp_path, src)
    base = jax.ShapeDtypeStruct(base_shape, jnp.float32)
    vmap = views_mod.ViewMap.trace(lambda x: x.reshape(view_shape), base)
    env = analysis.Environment(analysis.user_globals(fn), "<test>", {"v": vmap})
    return lambda: analysis.specialize(frontend.parse(fn), env, [jax.ShapeDtypeStruct(view_shape, jnp.float32)])


@pytest.mark.parametrize(
    "src, view_shape, base_shape, match",
    [
        (
            """
def f(v):
    out = jnp.zeros((200, N))
    for i in range(0, 200, T):
        out = out.at[i:i + T, :].set(v[i:i + T, :])
    return out
""",
            (200, N),
            (2, 100, N),
            "tiles of `v`, a view, run past its end",
        ),
        (
            """
def f(v):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        if v[0, 0] > 0:
            continue
        out = out.at[i:i + T, :].set(v[i:i + T, :])
    return out
""",
            (M, N),
            (2, T, N),
            "a mask reads `v`, a view",
        ),
        (
            """
def f(v):
    out = jnp.zeros_like(v)
    for i in range(0, M, T):
        out = out.at[i:i + T, :].set(v[i:i + T, :])
    return out
""",
            (M, N),
            (2, T, N),
            "an output allocation reads `v`, a view",
        ),
    ],
)
def test_views_a_nest_cant_read_as_windows(tmp_path, src, view_shape, base_shape, match):
    with pytest.raises(views_mod.ViewError, match=match) as e:
        _view_spec(tmp_path, src, view_shape, base_shape)()
    assert e.value.param == "v"


# --------------------------------------------------------------------------- narrow integer scalars


def _elementwise(value):
    """A kernel storing `y = value` (of the tile `t = x[i:i + T, :]` and the scalar `n`) into every tile."""
    return f"""
def f(x, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        t = x[i:i + T, :]
        y = {value}
        out = out.at[i:i + T, :].set(y)
    return out
"""


def test_jax_checkpoint_in_a_tile_is_rejected_before_mosaic(tmp_path):
    """Found on a v5e: Pallas has no TPU lowering for `remat2`; the interpreter ran it anyway."""
    kernel = _kernel(tmp_path, _elementwise("jax.checkpoint(lambda v: jnp.maximum(v, 0))(t)"))
    with pytest.raises(AutoKernelError, match=r"uses `remat2`, which Pallas can't lower to a TPU kernel.*call the"):
        kernel.lower(rand((M, N)), jnp.int32(1))


@pytest.mark.parametrize(
    "value, n, want",
    [
        ("t + jax.lax.max(n, jnp.int32(7))", jnp.int16(5), lambda t: t + 7),  # a strongly typed int32
        ("t * jax.lax.pow(2.0, n)", jnp.int16(3), lambda t: 8 * t),  # a weakly typed float with `n`
        ("t + jax.jit(lambda v: jnp.maximum(v, 0))(n)", jnp.int16(-4), lambda t: t),  # a nested jaxpr
        ("t + jax.lax.shift_right_logical(n, 1)", jnp.uint16(200), lambda t: t + 100),
    ],
)
def test_narrow_scalar_operations_followed_exactly(tmp_path, value, n, want):
    kernel = _kernel(tmp_path, _elementwise(value))
    x = rand((M, N))
    np.testing.assert_allclose(kernel(x, n), want(np.asarray(x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "value, n, op",
    [
        ("t + jax.lax.shift_right_logical(n, 1)", jnp.int16(-8), "shift_right_logical"),
        ("t + jax.lax.while_loop(lambda v: v > n, lambda v: v, n)", jnp.int8(1), "while"),
    ],
)
def test_narrow_scalar_operations_that_could_differ_are_rejected(tmp_path, value, n, op):
    match = rf"does `{op}` on `n` \({'an' if n.dtype.name.startswith('int') else 'a'} {n.dtype.name} runtime scalar\)"
    with pytest.raises(AutoKernelError, match=match):
        _kernel(tmp_path, _elementwise(value)).lower(rand((M, N)), n)


def _carried(init, update):
    """A two-step reduction carrying `c` (`init`, then `update` each step, of the scalars `n` and `u`)."""
    return f"""
def f(x, n, u):
    out = jnp.zeros((M, T))
    for i in range(0, M, T):
        c = {init}
        acc = jnp.zeros((T, T))
        for k in range(0, K, T):
            c = {update}
            acc += x[i:i + T, k:k + T]
        out = out.at[i:i + T, :].set(acc + c)
    return out
"""


def test_a_narrow_value_held_in_its_own_type_between_steps(tmp_path):
    kernel = _kernel(tmp_path, _carried("jnp.zeros((), jnp.uint16)", "n"))
    x = rand((M, K))
    want = np.asarray(x)[:, :T] + np.asarray(x)[:, T:] + 9
    np.testing.assert_allclose(kernel(x, jnp.uint16(9), jnp.uint16(0)), want, rtol=1e-5, atol=1e-5)


def test_a_narrow_value_widening_between_steps(tmp_path):
    # `c` is an int8 value on the first step, an int16 one (int8 with uint8) after: both exact on 32 bits
    kernel = _kernel(tmp_path, _carried("n", "jnp.maximum(c, u)"))
    x = rand((M, K))
    want = np.asarray(x)[:, :T] + np.asarray(x)[:, T:] + 200
    np.testing.assert_allclose(kernel(x, jnp.int8(-3), jnp.uint8(200)), want, rtol=1e-5, atol=1e-5)


def test_a_narrow_value_stored_in_a_narrower_state_is_rejected(tmp_path):
    # `c` is a uint16 (the kernel's 32-bit `n` mixes with it as a Python int would); Python's `c` would be an int16
    # on the first step, then an int32
    kernel = _kernel(tmp_path, _carried("jnp.zeros((), jnp.uint16)", "n"))
    match = r"`c = n` mixes `n` \(an int16 runtime scalar\) .* with uint16 values: Python computes that in int32"
    with pytest.raises(AutoKernelError, match=match):
        kernel.lower(rand((M, K)), jnp.int16(9), jnp.uint16(0))


def test_a_weakly_typed_conversion_keeps_a_narrow_value():
    from jax._src.lax.lax import _convert_element_type

    closed = jax.make_jaxpr(lambda v: _convert_element_type(v, np.dtype(np.int32), weak_type=True))(jnp.int8(1))
    narrow = analysis._Narrow(np.dtype(np.int8), "n")
    problem, outs = analysis._follow_narrow(closed.jaxpr, {closed.jaxpr.invars[0]: narrow})
    assert problem is None and outs == [narrow]


# --------------------------------------------------------------------------- scopes and metadata


def test_a_mapping_pattern_binds_its_rest():
    tree = ast.parse("match v:\n    case {'a': 1, **rest}:\n        pass\n")
    assert scopes.binds(tree.body, False) == {"rest"}


def test_the_version_falls_back_when_the_package_isnt_installed(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    before = autokernel.__version__
    try:
        with monkeypatch.context() as m:
            m.setattr(importlib.metadata, "version", missing)
            importlib.reload(autokernel)
            # Read from the source tree's pyproject.toml, so it can't drift from the package's version.
            pyproject = pathlib.Path(autokernel.__file__).parents[2] / "pyproject.toml"
            assert autokernel.__version__ == tomllib.loads(pyproject.read_text())["project"]["version"]
            m.setattr(autokernel, "_PYPROJECT", pyproject.parent / "missing.toml")
            assert autokernel._source_version() == "0+unknown"
    finally:
        importlib.reload(autokernel)
    assert autokernel.__version__ == before


def test_the_globals_of_a_function_include_its_closure():
    def make():
        def g():
            return early, late

        early = 1
        with pytest.raises(AutoKernelError, match="g: closure variable `late` has no value yet"):
            analysis.user_globals(g)
        late = 2
        return g

    names = analysis.user_globals(make())
    assert (names["early"], names["late"]) == (1, 2)
    assert names["np"] is np  # and the module's globals
