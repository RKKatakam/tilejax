"""Regression tests for the decorator's second review: constants read once per signature (no kernel, nor
copy of the module, kept per value), the reference without lowering, scope-aware global names, the
per-call fast path, and `verify` keywords."""

import ast
import gc
import inspect
import sys
import textwrap
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, api, frontend, scopes

M, N, T = 16, 128, 8
LR = 0.5
STATE = None  # unrelated module state the tests replace
a = jnp.ones((4, 4))  # a module array named like the lambda parameter below


def scaled_by_lr(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * LR)
    return out


def mapped(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(jax.tree.map(lambda a: a[:, :N] * 2, x[i : i + T, :]))
    return out


def narrow_rows(x):  # 4-row tiles of f32 break the (8, 128) layout rule: no kernel, but loops that run
    out = jnp.zeros((M, N))
    for i in range(0, M, 4):
        out = out.at[i : i + 4, :].set(x[i : i + 4, :] * 2)
    return out


def tolerant(x, rtol):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + rtol[i : i + T, :])
    return out


def set_global(name, value):
    sys.modules[__name__].__dict__[name] = value


# --------------------------------------------------------------------------- constants: once per signature


def test_changing_a_constant_builds_no_new_kernel_and_keeps_no_old_module_values():
    kernel = autokernel.tpu(scaled_by_lr)
    x = jnp.ones((M, N))
    refs = []
    try:
        for step in range(4):
            state = np.zeros((256, 1024), np.float32)  # module state the nest doesn't read, replaced every step
            refs.append(weakref.ref(state))
            set_global("STATE", state)
            set_global("LR", 0.25 * (step + 1))
            assert float(kernel(x)[0, 0]) == 0.25  # the value when this signature was first built
            del state
    finally:
        set_global("STATE", None)
        set_global("LR", 0.5)
    gc.collect()
    assert [r() is None for r in refs] == [True] * 4  # nothing holds a copy of the module's globals
    assert len(kernel._contexts) == len(kernel._lowered) == len(kernel._compiled) == 1
    (ctx,) = kernel._contexts.values()
    assert set(ctx.env.names) == {"jnp", "M", "N", "T", "LR"}  # only the names the code reads


def test_kernel_reference_derivatives_and_batches_share_the_snapshot(monkeypatch):
    kernel = autokernel.tpu(scaled_by_lr)
    x = rand((M, N))
    want = x * 0.5
    np.testing.assert_allclose(kernel(x), want, rtol=1e-6)
    monkeypatch.setattr(sys.modules[__name__], "LR", 3.0)
    np.testing.assert_allclose(kernel(x), want, rtol=1e-6)
    np.testing.assert_allclose(kernel.reference(x), want, rtol=1e-6)
    assert kernel.verify(x) == 0.0
    np.testing.assert_allclose(jax.grad(lambda x: kernel(x).sum())(x), jnp.full((M, N), 0.5))
    np.testing.assert_allclose(jax.jvp(kernel, (x,), (jnp.ones_like(x),))[1], jnp.full((M, N), 0.5))
    np.testing.assert_allclose(jax.vmap(kernel)(jnp.stack([x, x])), jnp.stack([want, want]), rtol=1e-6)
    np.testing.assert_allclose(jax.jit(kernel)(x), want, rtol=1e-6)
    (batched,) = kernel._batched.values()
    (ctx,) = kernel._contexts.values()
    assert batched.fn.names["LR"] == ctx.env.names["LR"] == 0.5


def test_an_unassigned_closure_variable_is_reported():
    def make():
        def f(x):
            out = jnp.zeros((M, N))
            for i in range(0, M, T):
                out = out.at[i : i + T, :].set(x[i : i + T, :] * later)
            return out

        kernel = autokernel.tpu(f)
        with pytest.raises(AutoKernelError, match="closure variable `later` has no value yet"):
            kernel(jnp.ones((M, N)))
        later = 2.0
        return kernel

    assert float(make()(jnp.ones((M, N)))[0, 0]) == 2.0


# --------------------------------------------------------------------------- the per-call fast path


def test_calls_snapshot_and_bind_once(monkeypatch):
    snapshots = []
    real = api._Reader.snapshot
    monkeypatch.setattr(api._Reader, "snapshot", lambda self, names: snapshots.append(1) or real(self, names))
    kernel = autokernel.tpu(scaled_by_lr)
    x = jnp.ones((M, N))
    for _ in range(3):
        kernel(x)
        kernel(x=x)
    assert len(snapshots) == 1
    binds = []
    monkeypatch.setattr(inspect.Signature, "bind", lambda *a, **k: binds.append(1))
    kernel(x)  # positional arguments skip `inspect.Signature.bind`
    assert binds == []


def test_the_signature_fast_path_matches_the_spec():
    cases = [
        jnp.ones((2, 3)),
        jnp.ones((2, 3), jnp.bfloat16),
        jnp.asarray(1.0),
        jnp.float32(1.0),
        np.ones((2, 3)),
        3,
        np.int64(3),
        jax.ShapeDtypeStruct((2, 3), jnp.int8),
    ]
    for x in cases:
        s = api._spec(x)
        assert api._arg_signature(x) == (tuple(s.shape), np.dtype(s.dtype), bool(s.weak_type)), x


# --------------------------------------------------------------------------- the reference


def test_the_reference_runs_without_a_kernel(monkeypatch):
    kernel = autokernel.tpu(narrow_rows)
    x = rand((M, N))
    np.testing.assert_allclose(kernel.reference(x), x * 2)
    assert kernel._lowered == {}  # nothing was lowered for it
    with pytest.raises(AutoKernelError, match="layout rules"):
        kernel(x)


# --------------------------------------------------------------------------- scope-aware global names


def free_and_sliced(src):
    fdef = ast.parse(textwrap.dedent(src)).body[0]
    nest = frontend.parse_function(fdef, fdef.name, 1)
    return set(api._free_names(nest)), api._sliced_names(nest)


def test_a_lambda_parameter_named_like_a_global_array_is_not_an_input():
    kernel = autokernel.tpu(mapped)
    x = rand((M, 2 * N))
    assert kernel.verify(x) == 0.0
    assert [op.param for op in kernel.lower(x).operands] == ["x"]
    np.testing.assert_allclose(kernel(x), x[:, :N] * 2)


def test_free_names_know_their_scopes():
    free, sliced = free_and_sliced(
        """
        def f(x):
            out = jnp.zeros((M, N))
            for i in range(0, M, T):
                t = x[i : i + T, :]
                t = jax.tree.map(lambda a, b=B: a[:, :N] + b, t)
                t = t + sum(v[i : i + T, :] for v in (t, t))
                out = out.at[i : i + T, :].set(t + TABLE[i : i + T, :] + a)
            return out
        """
    )
    # Bound: a lambda's parameters, comprehension targets.
    assert free == {"jnp", "M", "N", "T", "range", "jax", "B", "sum", "TABLE", "a"}
    assert sliced == {"TABLE"}  # `a[:, :N]` and `v[i : i + T, :]` slice the lambda's and generator's own


def test_free_loads_of_nested_functions_and_classes():
    body = ast.parse(
        textwrap.dedent(
            """
            @D
            def g(c, *rest, k=K, **kw) -> R:
                global G
                G = G + c + W + k
                import numpy as onp
                try:
                    pass
                except E as err:
                    return err
                return onp, rest, kw, g
            class C(Base):
                z = Z
            [(y := w) for w in V for u in w if u > U]
            y + 1
            """
        )
    ).body
    loads = {n.id for n in scopes.global_loads(body, scopes.binds(body))}
    assert loads == {"D", "K", "R", "G", "W", "E", "Base", "Z", "V", "U"}


def test_a_global_sliced_at_the_top_level_and_in_a_lambda_is_an_input():
    free, sliced = free_and_sliced(
        """
        def f(x):
            out = jnp.zeros((M, N))
            for i in range(0, M, T):
                t = jax.tree.map(lambda a: a[:, :N], x[i : i + T, :])
                out = out.at[i : i + T, :].set(t + a[i : i + T, :])
            return out
        """
    )
    assert "a" in free and sliced == {"a"}


# --------------------------------------------------------------------------- verify keywords


def test_a_parameter_named_rtol_is_passed_by_keyword_to_verify():
    kernel = autokernel.tpu(tolerant)
    x, r = rand((M, N)), rand((M, N), 1)
    assert kernel.verify(x, rtol=r) == 0.0  # the parameter, not the tolerance
    assert kernel.verify(x, r, atol=1.0) == 0.0  # `atol` is still a tolerance
    np.testing.assert_allclose(kernel(rtol=r, x=x), x + r)
