"""Regression tests for the second round of code generation fixes: values and user functions in the
generated source (so it runs on its own and computes what the kernel does), user names that collide with
the generated code's, narrow integer scalars, and the VMEM and SMEM budgets of single and fused kernels."""

import builtins
import enum
import importlib.util
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import codegen, fusion, hardware

M = N = K = 256
T = 128


def _load(tmp_path, src, name="ak_fixes2_mod"):
    """The module holding `src`, loaded from a file but not importable (as a script's `__main__` is): the
    generated source has to carry the source of the functions it calls."""
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _standalone(source, args):
    namespace = {}
    exec(source, namespace)  # nothing but the source
    entry = next(v for k, v in namespace.items() if k.endswith("_pallas"))
    return entry(*args, interpret=pltpu.InterpretParams())


def _check_standalone(lowered, args):
    """The source, run on its own, computes exactly what the kernel does."""
    want = lowered.entry(*args, interpret=pltpu.InterpretParams())
    got = _standalone(lowered.source, args)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    return want


# --------------------------------------------------------------------------- user functions


TUNED_HELPER = """
import jax.numpy as jnp

T = 128

def scaled(t):
    return t / T

def f(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i:i + T, j:j + T].set(scaled(x[i:i + T, j:j + T]) + T)
    return out
"""


def test_a_helper_reads_its_own_modules_constants_not_the_tuned_ones(tmp_path):
    # The kernel is tuned to T = 256, but `scaled` divides by its module's T = 128, in the kernel and in the
    # standalone source (which used to give it the tuned T, with only a comment saying so).
    mod = _load(tmp_path, TUNED_HELPER)
    x = rand((256, 256))
    lowered = autokernel.tpu(mod.f, tune={"T": (256,)}).lower(x)
    assert lowered.overrides == {"T": 256}
    want = _check_standalone(lowered, (x,))
    np.testing.assert_allclose(want, np.asarray(x) / 128 + 256, rtol=1e-6)
    assert "T = 256" in lowered.source and "_functions(T=128)" in lowered.source


HELPER_FORMS = """
import functools

import jax
import jax.numpy as jnp

T = 128
SCALE = 3.0


class Arr:
    pass


@functools.partial(jax.jit, static_argnums=1)
def decorated(t, n):
    return t * n


def defaulted(t, s=SCALE * 7):
    return t * s


def annotated(t: Arr) -> Arr:
    return t * 2


def calls_another(t):
    return defaulted(t) + annotated(t.T)  # `.T` is an attribute, not the global T


def f(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            t = x[i:i + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set(decorated(t, 2) + calls_another(t))
    return out
"""


def test_a_helpers_decorators_defaults_and_annotations_get_their_values(tmp_path):
    mod = _load(tmp_path, HELPER_FORMS)
    x = rand((256, 256))
    lowered = autokernel.tpu(mod.f, tune={"T": (256,)}).lower(x)
    want = _check_standalone(lowered, (x,))  # `functools`, `SCALE` and `Arr` used to be NameErrors
    xn = np.asarray(x, np.float64)
    np.testing.assert_allclose(want, 23 * xn + 2 * xn.T, rtol=1e-5, atol=1e-5)  # one 256 x 256 tile
    assert "Arr" not in lowered.source  # annotations are dropped, so `Arr` needs no source
    assert "T=128" not in lowered.source  # the functions read no global `T`


OTHER_MODULE = """
SCALE = 5.0

def helper(t):
    return t * SCALE
"""

USES_OTHER_MODULE = """
import jax.numpy as jnp
from ak_fixes2_other import helper as other_helper

T = 128
SCALE = 3.0

def helper(t):
    return other_helper(t) + SCALE

def f(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i:i + T, j:j + T].set(helper(x[i:i + T, j:j + T]) * SCALE)
    return out
"""


def test_helpers_from_two_modules_read_each_their_own_globals(tmp_path, monkeypatch):
    import sys

    other = _load(tmp_path, OTHER_MODULE, "ak_fixes2_other")
    other.helper.__module__ = "<not importable>"  # so its source goes into the kernel's too
    monkeypatch.setitem(sys.modules, "ak_fixes2_other", other)
    mod = _load(tmp_path, USES_OTHER_MODULE)
    x = rand((256, 256))
    lowered = autokernel.tpu(mod.f).lower(x)
    want = _check_standalone(lowered, (x,))
    np.testing.assert_allclose(want, (np.asarray(x) * 5 + 3) * 3, rtol=1e-6, atol=1e-5)


FUSED_HELPER = """
import jax.numpy as jnp

M = N = K = 256
T = 128

def helper(t):
    return t * 3.0

def f(x, y, keep):
    a = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if keep[i // T, j // T] == 0:
                continue
            a = a.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * 2)
    c = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                if keep[i // T, k // T] == 0:
                    continue
                acc += a[i:i + T, k:k + T] @ y[k:k + T, j:j + T]
            c = c.at[i:i + T, j:j + T].set(helper(acc))
    return a, c
"""


def test_a_fused_kernels_source_defines_the_user_functions_it_calls(tmp_path):
    mod = _load(tmp_path, FUSED_HELPER)
    args = {"x": rand((M, N), 1), "y": rand((K, N), 2), "keep": jnp.array([[1, 0], [1, 1]], jnp.int32)}
    [(_, fused)] = list(autokernel.tpu(mod.f).lower(*args.values()))
    assert isinstance(fused, autokernel.FusedLowered)
    _check_standalone(fused, [args[n] for n in fused.inputs])  # `helper` used to be a NameError


# --------------------------------------------------------------------------- values as source


class Cfg(NamedTuple):
    scale: float
    bias: float


class Mode(enum.IntEnum):
    A = 1
    B = 2


C32 = np.float32(0.5)
BIG = np.int64(2**40)
HALF = np.dtype(jnp.bfloat16).type(1.5)
NEG = np.float64(-np.inf)
CFG = Cfg(2.0, 1.0)
MODE = Mode.B


def numpy_scalars(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            t = jnp.maximum(x[i : i + T, j : j + T], NEG)
            out = out.at[i : i + T, j : j + T].set(t * C32 * HALF + BIG // 2**39 + MODE)
    return out


def test_numpy_scalars_are_numpy_scalars_in_the_source():
    # `jnp.float32(0.5)` is a JAX array, which the kernel captured ("pass them as inputs"); `jnp.int64(2**40)`
    # overflowed; a non-finite np.float64 became a (weak) Python float; an IntEnum was `<Mode.B: 2>`.
    x = rand((256, 256))
    lowered = autokernel.tpu(numpy_scalars).lower(x)
    for line in [
        "C32 = np.float32(0.5)",
        f"BIG = np.int64({2**40})",
        "HALF = np.dtype(jnp.bfloat16).type(1.5)",
        "NEG = np.float64(float('-inf'))",
        "from test_fixes2_codegen import Mode as MODE\nMODE = MODE.B",
    ]:
        assert line in lowered.source
    want = _check_standalone(lowered, (x,))
    np.testing.assert_allclose(want, np.asarray(x) * 0.75 + 4, rtol=1e-6)


@pytest.mark.parametrize(
    "value",
    [
        np.float32(0.5),
        np.int64(2**40),
        np.uint8(200),
        np.bool_(True),
        np.complex64(1 - 2j),
        np.float64(np.nan),
        np.dtype(jnp.bfloat16).type(np.inf),
        np.array([1.5, np.nan, -np.inf], jnp.bfloat16),
        np.array([1 + 2j, complex(np.nan, np.inf)], np.complex64),
        np.array([np.nan, 2.0], np.float16),
        np.zeros((0, 3), np.int8),
        jnp.asarray([1.0, jnp.inf]),
        complex(np.nan, 1.0),
        (1, (2.5, float("inf"))),
        slice(None, 4, None),
        np.dtype(jnp.bfloat16),
    ],
)
def test_literals_have_the_values_type_and_value(value):
    text = codegen._literal(value)
    assert text is not None
    got = eval(text, {"np": np, "jnp": jnp})
    assert type(got) is type(value)
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        assert got.dtype == value.dtype and got.shape == value.shape
        np.testing.assert_array_equal(np.asarray(got, np.complex128), np.asarray(value, np.complex128))
    elif isinstance(value, complex):
        assert np.isnan(got.real) and got.imag == value.imag
    else:
        assert got == value


@pytest.mark.parametrize("value", [CFG, MODE, np.str_("a"), np.array(["a"]), np.datetime64("2026-01-01")])
def test_subclasses_and_non_numbers_have_no_literal(value):
    assert codegen._literal(value) is None  # a NamedTuple used to become a plain tuple, an IntEnum `<Mode.B: 2>`


def named_tuple_constant(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * CFG.scale + CFG.bias)
    return out


def test_a_named_tuple_constant_says_it_has_no_source():
    source = autokernel.tpu(named_tuple_constant).lower(rand((256, 256))).source
    assert "CFG = (2.0, 1.0)" not in source  # a tuple has no `.scale`
    assert "# `CFG` (a Cfg) has no source here" in source


# --------------------------------------------------------------------------- names the generated code uses


ALIASED_NUMPY = """
import jax
import jax.numpy as np  # an old-style alias: `np` is jax.numpy here

M = N = 256
T = 128
zip = None  # a module global hiding a builtin the generated step-table code calls

def mask_reads_np(x, keep):
    out = jax.numpy.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if np.maximum(keep[i // T, j // T], 0) == 0:
                continue
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * 2.0)
    return out

def allocation_reads_np(x, keep):
    out = np.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if np.sum(keep[i // T]) == 0:
                continue
            out = out.at[i:i + T, j:j + T].set(x[i:i + T, j:j + T] * np.float32(2.0))
    return out
"""


@pytest.mark.parametrize("name", ["mask_reads_np", "allocation_reads_np"])
def test_user_code_aliasing_np_to_jax_numpy(tmp_path, name):
    # The runtime mask's table builder ran with the generated code's `np` (numpy: TracerArrayConversionError),
    # and an allocation reading `np` was rejected; now the user's `np` is renamed where the kernel embeds it.
    mod = _load(tmp_path, ALIASED_NUMPY)
    x, keep = rand((M, N)), jnp.array([[1, 0], [0, 0]], jnp.int32)
    kern = autokernel.tpu(getattr(mod, name))
    lowered = kern.lower(x, keep)
    assert "from jax import numpy as np_" in lowered.source
    want = _check_standalone(lowered, (x, keep))
    np.testing.assert_array_equal(np.asarray(want), np.asarray(kern.reference(x, keep)))


def loops_named_pl_and_max(x):
    out = jnp.zeros((256, 256))
    for pl in range(0, 256, T):  # loop variables named like an import and a builtin of the kernel
        for max in range(0, 256, T):
            out = out.at[pl : pl + T, max : max + T].set(x[pl : pl + T, max : max + T] + pl + max)
    return out


def test_loop_variables_named_like_the_generated_codes_names():
    x = rand((256, 256))
    kern = autokernel.tpu(loops_named_pl_and_max)
    want = _check_standalone(kern.lower(x), (x,))
    np.testing.assert_array_equal(np.asarray(want), np.asarray(kern.reference(x)))


def test_the_source_reads_no_name_it_doesnt_bind_but_the_builtins_it_lists(tmp_path):
    mod = _load(tmp_path, ALIASED_NUMPY)
    x, keep = rand((M, N)), jnp.array([[1, 0], [0, 0]], jnp.int32)
    sources = [autokernel.tpu(numpy_scalars).lower(x).source, autokernel.tpu(mod.mask_reads_np).lower(x, keep).source]
    args = (rand((M, N), 1), rand((K, N), 2), jnp.array([[1, 0], [1, 1]], jnp.int32))
    [(_, fused)] = list(autokernel.tpu(_load(tmp_path, FUSED_HELPER, "ak_fixes2_fused").f).lower(*args))
    for source in [*sources, fused.source]:
        unbound = codegen._globals_read(source)
        assert unbound <= set(codegen._BUILTINS) and all(hasattr(builtins, n) for n in unbound), unbound


# --------------------------------------------------------------------------- narrow integer scalars


def narrow_scalar(x, w, n):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            t = w[n, i : i + T, j : j + T] * n + x[i : i + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(jnp.where(n > 0, t, -t))
    return out


@pytest.mark.parametrize("n", [jnp.int8(1), jnp.uint8(1), jnp.int16(-1)])
def test_narrow_integer_scalars_are_held_as_weak_int32s(n):
    x, w = rand((M, N)), rand((2, M, N), 1)
    kern = autokernel.tpu(narrow_scalar)
    lowered = kern.lower(x, w, n)
    assert "n = _weak(scalars[0])" in lowered.source  # its 32-bit value, as base did
    assert f".astype(jnp.{np.dtype(n.dtype).name})" not in lowered.source  # not rebound to its narrow type
    t = np.asarray(w)[int(n)] * np.float32(n) + np.asarray(x)
    np.testing.assert_allclose(lowered.entry(x, w, n, interpret=pltpu.InterpretParams()), t if n > 0 else -t)


# --------------------------------------------------------------------------- budgets


def big_vmem_limit(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    return out


def test_a_user_vmem_limit_equal_to_the_chips_is_still_the_fused_budget():
    hw = hardware.target()
    x = rand((256, 256))
    mine = autokernel.tpu(big_vmem_limit, vmem_limit_bytes=hw.vmem_limit).lower(x)
    default = autokernel.tpu(big_vmem_limit).lower(x)
    assert mine.user_vmem_limit == hw.vmem_limit and default.user_vmem_limit is None
    assert fusion.vmem_budget([mine], hw) == hw.vmem_limit  # was the 75% default budget
    assert fusion.vmem_budget([default], hw) == hw.vmem_budget


def triangle(x):
    out = jnp.zeros((4096, 256 * 128))
    for i in range(0, 4096, 8):
        for j in range(0, 256 * 128, 128):
            if j // 128 > i // 8:
                continue
            out = out.at[i : i + 8, j : j + 128].set(x[i : i + 8, j : j + 128] * 2.0)
    return out


def test_tile_tables_that_fit_the_chips_smem_are_planned():
    # 769 KiB of tile tables (98432 tiles): on a v5e (1 MiB of SMEM) tables of 961 KiB ran; this was
    # rejected against half of SMEM.
    hw = hardware.target("v5e")
    assert hw.smem_budget == hw.smem_bytes - hardware.SMEM_RESERVE == 960 * 1024
    x = jax.ShapeDtypeStruct((4096, 256 * 128), jnp.float32)
    lowered = autokernel.tpu(triangle, background_fill=False, target="v5e").lower(x)
    assert fusion.smem_bytes(lowered) == 98432 * 2 * 4


def small_triangle(x):  # 513 KiB of tile tables
    out = jnp.zeros((3072, 256 * 128))
    for i in range(0, 3072, 8):
        for j in range(0, 256 * 128, 128):
            if j // 128 > i // 8:
                continue
            out = out.at[i : i + 8, j : j + 128].set(x[i : i + 8, j : j + 128] * 2.0)
    return out


def test_nests_whose_tables_dont_fit_smem_together_are_not_fused(monkeypatch):
    monkeypatch.setattr(fusion, "not_worth_it", lambda before, after: None)  # only SMEM can decide
    hw = hardware.target("v5e")
    x = jax.ShapeDtypeStruct((3072, 256 * 128), jnp.float32)
    lowered = autokernel.tpu(small_triangle, background_fill=False, target="v5e").lower(x)
    assert 2 * fusion.smem_bytes(lowered) > hw.smem_budget >= fusion.smem_bytes(lowered)
    nests = [fusion.NestPlan("a", lowered), fusion.NestPlan("b", lowered)]
    segments = fusion.plan_segments(nests, 3, hw)
    assert [(s.start, s.end) for s in segments] == [(0, 1), (1, 2)]
    assert "KiB of SMEM, more than the 960 KiB budget" in segments[0].reason


# --------------------------------------------------------------------------- dead code


def test_the_fused_prelude_is_just_the_imports():
    assert codegen.prelude() == codegen._IMPORTS
    assert not hasattr(codegen.Emitter, "weak_helper")
