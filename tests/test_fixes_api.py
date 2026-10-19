"""Regression tests for the decorator, tuning and packaging: weak types, changed constants, keyword arguments,
global arrays, VMEM pruning, option validation, thread safety, and more."""

import importlib.metadata
import os
import re
import subprocess
import sys
import threading
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import AutoKernelError, api, frontend, hardware, lowering
from autokernel import cost as cost_lib
from autokernel import tune as tune_lib

M, N, T = 32, 128, 16
SCALE = 2.0
BIAS = jnp.ones((M, N))
HERE = os.path.dirname(os.path.abspath(__file__))


def scale_by(x, s):
    out = jnp.zeros((M, N), jnp.bfloat16)
    for i in range(0, M, T):
        y = x[i : i + T, :] * s
        # bf16 * weak f32 stays bf16; bf16 * strong f32 is f32: the two give different results on purpose
        out = out.at[i : i + T, :].set(y.astype(jnp.bfloat16) if y.dtype == jnp.float32 else y + 100)
    return out


def scaled(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] * SCALE)
    return out


def add(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + y[i : i + T, :])
    return out


def biased(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + BIAS[i : i + T, :])
    return out


def two(x, y):
    a = jnp.zeros((M, N))
    b = jnp.zeros((M, N), jnp.int32)
    for i in range(0, M, T):
        t = x[i : i + T, :] + y[i : i + T, :]
        a = a.at[i : i + T, :].set(t)
        b = b.at[i : i + T, :].set((t > 0).astype(jnp.int32))
    return a, b


def plain_two(x, y):
    return x + y, (x + y > 0).astype(jnp.int32)


def close(got, want, tol=1e-5):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        assert g.shape == w.shape and g.dtype == w.dtype
        np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(w, np.float32), rtol=tol, atol=tol)


# --------------------------------------------------------------------------- weak types


def test_a_weak_scalar_mixes_with_tiles_as_in_the_reference():
    kernel = autokernel.tpu(scale_by)
    x = jnp.ones((M, N), jnp.bfloat16)
    weak, strong = jnp.asarray(1.0), jnp.float32(1.0)
    assert weak.weak_type and not strong.weak_type
    assert float(kernel(x, weak)[0, 0]) == float(kernel.reference(x, weak)[0, 0]) == 101  # stayed bf16
    assert float(kernel(x, strong)[0, 0]) == float(kernel.reference(x, strong)[0, 0]) == 1  # promoted to f32
    kernel.verify(x, weak)
    kernel.verify(x, strong)
    assert len(kernel._lowered) == 2  # weak and strong are different signatures


# --------------------------------------------------------------------------- changed constants


def test_a_changed_module_constant_gives_a_new_kernel_everywhere(monkeypatch):
    kernel = autokernel.tpu(scaled)
    x = jnp.ones((M, N))
    assert float(kernel(x)[0, 0]) == 2.0
    monkeypatch.setattr(sys.modules[__name__], "SCALE", 3.0)
    assert float(kernel(x)[0, 0]) == 3.0  # as the Python would compute now
    assert float(kernel.reference(x)[0, 0]) == 3.0
    assert float(jax.grad(lambda x: kernel(x).sum())(x)[0, 0]) == 3.0
    assert float(jax.vmap(kernel)(x[None])[0, 0, 0]) == 3.0
    assert float(jax.jvp(kernel, (x,), (x,))[1][0, 0]) == 3.0
    monkeypatch.setattr(sys.modules[__name__], "SCALE", 2.0)
    assert float(kernel(x)[0, 0]) == 2.0
    assert len(kernel._contexts) == 2  # the first kernel is reused for the old value


def test_a_changed_closure_constant_gives_a_new_kernel():
    def make():
        factor = 2.0

        def f(x):
            out = jnp.zeros((M, N))
            for i in range(0, M, T):
                out = out.at[i : i + T, :].set(x[i : i + T, :] * factor)
            return out

        def change(v):
            nonlocal factor
            factor = v

        return f, change

    f, change = make()
    kernel = autokernel.tpu(f)
    x = jnp.ones((M, N))
    assert float(kernel(x)[0, 0]) == 2.0
    change(5.0)
    assert float(kernel(x)[0, 0]) == float(jax.grad(lambda x: kernel(x).sum())(x)[0, 0]) == 5.0


def test_constant_keys_tell_values_apart():
    key = api._value_key
    assert key(2) != key(2.0) != key(True)
    assert key(float("nan")) == key(float("nan"))  # nan as a constant still reuses its kernel
    assert key((1, 2)) == key((1, 2)) != key((1, 3))
    assert key(np.arange(4)) == key(np.arange(4)) != key(np.arange(1, 5))  # small tables: by value
    assert key(jnp) == key(jnp) != key(np)


# --------------------------------------------------------------------------- keyword arguments


def test_every_entry_point_takes_keyword_arguments():
    kernel = autokernel.tpu(add)
    x, y = rand((M, N)), rand((M, N), 1)
    close(kernel(x, y=y), x + y)
    close(kernel(y=y, x=x), x + y)
    close(kernel.reference(x=x, y=y), x + y)
    assert kernel.source(x=x, y=y) == kernel.source(x, y)
    assert kernel.lower(x, y=jax.ShapeDtypeStruct(y.shape, y.dtype)) is kernel.lower(x, y)
    assert kernel.verify(x, y=y) == 0.0
    assert kernel.tuning(x, y=y) == []
    with pytest.raises(AutoKernelError, match="takes 2 arrays, got 1"):
        kernel(x)
    with pytest.raises(AutoKernelError, match="unexpected keyword argument 'z'"):
        kernel(x, y, z=y)


# --------------------------------------------------------------------------- global arrays


def test_a_global_array_read_through_slices_is_an_input(monkeypatch):
    kernel = autokernel.tpu(biased)
    assert isinstance(kernel, autokernel.TpuKernel)
    x = rand((M, N))
    close(kernel(x), x + 1)
    assert kernel.lower(x).operands[-1].param == "BIAS"
    kernel.verify(x)
    monkeypatch.setattr(sys.modules[__name__], "BIAS", jnp.full((M, N), 5.0))
    close(kernel(x), x + 5)  # its current value, on every call
    close(kernel.reference(x), x + 5)
    close(jax.jit(kernel)(x), x + 5)
    close(jax.vmap(kernel)(jnp.stack([x, 2 * x])), jnp.stack([x + 5, 2 * x + 5]))
    close(jax.grad(lambda x: kernel(x).sum())(x), jnp.ones((M, N)))
    monkeypatch.setattr(sys.modules[__name__], "BIAS", np.full((M, N), 2.0, np.float32))
    close(kernel(x), x + 2)


def test_a_global_array_with_a_user_backward_gets_no_gradient():
    kernel = autokernel.tpu(biased, backward=lambda x, ct: (3 * ct,))
    x = rand((M, N))
    close(jax.grad(lambda x: kernel(x).sum())(x), jnp.full((M, N), 3.0))


# --------------------------------------------------------------------------- verify


def test_verify_reports_a_mismatch_on_every_call(monkeypatch):
    kernel = autokernel.tpu(add, verify=True)
    monkeypatch.setattr(kernel, "_reference_of", lambda ctx, x, y: x + y + 1)  # a wrong reference
    x, y = rand((M, N)), rand((M, N), 1)
    for _ in range(2):
        with pytest.raises(AssertionError, match="disagrees"):
            kernel(x, y)
    good = autokernel.tpu(add, verify=True)
    good(x, y)
    assert len(good._verified) == 1


# --------------------------------------------------------------------------- options


def test_bad_tune_values_fail_at_decoration_time():
    with pytest.raises(AutoKernelError, match=r"tune\['T'\] must be 'auto' or a sequence"):
        autokernel.tpu(scaled, tune={"T": 8})


def test_tune_values_of_different_types_are_different_candidates():
    kernel = autokernel.tpu(scaled, tune={"SCALE": (2, 2.0, 2)})
    candidates = kernel.tuning(jax.ShapeDtypeStruct((M, N), jnp.float32))
    assert sorted(type(c.overrides["SCALE"]).__name__ for c in candidates) == ["float", "int"]
    assert tune_lib._distinct((2, 2.0, np.float32(2), 2)) == (2, 2.0, np.float32(2))
    assert api._typed({"S": 2}) != api._typed({"S": 2.0})


def test_measuring_without_tune_measures_the_buffer_depths():
    kernel = autokernel.tpu(scaled, autotune="measure")
    candidates = kernel.tuning(jax.ShapeDtypeStruct((M, N), jnp.float32))
    assert sorted(c.buffers for c in candidates) == [2, 3] and all(c.overrides == {} for c in candidates)
    x = rand((M, N))
    close(kernel(x), x * SCALE)


def test_time_kernel_runs_a_kernel():
    kernel = autokernel.tpu(add)
    x, y = rand((M, N)), rand((M, N), 1)
    seconds = tune_lib.time_kernel(kernel.lower(x, y), (x, y), runs=2, repeats=1, interpret=pltpu.InterpretParams())
    assert 0 < seconds < 60


def test_the_tpu_docstring_states_the_default_vmem_limit():
    assert "90% of its VMEM" in autokernel.tpu.__doc__
    assert hardware.target("v5e").vmem_limit == int(0.9 * hardware.target("v5e").vmem_bytes)


# --------------------------------------------------------------------------- VMEM pruning

K8 = 8192
TM, TK = 512, 4096


def deep_matmul(x, y):
    out = jnp.zeros((1024, 1024))
    for i in range(0, 1024, TM):
        for j in range(0, 1024, 512):
            acc = jnp.zeros((TM, 512))
            for k in range(0, K8, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + 512]
            out = out.at[i : i + TM, j : j + 512].set(acc)
    return out


def test_a_single_step_reduction_is_not_pruned_by_a_smaller_block():
    # With six buffers, TK=4096 doesn't fit, but TK=8192 (one step: its tiles fetched once per tile, into
    # two slots, no accumulator) does: a smaller block that doesn't fit says nothing about it.
    specs = [jax.ShapeDtypeStruct((1024, K8), jnp.float32), jax.ShapeDtypeStruct((K8, 1024), jnp.float32)]
    kernel = autokernel.tpu(deep_matmul, buffers=6, tune={"TM": (256, 512), "TK": (2048, 4096, 8192)})
    by = {(c.overrides["TM"], c.overrides["TK"]): c for c in kernel.tuning(*specs)}
    assert by[512, 4096].error is not None and "VMEM" in by[512, 4096].error
    assert by[512, 8192].error is None  # planned, and it fits
    options = lowering.Options(buffers=6, tune=lowering.Options.make_tune({"TK": (2048, 4096, 8192)}))
    space = tune_lib.Space(frontend.parse(deep_matmul), deep_matmul, specs, options, [("TK", (2048, 4096, 8192))])
    assert space.whole == [{2}]
    assert not space._covers((1,), (2,)) and not space._covers((2,), (1,)) and space._covers((0,), (1,))


# --------------------------------------------------------------------------- VMEM of 1-D tiles


def rows(x):
    out = jnp.zeros((8, 1024))
    for b in range(8):
        out = out.at[b, :].set(x[b, :] * 2)
    return out


def test_one_dimensional_slots_are_padded_to_whole_sublane_tiles():
    kernel = autokernel.tpu(rows)
    lowered = kernel.lower(jax.ShapeDtypeStruct((8, 1024), jnp.float32))
    assert [op.block.kernel_shape for op in lowered.operands] == [(1024,)]
    # Each (2, 1024) f32 ring (the input's and the output's) takes eight rows of 1024 in VMEM, not two.
    assert cost_lib.slots_nbytes(2, (1024,), jnp.float32) == 8 * 1024 * 4
    assert cost_lib.slots_nbytes(2, (1024,), jnp.bfloat16) == 16 * 1024 * 2
    assert cost_lib.slots_nbytes(3, (16, 128), jnp.float32) == 3 * 16 * 128 * 4  # 2-D tiles: as before
    assert lowered.estimate.vmem_bytes >= 2 * 8 * 1024 * 4
    close(kernel(rand((8, 1024))), rand((8, 1024)) * 2)


# --------------------------------------------------------------------------- thread safety


def test_concurrent_first_calls_tune_once(monkeypatch):
    calls = []
    real = tune_lib.search

    def slow_search(*args, **kwargs):  # slow enough for the other threads to arrive while it runs
        calls.append(1)
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(tune_lib, "search", slow_search)
    kernel = autokernel.tpu(add, tune={"T": (8, 16)})
    spec = jax.ShapeDtypeStruct((M, N), jnp.float32)
    barrier = threading.Barrier(6)
    results, errors = [], []

    def work():
        try:
            barrier.wait()
            results.append(kernel.lower(spec, spec))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(calls) == 1 and all(r is results[0] for r in results)


# --------------------------------------------------------------------------- jax.vmap


def test_vmap_with_out_axes():
    kernel = autokernel.tpu(two)
    xs, y = rand((3, M, N)), rand((M, N), 1)
    got = jax.vmap(kernel, in_axes=(0, None), out_axes=1)(xs, y)
    close(got, jax.vmap(plain_two, in_axes=(0, None), out_axes=1)(xs, y))
    got = jax.vmap(kernel, in_axes=(0, None), out_axes=(1, 0))(xs, y)
    close(got, jax.vmap(plain_two, in_axes=(0, None), out_axes=(1, 0))(xs, y))


def test_vmap_with_no_batched_input():
    kernel = autokernel.tpu(two)
    x, y = rand((M, N)), rand((M, N), 1)
    got = jax.vmap(kernel, in_axes=(None, None), axis_size=2)(x, y)
    close(got, jax.vmap(plain_two, in_axes=(None, None), axis_size=2)(x, y))


# --------------------------------------------------------------------------- examples and packaging


def _subprocess_env(**extra: str) -> dict[str, str]:
    """The environment for a child Python on the CPU (the TPU is held by this process), with `src` importable
    whether or not autokernel is installed."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", **extra}
    src = os.path.join(os.path.dirname(HERE), "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    return env


def test_the_tuned_matmul_example_runs():
    root = os.path.dirname(HERE)
    run = subprocess.run(
        [sys.executable, os.path.join(root, "examples", "tuned_matmul.py")],
        env=_subprocess_env(),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "max diff vs Python loops" in run.stdout


def test_the_version_is_the_installed_package_version():
    try:
        installed = importlib.metadata.version("autokernel")
    except importlib.metadata.PackageNotFoundError:  # run from a checkout (PYTHONPATH=src): the fallback
        with open(os.path.join(os.path.dirname(HERE), "pyproject.toml")) as f:
            pyproject = f.read()
        installed = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert autokernel.__version__ == installed


# --------------------------------------------------------------------------- sharded inputs that are gathered

SM, SN, ST = 256, 128, 128


def sharded_add(x, z):
    out = jnp.zeros((SM, SN))
    for i in range(0, SM, ST):
        out = out.at[i : i + ST, :].set(x[i : i + ST, :] + z[i : i + ST, :])
    return out


def sharded_half(x):
    out = jnp.zeros((SM // 2, SN))
    for i in range(0, SM // 2, 64):
        out = out.at[i : i + 64, :].set(x[i : i + 64, :] * 2)
    return out


def sharded_shifted(x, y):  # a program: its nest updates `y` from row 64 on
    for i in range(64, SM, 64):
        y = y.at[i : i + 64, :].set(x[i : i + 64, :] * 2)
    return y + 0


def sharding_fallbacks() -> None:
    """Run by `test_sharded_dims_that_cannot_split_are_gathered` in a process with four CPU devices."""
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    assert jax.device_count() == 4
    m4 = jax.make_mesh((4,), ("d",), axis_types=(AxisType.Explicit,))
    m22 = jax.make_mesh((2, 2), ("a", "b"), axis_types=(AxisType.Explicit,) * 2)
    x, z = rand((SM, SN)), rand((SM, SN), 1)

    def put(a, m, *spec):
        return jax.device_put(a, NamedSharding(m, P(*spec)))

    cases = [
        ("don't split evenly into 4 shards", lambda: autokernel.tpu(sharded_add)(put(x, m4, "d", None), z), x + z),
        (
            "is already split another way",
            lambda: autokernel.tpu(sharded_add)(put(x, m22, "a", None), put(z, m22, "b", None)),
            x + z,
        ),
        ("covers only part of `x`'s dim 0", lambda: autokernel.tpu(sharded_half)(put(x, m4, "d", None)), x[:128] * 2),
        (
            "doesn't start at 0",
            lambda: autokernel.tpu(sharded_shifted)(put(x, m4, "d", None), z),
            z.at[64:].set(x[64:] * 2),
        ),
    ]
    for why, run, want in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            close(run(), want)
        messages = [str(w.message) for w in caught]
        assert any("gathered onto each device" in m and why in m for m in messages), (why, messages)
    print("sharding fallbacks: ok")


def test_sharded_dims_that_cannot_split_are_gathered():
    flags = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
    env = _subprocess_env(XLA_FLAGS=flags)
    run = subprocess.run(
        [sys.executable, "-c", "import test_fixes_api as t; t.sharding_fallbacks()"],
        env=env,
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0 and "sharding fallbacks: ok" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
