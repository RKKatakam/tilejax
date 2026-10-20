"""Shared test helpers."""

from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

import autokernel

V5E = jax.sharding.AbstractDevice(device_kind="TPU v5 lite", num_cores=1, platform="tpu")


def rand(shape: tuple[int, ...], seed: int = 0, dtype: Any = jnp.float32) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), shape, dtype)


def spec(*shape: int, dtype: Any = jnp.float32) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def run_checking_races(kernel: autokernel.Kernel, *args: Any) -> Any:
    """Run a kernel (on its inputs, and tunable if given) in the Pallas TPU interpreter with its race
    detector, uninitialized memory as NaN."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    arrays = args[: len(kernel.nest.params)]
    out = jax.block_until_ready(kernel.lower(*args).entry(*arrays, interpret=params))
    assert interpreter.races is not None and not interpreter.races.races_found, "the kernel has data races"
    return out


def lower_for_v5e(lowered: autokernel.Lowered, *specs: jax.ShapeDtypeStruct) -> str:
    """Cross-lower a generated kernel to Mosaic for a TPU v5e, on any machine; the HLO text."""
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=V5E)
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(lambda *xs: lowered.entry(*xs, interpret=False)).trace(*specs)
        return traced.lower(lowering_platforms=("tpu",)).as_text()


def f64(x: Any) -> np.ndarray:
    return np.asarray(x, np.float64)


_sources = itertools.count()


def source_function(source: str, **namespace: Any) -> Any:
    """The function a source string defines (its last top-level `def`), with readable source, so autokernel
    can parse it. `namespace` holds its globals, with jax, jnp, np and math there already."""
    import linecache
    import math
    import textwrap

    source = textwrap.dedent(source)
    filename = f"<test source {next(_sources)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    globals_ = {"jax": jax, "jnp": jnp, "np": np, "math": math, **namespace}
    exec(compile(source, filename, "exec"), globals_)
    name = [line.split("(")[0][4:] for line in source.splitlines() if line.startswith("def ")][-1]
    return globals_[name]


def make_kernel(source: str, **namespace: Any) -> autokernel.Kernel:
    """`autokernel.kernel` of the function a source string defines (see `source_function`)."""
    return autokernel.kernel(source_function(source, **namespace))
