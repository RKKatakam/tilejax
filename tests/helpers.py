"""Shared test helpers."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def rand(shape, seed=0, dtype=jnp.float32):
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype)


def run_checking_races(kernel, *args):
    """Run a kernel in the TPU interpreter with the race detector on; return its output."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
    out = jax.block_until_ready(kernel.lower(*args).entry(*args, interpret=params))
    assert interpreter.races is not None and not interpreter.races.races_found, "the kernel has data races"
    return out
