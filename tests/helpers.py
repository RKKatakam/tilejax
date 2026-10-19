"""Shared test helpers."""

import jax
import jax.numpy as jnp


def rand(shape, seed=0, dtype=jnp.float32):
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype)
