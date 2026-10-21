"""`pairs`: a 16-bit tile's pairs of rows as 32-bit words, the way a TPU packs them.

A TPU keeps 16-bit values two rows to a 32-bit sublane. A tile whose second-to-last dim is 2, a pair of
rows (a cache page holding each head's key and value side by side, (..., 2, d)), is in VMEM a tile of words
(..., d): row 0 in each word's low half, row 1 in its high half. Kernel code that reads it so
(`pairs(cache[...])`) gets those words straight from the VMEM buffer, and splits them with shifts;
slicing the rows out of the 16-bit tile instead costs sublane shuffles (on a v5e, 0.34 vs 10 us for a page
of 128 positions of 8 heads).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["pairs"]


def pairs(x: Any) -> Any:
    """The tile `x` (..., 2, d) of a 16-bit dtype as uint32 words (..., d): `x[..., 0, i]` in each word's low
    16 bits, `x[..., 1, i]` in its high 16 bits. In kernel code, `x` must be a tile read (`pairs(c[...])`)."""
    return jax.lax.bitcast_convert_type(jnp.swapaxes(x, -1, -2), jnp.uint32)
