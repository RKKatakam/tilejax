"""`rows`: a window of rows chosen when the kernel runs, as many as are valid.

Ragged batches (sequences of many lengths, their tokens concatenated) read and store a tile's rows from a
row the kernel learns as it runs, up to a bound: `q[rows(start, stop, n), :, :]` reads rows start.. (n of
them), and `out = out.at[rows(start, stop, n), :, :].set(v)` stores v's first stop - start rows there,
nothing past `stop`. In the kernel both are copies of stop - start rows (fewer than n: the rest of the tile
keeps what its VMEM slot held), but along one of an array's last two dims (tiled: (8, 128) for 32 bits) a
read starts at a whole tile (`start` must be a multiple of 8 rows for 32 bits, 16 for 16, 128 lanes) and
copies whole tiles, up to the next tile boundary past `stop`; stores can't be ragged there.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

__all__ = ["rows"]

_PAST = 2**31 - 1  # an index out of every array: JAX drops stores there (and clamps reads)


def rows(start: Any, stop: Any, size: int) -> Any:
    """The row indices start, start + 1, ..., start + size - 1, those at or past `stop` out of bounds: as a
    tile read's index, `size` rows from `start` (past `stop`, whatever is there); as a store's, only the rows
    before `stop` are written. `size` is an int when the kernel is built."""
    index = start + jnp.arange(size, dtype=jnp.int32)
    return jnp.where(index < stop, index, _PAST)
