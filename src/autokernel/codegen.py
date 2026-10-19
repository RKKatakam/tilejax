"""Generate a Pallas TPU kernel for a parsed loop nest and compile it into a callable.

Two backends share the tile body and differ in how tiles are visited and data is moved:
`manual` (default) writes the pipeline by hand in one loop with no grid; `grid` uses a Pallas grid with
BlockSpecs. `Lowered.source` holds the generated code so users can inspect exactly what runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

from .analysis import specialize
from .emit import FILL_DMA_DEPTH, MATMUL_PRECISIONS, PIPELINES, Lowered, Options
from .frontend import LoopNest
from .grid import GridEmitter
from .manual import ManualEmitter

__all__ = ["FILL_DMA_DEPTH", "MATMUL_PRECISIONS", "PIPELINES", "Lowered", "Options", "lower"]

_BACKENDS = {"manual": ManualEmitter, "grid": GridEmitter}


def lower(
    nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct], options: Options | None = None
) -> Lowered:
    """Specialise `nest` to `specs`, generate the Pallas kernel and compile its source."""
    options = options or Options()
    return _BACKENDS[options.pipeline](specialize(nest, fn, specs), options).lower()
