"""`autokernel.tile`: block sizes declared in the function, tunable like module constants.

A block size is usually a module constant (`TM = 512`) that the tuner overrides by name. One computed in the
prologue from the input shapes can't be overridden that way: the prologue would compute it again. Declaring
it with `tile` makes it a tunable block size wherever it is computed:

    @autokernel.tpu(tune="auto")
    def matmul(x, y):
        M, K = x.shape
        TM = autokernel.tile(M, 512)  # 512 rows per tile (all of M if it is smaller) unless tuned
        ...

When the tuner tries a value for `TM`, the declaration is skipped and `TM` is that value; otherwise it is
`min(block, extent)`. The kernel, `reference` and `verify` all see the same value.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import AutoKernelError

__all__ = ["declarations", "tile"]


def tile(extent: int, block: int = 512) -> int:
    """A block size for a loop over `extent` elements: `block`, or all of `extent` when that is smaller.

    Assigned to a name in a kernel's prologue (`TM = autokernel.tile(M, 512)`), it is a block size the
    tuner can set (`tune="auto"`, or `tune={"TM": (...)}`), like a module constant."""
    for name, v in (("extent", extent), ("block", block)):
        if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1:
            raise AutoKernelError(f"autokernel.tile: {name} must be a positive int, got {v!r}")
    return int(min(block, extent))


def declarations(stmts: Sequence[ast.stmt], env: Mapping[str, Any]) -> dict[str, ast.Assign]:
    """The statements `NAME = tile(...)` among `stmts` (the callee written any way that evaluates to `tile`
    in `env`, `autokernel.tile` say), by the name they assign."""
    found = {}
    for s in stmts:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)):
            continue
        call = s.value
        if not isinstance(call, ast.Call):
            continue
        f = call.func
        if not (isinstance(f, ast.Name) and f.id == "tile") and not (isinstance(f, ast.Attribute) and f.attr == "tile"):
            continue
        expr = ast.fix_missing_locations(ast.Expression(copy.deepcopy(f)))
        try:
            callee = eval(compile(expr, "<autokernel.tile>", "eval"), dict(env))
        except Exception:
            continue
        if callee is tile:
            found[s.targets[0].id] = s
    return found
