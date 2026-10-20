"""The reference a kernel is checked against: the kernel function itself, run as plain JAX.

The loop nest is ordinary Python and JAX, so running it is the specification of what the kernel
computes. The one change: `@` in the loop nest means what it means in the kernel (mxu.py), a matmul
accumulating in f32, not plain JAX's `bf16 @ bf16 -> bf16`.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable
from typing import Any

import jax

from . import mxu
from .parse import Nest

__all__ = ["make_reference"]


def make_reference(nest: Nest, fn: Callable) -> Callable[..., Any]:
    """`fn` as a jitted JAX function with `@` on the MXU, reading the module's and closure's values as they
    are now (as the kernel built now does)."""
    fdef = copy.deepcopy(nest.function)
    fdef.decorator_list = []
    at = nest.function.body.index(nest.parallel[0].node)
    fdef.body[at] = mxu.rewrite_matmuls(fdef.body[at])

    free = fn.__code__.co_freevars
    maker = ast.FunctionDef(
        name="__make__",
        args=ast.arguments([], [ast.arg(name) for name in free], None, [], [], None, []),
        body=[fdef, ast.Return(ast.Name(fdef.name, ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module([ast.copy_location(maker, fdef)], []))
    namespace = {**fn.__globals__, **mxu.FUNCTIONS}
    exec(compile(module, nest.filename, "exec"), namespace)
    cells = [cell.cell_contents for cell in fn.__closure__ or ()]
    return jax.jit(namespace["__make__"](*cells))
