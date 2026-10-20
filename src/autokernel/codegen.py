"""Generate a gridless Pallas TPU kernel from a `Specialization` and its `Schedule`, then compile it.

The kernel runs every step of its share of the work in one `pl.loop` and moves data with hand-written
async copies. One step:

1. decode the step: tile ordinal, phase, reduction position, first/last/empty flags, and from those the
   block index of every tile loop (computed once, shared by every copy of the step);
2. start copying the inputs of the step `buffers - 1` ahead into a ring of VMEM slots;
3. at a tile's first step: start fetching the next tile's per-tile inputs, wait for this tile's;
4. wait for this step's inputs;
5. run the user's units: a phase's leading segment on its first step, its reduction body on non-empty
   steps, and the final segment plus output stores on the tile's last step;
6. stores write each output tile into one of two VMEM slots and copy it to HBM asynchronously.

Kernels with one unmasked reduction (or none) have no phase or empty-step tests at all: every step runs
the body. On chips with two TensorCores per device the kernel gets a one-axis "parallel" grid, one
point per TensorCore, and each core runs a contiguous run of tiles.

The source is self-contained: it imports what it uses and binds the values it reads from the user's module
(constants, functions the tile code calls, each with its own module's values), its cost estimate and its
SMEM tables, so it runs on its own. Every name it defines where the user's code can see it comes from the
nest's `Namer`, so it can't hide a name of the user's; and where the user's code means something else by a
name the generated code uses (`np` for jax.numpy, a loop variable `max`), the user's code gets a fresh name.
"""

from __future__ import annotations

import ast
import builtins
import copy
import enum
import importlib
import inspect
import math
import symtable
import textwrap
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import analysis, hardware, mxu
from . import schedule as schedule_lib
from .analysis import Block, LoopBounds, Operand, Specialization, StateVar, Unit
from .errors import AutoKernelError
from .frontend import assigned_name, reads
from .schedule import COLUMN_SHIFT, FIRST, LAST, RUNTIME_TABLES, Schedule
from .views import div_src, rem_src

__all__ = [
    "FILL_DMA_DEPTH",
    "MODULES",
    "Emitter",
    "Generated",
    "assignments",
    "check_layout",
    "generate",
    "layout_error",
    "plan_fills",
    "prelude",
]

# Constant (masked-out) tiles are written with at most this many fill DMAs in flight.
FILL_DMA_DEPTH = 4

# What generated code imports, by the names it uses them under.
_IMPORTS = """\
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
"""
MODULES = {"jax": jax, "jnp": jnp, "np": np, "pl": pl, "pltpu": pltpu}
# The builtins generated code uses (the step-table builders it carries, literals, ...).
_BUILTINS = ("bool", "complex", "float", "frozenset", "int", "len", "max", "range", "set", "slice", "tuple", "zip")
# Names the generated code reads as these values: user code meaning something else by one gets a fresh name.
_RESERVED = {**MODULES, **{name: getattr(builtins, name) for name in _BUILTINS}}

_PRELUDE = """\
def {_div}(a, b):
    # a // b for the non-negative step and tile indices: one integer division, no floor-sign fix-ups.
    return jax.lax.div(a, b)


def {_rem}(a, b):
    return jax.lax.rem(a, b)
"""

_SCALAR_PRELUDE = """\
def {_index}(i, n):
    # A runtime index into a dim of size n: Python's rules, so a negative index counts from the end. One
    # outside the dim is clamped into it, so a copy never reads outside the array (that would halt the TPU
    # core); the caller reports it (`analysis.check_indices`).
    return jnp.minimum(jnp.maximum(jnp.where(i < 0, i + n, i), 0), n - 1)
"""

# JAX has no public way to make a traced value weakly typed; this is what `jax.lax` itself uses.
_WEAK = """\
def {name}(x):
    # A Python scalar's weak type, so it takes the other operand's type: `bf16_tile * i` stays bf16.
    return jax._src.lax.lax._convert_element_type(x, weak_type=True)
"""


@dataclass(frozen=True)
class Generated:
    """Generated source and the callable its wrapper defines."""

    source: str
    entry: Callable[..., Any]  # entry(*inputs, interpret=False) -> output(s)


def generate(
    spec: Specialization,
    sched: Schedule,
    *,
    buffers: int,
    matmul_precision: str,
    fills: Sequence[str],
    aliased: Sequence[str],
    vmem_limit_bytes: int,
    cost: pl.CostEstimate | None,
    trace: bool = False,
) -> Generated:
    """`fills` and `aliased` are the outputs `plan_fills` assigned to fill DMAs and to an aliased input.
    With `trace`, each step marks its waits and compute as profiler trace regions (see `profile`)."""
    emitter = Emitter(spec, sched, buffers, matmul_precision, fills, aliased, vmem_limit_bytes, cost, trace=trace)
    return emitter.emit()


def plan_fills(spec: Specialization, background_fill: bool) -> tuple[list[str], list[str]]:
    """How skipped tiles of each output get their value: (outputs written by fill DMAs, outputs aliased).

    An output with skipped tiles whose allocation is a known constant can have those tiles written by
    DMAs from the kernel (overlapping the compute); otherwise the allocation is passed in and aliased to
    the output so the skipped tiles keep its values.
    """
    skipped = bool(spec.constant_tiles) or spec.tile_mask_runtime
    if not skipped:  # outputs the tiles don't cover keep their allocation's values
        return [], [o.var for o in spec.outputs if o.partial]
    fill_ok = background_fill and not spec.tile_mask_runtime
    fills = [o.var for o in spec.outputs if fill_ok and o.fill is not None]
    return fills, [o.var for o in spec.outputs if o.var not in fills]


def check_layout(spec: Specialization, lanes: int) -> list[str]:
    """TPU tiling problems of every input and output tile."""
    problems = []
    for op in spec.operands:
        shape = op.block.kernel_shape
        full = op.block.kept(op.full_shape)
        problems += hardware.check_block(
            f"input `{op.param}[{op.block.text}]` tile {shape}", shape, full, op.dtype, lanes
        )
    for out in spec.outputs:
        shape = out.block.kernel_shape
        full = out.block.kept(tuple(out.kernel_out.shape))
        what = f"output `{out.var}[{out.block.text}]` tile {shape}"
        problems += hardware.check_block(what, shape, full, out.out.dtype, lanes)
    return problems


def layout_error(spec: Specialization, problems: list[str]) -> AutoKernelError:
    return AutoKernelError(
        f"{spec.nest.fn_name}: tiles don't satisfy TPU (Mosaic) layout rules:\n  - "
        + "\n  - ".join(problems)
        + "\nPick tile sizes that fit, or pass check_tiling=False to try anyway."
    )


# --------------------------------------------------------------------------- source helpers


def _dtype_src(dtype: Any) -> str:
    return f"jnp.{np.dtype(dtype).name}"


def _stored_dtype(sv: StateVar) -> np.dtype:
    """The dtype a state variable's scratch holds: its own, except that SMEM scalars take a 32-bit word."""
    dtype = np.dtype(sv.aval.dtype)
    if not sv.smem or dtype.itemsize >= 4:
        return dtype
    return np.dtype(np.float32 if jnp.issubdtype(dtype, jnp.floating) else np.int32)


def _scalar_src(value: Any) -> str:
    """A Python (or NumPy) number as a Python number's source: `float('nan')` for nan, and so on."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return f"complex({_scalar_src(value.real)}, {_scalar_src(value.imag)})"
    if isinstance(value, float) and not math.isfinite(value):
        return f"float('{value}')"
    return repr(value)


# Profiler trace regions a traced kernel marks in every step (`generate(trace=True)`): waiting for the step's
# inputs, the user's code, and (inside it) waiting for an output slot to be free.
WAIT, COMPUTE, OUT_WAIT = "wait", "compute", "out_wait"
REGIONS = (WAIT, COMPUTE, OUT_WAIT)


def region_name(kernel: str, what: str) -> str:
    return f"{kernel}/{what}"


def _pad_widths(shape: Sequence[int], padded: Sequence[int]) -> str:
    """`jnp.pad`'s widths growing `shape` to `padded` at the end of each dim."""
    return repr(tuple((0, p - n) for n, p in zip(shape, padded, strict=True)))


def _indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line if line else line for line in lines]


def _fn(name: str, args: str, body: list[str], when: str | None = None, loop: str | None = None) -> list[str]:
    """A nested function, optionally under `@pl.when(when)` or `@pl.loop(loop)`."""
    decorator = [f"@pl.when({when})"] if when else [f"@pl.loop({loop})"] if loop else []
    return [*decorator, f"def {name}({args}):", *_indent(body or ["pass"])]


def _block(name: str, body: list[str], when: str | None) -> list[str]:
    """`body` as a function under `@pl.when(when)`, or inline when there's nothing to test."""
    return _fn(name, "", body, when=when) if when else list(body)


def _by_phase(values: Sequence[Any]) -> str:
    """Pick `values[ph]` with a chain of selects (Mosaic can't lower jnp.select's argmax over booleans)."""
    if len(set(values)) == 1:
        return str(values[0])
    out = str(values[-1])
    for p in range(len(values) - 2, -1, -1):
        out = f"jnp.where(ph == {p}, {values[p]}, {out})"
    return out


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _split_src(name: str, x: str, quotient: str, n: int, rem: str) -> str:
    """`name = x % n` given `quotient = x // n`: a mask for powers of two, else one multiply and subtract."""
    if n == 1:
        return f"{name} = 0"
    return f"{name} = {rem_src(x, n, rem)}" if _is_pow2(n) else f"{name} = {x} - {quotient} * {n}"


def _radix(index: str, loops: Sequence[LoopBounds], k: int, div: str, rem: str) -> str:
    """Absolute block index of `loops[k]` when `index` counts the loops' blocks row-major."""
    stride = int(np.prod([lb.count for lb in loops[k + 1 :]], dtype=np.int64))
    idx = div_src(index, stride, div)
    if k:
        idx = rem_src(idx, loops[k].count, rem)
    base = loops[k].start // loops[k].step
    return f"({idx} + {base})" if base else idx


# --------------------------------------------------------------------------- values as source


def _globals_read(src: str) -> set[str]:
    """The module-level names the code `src` reads and doesn't define itself."""
    top = symtable.symtable(src, "<autokernel>", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    found: set[str] = set()

    def visit(table: symtable.SymbolTable) -> None:
        for sym in table.get_symbols():
            if sym.is_referenced() and (table is top or sym.is_global()):
                found.add(sym.get_name())
        for child in table.get_children():
            visit(child)

    visit(top)
    return found - defined


def _numeric(dtype: np.dtype) -> bool:
    """Is `dtype` a number type (NumPy's, or one of ml_dtypes' like bfloat16) or bool?"""
    return dtype == np.bool_ or bool(jnp.issubdtype(dtype, jnp.number))


def _items_src(items: Any) -> str:
    return f"[{', '.join(_items_src(v) for v in items)}]" if isinstance(items, list) else _scalar_src(items)


def _array_src(value: np.ndarray) -> str | None:
    if not _numeric(value.dtype):
        return None
    finite = not jnp.issubdtype(value.dtype, jnp.inexact) or bool(np.isfinite(value).all())
    items = repr(value.tolist()) if finite else _items_src(value.tolist())  # nan and inf as float(...)
    out = f"np.array({items}, {_dtype_src(value.dtype)})"
    return out if value.size else f"{out}.reshape({value.shape})"


def _number_type_src(dtype: np.dtype) -> str:
    """The NumPy scalar type of a number dtype (`jnp.float32(x)` would make a JAX array instead)."""
    if getattr(np, dtype.name, None) is dtype.type:
        return f"np.{dtype.name}"
    return f"np.dtype({_dtype_src(dtype)}).type"  # bfloat16 & co. (ml_dtypes)


def _literal(value: Any) -> str | None:
    """`value` as a Python expression over the generated code's imports, or None if it has none.

    Types are matched exactly: a subclass's value (a NamedTuple's, an IntEnum's) isn't its base's literal.
    """
    kind = type(value)
    if value is None or kind in (bool, int, str, bytes):
        return repr(value)
    if kind in (float, complex):
        return _scalar_src(value)
    if kind in (tuple, list, set, frozenset):
        items = [_literal(v) for v in value]
        if None in items:
            return None
        if kind is tuple:
            return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
        text = ", ".join(items)
        return f"[{text}]" if kind is list else f"{kind.__name__}([{text}])"
    if kind is dict:
        pairs = [(_literal(k), _literal(v)) for k, v in value.items()]
        if any(k is None or v is None for k, v in pairs):
            return None
        return "{" + ", ".join(f"{k}: {v}" for k, v in pairs) + "}"
    if kind is slice:
        parts = [_literal(value.start), _literal(value.stop), _literal(value.step)]
        return None if None in parts else f"slice({', '.join(parts)})"
    if isinstance(value, np.dtype):
        return f"np.dtype({_dtype_src(value)})" if _numeric(value) else None
    if isinstance(value, np.generic) and kind is value.dtype.type and _numeric(value.dtype):
        return f"{_number_type_src(value.dtype)}({_scalar_src(value)})"  # a NumPy scalar, strongly typed
    if kind is np.ndarray:
        return _array_src(value)
    if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer):
        array = _array_src(np.asarray(value))
        return None if array is None else f"jnp.asarray({array})"
    if kind is pl.CostEstimate:
        fields = ["flops", "transcendentals", "bytes_accessed"]
        fields += ["remote_bytes_transferred"] if value.remote_bytes_transferred else []
        return f"pl.CostEstimate({', '.join(f'{f}={getattr(value, f)!r}' for f in fields)})"
    return None


def _import(name: str, value: Any) -> list[str] | None:
    """Statements binding `name` to `value` by importing it (a module, something a module defines by that
    name, or a member of an enum class a module defines), if there are any."""
    if isinstance(value, types.ModuleType):
        if "." in value.__name__:
            parent, attr = value.__name__.rsplit(".", 1)
            return [f"from {parent} import {attr}" + (f" as {name}" if attr != name else "")]
        return [f"import {value.__name__}" + (f" as {name}" if value.__name__ != name else "")]
    if isinstance(value, enum.Enum):
        cls = _import(name, type(value)) if getattr(type(value), value.name, None) is value else None
        return None if cls is None else [*cls, f"{name} = {name}.{value.name}"]  # the class, then its member
    attr = getattr(value, "__name__", None)
    if not isinstance(attr, str) or not attr.isidentifier():
        return None
    # The public module that has it when one of the generated code's does (`jnp.maximum`, not jax._src...).
    module = getattr(value, "__module__", None)
    homes = [m.__name__ for m in MODULES.values()] + ["jax.nn", "math"]
    importable = isinstance(module, str) and module != "__main__" and all(p.isidentifier() for p in module.split("."))
    for home in [*homes, *([module] if importable else [])]:
        try:
            found = getattr(importlib.import_module(home), attr, None)
        except ImportError:
            continue
        if found is value:
            return [f"from {home} import {attr}" + (f" as {name}" if attr != name else "")]
    return None


@dataclass(frozen=True)
class _Function:
    """A user function the kernel calls, as source: `fn` is `value` unwrapped, `source` its definition and
    `reads` the names (and values) it reads from its module."""

    value: Any
    fn: types.FunctionType
    source: str
    reads: dict[str, Any]


def _function_source(value: Any) -> _Function | None:
    """A Python function with source of its own, no closure and no `global` statements, as source."""
    fn = inspect.unwrap(value)
    if not isinstance(fn, types.FunctionType) or fn.__code__.co_freevars or fn.__name__ == "<lambda>":
        return None
    # Its definition binds its name (in its module, if defined there) to `value` (decorated as the source
    # says), or to the undecorated function `value` wraps (`g = jax.jit(f)`), which runs the same code.
    bound = fn.__globals__.get(fn.__name__, value)
    if bound is not value and bound is not fn:
        return None
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != fn.__name__:
        return None
    fdef = tree.body[0]
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(fdef)):
        return None
    changed = False
    if bound is fn and fdef.decorator_list:  # decorators that returned the function itself (registries, ...)
        fdef.decorator_list, changed = [], True
    # Annotations do nothing when it runs: dropped, so what they name needn't have a source.
    args = fdef.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None and arg.annotation is not None:
            arg.annotation, changed = None, True
    if fdef.returns is not None:
        fdef.returns, changed = None, True
    src = ast.unparse(fdef) if changed else src.rstrip()
    # What it reads from its module: the names its definition (decorators, defaults, body) loads and doesn't
    # bind itself, scope by scope (an attribute's name isn't a read).
    reads = {n: fn.__globals__[n] for n in sorted(_globals_read(src)) if n in fn.__globals__ and not n.startswith("__")}
    return _Function(value, fn, src, reads)


class _Rename(ast.NodeTransformer):
    """Renames names (and lambdas' parameters) of user code: `names[old]` is the new name."""

    def __init__(self, names: dict[str, str]):
        self.names = names

    def visit_Name(self, n: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(self.names[n.id], n.ctx), n) if n.id in self.names else n

    def visit_arg(self, n: ast.arg) -> ast.AST:
        n.arg = self.names.get(n.arg, n.arg)
        return n


class Emitter:
    """Generates one nest's kernel: as a whole `pallas_call` (`emit`), or as one part of a fused kernel
    (`part` and `prepare`, used by `fusion`), where `dependent` names the inputs an earlier part writes."""

    def __init__(
        self, spec, sched, buffers, precision, fills, aliased, vmem_limit, cost, dependent=frozenset(), trace=False
    ):
        self.spec, self.sched, self.nest = spec, sched, spec.nest
        self.trace = trace  # mark each step's waits and compute as profiler trace regions
        self.buffers, self.precision, self.vmem_limit, self.cost = buffers, precision, vmem_limit, cost
        self.dependent = frozenset(dependent)
        self.names = spec.names.copy()  # emitting twice gives the same names
        self.P = len(spec.phases)
        self.cores = sched.cores
        # One unmasked reduction (or none): every step belongs to phase 0 and none is empty.
        self.trivial = sched.mode == "uniform" and self.P <= 1
        self.used_names: set[str] = set()  # names the user's code (tile code, masks, allocations) reads
        self.weak: str | None = None  # the weak-type helper's name, once something needs it
        self._param_names: dict[str, str] | None = None
        self._tables_local: str | None = None
        self._fresh: dict[str, str] = {}  # generated identifiers, by what they'd be called without the user's
        self.arrays: dict[str, np.ndarray] = {}  # constant arrays the runtime tables are built from
        self.exported: dict[str, Any] = {}  # a fused part's user functions, by their module-level names
        self.module_defs: list[str] = []  # a fused part's source for module level: its user functions
        self._fn_exports: dict[int, str] = {}  # id(user function) -> the module-level name defined for it
        self.renamed = self._renames()
        self._names(fills, aliased)
        self._tables()

    def _renames(self) -> dict[str, str]:
        """The user's names that would hide a name the generated code uses (`np` meaning jax.numpy, a local
        `max`, ...), and the fresh names the generated code gives them (bound to the user's values)."""
        fdef = self.nest.fdef
        seen = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)}
        seen |= {a.arg for a in ast.walk(fdef) if isinstance(a, ast.arg)}
        bound = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
        bound |= {a.arg for a in ast.walk(fdef) if isinstance(a, ast.arg)}
        renamed = {}
        for name in sorted(seen & _RESERVED.keys()):
            if name in bound or self.spec.consts.get(name, _RESERVED[name]) is not _RESERVED[name]:
                renamed[name] = self.names.fresh(name)
        return renamed

    def _user(self, name: str) -> str:
        """What the generated code calls the user's name `name`."""
        return self.renamed.get(name, name)

    def _user_code(self, node: ast.AST) -> ast.AST:
        """A copy of the user's code `node` with the names it uses as the generated code calls them."""
        return _Rename(self.renamed).visit(copy.deepcopy(node))

    def _n(self, base: str) -> str:
        """The identifier generated code uses for `base`: `base`, unless that is one of the user's names."""
        if base not in self._fresh:
            self._fresh[base] = self.names.fresh(base)
        return self._fresh[base]

    def _div_src(self, x: str, n: int) -> str:
        return div_src(x, n, self._n("_div"))

    def _rem_src(self, x: str, n: int) -> str:
        return rem_src(x, n, self._n("_rem"))

    def _names(self, fills: Sequence[str], aliased: Sequence[str]) -> None:
        """Identifiers the generated code binds where the user's code can see them."""
        spec, names = self.spec, self.names
        self.fills = {var: names.fresh(f"{var}_fill") for var in fills}
        self.aliased = {var: names.fresh(f"{var}_init") for var in aliased}
        self.v = {
            k: names.fresh(k)
            for k in (
                "s",
                "o",
                "ph",
                "d",
                "first",
                "last",
                "empty",
                "blocks",
                "core",
                "s_begin",
                "s_end",
                "o_begin",
                "o_end",
            )
        }
        self.blk = {var: names.fresh(f"{var}_blk") for var in spec.loops}
        self.streamed = [op for op in spec.operands if op.phase is not None]
        self.per_tile = [op for op in spec.operands if op.phase is None]
        self.hbm: dict[str, str] = {}
        for op in spec.operands:
            self.hbm.setdefault(op.param, names.fresh(f"{op.param}_hbm"))
        self.slots = {op.ref: names.fresh(f"{op.ref}_slots") for op in spec.operands}
        self.sems = {op.ref: names.fresh(f"{op.ref}_sems") for op in spec.operands}
        self.out_hbm = {o.var: names.fresh(f"{o.var}_hbm") for o in spec.outputs}
        self.out_slots = {o.var: names.fresh(f"{o.ref}_slots") for o in spec.outputs}
        self.out_sems = {o.var: names.fresh(f"{o.ref}_sems") for o in spec.outputs}
        # Runtime scalars live in SMEM: integers in one int32 array, floats in one f32 array.
        avals = dict(zip(spec.nest.params, spec.specs, strict=True))
        self.int_scalars = [p for p in spec.scalars if jnp.issubdtype(avals[p].dtype, jnp.integer)]
        self.float_scalars = [(p, bool(avals[p].weak_type)) for p in spec.scalars if p not in self.int_scalars]
        self.scalars_ref = names.fresh("scalars") if self.int_scalars else None
        self.fscalars_ref = names.fresh("fscalars") if self.float_scalars else None
        self.indices: dict[tuple[str, int], str] = {}  # (runtime index expression, dim size) -> its variable
        for op in spec.operands:
            for d, expr in enumerate(op.block.dynamic):
                if expr is not None and (expr, op.full_shape[d]) not in self.indices:
                    self.indices[(expr, op.full_shape[d])] = names.fresh(f"{op.param}_index{d}")
        # A fused kernel's part: its pieces, run in turn by the fused kernel.
        self.pieces = {
            k: names.fresh(f"_{k}") for k in ("early", "with_scratch", "late", "run", "drain", "hook", "hook_at")
        }

    def _tables(self) -> None:
        """Static SMEM tables (embedded as constants of the wrapper) and the runtime tables' names."""
        spec, sched, names = self.spec, self.sched, self.names
        self.tables: dict[str, np.ndarray] = {}  # kernel parameter -> contents
        self.table_consts: dict[str, str] = {}  # kernel parameter -> the wrapper's constant

        def table(base: str, value: np.ndarray) -> str:
            name = names.fresh(base)
            self.tables[name] = np.asarray(value, np.int32)
            self.table_consts[name] = names.fresh(name.upper())
            return name

        self.tile_tables: dict[str, str] = {}
        if spec.nest.guard is not None and not spec.tile_mask_runtime:  # a static tile mask: list the tiles
            for n, lb in enumerate(spec.parallel):
                self.tile_tables[lb.var] = table(f"{lb.var}_blocks", [t[n] for t in spec.tiles])
        if self.fills:
            self.fill_tables = {
                lb.var: table(f"{lb.var}_fill_blocks", [t[n] for t in spec.constant_tiles])
                for n, lb in enumerate(spec.parallel)
            }
            n_tiles, n_fill = len(spec.tiles), len(spec.constant_tiles)
            # Tile ordinal o writes the fills [fill_start[o], fill_start[o + 1]).
            self.fill_start = table("fill_start", [t * n_fill // n_tiles for t in range(n_tiles + 1)])
        self.step_tables: dict[str, str] = {}
        if sched.mode == "table":
            self.step_tables = {key: table(key, value) for key, value in sched.tables.items()}
        elif sched.mode == "runtime":
            self.step_tables = {k: names.fresh(k) for k in RUNTIME_TABLES}
        self.bounds_table = None
        if sched.mode != "runtime" and self.cores > 1:  # [step bound per core..., tile bound per core...]
            steps = [b for b, _ in sched.core_steps] + [sched.core_steps[-1][1]]
            tiles = [b for b, _ in sched.core_tiles] + [sched.core_tiles[-1][1]]
            self.bounds_table = table("core_bounds", steps + tiles)

    # ------------------------------------------------------------------ coordinates

    def _tile_block(self, var: str, o: str) -> str:
        """Absolute block index of parallel loop `var` for tile ordinal `o`."""
        if var in self.tile_tables:
            return f"{self.tile_tables[var]}[{o}]"
        t = f"{self.step_tables['tile_ids']}[{o}]" if self.spec.tile_mask_runtime else o
        k = [lb.var for lb in self.spec.parallel].index(var)
        return _radix(t, self.spec.parallel, k, self._n("_div"), self._n("_rem"))

    def _reduction_block(self, var: str, d: str) -> str:
        """Absolute block index of reduction loop `var` at flattened phase position `d`."""
        phase = next(p for p in self.spec.phases if any(lb.var == var for lb in p.loops))
        k = [lb.var for lb in phase.loops].index(var)
        return _radix(d, phase.loops, k, self._n("_div"), self._n("_rem"))

    def _blocks(self) -> list[str]:
        """`_blocks(o, d)`: the block index of every tile loop, for the copies and the loop values."""
        body = []
        for lb in self.spec.parallel:
            body.append(f"{self.blk[lb.var]} = {self._tile_block(lb.var, 'o')}")
        for phase in self.spec.phases:
            for lb in phase.loops:
                body.append(f"{self.blk[lb.var]} = {self._reduction_block(lb.var, 'd')}")
        body.append(f"return {self._blocks_tuple(self.blk)}")
        return _fn(self._n("_blocks"), "o, d", body)

    def _blocks_tuple(self, names: dict[str, str]) -> str:
        items = [names.get(var, "_") for var in self.spec.loops]
        return f"({items[0]},)" if len(items) == 1 else ", ".join(items)

    def _slices(self, block: Block, blocks: dict[str, str], fill: str | None = None, full: tuple[int, ...] = ()) -> str:
        """The HBM window of a block: `pl.ds` per tiled dim, a plain index per squeezed dim."""
        parts = []
        for d, (var, size, c, sq) in enumerate(zip(block.loops, block.shape, block.const, block.squeezed, strict=True)):
            expr = block.dynamic_index(d)
            if expr is not None:  # a runtime index, read from SMEM at the start of the kernel
                start = self.indices[(expr, full[d])]
            elif var is None:
                start = str(c * size)
            elif self.spec.loops[var].count == 1:
                # A loop of one block: a constant offset, so Mosaic sees e.g. a window spanning whole rows and
                # copies it in one go (a traced 0 offset cost a single-step v5e matmul 2%).
                lb = self.spec.loops[var]
                start = str(lb.start // lb.step * size)
            else:
                idx = f"{self.fill_tables[var]}[{fill}]" if fill is not None else blocks[var]
                start = idx if size == 1 else f"{idx} * {size}"
            parts.append(start if sq else f"pl.ds({start}, {size})")
        return ", ".join(parts)

    def _unpack_blocks(self, block: Block) -> list[str]:
        """Bind the block indices a copy needs from the `blocks` tuple."""
        wanted = {var: self.blk[var] for var in block.loops if var}
        return [f"{self._blocks_tuple(wanted)} = blocks"] if wanted else []

    def _loop_values(self) -> list[str]:
        """The user's loop variables, for code that reads them as values."""
        lines = []
        for var in sorted(self.used_names & set(self.spec.loops)):
            step = self.spec.loops[var].step
            value = f"{self.blk[var]} * {step}" if step != 1 else self.blk[var]
            lines.append(f"{self._user(var)} = {self._weak()}({value})")
        return lines

    def _weak(self) -> str:
        """The prelude helper giving a value a Python scalar's weak type (`bf16_tile * i` stays bf16)."""
        if self.weak is None:
            self.weak = self._n("_weak")
        return self.weak

    # ------------------------------------------------------------------ step decoding

    def _decode(self) -> list[str]:
        """`_decode(s)` -> (tile ordinal, phase, reduction position, first, last, empty)."""
        sched = self.sched
        if sched.mode == "uniform":
            per = sched.steps_per_tile
            body = [f"o = {self._div_src('s', per)}", _split_src("d", "s", "o", per, self._n("_rem"))]
            if self.P <= 1:
                body += ["first = d == 0", f"last = d == {per - 1}", "return o, 0, d, first, last, False"]
            else:
                offsets = np.cumsum([0, *sched.phase_steps]).tolist()
                body.append("ph = " + " + ".join(f"jnp.asarray(d >= {off}, jnp.int32)" for off in offsets[1:-1]))
                body += [f"d = d - {_by_phase(offsets[:-1])}", "first = d == 0"]
                body.append(f"last = d == {_by_phase([k - 1 for k in sched.phase_steps])}")
                body.append("return o, ph, d, first, last, False")
            return _fn(self._n("_decode"), "s", body)
        t = self.step_tables
        if sched.mode == "table":
            ps, pt = sched.pattern_steps, sched.pattern_tiles
            if sched.steps != ps:
                body = [f"rep = {self._div_src('s', ps)}", _split_src("r", "s", "rep", ps, self._n("_rem"))]
                body.append(f"o = rep * {pt} + {t['step_tile']}[r]")
            else:
                body = ["r = s", f"o = {t['step_tile']}[r]"]
            body.append(f"code = {t['step_code']}[r]")
        else:
            body = [f"o = {t['step_tile']}[s]", f"code = {t['step_code']}[s]"]
        body += [f"c = code >> {COLUMN_SHIFT}", f"first = (code & {FIRST}) != 0", f"last = (code & {LAST}) != 0"]
        if self.P <= 1:
            k = sched.phase_steps[0] if self.P else 0
            body += ["d = c", f"empty = d == {k}" if self.P else "empty = False", "return o, 0, d, first, last, empty"]
        else:
            offsets = sched.column_offsets()
            body.append("ph = " + " + ".join(f"jnp.asarray(c >= {off}, jnp.int32)" for off in offsets[1:]))
            body.append(f"d = c - {_by_phase(offsets)}")
            body.append(f"empty = d == {_by_phase(sched.phase_steps)}")
            body.append("return o, ph, d, first, last, empty")
        return _fn(self._n("_decode"), "s", body)

    def _bounds(self) -> list[str]:
        """This core's [begin, end) steps and tile ordinals."""
        v, sched = self.v, self.sched
        table = self.step_tables["bounds"] if sched.mode == "runtime" else self.bounds_table
        if table is None:
            return [
                f"{v['s_begin']}, {v['s_end']} = 0, {sched.steps}",
                f"{v['o_begin']}, {v['o_end']} = 0, {sched.tiles}",
            ]
        n = self.cores
        return [
            f"{v['core']} = {'pl.program_id(0)' if n > 1 else '0'}",
            f"{v['s_begin']}, {v['s_end']} = {table}[{v['core']}], {table}[{v['core']} + 1]",
            f"{v['o_begin']}, {v['o_end']} = {table}[{n + 1} + {v['core']}], {table}[{n + 2} + {v['core']}]",
        ]

    def _in_phase(self, p: int, ph: str) -> str | None:
        """The test for 'this step belongs to phase p', or None when there is at most one phase."""
        return None if self.P <= 1 else f"{ph} == {p}"

    def _phase_test(self, p: int, ph: str, empty: str) -> str | None:
        """The test for 'this step is a non-empty step of phase p', or None when every step is."""
        if self.trivial:
            return None
        in_phase = self._in_phase(p, ph)
        if self.sched.mode == "uniform":  # no masks: no empty steps
            return in_phase
        return f"~{empty}" if in_phase is None else f"({in_phase}) & ~{empty}"

    def _flag_test(self, p: int, ph: str, flag: str) -> str:
        """The test for 'this step is phase p's step with `flag` (first / last) set'."""
        in_phase = self._in_phase(p, ph)
        return flag if in_phase is None else f"({in_phase}) & {flag}"

    # ------------------------------------------------------------------ copies

    def _copy_fn(self, op: Operand) -> str:
        return self._n(f"_{op.ref}_copy")

    def _out_copy_fn(self, out: Any) -> str:
        return self._n(f"_{out.ref}_copy")

    def _fill_copy_fn(self, var: str) -> str:
        return self._n(f"_{var}_fill_copy")

    def _window(self, op: Operand) -> str:
        """The window of a view's base array that the view's tile `op` is (see `views`)."""
        starts = []
        for d, (var, size, c) in enumerate(zip(op.block.loops, op.block.shape, op.block.const, strict=True)):
            expr = op.block.dynamic_index(d)
            if expr is not None:
                starts.append((self.indices[(expr, op.full_shape[d])], 1, 0))
            elif var is None:
                starts.append((None, 0, c * size))
            else:
                starts.append((self.blk[var], size, 0))
        return ", ".join(op.window.render(starts, self._n("_div"), self._n("_rem")))

    def _copies(self, ops: Sequence[Operand] | None = None, outputs: bool = True) -> list[str]:
        """The copy functions of `ops` (default: every input) and, with `outputs`, of the outputs and fills."""
        lines = []
        for op in self.spec.operands if ops is None else ops:
            if op.window is not None:  # a view: copy the window of its base array
                src = f"{self.hbm[op.param]}.at[{self._window(op)}]"
            else:
                src = f"{self.hbm[op.param]}.at[{self._slices(op.block, self.blk, full=op.full_shape)}]"
            dst = f"{self.slots[op.ref]}.at[slot], {self.sems[op.ref]}.at[slot]"
            body = [
                *self._unpack_blocks(op.block),
                f"src = {src}  # {op.param}[{op.block.text}]",
                f"return pltpu.make_async_copy(src, {dst})",
            ]
            lines += [*_fn(self._copy_fn(op), "blocks, slot", body), ""]
        for out in self.spec.outputs if outputs else []:
            dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, self.blk)}]"
            src = f"{self.out_slots[out.var]}.at[slot], dst, {self.out_sems[out.var]}.at[slot]"
            body = [
                *self._unpack_blocks(out.block),
                f"dst = {dst}  # {out.var}[{out.block.text}]",
                f"return pltpu.make_async_copy({src})",
            ]
            lines += [*_fn(self._out_copy_fn(out), "blocks, slot", body), ""]
            if out.var in self.fills:
                fill = self.fills[out.var]
                dst = f"{self.out_hbm[out.var]}.at[{self._slices(out.block, {}, fill='c')}]"
                body = [f"dst = {dst}", f"return pltpu.make_async_copy({fill}, dst, {self._n(f'{fill}_sem')})"]
                lines += [*_fn(self._fill_copy_fn(out.var), "c", body), ""]
        return lines

    # ------------------------------------------------------------------ user code -> kernel code

    def _rewrite(self, node: ast.AST, unit: Unit | None) -> ast.AST:
        """User code of `unit` (None: the stores) as kernel code: inputs become their tile refs, state variables
        their scratch refs, `@` an MXU matmul, and finished reduction loops' variables their last values."""
        spec, params, state = self.spec, self.nest.params, self.spec.state
        phase = unit.phase if unit is not None else None
        weak = self._weak
        calls = {call: self._n(name) for name, call in mxu.CALLS.items()}  # `@` -> this kernel's MXU helpers
        renamed = self.renamed

        class Rewriter(_Rename):
            def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
                if isinstance(n.value, ast.Name) and n.value.id in params:
                    ref = spec.operand(n, phase).ref
                    return ast.copy_location(ast.parse(f"{ref}[...]", mode="eval").body, n)
                return self.generic_visit(n)

            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in calls:
                    return ast.copy_location(ast.Name(calls[n.id], n.ctx), n)
                if n.id not in state:
                    return super().visit_Name(n)
                sv = state[n.id]
                index = ast.Constant(0) if sv.smem else ast.Constant(...)
                ref = ast.copy_location(ast.Subscript(ast.Name(sv.ref, ast.Load()), index, n.ctx), n)
                if not isinstance(n.ctx, ast.Load):
                    return ref
                if _stored_dtype(sv) != sv.aval.dtype:  # back from its 32-bit SMEM word
                    ref = ast.parse(f"{ast.unparse(ref)}.astype({_dtype_src(sv.aval.dtype)})", mode="eval").body
                if sv.aval.weak_type:  # it holds Python scalars: read it as one
                    ref = ast.Call(ast.Name(weak(), ast.Load()), [ref], [])
                return ast.copy_location(ref, n)

        code = spec.unit_code(node, unit)
        self.used_names |= {n.id for n in ast.walk(code) if isinstance(n, ast.Name)}
        return Rewriter(renamed).visit(code)

    def _statements(self, unit: Unit) -> list[str]:
        lines = []
        accumulators = {id(a.update): a for a in self.spec.accumulators}
        skipped = {id(a.init) for a in self.spec.accumulators}
        for original in unit.statements:
            if id(original) in skipped:  # an accumulator's zeros: its first step stores instead of adding
                continue
            acc = accumulators.get(id(original))
            if acc is not None:
                lines += self._accumulate(acc, original, unit)
                continue
            lines.append(self._statement(original, original, unit))
        return lines

    def _statement(self, s: ast.stmt, original: ast.stmt, unit: Unit) -> str:
        """`s` (the user's statement `original`, or one standing in for it) as kernel code."""
        target = assigned_name(s)
        if isinstance(s, ast.AugAssign) and target in self.spec.state:  # x += v  ->  x = x + v, on the ref
            value = ast.BinOp(ast.Name(target, ast.Load()), s.op, s.value)
            s = ast.copy_location(ast.Assign([ast.Name(target, ast.Store())], value), s)
        out = self._rewrite(s, unit)
        dtypes = [self.spec.casts.get(id(original))]  # to the state variable's type
        sv = self.spec.state.get(target)
        if sv is not None and _stored_dtype(sv) != sv.aval.dtype:
            dtypes.append(_stored_dtype(sv))  # then into its 32-bit SMEM word
        for dtype in dtypes:
            if dtype is not None and isinstance(out, ast.Assign):
                value = f"jnp.asarray({ast.unparse(out.value)}, {_dtype_src(dtype)})"
                out.value = ast.parse(value, mode="eval").body
        return ast.unparse(out)

    def _accumulate(self, acc: Any, original: ast.stmt, unit: Unit) -> list[str]:
        """`var += expr` on a zero-initialised accumulator in a tile of one step: just `var = expr`."""
        aval = self.spec.avals[id(original)]
        # `zeros + expr` has the sum's type and shape (`expr` may broadcast, like a (rows, 1) column), and is
        # weakly typed when both are (`acc = 0.0; acc += 0.1 * i` holds a Python float).
        value = f"jnp.asarray({ast.unparse(self._rewrite(acc.expr, unit))}, {_dtype_src(aval.dtype)})"
        value = f"jnp.broadcast_to({value}, {tuple(aval.shape)})"
        var = self._user(acc.var)
        return [f"{var} = {self._weak()}({value})" if aval.weak_type else f"{var} = {value}"]

    def _stores(self) -> list[str]:
        v, lines = self.v, []
        for out in self.spec.outputs:
            store = next(st for st in self.nest.stores if st.out == out.var)
            value = f"jnp.asarray({ast.unparse(self._rewrite(store.value, None))}).astype({out.ref}.dtype)"
            if out.var in self.spec.broadcast_stores:  # `.at[...].set(v)` broadcasts v to the tile
                value = f"jnp.broadcast_to({value}, {out.ref}.shape)"
            copy_fn = self._out_copy_fn(out)
            lines += [
                f"{out.ref} = {self.out_slots[out.var]}.at[{v['o']} & 1]",
                f"{out.ref}[...] = {value}",
                f"{copy_fn}({v['blocks']}, {v['o']} & 1).start()",
            ]
        return lines

    def _free_output_slots(self) -> list[str]:
        """Wait for the copies that last used this tile's output slots (two tiles ago) to leave.

        Done before the tile's last step runs any of the user's code: a wait is a barrier Mosaic schedules
        nothing across, so a value computed before it and stored after it (the tile's matmul) would be parked
        in VMEM and read back (on a v5e, 2% of a matmul).
        """
        v, blocks = self.v, self._n("_blocks")
        waits = [
            f"{self._out_copy_fn(out)}({blocks}({v['o']} - 2, 0), {v['o']} & 1).wait()" for out in self.spec.outputs
        ]
        waits = self._region(OUT_WAIT, waits)
        return _fn(self._n("_free_output_slots"), "", waits, when=f"{v['o']} - 2 >= {v['o_begin']}")

    def _units(self) -> list[str]:
        """The user's code for one step, guarded by the step's phase and flags."""
        v, P, units = self.v, self.P, self.spec.units
        segments = [u for u in units if u.kind == "segment"]
        bodies = [u for u in units if u.kind == "body"]
        fill_issue = self._fill_issue()
        final = self._statements(segments[-1]) + self._stores()
        free = self._free_output_slots()
        if P == 0:
            return free + fill_issue + final
        if self.spec.single_step:  # the tile's one step: its code in order, values as plain locals
            return free + fill_issue + self._statements(segments[0]) + self._statements(bodies[0]) + final
        lines = _block(self._n("_last_step"), free, self._flag_test(P - 1, v["ph"], v["last"]))
        for p in range(P):
            lead = (fill_issue if p == 0 else []) + self._statements(segments[p])
            if lead:
                lines += _fn(self._n(f"_segment_{p}"), "", lead, when=self._flag_test(p, v["ph"], v["first"]))
            body = self._statements(bodies[p])
            lines += _block(self._n(f"_body_{p}"), body, self._phase_test(p, v["ph"], v["empty"]))
        return lines + _fn(self._n("_finish_tile"), "", final, when=self._flag_test(P - 1, v["ph"], v["last"]))

    # ------------------------------------------------------------------ masked-out tiles

    def _fill_issue(self) -> list[str]:
        v, lines = self.v, []
        for var in self.fills:
            limit = _fn(
                self._n("_limit_in_flight"),
                "",
                [f"{self._fill_copy_fn(var)}(0).wait()"],
                when=f"c - {self.fill_start}[{v['o_begin']}] >= {FILL_DMA_DEPTH}",
            )
            lines += _fn(
                self._n(f"_start_{var}_fills"),
                "c",
                [*limit, f"{self._fill_copy_fn(var)}(c).start()"],
                loop=f"{self.fill_start}[{v['o']}], {self.fill_start}[{v['o']} + 1]",
            )
        return lines

    def _fill_drain(self) -> list[str]:
        v, lines = self.v, []
        for var in self.fills:
            issued = f"{self.fill_start}[{v['o_end']}] - {self.fill_start}[{v['o_begin']}]"
            lines += _fn(
                self._n(f"_drain_{var}_fills"),
                "_",
                [f"{self._fill_copy_fn(var)}(0).wait()"],
                loop=f"0, jnp.minimum({issued}, {FILL_DMA_DEPTH})",
            )
        return lines

    # ------------------------------------------------------------------ kernel

    def _by_phase_ops(self, ops: Sequence[Operand]) -> list[tuple[int, list[Operand]]]:
        return [(p, [op for op in ops if op.phase == p]) for p in range(self.P) if any(op.phase == p for op in ops)]

    def _fetch_ahead(self, step: str, name: str, ops: Sequence[Operand] | None = None) -> list[str]:
        """Start copying step `step`'s streamed inputs (or just `ops`) into their ring slots."""
        slot = self._rem_src(f"({step})", self.buffers)
        body = [f"o2, ph2, d2, _, _, empty2 = {self._n('_decode')}({step})", f"blocks2 = {self._n('_blocks')}(o2, d2)"]
        for p, phase_ops in self._by_phase_ops(self.streamed if ops is None else ops):
            starts = [f"{self._copy_fn(op)}(blocks2, {slot}).start()" for op in phase_ops]
            body += _block(self._n(f"_fetch_phase_{p}"), starts, self._phase_test(p, "ph2", "empty2"))
        return _fn(self._n(name), "", body, when=f"{step} < {self.v['s_end']}")

    def _step(self, hook: bool = False) -> list[str]:
        v, N, blocks = self.v, self.buffers, self._n("_blocks")
        slot = self._rem_src(v["s"], N)
        lines = [
            f"{v['o']}, {v['ph']}, {v['d']}, {v['first']}, {v['last']}, {v['empty']} = {self._n('_decode')}({v['s']})",
            f"{v['blocks']} = {blocks}({v['o']}, {v['d']})",
        ]
        if hook:  # once this nest's own copies are all issued: start the next nest's first copies
            hook_at, call = self.pieces["hook_at"], f"{self.pieces['hook']}()"
            lines += _fn(self._n("_start_next_nest"), "", [call], when=f"{v['s']} == {hook_at}")
        if self.streamed:  # 1. prefetch the step N - 1 ahead
            lines += self._fetch_ahead(f"{v['s']} + {N - 1}", "_prefetch")
        if self.per_tile:  # 2. per-tile inputs, at a tile's first step
            next_tile = [
                f"{self._copy_fn(op)}({blocks}({v['o']} + 1, 0), ({v['o']} + 1) & 1).start()" for op in self.per_tile
            ]
            body = [
                *_fn(self._n("_next_tile_inputs"), "", next_tile, when=f"{v['o']} + 1 < {v['o_end']}"),
                *[f"{self._copy_fn(op)}({v['blocks']}, {v['o']} & 1).wait()" for op in self.per_tile],
            ]
            # A tile of one step is always at its first step: no test (Mosaic schedules unguarded waits better).
            guard = self._flag_test(0, v["ph"], v["first"]) if self.P and not self.spec.single_step else None
            lines += self._region(WAIT, _block(self._n("_tile_inputs"), body, guard))
        for p, ops in self._by_phase_ops(self.streamed):  # 3. this step's inputs
            waits = [f"{self._copy_fn(op)}({v['blocks']}, {slot}).wait()" for op in ops]
            lines += self._region(
                WAIT, _block(self._n(f"_wait_phase_{p}"), waits, self._phase_test(p, v["ph"], v["empty"]))
            )
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{slot}]" for op in self.streamed]
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{v['o']} & 1]" for op in self.per_tile]
        units = self._units()  # 4. the user's code
        values = self._loop_values()
        if values:
            lines.append(f"{self._blocks_tuple(self.blk)} = {v['blocks']}")
        return lines + values + self._region(COMPUTE, units)

    def _region(self, what: str, lines: list[str]) -> list[str]:
        """`lines` as a profiler trace region named `<kernel>/<what>` when tracing (see `profile`)."""
        if not self.trace or not lines:
            return lines
        return [f"with jax.named_scope({region_name(self.nest.fn_name, what)!r}):", *_indent(lines)]

    def _arg_groups(self) -> dict[str, list[str]]:
        """The kernel's parameters by kind, in order: SMEM inputs, HBM inputs, aliased inputs, outputs, scratch."""
        runtime = [self.step_tables[k] for k in RUNTIME_TABLES] if self.sched.mode == "runtime" else []
        return {
            "smem": [*self.tables, *runtime, *(r for r in (self.scalars_ref, self.fscalars_ref) if r)],
            "hbm": list(self.hbm.values()),
            "aliased": list(self.aliased.values()),
            "outputs": list(self.out_hbm.values()),
            "scratch": [name for names, _, _ in self._scratch_buffers() for name in names],
        }

    def _kernel_args(self) -> list[str]:
        return [a for group in self._arg_groups().values() for a in group]

    def _scalar_reads(self) -> list[str]:
        """Bind the runtime scalars (as the user's names) and the runtime indices computed from them."""
        # Integers as weak int32s, whatever their type: a narrow one holds its 32-bit value (`analysis` rejects
        # arithmetic that would wrap in its own type; comparisons, min and max are exact in int32).
        weak, lines = self._weak, []
        for n, p in enumerate(self.int_scalars):
            lines.append(f"{self._user(p)} = {weak()}({self.scalars_ref}[{n}])")
        for n, (p, is_weak) in enumerate(self.float_scalars):
            value = f"{self.fscalars_ref}[{n}]"
            lines.append(f"{self._user(p)} = {weak()}({value})" if is_weak else f"{self._user(p)} = {value}")
        for (expr, size), name in self.indices.items():
            expr = ast.unparse(self._user_code(ast.parse(expr, mode="eval"))) if self.renamed else expr
            lines.append(f"{name} = {self._n('_index')}({expr}, {size})")
        return lines

    def _setup(self) -> list[str]:
        """The kernel's definitions: this core's bounds, the runtime scalars, step decoding and copies."""
        scalar_reads = self._scalar_reads()
        lines = [*self._bounds(), *(["", *scalar_reads] if scalar_reads else []), ""]
        return [*lines, *self._decode(), "", *self._blocks(), "", *self._copies()]

    def _fill_init(self) -> list[str]:
        lines = []
        for out in self.spec.outputs:
            if out.var in self.fills:
                fill = self.fills[out.var]
                lines.append(f"{fill}[...] = jnp.full({fill}.shape, {_scalar_src(out.fill)}, {fill}.dtype)")
        return lines

    def _first_fetches(self, dependent: bool | None = None) -> list[str]:
        """Start the first steps' copies: of every input (None), or only those an earlier fused nest writes
        (True) or doesn't (False)."""
        v, N, lines = self.v, self.buffers, []

        def wanted(op: Operand) -> bool:
            return dependent is None or (op.param in self.dependent) == dependent

        per_tile = [op for op in self.per_tile if wanted(op)]
        streamed = [op for op in self.streamed if wanted(op)]
        if per_tile:
            blocks = self._n("_blocks")
            first = [f"{self._copy_fn(op)}({blocks}({v['o_begin']}, 0), {v['o_begin']} & 1).start()" for op in per_tile]
            lines += _fn(self._n("_first_tile_inputs"), "", first, when=f"{v['o_begin']} < {v['o_end']}")
        if streamed:
            for k in range(N - 1):
                lines += self._fetch_ahead(f"{v['s_begin']} + {k}", f"_prefetch_{k}", streamed)
        return lines

    def _run(self, hook: bool = False) -> list[str]:
        """The step loop. With `hook`, it calls the next fused nest's prologue once its own copies are issued."""
        v, N = self.v, self.buffers
        lines = []
        # The step bounds are read from SMEM (not Python ints) with runtime tables or several TensorCores.
        traced = self.sched.mode == "runtime" or self.cores > 1
        if hook:
            at = f"{v['s_end']} - {N - 1}, {v['s_begin']}"
            at = f"jnp.maximum({at})" if traced else f"max({at})"
            lines.append(f"{self.pieces['hook_at']} = {at}")
        lines += _fn(self._n("_steps"), v["s"], self._step(hook), loop=f"{v['s_begin']}, {v['s_end']}")
        if hook and traced:  # no steps at all (on this core): the loop never started the next nest
            no_steps = [f"{self.pieces['hook']}()"]
            lines += _fn(self._n("_no_steps"), "", no_steps, when=f"{v['s_end']} <= {v['s_begin']}")
        return lines

    def _drain(self) -> list[str]:
        v, lines, blocks = self.v, [], self._n("_blocks")
        for out in self.spec.outputs:
            for back in (2, 1):
                at = f"{v['o_end']} - {back}"
                wait = [f"{self._out_copy_fn(out)}({blocks}({at}, 0), ({at}) & 1).wait()"]
                lines += _fn(self._n(f"_drain_{out.ref}_{back}"), "", wait, when=f"{at} >= {v['o_begin']}")
        return lines + self._fill_drain()

    def _kernel(self, name: str) -> list[str]:
        body = [*self._setup(), *self._fill_init(), "", "# Fetch the first steps' inputs.", *self._first_fetches()]
        body += ["", *self._run(), "", "# Wait for the copies still in flight.", *self._drain()]
        return [f"def {name}({', '.join(self._kernel_args())}):", *_indent(body)]

    def part(self, name: str, hook: bool) -> list[str]:
        """This nest as one part of a fused kernel. `name(<its refs>, <its early scratch>)` defines its copies
        and returns two functions, which the fused kernel runs in turn:

        - early(): start the first copies of the inputs no earlier part writes (run while the previous part
          finishes), into the part's *early* scratch (`scratch_groups`), allocated before it;
        - with_scratch(<its other scratch>) -> (late, run, drain), once the rest of its scratch exists:
          - late(): start the first copies of the inputs an earlier part wrote (after it has drained);
          - run(hook): the step loop; `hook` (the next part's `early`) is called once this part's own copies
            have all been issued;
          - drain(): wait for this part's output copies.
        """
        p = self.pieces
        early_ops = [op for op in self.spec.operands if op.param not in self.dependent]
        late_ops = [op for op in self.spec.operands if op.param in self.dependent]
        groups = self._arg_groups()
        early_scratch, late_scratch = (names for names, _ in self.scratch_groups())
        args = [*groups["smem"], *groups["hbm"], *groups["aliased"], *groups["outputs"], *early_scratch]
        scalar_reads = self._scalar_reads()
        body = [*self._bounds(), *(["", *scalar_reads] if scalar_reads else []), ""]
        body += [*self._decode(), "", *self._blocks(), "", *self._copies(early_ops, outputs=False)]
        body += _fn(p["early"], "", [*self._fill_init(), *self._first_fetches(False)])
        inner = [*self._copies(late_ops, outputs=True)]
        inner += _fn(p["late"], "", self._first_fetches(True))
        inner += _fn(p["run"], p["hook"] if hook else "", self._run(hook)) + _fn(p["drain"], "", self._drain())
        inner.append(f"return {p['late']}, {p['run']}, {p['drain']}")
        body += _fn(p["with_scratch"], ", ".join(late_scratch), inner)
        body.append(f"return {p['early']}, {p['with_scratch']}")
        return self._contained(name, ", ".join(args), [*self._kernel_helpers(), "", *body])

    def scratch_groups(self) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
        """A fused part's scratch as ((names, shapes) early, (names, shapes) late): the early scratch is what
        its `early` copies write (the rings of inputs no earlier part writes, and the skipped tiles' fill
        values); the rest (the rings of inputs earlier parts write, output slots, state) is only needed once
        the part runs."""
        early: tuple[list[str], list[str]] = ([], [])
        late: tuple[list[str], list[str]] = ([], [])
        for names, shapes, is_early in self._scratch_buffers():
            group = early if is_early else late
            group[0].extend(names)
            group[1].extend(shapes)
        return early, late

    # ------------------------------------------------------------------ wrapper

    def _scratch_buffers(self) -> list[tuple[list[str], list[str], bool]]:
        """Each scratch buffer, in kernel-argument order: (its names, its shapes, written by a fused part's
        `early` copies)."""
        spec, N, out = self.spec, self.buffers, []
        for op in spec.operands:
            depth = N if op.phase is not None else 2
            shape = (depth, *op.block.kernel_shape)
            lines = [
                f"pltpu.VMEM({shape}, {_dtype_src(op.dtype)}),  # {op.param}[{op.block.text}]",
                f"pltpu.SemaphoreType.DMA(({depth},)),",
            ]
            out.append(([self.slots[op.ref], self.sems[op.ref]], lines, op.param not in self.dependent))
        for o in spec.outputs:
            shape = (2, *o.block.kernel_shape)
            lines = [
                f"pltpu.VMEM({shape}, {_dtype_src(o.out.dtype)}),  # {o.var}[{o.block.text}]",
                "pltpu.SemaphoreType.DMA((2,)),",
            ]
            out.append(([self.out_slots[o.var], self.out_sems[o.var]], lines, False))
        for sv in spec.state.values():
            dtype = _dtype_src(_stored_dtype(sv))
            mem = f"pltpu.SMEM((1,), {dtype})" if sv.smem else f"pltpu.VMEM({tuple(sv.aval.shape)}, {dtype})"
            out.append(([sv.ref], [f"{mem},  # {sv.name}"], False))
        for o in spec.outputs:
            if o.var in self.fills:
                fill = self.fills[o.var]
                lines = [
                    f"pltpu.VMEM({o.block.kernel_shape}, {_dtype_src(o.out.dtype)}),  # {o.var}'s skipped tiles",
                    "pltpu.SemaphoreType.DMA(()),",
                ]
                out.append(([fill, self._n(f"{fill}_sem")], lines, True))
        return out

    def _scratch(self) -> list[str]:
        return [line for _, lines, _ in self._scratch_buffers() for line in lines]

    def _params(self) -> dict[str, str]:
        """The wrapper's parameters: the kernel's, renamed where they would hide a name the wrapper uses."""
        if self._param_names is None:  # (every other name the wrapper uses is fresh)
            names = {p: self.renamed.get(p) for p in self.nest.params}  # `np` & co. as everywhere else
            self._param_names = {
                p: n or (self.names.fresh(f"{p}_arg") if p == "interpret" else p) for p, n in names.items()
            }
        return self._param_names

    def _wrapper_inputs(self) -> tuple[list[str], list[str], list[str]]:
        """What the wrapper computes before the call: (its lines, the SMEM inputs, the aliased initial values)."""
        spec, param = self.spec, self._params()
        lines, smem, inits = [], list(self.table_consts.values()), []
        if self.sched.mode == "runtime":
            if self._tables_local is None:
                self._tables_local = self.names.fresh("tables")
            tables = self._tables_local
            lines.append("# A mask reads inputs, so the steps are listed on device.")
            inputs = ", ".join(f"{self._user(p)}={a}" for p, a in param.items())
            lines.append(f"{tables} = {self._n('_runtime_tables')}({inputs})")
            smem += [f"{tables}[{k!r}]" for k in RUNTIME_TABLES]
        if self.scalars_ref:
            values = ", ".join(f"jnp.asarray({param[p]}, jnp.int32)" for p in self.int_scalars)
            lines.append(f"{self.scalars_ref} = jnp.stack([{values}])  # runtime scalars, read from SMEM")
            smem.append(self.scalars_ref)
        if self.fscalars_ref:
            values = ", ".join(f"jnp.asarray({param[p]}, jnp.float32)" for p, _ in self.float_scalars)
            lines.append(f"{self.fscalars_ref} = jnp.stack([{values}])  # float runtime scalars, read from SMEM")
            smem.append(self.fscalars_ref)

        for var, init in self.aliased.items():
            out = next(o for o in spec.outputs if o.var == var)
            self.used_names |= reads(out.alloc)
            alloc = ast.unparse(_Rename({**self.renamed, **param}).visit(copy.deepcopy(out.alloc)))
            if out.padded:
                alloc = f"jnp.pad({alloc}, {_pad_widths(out.out.shape, out.padded)})"
            lines.append(f"{init} = {alloc}  # skipped tiles keep these values")
            inits.append(init)
        for p in spec.arrays:  # inputs whose last tiles run past their end: zeros there
            if p in spec.pads:
                widths = _pad_widths(spec.given_shape(p), spec.pads[p])
                lines.append(f"{param[p]} = jnp.pad({param[p]}, {widths})  # its last tiles run past its end")
        return lines, smem, inits

    def _wrapper(self, kernel: str) -> list[str]:
        spec, nest = self.spec, self.nest
        smem, anyspace = "pl.BlockSpec(memory_space=pltpu.SMEM)", "pl.BlockSpec(memory_space=pl.ANY)"
        prep, smem_args, inits = self._wrapper_inputs()
        in_specs = [smem] * len(smem_args) + [anyspace] * (len(self.hbm) + len(inits))
        outs = [f"jax.ShapeDtypeStruct({tuple(o.kernel_out.shape)}, {_dtype_src(o.out.dtype)})" for o in spec.outputs]
        out_shape = f"({', '.join(outs)},)" if nest.returns_tuple else outs[0]
        out_specs = f"({', '.join([anyspace] * len(outs))},)" if nest.returns_tuple else anyspace

        param = self._params()
        lines = [f"def {nest.fn_name}_pallas({', '.join([*param.values(), '*', 'interpret=False'])}):"]
        lines.append(f"    # No grid: one loop over {self._describe()}{self._buffering()}.")
        lines += _indent(prep)
        args = [*smem_args, *[param[p] for p in self.hbm], *inits]
        aliases = {len(args) - len(inits) + n: nest.out_vars.index(var) for n, var in enumerate(self.aliased)}
        params = [f"vmem_limit_bytes={self.vmem_limit}"] + (
            ['dimension_semantics=("parallel",)'] if self.cores > 1 else []
        )
        cropped = any(o.padded for o in spec.outputs)
        result = self._n("out") if cropped else None
        lines += [
            f"    {result} = pl.pallas_call(" if cropped else "    return pl.pallas_call(",
            f"        {kernel},",
            f"        out_shape={out_shape},",
            *([f"        grid=({self.cores},),"] if self.cores > 1 else []),
            f"        in_specs=[{', '.join(in_specs)}],",
            f"        out_specs={out_specs},",
            "        scratch_shapes=[",
            *_indent(self._scratch(), 3),
            "        ],",
            *([f"        input_output_aliases={aliases},"] if aliases else []),
            f"        compiler_params=pltpu.CompilerParams({', '.join(params)}),",
            *([f"        cost_estimate={self._n('_COST')},"] if self.cost is not None else []),
            "        interpret=interpret,",
            f"        name={nest.fn_name!r},",
            f"    )({', '.join(args)})",
        ]
        if cropped:  # what the last tiles wrote past the outputs' ends is dropped
            crops = [f"{result}[{k}]" if nest.returns_tuple else result for k in range(len(spec.outputs))]
            crops = [
                f"{c}[{', '.join(f':{n}' for n in o.out.shape)}]" if o.padded else c
                for c, o in zip(crops, spec.outputs, strict=True)
            ]
            lines.append(f"    return ({', '.join(crops)},)" if nest.returns_tuple else f"    return {crops[0]}")
        return lines

    def prepare(self, name: str) -> list[str]:
        """For a fused kernel: `name(*inputs)` -> (SMEM inputs, aliased initial values) of this part.

        Inputs an earlier part writes are passed as None: the fused kernel only fuses nests whose masks and
        aliased allocations don't read them.
        """
        prep, smem, inits = self._wrapper_inputs()
        helpers = self._wrapper_helpers()
        weak = [*_WEAK.format(name=self.weak).splitlines(), "", ""] if helpers and self.weak is not None else []
        body = [*weak, *helpers, *prep, f"return [{', '.join(smem)}], [{', '.join(inits)}]"]
        return self._contained(name, ", ".join(self._params().values()), body)

    def arg_groups(self) -> dict[str, list[str]]:
        return self._arg_groups()

    def describe(self) -> str:
        return self._describe()

    def buffering(self) -> str:
        return self._buffering()

    def _buffering(self) -> str:
        """How the wrapper comment describes the inputs' VMEM slots."""
        streamed, per_tile = f"{self.buffers}-buffered", "fetched once per tile, 2-buffered"
        if self.streamed and self.per_tile:
            return f"; streamed inputs {streamed}, others {per_tile}, by hand"
        if self.streamed or self.per_tile:
            return f"; inputs {streamed if self.streamed else per_tile} by hand"
        return ""

    def _describe(self) -> str:
        sched, spec = self.sched, self.spec
        total = spec.total_tiles
        tiles = f"{sched.tiles} tiles" + (f" (of {total})" if sched.tiles != total else "")
        if sched.mode == "runtime":
            return f"the steps of up to {tiles} that the masks keep at run time"
        cores = f", split over {self.cores} TensorCores" if self.cores > 1 else ""
        return f"{sched.steps} steps of {tiles}{cores}"

    # ------------------------------------------------------------------ assembly

    def header(self) -> str:
        spec, nest = self.spec, self.nest
        inputs = ", ".join(
            f"{p}: {_dtype_src(s.dtype)}{tuple(s.shape)}" for p, s in zip(nest.params, spec.specs, strict=True)
        )
        header = f"# Generated by autokernel from `{nest.fn_name}` ({spec.filename}:{nest.first_line}) for {inputs}"
        if spec.overrides:
            header += "\n# Tuned: " + ", ".join(f"{k}={v}" for k, v in sorted(spec.overrides.items()))
        return header

    def _kernel_helpers(self) -> list[str]:
        """The helpers the kernel's code calls (once that code is generated): index arithmetic, `@`, runtime
        indices and weak types."""
        names = {k: self._n(k) for k in ("_div", "_rem", "_index", *mxu.HELPERS)}
        parts = [_PRELUDE.format(**names), mxu.prelude(self.precision, names)]
        parts += [_SCALAR_PRELUDE.format(**names)] if self.indices else []
        parts += [_WEAK.format(name=self.weak)] if self.weak is not None else []
        return "\n\n".join(parts).splitlines()

    def _wrapper_helpers(self) -> list[str]:
        """The functions the wrapper calls to build runtime step tables, if it has any."""
        if self.sched.mode != "runtime":
            return []
        return [*self._step_table_functions(), "", "", *self._runtime_tables()]

    def _step_table_functions(self) -> list[str]:
        """`schedule.step_tables` and the functions it calls, as source (under fresh names)."""
        fns = (schedule_lib.step_tables, schedule_lib.column_mask, schedule_lib.build_tables, schedule_lib.core_bounds)
        renamed = {f.__name__: self._n(f"_{f.__name__}") for f in fns}
        consts = {"FIRST": FIRST, "LAST": LAST, "COLUMN_SHIFT": COLUMN_SHIFT}

        class Rename(ast.NodeTransformer):
            def visit_FunctionDef(self, n: ast.FunctionDef) -> ast.AST:
                self.generic_visit(n)
                n.name = renamed.get(n.name, n.name)
                return n

            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in consts:
                    return ast.copy_location(ast.Constant(consts[n.id]), n)
                return ast.copy_location(ast.Name(renamed.get(n.id, n.id), n.ctx), n)

        lines = ["# The step tables of masks that read inputs, built on device (from autokernel.schedule)."]
        for f in fns:
            tree = Rename().visit(ast.parse(textwrap.dedent(inspect.getsource(f))))
            lines += [*ast.unparse(tree).splitlines(), "", ""]
        return lines[:-2]

    def _array(self, base: str, value: Any) -> str:
        """A constant array the generated code reads, by name."""
        name = self._n(base)
        self.arrays[name] = np.asarray(value)
        return name

    def _runtime_tables(self) -> list[str]:
        """`_runtime_tables(*inputs)`: the step tables of masks that read inputs (`schedule.runtime_tables`)."""
        spec, sched, xp = self.spec, self.sched, self._n("xp")
        body = [
            "# Each mask that reads inputs runs on device, once per combination of the loop values it reads (as",
            "# jax.numpy: its Python operators in their array forms), and is spread over the tiles and steps.",
            f"{xp} = jnp",
        ]
        tile_keep = "None"
        if spec.tile_mask_runtime:
            tile_keep = self._n("tile_keep")
            body += self._mask(None, tile_keep, xp)
        masks = []
        for p, phase in enumerate(spec.phases):
            if phase.reduction.guard is None:
                masks.append("None")
            elif not spec.phase_runtime[p]:  # a static mask: its value
                masks.append(f"jnp.asarray({self._array(f'phase{p}_mask', spec.phase_masks[p])})")
            else:
                masks.append(self._n(f"keep{p}"))
                body += self._mask(p, masks[-1], xp)
        shape = f"{list(sched.phase_steps)}, {len(spec.tiles)}, {self.cores}"  # steps per phase, tiles, cores
        body.append(f"return {self._n('_step_tables')}({tile_keep}, [{', '.join(masks)}], {shape})")
        return _fn(self._n("_runtime_tables"), ", ".join(map(self._user, self.nest.params)), body)

    def _mask(self, phase: int | None, var: str, xp: str) -> list[str]:
        """`var = ...`: the tile mask (`phase` None) or a phase's mask, as `analysis.evaluate_mask` computes it."""
        spec = self.spec
        cond = spec.nest.guard if phase is None else spec.phases[phase].reduction.guard
        loops = list(spec.parallel) + (list(spec.phases[phase].loops) if phase is not None else [])
        used = reads(cond)
        self.used_names |= used
        read = [lb for lb in loops if lb.var in used]
        done = spec.finished_loops(phase) if phase is not None else {}  # earlier phases' loops: their last values
        user = self._user
        args = [user(lb.var) for lb in read]
        args += [f"{user(k)}={v}" for k, v in done.items() if k in used and k not in loops]
        # Renamed first: a user's own `max` isn't the builtin `vectorize` makes elementwise.
        value = _Rename({analysis.XP: xp}).visit(analysis.vectorize(self._user_code(cond)))
        value = f"jnp.asarray({ast.unparse(value)})"
        tag = "tile" if phase is None else f"phase{phase}"
        steps = spec.phases[phase].steps if phase is not None else None
        shape = (len(spec.tiles),) if phase is None else (len(spec.tiles), steps)
        if not read:
            value = f"(lambda {', '.join(args)}: {value})()" if args else value
            return [f"{var} = jnp.broadcast_to({value}.astype(bool), {shape})"]
        grids = [np.arange(lb.start, lb.stop, lb.step, dtype=np.int32) for lb in read]
        mesh = np.meshgrid(*grids, indexing="ij")
        coords = [self._array(f"{tag}_{lb.var}_values", m.reshape(-1)) for lb, m in zip(read, mesh, strict=True)]
        positions = []  # the position of each tile's (and step's) value in each loop the mask reads
        parallel = [lb.var for lb in spec.parallel]
        for lb in read:
            if lb.var in parallel:
                pos = np.array([t[parallel.index(lb.var)] for t in spec.tiles], np.int32) - lb.start // lb.step
                pos = pos if phase is None else pos[:, None]
            else:
                counts = [b.count for b in spec.phases[phase].loops]
                k = [b.var for b in spec.phases[phase].loops].index(lb.var)
                pos = np.unravel_index(np.arange(steps), counts)[k][None, :].astype(np.int32)
            positions.append(self._array(f"{tag}_{lb.var}_positions", pos))
        # The coordinates as weakly typed int32s: the Python ints they are in the loops.
        names = [user(lb.var) for lb in read]
        weak = ", ".join(f"{self._weak()}({v})" for v in names)
        return [
            f"{var} = jax.vmap(lambda {', '.join(names)}: (lambda {', '.join(args)}: {value})({weak}))"
            f"({', '.join(coords)})",
            f"{var} = {var}.astype(bool).reshape({tuple(len(g) for g in grids)})",
            f"{var} = jnp.broadcast_to({var}[{', '.join(positions)}], {shape})",
        ]

    def namespace(self) -> dict[str, Any]:
        """What the generated code runs with: the user's constants (the builtins it calls stay builtins, and
        names it renamed are bound under their new names), the imports, the cost estimate and tables."""
        consts = self.spec.consts
        namespace: dict[str, Any] = {k: v for k, v in consts.items() if k not in _BUILTINS}
        namespace.update(MODULES)
        namespace.update({new: consts[old] for old, new in self.renamed.items() if old in consts})
        if self.cost is not None:
            namespace[self._n("_COST")] = self.cost
        namespace.update({self.table_consts[name]: value for name, value in self.tables.items()})
        namespace.update(self.arrays)
        namespace.update(self.exported)
        return namespace

    def _constants(self, code: str, prefix: str | None = None) -> tuple[list[str], list[str]]:
        """Source binding the names `code` reads from the namespace (the user's constants as literals and
        imports, the user functions it calls, the cost estimate and the tables): (bindings, definitions).

        The definitions are the user functions' (`_function_defs`), for module level. Without `prefix` they
        bind the names `code` reads; with it (a fused part, whose code binds what it reads inside itself)
        they bind names starting with `prefix`, which the bindings read.
        """
        namespace = self.namespace()
        generated = {*self.table_consts.values(), *self.arrays, self._n("_COST")}
        wanted = sorted((n for n in _globals_read(code) if n in namespace), key=lambda n: (n in generated, n))
        imports, values, missing, functions = [], [], [], {}
        for name in wanted:
            value = namespace[name]
            if name.startswith("__") or MODULES.get(name) is value:
                continue
            if (text := _literal(value)) is not None:
                values.append(f"{name} = {text}")
            elif (lines := _import(name, value)) is not None:
                imports += lines
            elif (fn := _function_source(value)) is not None:
                functions[name] = fn
            else:
                missing.append(f"# `{name}` (a {type(value).__name__}) has no source here: define it to run this alone")
        # A fused part's `_part` and `_prepare` share the definitions.
        known = {name: self._fn_exports[id(f.fn)] for name, f in functions.items() if id(f.fn) in self._fn_exports}
        defs, exports = self._function_defs({n: f for n, f in functions.items() if n not in known}, prefix)
        exports.update(known)
        self._fn_exports.update({id(functions[name].fn): export for name, export in exports.items()})
        binds = [f"{name} = {exports[name]}" for name in functions if exports[name] != name]
        if prefix is not None:  # the fused kernel runs with the user's functions under those names
            self.exported.update({exports[name]: functions[name].value for name in functions})
        lines = [*imports, *values, *binds, *missing]
        what = f"# What the kernel reads from `{self.nest.fn_name}`'s module, and the values it was planned with."
        return ([what, *lines] if lines or defs else []), defs

    def _function_defs(self, functions: dict[str, _Function], prefix: str | None) -> tuple[list[str], dict[str, str]]:
        """Definitions of the user functions the kernel reads (by name) and of the ones they call, and the
        module-level name each kernel name is bound to.

        A function reads its own module's globals, which needn't be the kernel's (a tuned `T`, a helper
        from another module), so each module's functions are defined inside a function binding what they
        read there (literals as its arguments, imports, other modules' functions as arguments too):

            def _mod_functions(T):
                def scaled(t):
                    return t / T
                return scaled
            scaled = _mod_functions(T=128)
        """
        groups: dict[int, dict[str, Any]] = {}  # id(module globals) -> what its definitions need
        export: dict[int, str] = {}  # id(function) -> the module-level name bound to it
        for name, f in functions.items():
            export.setdefault(id(f.fn), name if prefix is None else self._n(f"{prefix}_{name}"))
        queue = list(functions.values())
        while queue:
            f = queue.pop(0)
            g = groups.setdefault(
                id(f.fn.__globals__),
                {"module": f.fn.__module__, "functions": {}, "bound": set(), "args": {}, "body": [], "aliases": []},
            )
            if id(f.fn) in g["functions"]:
                continue
            g["functions"][id(f.fn)] = f
            for n, v in f.reads.items():  # one value per name in a module
                if n in g["bound"]:
                    continue
                g["bound"].add(n)
                if (text := _literal(v)) is not None:
                    g["args"][n] = text
                elif (lines := _import(n, v)) is not None:
                    g["body"] += lines
                elif (dep := _function_source(v)) is not None:
                    queue.append(dep)
                    if dep.fn.__globals__ is f.fn.__globals__:  # defined next to it
                        if n != dep.fn.__name__:
                            g["aliases"].append(f"{n} = {dep.fn.__name__}")
                    else:
                        g["args"][n] = dep
                        export.setdefault(id(dep.fn), self._n(f"{prefix or ''}_{dep.fn.__name__}"))
                else:
                    g["body"].append(f"# `{n}` (a {type(v).__name__}) has no source here: define it to run this alone")

        lines: list[str] = []
        done: set[int] = set()

        def define(key: int, within: frozenset[int]) -> None:
            """The definitions of module `key`'s functions, after those of the modules they call."""
            g, kwargs, params = groups[key], [], []
            for n, arg in g["args"].items():
                if isinstance(arg, _Function):
                    other = id(arg.fn.__globals__)
                    if other in within:  # modules calling each other's functions: no order works
                        g["body"].append(f"# `{n}`: its module calls back into this one: define it to run this alone")
                        continue
                    if other not in done:
                        define(other, within | {key})
                    arg = export[id(arg.fn)]
                params.append(n)
                kwargs.append(f"{n}={arg}")
            fns = list(g["functions"].values())
            names = ", ".join(f"`{f.fn.__name__}`" for f in fns)
            module = "".join(c if c.isalnum() else "_" for c in g["module"].rsplit(".", 1)[-1]).strip("_")
            factory = self._n(f"{prefix or ''}_{module or 'user'}_functions")
            out = [f.fn.__name__ for f in fns if id(f.fn) in export]
            body = [f"# From `{g['module']}`: {names}, with the values they read there.", *g["body"]]
            for f in fns:
                body += [*f.source.splitlines(), ""]
            body += [*g["aliases"], f"return {', '.join(out)}"]
            targets = ", ".join(export[id(f.fn)] for f in fns if id(f.fn) in export)
            lines.extend([*_fn(factory, ", ".join(params), body), f"{targets} = {factory}({', '.join(kwargs)})", ""])
            done.add(key)

        for key in groups:
            if key not in done:
                define(key, frozenset())
        return lines[:-1], {name: export[id(f.fn)] for name, f in functions.items()}

    def _contained(self, name: str, args: str, body: list[str]) -> list[str]:
        """`def name(args): body` with the values it reads bound first, so a fused kernel's source runs alone.
        Only literals, imports and names of functions: running it again (as the fused kernel does) changes
        nothing. The user functions it calls are defined at module level (`module_defs`)."""
        constants, defs = self._constants("\n".join(_fn(name, args, body)), prefix=name)
        self.module_defs += [*defs, ""] if defs else []
        return _fn(name, args, [*constants, *([""] if constants else []), *body])

    def emit(self) -> Generated:
        nest = self.nest
        kernel_name = self._n(f"{nest.fn_name}_kernel")
        kernel = self._kernel(kernel_name)
        wrapper = self._wrapper(kernel_name)
        wrapper_helpers = self._wrapper_helpers()  # first: they may need the weak-type helper too
        helpers = [*self._kernel_helpers(), *(["", "", *wrapper_helpers] if wrapper_helpers else [])]
        code = "\n".join([*helpers, "", "", *kernel, "", "", *wrapper, ""])
        constants, defs = self._constants(code)
        constants += ["", *defs] if defs else []
        head = "\n".join([self.header(), "", _IMPORTS, *constants, *([""] if constants else []), ""])
        # Run the code after the imports and constants: the namespace holds those, as the user's module does.
        namespace = self.namespace()
        exec(compile("\n" * head.count("\n") + code, f"<autokernel:{nest.fn_name}>", "exec"), namespace)
        return Generated(head + code, namespace[f"{nest.fn_name}_pallas"])


def assignments(values: dict[str, Any]) -> list[str]:
    """`name = <value as source>` for each value (literals, arrays, cost estimates; see `_literal`)."""
    return [f"{name} = {_literal(value)}" for name, value in values.items()]


def prelude() -> str:
    """What the source of a fused kernel starts with: the imports (each part defines its own helpers, see
    `Emitter.part`)."""
    return _IMPORTS
