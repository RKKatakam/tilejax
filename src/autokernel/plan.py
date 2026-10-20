"""Resolve a parsed loop nest for concrete input shapes: what the kernel copies, keeps and computes.

`make_plan` runs the prologue on the inputs' shapes, evaluates the loop ranges, turns every tile read and
store into a window of an array, finds the values a tile carries from one reduction step to the next, and
traces the tile code once with abstract values to learn every type. Nothing here depends on how the kernel
is pipelined (pipeline.py) or the order its steps run in (schedule.py).
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import mxu, v5e
from .errors import KernelError
from .parse import Loop, Nest
from .tilecode import assigned, first_reads, flatten, reads

__all__ = ["Carried", "Kind", "Operand", "Output", "Plan", "Range", "Tile", "Window", "make_plan"]


@dataclass(frozen=True)
class Range:
    """A loop's iterations: `start + block * step` for block in `range(count)`."""

    var: str
    start: int
    step: int
    count: int
    mask: ast.expr | None  # `if mask: continue` opening the loop's body

    def value(self, block: int) -> int:
        return self.start + block * self.step


@dataclass(frozen=True)
class Window:
    """How a tile indexes one array dim: from element `offset` (plus the value of `loop`, if any), `size`
    elements; `size` None is an integer index, which drops the dim from the tile."""

    loop: str | None
    offset: int
    size: int | None


@dataclass(frozen=True)
class Tile:
    """A window of an array that the kernel copies between HBM and VMEM."""

    array: str
    windows: tuple[Window, ...]
    dtype: np.dtype
    text: str  # as written, e.g. "x[i:i + TM, k:k + TK]"

    @property
    def shape(self) -> tuple[int, ...]:
        """The tile's shape in VMEM: the sliced dims."""
        return tuple(w.size for w in self.windows if w.size is not None)

    @property
    def loops(self) -> frozenset[str]:
        """The loop variables that move the window."""
        return frozenset(w.loop for w in self.windows if w.loop)


class Kind(Enum):
    """When an input tile changes, and so how the kernel buffers it."""

    STREAM = "stream"  # every step: a ring of slots, fetched ahead
    TILE = "tile"  # every output tile: two slots, the next tile's fetched at a tile's first step
    ONCE = "once"  # never: fetched once, before the first step


@dataclass(frozen=True)
class Operand:
    """An input tile the kernel reads."""

    name: str  # unique among operands: "x", or "x2" for a second window of x
    param: int  # which input
    tile: Tile
    kind: Kind


@dataclass(frozen=True)
class Output:
    """An output: its array, the fill of tiles a mask skips, and the tile each step stores."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fill: Any  # the allocation's value (0 for jnp.zeros)
    tile: Tile
    value: jax.ShapeDtypeStruct  # the type of the value the tile code stores


@dataclass(frozen=True)
class Carried:
    """A value a tile carries from one reduction step to the next: VMEM scratch in the kernel."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class Plan:
    """A loop nest resolved for one input signature."""

    nest: Nest
    inputs: tuple[jax.ShapeDtypeStruct, ...]
    tunable: Any  # the tunable's values it was planned with (None: the kernel has none)
    parallel: tuple[Range, ...]  # outermost first
    reduction: Range | None
    operands: tuple[Operand, ...]
    reads: Mapping[str, Operand]  # `read_key` of a tile read -> its operand
    outputs: tuple[Output, ...]
    carried: Mapping[str, Carried]
    assigned_types: Mapping[int, Mapping[str, jax.ShapeDtypeStruct]]  # id(stmt) -> the types it assigns
    constants: Mapping[str, Any]  # names the tile code reads from the module, closure and prologue
    scopes: tuple[str, ...]  # the names of the tile code's `jax.named_scope`s
    env: dict[str, Any]  # the function's module and closure values, with the prologue's names over them

    @property
    def single_step(self) -> bool:
        """Does every tile run in one step? Then nothing is carried between steps."""
        return _single_step(self.reduction)

    @property
    def loops(self) -> tuple[Range, ...]:
        return self.parallel + ((self.reduction,) if self.reduction else ())

    def eval(self, node: ast.expr, values: Mapping[str, Any] | None = None) -> Any:
        """Evaluate an expression of the function on the prologue's names (and `values`, e.g. loop values)."""
        return evaluate(self.nest, self.env, node, values)


def _single_step(reduction: Range | None) -> bool:
    return reduction is None or (reduction.count == 1 and reduction.mask is None)


def evaluate(nest: Nest, env: dict[str, Any], node: ast.expr, values: Mapping[str, Any] | None = None) -> Any:
    """Evaluate an expression of `nest`'s function with `env` as its globals and `values` (loop values) as
    its locals; KernelError if it fails."""
    try:
        code = compile(ast.Expression(node), nest.filename, "eval")
        return eval(code, env, dict(values or {}))
    except Exception as e:
        raise nest.error(f"can't evaluate `{ast.unparse(node)}` when planning: {e!r}", node) from e


def read_key(node: ast.Subscript) -> str:
    """Identifies a tile read `x[...]` (the same window of the same input is one operand)."""
    return ast.dump(node)


def make_plan(nest: Nest, fn: Callable, inputs: Sequence[jax.ShapeDtypeStruct], tunable: Any = None) -> Plan:
    """Resolve `nest` (parsed from `fn`) for inputs of these shapes and dtypes, and its tunable's values."""
    return _Planner(nest, fn, tuple(inputs), tunable).plan()


class _Planner:
    def __init__(self, nest: Nest, fn: Callable, inputs: tuple[jax.ShapeDtypeStruct, ...], tunable: Any) -> None:
        self.nest, self.fn, self.inputs = nest, fn, inputs
        closure = inspect.getclosurevars(fn).nonlocals
        # One namespace (code in the prologue, masks and bounds sees its names from lambdas and generator
        # expressions too): the module's values, the closure's over them, the prologue's over those.
        self.env: dict[str, Any] = {"__builtins__": builtins, **fn.__globals__, **closure}
        self.env.update(zip(nest.params, inputs, strict=True))
        if nest.tunable:
            self.env[nest.tunable] = tunable
        self.ranges: dict[str, Range] = {}

    def error(self, message: str, node: ast.AST | None = None) -> KernelError:
        return self.nest.error(message, node)

    def eval(self, node: ast.expr) -> Any:
        return evaluate(self.nest, self.env, node)

    def plan(self) -> Plan:
        outputs = self._prologue()
        parallel = tuple(self._range(loop) for loop in self.nest.parallel)
        reduction = self._range(self.nest.reduction) if self.nest.reduction else None
        self.ranges = {r.var: r for r in (*parallel, *([reduction] if reduction else []))}
        single_step = _single_step(reduction)
        operands, reads_ = self._operands(reduction, single_step)
        output_tiles = {name: self._output_tile(name, *outputs[name], parallel) for name in self.nest.outputs}
        self._check_reads_before_assignment()
        carried_names = set() if single_step else self._carried_names()
        scopes = self._scopes()
        types = _TypeTrace(self, operands, reads_, carried_names, single_step).run()
        for name in self.nest.outputs:
            self._check_store(output_tiles[name], types.stores[name])
        carried = {name: Carried(name, aval.shape, aval.dtype) for name, aval in types.carried.items()}
        return Plan(
            nest=self.nest,
            inputs=self.inputs,
            tunable=self.env.get(self.nest.tunable) if self.nest.tunable else None,
            parallel=parallel,
            reduction=reduction,
            operands=operands,
            reads=reads_,
            outputs=tuple(
                Output(name, *outputs[name], output_tiles[name], types.stores[name]) for name in self.nest.outputs
            ),
            carried=carried,
            assigned_types=types.assigned,
            constants=self._constants(),
            scopes=scopes,
            env=self.env,
        )

    # ------------------------------------------------------------------ the prologue and outputs

    def _prologue(self) -> dict[str, tuple[tuple[int, ...], np.dtype, Any]]:
        """Run the prologue on the inputs' shapes; each output's (shape, dtype, fill)."""
        allocation_of = {id(stmt): name for name, stmt in self.nest.allocations.items()}
        outputs = {}
        for stmt in self.nest.prologue:
            name = allocation_of.get(id(stmt))
            if name is not None:
                outputs[name] = self._allocation(name, stmt.value)  # type: ignore[attr-defined]
                self.env[name] = jax.ShapeDtypeStruct(*outputs[name][:2])
                continue
            try:
                exec(compile(ast.Module([stmt], []), self.nest.filename, "exec"), self.env)
            except Exception as e:
                raise self.error(f"the prologue runs on the inputs' shapes and dtypes, and failed: {e!r}", stmt) from e
        return outputs

    def _allocation(self, name: str, call: ast.expr) -> tuple[tuple[int, ...], np.dtype, Any]:
        usage = f"allocate output `{name}` with jnp.zeros(shape, dtype) or jnp.full(shape, value, dtype)"
        if not isinstance(call, ast.Call):
            raise self.error(usage, call)
        func = self.eval(call.func)
        if func is not jnp.zeros and func is not jnp.full:
            raise self.error(usage, call)
        args = [self.eval(a) for a in call.args]
        kwargs = {k.arg: self.eval(k.value) for k in call.keywords if k.arg}
        try:
            aval = jax.eval_shape(lambda: func(*args, **kwargs))
        except Exception as e:
            raise self.error(f"{usage}: {e!r}", call) from e
        fill = 0 if func is jnp.zeros else (args[1] if len(args) > 1 else kwargs.get("fill_value"))
        if not np.isscalar(fill):
            raise self.error(f"output `{name}` is filled with a scalar constant, not {type(fill).__name__}", call)
        return aval.shape, aval.dtype, fill

    # ------------------------------------------------------------------ loops

    def _range(self, loop: Loop) -> Range:
        bounds = [self.eval(b) for b in loop.bounds]
        if not all(isinstance(b, int | np.integer) for b in bounds):
            raise self.error(f"loop bounds are ints when planning, got {bounds}", loop.node)
        bounds = [int(b) for b in bounds]
        if len(bounds) == 3 and bounds[2] <= 0:
            raise self.error(f"`{loop.var}` steps by {bounds[2]}: loops count up", loop.node)
        r = range(*bounds)
        if not r:
            raise self.error(f"`{loop.var}` loops over {r}, which is empty", loop.node)
        if (r.stop - r.start) % r.step:
            raise self.error(
                f"`{loop.var}` loops over {r}: the range must be a whole number of steps of {r.step}", loop.node
            )
        if r.start % r.step:
            raise self.error(f"`{loop.var}` loops over {r}: it must start at a multiple of its step", loop.node)
        return Range(loop.var, r.start, r.step, len(r), loop.mask)

    # ------------------------------------------------------------------ tiles

    def _window(self, index: ast.expr, dim: int, text: str) -> Window:
        """How the index expression `index` selects from a dim of size `dim`."""
        usage = (
            f"{text}: each index is `v:v + size` or `v` for a loop variable v, `:`, or constants (`start:stop`, an int)"
        )
        loop_vars = reads(index) & self.ranges.keys()
        if isinstance(index, ast.Slice):
            lower, upper = index.lower, index.upper
            if index.step is not None:
                raise self.error(usage, index)
            if isinstance(lower, ast.Name) and lower.id in self.ranges:  # v:v + size
                if not (
                    isinstance(upper, ast.BinOp)
                    and isinstance(upper.op, ast.Add)
                    and ast.dump(upper.left) == ast.dump(lower)
                    and not reads(upper.right) & self.ranges.keys()
                ):
                    raise self.error(usage, index)
                return self._checked(Window(lower.id, 0, self._int(upper.right, text)), dim, text, index)
            if loop_vars:
                raise self.error(usage, index)
            start = 0 if lower is None else self._int(lower, text)
            stop = dim if upper is None else self._int(upper, text)
            return self._checked(Window(None, start, stop - start), dim, text, index)
        if isinstance(index, ast.Name) and index.id in self.ranges:
            return self._checked(Window(index.id, 0, None), dim, text, index)
        if loop_vars:
            raise self.error(usage, index)
        return self._checked(Window(None, self._int(index, text), None), dim, text, index)

    def _int(self, node: ast.expr, text: str) -> int:
        value = self.eval(node)
        if not isinstance(value, int | np.integer):
            raise self.error(f"{text}: tile bounds are ints when planning, not {value!r}", node)
        return int(value)

    def _checked(self, window: Window, dim: int, text: str, node: ast.AST) -> Window:
        """`window`, once it is known to stay inside the dim."""
        if window.size is not None and window.size <= 0:
            raise self.error(f"{text}: a slice takes at least one element", node)
        first = window.offset
        last = window.offset + (window.size or 1)
        if window.loop:
            r = self.ranges[window.loop]
            first += r.start
            last += r.value(r.count - 1)
        if first < 0 or last > dim:
            raise self.error(f"{text}: reads elements {first} to {last - 1} of a dim of {dim}", node)
        return window

    def _tile(
        self, array: str, shape: tuple[int, ...], dtype: np.dtype, index: Sequence[ast.expr], node: ast.AST
    ) -> Tile:
        text = f"{array}[{', '.join(ast.unparse(i) for i in index)}]"
        if len(index) != len(shape):
            raise self.error(f"{text}: index every one of {array}'s {len(shape)} dims", node)
        windows = tuple(self._window(i, d, text) for i, d in zip(index, shape, strict=True))
        sliced = [w.size is not None for w in windows]
        if not any(sliced):
            raise self.error(f"{text}: a tile keeps at least one dim (slice it, `v:v + 1`)", node)
        if not all(sliced[sliced.index(True) :]):
            raise self.error(f"{text}: integer indices go before every sliced dim", node)
        tile = Tile(array, windows, np.dtype(dtype), text)
        dims = [d for d, w in zip(shape, windows, strict=True) if w.size is not None]
        problems = v5e.layout_problems(tile.shape, dims, dtype)
        if problems:
            raise self.error(f"{text}: a {tile.shape} tile breaks the TPU layout: {'; '.join(problems)}", node)
        return tile

    def _operands(self, reduction: Range | None, single_step: bool) -> tuple[tuple[Operand, ...], dict[str, Operand]]:
        operands: list[Operand] = []
        by_key: dict[str, Operand] = {}
        params = {name: i for i, name in enumerate(self.nest.params)}
        subscripts = [n for stmt in self.nest.tile for n in ast.walk(stmt) if isinstance(n, ast.Subscript)]
        for node in sorted(subscripts, key=lambda n: (n.lineno, n.col_offset)):  # in source order
            param = params.get(node.value.id) if isinstance(node.value, ast.Name) else None
            key = read_key(node)
            if param is None or key in by_key:
                continue
            spec = self.inputs[param]
            index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            tile = self._tile(node.value.id, spec.shape, spec.dtype, index, node)
            if not tile.loops:
                kind = Kind.ONCE
            elif (reduction and reduction.var in tile.loops) or single_step:
                kind = Kind.STREAM
            else:
                kind = Kind.TILE
            taken = {*self.nest.params, *(op.name for op in operands)}
            name = tile.array if tile.array not in {op.tile.array for op in operands} else _fresh(tile.array, taken)
            operand = Operand(name, param, tile, kind)
            operands.append(operand)
            by_key[key] = operand
        return tuple(operands), by_key

    def _output_tile(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype, fill: Any, parallel: tuple[Range, ...]
    ) -> Tile:
        store = self.nest.stores[name]
        tile = self._tile(name, shape, dtype, store.index, store.node)
        uses = [w.loop for w in tile.windows if w.loop]
        for r in parallel:
            if uses.count(r.var) != 1:
                raise self.error(
                    f"{tile.text}: every parallel loop places each output tile, so `{r.var}` indexes exactly one "
                    "dim of the output",
                    store.node,
                )
        for window, dim in zip(tile.windows, shape, strict=True):
            r = self.ranges.get(window.loop) if window.loop else None
            if r is None:
                if window.offset != 0 or window.size != dim:
                    raise self.error(f"{tile.text}: dims no loop places are stored whole, with `:`", store.node)
            elif r.start != 0 or r.count * r.step != dim or (window.size or 1) != r.step:
                raise self.error(
                    f"{tile.text}: loop `{r.var}` must cover the dim of {dim} exactly, one step per tile", store.node
                )
        return tile

    def _check_store(self, tile: Tile, value: jax.ShapeDtypeStruct | None) -> None:
        """A store's value must broadcast to its tile, as `.at[...].set` requires."""
        node = self.nest.stores[tile.array].node
        if value is None:
            raise self.error(f"{tile.text}: the stored value must be an array or a number", node)
        try:
            ok = np.broadcast_shapes(value.shape, tile.shape) == tile.shape
        except ValueError:
            ok = False
        if not ok:
            raise self.error(f"{tile.text}: can't store a value of shape {value.shape} in a {tile.shape} tile", node)

    # ------------------------------------------------------------------ names

    def _check_reads_before_assignment(self) -> None:
        """Python raises UnboundLocalError for a local read before it is assigned: in a tile, that is a
        value expected to carry over from the previous tile, which nothing does."""
        nest = self.nest
        early = (first_reads(nest.pre + nest.body + nest.post) & assigned(nest.tile)) - set(nest.outputs)
        for stmt in flatten(nest.tile):
            names = sorted(reads(stmt) & early)
            if names:
                raise self.error(
                    f"`{names[0]}` is read before the tile assigns it: nothing carries over from one tile to the next",
                    stmt,
                )

    def _carried_names(self) -> set[str]:
        """Values a tile carries between steps: read by the reduction body before it assigns them, or assigned
        before the reduction and read after it."""
        nest = self.nest
        made = assigned(nest.pre) | assigned(nest.body)
        carried = first_reads(nest.body) & made
        carried |= (first_reads(nest.post) & assigned(nest.pre)) - assigned(nest.body)
        for stmt in flatten(nest.pre + nest.body):
            targets = stmt.targets[0] if isinstance(stmt, ast.Assign) else None
            if isinstance(targets, ast.Tuple | ast.List) and assigned([stmt]) & carried:
                raise self.error("assign values carried between steps one at a time, not by unpacking", stmt)
        return carried

    def _constants(self) -> dict[str, Any]:
        """The names tile code reads that aren't its own: the module's, the closure's and the prologue's."""
        nest = self.nest
        own = assigned(nest.tile) | {r.var for r in nest.loops} | set(nest.params) | set(nest.outputs)
        names = set().union(*(reads(stmt) for stmt in flatten(nest.tile))) - own
        return {name: self.env[name] for name in sorted(names) if name in self.env}

    def _scopes(self) -> tuple[str, ...]:
        scopes = []
        for stmt in flatten(self.nest.tile):
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    call: ast.Call = item.context_expr  # type: ignore[assignment]  # parse.py: calls
                    name = self.eval(call.args[0]) if len(call.args) == 1 and not call.keywords else None
                    if self.eval(call.func) is not jax.named_scope or not isinstance(name, str):
                        raise self.error("tile code opens only `with jax.named_scope(name):` blocks", stmt)
                    scopes.append(name)
        return tuple(scopes)


@dataclass(frozen=True)
class _Types:
    carried: dict[str, jax.ShapeDtypeStruct]
    assigned: dict[int, dict[str, jax.ShapeDtypeStruct]]
    stores: dict[str, jax.ShapeDtypeStruct]


class _TypeTrace:
    """Runs the tile code once with abstract values (`jax.eval_shape`): pre, the reduction body twice (a
    value's type after the first step must be its type after every step), post. Records the type of every
    value assigned and stored."""

    def __init__(
        self,
        planner: _Planner,
        operands: tuple[Operand, ...],
        reads_: Mapping[str, Operand],
        carried: set[str],
        single_step: bool,
    ) -> None:
        self.planner, self.nest = planner, planner.nest
        self.operands, self.reads, self.carried, self.single_step = operands, reads_, carried, single_step
        self.notes: dict[tuple[str, int], dict[str, Any]] = {}  # (segment, id(stmt)) -> name -> type
        self.stores: dict[str, jax.ShapeDtypeStruct] = {}
        self.stmts: dict[int, ast.stmt] = {}

    def run(self) -> _Types:
        fn = self._function()
        specs = [jax.ShapeDtypeStruct(op.tile.shape, op.tile.dtype) for op in self.operands]
        loop_vars = [jax.ShapeDtypeStruct((), jnp.int32, weak_type=True) for _ in self.planner.ranges]
        try:
            jax.eval_shape(fn, *specs, *loop_vars)
        except Exception as e:
            frame = next(
                (f for f in reversed(traceback.extract_tb(e.__traceback__)) if f.filename == self.nest.filename),
                None,
            )
            node = ast.Pass(lineno=frame.lineno) if frame else None
            raise self.nest.error(f"the tile code fails for these input shapes: {e!r}", node) from e
        return _Types(self._carried_types(), self._assigned(), self.stores)

    def _function(self) -> Callable:
        """The tile code as one function of its operands and loop variables, with type hooks."""
        segments = [("pre", self.nest.pre), ("body", self.nest.body)]
        if not self.single_step:
            segments.append(("body2", self.nest.body))
        segments.append(("post", self.nest.post))
        body: list[ast.stmt] = []
        for segment, stmts in segments:
            body += [s for stmt in stmts for s in self._instrument(stmt, segment)]
        args = [f"__operand{i}" for i in range(len(self.operands))] + list(self.planner.ranges)
        fdef = ast.FunctionDef(
            name="__tile__",
            args=ast.arguments([], [ast.arg(a) for a in args], None, [], [], None, []),
            body=body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        module = ast.fix_missing_locations(ast.Module([ast.copy_location(fdef, self.nest.function)], []))
        namespace = dict(self.planner.env)
        namespace |= mxu.FUNCTIONS | {"__note__": self._note, "__store__": self._store}
        exec(compile(module, self.nest.filename, "exec"), namespace)
        return namespace["__tile__"]

    def _instrument(self, stmt: ast.stmt, segment: str) -> list[ast.stmt]:
        """`stmt` reading operand arguments, calling `__note__` after it assigns, `__store__` for a store."""
        self.stmts[id(stmt)] = stmt
        store = self.nest.store(stmt)
        if store is not None:
            value = self._rewrite(copy.deepcopy(store.value))
            call = ast.Call(ast.Name("__store__", ast.Load()), [ast.Constant(store.output), value], [])
            return [ast.copy_location(ast.Expr(call), stmt)]
        if isinstance(stmt, ast.With):
            new = copy.copy(stmt)
            new.body = [s for inner in stmt.body for s in self._instrument(inner, segment)]
            return [new]
        new = self._rewrite(copy.deepcopy(stmt))
        names = sorted(assigned([stmt]))
        values = ast.Dict([ast.Constant(n) for n in names], [ast.Name(n, ast.Load()) for n in names])
        key = ast.Tuple([ast.Constant(segment), ast.Constant(id(stmt))], ast.Load())
        note = ast.Expr(ast.Call(ast.Name("__note__", ast.Load()), [key, values], []))
        return [new, ast.copy_location(note, stmt)]

    def _rewrite[T: ast.AST](self, tree: T) -> T:
        index = {op.name: i for i, op in enumerate(self.operands)}
        reads_ = self.reads

        class Reads(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                operand = reads_.get(read_key(node))
                if operand is None:
                    return self.generic_visit(node)  # type: ignore[return-value]
                return ast.copy_location(ast.Name(f"__operand{index[operand.name]}", ast.Load()), node)

        return mxu.rewrite_matmuls(Reads().visit(tree))

    def _note(self, key: tuple[str, int], values: dict[str, Any]) -> None:
        self.notes[key] = {name: _type(value) for name, value in values.items()}

    def _store(self, output: str, value: Any) -> None:
        self.stores[output] = _type(value)

    def _carried_types(self) -> dict[str, jax.ShapeDtypeStruct]:
        """The one type each carried value keeps: what the reduction body makes it.

        The kernel keeps a carried value in a VMEM buffer of that type, so every value assigned to it must
        have the type; the code before the reduction may also start it with a number (`m = -1e30`), which
        the kernel broadcasts to the carried type."""
        types = {}
        for name in sorted(self.carried):
            body = self._last("body", name)
            kept = body if body is not None else self._last("pre", name)
            node = self._assignment(name)
            if kept is None or not kept.shape:
                raise self.nest.error(
                    f"`{name}` is carried between reduction steps, so it must be an array (kept in VMEM)", node
                )
            for (segment, stmt), values in self.notes.items():
                value = values.get(name, kept)
                if segment == "post" or _show(value) == _show(kept) or (segment == "pre" and _starts(value, kept)):
                    continue  # (after the reduction, assignments make new values: codegen renames them)
                raise self.nest.error(
                    f"`{name}` is carried between reduction steps as {_show(kept)}, but this makes it "
                    f"{_show(value)}: a carried value keeps one type (start it with that type)",
                    self.stmts[stmt],
                )
            types[name] = jax.ShapeDtypeStruct(kept.shape, kept.dtype)
        return types

    def _last(self, segment: str, name: str) -> jax.ShapeDtypeStruct | None:
        """The type of `name` after the segment's last assignment of it, if it assigns it."""
        found = None
        for (seg, _), values in self.notes.items():
            if seg == segment and name in values:
                found = values[name]
        return found

    def _assignment(self, name: str) -> ast.stmt | None:
        return next((self.stmts[i] for (_, i), values in self.notes.items() if name in values), None)

    def _assigned(self) -> dict[int, dict[str, jax.ShapeDtypeStruct]]:
        assigned_: dict[int, dict[str, Any]] = {}
        for (_, stmt), values in self.notes.items():
            assigned_.setdefault(stmt, {}).update(values)  # a body statement: its type in the second step
        return assigned_


def _fresh(base: str, taken: set[str]) -> str:
    """`base2`, `base3`, ...: the first not in `taken`."""
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def _type(value: Any) -> jax.ShapeDtypeStruct | None:
    """The type of a value tile code computed, or None if it isn't an array or scalar."""
    try:
        aval = jax.typeof(value)
    except TypeError:
        return None
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)


def _starts(value: jax.ShapeDtypeStruct | None, kept: jax.ShapeDtypeStruct) -> bool:
    """Can `value` start a carried value of type `kept`? A number (a weakly typed scalar), or a scalar of
    its dtype: the kernel broadcasts it, which elementwise code at the first step does too."""
    return value is not None and not value.shape and (value.weak_type or value.dtype == kept.dtype)


def _show(t: jax.ShapeDtypeStruct | None) -> str:
    if t is None:
        return "not an array"
    return f"{t.dtype}{list(t.shape)}"
