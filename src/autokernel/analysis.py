"""Specialise a parsed loop nest to concrete input shapes.

Everything here is analysis: it evaluates the user's loop bounds, slices, tile mask and output
allocation at trace time and records what the kernel must do, without generating any code.
`codegen` turns the resulting `Specialization` into Pallas source.
"""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal

from .errors import AutoKernelError
from .frontend import LoopNest

__all__ = ["Accumulator", "Block", "LoopBounds", "Operand", "Specialization", "TileMask", "specialize"]

Scalar = bool | int | float

# Primitives that keep "every element is the same scalar" true, used to find constant fills.
_SHAPE_ONLY_OPS = frozenset({"broadcast_in_dim", "reshape", "transpose", "copy", "copy_p", "squeeze", "expand_dims"})
_ELEMENTWISE_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "max",
        "min",
        "neg",
        "abs",
        "sign",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "pow",
        "integer_pow",
        "floor",
        "ceil",
        "round",
        "convert_element_type",
    }
)


@dataclass(frozen=True)
class LoopBounds:
    """A tile loop's evaluated `range(start, stop, step)` and the grid axis it maps to."""

    var: str
    axis: int
    start: int
    stop: int
    step: int

    @property
    def grid(self) -> int:
        return (self.stop - self.start) // self.step

    @property
    def blocks(self) -> range:
        """Absolute block indices the loop visits (start and stop are multiples of step)."""
        return range(self.start // self.step, self.stop // self.step)


@dataclass(frozen=True)
class Block:
    """One BlockSpec: which tile of an array a grid step sees."""

    shape: tuple[int, ...]
    index: tuple[str, ...]  # per-dim block-index expression, in terms of index-map arguments
    loops: tuple[str | None, ...]  # loop variable tiling each dim (None for full / constant dims)
    text: str  # the original slice source, for comments and error messages


@dataclass(frozen=True)
class Operand:
    """An input read through one particular tiling (an input sliced two ways is two operands)."""

    param: str
    ref: str
    block: Block
    full_shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class Accumulator:
    """A value carried across the reduction loop, kept in a VMEM scratch buffer."""

    name: str
    ref: str
    init: jax.ShapeDtypeStruct  # as the user's code initialises it
    scratch: jax.ShapeDtypeStruct  # as stored: the steady-state dtype, narrow floats widened to f32


@dataclass(frozen=True)
class TileMask:
    """Which output tiles a tile mask keeps, as absolute block indices of the parallel loops."""

    active: tuple[tuple[int, ...], ...]  # computed tiles, in loop order
    constant: tuple[tuple[int, ...], ...]  # skipped tiles: they keep the output allocation's value
    fill: Scalar | None  # that value when it's a known constant, else None
    tables: dict[str, str]  # parallel loop var -> name of its scalar-prefetch table of block indices
    grid_var: str  # index-map argument for the flattened tile axis

    @property
    def total(self) -> int:
        return len(self.active) + len(self.constant)


@dataclass
class Specialization:
    """A loop nest with every trace-time fact resolved for one set of input shapes/dtypes."""

    nest: LoopNest
    specs: list[jax.ShapeDtypeStruct]
    filename: str
    consts: dict[str, Any]  # names the user's code can see, minus the inputs
    loops: dict[str, LoopBounds]
    out: jax.ShapeDtypeStruct
    out_block: Block
    out_ref: str
    operands: list[Operand]
    accumulators: list[Accumulator]
    mask: TileMask | None
    names: Namer
    code: UserCode

    @property
    def reduction(self) -> LoopBounds | None:
        return self.loops[self.nest.reduction.var] if self.nest.reduction else None

    def operand(self, node: ast.Subscript) -> Operand:
        """The operand a tile slice of an input refers to."""
        param = node.value.id  # type: ignore[attr-defined]
        blk = _resolve_block(
            self, node.slice, self._input_shape(param), _slice_label(param, node.slice), set(self.loops)
        )
        for op in self.operands:
            if (op.param, op.block.shape, op.block.index) == (param, blk.shape, blk.index):
                return op
        raise AssertionError(f"operand for {ast.unparse(node)} was not collected")  # analysis bug

    def _input_shape(self, param: str) -> tuple[int, ...]:
        return tuple(self.specs[self.nest.params.index(param)].shape)


class Namer:
    """Hands out identifiers for generated code that can't collide with the user's names."""

    def __init__(self, taken: set[str]):
        self._taken = set(taken)

    def fresh(self, base: str) -> str:
        name = base
        while name in self._taken:
            name += "_"
        self._taken.add(name)
        return name


class UserCode:
    """Compiles fragments of the user's function so tracebacks point at their file and lines."""

    def __init__(self, nest: LoopNest, filename: str):
        self.nest, self.filename = nest, filename

    def compile(self, node: ast.AST | list[ast.stmt], mode: str):
        node = copy.deepcopy(node)
        wrapped = ast.Expression(node) if mode == "eval" else ast.Module(body=node, type_ignores=[])
        ast.increment_lineno(wrapped, self.nest.first_line - 1)
        ast.fix_missing_locations(wrapped)
        return compile(wrapped, self.filename, mode)

    def eval_int(self, node: ast.expr, what: str, env: dict[str, Any]) -> int:
        try:
            value = eval(self.compile(node, "eval"), dict(env))
        except Exception as e:
            raise self.nest.error(
                node, f"can't evaluate {what} `{ast.unparse(node)}` at trace time ({type(e).__name__}: {e})"
            ) from None
        if not isinstance(value, (int, np.integer)):
            raise self.nest.error(node, f"{what} `{ast.unparse(node)}` must be a Python int, got {value!r}")
        return int(value)


def specialize(nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct]) -> Specialization:
    """Resolve `nest` for inputs of the given shapes and dtypes, validating it along the way."""
    specs = list(specs)
    if len(specs) != len(nest.params):
        raise AutoKernelError(f"{nest.fn_name} takes {len(nest.params)} arrays, got {len(specs)}")
    code = UserCode(nest, inspect.getsourcefile(fn) or "<autokernel>")

    # Names the user's code can see: module globals, closure cells, then the prologue's constants.
    # Inputs are bound to ShapeDtypeStructs, so the prologue can only use their shapes and dtypes.
    env: dict[str, Any] = dict(fn.__globals__)
    env.update(inspect.getclosurevars(fn).nonlocals)
    env.update(zip(nest.params, specs, strict=True))
    if nest.prologue:
        exec(code.compile(nest.prologue, "exec"), env)
    consts = {k: v for k, v in env.items() if k not in nest.params}

    masked = nest.guard is not None
    loops = {}
    for axis, loop in enumerate(nest.loops):
        start, stop, step = (
            code.eval_int(e, f"`range` argument of loop `{loop.var}`", env) for e in (loop.start, loop.stop, loop.step)
        )
        if step <= 0 or start % step or (stop - start) % step or stop <= start:
            raise nest.error(
                loop.node,
                f"loop `{loop.var}`: range({start}, {stop}, {step}) must be "
                "non-empty, positive-step and aligned (start and stop-start divisible by step)",
            )
        if masked:  # the parallel loops collapse into one grid axis over the active tiles
            axis = 1 if loop is nest.reduction else 0
        loops[loop.var] = LoopBounds(loop.var, axis, start, stop, step)

    user_names = {n.id for s in _kernel_statements(nest) for n in ast.walk(s) if isinstance(n, ast.Name)}
    names = Namer(user_names | set(nest.params) | set(consts))
    spec = Specialization(
        nest=nest,
        specs=specs,
        filename=code.filename,
        consts=consts,
        loops=loops,
        out=None,
        out_block=None,
        out_ref="",
        operands=[],
        accumulators=[],  # type: ignore[arg-type]
        mask=None,
        names=names,
        code=code,
    )
    if masked:
        spec.mask = _tile_mask(spec)  # before any Block is resolved: index maps read its tables
    spec.out = _output_shape(spec)
    spec.out_block = _output_block(spec)
    spec.operands = _collect_operands(spec)
    spec.accumulators = _accumulators(spec)
    spec.out_ref = names.fresh(f"{nest.out_var}_ref")
    if spec.mask is not None:
        spec.mask = TileMask(
            spec.mask.active, spec.mask.constant, _constant_fill(spec), spec.mask.tables, spec.mask.grid_var
        )
    return spec


# --------------------------------------------------------------------------- tiles and slices


def _kernel_statements(nest: LoopNest) -> list[ast.stmt]:
    return nest.pre + nest.body + nest.post + [nest.store_node]


def _slice_label(name: str, index: ast.expr) -> str:
    return f"`{name}[{ast.unparse(index)}]`"


def _block_index_src(var: str, offset: int) -> str:
    return var if offset == 0 else f"{var} + {offset}"


def _resolve_block(spec: Specialization, index: ast.expr, full: tuple[int, ...], what: str, allowed: set[str]) -> Block:
    """Turn a tile slice like `x[i:i+T, :]` into a Block, checking it against the loops."""
    nest = spec.nest
    env = {**spec.consts, **dict(zip(nest.params, spec.specs, strict=True))}  # tile sizes may use x.shape
    dims = list(index.elts) if isinstance(index, ast.Tuple) else [index]
    if len(dims) > len(full):
        raise nest.error(index, f"{what}: {len(dims)} indices for a rank-{len(full)} array")
    dims += [ast.Slice()] * (len(full) - len(dims))
    shape: list[int] = []
    idx: list[str] = []
    loops: list[str | None] = []
    for d, (sl, size) in enumerate(zip(dims, full, strict=True)):
        if not isinstance(sl, ast.Slice) or sl.step is not None:
            raise nest.error(
                index,
                f"{what}: dim {d} must be a tile slice `v:v+T`, `c:c+T` or `:` "
                "(integer indexing and strided slices aren't supported)",
            )
        if sl.lower is None and sl.upper is None:
            shape.append(size)
            idx.append("0")
            loops.append(None)
            continue
        var = sl.lower.id if isinstance(sl.lower, ast.Name) and sl.lower.id in spec.loops else None
        if var is not None:
            if var not in allowed:
                raise nest.error(index, f"{what}: can't be tiled by loop `{var}` here")
            tile = _tile_size(sl.upper, var)
            if tile is None:
                raise nest.error(index, f"{what}: dim {d} must look like `{var}:{var} + T`")
            lb = spec.loops[var]
            t = spec.code.eval_int(tile, "tile size", env)
            if t != lb.step:
                raise nest.error(
                    index, f"{what}: tile size {t} in dim {d} must equal the step ({lb.step}) of loop `{var}`"
                )
            if lb.stop > size:
                raise nest.error(index, f"{what}: loop `{var}` runs to {lb.stop} but dim {d} only has size {size}")
            shape.append(t)
            loops.append(var)
            if spec.mask is not None and var in spec.mask.tables:  # tables hold absolute block indices
                idx.append(f"{spec.mask.tables[var]}[{spec.mask.grid_var}]")
            else:
                idx.append(_block_index_src(var, lb.start // t))
            continue
        lo = 0 if sl.lower is None else spec.code.eval_int(sl.lower, "slice bound", env)
        hi = size if sl.upper is None else spec.code.eval_int(sl.upper, "slice bound", env)
        t = hi - lo
        if t <= 0 or lo % t or hi > size:
            raise nest.error(
                index, f"{what}: constant slice {lo}:{hi} in dim {d} must be a whole aligned block inside the array"
            )
        shape.append(t)
        idx.append(str(lo // t))
        loops.append(None)
    text = ", ".join(map(ast.unparse, index.elts)) if isinstance(index, ast.Tuple) else ast.unparse(index)
    return Block(tuple(shape), tuple(idx), tuple(loops), text)


def _tile_size(upper: ast.expr | None, var: str) -> ast.expr | None:
    """T in an upper bound written `var + T` or `T + var`."""
    if isinstance(upper, ast.BinOp) and isinstance(upper.op, ast.Add):
        if isinstance(upper.left, ast.Name) and upper.left.id == var:
            return upper.right
        if isinstance(upper.right, ast.Name) and upper.right.id == var:
            return upper.left
    return None


def _input_subscripts(nest: LoopNest) -> Iterator[ast.Subscript]:
    """Every read of an input inside the loop nest, in source order. Rejects reads that aren't slices."""
    found: list[ast.Subscript] = []

    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in nest.params:
                found.append(node)
                self.visit(node.slice)
            else:
                self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in nest.params:
                raise nest.error(
                    node,
                    f"input `{node.id}` may only be read through a tile slice like "
                    f"`{node.id}[i:i+T, ...]` inside the loop nest",
                )
            if node.id == nest.out_var:
                raise nest.error(node, f"the output `{node.id}` can't be read inside the loop nest")

    visitor = Visitor()
    for stmt in nest.pre + nest.body + nest.post:
        visitor.visit(stmt)
    visitor.visit(nest.store_value)
    return iter(found)


def _collect_operands(spec: Specialization) -> list[Operand]:
    operands: list[Operand] = []
    for node in _input_subscripts(spec.nest):
        param = node.value.id  # type: ignore[attr-defined]
        shape = spec._input_shape(param)
        blk = _resolve_block(spec, node.slice, shape, _slice_label(param, node.slice), set(spec.loops))
        if any((op.param, op.block.shape, op.block.index) == (param, blk.shape, blk.index) for op in operands):
            continue
        existing = sum(op.param == param for op in operands)
        ref = spec.names.fresh(f"{param}_ref" if existing == 0 else f"{param}_ref{existing}")
        dtype = np.dtype(spec.specs[spec.nest.params.index(param)].dtype)
        operands.append(Operand(param, ref, blk, shape, dtype))
    return operands


# --------------------------------------------------------------------------- output


def _output_shape(spec: Specialization) -> jax.ShapeDtypeStruct:
    nest = spec.nest
    alloc = spec.code.compile(nest.out_alloc, "eval")
    try:
        out = jax.eval_shape(
            lambda *xs: eval(alloc, {**spec.consts, **dict(zip(nest.params, xs, strict=True))}), *spec.specs
        )
    except Exception as e:
        raise nest.error(nest.out_alloc, f"couldn't evaluate the output allocation ({type(e).__name__}: {e})") from None
    if not isinstance(out, jax.ShapeDtypeStruct):
        raise nest.error(nest.out_alloc, "the output allocation must produce a single array")
    return out


def _output_block(spec: Specialization) -> Block:
    nest = spec.nest
    par_vars = {lp.var for lp in nest.parallel}
    out_block = _resolve_block(spec, nest.store_index, tuple(spec.out.shape), f"`{nest.out_var}.at[...]`", par_vars)
    for var in sorted(par_vars - {v for v in out_block.loops if v}):
        raise nest.error(
            nest.store_node,
            f"parallel loop `{var}` must tile one of the output's dims, "
            "otherwise grid steps race on the same output block",
        )
    for d, (var, n, t) in enumerate(zip(out_block.loops, spec.out.shape, out_block.shape, strict=True)):
        covered = (spec.loops[var].start, spec.loops[var].stop) == (0, n) if var else t == n
        if not covered:
            raise nest.error(
                nest.store_node,
                f"the grid doesn't write all of output dim {d} (size {n}); "
                "every output element must be produced by the kernel",
            )
    return out_block


# --------------------------------------------------------------------------- accumulators


def _accumulators(spec: Specialization) -> list[Accumulator]:
    """Shapes of loop-carried values after initialisation and after one reduction step."""
    nest = spec.nest
    if not nest.carried:
        return []
    pre_code, body_code = spec.code.compile(nest.pre, "exec"), spec.code.compile(nest.body, "exec")

    def run(*arrays):
        ns = dict(spec.consts)
        ns.update(zip(nest.params, arrays, strict=True))
        ns.update({v: lb.start for v, lb in spec.loops.items()})
        exec(pre_code, ns)
        init = {c: ns[c] for c in nest.carried}
        exec(body_code, ns)
        return init, {c: ns[c] for c in nest.carried}

    try:
        init, step = jax.eval_shape(run, *spec.specs)
    except Exception as e:
        raise nest.error(nest.reduction.node, f"couldn't trace one reduction step ({type(e).__name__}: {e})") from None

    accumulators = []
    for c in nest.carried:
        if init[c].shape != step[c].shape:
            raise nest.error(
                nest.reduction.node,
                f"accumulator `{c}` changes shape from {init[c].shape} to {step[c].shape} inside the reduction loop",
            )
        scratch = step[c]
        # Narrow float accumulators live in f32 scratch: the MXU produces f32 anyway.
        if jnp.issubdtype(scratch.dtype, jnp.floating) and np.dtype(scratch.dtype).itemsize < 4:
            scratch = jax.ShapeDtypeStruct(scratch.shape, jnp.float32)
        accumulators.append(Accumulator(c, spec.names.fresh(f"{c}_ref"), init[c], scratch))
    return accumulators


# --------------------------------------------------------------------------- tile masks


def _tile_mask(spec: Specialization) -> TileMask:
    """Evaluate the mask for every tile at trace time; the grid will walk only the active ones."""
    nest = spec.nest
    assert nest.guard is not None and nest.guard_node is not None
    if nest.reduction and any(isinstance(n, ast.Name) and n.id == nest.reduction.var for n in ast.walk(nest.guard)):
        raise nest.error(nest.guard_node, f"the tile mask can't depend on the reduction loop `{nest.reduction.var}`")
    guard = spec.code.compile(nest.guard, "eval")
    bounds = [spec.loops[lp.var] for lp in nest.parallel]
    active, constant = [], []
    for blocks in itertools.product(*(lb.blocks for lb in bounds)):
        env = dict(spec.consts)
        env.update({lb.var: b * lb.step for lb, b in zip(bounds, blocks, strict=True)})
        try:
            keep = bool(eval(guard, env))
        except Exception as e:
            raise nest.error(
                nest.guard_node,
                f"can't evaluate the tile mask `{ast.unparse(nest.guard)}` at trace time ({type(e).__name__}: {e})",
            ) from None
        (active if keep else constant).append(blocks)
    if not active:
        raise nest.error(nest.guard_node, "the tile mask skips every tile")
    grid_var = spec.names.fresh("tile")
    tables = {lp.var: spec.names.fresh(f"{lp.var}_blocks") for lp in nest.parallel}
    return TileMask(tuple(active), tuple(constant), None, tables, grid_var)


def _constant_fill(spec: Specialization) -> Scalar | None:
    """The scalar every element of the output allocation holds, or None if it isn't uniform.

    Traces the allocation and follows values that are the same everywhere: a broadcast scalar,
    elementwise ops on such values, reshapes of them. So `jnp.zeros`, `jnp.full(s, -inf)` and
    `jnp.zeros(s) + 1` all count, however the code spells them.
    """
    nest = spec.nest
    alloc = spec.code.compile(nest.out_alloc, "eval")
    try:
        closed = jax.make_jaxpr(lambda *xs: eval(alloc, {**spec.consts, **dict(zip(nest.params, xs, strict=True))}))(
            *spec.specs
        )
    except Exception:
        return None
    jaxpr = closed.jaxpr
    uniform = {
        v: np.asarray(c).reshape(()) for v, c in zip(jaxpr.constvars, closed.consts, strict=True) if np.size(c) == 1
    }

    def read(atom):
        if isinstance(atom, Literal):
            return np.asarray(atom.val)
        return uniform[atom]  # KeyError: depends on something that isn't uniform

    try:
        for eqn in jaxpr.eqns:
            name, ins = eqn.primitive.name, eqn.invars
            if name == "empty":  # unspecified contents: any constant is a valid choice
                val = np.zeros((), eqn.outvars[0].aval.dtype)
            elif name in _SHAPE_ONLY_OPS:
                val = read(ins[0])
            elif name in _ELEMENTWISE_OPS:
                val = np.asarray(eqn.primitive.bind(*[jnp.asarray(read(a)) for a in ins], **eqn.params))
            else:
                return None
            uniform[eqn.outvars[0]] = val.reshape(())
        return read(jaxpr.outvars[0]).item()
    except (KeyError, TypeError, ValueError):
        return None
