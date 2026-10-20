"""Parse a kernel function into its loop nest. Syntax only: plan.py resolves the nest for concrete shapes.

A kernel function is

    def f(x, y, tiles=Tiles()):                    # the input arrays; optionally last, a tunable
        <prologue>                                 # assignments on shapes, run when the kernel is planned
        out = jnp.zeros((M, N), dtype)             # each output allocated once in the prologue
        for i in range(0, M, tiles.tm):            # parallel loops: every iteration writes its own tiles
            for j in range(0, N, TN):
                if <mask>:                         # optional: skip this output tile
                    continue
                <segment>                          # straight-line code, run once per tile
                for k in range(0, K, TK):          # any number of reduction loops, one after another:
                    if <mask>:                     #   optional: skip this step (it may read elements of
                        continue                   #   index arrays, `k >= lengths[b]`: decided at run time)
                    <body>                         #   runs once per step
                <segment>                          # between and after them, more straight-line code
                out = out.at[i : i + TM, j : j + TN].set(value)   # each output stored once, at the end
        return out                                 # the outputs, one name or a tuple of names

A tile's reduction loops run one after another (the kernel's steps for a tile are those of its first
reduction loop, then its second's, ...): a phase each. A loop around reduction loops is a repeat loop

                for l in range(L):                 # a repeat loop: its reduction loops' steps, L times over
                    <code>                         #   (one level of them: a repeat loop holds reduction loops
                    for k in range(0, K, TK):      #   and code, and values carry from one iteration to the
                        <body>                     #   next, as across steps)
                    <code>

whose steps are its reduction loops' steps, iteration after iteration: a whole model's layers in one
kernel, the next layer's weights fetched while this one's run.

An input read only element by element, `lengths[b]` (integer indices, no slices), is an index array: the
kernel keeps it whole in SMEM, and tile code, reduction masks and the indices of tile reads
(`w[expert[e], :, :]`) read its elements when the kernel runs.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .errors import KernelError
from .tilecode import Store, as_store, assigned, element_reads, flatten, reads

__all__ = ["Loop", "Nest", "Repeat", "parse"]

# Prologue statements: Python run on shapes when the kernel is planned.
_PROLOGUE_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Assert, ast.Pass)


@dataclass(frozen=True)
class Loop:
    """`for var in range(*bounds):`, possibly opening with `if mask: continue`."""

    var: str
    bounds: tuple[ast.expr, ...]
    mask: ast.expr | None
    node: ast.For


@dataclass(frozen=True)
class Repeat:
    """A repeat loop: `for var in range(...)` around the reduction loops `first` to `last` (exclusive), with
    its own code before the first and after the last (the code between them is in `Nest.segments`)."""

    loop: Loop
    first: int
    last: int
    before: tuple[ast.stmt, ...]
    after: tuple[ast.stmt, ...]


@dataclass(frozen=True)
class Nest:
    """A kernel function's loop nest, as written."""

    name: str
    filename: str
    function: ast.FunctionDef  # line numbers are the file's
    params: tuple[str, ...]  # the input arrays
    tunable: str | None  # the last parameter, if it has a default: the tunable autokernel optimizes over
    prologue: tuple[ast.stmt, ...]
    allocations: dict[str, ast.Assign]  # output -> the prologue statement allocating it
    outputs: tuple[str, ...]  # in the order the function returns them
    parallel: tuple[Loop, ...]  # outermost first
    reductions: tuple[Loop, ...]  # in the order they run
    bodies: tuple[tuple[ast.stmt, ...], ...]  # each reduction loop's body
    # The code before, between and after them (one more): at a repeat loop's first and last reduction
    # loops, the code outside the repeat loop (before and after it).
    segments: tuple[tuple[ast.stmt, ...], ...]
    repeats: tuple[Repeat, ...]
    stores: dict[str, Store]  # output -> its store: in the last segment, or a reduction loop's body
    stored_in: dict[str, int | None]  # output -> the reduction loop storing a tile of it per step (None: per tile)
    _store_nodes: dict[int, Store] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._store_nodes.update({id(store.node): store for store in self.stores.values()})

    @property
    def tile(self) -> tuple[ast.stmt, ...]:
        """Everything a tile runs, in order (as written): the segments and, between them, the reduction
        bodies, and the repeat loops' own code."""
        before = {rep.first: rep.before for rep in self.repeats}
        after = {rep.last - 1: rep.after for rep in self.repeats}
        out = self.segments[0]
        for r, (body, segment) in enumerate(zip(self.bodies, self.segments[1:], strict=True)):
            out += before.get(r, ()) + body + after.get(r, ()) + segment
        return out

    @property
    def loops(self) -> tuple[Loop, ...]:
        return self.parallel + self.reductions + tuple(rep.loop for rep in self.repeats)

    def repeat_of(self, reduction: int) -> Repeat | None:
        """The repeat loop around the `reduction`-th reduction loop, if any."""
        return next((rep for rep in self.repeats if rep.first <= reduction < rep.last), None)

    @property
    def runtime_masks(self) -> bool:
        """Do masks read elements of index arrays? Then they are evaluated when the kernel runs."""
        return any(loop.mask is not None and element_reads(loop.mask, self.params) for loop in self.reductions)

    def store(self, stmt: ast.stmt) -> Store | None:
        """The output store `stmt` is, if it is one."""
        return self._store_nodes.get(id(stmt))

    def error(self, message: str, node: ast.AST | None = None) -> KernelError:
        """A KernelError about `node` (a line of this function), naming the function."""
        return _error(self.name, self.filename, message, node)


def _error(name: str, filename: str, message: str, node: ast.AST | None) -> KernelError:
    return KernelError(f"{name}: {message}", filename=filename, node=node)


def parse(fn: Callable) -> Nest:
    """The loop nest of the kernel function `fn`; raises KernelError naming the line it can't compile."""
    name = getattr(fn, "__name__", repr(fn))
    try:
        lines, first_line = inspect.getsourcelines(fn)
        filename = inspect.getsourcefile(fn) or fn.__code__.co_filename
    except (OSError, TypeError) as e:
        raise KernelError(f"{name}: can't read its source; autokernel compiles a function's Python source") from e
    tree = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(tree, first_line - 1)
    fdef = tree.body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise _error(name, filename, "a kernel is a function defined with `def`", fdef)
    return _Parser(name, filename, fdef).parse()


class _Parser:
    def __init__(self, name: str, filename: str, fdef: ast.FunctionDef) -> None:
        self.name, self.filename, self.fdef = name, filename, fdef

    def error(self, message: str, node: ast.AST | None = None) -> KernelError:
        return _error(self.name, self.filename, message, node)

    def parse(self) -> Nest:
        params, tunable = self._params()
        body = self.fdef.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # the docstring
        at = next((i for i, stmt in enumerate(body) if isinstance(stmt, ast.For)), None)
        if at is None:
            raise self.error("a kernel is a loop nest over output tiles, but there is no `for` loop", self.fdef)
        prologue = tuple(body[:at])
        for stmt in prologue:
            if not isinstance(stmt, _PROLOGUE_STATEMENTS):
                raise self.error("statements before the loop nest compute shapes: assignments and asserts", stmt)
        if len(body) != at + 2 or not isinstance(body[at + 1], ast.Return):
            node = body[at + 1] if len(body) > at + 1 else body[at]
            raise self.error("the loop nest must be followed by `return` of the outputs, and nothing else", node)
        outputs = self._outputs(body[at + 1])  # type: ignore[arg-type]
        allocations = self._allocations(prologue, outputs, body[at + 1])

        parallel, tile = self._parallel_loops(body[at])  # type: ignore[arg-type]
        reductions, bodies, segments, repeats = self._split_tile(tile)
        stores, stored_in = self._stores(bodies, segments, repeats, outputs)
        nest = Nest(
            name=self.name,
            filename=self.filename,
            function=self.fdef,
            params=params,
            tunable=tunable,
            prologue=prologue,
            allocations=allocations,
            outputs=outputs,
            parallel=parallel,
            reductions=reductions,
            bodies=bodies,
            segments=segments,
            repeats=repeats,
            stores=stores,
            stored_in=stored_in,
        )
        self._check_names(nest)
        return nest

    # ------------------------------------------------------------------ the signature and the outputs

    def _params(self) -> tuple[tuple[str, ...], str | None]:
        """The input arrays' parameters, and the tunable's (the last parameter, given a default)."""
        args = self.fdef.args
        names = tuple(a.arg for a in [*args.posonlyargs, *args.args])
        if args.vararg or args.kwarg or args.kwonlyargs or len(args.defaults) > 1:
            raise self.error(
                "a kernel's parameters are its input arrays, then optionally a tunable with a default, `tiles=Tiles()`",
                args,
            )
        return (names[:-1], names[-1]) if args.defaults else (names, None)

    def _outputs(self, ret: ast.Return) -> tuple[str, ...]:
        value = ret.value
        elts = value.elts if isinstance(value, ast.Tuple) else [value]
        if not elts or not all(isinstance(e, ast.Name) for e in elts):
            raise self.error("return the outputs by name: `return out` or `return out1, out2`", ret)
        names = tuple(e.id for e in elts)  # type: ignore[union-attr]
        if len(set(names)) != len(names):
            raise self.error("each output is returned once", ret)
        return names

    def _allocations(
        self, prologue: tuple[ast.stmt, ...], outputs: tuple[str, ...], ret: ast.stmt
    ) -> dict[str, ast.Assign]:
        allocations: dict[str, ast.Assign] = {}
        for stmt in prologue:
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign | ast.AugAssign):
                targets = [stmt.target]
            else:
                continue  # assert, pass
            for name in sorted({n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)} & set(outputs)):
                single = isinstance(stmt, ast.Assign) and len(targets) == 1 and isinstance(targets[0], ast.Name)
                if name in allocations or not single:
                    raise self.error(f"allocate output `{name}` once, as `{name} = jnp.zeros(...)`", stmt)
                allocations[name] = stmt  # type: ignore[assignment]
        for name in outputs:
            if name not in allocations:
                raise self.error(f"output `{name}` must be allocated before the loops: `{name} = jnp.zeros(...)`", ret)
        return allocations

    # ------------------------------------------------------------------ loops

    def _loop(self, node: ast.For) -> tuple[Loop, list[ast.stmt]]:
        """The loop `node` is, and its body without the mask."""
        call = node.iter
        if not (
            isinstance(node.target, ast.Name)
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and 1 <= len(call.args) <= 3
            and not call.keywords
        ):
            raise self.error("loops are `for v in range(start, stop, step)`", node)
        if node.orelse:
            raise self.error("loops can't have an `else` block", node)
        body, mask = list(node.body), None
        first = body[0]
        if isinstance(first, ast.If) and not first.orelse and [type(s) for s in first.body] == [ast.Continue]:
            mask, body = first.test, body[1:]
        if not body:
            raise self.error("the loop has nothing to run", node)
        return Loop(node.target.id, tuple(call.args), mask, node), body

    def _parallel_loops(self, node: ast.For) -> tuple[tuple[Loop, ...], list[ast.stmt]]:
        """The perfectly nested loops over output tiles, and the innermost one's body: the tile."""
        loops = []
        while True:
            loop, body = self._loop(node)
            loops.append(loop)
            if len(body) == 1 and isinstance(body[0], ast.For):
                node = body[0]
                continue
            return tuple(loops), body

    def _split_tile(
        self, tile: list[ast.stmt]
    ) -> tuple[
        tuple[Loop, ...], tuple[tuple[ast.stmt, ...], ...], tuple[tuple[ast.stmt, ...], ...], tuple[Repeat, ...]
    ]:
        """(the reduction loops, their bodies, the segments before, between and after them, the repeat loops)."""
        reductions: list[Loop] = []
        bodies: list[tuple[ast.stmt, ...]] = []
        segments: list[tuple[ast.stmt, ...]] = []
        repeats: list[Repeat] = []
        segment: list[ast.stmt] = []
        for stmt in tile:
            if not isinstance(stmt, ast.For):
                segment.append(stmt)
                continue
            loop, body = self._loop(stmt)
            if not any(isinstance(s, ast.For) for s in body):
                self._check_statements(body)
                reductions.append(loop)
                bodies.append(tuple(body))
                segments.append(tuple(segment))
                segment = []
                continue
            if loop.mask is not None:
                raise self.error("a repeat loop (around reduction loops) can't be masked", loop.mask)
            inner, inner_bodies, inner_segments, nested = self._split_tile(body)
            if nested:
                raise self.error("repeat loops don't nest: a repeat loop holds reduction loops", nested[0].loop.node)
            segments.append(tuple(segment))
            segments.extend(inner_segments[1:-1])
            segment = []
            first = len(reductions)
            reductions.extend(inner)
            bodies.extend(inner_bodies)
            repeats.append(Repeat(loop, first, len(reductions), inner_segments[0], inner_segments[-1]))
        self._check_statements(segment)
        segments.append(tuple(segment))
        for done in segments[:-1]:
            self._check_statements(done)
        return tuple(reductions), tuple(bodies), tuple(segments), tuple(repeats)

    # ------------------------------------------------------------------ tile statements

    def _check_statements(self, stmts: Iterable[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                raise self.error("loops nest only as parallel loops around reduction (and repeat) loops", stmt)
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not _names_only(stmt.targets[0]):
                    raise self.error("tile code assigns names: `a = ...` or `a, b = ...`", stmt)
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, ast.Name):
                    raise self.error("tile code augments names: `a += ...`", stmt)
            elif isinstance(stmt, ast.With):
                if any(item.optional_vars or not isinstance(item.context_expr, ast.Call) for item in stmt.items):
                    raise self.error("tile code opens only `with jax.named_scope(name):` blocks", stmt)
                self._check_statements(stmt.body)
            else:
                raise self.error(
                    "tile code is assignments, output stores and `with jax.named_scope(...)` blocks; "
                    "use jnp.where instead of `if`",
                    stmt,
                )
            for node in ast.walk(stmt):
                if isinstance(node, ast.NamedExpr):
                    raise self.error("tile code can't assign inside expressions (`:=`)", node)

    def _stores(
        self,
        bodies: tuple[tuple[ast.stmt, ...], ...],
        segments: tuple[tuple[ast.stmt, ...], ...],
        repeats: tuple[Repeat, ...],
        outputs: tuple[str, ...],
    ) -> tuple[dict[str, Store], dict[str, int | None]]:
        """Each output's store, and where: after the reduction loops (a tile per tile), or in the body of a
        reduction loop outside repeat loops (a tile per step)."""
        in_repeats = {r for rep in repeats for r in range(rep.first, rep.last)}
        elsewhere = [s for part in segments[:-1] for s in part] + [s for rep in repeats for s in rep.before + rep.after]
        elsewhere += [s for r, body in enumerate(bodies) if r in in_repeats for s in body]
        for stmt in flatten(elsewhere):
            store = as_store(stmt)
            if store and store.output in outputs:
                raise self.error(
                    "store outputs after the reduction loops (a tile per tile), or in the body of a reduction loop "
                    "outside repeat loops (a tile per step)",
                    stmt,
                )
        places = [(None, segments[-1])] + [(r, body) for r, body in enumerate(bodies) if r not in in_repeats]
        stores: dict[str, Store] = {}
        stored_in: dict[str, int | None] = {}
        for r, stmts in places:
            for stmt in flatten(stmts):
                store = as_store(stmt)
                if store and store.output in outputs:
                    if store.output in stores:
                        raise self.error(f"output `{store.output}` is stored twice in a tile", stmt)
                    stores[store.output], stored_in[store.output] = store, r
        for name in outputs:
            if name not in stores:
                raise self.error(f"no tile stores output `{name}`: `{name} = {name}.at[...].set(...)`")
        return stores, stored_in

    # ------------------------------------------------------------------ names

    def _check_names(self, nest: Nest) -> None:
        params, outputs = set(nest.params), set(nest.outputs)
        for stmt in flatten(nest.tile):
            store = nest.store(stmt)
            for part in _own_expressions(stmt, store):
                bases = {
                    id(n.value)
                    for n in ast.walk(part)
                    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in params
                }
                fields = {
                    id(n.value)
                    for n in ast.walk(part)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == nest.tunable
                }
                for n in ast.walk(part):
                    if not isinstance(n, ast.Name):
                        continue
                    if n.id in params and id(n) not in bases:
                        raise self.error(f"`{n.id}` is an input: kernel code reads tiles of it, `{n.id}[...]`", n)
                    if n.id == nest.tunable and id(n) not in fields:
                        raise self.error(f"`{n.id}` is the kernel's tunable: tile code reads its fields, `{n.id}.x`", n)
                    if n.id in outputs:
                        what = "can't be read" if store else f"are written as `{n.id} = {n.id}.at[...].set(...)`"
                        raise self.error(f"output `{n.id}`'s tiles {what}", n)
        loop_vars = {loop.var for loop in nest.loops}
        parallel_vars = [loop.var for loop in nest.parallel]
        for depth, loop in enumerate(nest.loops):
            if loop.mask is None:
                continue
            # A parallel loop's mask sees the loops around it; a reduction's, the parallel loops, the repeat loop
            # around it and its own.
            reduction = depth >= len(parallel_vars)
            visible = {*parallel_vars, loop.var} if reduction else set(parallel_vars[: depth + 1])
            around = nest.repeat_of(depth - len(parallel_vars)) if reduction else None
            if around is not None:
                visible.add(around.loop.var)
            elements = {id(n.value) for n in element_reads(loop.mask, params)} if reduction else set()
            arrays = [
                n.id
                for n in ast.walk(loop.mask)
                if isinstance(n, ast.Name) and n.id in params | outputs and id(n) not in elements
            ]
            if arrays and reduction:
                raise self.error(
                    f"a reduction loop's mask reads loop variables, constants and elements of index arrays "
                    f"(`lengths[b]`), not `{arrays[0]}` itself",
                    loop.mask,
                )
            if arrays:
                raise self.error(
                    f"masks are evaluated when the kernel is built, on loop variables and constants: this one reads "
                    f"`{arrays[0]}`",
                    loop.mask,
                )
            inner = reads(loop.mask) & (loop_vars - visible)
            if inner:
                raise self.error(
                    f"a mask reads its own and enclosing loops' variables, not `{sorted(inner)[0]}`", loop.mask
                )
        written = assigned(nest.tile)
        for loop in nest.loops:
            if loop.var in written:
                raise self.error(f"loop variable `{loop.var}` can't be assigned in the tile", loop.node)
        repeat_code = [s for rep in nest.repeats for s in rep.before + rep.after]
        for r, reduction in enumerate(nest.reductions):
            elsewhere = [s for b, body in enumerate(nest.bodies) if b != r for s in body]
            for stmt in flatten([*elsewhere, *(s for segment in nest.segments for s in segment), *repeat_code]):
                if reduction.var in reads(stmt):
                    raise self.error(f"`{reduction.var}` is only defined inside its reduction loop", stmt)
        for rep in nest.repeats:
            inside = range(rep.first, rep.last)
            outside = [s for b, body in enumerate(nest.bodies) if b not in inside for s in body]
            outside += [s for i, segment in enumerate(nest.segments) if not rep.first < i < rep.last for s in segment]
            outside += [s for other in nest.repeats if other is not rep for s in other.before + other.after]
            for stmt in flatten(outside):  # (masks outside it can't read it: they read only enclosing loops)
                if rep.loop.var in reads(stmt):
                    raise self.error(f"`{rep.loop.var}` is only defined inside its repeat loop", stmt)


def _names_only(target: ast.expr) -> bool:
    if isinstance(target, ast.Tuple | ast.List):
        return all(_names_only(e) for e in target.elts)
    return isinstance(target, ast.Name)


def _own_expressions(stmt: ast.stmt, store: Store | None) -> list[ast.AST]:
    """The parts of `stmt` whose names it uses itself: a store's index and value, a `with`'s context (its
    body's statements are checked on their own), else the whole statement."""
    if store:
        return [*store.index, store.value]
    if isinstance(stmt, ast.With):
        return [item.context_expr for item in stmt.items]
    return [stmt]
