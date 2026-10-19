"""Frontend: parse a Python tile loop nest into a small IR.

The accepted shape of a function is::

    def f(x, y, ...):                           # every parameter is an input array
        <prologue>                              # shape-level constants (M, K = x.shape, ...)
        out = jnp.zeros((M, N))                 # one allocation per output
        for i in range(0, M, TM):               # parallel tile loops, outermost first; statements
            for j in range(0, N, TN):           # between them run once per tile
                if <mask>: continue             # optional tile mask
                acc = jnp.zeros((TM, TN))       # the tile body: statements and reduction loops
                for k in range(0, K, TK):       # a reduction loop (itself possibly a perfect nest),
                    if <mask>: continue         # with an optional reduction mask
                    acc += x[i:i+TM, k:k+TK] @ y[k:k+TK, j:j+TN]
                out = out.at[i:i+TM, j:j+TN].set(acc)   # one store per output, at the end
        return out                              # or a tuple of outputs

Masks are `if cond: continue` at the top of a loop body or `if cond:` around all of it. They depend on
loop variables and constants, and may read input arrays (`block_mask[i // T, k // T]`), which makes them
runtime masks. Expressions stay as AST until `analysis` evaluates them for concrete input shapes.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

from .errors import AutoKernelError

__all__ = ["Loop", "LoopNest", "Output", "Reduction", "Store", "assigned_name", "parse"]


def assigned_name(s: ast.stmt) -> str:
    """The variable a tile-body statement (`name = ...` or `name op= ...`) assigns."""
    return s.targets[0].id if isinstance(s, ast.Assign) else s.target.id


@dataclass
class Loop:
    """`for var in range(start, stop, step)`, with the range arguments still as source expressions."""

    var: str
    start: ast.expr
    stop: ast.expr
    step: ast.expr
    node: ast.For


@dataclass
class Reduction:
    """A reduction loop: a perfect nest of loops (outermost first) around an optional mask and a body."""

    loops: list[Loop]
    guard: ast.expr | None  # the body runs only where this is true
    guard_node: ast.stmt | None
    body: list[ast.stmt]


@dataclass
class Output:
    var: str
    alloc: ast.expr  # the allocation expression, e.g. `jnp.zeros((M, N))`


@dataclass
class Store:
    """`out = out.at[index].set(value)`: the tile's contribution to one output."""

    out: str
    index: ast.expr
    value: ast.expr
    node: ast.stmt


@dataclass
class LoopNest:
    """A parsed tile loop nest.

    The tile body is `segments[0]`, `reductions[0]`, `segments[1]`, ..., `reductions[-1]`, `segments[-1]`,
    then `stores`. Statements found between parallel loops are already folded into `segments[0]`.
    """

    fn_name: str
    first_line: int
    params: list[str]
    prologue: list[ast.stmt]
    outputs: list[Output]
    returns_tuple: bool
    parallel: list[Loop]
    guard: ast.expr | None  # tile mask: the tile is computed only where this is true
    guard_node: ast.stmt | None
    segments: list[list[ast.stmt]]
    reductions: list[Reduction]
    stores: list[Store]

    def error(self, node: ast.AST | None, msg: str) -> AutoKernelError:
        if node is not None and hasattr(node, "lineno"):
            return AutoKernelError(f"{self.fn_name} (line {self.first_line + node.lineno - 1}): {msg}")
        return AutoKernelError(f"{self.fn_name}: {msg}")

    @property
    def loops(self) -> list[Loop]:
        return self.parallel + [lp for r in self.reductions for lp in r.loops]

    @property
    def out_vars(self) -> list[str]:
        return [o.var for o in self.outputs]

    def tile_statements(self) -> list[ast.stmt]:
        return [s for seg in self.segments for s in seg] + [s for r in self.reductions for s in r.body]


def parse(fn: Callable) -> LoopNest:
    """Parse `fn`'s source into a LoopNest, or raise AutoKernelError naming the offending line."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise AutoKernelError(f"can't read the source of {fn!r}: {e}") from e
    fdef = ast.parse(src).body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise AutoKernelError(f"{fn.__name__}: expected a plain `def` function")
    return _Parser(fn.__name__, fn.__code__.co_firstlineno, fdef).parse()


class _Parser:
    def __init__(self, name: str, first_line: int, fdef: ast.FunctionDef):
        self.name, self.first_line, self.fdef = name, first_line, fdef

    def err(self, node: ast.AST | None, msg: str) -> AutoKernelError:
        if node is not None and hasattr(node, "lineno"):
            return AutoKernelError(f"{self.name} (line {self.first_line + node.lineno - 1}): {msg}")
        return AutoKernelError(f"{self.name}: {msg}")

    # ------------------------------------------------------------------ top level

    def parse(self) -> LoopNest:
        fdef = self.fdef
        a = fdef.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
            raise self.err(fdef, "only plain positional array parameters are supported")
        self.params = [p.arg for p in a.args]

        body = list(fdef.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        if not body or not isinstance(body[-1], ast.Return):
            raise self.err(body[-1] if body else fdef, "function must end with `return <output>` or a tuple of outputs")
        ret = body[-1].value
        names = ret.elts if isinstance(ret, ast.Tuple) else [ret]
        if not names or not all(isinstance(n, ast.Name) for n in names):
            raise self.err(body[-1], "return the output arrays by name: `return out` or `return out1, out2`")
        self.out_vars = [n.id for n in names]
        if len(set(self.out_vars)) != len(self.out_vars):
            raise self.err(body[-1], "each output may be returned only once")
        body = body[:-1]

        fors = [s for s in body if isinstance(s, ast.For)]
        if len(fors) != 1 or body[-1] is not fors[0]:
            raise self.err(
                fors[1] if len(fors) > 1 else fdef, "expected one top-level tile loop nest before the return"
            )
        top = fors[0]

        prologue, allocs = [], {}
        for s in body[:-1]:
            if self._is_assign_to(s, self.out_vars):
                allocs[s.targets[0].id] = s.value
                continue
            if not isinstance(s, (ast.Assign, ast.AnnAssign, ast.Assert, ast.Pass)):
                raise self.err(s, "only simple constant assignments / asserts may precede the loop nest")
            for t in ast.walk(s):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id in self.params:
                    raise self.err(s, f"the prologue may not rebind input `{t.id}`")
            prologue.append(s)
        for var in self.out_vars:
            if var not in allocs:
                raise self.err(top, f"missing output allocation `{var} = jnp.zeros(...)` before the loop nest")
        outputs = [Output(var, allocs[var]) for var in self.out_vars]

        parallel, hoisted, inner = self._parallel_loops(top)
        guard, guard_node, inner = self._split_guard(inner, "tile mask")
        inner, stores = self._split_stores(inner, parallel[-1].node)
        segments, reductions = self._tile_body(hoisted + inner)

        nest = LoopNest(
            fn_name=self.name,
            first_line=self.first_line,
            params=self.params,
            prologue=prologue,
            outputs=outputs,
            returns_tuple=isinstance(ret, ast.Tuple),
            parallel=parallel,
            guard=guard,
            guard_node=guard_node,
            segments=segments,
            reductions=reductions,
            stores=stores,
        )
        self._validate(nest)
        return nest

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _is_assign_to(s: ast.stmt, names: list[str]) -> bool:
        return (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id in names
        )

    def _loop(self, node: ast.For) -> Loop:
        if node.orelse or not isinstance(node.target, ast.Name):
            raise self.err(node, "tile loops must be `for v in range(start, stop, step)`")
        it = node.iter
        ok = (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
            and len(it.args) == 3
        )
        if not ok:
            raise self.err(node, "tile loops must use the 3-argument form `range(start, stop, step)`")
        return Loop(node.target.id, *it.args, node=node)

    def _is_tile_body(self, body: list[ast.stmt]) -> bool:
        """Does this loop body hold the output stores (possibly under a tile mask)?"""
        if body and isinstance(body[0], ast.If) and not body[0].orelse:
            guarded = body[1:] if len(body[0].body) == 1 and isinstance(body[0].body[0], ast.Continue) else body[0].body
            if any(self._is_store(s) for s in guarded):
                return True
        return any(self._is_store(s) for s in body)

    def _parallel_loops(self, top: ast.For) -> tuple[list[Loop], list[ast.stmt], list[ast.stmt]]:
        """Walk down the parallel loops: the loops enclosing the output stores.

        Statements before an inner parallel loop are hoisted into the tile body: they only read outer
        loop variables, so recomputing them per tile gives the same values.
        """
        loops, hoisted, node = [], [], top
        while True:
            loops.append(self._loop(node))
            body = node.body
            if self._is_tile_body(body):
                return loops, hoisted, body
            if not body or not isinstance(body[-1], ast.For) or any(isinstance(s, ast.For) for s in body[:-1]):
                raise self.err(
                    node, "the innermost parallel loop must end with the output stores `out = out.at[...].set(...)`"
                )
            hoisted += body[:-1]
            node = body[-1]

    def _split_guard(self, body: list[ast.stmt], what: str) -> tuple[ast.expr | None, ast.stmt | None, list[ast.stmt]]:
        """Peel a mask off a loop body: `if cond: <body>` or a leading `if cond: continue`.

        Returns (condition under which the body runs, the `if` node, the remaining body).
        """
        if not body or not isinstance(body[0], ast.If) or body[0].orelse:
            return None, None, body
        node = body[0]
        if len(node.body) == 1 and isinstance(node.body[0], ast.Continue):
            cond, rest = ast.copy_location(ast.UnaryOp(ast.Not(), node.test), node.test), body[1:]
        elif len(body) == 1:
            cond, rest = node.test, node.body
        else:
            raise self.err(
                node, f"a {what} must be `if cond:` around the whole loop body, or a leading `if cond: continue`"
            )
        if not rest:
            raise self.err(node, f"the body under the {what} is empty")
        return cond, node, rest

    def _is_store(self, s: ast.stmt) -> bool:
        if not self._is_assign_to(s, self.out_vars):
            return False
        v = s.value
        return (
            isinstance(v, ast.Call)
            and len(v.args) == 1
            and not v.keywords
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "set"
            and isinstance(v.func.value, ast.Subscript)
            and isinstance(v.func.value.value, ast.Attribute)
            and v.func.value.value.attr == "at"
            and isinstance(v.func.value.value.value, ast.Name)
            and v.func.value.value.value.id == s.targets[0].id
        )

    def _split_stores(self, body: list[ast.stmt], loop: ast.For) -> tuple[list[ast.stmt], list[Store]]:
        """The tile body ends with one store per output."""
        n = len(body)
        while n and self._is_store(body[n - 1]):
            n -= 1
        stores = []
        for s in body[n:]:
            call = s.value
            stores.append(Store(s.targets[0].id, call.func.value.slice, call.args[0], s))
        written = [st.out for st in stores]
        for var in self.out_vars:
            if written.count(var) != 1:
                raise self.err(
                    body[-1] if body else loop,
                    f"the innermost parallel loop must end with exactly one `{var} = {var}.at[...].set(...)`",
                )
        return body[:n], stores

    def _tile_body(self, body: list[ast.stmt]) -> tuple[list[list[ast.stmt]], list[Reduction]]:
        segments: list[list[ast.stmt]] = [[]]
        reductions: list[Reduction] = []
        for s in body:
            if isinstance(s, ast.For):
                reductions.append(self._reduction(s))
                segments.append([])
            else:
                segments[-1].append(s)
        return segments, reductions

    def _reduction(self, node: ast.For) -> Reduction:
        loops, body = [self._loop(node)], node.body
        while len(body) == 1 and isinstance(body[0], ast.For):  # a perfect nest of reduction loops
            loops.append(self._loop(body[0]))
            body = body[0].body
        guard, guard_node, body = self._split_guard(body, "reduction mask")
        for s in body:
            if isinstance(s, ast.For):
                raise self.err(s, "reduction loops may only nest perfectly (a loop whose body is just another loop)")
        return Reduction(loops, guard, guard_node, body)

    # ------------------------------------------------------------------ validation

    def _validate(self, nest: LoopNest) -> None:
        loop_vars = [lp.var for lp in nest.loops]
        if len(set(loop_vars)) != len(loop_vars):
            raise self.err(nest.parallel[0].node, "loop variables must have distinct names")
        for s in nest.tile_statements():
            ok = (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)) or (
                isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name)
            )
            if not ok:
                raise self.err(s, "loop bodies may only contain `name = expr` / `name op= expr` statements")
            target = assigned_name(s)
            if target in nest.out_vars:
                raise self.err(s, f"`{target}` may only be written by its final `.at[...].set(...)`")
            if target in self.params or target in loop_vars:
                raise self.err(s, f"can't assign to input / loop variable `{target}` inside the loop nest")
        temporaries = {assigned_name(s) for s in nest.tile_statements()}
        masks = [(nest.guard, nest.guard_node, "tile mask")]
        masks += [(r.guard, r.guard_node, "reduction mask") for r in nest.reductions]
        for cond, node, what in masks:
            if cond is None:
                continue
            for n in ast.walk(cond):
                if isinstance(n, ast.Name) and (n.id in nest.out_vars or n.id in temporaries):
                    raise self.err(
                        node, f"the {what} can only depend on loop variables, constants and inputs, not `{n.id}`"
                    )
