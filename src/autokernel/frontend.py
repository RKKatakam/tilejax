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

The tiles must write every element of an output, except when its allocation is an input (`out = x`):
then the elements no tile writes keep the input's values, as they would in Python.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from . import scopes
from .errors import AutoKernelError

__all__ = [
    "Loop",
    "LoopNest",
    "Output",
    "Reduction",
    "Store",
    "assigned_name",
    "free_loads",
    "function_ast",
    "is_simple_assignment",
    "parse",
    "parse_function",
    "reads",
    "scope_bound",
]


def is_simple_assignment(s: ast.stmt) -> bool:
    """`name = expr` or `name op= expr`: the only statements allowed inside the loop nest."""
    if isinstance(s, ast.Assign):
        return len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
    return isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name)


def assigned_name(s: ast.stmt) -> str:
    """The variable a simple assignment assigns."""
    return s.targets[0].id if isinstance(s, ast.Assign) else s.target.id


def _error(fn_name: str, first_line: int, node: ast.AST | None, msg: str) -> AutoKernelError:
    """An error naming the function and, when `node` has one, the line in the user's source file."""
    if node is not None and hasattr(node, "lineno"):
        return AutoKernelError(f"{fn_name} (line {first_line + node.lineno - 1}): {msg}")
    return AutoKernelError(f"{fn_name}: {msg}")


@dataclass
class Loop:
    """`for var in range(...)`, with the range arguments still as source expressions."""

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
    # The allocation is an input array (`out = x`): elements no tile writes keep its values, so the tiles
    # needn't cover the whole output (a program passes an output's current value this way).
    keeps: bool = False


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
    first_line: int  # of the function's source (its first decorator) in its file
    fdef: ast.FunctionDef  # the whole function, lines relative to `first_line`
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
    # Reads of a value only masked reduction loops assign: (its name, the statement reading it, the indices of
    # those reductions). Analysis checks that every tile runs a step of one of them, as Python needs.
    masked_reads: list[tuple[str, ast.AST, tuple[int, ...]]] = field(default_factory=list)
    # Compiled fragments of the nest, by what they depend on: specialising it for many tile sizes (tuning)
    # reuses them.
    cache: dict[Any, Any] = field(default_factory=dict, repr=False, compare=False)

    def error(self, node: ast.AST | None, msg: str) -> AutoKernelError:
        return _error(self.fn_name, self.first_line, node, msg)

    @property
    def loops(self) -> list[Loop]:
        return self.parallel + [lp for r in self.reductions for lp in r.loops]

    @property
    def out_vars(self) -> list[str]:
        return [o.var for o in self.outputs]

    def tile_statements(self) -> list[ast.stmt]:
        return [s for seg in self.segments for s in seg] + [s for r in self.reductions for s in r.body]


def function_ast(fn: Callable) -> ast.FunctionDef:
    """`fn`'s source as an AST, with line numbers relative to its first line (its first decorator)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise AutoKernelError(f"can't read the source of {fn!r}: {e}") from e
    fdef = ast.parse(src).body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise AutoKernelError(f"{fn.__name__}: expected a plain `def` function")
    return fdef


def parse(fn: Callable) -> LoopNest:
    """Parse `fn`'s source into a LoopNest, or raise AutoKernelError naming the offending line."""
    return parse_function(function_ast(fn), fn.__name__, fn.__code__.co_firstlineno)


def parse_function(fdef: ast.FunctionDef, fn_name: str, first_line: int) -> LoopNest:
    """Parse an already-parsed function (lines relative to `first_line` of its file) into a LoopNest.

    `fn_name` names the kernel in errors and generated code, so it must be an identifier. `a @= b` is
    rewritten to `a = a @ b` in `fdef` (the same thing for arrays), so `@` means one thing everywhere.
    """
    _MatmulAssign().visit(fdef)
    return _Parser(fn_name, first_line, fdef).parse()


class _MatmulAssign(ast.NodeTransformer):
    """`a @= b` -> `a = a @ b`: `mxu.rewrite_matmuls` then gives it the MXU's accumulation, like any `@`."""

    def visit_AugAssign(self, s: ast.AugAssign) -> ast.stmt:
        if not (isinstance(s.op, ast.MatMult) and isinstance(s.target, ast.Name)):
            return s
        value = ast.copy_location(ast.BinOp(ast.Name(s.target.id, ast.Load()), ast.MatMult(), s.value), s)
        return ast.fix_missing_locations(ast.copy_location(ast.Assign([s.target], value), s))


def _inline(value: ast.expr, allocs: dict[str, ast.expr]) -> ast.expr:
    """An output allocation reading outputs allocated before it (`allocs`), with their allocations in their place:
    the value they have there, in Python."""
    value = copy.deepcopy(value)
    earlier = {id(n) for n in free_loads(value) if n.id in allocs}

    class Inline(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            return copy.deepcopy(allocs[n.id]) if id(n) in earlier else n

    return Inline().visit(value) if earlier else value


class _Parser:
    def __init__(self, fn_name: str, first_line: int, fdef: ast.FunctionDef):
        self.fn_name, self.first_line, self.fdef = fn_name, first_line, fdef
        self.params = [p.arg for p in fdef.args.args]
        self.out_vars: list[str] = []
        self.hoisted_depth: dict[int, int] = {}  # id of a statement between parallel loops -> its loop's depth
        self.rebinds: dict[str, ast.stmt] = {}  # outputs named like a parameter -> their allocation

    def err(self, node: ast.AST | None, msg: str) -> AutoKernelError:
        return _error(self.fn_name, self.first_line, node, msg)

    # ------------------------------------------------------------------ top level

    def parse(self) -> LoopNest:
        fdef = self.fdef
        a = fdef.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
            raise self.err(fdef, "only plain positional array parameters are supported")

        body = list(fdef.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        ret = self._returns(body)
        body = body[:-1]

        fors = [s for s in body if isinstance(s, ast.For)]
        if len(fors) != 1 or body[-1] is not fors[0]:
            raise self.err(
                fors[1] if len(fors) > 1 else fdef, "expected one top-level tile loop nest before the return"
            )
        top = fors[0]
        prologue, outputs = self._prologue(body[:-1], top)

        parallel, hoisted, inner = self._parallel_loops(top)
        guard, guard_node, inner = self._split_guard(inner, "tile mask")
        inner, stores = self._split_stores(inner, parallel[-1].node)
        segments, reductions = self._tile_body(hoisted + inner)

        nest = LoopNest(
            fn_name=self.fn_name,
            first_line=self.first_line,
            fdef=fdef,
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

    def _returns(self, body: list[ast.stmt]) -> ast.expr:
        """The final `return out` / `return out1, out2`; records the output names."""
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise self.err(
                body[-1] if body else self.fdef, "function must end with `return <output>` or a tuple of outputs"
            )
        ret = body[-1].value
        names = ret.elts if isinstance(ret, ast.Tuple) else [ret]
        if not names or not all(isinstance(n, ast.Name) for n in names):
            raise self.err(body[-1], "return the output arrays by name: `return out` or `return out1, out2`")
        self.out_vars = [n.id for n in names]
        if len(set(self.out_vars)) != len(self.out_vars):
            raise self.err(body[-1], "each output may be returned only once")
        return ret

    def _prologue(self, stmts: list[ast.stmt], top: ast.For) -> tuple[list[ast.stmt], list[Output]]:
        """Split the statements before the loop nest into shape-level constants and output allocations.

        Each runs, as in Python, with the names assigned before it; but the kernel evaluates the allocations
        after the whole prologue, each on its own, so a name an allocation reads must not change after it, an
        output an allocation reads stands for its own allocation, and the rest of the prologue can't read one.
        """
        prologue, allocs = [], {}
        local = scopes.local_names(self.fdef)
        assigned = set(self.params)
        read_by: dict[str, ast.stmt] = {}  # a name an allocation reads -> that allocation
        for s in stmts:
            for name in sorted(reads(s) & local - assigned):
                raise self.err(
                    s,
                    f"`{name}` is read before it is assigned (the function assigns it later, so Python would raise "
                    "UnboundLocalError here); assign it before this line or use another name",
                )
            if self._is_assign_to(s, self.out_vars):
                var, value = s.targets[0].id, s.value
                if var in self.params and not (isinstance(value, ast.Name) and value.id == var):
                    self.rebinds[var] = s  # fine unless the loop nest reads `var` (`_check_rebinds`)
                allocs[var] = _inline(value, allocs)
                read_by.update((name, s) for name in reads(s) if name not in read_by)
                assigned.add(var)
                continue
            if not isinstance(s, (ast.Assign, ast.AnnAssign, ast.Assert, ast.Pass)):
                raise self.err(s, "only simple constant assignments / asserts may precede the loop nest")
            for name in sorted(reads(s) & set(allocs)):
                raise self.err(
                    s,
                    f"the prologue reads output `{name}` after its allocation; the kernel runs the prologue before "
                    "the allocations, so compute this from the inputs and constants instead",
                )
            stored = {t.id for t in ast.walk(s) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
            for name in sorted(stored & set(self.params)):
                raise self.err(s, f"the prologue may not rebind input `{name}`")
            for name in sorted(stored & set(read_by)):
                alloc = read_by[name]
                raise self.err(
                    s,
                    f"`{name}` is reassigned after the output allocation `{ast.unparse(alloc)}` (line "
                    f"{self.first_line + alloc.lineno - 1}) reads it; the kernel evaluates the allocations after the "
                    "prologue, so use another name here",
                )
            assigned |= stored
            prologue.append(s)
        for var in self.out_vars:
            if var not in allocs:
                raise self.err(top, f"missing output allocation `{var} = jnp.zeros(...)` before the loop nest")
        keeps = {v for v, a in allocs.items() if isinstance(a, ast.Name) and a.id in self.params}
        return prologue, [Output(var, allocs[var], var in keeps) for var in self.out_vars]

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _is_assign_to(s: ast.stmt, names: list[str]) -> bool:
        return isinstance(s, ast.Assign) and is_simple_assignment(s) and s.targets[0].id in names

    def _loop(self, node: ast.For) -> Loop:
        """`for v in range(stop)`, `range(start, stop)` or `range(start, stop, step)`."""
        if node.orelse or not isinstance(node.target, ast.Name):
            raise self.err(node, "tile loops must be `for v in range(start, stop, step)`")
        it = node.iter
        ok = (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
            and 1 <= len(it.args) <= 3
        )
        if not ok:
            raise self.err(node, "tile loops must be `for v in range(start, stop, step)`")
        args = list(it.args)
        if len(args) == 1:
            args.insert(0, ast.Constant(0))
        if len(args) == 2:
            args.append(ast.Constant(1))
        for arg in args:
            ast.copy_location(arg, it)
        return Loop(node.target.id, *args, node=node)

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
            if isinstance(body[0], ast.If):
                raise self.err(body[0], "a tile mask goes on the innermost parallel loop, around the output stores")
            self.hoisted_depth.update((id(s), len(loops) - 1) for s in body[:-1])
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
        stores = [Store(s.targets[0].id, s.value.func.value.slice, s.value.args[0], s) for s in body[n:]]
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
        for var in loop_vars:
            if loop_vars.count(var) > 1:
                node = next(lp.node for lp in reversed(nest.loops) if lp.var == var)
                raise self.err(node, f"loop variable `{var}` is used by two loops; loop variables must be distinct")
        for s in nest.tile_statements():
            if not is_simple_assignment(s):
                raise self.err(s, "loop bodies may only contain `name = expr` / `name op= expr` statements")
            target = assigned_name(s)
            if target in nest.out_vars:
                raise self.err(s, f"`{target}` may only be written by its final `.at[...].set(...)`")
            if target in self.params or target in loop_vars:
                raise self.err(s, f"can't assign to input / loop variable `{target}` inside the loop nest")
        temporaries = {assigned_name(s) for s in nest.tile_statements()}
        code = [*nest.tile_statements(), *(st.node for st in nest.stores)]
        guards = [c for c in [nest.guard, *(r.guard for r in nest.reductions)] if c is not None]
        for node in code + guards:
            for n in ast.walk(node):
                if isinstance(n, ast.NamedExpr):
                    raise self.err(n, "`:=` isn't supported inside the loop nest; assign in a statement of its own")
        # The kernel reads an input tile wherever the code subscripts the input's name, whatever scope binds it
        # (analysis checks the same for state variables, which the kernel renames; masks run as Python).
        for node in code:
            for name in sorted(scope_bound(node) & set(self.params)):
                raise self.err(node, f"a comprehension or lambda binds `{name}`, an input of the loop nest; rename it")
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
        self._check_rebinds(nest)
        self._check_hoisted(nest)
        self._check_order(nest, temporaries | set(loop_vars))

    def _check_rebinds(self, nest: LoopNest) -> None:
        """An output named like a parameter (`x = jnp.zeros(...)`) is the output wherever the loop nest names
        it, in Python; but the kernel would read the input there, so the nest must not read it at all."""
        if not self.rebinds:
            return
        ranges = [e for lp in nest.loops for e in (lp.start, lp.stop, lp.step)]
        masks = [c for c in [nest.guard, *(r.guard for r in nest.reductions)] if c is not None]
        stores = [e for st in nest.stores for e in (st.index, st.value)]
        for node in [*ranges, *masks, *nest.tile_statements(), *stores]:
            for name in sorted(reads(node) & set(self.rebinds)):
                line = self.first_line + self.rebinds[name].lineno - 1
                raise self.err(
                    node,
                    f"output `{name}` rebinds parameter `{name}` (line {line}), so `{name}` here is the output in "
                    "Python, but the kernel would read the input; rename the output",
                )

    def _check_hoisted(self, nest: LoopNest) -> None:
        """Values assigned between parallel loops must not be reassigned further in.

        The kernel runs those statements once per tile, which matches Python only while nothing deeper
        changes their values: Python would carry such a change on to the inner loops' next iteration.
        """
        first: dict[str, tuple[int, ast.stmt]] = {}  # name -> (depth, statement) where a hoisted one assigns it
        for s in nest.tile_statements():
            name, depth = assigned_name(s), self.hoisted_depth.get(id(s), len(nest.parallel))
            if name in first and first[name][0] < depth:
                line = self.first_line + first[name][1].lineno - 1
                raise self.err(
                    s,
                    f"`{name}` is assigned between the parallel loops (line {line}) and again further in; Python "
                    "would carry the new value on to the next iteration of the inner loops, but the kernel "
                    "recomputes it for every tile: use another name here",
                )
            if id(s) in self.hoisted_depth:
                first.setdefault(name, (depth, s))

    def _check_order(self, nest: LoopNest, tile_names: set[str]) -> None:
        """Every loop variable and temporary the tile reads must be bound earlier in the tile, as in Python.

        The kernel can't carry a value in from before the loop nest or from one tile to the next. A
        reduction loop's variable keeps its last value after the loop, as in Python. A value only masked
        reduction loops assign is recorded in `nest.masked_reads`: whether a tile may skip all their steps
        is known once the masks are evaluated.
        """
        defined: set[str] = set()
        masked: dict[str, list[int]] = {}  # a name only masked reduction loops assign -> their indices

        def check(node: ast.AST, names: set[str]) -> None:
            for name in sorted(names & tile_names - defined):
                if name in {lp.var for lp in nest.loops}:
                    raise self.err(node, f"loop variable `{name}` is read before its loop starts")
                if name in masked:
                    nest.masked_reads.append((name, node, tuple(masked[name])))
                    continue
                raise self.err(
                    node,
                    f"`{name}` is read before it is assigned in the tile; assign it inside the loop nest first "
                    "(a value can't be carried in from before the loop nest, or from one tile to the next)",
                )

        def run(statements: list[ast.stmt]) -> None:
            for s in statements:
                depth = self.hoisted_depth.get(id(s), len(nest.parallel) - 1)
                defined.update(lp.var for lp in nest.parallel[: depth + 1])
                check(s, reads(s))
                defined.add(assigned_name(s))

        run(nest.segments[0])
        defined.update(lp.var for lp in nest.parallel)
        for p, (r, seg) in enumerate(zip(nest.reductions, nest.segments[1:], strict=True)):
            defined.update(lp.var for lp in r.loops)
            if r.guard is not None:
                check(r.guard_node, reads(r.guard))
            before = set(defined)
            run(r.body)
            if r.guard is not None:  # the mask may skip the whole loop: only what was assigned before is sure
                for name in defined - before:
                    masked.setdefault(name, []).append(p)
                defined.intersection_update(before)
            run(seg)
        for st in nest.stores:
            check(st.node, reads(st.value))


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def free_loads(node: ast.AST, bound: frozenset[str] = frozenset()) -> Iterator[ast.Name]:
    """The names `node` reads from its scope: not those its comprehensions and lambdas bind themselves."""
    if isinstance(node, _COMPREHENSIONS):
        gens = node.generators
        inner = bound | {n.id for g in gens for n in ast.walk(g.target) if isinstance(n, ast.Name)}
        yield from free_loads(gens[0].iter, bound)  # evaluated outside the comprehension
        for g in gens:
            for part in ([g.iter] if g is not gens[0] else []) + g.ifs:
                yield from free_loads(part, inner)
        for part in (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,):
            yield from free_loads(part, inner)
        return
    if isinstance(node, ast.Lambda):
        a = node.args
        for part in [*a.defaults, *(d for d in a.kw_defaults if d is not None)]:
            yield from free_loads(part, bound)
        yield from free_loads(node.body, bound | {p.arg for p in ast.walk(a) if isinstance(p, ast.arg)})
        return
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in bound:
            yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from free_loads(child, bound)


def scope_bound(node: ast.AST) -> set[str]:
    """The names the comprehensions and lambdas in `node` bind for themselves."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.comprehension):
            names |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
        elif isinstance(n, ast.Lambda):
            names |= {a.arg for a in ast.walk(n.args) if isinstance(a, ast.arg)}
    return names


def reads(node: ast.AST) -> set[str]:
    """The names a statement (or expression) reads, not counting those its comprehensions and lambdas bind
    where they bind them (`t + sum(t for t in r)` reads `t`).

    An augmented assignment `x += v` reads `x`.
    """
    expr = node.value if isinstance(node, (ast.Assign, ast.AugAssign)) else node
    loaded = {n.id for n in free_loads(expr)}
    if isinstance(node, ast.AugAssign):
        loaded.add(node.target.id)
    return loaded
