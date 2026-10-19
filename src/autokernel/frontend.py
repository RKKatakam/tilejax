"""Frontend: parse a Python tile loop nest into a small IR.

The accepted shape of a function is:

    def f(x, y, ...):                      # every parameter is an input array
        <prologue>                         # shape-level constants (M, K = x.shape, ...)
        out = jnp.zeros((M, N))            # output allocation
        for i in range(0, M, TM):          # one or more *parallel* tile loops
            for j in range(0, N, TN):
                acc = jnp.zeros((TM, TN))  # optional loop-carried accumulators
                for k in range(0, K, TK):  # optional single *reduction* loop
                    acc += x[i:i+TM, k:k+TK] @ y[k:k+TK, j:j+TN]
                out = out.at[i:i+TM, j:j+TN].set(acc)
        return out

Input arrays may only be read through tile slices: each dimension is
`v:v+T` (v a loop variable stepping by T), `c:c+T` (a constant block) or `:`.

The body of the innermost parallel loop may be guarded by a tile mask that depends only on the
parallel loop variables, either `if cond: <body>` or a leading `if not_cond: continue`. Masked
tiles are skipped entirely and keep the output allocation's value (e.g. zeros for causal masks).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import AutoKernelError

__all__ = ["Loop", "LoopNest", "parse"]


def _err(node: ast.AST | None, msg: str, fn_name: str, first_line: int) -> AutoKernelError:
    if node is not None and hasattr(node, "lineno"):
        return AutoKernelError(f"{fn_name} (line {first_line + node.lineno - 1}): {msg}")
    return AutoKernelError(f"{fn_name}: {msg}")


@dataclass
class Loop:
    """`for var in range(start, stop, step)`, with the range arguments still as source expressions."""

    var: str
    start: ast.expr
    stop: ast.expr
    step: ast.expr
    node: ast.For


@dataclass
class LoopNest:
    """A parsed tile loop nest. Expressions stay as AST: they are evaluated once input shapes are known."""

    fn_name: str
    first_line: int
    params: list[str]
    prologue: list[ast.stmt]
    out_var: str
    out_alloc: ast.expr
    parallel: list[Loop]
    pre: list[ast.stmt]  # statements before the reduction loop (accumulator inits, temporaries)
    reduction: Loop | None
    body: list[ast.stmt]  # reduction loop body
    post: list[ast.stmt]  # statements between the reduction loop and the store
    store_index: ast.expr  # the `[...]` in out.at[...]
    store_value: ast.expr
    store_node: ast.stmt
    carried: list[str] = field(default_factory=list)  # accumulators carried across the reduction
    guard: ast.expr | None = None  # tile mask: the tile is computed only where this is true
    guard_node: ast.stmt | None = None
    red_guard: ast.expr | None = None  # reduction mask: the reduction step runs only where this is true
    red_guard_node: ast.stmt | None = None

    def error(self, node: ast.AST | None, msg: str) -> AutoKernelError:
        return _err(node, msg, self.fn_name, self.first_line)

    @property
    def loops(self) -> list[Loop]:
        return self.parallel + ([self.reduction] if self.reduction else [])


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for s in stmts:
        if isinstance(s, ast.Assign):
            names.add(s.targets[0].id)
        elif isinstance(s, ast.AugAssign):
            names.add(s.target.id)
    return names


def _split_guard(body: list[ast.stmt], what: str, err) -> tuple[ast.expr | None, ast.stmt | None, list[ast.stmt]]:
    """Peel a mask off a loop body: `if cond: <body>` or a leading `if cond: continue`.

    Returns (condition under which the body runs, the `if` node, the remaining body).
    """
    if not body or not isinstance(body[0], ast.If) or body[0].orelse:
        return None, None, body
    node = body[0]
    if len(node.body) == 1 and isinstance(node.body[0], ast.Continue):
        return ast.copy_location(ast.UnaryOp(ast.Not(), node.test), node.test), node, body[1:]
    if len(body) == 1:
        return node.test, node, node.body
    raise err(node, f"a {what} must be `if cond:` around the whole loop body, or a leading `if cond: continue`")


def parse(fn: Callable) -> LoopNest:
    """Parse `fn`'s source into a LoopNest, or raise AutoKernelError naming the offending line."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise AutoKernelError(f"can't read the source of {fn!r}: {e}") from e
    first_line = fn.__code__.co_firstlineno
    name = fn.__name__
    fdef = ast.parse(src).body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise AutoKernelError(f"{name}: expected a plain `def` function")

    def err(node, msg):
        return _err(node, msg, name, first_line)

    a = fdef.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
        raise err(fdef, "only plain positional array parameters are supported")
    params = [p.arg for p in a.args]

    body = list(fdef.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring

    if not body or not isinstance(body[-1], ast.Return) or not isinstance(body[-1].value, ast.Name):
        raise err(body[-1] if body else fdef, "function must end with `return <output array>`")
    out_var = body[-1].value.id
    body = body[:-1]

    fors = [s for s in body if isinstance(s, ast.For)]
    if len(fors) != 1 or body[-1] is not fors[0]:
        raise err(
            fors[1] if len(fors) > 1 else fdef, "expected exactly one top-level tile loop nest right before the return"
        )
    top = fors[0]

    prologue, out_alloc = [], None
    for s in body[:-1]:
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id == out_var
        ):
            out_alloc = s.value
            continue
        if not isinstance(s, (ast.Assign, ast.AnnAssign, ast.Assert, ast.Pass)):
            raise err(s, "only simple constant assignments / asserts may precede the loop nest")
        for t in ast.walk(s):
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id in params:
                raise err(s, f"the prologue may not rebind input `{t.id}`")
        prologue.append(s)
    if out_alloc is None:
        raise err(top, f"missing output allocation `{out_var} = jnp.zeros(...)` before the loop nest")

    def as_loop(node: ast.For) -> Loop:
        if node.orelse or not isinstance(node.target, ast.Name):
            raise err(node, "tile loops must be `for v in range(start, stop, step)`")
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
            and len(it.args) == 3
        ):
            raise err(node, "tile loops must use the 3-argument form `range(start, stop, step)`")
        return Loop(node.target.id, *it.args, node=node)

    parallel = [as_loop(top)]
    inner = top.body
    while len(inner) == 1 and isinstance(inner[0], ast.For):
        parallel.append(as_loop(inner[0]))
        inner = inner[0].body

    def check_mask(cond, node, what, body):
        for n in ast.walk(cond):
            if isinstance(n, ast.Name) and n.id in [*params, out_var]:
                raise err(node, f"the {what} can only depend on loop variables and constants, not `{n.id}`")
        if not body:
            raise err(node, f"the body under the {what} is empty")

    guard, guard_node, inner = _split_guard(inner, "tile mask", err)
    if guard is not None:
        check_mask(guard, guard_node, "tile mask", inner)

    last = inner[-1]
    store_ok = (
        isinstance(last, ast.Assign)
        and len(last.targets) == 1
        and isinstance(last.targets[0], ast.Name)
        and last.targets[0].id == out_var
        and isinstance(last.value, ast.Call)
        and len(last.value.args) == 1
        and not last.value.keywords
        and isinstance(last.value.func, ast.Attribute)
        and last.value.func.attr == "set"
        and isinstance(last.value.func.value, ast.Subscript)
        and isinstance(last.value.func.value.value, ast.Attribute)
        and last.value.func.value.value.attr == "at"
        and isinstance(last.value.func.value.value.value, ast.Name)
        and last.value.func.value.value.value.id == out_var
    )
    if not store_ok:
        raise err(last, f"the innermost parallel loop must end with `{out_var} = {out_var}.at[...].set(...)`")
    store_index = last.value.func.value.slice
    store_value = last.value.args[0]
    inner = inner[:-1]

    red_idx = [n for n, s in enumerate(inner) if isinstance(s, ast.For)]
    if len(red_idx) > 1:
        raise err(inner[red_idx[1]], "only one reduction loop is supported")
    red_guard, red_guard_node = None, None
    if red_idx:
        n = red_idx[0]
        pre, red_node, post = inner[:n], inner[n], inner[n + 1 :]
        reduction = as_loop(red_node)
        red_guard, red_guard_node, red_body = _split_guard(red_node.body, "reduction mask", err)
        if red_guard is not None:
            check_mask(red_guard, red_guard_node, "reduction mask", red_body)
    else:
        pre, reduction, red_body, post = inner, None, [], []

    loop_vars = {lp.var for lp in parallel} | ({reduction.var} if reduction else set())
    if len(loop_vars) != len(parallel) + (1 if reduction else 0):
        raise err(top, "loop variables must have distinct names")

    for s in pre + red_body + post:
        ok = (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)) or (
            isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name)
        )
        if not ok:
            raise err(s, "loop bodies may only contain `name = expr` / `name op= expr` statements")
        target = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id
        if target == out_var:
            raise err(s, f"`{out_var}` may only be written by the final `.at[...].set(...)`")
        if target in params or target in loop_vars:
            raise err(s, f"can't assign to input / loop variable `{target}` inside the loop nest")

    carried = sorted(_assigned_names(pre) & _assigned_names(red_body))
    for s in red_body:
        if isinstance(s, ast.AugAssign) and s.target.id not in carried:
            raise err(s, f"`{s.target.id}` is updated in the reduction loop but never initialised before it")

    return LoopNest(
        fn_name=name,
        first_line=first_line,
        params=params,
        prologue=prologue,
        out_var=out_var,
        out_alloc=out_alloc,
        parallel=parallel,
        pre=pre,
        reduction=reduction,
        body=red_body,
        post=post,
        store_index=store_index,
        store_value=store_value,
        store_node=last,
        carried=carried,
        guard=guard,
        guard_node=guard_node,
        red_guard=red_guard,
        red_guard_node=red_guard_node,
    )
