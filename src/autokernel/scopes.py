"""Python's scoping rules over the AST: which names a piece of code binds in its own scope, and which it
reads from scopes around it. Every pass that asks "what does this code assign?" (the frontend, kernels'
constants, programs' nests and rolled loops) asks it here, so they agree.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

__all__ = [
    "COMPREHENSIONS",
    "FUNCTIONS",
    "SCOPES",
    "binds",
    "global_loads",
    "in_scope",
    "local_names",
    "params",
    "parts",
]

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
SCOPES = (*FUNCTIONS, ast.ClassDef, *COMPREHENSIONS)


def parts(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """A nested scope's parts evaluated in the scope around it (defaults, decorators, a comprehension's
    first iterable), and those evaluated in its own."""
    if isinstance(scope, FUNCTIONS):
        a = scope.args
        outer: list[ast.AST] = [*a.defaults, *(d for d in a.kw_defaults if d is not None)]
        if isinstance(scope, ast.Lambda):
            return outer, [scope.body]
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
        outer += [*scope.decorator_list, *(p.annotation for p in params if p is not None and p.annotation)]
        return outer + ([scope.returns] if scope.returns else []), list(scope.body)
    if isinstance(scope, ast.ClassDef):
        return [*scope.decorator_list, *scope.bases, *scope.keywords], list(scope.body)
    first, *rest = scope.generators
    elts = [scope.key, scope.value] if isinstance(scope, ast.DictComp) else [scope.elt]
    return [first.iter], [*elts, first.target, *first.ifs, *rest]


def in_scope(nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
    """Every node of `nodes` that belongs to their scope: nested scopes are yielded, but only their parts
    evaluated in this scope are entered."""
    todo = list(nodes)[::-1]
    while todo:
        node = todo.pop()
        yield node
        todo.extend(reversed(parts(node)[0] if isinstance(node, SCOPES) else list(ast.iter_child_nodes(node))))


def params(scope: ast.AST) -> set[str]:
    if not isinstance(scope, FUNCTIONS):
        return set()
    a = scope.args
    return {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if p is not None}


def binds(nodes: list[ast.AST], comprehension: bool = False, declared: bool = False) -> set[str]:
    """The names code in one scope binds: assigned, imported, defined or caught (walrus targets in its
    comprehensions bind in the function around them), less those it declares `global` or `nonlocal` unless
    `declared` (then those too: the names its statements assign, wherever the value goes)."""
    bound, names_declared = set(), set()
    for node in in_scope(nodes):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names_declared.update(node.names)
        elif isinstance(node, COMPREHENSIONS) and not comprehension:
            bound |= {n.target.id for n in ast.walk(node) if isinstance(n, ast.NamedExpr)}
    if comprehension:  # its own walrus targets aren't its own
        bound -= {n.target.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.NamedExpr)}
    return bound if declared else bound - names_declared


def global_loads(nodes: list[ast.AST], bound: set[str]) -> Iterable[ast.Name]:
    """The `Name`s in `nodes` (code in a scope that binds `bound`) that read a module global or closure
    variable: not bound there, in a scope around them or in their own (a lambda's parameters, a
    comprehension's targets, a nested function's locals)."""
    for node in in_scope(nodes):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            yield node
        elif isinstance(node, SCOPES):
            inner = parts(node)[1]
            own = params(node) | binds(inner, isinstance(node, COMPREHENSIONS))
            yield from global_loads(inner, bound | own)


def local_names(fdef: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names local to a function: its parameters and everything its body binds (not nested scopes, and not
    names it declares `global` or `nonlocal`)."""
    return params(fdef) | binds(list(fdef.body))
