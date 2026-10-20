"""Kernel code: the statements a tile runs, the names they read and assign, and the output stores among them.

Tile statements are assignments, augmented assignments, output stores and `with jax.named_scope(...)`
blocks of those (parse.py checks), so the analyses here are straight-line: a `with` block's statements run
in order, as if it weren't there.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

__all__ = ["Store", "as_store", "assigned", "first_reads", "flatten", "reads"]


@dataclass(frozen=True)
class Store:
    """`out = out.at[index].set(value)`: writing a tile of an output."""

    output: str
    index: tuple[ast.expr, ...]
    value: ast.expr
    node: ast.Assign


def as_store(stmt: ast.stmt) -> Store | None:
    """The output store `stmt` is, or None if it isn't one."""
    match stmt:
        case ast.Assign(
            targets=[ast.Name(id=target)],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(value=ast.Attribute(value=ast.Name(id=output), attr="at"), slice=index),
                    attr="set",
                ),
                args=[value],
                keywords=[],
            ),
        ) if target == output:
            dims = tuple(index.elts) if isinstance(index, ast.Tuple) else (index,)
            return Store(output, dims, value, stmt)
    return None


def flatten(stmts: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """`stmts` in the order they run, each `with` block followed by its statements."""
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, ast.With):
            yield from flatten(stmt.body)


class _Reads(ast.NodeVisitor):
    """The free names an expression reads: names bound by its lambdas and comprehensions are their own."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)  # tile code assigns only in statements, whose targets aren't visited

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            self.visit(default)  # type: ignore[arg-type]
        params = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        self.names |= reads(node.body) - params

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp) -> None:
        bound: set[str] = set()
        inner: set[str] = set()
        for gen in node.generators:
            bound |= {n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
            inner |= reads(gen.iter)
            for cond in gen.ifs:
                inner |= reads(cond)
        parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
        for part in parts:
            inner |= reads(part)
        self.names |= inner - bound

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comprehension


def reads(node: ast.AST) -> set[str]:
    """The names an expression reads (for a statement: the names its own expressions read, not a `with`
    block's body; an augmented assignment reads its target)."""
    visitor = _Reads()
    if isinstance(node, ast.With):
        for item in node.items:
            visitor.visit(item.context_expr)
    elif isinstance(node, ast.AugAssign):
        visitor.visit(node.value)
        visitor.names.add(node.target.id)  # type: ignore[union-attr]  # parse.py: only names are augmented
    elif isinstance(node, ast.Assign):
        visitor.visit(node.value)
    else:
        visitor.visit(node)
    return visitor.names


def _assigns(stmt: ast.stmt) -> set[str]:
    """The names one statement assigns itself (a `with` block: none; its body's statements do)."""
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(stmt, ast.AugAssign):
        return {stmt.target.id}  # type: ignore[union-attr]
    return set()


def assigned(stmts: Iterable[ast.stmt]) -> set[str]:
    """The names `stmts` assign."""
    return {name for stmt in flatten(stmts) for name in _assigns(stmt)}


def first_reads(stmts: Iterable[ast.stmt]) -> set[str]:
    """The names `stmts` read before assigning them: the values they need from before they run."""
    needed: set[str] = set()
    done: set[str] = set()
    for stmt in flatten(stmts):
        needed |= reads(stmt) - done
        done |= _assigns(stmt)
    return needed
