"""Programs: a whole model written as tile loop nests mixed with ordinary JAX and Python code.

    @autokernel.tpu
    def mlp(x, w1, w2):
        h = jnp.zeros((M, F))
        for i in range(0, M, TM):           # a tile loop nest: one gridless Pallas kernel
            for j in range(0, F, TF):
                ...
                h = h.at[i : i + TM, j : j + TF].set(jnp.maximum(acc, 0))
        h = h.reshape(...)                  # ordinary JAX, compiled by XLA around the kernels
        out = jnp.zeros((M, D))
        for i in range(0, M, TM):           # another kernel
            ...
        return out + x

A *tile loop nest* is a `for` statement whose `range` loops end in stores `v = v.at[i : i + T, ...].set(...)`
of tiles placed by the loop variables, and which doesn't read the arrays it stores to (`_nest_stores` has the
exact rule); any other loop, such as `h = h.at[t].set(f(h[t - 1]))`, is ordinary Python. Nests may sit at
any depth of Python control flow: in the function body, or inside a plain Python `for`, `if` or `with`,
which runs at trace time like any other Python (a `for layer in range(L)` around nests runs as a
`jax.lax.fori_loop` where it can, else unrolled).

A nest's output `v` is usually allocated just before it (`v = jnp.zeros(...)`): that allocation moves into
the kernel. Otherwise the nest starts from `v`'s current value, whatever computed it (an allocation before an
enclosing `if` or loop, an argument, an earlier nest): it is an input of the kernel, aliased to the output,
so elements no tile writes keep their values, as in Python. The nest's loop variables and temporaries
don't exist after it, so reading one after the nest is an error at decoration time.

The function is rewritten so that each nest and its allocations become one call,
`o1, o2 = __autokernel_nests[k]({"x": x, "l": l, ...})`, passing the function's local values the nest
reads; global and closure arrays it reads through subscripts are passed in the same way (their current
values, on every call); other globals and closure variables are read directly, as in a single kernel. At
call time the nest splits those values into arrays (the kernel's inputs) and everything else (numbers,
dtypes, tuples: constants compiled into the kernel), and builds one `TpuKernel` per distinct set of them.
The rewritten function is jit-compiled, so the code between the kernels is compiled by XLA.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import inspect
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import analysis, autodiff, frontend, fusion, hardware
from . import tune as tune_lib
from .api import TpuKernel, check_outputs
from .errors import AutoKernelError
from .lowering import Lowered, Options
from .views import View, ViewError, make_view

__all__ = ["ProgramLowered", "ProgramNest", "TpuProgram", "is_single_nest"]

_NESTS = "__autokernel_nests"  # the rewritten function's name for the list of nests
_GROUPS = "__autokernel_groups"  # ... for the runs of consecutive nests (fused into one kernel)
_LOOP = "__autokernel_loop"  # ... for the runner of Python loops around nests (as `jax.lax.fori_loop`)
_BODY = "__autokernel_body"  # prefix of the functions holding such loops' bodies
_CARRY = "__autokernel_carry"
_SHAPE_ATTRS = frozenset({"shape", "dtype", "ndim", "size"})  # what a single kernel's prologue may read


# --------------------------------------------------------------------------- single kernel or program


def is_single_nest(fdef: ast.FunctionDef) -> bool:
    """Is `fdef` in the single-kernel form: plain array parameters, a prologue of constants and output
    allocations, one top-level tile loop nest, then `return` of the outputs by name?

    Such a function becomes a `TpuKernel` (and keeps its error messages when the nest is malformed);
    anything else is a program. A prologue that uses an input's values rather than its shape and dtype
    (`h = x @ w`) is ordinary JAX code, so it makes the function a program too.
    """
    a = fdef.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
        return False
    body = _strip_docstring(fdef.body)
    loops = [n for n, s in enumerate(body) if isinstance(s, ast.For)]
    if len(loops) != 1:
        return False
    top = body[loops[0]]
    stores = _nest_stores(top)
    if not stores and _holds_nests(fdef, top):
        return False  # a Python loop around nests
    before, after = body[: loops[0]], body[loops[0] + 1 :]
    if not all(isinstance(s, (ast.Assign, ast.AnnAssign, ast.Assert, ast.Pass)) for s in before):
        return False
    if {_store_target(s) for s in stores} - {_target(s) for s in before}:
        return False  # it updates an array it doesn't allocate (an argument): a program's nest can
    if len(after) > 1:
        return False
    outputs: set[str] = set()
    if after:
        ret = after[0]
        if not isinstance(ret, ast.Return) or ret.value is None:
            return False
        names = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
        if not all(isinstance(n, ast.Name) for n in names):
            return False
        outputs = {n.id for n in names}
    params = {p.arg for p in a.args}
    # Allocations may use the inputs' values (`out = x`); other prologue statements only their shapes.
    return not any(_target(s) not in outputs and _reads_values(s, params) for s in before)


def _holds_nests(fdef: ast.FunctionDef, loop: ast.For) -> bool:
    """Does the body of `loop` (in `fdef`) hold tile loop nests of a program?"""
    return bool(_Finder(fdef, fdef.name, 1, strict=False, body=loop.body).regions)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[1:]
    return body


def _target(s: ast.stmt) -> str | None:
    """The name a `name = expr` statement assigns, else None."""
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
        return s.targets[0].id
    return None


def _reads_values(s: ast.stmt, params: set[str]) -> bool:
    """Does `s` use one of `params` other than through `.shape`, `.dtype`, `.ndim` or `.size`?"""
    shape_reads = {id(n.value) for n in ast.walk(s) if isinstance(n, ast.Attribute) and n.attr in _SHAPE_ATTRS}
    return any(isinstance(n, ast.Name) and n.id in params and id(n) not in shape_reads for n in ast.walk(s))


# --------------------------------------------------------------------------- finding the nests


@dataclass
class _Region:
    """A tile loop nest of a program, with its output allocations."""

    name: str  # the kernel's name: an identifier
    block: list[ast.stmt]  # the statement list holding it
    # One per output, in output order: the statement allocating it just before the nest, or for an output
    # with no such statement (`existing`), `v = v`: the nest starts from the array's current value.
    allocs: list[ast.Assign]
    loop: ast.For
    outputs: list[str]
    inputs: list[str]  # the function's local names it reads, in source order: passed in at call time
    names: frozenset[str]  # every name it reads
    existing: frozenset[str] = frozenset()  # outputs whose current value the nest is given (no allocation)
    globals: tuple[str, ...] = ()  # global and closure names it reads through subscripts: arrays become inputs


def _store_target(s: ast.stmt) -> str | None:
    """`v` for an output store `v = v.at[...].set(value)`, else None."""
    v = getattr(s, "value", None)
    name = _target(s)
    ok = (
        name is not None
        and isinstance(v, ast.Call)
        and len(v.args) == 1
        and not v.keywords
        and isinstance(v.func, ast.Attribute)
        and v.func.attr == "set"
        and isinstance(v.func.value, ast.Subscript)
        and isinstance(v.func.value.value, ast.Attribute)
        and v.func.value.value.attr == "at"
        and isinstance(v.func.value.value.value, ast.Name)
        and v.func.value.value.value.id == name
    )
    return name if ok else None


def _tile_stores(body: list[ast.stmt]) -> list[ast.Assign]:
    """The output stores ending a tile body (possibly under a tile mask `if`)."""
    stmts = list(body)
    if stmts and isinstance(stmts[0], ast.If) and not stmts[0].orelse:
        guard = stmts[0]
        if not (len(guard.body) == 1 and isinstance(guard.body[0], ast.Continue)):
            stmts = list(guard.body) + stmts[1:]
    n = len(stmts)
    while n and _store_target(stmts[n - 1]):
        n -= 1
    return stmts[n:]


def _is_range_loop(node: ast.For) -> bool:
    """`for v in range(...)`, with no `else`."""
    it = node.iter
    return (
        not node.orelse
        and isinstance(node.target, ast.Name)
        and isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id == "range"
        and not it.keywords
        and 1 <= len(it.args) <= 3
    )


def _tiles(store: ast.Assign, loop_vars: set[str]) -> bool:
    """Does `store` (`v = v.at[index].set(...)`) write tiles placed by the loop variables, the way a kernel
    places them: does its index spell out a slice (`:`, `i : i + T`), mention every loop variable, and read
    each one only as a whole dim (`hd`, a tile of one) or as the start of a slice (`i : i + T`)?"""
    index = store.value.func.value.slice
    dims = index.elts if isinstance(index, ast.Tuple) else [index]
    if not any(isinstance(d, ast.Slice) for d in dims) or not loop_vars <= _loaded(index):
        return False
    for d in dims:
        if isinstance(d, ast.Name) or not _loaded(d) & loop_vars:
            continue
        lower = d.lower if isinstance(d, ast.Slice) else None
        if not (isinstance(lower, ast.Name) and lower.id in loop_vars):
            return False
    return True


def _bound(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _loaded(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _nest_stores(top: ast.For) -> list[ast.Assign]:
    """If `top` is a tile loop nest, the output stores its loops end in; else [] (an ordinary loop).

    A `for` statement is a tile loop nest when
    1. its loops are `for v in range(...)`, each body ending in the next loop (other statements may come
       before it, but no other loops), down to a body ending in stores `v = v.at[index].set(value)`
       (possibly under a tile mask `if`), and
    2. every store writes tiles placed by those loops' variables: its index spells out at least one slice
       (`:` or `i : i + T`), mentions every loop variable, and uses each one only as a whole dim (`hd`, a
       tile of one) or as the start of a slice (`i : i + T`),
    3. the loops don't read the arrays they store to (other than as the `v` of `v.at`): a loop that does
       carries values from one tile to the next, as a scan does, and
    4. no loop inside its tile body is itself a nest (that would be a Python loop around nests).
    Anything else is ordinary Python, which runs as written when the program is traced: an element-wise
    update `h = h.at[t].set(f(h[t - 1]))`, collecting results `outs = outs.at[layer].set(h)`, a loop over a
    list, a loop whose stores aren't at its end. A loop that is a nest by this rule runs as a kernel, so
    whatever in it a kernel can't do is an error naming the line.

    Walking down also gives up when the stored outputs are allocated on the way down: `for layer in
    range(L): h = jnp.zeros(...); for i in ...: h = h.at[...].set(...)` is a Python loop around a nest.
    """
    node, hoisted, loop_vars = top, [], set()
    while True:
        if not _is_range_loop(node):
            return []
        loop_vars.add(node.target.id)
        stores = _tile_stores(node.body)
        if stores:
            outputs = {_store_target(s) for s in stores}
            if any(outputs & _bound(s) for s in hoisted) or not all(_tiles(s, loop_vars) for s in stores):
                return []
            inner = [n for s in node.body for n in ast.walk(s) if isinstance(n, ast.For)]
            if any(_nest_stores(n) for n in inner):
                return []  # a Python loop around nests (that also stores something itself)
            bases = {id(s.value.func.value.value.value) for s in stores}  # the `v` of each `v.at`
            for n in ast.walk(top):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in outputs and id(n) not in bases:
                    return []
            return stores
        body = node.body
        if not body or not isinstance(body[-1], ast.For):
            return []
        if any(isinstance(n, (ast.For, ast.While)) for s in body[:-1] for n in ast.walk(s)):
            return []
        hoisted += body[:-1]
        node = body[-1]


def _blocks(s: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists inside a compound statement (not nested functions or classes)."""
    if isinstance(s, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        return [s.body, s.orelse]
    if isinstance(s, (ast.With, ast.AsyncWith)):
        return [s.body]
    if isinstance(s, (ast.Try, ast.TryStar)):
        return [s.body, *(h.body for h in s.handlers), s.orelse, s.finalbody]
    if isinstance(s, ast.Match):
        return [c.body for c in s.cases]
    return []


def _local_names(fdef: ast.FunctionDef) -> set[str]:
    """Names local to `fdef`: its parameters and everything its body binds (not nested scopes)."""
    a = fdef.args
    names = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    names |= {p.arg for p in (a.vararg, a.kwarg) if p is not None}
    declared: set[str] = set()
    stack: list[ast.AST] = list(fdef.body)
    scopes = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, scopes):
            continue
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    return names - declared


class _Finder:
    """Finds the tile loop nests of a function.

    With `strict`, it also checks each nest (with the frontend's checks too) and raises AutoKernelError
    naming the line on anything wrong; otherwise it only finds them.
    """

    def __init__(
        self, fdef: ast.FunctionDef, fn_name: str, first_line: int, strict: bool, body: list[ast.stmt] | None = None
    ):
        self.fn_name, self.first_line, self.strict = fn_name, first_line, strict
        self.locals = _local_names(fdef)
        self.regions: list[_Region] = []
        self.block(fdef.body if body is None else body)

    def line(self, node: ast.AST) -> int:
        return self.first_line + node.lineno - 1

    def block(self, stmts: list[ast.stmt]) -> None:
        for n, s in enumerate(stmts):
            if not isinstance(s, ast.For):
                for sub in _blocks(s):
                    self.block(sub)
                continue
            stores = _nest_stores(s)
            if not stores:
                self.block(s.body)  # an ordinary Python loop, possibly around nests
                self.block(s.orelse)
                continue
            outputs = list(dict.fromkeys(_store_target(st) for st in stores))
            self.region(stmts, s, outputs, {v: self.allocation(stmts, n, v) for v in outputs})

    @staticmethod
    def allocation(stmts: list[ast.stmt], n: int, var: str) -> ast.Assign | None:
        """The statement allocating output `var` of the nest `stmts[n]` that can move into the nest: the
        `var = expr` last binding `var` in the same block, when nothing between them reads `var` or
        changes what `expr` reads (and the nest doesn't bind what it reads either). Otherwise None: the
        nest starts from `var`'s current value, whatever computed it."""
        for a in range(n - 1, -1, -1):
            s = stmts[a]
            if var not in _bound(s):
                continue
            if _target(s) != var:
                return None  # last bound some other way: a loop variable, `var += ...`, unpacking, ...
            reads = _loaded(s.value)
            if reads & _bound(stmts[n]):
                return None
            for t in stmts[a + 1 : n]:
                if var in _loaded(t) or _bound(t) & reads:
                    return None
            return s
        return None

    def region(self, block: list[ast.stmt], loop: ast.For, outputs: list[str], found: dict) -> None:
        existing = [v for v in outputs if found[v] is None]
        allocs = []
        for v in outputs:  # `v = v` for an output the nest updates in place: its current value is the input
            a = found[v] or ast.Assign([ast.Name(v, ast.Store())], ast.Name(v, ast.Load()))
            allocs.append(ast.fix_missing_locations(ast.copy_location(a, loop)) if found[v] is None else a)
        loads = (n for node in [*allocs, loop] for n in ast.walk(node) if isinstance(n, ast.Name))
        reads = sorted((n for n in loads if isinstance(n.ctx, ast.Load)), key=lambda n: (n.lineno, n.col_offset))
        own = (_bound(loop) | set(outputs)) - set(existing)  # loop variables, temporaries and allocated outputs
        inputs = list(dict.fromkeys(n.id for n in reads if n.id not in own and n.id in self.locals))
        # Global and closure arrays it reads through subscripts are passed in like local ones.
        bases = sorted(
            (n.value for n in ast.walk(loop) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        outer = tuple(dict.fromkeys(g.id for g in bases if g.id not in own and g.id not in self.locals))
        taken = {r.name for r in self.regions}
        name = base = f"{self.fn_name}_{outputs[0]}"
        k = 1
        while name in taken:
            k += 1
            name = f"{base}_{k}"
        names = frozenset(n.id for n in reads)
        region = _Region(name, block, allocs, loop, outputs, inputs, names, frozenset(existing), outer)
        if self.strict:  # the frontend's checks, now: fail at decoration time
            frontend.parse_function(_nest_ast(region, inputs + list(outer)), name, self.first_line)
        self.regions.append(region)


def _nest_ast(region: _Region, params: list[str]) -> ast.FunctionDef:
    """The nest as a single-kernel function: `def name(*params): allocations; nest; return outputs`."""
    outs = [ast.Name(v, ast.Load()) for v in region.outputs]
    ret = ast.copy_location(ast.Return(outs[0] if len(outs) == 1 else ast.Tuple(outs, ast.Load())), region.loop)
    args = ast.arguments(posonlyargs=[], args=[ast.arg(p) for p in params], kwonlyargs=[], kw_defaults=[], defaults=[])
    body = [*copy.deepcopy(region.allocs), copy.deepcopy(region.loop), ret]
    fdef = ast.FunctionDef(name=region.name, args=args, body=body, decorator_list=[], type_params=[])
    return ast.fix_missing_locations(ast.copy_location(fdef, region.allocs[0]))


def _find(fdef: ast.FunctionDef, fn_name: str, first_line: int, strict: bool = False) -> list[_Region]:
    return _Finder(fdef, fn_name, first_line, strict).regions


def _index(block: list[ast.stmt], node: ast.stmt) -> int:
    return next(n for n, s in enumerate(block) if s is node)


# --------------------------------------------------------------------------- names a nest binds

_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def _scope_loads(node: ast.AST, bound: frozenset[str] = frozenset()) -> Iterator[ast.Name]:
    """The names `node` reads from its scope: not those its comprehensions, lambdas and nested functions
    bind themselves. Reads inside nested functions count (a closure reads the variable when it runs)."""
    if isinstance(node, _COMPREHENSIONS):
        gens = node.generators
        inner = bound | {n.id for g in gens for n in ast.walk(g.target) if isinstance(n, ast.Name)}
        yield from _scope_loads(gens[0].iter, bound)  # evaluated outside the comprehension
        for g in gens:
            for part in ([g.iter] if g is not gens[0] else []) + g.ifs:
                yield from _scope_loads(part, inner)
        for part in (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,):
            yield from _scope_loads(part, inner)
        return
    if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for part in [*a.defaults, *(d for d in a.kw_defaults if d is not None), *getattr(node, "decorator_list", [])]:
            yield from _scope_loads(part, bound)
        if isinstance(node, ast.Lambda):
            inner = bound | {p.arg for p in ast.walk(a) if isinstance(p, ast.arg)}
            yield from _scope_loads(node.body, inner)
        else:
            inner = bound | _local_names(node)
            for s in node.body:
                yield from _scope_loads(s, inner)
        return
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in bound:
            yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _scope_loads(child, bound)


def _scope_stores(s: ast.stmt) -> set[str]:
    """The names a simple statement binds in its scope."""
    inner = {
        n.id
        for c in ast.walk(s)
        if isinstance(c, (*_COMPREHENSIONS, ast.Lambda))
        for n in ast.walk(c)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    return _stores([s]) - inner


class _ReadsAfter:
    """Finds reads that may see a value bound inside a tile loop nest, following Python's control flow."""

    def __init__(self, regions: list[_Region]):
        self.nests = {id(r.loop): r for r in regions}

    @staticmethod
    def first(node: ast.AST | None, live: set[str]) -> ast.Name | None:
        if node is None:
            return None
        hits = [n for n in _scope_loads(node) if n.id in live]
        return min(hits, key=lambda n: (n.lineno, n.col_offset), default=None)

    def block(self, stmts: list[ast.stmt], live: set[str]) -> tuple[ast.Name | None, set[str]]:
        """The first read in `stmts` of a name in `live` that may happen before they bind it again, and the
        names of `live` they certainly bind."""
        start, live = set(live), set(live)
        for s in stmts:
            if not live:
                break
            hit, killed = self.stmt(s, live)
            if hit is not None:
                return hit, start - live
            live -= killed
        return None, start - live

    def stmt(self, s: ast.stmt, live: set[str]) -> tuple[ast.Name | None, set[str]]:
        if id(s) in self.nests:  # another nest reads its inputs (and the outputs it updates)
            r = self.nests[id(s)]
            own = _bound(r.loop) - r.existing
            return self.first(s, live - own), set()
        if isinstance(s, ast.If):
            h1, k1 = self.block(s.body, live)
            h2, k2 = self.block(s.orelse, live)
            return self.first(s.test, live) or h1 or h2, k1 & k2
        if isinstance(s, (ast.For, ast.AsyncFor)):
            hit = self.first(s.iter, live) or self.first(s.target, live)
            hit = hit or self.block(s.body, live - _bound(s.target))[0] or self.block(s.orelse, live)[0]
            return hit, set()
        if isinstance(s, ast.While):
            return self.first(s.test, live) or self.block(s.body, live)[0] or self.block(s.orelse, live)[0], set()
        if isinstance(s, (ast.With, ast.AsyncWith)):
            hit, inner = None, set(live)
            for item in s.items:
                hit = hit or self.first(item.context_expr, inner)
                if item.optional_vars is not None:
                    inner -= _bound(item.optional_vars)
            h, k = self.block(s.body, inner)
            return hit or h, k | (live - inner)
        if isinstance(s, (ast.Try, ast.TryStar)):
            hit, k = self.block(s.body, live)
            for handler in s.handlers:
                hit = hit or self.first(handler.type, live) or self.block(handler.body, live - {handler.name})[0]
            hit = hit or self.block(s.orelse, live - k)[0]
            h, k = self.block(s.finalbody, live)
            return hit or h, k
        if isinstance(s, ast.Match):
            hit = self.first(s.subject, live)
            for case in s.cases:
                hit = hit or self.first(case.guard, live) or self.block(case.body, live)[0]
            return hit, set()
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return self.first(s, live), {s.name}
        hit = self.first(s, live)
        if hit is None and isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name) and s.target.id in live:
            hit = s.target
        return hit, _scope_stores(s)


def _path(stmts: list[ast.stmt], node: ast.stmt) -> list[tuple[list[ast.stmt], int]] | None:
    """The (block, index) of each statement from the function body down to `node`."""
    for n, s in enumerate(stmts):
        if s is node:
            return [(stmts, n)]
        for sub in _blocks(s):
            inner = _path(sub, node)
            if inner is not None:
                return [(stmts, n), *inner]
    return None


def _check_reads_after(fdef: ast.FunctionDef, regions: list[_Region], fn_name: str, first_line: int) -> None:
    """Reject reads of a nest's loop variables and temporaries that may happen after the nest.

    In Python they would see the values the nest's loops left behind; a nest runs as a kernel, so those
    names aren't bound by it at all (the program would read an earlier value, or fail)."""
    finder = _ReadsAfter(regions)
    for r in reversed(regions):
        live = _bound(r.loop) - set(r.outputs)
        path = _path(fdef.body, r.loop)
        hit = None
        for depth in range(len(path) - 1, -1, -1):
            block, index = path[depth]
            hit, killed = finder.block(block[index + 1 :], live)
            live = live - killed
            if hit is not None or not live or depth == 0:
                break
            parent = path[depth - 1][0][path[depth - 1][1]]
            if isinstance(parent, (ast.For, ast.AsyncFor, ast.While)) and block is parent.body:
                # The loop's next iteration, then its `else`.
                again = live - _bound(parent.target) if isinstance(parent, (ast.For, ast.AsyncFor)) else live
                hit = finder.first(getattr(parent, "test", None), live) or finder.block(parent.body, again)[0]
                hit = hit or finder.block(parent.orelse, live)[0]
            elif isinstance(parent, (ast.Try, ast.TryStar)) and block is parent.body:
                hit = finder.block(parent.orelse, live)[0] or finder.block(parent.finalbody, live)[0]
            if hit is not None:
                break
        if hit is not None:
            targets = {id(n.target) for n in ast.walk(r.loop) if isinstance(n, ast.For)}
            loop_vars = {n.id for n in ast.walk(r.loop) if isinstance(n, ast.Name) and id(n) in targets}
            kind = "a loop variable" if hit.id in loop_vars else "a temporary"
            raise AutoKernelError(
                f"{fn_name} (line {first_line + hit.lineno - 1}): `{hit.id}` is read here, but its value may come "
                f"from the tile loop nest at line {first_line + r.loop.lineno - 1}, where it is {kind}; the nest "
                "runs as a kernel, so its loop variables and temporaries don't exist after it: use another name "
                "inside the nest, or compute the value outside it"
            )


def _program_ast(
    fdef: ast.FunctionDef, fn_name: str, first_line: int
) -> tuple[ast.FunctionDef, list[_Region], list[tuple[list[int], list[str], list[str]]]]:
    """The function with each nest (and its allocations) replaced by a call of the nest's kernels, each run
    of consecutive nests by one call of the run (a *group*: the nests a fused kernel can run), and each
    Python `for` loop around nests by a call of `_LOOP` (which runs it as a `jax.lax.fori_loop` if it can).

    Returns the function, the nests, and per group: (its nests' indices, its inputs, its outputs).
    """
    fdef = copy.deepcopy(fdef)
    regions = _find(fdef, fn_name, first_line, strict=True)
    _check_reads_after(fdef, regions, fn_name, first_line)
    calls: dict[int, int] = {}  # id(call statement) -> nest index
    for k, r in enumerate(regions):
        values = ast.Dict([ast.Constant(n) for n in r.inputs], [ast.Name(n, ast.Load()) for n in r.inputs])
        call = ast.Call(ast.Subscript(ast.Name(_NESTS, ast.Load()), ast.Constant(k), ast.Load()), [values], [])
        names = [ast.Name(v, ast.Store()) for v in r.outputs]
        target = names[0] if len(names) == 1 else ast.Tuple(names, ast.Store())
        stmt = ast.copy_location(ast.Assign([target], call), r.loop)
        r.block[_index(r.block, r.loop)] = stmt
        calls[id(stmt)] = k
        for var, alloc in zip(r.outputs, r.allocs, strict=True):
            if var not in r.existing:  # moved into the nest
                del r.block[_index(r.block, alloc)]
    groups = []
    local = _local_names(fdef)
    for block in {id(r.block): r.block for r in regions}.values():
        groups += _group_runs(block, calls, regions, local)
    _roll_loops(fdef)
    return ast.fix_missing_locations(fdef), regions, _number_groups(groups)


# Layout glue between nests that only relabels an array's elements: `x.reshape(...)`, `x.T`, `x[0]`, ...
_VIEW_METHODS = frozenset({"reshape", "transpose", "swapaxes", "squeeze"})
_VIEW_FUNCTIONS = frozenset({"reshape", "transpose", "swapaxes", "moveaxis", "squeeze", "expand_dims"})
_ENV = "__autokernel_env"  # the parameter of the functions computing views (`_ViewStmt`)


@dataclass(frozen=True)
class _ViewStmt:
    """`target = expr` between the nests of a run, where `expr` reshapes, transposes or indexes `base`
    (integers and whole slices): possibly a *view* of `base` (see `views`), which a later nest can read
    without the copy. Whether it is one is decided when the program is traced, from what JAX makes of it."""

    target: str
    base: str
    reads: tuple[str, ...]  # the local names `expr` reads, `base` first
    line: int


def _view_stmt(s: ast.stmt, local: set[str], first_line: int = 1) -> _ViewStmt | None:
    """`s` as a `_ViewStmt`, if it looks like one: `name = <reshapes, transposes, indexing of a local>`."""
    if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)):
        return None
    if any(isinstance(n, (ast.Lambda, *_COMPREHENSIONS, ast.NamedExpr, ast.Await)) for n in ast.walk(s.value)):
        return None
    e, ops = s.value, 0
    while not isinstance(e, ast.Name):
        if isinstance(e, ast.Attribute) and e.attr in ("T", "mT"):
            e = e.value
        elif isinstance(e, ast.Subscript):
            dims = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
            whole = all(
                d.lower is None and d.upper is None and d.step is None for d in dims if isinstance(d, ast.Slice)
            )
            if not whole:
                return None
            e = e.value
        elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and not e.keywords:
            f = e.func
            if any(isinstance(a, ast.Starred) for a in e.args):
                return None
            if isinstance(f.value, ast.Name) and f.value.id not in local:  # `jnp.reshape(x, ...)`
                if f.attr not in _VIEW_FUNCTIONS or not e.args:
                    return None
                e = e.args[0]
            elif f.attr in _VIEW_METHODS:  # `x.reshape(...)`
                e = f.value
            else:
                return None
        else:
            return None
        ops += 1
    if not ops or e.id not in local:
        return None
    reads = [
        n.id for n in ast.walk(s.value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in local
    ]
    reads = list(dict.fromkeys([e.id, *reads]))
    return _ViewStmt(s.targets[0].id, e.id, tuple(reads), first_line + s.lineno - 1)


def _view_fn(s: ast.Assign, local: set[str]) -> ast.Lambda:
    """`lambda env: expr` for a view statement, its local names read from `env`."""

    class FromEnv(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if isinstance(n.ctx, ast.Load) and n.id in local:
                key = ast.Subscript(ast.Name(_ENV, ast.Load()), ast.Constant(n.id), ast.Load())
                return ast.copy_location(key, n)
            return n

    body = FromEnv().visit(copy.deepcopy(s.value))
    args = ast.arguments(posonlyargs=[], args=[ast.arg(_ENV)], kwonlyargs=[], kw_defaults=[], defaults=[])
    return ast.copy_location(ast.Lambda(args, body), s)


def _group_runs(
    block: list[ast.stmt], calls: dict[int, int], regions: list[_Region], local: set[str]
) -> list[tuple[list[int | _ViewStmt], list[str], list[str], ast.Constant]]:
    """Replace each run of consecutive nest calls in `block`, possibly with view statements (`_ViewStmt`)
    between and before them, by one call of the run (a group), when it holds two nests or a nest and a view:
    `outputs = __autokernel_groups[g]({inputs}, (lambda env: <view 1>, ...))`."""
    kinds: list[int | _ViewStmt | None] = []
    for s in block:
        kinds.append(calls[id(s)] if id(s) in calls else _view_stmt(s, local))
    runs: list[list[int]] = [[]]
    for n, kind in enumerate(kinds):
        if kind is not None:
            runs[-1].append(n)
        elif runs[-1]:
            runs.append([])
    groups = []
    for run in reversed(runs):  # from the end: indices before stay valid
        while run and not isinstance(kinds[run[-1]], int):  # views after the last nest stay plain JAX
            run.pop()
        nests = sum(isinstance(kinds[n], int) for n in run)
        if nests == 0 or (nests == 1 and len(run) == 1):
            continue
        items = [kinds[n] for n in run]
        inputs: list[str] = []
        outputs: list[str] = []
        for item in items:
            reads, writes = (regions[item].inputs, regions[item].outputs) if isinstance(item, int) else (
                item.reads, [item.target])  # fmt: skip
            inputs += [v for v in reads if v not in outputs and v not in inputs]
            outputs += [v for v in writes if v not in outputs]
        values = ast.Dict([ast.Constant(v) for v in inputs], [ast.Name(v, ast.Load()) for v in inputs])
        fns = ast.Tuple([_view_fn(block[n], local) for n in run if not isinstance(kinds[n], int)], ast.Load())
        g = ast.Constant(-1)  # numbered below, in source order
        call = ast.Call(ast.Subscript(ast.Name(_GROUPS, ast.Load()), g, ast.Load()), [values, fns], [])
        names = [ast.Name(v, ast.Store()) for v in outputs]
        target = names[0] if len(names) == 1 else ast.Tuple(names, ast.Store())
        block[run[0] : run[-1] + 1] = [ast.copy_location(ast.Assign([target], call), block[run[0]])]
        groups.append((items, inputs, outputs, g))
    return groups


def _number_groups(groups: list) -> list[tuple[list[int | _ViewStmt], list[str], list[str]]]:
    """Number the groups in the order of their first nest, and point their calls at those numbers."""
    groups = sorted(groups, key=lambda g: min(k for k in g[0] if isinstance(k, int)))
    for n, (_, _, _, const) in enumerate(groups):
        const.value = n
    return [(items, inputs, outputs) for items, inputs, outputs, _ in groups]


# --------------------------------------------------------------------------- Python loops around nests


def _calls_kernels(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id in (_NESTS, _GROUPS) for n in ast.walk(node))


def _walk_scope(nodes: list[ast.stmt]) -> Iterator[ast.AST]:
    """Every node in `nodes`, not descending into nested functions, classes or lambdas."""
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _escapes(body: list[ast.stmt]) -> bool:
    """Does control leave `body` other than by falling off its end (return, break, yield, ...)?"""
    for node in _walk_scope(body):
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
            return True
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
            return True
    stack: list[ast.AST] = list(body)  # break / continue of this loop (not of loops inside it)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Break, ast.Continue)):
            return True
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Lambda)):  # fmt: skip
            stack.extend(ast.iter_child_nodes(node))
    return False


def _stores(nodes: list[ast.stmt]) -> set[str]:
    names = set()
    for node in _walk_scope(nodes):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _upward_exposed(body: list[ast.stmt]) -> set[str]:
    """Names `body` may read before writing them: the values a loop carries from one iteration to the next.

    Statement by statement, a statement's reads count as happening before its writes.
    """
    written: set[str] = set()
    exposed: set[str] = set()
    for s in body:
        exposed |= {n.id for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - written
        written |= _stores([s])
    return exposed


def _roll_loops(fdef: ast.FunctionDef) -> None:
    """Replace each Python `for v in range(...)` loop around nests whose body can run as a function of `v`
    and the values it carries by a call of `_LOOP` (innermost loops first)."""
    a = fdef.args
    params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    params |= {p.arg for p in (a.vararg, a.kwarg) if p is not None}
    counter = iter(range(1 << 30))

    def outside_loads(loop: ast.For) -> set[str]:
        inside = {id(n) for n in ast.walk(loop)}
        return {
            n.id
            for n in ast.walk(fdef)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in inside
        }

    def roll(loop: ast.For, bound: set[str]) -> list[ast.stmt] | None:
        it = loop.iter
        ok = (
            not loop.orelse
            and isinstance(loop.target, ast.Name)
            and isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
            and 1 <= len(it.args) <= 3
            and not any(isinstance(x, ast.Starred) for x in it.args)
        )
        if not ok or _escapes(loop.body):
            return None
        var = loop.target.id
        assigned = _stores(loop.body)
        loads = outside_loads(loop)
        if var in assigned or var in loads:
            return None
        carried = sorted((assigned & (loads | _upward_exposed(loop.body))) - {var})
        if any(c not in bound for c in carried):
            return None  # a value the loop leaves behind that has none before it
        k = next(counter)
        name = f"{_BODY}_{k}"

        def names(ctx: Any) -> ast.Tuple:
            return ast.Tuple([ast.Name(c, ctx()) for c in carried], ctx())

        body = [ast.Assign([names(ast.Store)], ast.Name(_CARRY, ast.Load()))] if carried else []
        body += [*loop.body, ast.Return(names(ast.Load))]
        args = ast.arguments(
            posonlyargs=[], args=[ast.arg(var), ast.arg(_CARRY)], kwonlyargs=[], kw_defaults=[], defaults=[]
        )
        fn = ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], type_params=[])
        bounds = ast.Tuple(list(it.args), ast.Load())
        call = ast.Call(ast.Name(_LOOP, ast.Load()), [bounds, ast.Name(name, ast.Load()), names(ast.Load)], [])
        run = ast.Assign([names(ast.Store)], call) if carried else ast.Expr(call)
        out = [ast.copy_location(fn, loop), ast.copy_location(run, loop)]
        for node in out:
            for sub in ast.walk(node):
                if not hasattr(sub, "lineno"):
                    ast.copy_location(sub, loop)
        return out

    def process(block: list[ast.stmt], bound: set[str]) -> None:
        bound = set(bound)
        out: list[ast.stmt] = []
        for s in block:
            inner = bound | (_stores([s]) if isinstance(s, (ast.For, ast.AsyncFor, ast.While)) else set())
            for sub in _blocks(s):
                process(sub, inner)
            rolled = roll(s, bound) if isinstance(s, ast.For) and _calls_kernels(s) else None
            out += rolled if rolled is not None else [s]
            bound |= _stores([s])
        block[:] = out

    process(fdef.body, params)


def _compile(fdef: ast.FunctionDef, fn: Callable, filename: str, first_line: int, extra: dict[str, Any]) -> Callable:
    """Compile a rewritten version of `fn` with `fn`'s globals (live), closure values and `extra` names."""
    fdef = copy.deepcopy(fdef)
    fdef.decorator_list = []
    closure = {k: v for k, v in inspect.getclosurevars(fn).nonlocals.items() if k not in extra}
    free = {**closure, **extra}
    # def __autokernel_factory(<free names>): def fn(...): ...; return fn
    args = ast.arguments(posonlyargs=[], args=[ast.arg(n) for n in free], kwonlyargs=[], kw_defaults=[], defaults=[])
    ret = ast.copy_location(ast.Return(ast.Name(fdef.name, ast.Load())), fdef)
    factory = ast.FunctionDef("__autokernel_factory", args, [fdef, ret], [], type_params=[])
    module = ast.fix_missing_locations(ast.Module([ast.copy_location(factory, fdef)], type_ignores=[]))
    ast.increment_lineno(module, first_line - 1)
    code = compile(module, filename, "exec", dont_inherit=True)
    factory_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    return types.FunctionType(factory_code, fn.__globals__)(*free.values())


# --------------------------------------------------------------------------- how a nest uses names


def _uses(region: _Region) -> dict[str, set[tuple[str, ...]]]:
    """How the nest uses each name it reads, as a set of contexts per name:

    - `("index", base)`: in an integer index of `base[...]` (an input read through a tile slice);
    - `("body",)`: as a value in the tile body;
    - `("static",)`: where the value is needed at trace time: loop bounds, masks, slice bounds and tile
      sizes, output allocations and store indices, or anything else.

    An integer read only in the first two ways can be a runtime scalar of the kernel instead of a constant.
    """
    uses: dict[str, set[tuple[str, ...]]] = {}
    static = ("static",)

    def visit(node: ast.AST, ctx: tuple[str, ...]) -> None:
        if ctx[0] == "index" and isinstance(node, (ast.Call, ast.Subscript, ast.Attribute)):
            ctx = static  # a table lookup or a call on the index: needs its value
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                uses.setdefault(node.id, set()).add(ctx)
            return
        if isinstance(node, ast.For):
            visit(node.iter, static)
            for st in node.body + node.orelse:
                visit(st, ("body",))
            return
        if isinstance(node, ast.If):
            visit(node.test, static)
            for st in node.body + node.orelse:
                visit(st, ("body",))
            return
        if isinstance(node, ast.Assign) and _store_target(node):
            visit(node.value.func.value.slice, static)
            visit(node.value.args[0], ("body",))
            return
        if isinstance(node, ast.Subscript) and ctx[0] == "body":
            visit(node.value, ctx)
            dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            base = node.value.id if isinstance(node.value, ast.Name) else None
            for d in dims:
                if isinstance(d, ast.Slice):
                    for bound in (d.lower, d.upper, d.step):
                        if bound is not None:
                            visit(bound, static)
                else:
                    visit(d, ("index", base) if base is not None else static)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, ctx)

    for alloc in region.allocs:
        visit(alloc.value, static)
    visit(region.loop, ("body",))
    return uses


# --------------------------------------------------------------------------- runtime


_INT32 = np.iinfo(np.int32)


def _is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct, View))


def _materialise(x: Any) -> Any:
    """A view as the array it stands for (an XLA copy); anything else as it is."""
    return x.materialise() if isinstance(x, View) else x


def _is_int(x: Any) -> bool:
    """A single integer: a Python or NumPy int, or a 0-d integer array (possibly traced)."""
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    return _is_array(x) and tuple(x.shape) == () and jnp.issubdtype(x.dtype, jnp.integer)


def _is_float(x: Any) -> bool:
    """A 0-d float array (possibly traced): `scale = 1 / jnp.sqrt(d)`. (Python floats are constants.)"""
    return _is_array(x) and tuple(x.shape) == () and jnp.issubdtype(x.dtype, jnp.floating)


def _weak_type(x: Any) -> bool:
    return bool(getattr(getattr(x, "aval", x), "weak_type", False))


def _known(x: Any) -> bool:
    """Is the value of `x` known now (not traced, not just a shape)?"""
    return not isinstance(x, (jax.core.Tracer, jax.ShapeDtypeStruct))


def _is_static(arg: Any) -> bool:
    """A program argument with no arrays in it (a number, a string, a tuple of numbers): static under jit."""
    return not any(_is_array(leaf) for leaf in jax.tree.leaves(arg))


def _concrete(args: tuple[Any, ...]) -> bool:
    leaves = [leaf for leaf in jax.tree.leaves(args) if _is_array(leaf)]
    return all(isinstance(x, np.ndarray) or not isinstance(x, (jax.core.Tracer, jax.ShapeDtypeStruct)) for x in leaves)


def _arg_key(arg: Any) -> Any:
    if _is_static(arg):
        return ("static", arg)
    leaves, tree = jax.tree.flatten(arg)
    specs = tuple((tuple(x.shape), np.dtype(x.dtype)) if _is_array(x) else type(x) for x in leaves)
    return (tree, specs)


def _spec(x: Any) -> jax.ShapeDtypeStruct:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    if isinstance(x, View):
        return x.spec()
    if isinstance(x, (int, np.integer)):
        return jax.ShapeDtypeStruct((), jnp.int32)
    if _is_float(x):  # a float runtime scalar mixes with tiles as its weak type says
        return jax.ShapeDtypeStruct((), x.dtype, weak_type=_weak_type(x))
    return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)


Kernel = Lowered | fusion.FusedLowered


@dataclass(frozen=True)
class ProgramLowered:
    """The kernels a program runs for one set of inputs, in the order it first runs them. A fused kernel
    (several nests in one `pallas_call`) is a `FusedLowered`, whose `parts` are its nests' kernels."""

    kernels: tuple[tuple[str, Kernel], ...]  # (name, kernel)
    inputs: tuple[tuple[jax.ShapeDtypeStruct, ...], ...] = ()  # each kernel's inputs: `kernel.entry(*inputs)`
    # Where runs of nests were cut into several kernels, and why (fewest kernels, least data between them).
    cuts: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self) -> Iterator[tuple[str, Kernel]]:
        return iter(self.kernels)

    def __getitem__(self, key: int | str) -> Any:
        """`lowered[n]`: the n-th (name, kernel); `lowered["name"]`: the kernel of that name, or the part of a
        fused kernel of that name."""
        if isinstance(key, str):
            for name, lowered in self.kernels:
                if name == key:
                    return lowered
            for _, lowered in self.kernels:
                for name, part in getattr(lowered, "parts", ()):
                    if name == key:
                        return part
            raise KeyError(f"no kernel named {key!r}; the kernels are {self.names}")
        return self.kernels[key]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.kernels]

    @property
    def source(self) -> str:
        """Every kernel's generated Pallas source, each under a header naming it."""
        parts = [
            f"# {'=' * 20} kernel {n + 1}/{len(self)}: {name} {'=' * 20}\n\n{lowered.source}"
            for n, (name, lowered) in enumerate(self.kernels)
        ]
        return "\n\n".join(parts)


@dataclass(frozen=True)
class NestCall:
    """How a nest runs for one set of values: its kernel, which of the values it takes (in order), and the
    kernel lowered for them."""

    nest: ProgramNest
    key: tuple  # the kernel's key in `nest.kernels`
    kernel: TpuKernel
    params: tuple[str, ...]  # the names of its arrays and runtime scalars, in kernel-parameter order
    lowered: Lowered
    views: frozenset[str] = frozenset()  # params that are views, read as windows of their base (given the base)

    def args(self, values: dict[str, Any]) -> list[Any]:
        """The kernel's arguments for these values: a view's base for the views it reads as windows of it,
        other views made into arrays."""
        return [values[n].root if n in self.views else _materialise(values[n]) for n in self.params]


class ProgramNest:
    """One tile loop nest of a program, and its kernels: one per distinct set of constants it is given.

    Integers the nest only uses as indices of its inputs or as values in its tile body (a layer number,
    `w[layer, ...]`) are runtime scalars of the kernel rather than constants, so one kernel serves every
    value; integers it needs at trace time (loop bounds, masks, slices, allocations) are constants.
    """

    def __init__(self, program: TpuProgram, region: _Region):
        self.program, self.region = program, region
        self.name = region.name
        self.line = program.first_line + region.loop.lineno - 1  # of the nest's outer `for`
        self.inputs = list(region.inputs)  # the program's local names the nest reads
        self.globals = list(region.globals)  # global and closure names it reads through subscripts
        self.outputs = list(region.outputs)
        self.kernels: dict[tuple, TpuKernel] = {}  # (arrays, scalars, constants) -> kernel
        self.uses = _uses(region)
        self.compiled_in: set[str] = set()  # Python ints that turned out to be needed at trace time
        self.unmapped: set[tuple[str, tuple]] = set()  # (param, view) pairs it can't read as windows of a base
        self._compiled: dict[int, Callable] = {}  # kernels reading views: by id(lowered)

    def __repr__(self) -> str:
        return f"<ProgramNest {self.name} (line {self.line}): {', '.join(self.outputs)}>"

    def names(self, values: dict[str, Any]) -> list[str]:
        """The names of the nest's values: its local inputs, then the global arrays it reads (`with_globals`)."""
        return self.inputs + [g for g in self.globals if g in values]

    def with_globals(self, values: dict[str, Any]) -> dict[str, Any]:
        """`values` (the local names the nest reads), plus the global and closure arrays it reads through
        subscripts, as they are now: those are kernel inputs like local arrays, not constants."""
        return {**self.program._outer_arrays(self.globals), **values}

    def _split(self, values: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
        """The nest's values as (arrays, runtime scalars, constants)."""
        # A 0-d float array only used as a value in the tile body is a runtime scalar (read from SMEM).
        floats = [
            n
            for n in self.names(values)
            if _is_float(values[n]) and n not in self.compiled_in and all(c == ("body",) for c in self.uses.get(n, ()))
        ]
        arrays = [n for n in self.names(values) if _is_array(values[n]) and not _is_int(values[n]) and n not in floats]
        scalars, consts = [], {}
        for n in self.names(values):
            v = values[n]
            if n in arrays:
                continue
            if n in floats:
                scalars.append(n)
                continue
            if not _is_int(v):
                consts[n] = v
                continue
            runtime = all(c == ("body",) or (c[0] == "index" and c[1] in arrays) for c in self.uses.get(n, ()))
            if _known(v):
                v = int(v)
                runtime = runtime and n not in self.compiled_in and _INT32.min <= v <= _INT32.max
            if runtime:
                scalars.append(n)
            elif _known(v):
                consts[n] = v
            else:
                raise AutoKernelError(
                    f"{self.name} (line {self.line}): `{n}` is a traced integer (computed by JAX, or the variable "
                    "of a loop run as `jax.lax.fori_loop`), but the nest needs its value at trace time (a loop "
                    "bound, a mask, a slice or an allocation); only integer indices of inputs and values in the "
                    "tile body can be runtime values"
                )
        return arrays, scalars, consts

    def prepare(self, values: dict[str, Any]) -> NestCall:
        """The kernel for these values (arrays, or their shapes), lowered."""
        values = self.with_globals(values)
        while True:
            arrays, scalars, consts = self._split(values)
            # Float scalars: one kernel per dtype and weak type (how they mix with tiles).
            scalar_key = tuple((n, np.dtype(values[n].dtype).name, _weak_type(values[n])) if _is_float(values[n])
                               else n for n in scalars)  # fmt: skip
            key = (tuple(arrays), scalar_key, tuple((n, type(v), self._hashable(n, v)) for n, v in consts.items()))
            # Views (of an earlier nest's output, say) it reads as windows of their base array: no copy.
            remap = {
                n: values[n]
                for n in arrays
                if isinstance(values[n], View) and not self.program._eager and (n, values[n].key) not in self.unmapped
            }
            if remap:
                key += (tuple((n, v.key) for n, v in remap.items()),)
            params = tuple(n for n in self.names(values) if n in arrays or n in scalars)
            kernel = self.kernels.get(key)
            if kernel is None:
                kernel = self.kernels[key] = self._kernel(list(params), consts, {n: v.map for n, v in remap.items()})
            try:
                lowered = kernel.lower(*[_spec(values[n]) for n in params])
            except AutoKernelError as e:
                if remap:  # a view it can't read as windows of its base: it reads a copy (made by XLA) instead
                    del self.kernels[key]
                    bad = [e.param] if isinstance(e, ViewError) and e.param in remap else list(remap)
                    self.unmapped |= {(n, remap[n].key) for n in bad}
                    continue
                # A Python int used in the tile body in a way that needs its value: compile it in instead.
                retry = {n for n in scalars if _known(values[n]) and ("body",) in self.uses.get(n, ())}
                if not retry:
                    raise
                del self.kernels[key]
                self.compiled_in |= retry
                continue
            lowered.check_indices({n: values[n] for n in scalars if _known(values[n]) and _is_int(values[n])})
            return NestCall(self, key, kernel, params, lowered, frozenset(remap))

    def reference(self, values: dict[str, Any]) -> Any:
        """Run the nest as Python loops, as its kernel's `reference` does (with its tuned constants)."""
        values = {n: _materialise(v) for n, v in self.with_globals(values).items()}
        call = self.prepare(values)
        out = call.kernel.reference(*[values[n] for n in call.params])
        if self.program._nest_log is not None:  # `verify_nests`: what each nest ran on, and what it gave
            self.program._nest_log.append((self, values, out))
        return out

    def __call__(self, values: dict[str, Any]) -> Any:
        values = self.with_globals(values)
        call = self.prepare(values)
        args = call.args(values)
        recording = self.program._recording
        if recording is None:
            return call.kernel(*args) if not call.views else self._run_views(call, values, args)
        # Tracing for `lower`: record the kernel, and stand in for its outputs without running it.
        recording.append(("nest", call, tuple(map(_spec, args))))
        outs = tuple(jnp.zeros(o.out.shape, o.out.dtype) for o in call.lowered.outputs)
        return outs if len(outs) > 1 else outs[0]

    def _hashable(self, name: str, value: Any) -> Any:
        try:
            hash(value)
        except TypeError:
            raise AutoKernelError(
                f"{self.name} (line {self.line}): `{name}` is a {type(value).__name__}; a tile loop nest can only "
                "read arrays and constants that can be compiled into the kernel (numbers, dtypes, tuples)"
            ) from None
        return value

    def _run_views(self, call: NestCall, values: dict[str, Any], args: list[Any]) -> Any:
        """Run a kernel that reads views as windows of their bases (it takes the bases, so it can't run as
        `call.kernel(...)`, which takes the views' arrays)."""
        lowered, kernel = call.lowered, call.kernel
        lowered.check_indices({p: a for p, a in zip(call.params, args, strict=True) if p in lowered.scalars})
        fn = self._compiled.get(id(lowered))
        if fn is None:
            fn = self._compiled[id(lowered)] = jax.jit(
                functools.partial(lowered.entry, interpret=kernel._interpret_mode())
            )
        views = {n: values[n] for n in call.views}

        def arrays(*xs: Any) -> list[Any]:
            return [views[n].on(x).materialise() if n in views else x for n, x in zip(call.params, xs, strict=True)]

        def reference(*xs: Any) -> Any:
            return kernel.reference(*arrays(*xs))

        def unfused(*xs: Any) -> Any:  # the kernel on the views' arrays: what runs batched or sharded
            return kernel(*arrays(*xs))

        return autodiff.differentiable(fn, reference, None, fallback=unfused)(*args)

    def _kernel(self, params: list[str], consts: dict[str, Any], views: dict | None = None) -> TpuKernel:
        program = self.program
        nest = frontend.parse_function(_nest_ast(self.region, params), self.name, program.first_line)
        names = analysis.user_globals(program.fn)
        names.update(consts)
        options = program.options
        # Tune only the constants this nest uses (names that aren't identifiers stand for "every block size").
        tune = tuple((k, v) for k, v in options.tune if k in self.region.names or not k.isidentifier())
        if tune != options.tune:
            options = dataclasses.replace(options, tune=tune)
        env = analysis.Environment(names, program.filename, dict(views or {}))
        return TpuKernel(env, nest=nest, interpret=program.interpret, options=options)


@dataclass(frozen=True, eq=False)
class _Ext:
    """A value the group was given, by name."""

    name: str


@dataclass(frozen=True, eq=False)
class _Out:
    """Output `var` of the group's k-th nest."""

    k: int
    var: str


@dataclass(frozen=True, eq=False)
class _Viewed:
    """A view of `root` (an `_Ext`, `_Out` or `_Glue`), as a `views.View` of root's plan-time value."""

    root: Any
    view: View


@dataclass(frozen=True, eq=False)
class _Glue:
    """The value of a view statement that isn't a view: JAX code on `deps` (name -> value)."""

    item: _ViewStmt
    deps: dict[str, Any]


@dataclass
class _Plan:
    """How a group runs for one set of values: each nest's kernel (`calls`), the value of each name it reads
    (`names`), the names bound at the end (`binding`), and the kernels it runs (`segments`)."""

    calls: list[NestCall]
    names: list[dict[str, Any]]
    binding: dict[str, Any]
    nests: list[fusion.NestPlan]
    segments: list[fusion.Segment]
    out_specs: dict[tuple[int, str], jax.ShapeDtypeStruct]


class ProgramGroup:
    """A run of consecutive nests, with no code between them but view statements (`x = y.reshape(...)`,
    `.transpose`, integer indexing: see `views`): fused into one kernel where they can be.

    It plans every nest first (standing in for the arrays earlier nests write with their shapes), reading
    a view of an array as windows of the array itself where the nest's tiles allow it (no copy), cuts the run
    into segments that can share a kernel (`fusion.plan_segments`), and runs each segment as one fused
    kernel, or a lone nest as its own kernel. The group's outputs (views included, as arrays) are what the
    statements would have bound; XLA drops the ones nothing uses.
    """

    def __init__(
        self, program: TpuProgram, items: list[ProgramNest | _ViewStmt], inputs: list[str], outputs: list[str]
    ):
        self.program, self.items, self.inputs, self.outputs = program, items, inputs, outputs
        self.nests = [item for item in items if isinstance(item, ProgramNest)]
        self.views = [item for item in items if isinstance(item, _ViewStmt)]
        self.fused: dict[tuple, fusion.FusedLowered] = {}
        self._compiled: dict[int, Callable] = {}

    def __repr__(self) -> str:
        return f"<ProgramGroup {', '.join(n.name for n in self.nests)}>"

    def _env(self, values: dict[str, Any], view_fns: tuple) -> tuple[dict[str, Any], dict[int, Callable]]:
        env = dict(values)
        for nest in self.nests:
            env = nest.with_globals(env)
        return env, {id(item): fn for item, fn in zip(self.views, view_fns, strict=True)}

    def __call__(self, values: dict[str, Any], view_fns: tuple = ()) -> Any:
        program = self.program
        env, fns = self._env(values, view_fns)
        if not program.fuse or program._eager:
            return self._sequential(env, fns, reference=False)
        plan = self._plan(env, fns)
        results: dict[tuple[int, str], Any] = {}
        glue: dict[int, Any] = {}

        def actual(v: Any) -> Any:
            """The value of a plan's value now: an array, or a `View` of one."""
            if isinstance(v, _Ext):
                return env[v.name]
            if isinstance(v, _Out):
                return results[(v.k, v.var)]
            if isinstance(v, _Viewed):
                return v.view.on(_materialise(actual(v.root)))
            if id(v) not in glue:
                fn = fns[id(v.item)]
                glue[id(v)] = fn({r: _materialise(actual(d)) for r, d in v.deps.items()})
            return glue[id(v)]

        recording = program._recording
        for seg in plan.segments:
            if recording is not None and seg.end < len(self.nests):
                recording.append(("cut", seg.reason))
            if seg.end - seg.start == 1:
                k, nest = seg.start, self.nests[seg.start]
                outs = nest({n: actual(v) for n, v in plan.names[k].items()})
                outs = outs if len(nest.outputs) > 1 else (outs,)
                results.update(zip([(k, v) for v in nest.outputs], outs, strict=True))
                continue
            fused, given, sources = self._fuse(plan, seg)
            args = []
            for key in fused.inputs:
                v, remapped = given[key]
                args.append(_materialise(actual(v.root)) if remapped else _materialise(actual(v)))
            calls = plan.calls[seg.start : seg.end]
            consts = [
                {n: actual(v) for n, v in plan.names[k].items() if n not in call.params}
                for k, call in zip(range(seg.start, seg.end), calls, strict=True)
            ]
            for k, call in zip(range(seg.start, seg.end), calls, strict=True):  # traced indices: checkify checks
                if call.lowered.scalars:
                    call.lowered.check_indices({p: actual(plan.names[k][p]) for p in call.lowered.scalars})
            if recording is not None:
                recording.append(("fused", fused, tuple(map(_spec, args)), calls))
                outs = [jnp.zeros(o.out.shape, o.out.dtype) for _, lw in fused.parts for o in lw.outputs]
            else:
                outs = self._run(fused, plan, seg, args, sources, consts)
            for (j, var), out in zip(fused.outputs, outs, strict=True):
                results[(seg.start + j, var)] = out
        outs = tuple(_materialise(actual(plan.binding[v])) for v in self.outputs)
        return outs if len(outs) > 1 else outs[0]

    def reference(self, values: dict[str, Any], view_fns: tuple = ()) -> Any:
        """Run the statements one after another: nests as Python loops (each with its kernel's `reference`),
        views as the JAX code they are."""
        env, fns = self._env(values, view_fns)
        return self._sequential(env, fns, reference=True)

    def _sequential(self, env: dict[str, Any], fns: dict[int, Callable], reference: bool) -> Any:
        """Run the statements one after another, each nest as its own kernel (or its reference)."""
        env = dict(env)
        for item in self.items:
            if isinstance(item, _ViewStmt):
                env[item.target] = self._view(item, fns[id(item)], env, views=not reference)
                continue
            vals = {n: env[n] for n in item.names(env)}
            outs = item.reference(vals) if reference else item(vals)
            env.update(zip(item.outputs, outs if len(item.outputs) > 1 else (outs,), strict=True))
        outs = tuple(_materialise(env[v]) for v in self.outputs)
        return outs if len(outs) > 1 else outs[0]

    def _view(self, item: _ViewStmt, fn: Callable, env: dict[str, Any], views: bool) -> Any:
        """A view statement's value: a `View` of its base where it is one (and `views`), else its array."""
        base, others = env.get(item.base), {r: env[r] for r in item.reads if r != item.base}
        if views and not self.program._eager and _is_array(base) and not any(map(_is_array, others.values())):
            view = make_view(lambda b: fn({**others, item.base: b}), base)
            if view is not None:
                return view
        return fn({r: _materialise(env[r]) for r in item.reads})

    def _plan(self, env: dict[str, Any], fns: dict[int, Callable]) -> _Plan:
        """Plan every nest and view statement (on shapes), then cut the nests into kernels."""
        binding: dict[str, Any] = {n: _Ext(n) for n in env}
        out_specs: dict[tuple[int, str], jax.ShapeDtypeStruct] = {}
        shapes: dict[int, Any] = {}

        def now(v: Any) -> Any:  # a plan value as the nests are planned with: an array or spec, or a View
            if isinstance(v, _Ext):
                return env[v.name]
            if isinstance(v, _Out):
                return out_specs[(v.k, v.var)]
            if isinstance(v, _Viewed):
                return v.view
            if id(v) not in shapes:
                values = {r: now(d) for r, d in v.deps.items()}
                arrays = {r: _spec(x) for r, x in values.items() if _is_array(x)}
                others = {r: x for r, x in values.items() if r not in arrays}
                fn = fns[id(v.item)]
                shapes[id(v)] = jax.eval_shape(lambda a, fn=fn, others=others: fn({**others, **a}), arrays)
            return shapes[id(v)]

        calls, names = [], []
        for item in self.items:
            if isinstance(item, _ViewStmt):
                base = binding.get(item.base)
                others = {r: binding[r] for r in item.reads if r != item.base}
                view = None
                if base is not None and _is_array(now(base)) and not any(_is_array(now(o)) for o in others.values()):
                    consts, fn = {r: now(o) for r, o in others.items()}, fns[id(item)]
                    view = make_view(lambda b, c=consts, f=fn, i=item: f({**c, i.base: b}), now(base))
                if view is not None:
                    binding[item.target] = _Viewed(base.root if isinstance(base, _Viewed) else base, view)
                else:
                    binding[item.target] = _Glue(item, {r: binding[r] for r in item.reads})
                continue
            here = {n: binding[n] for n in item.names(binding)}
            call = item.prepare({n: now(v) for n, v in here.items()})
            k = len(calls)
            calls.append(call)
            names.append(here)
            for o in call.lowered.outputs:
                out_specs[(k, o.var)] = jax.ShapeDtypeStruct(o.out.shape, o.out.dtype)
                binding[o.var] = _Out(k, o.var)
        nests = [self._nest_plan(call, here, out_specs) for call, here in zip(calls, names, strict=True)]
        hw = hardware.target(self.program.options.target)
        segments = fusion.plan_segments(nests, self.program.max_fused, hw)
        return _Plan(calls, names, binding, nests, segments, out_specs)

    @staticmethod
    def _nest_plan(call: NestCall, here: dict[str, Any], out_specs: dict) -> fusion.NestPlan:
        """What cutting the run into kernels needs to know about a nest: which earlier outputs it reads, and
        whether it can read them in the same kernel."""

        def roots(v: Any) -> set[tuple[int, str]]:
            if isinstance(v, _Out):
                return {(v.k, v.var)}
            if isinstance(v, _Viewed):
                return roots(v.root)
            if isinstance(v, _Glue):
                return set().union(*map(roots, v.deps.values()))
            return set()

        lw = call.lowered
        wrapper = fusion.wrapper_reads(lw) if lw.plan is not None else set()
        sources, blocking, nbytes = {}, {}, {}
        for p in call.params:
            v = here[p]
            direct = None
            if p not in wrapper and isinstance(v, _Out):
                direct = (v.k, v.var)
            elif p not in wrapper and isinstance(v, _Viewed) and isinstance(v.root, _Out) and p in call.views:
                direct = (v.root.k, v.root.var)
            if direct is not None:
                sources[p] = direct
            for r in roots(v):
                spec = out_specs[r]
                nbytes[r] = int(np.prod(spec.shape, dtype=np.int64)) * np.dtype(spec.dtype).itemsize
                if r == direct:
                    continue
                if p in wrapper:
                    why = f"a mask or an output allocation reads `{p}`"
                elif isinstance(v, _Viewed):
                    why = f"it can't read the view `{p}` as windows of `{r[1]}`, so XLA copies it first"
                else:
                    why = f"`{p}` is computed from `{r[1]}` by JAX code"
                blocking.setdefault(r[0], why)
        return fusion.NestPlan(call.nest.name, lw, sources, blocking, nbytes)

    def _fuse(
        self, plan: _Plan, seg: fusion.Segment
    ) -> tuple[fusion.FusedLowered, dict[str, tuple[Any, bool]], list[dict[str, tuple]]]:
        """The fused kernel of a segment, what each of its inputs is (plan value, read as the root of a view),
        and where each part's inputs come from (`fusion.Part.sources`)."""
        parts, given = [], {}
        for k in range(seg.start, seg.end):
            call, here = plan.calls[k], plan.names[k]
            sources = {}
            for p in call.params:
                src = plan.nests[k].sources.get(p)
                if src is not None and seg.start <= src[0] < k:
                    sources[p] = ("part", src[0] - seg.start, src[1])
                    continue
                v, remapped = here[p], p in call.views
                if isinstance(v, _Ext):
                    key = v.name
                elif remapped and isinstance(v.root, _Ext):
                    key = v.root.name
                else:
                    key = f"_in{k}_{p}"
                    while key in plan.binding:
                        key += "_"
                sources[p] = ("input", key)
                given[key] = (v, remapped)
            parts.append(fusion.Part(call.nest.name, call.lowered, call.params, sources))
        key = (seg.reason, *((id(p.lowered), p.params, tuple(sorted(p.sources.items()))) for p in parts))
        fused = self.fused.get(key)
        if fused is None:
            program = self.program
            name = "_".join([program.__name__, *(c.nest.outputs[0] for c in plan.calls[seg.start : seg.end]), "fused"])
            hw = hardware.target(program.options.target)
            cut = seg.reason if seg.end < len(self.nests) else ""
            fused = fusion.fuse(name, parts, program.options.matmul_precision, hw, cut=cut)
            self.fused[key] = fused
        return fused, given, [p.sources for p in parts]

    def _run(
        self,
        fused: fusion.FusedLowered,
        plan: _Plan,
        seg: fusion.Segment,
        args: list[Any],
        sources: list[dict[str, tuple]],
        consts: list[dict[str, Any]],
    ) -> Any:
        """Run a fused kernel (with derivatives: those of its nests' Python loops)."""
        calls = plan.calls[seg.start : seg.end]
        fn = self._compiled.get(id(fused))
        if fn is None:
            interpret = calls[0].kernel._interpret_mode()
            if fused.cores > 1 and dataclasses.is_dataclass(interpret):  # the parts meet at barriers: simulate
                interpret = dataclasses.replace(interpret, num_cores_or_threads=fused.cores)  # every core
            entry = functools.partial(fused.entry, interpret=interpret)
            fn = self._compiled[id(fused)] = jax.jit(lambda *a: tuple(entry(*a)))

        def reference(*arrays: Any, kernels: bool = False) -> tuple[Any, ...]:
            # The fused nests as Python loops, one after another: what `jax.grad` differentiates (with
            # `kernels`, each nest as its own kernel: how a batch runs under `jax.vmap`, or sharded inputs).
            inputs = dict(zip(fused.inputs, arrays, strict=True))
            outs: dict[tuple[int, str], Any] = {}
            for j, call in enumerate(calls):
                nest, here = call.nest, plan.names[seg.start + j]
                vals = dict(consts[j])
                for p in call.params:
                    src = sources[j][p]
                    x = outs[(src[1], src[2])] if src[0] == "part" else inputs[src[1]]
                    vals[p] = here[p].view.on(x).materialise() if p in call.views else x
                res = (nest if kernels else nest.reference)(vals)
                res = res if len(nest.outputs) > 1 else (res,)
                outs.update(zip([(j, v) for v in nest.outputs], res, strict=True))
            return tuple(outs[(j, var)] for j, var in fused.outputs)

        unfused = functools.partial(reference, kernels=True)
        return autodiff.differentiable(fn, reference, None, fallback=unfused)(*args)


def _keywords(fn: Callable, names: tuple[str, ...]) -> Callable:
    """`fn` taking its arguments as one tuple: positional values, then the values of keywords `names`."""
    if not names:
        return fn

    def call(*flat: Any) -> Any:
        k = len(flat) - len(names)
        return fn(*flat[:k], **dict(zip(names, flat[k:], strict=True)))

    return call


def _python_loop(bounds: tuple[int, ...], body: Callable, carry: tuple) -> tuple:
    """Run a Python loop around nests as the plain loop it is (for the reference)."""
    for i in range(*bounds):
        carry = body(i, carry)
    return carry


class TpuProgram:
    """A Python function made of tile loop nests and ordinary JAX code; each nest runs as a Pallas kernel.

    Calling it runs the function jit-compiled, with every nest as a generated gridless kernel and the code
    between them compiled by XLA. Consecutive nests (nothing between them but views: reshapes, transposes
    and integer indexing, which the nests read in place) are fused into one kernel
    (`fuse=True`: up to `fusion.MAX_PARTS` nests per kernel; an int: up to that many); a Python `for`
    loop around nests runs as a `jax.lax.fori_loop` where its body allows it
    (`unroll=False`), so a deep model compiles each kernel once. Other entry points, like `TpuKernel`'s:

    - `lower(*args)`: a `ProgramLowered`: (name, `Lowered` or `FusedLowered`) for every kernel it runs
    - `source(*args)`: the generated Pallas source of every kernel
    - `reference(*args)`: run the original Python, the nests as Python loops (with the kernels' `@`)
    - `verify(*args)`: run both and raise if they disagree, naming the first nest whose kernel disagrees
      with its Python loops; returns the max abs error (`per_nest=True`: check every nest first)
    - `verify_nests(*args)`: run each nest's kernel on the inputs the Python gives that nest, against its
      Python loops; raises naming the first nest that disagrees, else returns each nest's max abs error
    - `tuning(*args)`: each nest's tuning candidates
    - `kernels`: the program's nests (`ProgramNest`), with the kernels built for each so far

    Arguments are arrays (or pytrees of them), and for `lower`, `source` and `tuning` also
    `jax.ShapeDtypeStruct`s; arguments holding no arrays are static: Python values the code branches on.
    Every entry point takes them as the function does: positionally or by keyword, with its defaults
    (`rtol`, `atol` and `per_nest` are `verify`'s own keywords). With `backward`, it gets the arguments
    bound to the signature in order: positional ones, then keyword-only ones.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        interpret: Any = None,
        verify: bool = False,
        options: Options | None = None,
        fdef: ast.FunctionDef | None = None,
        fuse: bool | int = True,
        unroll: bool = False,
        backward: Callable | None = None,
    ):
        self.fn = fn
        self.backward = backward
        self.interpret = interpret
        self.verify_on_first_call = verify
        self.options = options or Options()
        self.fuse, self.unroll = bool(fuse), unroll
        # `fuse=True`: up to fusion.MAX_PARTS nests per kernel; an int: up to that many.
        self.max_fused = fusion.MAX_PARTS if fuse is True else max(1, int(fuse))
        self.filename = inspect.getsourcefile(fn) or "<autokernel>"
        self.first_line = fn.__code__.co_firstlineno
        fdef = fdef if fdef is not None else frontend.function_ast(fn)
        self._program_ast, regions, groups = _program_ast(fdef, fn.__name__, self.first_line)
        if not regions:
            raise AutoKernelError(
                f"{fn.__name__}: found no tile loop nest (`for v in range(...)` loops ending in stores of tiles "
                "placed by their variables, `out = out.at[v : v + T, ...].set(...)`)"
            )
        self.nests = [ProgramNest(self, r) for r in regions]
        self.groups = [
            ProgramGroup(self, [self.nests[k] if isinstance(k, int) else k for k in items], ins, outs)
            for items, ins, outs in groups
        ]
        self._program: Callable | None = None
        self._reference: Callable | None = None
        self._jitted: dict[tuple[int, ...], Callable] = {}
        self._measured: set[tuple] = set()
        self._verified: set[tuple] = set()
        self._recording: list | None = None
        self._eager = False  # running eagerly (to measure kernels): no fusion, no rolled loops
        self.unrolled: list[str] = []  # why loops around nests were unrolled rather than rolled, most recent last
        self._signature = inspect.signature(fn)
        self._outer_names = list(dict.fromkeys(g for nest in self.nests for g in nest.globals))
        self._outer: dict[str, Any] | None = None  # while tracing: the global arrays nests read, as traced values
        self._nest_log: list | None = None  # for `verify_nests`: (nest, values, reference output) per nest call
        functools.update_wrapper(self, fn)

    @property
    def kernels(self) -> list[ProgramNest]:
        """The program's tile loop nests, in source order; `nest.kernels` holds the kernels built so far."""
        return list(self.nests)

    def _program_fn(self) -> Callable:
        if self._program is None:
            extra = {_NESTS: self.nests, _GROUPS: self.groups, _LOOP: self._loop}
            self._program = _compile(self._program_ast, self.fn, self.filename, self.first_line, extra)
        return self._program

    def _loop(self, bounds: tuple[int, ...], body: Callable, carry: tuple) -> tuple:
        """Run a Python loop around nests: as `jax.lax.fori_loop` (each kernel traced once, the loop variable a
        runtime scalar), or, when unrolling is asked for or the body needs the loop variable's value at trace
        time, as the Python loop it is."""
        r = range(*bounds)
        if not self.unroll and not self._eager and len(r) > 1:
            recording = self._recording
            mark = len(recording) if recording is not None else 0
            try:
                return jax.lax.fori_loop(0, len(r), lambda n, c: tuple(body(r.start + n * r.step, c)), tuple(carry))
            except Exception as e:  # the body needs Python values: unroll it
                if recording is not None:
                    del recording[mark:]
                self.unrolled.append(f"{type(e).__name__}: {e}")
        for i in r:
            carry = body(i, carry)
        return carry

    def _reference_fn(self) -> Callable:
        if self._reference is None:  # the program, with each nest run by its kernel's `reference`
            extra = {
                _NESTS: [nest.reference for nest in self.nests],
                _GROUPS: [group.reference for group in self.groups],
                _LOOP: _python_loop,
            }
            self._reference = _compile(self._program_ast, self.fn, self.filename, self.first_line, extra)
        return self._reference

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        """The arguments bound to the function's signature with its defaults filled in, as one tuple: the
        positional values, then the keyword-only ones (and `**kwargs`), whose names it also returns."""
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{self.__name__}(): {e}") from None
        bound.apply_defaults()
        return (*bound.args, *bound.kwargs.values()), tuple(bound.kwargs)

    def _outer_value(self, name: str) -> Any:
        """The current value of a global or closure variable of the function (None if unbound)."""
        code = self.fn.__code__
        if name in code.co_freevars:
            try:
                return self.fn.__closure__[code.co_freevars.index(name)].cell_contents
            except ValueError:  # an empty cell
                return None
        return self.fn.__globals__.get(name)

    def _outer_arrays(self, names: list[str]) -> dict[str, Any]:
        """The global and closure arrays among `names` (not scalars): their traced values while the program is
        traced, else their current values."""
        out = {}
        for name in names:
            if self._outer is not None and name in self._outer:
                out[name] = self._outer[name]
                continue
            value = self._outer_value(name)
            if _is_array(value) and len(value.shape) >= 1:
                out[name] = value
        return out

    def _with_outer(self, fn: Callable) -> Callable:
        """`fn(*args)` as `run(outer, *args)`: `outer` holds the global arrays the nests read, passed as
        arguments so that each call sees their current values (as a traced function would not)."""

        def run(outer: dict[str, Any], *args: Any) -> Any:
            saved, self._outer = self._outer, outer
            try:
                return fn(*args)
            finally:
                self._outer = saved

        return run

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        flat, names = self._bind(args, kwargs)
        key = (names, tuple(map(_arg_key, flat)))
        concrete = _concrete(flat)
        measure = self.options.tune and self.options.autotune == "measure" and hardware.on_tpu()
        if measure and concrete and key not in self._measured:
            # Timing kernels needs real arrays: run this first call eagerly, so each kernel measures its
            # candidates on its actual inputs; later calls (re)trace with the winners.
            self._measured.add(key)
            self._eager = True
            try:
                out = self._program_fn()(*args, **kwargs)
            finally:
                self._eager = False
            self._jitted.clear()
        else:
            static = tuple(n for n, a in enumerate(flat) if _is_static(a))
            jitted = self._jitted.get((names, len(flat), static))
            if jitted is None:
                fn = _keywords(self._program_fn(), names)
                if self.backward is not None:  # otherwise jax.grad goes through each kernel's own VJP
                    if static:
                        raise AutoKernelError(f"{self.__name__}: `backward` needs every argument to be an array")
                    reference = _keywords(self._reference_fn(), names)
                    fn = autodiff.differentiable(fn, reference, self.backward)
                run = self._with_outer(fn)
                jitted = self._jitted[(names, len(flat), static)] = jax.jit(run, static_argnums=[1 + n for n in static])
            # With `backward`, global arrays are read when tracing (its VJP takes only the arguments).
            outer = self._outer_arrays(self._outer_names) if self.backward is None else {}
            out = jitted(outer, *flat)
        if self.verify_on_first_call and concrete and key not in self._verified:
            self._verified.add(key)
            self._check(out, self.reference(*args, **kwargs), args, kwargs)
        return out

    def reference(self, *args: Any, **kwargs: Any) -> Any:
        """Run the original function eagerly, nests as Python loops, with the kernels' `@` semantics in the
        nests (MXU dots accumulating in f32 / i32) and their tuned constants; code outside the nests runs
        exactly as written."""
        return self._reference_fn()(*args, **kwargs)

    def verify(
        self, *args: Any, rtol: float | None = None, atol: float | None = None, per_nest: bool = False, **kwargs: Any
    ) -> float:
        """Run the program and its `reference` and raise AssertionError if their outputs disagree; returns the
        max abs error. The error names the first nest whose kernel disagrees with its Python loops on the
        inputs the program gives it (see `verify_nests`). With `per_nest`, check every nest that way first."""
        if per_nest:
            self.verify_nests(*args, rtol=rtol, atol=atol, **kwargs)
        got, want = self(*args, **kwargs), self.reference(*args, **kwargs)
        return self._check(got, want, args, kwargs, rtol, atol)

    def verify_nests(
        self, *args: Any, rtol: float | None = None, atol: float | None = None, **kwargs: Any
    ) -> dict[str, float]:
        """Check each nest on its own: run the program's `reference` and, for every nest it runs, the nest's
        kernel on the very inputs the Python gave that nest, comparing the two outputs. Raises AssertionError
        naming the first nest (and its line) whose kernel disagrees; returns each nest's max abs error."""
        log: list[tuple[ProgramNest, dict[str, Any], Any]] = []
        self._nest_log = log
        try:
            self.reference(*args, **kwargs)
        finally:
            self._nest_log = None
        total = {nest: sum(1 for n, _, _ in log if n is nest) for nest, _, _ in log}
        seen: dict[ProgramNest, int] = {}
        errors: dict[str, float] = {}
        for nest, values, want in log:
            seen[nest] = seen.get(nest, 0) + 1
            where = f"nest `{nest.name}` (line {nest.line})"
            if total[nest] > 1:
                where += f", run {seen[nest]} of {total[nest]}"
            try:
                err = check_outputs(nest.name, nest(values), want, self.options.matmul_precision, rtol, atol)
            except AssertionError as e:
                raise AssertionError(
                    f"{self.__name__}: {where} disagrees with its Python loops on the inputs the program gives it: {e}"
                ) from None
            errors[nest.name] = max(errors.get(nest.name, 0.0), err)
        return errors

    def _check(
        self, got: Any, want: Any, args: tuple, kwargs: dict, rtol: float | None = None, atol: float | None = None
    ) -> float:
        """`check_outputs`, naming the first disagreeing nest when the outputs disagree."""
        try:
            return check_outputs(self.__name__, got, want, self.options.matmul_precision, rtol, atol)
        except AssertionError as e:
            try:
                self.verify_nests(*args, rtol=rtol, atol=atol, **kwargs)
                why = (
                    "every nest's kernel agrees with its Python loops on the inputs the program gives it, so the "
                    "difference builds up across nests"
                )
            except AssertionError as nest_error:
                why = f"the first nest that disagrees: {nest_error}"
            except Exception as other:  # the diagnosis is a courtesy: don't hide the original error
                why = f"(couldn't check the nests one by one: {type(other).__name__}: {other})"
            raise AssertionError(f"{e}\n{why}") from None

    def _trace(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[tuple]:
        """Trace the program abstractly; what it runs, in order: ("nest", NestCall, specs) and
        ("fused", FusedLowered, specs, [NestCall])."""
        flat, names = self._bind(args, kwargs)
        fn = _keywords(self._program_fn(), names)
        dynamic = [n for n, a in enumerate(flat) if not _is_static(a)]

        def run(*values: Any) -> Any:
            full = list(flat)
            for n, v in zip(dynamic, values, strict=True):
                full[n] = v
            return fn(*full)

        self._recording = []
        try:
            outer = self._outer_arrays(self._outer_names)
            jax.eval_shape(self._with_outer(run), outer, *[flat[n] for n in dynamic])
            return self._recording
        finally:
            self._recording = None

    def lower(self, *args: Any, **kwargs: Any) -> ProgramLowered:
        calls = self._trace(args, kwargs)
        cuts = tuple(dict.fromkeys(c[1] for c in calls if c[0] == "cut"))
        calls = [c for c in calls if c[0] != "cut"]
        keys: dict[ProgramNest, list[tuple]] = {}
        for c in calls:
            for call in [c[1]] if c[0] == "nest" else c[3]:
                keys.setdefault(call.nest, [])
                if call.key not in keys[call.nest]:
                    keys[call.nest].append(call.key)
        kernels: dict[tuple, tuple[str, Kernel, tuple]] = {}
        for c in calls:
            specs = c[2]
            if c[0] == "nest":
                call = c[1]
                key = (id(call.lowered), tuple((s.shape, s.dtype) for s in specs))
                kernels.setdefault(key, (_kernel_name(call.nest, call.key, keys[call.nest]), call.lowered, specs))
            else:
                fused, segment = c[1], c[3]
                names = [_kernel_name(call.nest, call.key, keys[call.nest]) for call in segment]
                if [n for n, _ in fused.parts] != names:  # nests with several kernels: name them apart
                    parts = tuple(zip(names, (lw for _, lw in fused.parts), strict=True))
                    fused = dataclasses.replace(fused, parts=parts)
                kernels.setdefault(
                    (id(c[1]), tuple((s.shape, s.dtype) for s in specs)), ("+".join(names), fused, specs)
                )
        named = _unique_names(list(kernels.values()))
        return ProgramLowered(
            tuple((name, k) for name, k, _ in named), tuple(specs for _, _, specs in named), cuts=cuts
        )

    def source(self, *args: Any, **kwargs: Any) -> str:
        return self.lower(*args, **kwargs).source

    def tuning(self, *args: Any, **kwargs: Any) -> dict[str, list[tune_lib.Candidate]]:
        """Each nest kernel's tuning candidates (empty for kernels with nothing to tune), by kernel name."""
        calls = [c for c in self._trace(args, kwargs) if c[0] != "cut"]
        nest_calls = []
        for c in calls:
            if c[0] == "nest":
                nest_calls.append((c[1], c[2]))
            else:
                for call in c[3]:
                    nest_calls.append((call, tuple(_spec(s) for s in call.lowered.plan.spec.specs)))
        keys: dict[ProgramNest, list[tuple]] = {}
        for call, _ in nest_calls:
            keys.setdefault(call.nest, [])
            if call.key not in keys[call.nest]:
                keys[call.nest].append(call.key)
        out = {}
        for call, specs in nest_calls:
            out.setdefault(_kernel_name(call.nest, call.key, keys[call.nest]), call.kernel.tuning(*specs))
        return out


def _unique_names(kernels: list[tuple[str, Any, tuple]]) -> list[tuple[str, Any, tuple]]:
    """Number kernels that would otherwise share a name (the same nests at different shapes)."""
    counts: dict[str, int] = {}
    for name, _, _ in kernels:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for name, k, specs in kernels:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name}#{seen[name]}"
        out.append((name, k, specs))
    return out


def _kernel_name(nest: ProgramNest, key: tuple, keys: list[tuple]) -> str:
    """The nest's name, plus the constants that tell its kernels apart when it has several (`[l=1]`)."""
    if len(keys) == 1:
        return nest.name
    consts = dict((n, v) for n, _, v in key[2])
    differ = [n for n in consts if any(dict((m, w) for m, _, w in k[2]).get(n, object()) != consts[n] for k in keys)]
    if not differ:
        return f"{nest.name}[{keys.index(key)}]"
    return f"{nest.name}[{', '.join(f'{n}={consts[n]!r}' for n in differ)}]"
