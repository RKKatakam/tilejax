"""Views: arrays that are reshapes, transposes and integer or full indexing of another array.

Between the nests of a program, layout glue like

    qkv = qkv.reshape(B, S, 3, H, DH).transpose(2, 0, 3, 1, 4)

would make XLA copy `qkv` in HBM just so the next nest can read `qkv[0, b, h, i : i + T, :]`. But every
element of such a view is an element of its base, and a tile of the view is often a rectangular window of
the base: here rows `b * S + i : b * S + i + T` and columns `h * DH : h * DH + DH` of the original
`(B * S, 3 * D)` array. A nest reading a view through tiles like that gets the base array as its input and
copies those windows straight out of it: no copy of the whole array, and nothing between the nests.

A view is described by how it splits each dim of its base into *factors* (mixed-radix digits, row-major):
`(B * S, 3 * D)` reshaped to `(B, S, 3, H, DH)` splits dim 0 into factors (B, S) and dim 1 into (3, H, DH).
Each dim of the view is a run of factors (several when a reshape merges dims, like `(B, S)` back into
`B * S` rows); integer indexing fixes factors to a value. `ViewMap.trace` derives this from the jaxpr of the
glue code, so whatever spelling the code uses (`.T`, `jnp.swapaxes`, `x[0]`, ...) means what JAX says it
means; anything that isn't a pure relabelling of elements is not a view.

`ViewMap.window` maps one tile of the view to a window of the base, or says why it can't: a tile that
straddles a factor boundary (128 rows of a view whose rows are `(B, S)` with S = 64), a tile whose dims come
out in another order in the base (a transpose of the minor dims), or a window that breaks TPU layout rules.
The program then materialises the view (an XLA copy, as before) for that nest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from .errors import AutoKernelError

__all__ = ["View", "ViewError", "ViewMap", "Window", "div_src", "make_view", "rem_src"]


class ViewError(AutoKernelError):
    """A nest can't read input `param` (a view) as windows of its base array."""

    def __init__(self, param: str, msg: str):
        super().__init__(msg)
        self.param = param


# --------------------------------------------------------------------------- the structure of a view


@dataclass(frozen=True)
class ViewMap:
    """How a view's elements sit in its base array.

    `factors[d]` lists base dim d's factors, outermost first, as (size, fixed): `fixed` is the index an
    integer index of the view pins it to, or None. `dims[v]` lists the factors view dim v is made of,
    outermost first, as (base dim, position in `factors[base dim]`). Size-1 factors are always fixed (to 0).
    """

    base_shape: tuple[int, ...]
    dtype: np.dtype
    factors: tuple[tuple[tuple[int, int | None], ...], ...]
    dims: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(math.prod(self.factors[d][k][0] for d, k in dim) for dim in self.dims)

    @property
    def key(self) -> tuple:
        return (self.base_shape, self.dtype.name, self.factors, self.dims)

    @property
    def identity(self) -> bool:
        """Is the view the base itself?"""
        return all(len(f) == 1 and f[0][1] is None for f in self.factors) and self.dims == tuple(
            ((d, 0),) for d in range(len(self.base_shape))
        )

    @staticmethod
    def trace(fn: Callable[[Any], Any], base: Any) -> ViewMap | None:
        """The view `fn(base)` is of `base` (an array or its shape and dtype), or None if `fn` does anything
        but relabel elements: reshapes, transposes, squeezes and slices that are whole dims or single
        indices."""
        spec = jax.ShapeDtypeStruct(tuple(base.shape), base.dtype)
        try:
            closed = jax.make_jaxpr(fn)(spec)
        except Exception:
            return None
        jaxpr = closed.jaxpr
        if len(jaxpr.invars) != 1 or len(jaxpr.outvars) != 1:
            return None
        builder = _Builder(tuple(spec.shape))
        current = jaxpr.invars[0]
        for eqn in jaxpr.eqns:
            if len(eqn.invars) != 1 or eqn.invars[0] is not current or len(eqn.outvars) != 1:
                return None
            if not builder.apply(eqn):
                return None
            current = eqn.outvars[0]
        if jaxpr.outvars[0] is not current:
            return None
        out = jaxpr.outvars[0].aval
        if np.dtype(out.dtype) != np.dtype(spec.dtype):  # pragma: no cover - every step it follows keeps the dtype
            return None
        vmap = builder.result(np.dtype(spec.dtype))
        if vmap.shape != tuple(out.shape):  # pragma: no cover - the builder disagrees with JAX: don't guess
            return None
        return vmap

    def window(self, param: str, block: Any, text: str, lanes: int, check: Callable) -> Window:
        """The base window holding the view's tile `block` (an `analysis.Block` of the view); ViewError if
        there is none. `check(what, block_shape, full_shape)` returns the layout problems of a base window."""
        where = f"`{param}[{text}]`"
        values: list[list[Any]] = [[None] * len(f) for f in self.factors]  # per factor: _Value
        owner: dict[tuple[int, int], int] = {}  # factor -> the kept view dim whose range covers it
        for d, f in enumerate(self.factors):
            for k, (_, fixed) in enumerate(f):
                if fixed is not None:
                    values[d][k] = _Value((), fixed, 1)
        for v, dim in enumerate(self.dims):
            size = math.prod(self.factors[d][k][0] for d, k in dim)
            t = 1 if block.squeezed[v] else block.shape[v]
            if not block.squeezed[v] and t == 1 and size != 1:
                raise ViewError(param, f"{where}: a size-1 tile of view dim {v} isn't supported for views")
            sizes = [self.factors[d][k][0] for d, k in dim]
            suffix = [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]
            j = next((i for i in range(len(sizes)) if t % suffix[i] == 0 and (sizes[i] * suffix[i]) % t == 0), None)
            if dim and j is None:
                raise ViewError(
                    param,
                    f"{where}: a tile of {t} along view dim {v} (made of base factors {sizes}) doesn't fall in "
                    "whole blocks of the base array",
                )
            for i, (d, k) in enumerate(dim):
                if i > j:
                    values[d][k] = _Value((), 0, sizes[i])  # whole
                    owner[(d, k)] = v
                elif i == j:
                    length = t // suffix[i]
                    values[d][k] = _Value(((v, suffix[i], sizes[i] if i else None),), 0, length)
                    if length > 1:
                        owner[(d, k)] = v
                else:
                    values[d][k] = _Value(((v, suffix[i], sizes[i] if i else None),), 0, 1)
        out: list[BaseDim] = []
        kept_by: list[int | None] = []
        for d, f in enumerate(self.factors):
            sizes = [s for s, _ in f]
            suffix = [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]
            vals = values[d]
            # A window of one base dim: single indices of its outer factors, then a range of one factor,
            # then whole inner factors.
            k = next((i for i in range(len(f)) if vals[i].length > 1), None)
            if k is None:
                length, upto = 1, len(f)
            else:
                if any(vals[i].length != sizes[i] or not vals[i].whole for i in range(k + 1, len(f))):
                    raise ViewError(param, f"{where}: the tile isn't one rectangular window of base dim {d}")
                length, upto = vals[k].length * suffix[k], k + 1
            terms, const = [], 0
            for i in range(upto):
                for vd, div, mod in vals[i].terms:
                    terms.append(_Term(vd, div, mod, suffix[i], i == k))
                const += vals[i].const * suffix[i]
            owners = {owner[(d, i)] for i in range(len(f)) if (d, i) in owner}
            if len(owners) > 1:
                raise ViewError(param, f"{where}: two dims of the tile fall in base dim {d}")
            v = owners.pop() if owners else None
            if v is not None:
                covered = [(d, i) for i in range(len(f)) if (d, i) in owner]
                ranged = [(dd, kk) for dd, kk in self.dims[v] if (dd, kk) in owner]
                if k is None or covered != ranged or covered != [(d, i) for i in range(k, len(f)) if sizes[i] > 1]:
                    raise ViewError(param, f"{where}: the tile's elements aren't in the base's order along dim {d}")
                if length != (1 if block.squeezed[v] else block.shape[v]):  # pragma: no cover - caught above
                    raise ViewError(param, f"{where}: view dim {v} doesn't map onto one dim of the base")
            elif length != 1:  # pragma: no cover - a factor ranged over always has an owning view dim
                raise ViewError(param, f"{where}: base dim {d} would need a window of {length}")
            out.append(BaseDim(tuple(terms), const, length, v is None))
            kept_by.append(v)
        kept = [v for v in kept_by if v is not None]
        if kept != sorted(kept) or len(set(kept)) != len(kept):
            raise ViewError(
                param, f"{where}: the tile's dims come out in another order in the base array (a transpose)"
            )
        if len(kept) != sum(not s for s in block.squeezed):
            raise ViewError(param, f"{where}: the tile doesn't map onto a window of the base array")
        window = Window(tuple(out), self.base_shape)
        # TPU layout: the window's last two kept dims, and their offsets, must be tile-aligned.
        kept_shape = tuple(b.size for b in out if not b.squeezed)
        kept_full = tuple(n for b, n in zip(out, self.base_shape, strict=True) if not b.squeezed)
        problems = list(check(f"{where} (a view) as a window {kept_shape} of its base", kept_shape, kept_full))
        if out and out[-1].squeezed and self.base_shape[-1] > 1:
            problems.append(f"{where}: the tile picks single elements of the base's last dim")
        sub = 8 * max(1, 4 // self.dtype.itemsize)
        for pos, align in ((-1, lanes), (-2, sub)):
            if len(out) < -pos:
                continue
            b, n = out[pos], self.base_shape[pos]
            if b.squeezed or b.size == n:  # a whole dim starts at 0
                continue
            # A range's start is a multiple of its length (the window's size); single indices are anything.
            if b.const % align or any(t.coef % align for t in b.terms if not t.ranged):
                at = len(out) + pos
                problems.append(f"{where}: the window's offset in base dim {at} isn't a multiple of {align}")
        if problems:
            raise ViewError(param, "; ".join(problems))
        return window


@dataclass(frozen=True)
class _Value:
    """A factor's index in one tile: `const` plus the sum of `terms` (view dim, div, mod), for `length`."""

    terms: tuple[tuple[int, int, int | None], ...]
    const: int
    length: int

    @property
    def whole(self) -> bool:
        return not self.terms and self.const == 0


@dataclass(frozen=True)
class _Term:
    """`coef * ((start of view dim vd) // div % mod)` (no `% mod` when mod is None). `ranged`: the factor
    the window ranges over, whose index is a multiple of the window's length."""

    vd: int
    div: int
    mod: int | None
    coef: int
    ranged: bool = False


@dataclass(frozen=True)
class BaseDim:
    """One dim of a base window: start = const + sum of terms, `size` long, or a single index (squeezed)."""

    terms: tuple[_Term, ...]
    const: int
    size: int
    squeezed: bool


@dataclass(frozen=True)
class Window:
    """The window of the base array a tile of a view reads, as a function of the tile's start in each view
    dim."""

    dims: tuple[BaseDim, ...]
    base_shape: tuple[int, ...]

    def render(self, starts: Sequence[tuple[str | None, int, int]], div: str = "_div", rem: str = "_rem") -> list[str]:
        """Per base dim, the index (`pl.ds(start, size)`, or an integer for a squeezed dim), given each view
        dim's start as `atom * mult + const` (atom None: a constant). `div` and `rem` name the generated
        code's integer division and remainder helpers."""
        out = []
        for b in self.dims:
            expr = _Affine(None, 0, b.const)
            for t in b.terms:
                atom, mult, const = starts[t.vd]
                x = _Affine(atom, mult, const).floordiv(t.div, div)
                if t.mod is not None:
                    x = x.mod(t.mod, rem)
                expr = expr.add(x.scale(t.coef))
            start = expr.render()
            out.append(start if b.squeezed else f"pl.ds({start}, {b.size})")
        return out


@dataclass(frozen=True)
class _Affine:
    """`sum(coef * atom) + const` over source strings (atoms), for index arithmetic in generated code."""

    atom: str | None
    mult: int
    const: int
    rest: tuple[tuple[str, int], ...] = ()  # more (atom, coef) terms

    def terms(self) -> list[tuple[str, int]]:
        return ([(self.atom, self.mult)] if self.atom is not None and self.mult else []) + list(self.rest)

    def scale(self, c: int) -> _Affine:
        terms = [(a, m * c) for a, m in self.terms()]
        return _Affine.of(terms, self.const * c)

    def add(self, other: _Affine) -> _Affine:
        return _Affine.of(self.terms() + other.terms(), self.const + other.const)

    @staticmethod
    def of(terms: list[tuple[str, int]], const: int) -> _Affine:
        merged: dict[str, int] = {}
        for a, m in terms:
            merged[a] = merged.get(a, 0) + m
        items = [(a, m) for a, m in merged.items() if m]
        if not items:
            return _Affine(None, 0, const)
        return _Affine(items[0][0], items[0][1], const, tuple(items[1:]))

    def floordiv(self, n: int, div: str) -> _Affine:
        if n == 1:
            return self
        terms = self.terms()
        if all(m % n == 0 for _, m in terms) and self.const % n == 0:
            return _Affine.of([(a, m // n) for a, m in terms], self.const // n)
        if len(terms) == 1 and self.const == 0 and n % terms[0][1] == 0:  # (a * m) // n == a // (n / m)
            a, m = terms[0]
            return _Affine(div_src(a, n // m, div), 1, 0)
        return _Affine(div_src(self.render(), n, div), 1, 0)

    def mod(self, n: int, rem: str) -> _Affine:
        terms = [(a, m % n) for a, m in self.terms()]
        terms = [(a, m) for a, m in terms if m]
        if not terms:
            return _Affine(None, 0, self.const % n)
        if len(terms) == 1 and self.const % n == 0 and n % terms[0][1] == 0:  # (a * m) % n == (a % (n / m)) * m
            a, m = terms[0]
            return _Affine(rem_src(a, n // m, rem), m, 0)
        return _Affine(rem_src(_Affine.of(terms, self.const % n).render(), n, rem), 1, 0)

    def render(self) -> str:
        parts = [a if m == 1 else f"{a} * {m}" for a, m in self.terms()]
        if self.const or not parts:
            parts.append(str(self.const))
        return " + ".join(parts) if len(parts) == 1 else "(" + " + ".join(parts) + ")"


def div_src(x: str, n: int, div: str = "_div") -> str:
    """`x // n` for a non-negative x, as source: a shift for powers of two, else one integer division by the
    generated code's helper `div`."""
    if n == 1:
        return x
    return f"({x} >> {n.bit_length() - 1})" if n & (n - 1) == 0 else f"{div}({x}, {n})"


def rem_src(x: str, n: int, rem: str = "_rem") -> str:
    """`x % n` for a non-negative x, as source: a mask for powers of two, else the helper `rem`."""
    return f"({x} & {n - 1})" if n & (n - 1) == 0 else f"{rem}({x}, {n})"


class _Builder:
    """Follows reshapes, transposes and index slices from a base array, tracking each view dim's factors."""

    def __init__(self, shape: tuple[int, ...]):
        self.size: dict[int, int] = {}
        self.fixed: dict[int, int] = {}
        self.order: list[list[int]] = []  # per base dim: its factors, outermost first
        self.dims: list[list[int]] = []  # per view dim: its factors, outermost first
        self.next = 0
        for n in shape:
            f = self.new(n)
            self.order.append([f])
            self.dims.append([] if f in self.fixed else [f])

    def new(self, n: int) -> int:
        f = self.next
        self.next += 1
        self.size[f] = n
        if n == 1:
            self.fixed[f] = 0
        return f

    def split(self, f: int, outer: int) -> tuple[int, int]:
        """Split factor f into (outer, size // outer), in place in its base dim."""
        a, b = self.new(outer), self.new(self.size[f] // outer)
        for order in self.order:
            if f in order:
                k = order.index(f)
                order[k : k + 1] = [a, b]
        return a, b

    def apply(self, eqn: Any) -> bool:
        name, p = eqn.primitive.name, eqn.params
        if name in ("copy", "copy_p"):
            return True
        if name == "convert_element_type":
            return np.dtype(p["new_dtype"]) == np.dtype(eqn.invars[0].aval.dtype)
        if name == "transpose":
            self.dims = [self.dims[k] for k in p["permutation"]]
            return True
        if name == "squeeze":
            drop = set(p["dimensions"])
            if any(self.dims[k] for k in drop):  # pragma: no cover - JAX squeezes size-1 dims, which have no factors
                return False
            self.dims = [d for k, d in enumerate(self.dims) if k not in drop]
            return True
        if name == "reshape":
            if p.get("dimensions") is not None:
                return False
            return self.reshape(tuple(p["new_sizes"]))
        if name == "slice":
            if p.get("strides") is not None and any(s != 1 for s in p["strides"]):
                return False
            shape = eqn.invars[0].aval.shape
            new = []
            for dim, lo, hi, n in zip(self.dims, p["start_indices"], p["limit_indices"], shape, strict=True):
                if (lo, hi) == (0, n):
                    new.append(dim)
                elif hi - lo == 1:
                    v = lo
                    for f in reversed(dim):
                        self.fixed[f] = v % self.size[f]
                        v //= self.size[f]
                    new.append([])
                else:
                    return False
            self.dims = new
            return True
        if name == "broadcast_in_dim":  # only adding size-1 dims
            shape, keep = tuple(p["shape"]), tuple(p["broadcast_dimensions"])
            if list(keep) != sorted(keep) or len(keep) != len(self.dims):
                return False
            if any(shape[k] != math.prod(self.size[f] for f in d) for k, d in zip(keep, self.dims, strict=True)):
                return False
            if any(n != 1 for k, n in enumerate(shape) if k not in keep):
                return False
            it = iter(self.dims)
            self.dims = [next(it) if k in keep else [] for k in range(len(shape))]
            return True
        return False

    def reshape(self, sizes: tuple[int, ...]) -> bool:
        queue = [f for d in self.dims for f in d]
        out: list[list[int]] = []
        for t in sizes:
            dim: list[int] = []
            need = t
            while need > 1:
                if not queue:  # pragma: no cover - JAX checked the sizes' product
                    return False
                f = queue[0]
                s = self.size[f]
                if s <= need:
                    if need % s:
                        return False
                    dim.append(f)
                    queue.pop(0)
                    need //= s
                else:
                    if s % need:
                        return False
                    a, b = self.split(f, need)
                    dim.append(a)
                    queue[0] = b
                    need = 1
            out.append(dim)
        if queue:  # pragma: no cover - JAX checked the sizes' product
            return False
        self.dims = out
        return True

    def result(self, dtype: np.dtype) -> ViewMap:
        position = {f: (d, k) for d, order in enumerate(self.order) for k, f in enumerate(order)}
        factors = tuple(tuple((self.size[f], self.fixed.get(f)) for f in order) for order in self.order)
        dims = tuple(tuple(position[f] for f in dim) for dim in self.dims)
        return ViewMap(tuple(math.prod(self.size[f] for f in o) for o in self.order), dtype, factors, dims)


# --------------------------------------------------------------------------- views at run time


class View:
    """A view of `root` (an array, a traced value or a ShapeDtypeStruct): the array `fn(root)`, not yet made.

    A nest that can read it as windows of `root` is given `root`; anything else gets `materialise()`, the
    array itself (an XLA copy).
    """

    def __init__(self, root: Any, fn: Callable[[Any], Any], vmap: ViewMap):
        self.root, self.fn, self.map = root, fn, vmap

    @property
    def shape(self) -> tuple[int, ...]:
        return self.map.shape

    @property
    def dtype(self) -> np.dtype:
        return self.map.dtype

    @property
    def ndim(self) -> int:
        return len(self.map.shape)

    @property
    def key(self) -> tuple:
        return self.map.key

    def spec(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)

    def materialise(self) -> Any:
        if isinstance(self.root, jax.ShapeDtypeStruct):
            return self.spec()
        return self.fn(self.root)

    def on(self, root: Any) -> View:
        """The same view of another value of the root (its actual array, say)."""
        return View(root, self.fn, self.map)

    def __repr__(self) -> str:
        return f"<View {self.shape} of {tuple(self.map.base_shape)}>"


def make_view(fn: Callable[[Any], Any], base: Any) -> View | None:
    """`fn(base)` as a View of `base` (or of `base`'s own root, when it is a View), or None if it isn't one."""
    if isinstance(base, View):
        inner = base.fn

        def composed(r: Any) -> Any:
            return fn(inner(r))

        root, g = base.root, composed
    else:
        root, g = base, fn
    if not hasattr(root, "shape") or not hasattr(root, "dtype"):
        return None
    vmap = ViewMap.trace(g, root)
    if vmap is None or vmap.identity or vmap.shape == ():
        return None
    return View(root, g, vmap)
