"""Block-size tuning: generate or take the candidate values of the constants being tuned, rank their
combinations with the cost model, and optionally time the best few on the TPU.

The search is split in two so that thousands of combinations stay cheap:

1. *Screen* combinations with `lowering.plan`: specialise, check layout rules, schedule, estimate run
   time and VMEM. No code is generated, and unit costs are memoised across combinations, so a unit is
   only traced again when the tile shapes or constants it uses change. Combinations that can't fit in
   VMEM are pruned without planning; small spaces are screened exhaustively, large ones by coordinate
   descent over the model (see `Space`).
2. *Build* (generate the kernel for) only `Options.measure_top` of them: the best by the model's score,
   kept diverse (no more than half of them share any one tuned value), so that measuring can still find
   the winner where the model's best few differ by less than its error. The rest stay in the ranking
   with their predicted times; `Candidate.lowered` is None for them.

`tune="auto"` (or `{"TM": "auto", ...}`) generates the candidate values: see `block_sizes`.
"""

from __future__ import annotations

import ast
import functools
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from . import hardware, tiles
from .analysis import Environment, UserCode, environment
from .cost import Estimate
from .errors import AutoKernelError
from .frontend import LoopNest
from .lowering import ALL_NAMES, AUTO, Lowered, Options, VmemError, build, plan

__all__ = [
    "AUTO_MAX",
    "AUTO_MIN",
    "AUTO_WHOLE_MAX",
    "EXHAUSTIVE_MAX",
    "Candidate",
    "Space",
    "block_sizes",
    "diverse",
    "grid",
    "measure",
    "search",
    "time_kernel",
]

# Generated block sizes: powers of two, and three times powers of two, from the smallest the TPU layout
# allows for the dims a loop tiles (128 on the lane axis, 8/16/32 rows on the sublane axis by dtype,
# AUTO_MIN elsewhere) up to AUTO_MAX, plus the user's own value and a block covering the whole loop (up to
# AUTO_WHOLE_MAX: a single reduction step, as XLA's own matmul windows often take); only those dividing
# the loop are kept. VMEM pruning drops the ones too big to fit before they cost anything.
AUTO_MAX = 4096
AUTO_MIN = 8
AUTO_WHOLE_MAX = 8192
# Search spaces of at most this many combinations (after VMEM pruning) are planned exhaustively; larger
# ones are searched (see `Space`). Planning one takes 2-10 ms.
EXHAUSTIVE_MAX = 128


@dataclass
class Candidate:
    overrides: dict[str, Any]
    buffers: int | None  # None: the default depth, for a candidate that couldn't be planned
    lowered: Lowered | None  # the generated kernel: only for the best few by score
    error: str | None = None  # why it can't run (invalid tiling, too much VMEM, ...)
    measured_s: float | None = None
    estimate: Estimate | None = None  # the cost model's estimate, for every valid candidate
    cached: bool = False  # the pick of an earlier search, read from the tuning cache (see `cache`)
    bottleneck: str | None = None  # what profiling found limits it (`autotune="profile"`): dma/overhead/compute

    @property
    def predicted_s(self) -> float:
        est = self.lowered.estimate if self.lowered else self.estimate
        return est.seconds if est else float("inf")

    @property
    def score(self) -> float:
        """What candidates are ranked by: the predicted time, mildly penalised for VMEM use near the budget."""
        est = self.lowered.estimate if self.lowered else self.estimate
        return est.score if est else float("inf")


def search(
    nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct], options: Options
) -> list[Candidate]:
    """The candidates planned, valid ones first, each group best-scored (about: fastest-predicted) first.

    Each block-size combination is planned at the default buffer depth (see `Space` for which ones are);
    `options.measure_top` of the best are then built (see `diverse`). A measured search also builds those
    at the other depth (the cost model can't tell the depths apart well, only timing can).
    """
    space = Space(nest, fn, specs, options, grid(nest, fn, specs, options))
    candidates = space.search()
    candidates.sort(key=space.rank)
    if not candidates or candidates[0].error is not None:
        raise AutoKernelError(f"{nest.fn_name}: no tuning candidate is valid:\n  {_summarize(candidates)}")

    # Generate the best few. Plans are cheap to redo (unit costs are cached), so they aren't all kept.
    cache = space.cache
    for c in diverse([c for c in candidates if c.error is None], max(1, options.measure_top)):
        c.lowered = build(plan(nest, fn, specs, options, c.overrides, cost_cache=cache), options)
        if options.autotune in ("measure", "profile") and not options.buffers and c.buffers == 3:
            p2 = plan(nest, fn, specs, options, c.overrides, 2, cost_cache=cache)
            candidates.append(Candidate(c.overrides, 2, build(p2, options), estimate=p2.estimate))
    candidates.sort(key=space.rank)
    return candidates


Point = tuple[int, ...]  # an index into each tuned name's values


class Space:
    """The combinations of the tuned names' values, planned on demand, and how to search them.

    Planning a combination (specialise, schedule, estimate) takes a few milliseconds, so thousands of them
    would take tens of seconds. Three things keep a search short:

    1. *VMEM pruning.* A kernel's VMEM use grows with its block sizes, so once a combination doesn't fit,
       no combination with every block size at least as big does either: those are rejected without
       planning. Each block size's largest value that can fit at all (with the other block sizes at their
       smallest) is found first, by bisection. The exception is a block covering a whole loop: a reduction
       of one step fetches its tiles once per tile, into two slots rather than `buffers`, and needs no
       accumulator, so it can take less VMEM than a smaller block. Such a value is only compared with
       itself.
    2. Spaces of up to `EXHAUSTIVE_MAX` combinations left are planned exhaustively, so the ranking is
       exact.
    3. Larger ones are searched by coordinate descent over the model's score, from several starting points
       (the constants' own values and the middle of each range): each round tries every value of one name
       with the others fixed and moves to the best, until no name improves; then pairs of names are moved
       one step together (a bigger TM with a smaller TN, say), and the descent resumes from any
       improvement. Only the combinations planned are returned.
    """

    def __init__(
        self,
        nest: LoopNest,
        fn: Callable | Environment,
        specs: Sequence[jax.ShapeDtypeStruct],
        options: Options,
        tuned: list[tuple[str, tuple[Any, ...]]],
    ) -> None:
        self.nest, self.fn, self.specs, self.options = nest, fn, specs, options
        self.names = [name for name, _ in tuned]
        steps = _loop_steps(nest)
        # Block sizes (loop steps with int values) are searched in increasing order; VMEM grows with them.
        self.monotone = [
            name in steps and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values)
            for name, values in tuned
        ]
        self.values = [
            tuple(sorted(set(values))) if mono else _distinct(values)
            for (_, values), mono in zip(tuned, self.monotone, strict=True)
        ]
        # Per name, the values that are a whole loop it steps (VMEM isn't monotone across them).
        extents = _loop_extents(nest, fn, specs)
        self.whole = [
            {i for i, v in enumerate(values) if v in extents.get(name, ())} if mono else set()
            for name, values, mono in zip(self.names, self.values, self.monotone, strict=True)
        ]
        # Loops ragged as written: a candidate making another loop ragged would change what it computes (its
        # last tile reading zeros past an array's end), not just how fast.
        self.ragged = _ragged_loops(nest, fn, specs)
        self.cache: dict = {}  # unit costs, shared by every combination
        self.seen: dict[Point, Candidate] = {}
        self.too_big: list[Point] = []  # combinations that don't fit in VMEM (smallest known)

    @property
    def size(self) -> int:
        return int(np.prod([len(v) for v in self.values], dtype=np.int64))

    def rank(self, c: Candidate) -> tuple:
        """Sort key: valid first, best score first, ties in the order of the combinations."""
        point = tuple(_index(values, c.overrides[name]) for name, values in zip(self.names, self.values, strict=True))
        return (c.error is not None, c.score, point)

    def overrides(self, point: Point) -> dict[str, Any]:
        return {name: values[i] for name, values, i in zip(self.names, self.values, point, strict=True)}

    def dominated(self, point: Point) -> Point | None:
        """A combination known not to fit in VMEM whose block sizes are all at most `point`'s."""
        for big in self.too_big:
            if self._covers(big, point):
                return big
        return None

    def _at_most(self, k: int, small: int, big: int) -> bool:
        """Does value index `small` of name `k` take no more VMEM than `big`, the other names alike?"""
        if not self.monotone[k] or small in self.whole[k] or big in self.whole[k]:
            return small == big
        return small <= big

    def evaluate(self, point: Point) -> Candidate:
        if point in self.seen:
            return self.seen[point]
        overrides = self.overrides(point)
        if (big := self.dominated(point)) is not None:
            reason = f"too much VMEM: smaller blocks {self.overrides(big)} already don't fit"
            c = Candidate(overrides, self.options.buffers, None, reason)
        else:
            try:
                p = plan(self.nest, self.fn, self.specs, self.options, overrides, cost_cache=self.cache)
                made = [lb for lb in p.spec.loops.values() if lb.ragged and lb.var not in self.ragged]
                if made:
                    lb = made[0]
                    reason = (
                        f"loop `{lb.var}` would be ragged (range({lb.start}, {lb.end}, {lb.step})): its last tile "
                        "would read zeros past the end of an array, which can change the result"
                    )
                    c = Candidate(overrides, p.buffers, None, reason)
                else:
                    c = Candidate(overrides, p.buffers, None, estimate=p.estimate)
            except AutoKernelError as e:
                c = Candidate(overrides, self.options.buffers, None, " ".join(str(e).split()))
                if isinstance(e, VmemError):
                    self.too_big = [b for b in self.too_big if not self._covers(point, b)] + [point]
        self.seen[point] = c
        return c

    def _covers(self, small: Point, big: Point) -> bool:
        return all(self._at_most(k, s, b) for k, (s, b) in enumerate(zip(small, big, strict=True)))

    def search(self) -> list[Candidate]:
        self._prune()
        if self.size <= EXHAUSTIVE_MAX:
            points = list(itertools.product(*(range(len(v)) for v in self.values)))
        else:
            self._descend()
            if any(c.error is None for c in self.seen.values()):
                return list(self.seen.values())
            points = list(itertools.product(*(range(len(v)) for v in self.values)))  # nothing valid: say why
        return [self.evaluate(p) for p in points]

    def _prune(self) -> None:
        """Bisect each block size for its largest value that fits in VMEM with the others at their smallest."""
        low = tuple(0 for _ in self.values)
        for k, mono in enumerate(self.monotone):
            ordered = [i for i in range(len(self.values[k])) if i not in self.whole[k]]  # VMEM grows along these
            if not mono or len(ordered) <= 1:
                continue
            at = lambda i, k=k, ordered=ordered: tuple(ordered[i] if j == k else 0 for j in range(len(low)))  # noqa: E731
            if self._fits(at(len(ordered) - 1)) is not False:
                continue
            lo, hi = 0, len(ordered) - 1  # ordered[lo] isn't known not to fit; ordered[hi] doesn't fit
            while hi - lo > 1:
                mid = (lo + hi) // 2
                fits = self._fits(at(mid))
                if fits is None:  # fails for another reason: no conclusion
                    break
                lo, hi = (mid, hi) if fits else (lo, mid)

    def _fits(self, point: Point) -> bool | None:
        """Whether the combination fits in VMEM: None if it fails for some other reason."""
        c = self.evaluate(point)
        if c.error is None:
            return True
        return False if self.dominated(point) is not None else None

    def _descend(self) -> None:
        for seed in dict.fromkeys([self._nearest_current(), tuple(len(v) // 2 for v in self.values)]):
            self._descend_from(seed)

    def _score(self, point: Point) -> float:
        return self.evaluate(point).score

    def _descend_from(self, point: Point) -> Point:
        dims = range(len(self.values))
        while True:
            moved = True
            while moved:  # coordinate descent: the best value of each name in turn
                moved = False
                for k in dims:
                    line = [(*point[:k], i, *point[k + 1 :]) for i in range(len(self.values[k]))]
                    best = min(line, key=self._score)
                    if self._score(best) < self._score(point):
                        point, moved = best, True
            pairs = [
                tuple(point[d] + (da if d == a else db if d == b else 0) for d in dims)
                for a, b in itertools.combinations(dims, 2)
                for da in (-1, 1)
                for db in (-1, 1)
            ]
            pairs = [p for p in pairs if all(0 <= i < len(v) for i, v in zip(p, self.values, strict=True))]
            best = min(pairs, key=self._score, default=point)
            if self._score(best) >= self._score(point):
                return point
            point = best

    def _nearest_current(self) -> Point:
        """The constants' own values (the nearest candidate, for block sizes the candidates don't include)."""
        env = _prologue_env(self.nest, self.fn, self.specs)
        point = []
        for name, values, mono in zip(self.names, self.values, self.monotone, strict=True):
            current = env.get(name)
            if current in values:
                point.append(_index(values, current))
            elif mono and isinstance(current, (int, np.integer)) and current > 0:
                point.append(min(range(len(values)), key=lambda i: abs(np.log2(values[i] / current))))
            else:
                point.append(len(values) // 2)
        return tuple(point)


def _prologue_env(nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct]) -> dict[str, Any]:
    """The names the loops see: the constants, and what the prologue computes from the input shapes (block
    sizes declared with `autokernel.tile` among them); just the constants if the prologue fails."""
    source = environment(fn)
    env = {**source.names, **dict(zip(nest.params, specs, strict=True))}
    try:
        UserCode(nest, source.filename).run_prologue(env)
    except Exception:  # planning reports it
        return dict(source.names)
    return env


def _loop_steps(nest: LoopNest) -> set[str]:
    """Names used as the step of a loop (`range(0, M, TM)`): block sizes."""
    return {lp.step.id for lp in nest.loops if isinstance(lp.step, ast.Name)}


def _loop_extents(
    nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct]
) -> dict[str, set[int]]:
    """Per block size, the extents of the loops it steps (a block that size covers its loop in one step);
    empty when the bounds can't be worked out without the tuned values."""
    try:
        source = environment(fn)
        env = {**source.names, **dict(zip(nest.params, specs, strict=True))}
        code = UserCode(nest, source.filename)
        code.run_prologue(env)
        extents: dict[str, set[int]] = {}
        for lp in nest.loops:
            if isinstance(lp.step, ast.Name):
                start = code.eval_int(lp.start, f"`range` start of loop `{lp.var}`", env)
                stop = code.eval_int(lp.stop, f"`range` stop of loop `{lp.var}`", env)
                extents.setdefault(lp.step.id, set()).add(stop - start)
        return extents
    except Exception:  # planning reports it
        return {}


def _ragged_loops(nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct]) -> set[str]:
    """The loops whose range isn't a whole number of steps with the constants as written (empty when the bounds
    can't be worked out)."""
    try:
        source = environment(fn)
        env = {**source.names, **dict(zip(nest.params, specs, strict=True))}
        code = UserCode(nest, source.filename)
        code.run_prologue(env)
        ragged = set()
        for lp in nest.loops:
            start, stop, step = (code.eval_int(e, "`range` argument", env) for e in (lp.start, lp.stop, lp.step))
            if step > 0 and (stop - start) % step:
                ragged.add(lp.var)
        return ragged
    except Exception:  # planning reports it
        return set()


def _same(a: Any, b: Any) -> bool:
    """Are two tuned values the same candidate: equal and of the same type (`2` isn't `2.0`)?"""
    return type(a) is type(b) and bool(a == b)


def _distinct(values: Sequence[Any]) -> tuple[Any, ...]:
    """`values` without repeats, in order; values of different types (`2`, `2.0`) are different candidates."""
    out: list[Any] = []
    for v in values:
        if not any(_same(v, w) for w in out):
            out.append(v)
    return tuple(out)


def _index(values: Sequence[Any], value: Any) -> int:
    """Where `value` is among `values`: the same candidate, else an equal one (a block size as another int type)."""
    same = [i for i, v in enumerate(values) if _same(v, value)]
    return same[0] if same else values.index(value)


def diverse(ranked: list[Candidate], top: int) -> list[Candidate]:
    """`top` of the best-ranked candidates, of which at most half share any one value of a tuned name.

    The model's best few often differ by less than its error and all share one trait (the same TK, say);
    measuring a spread of them is what lets timing find the real winner. Names with a single value among
    the candidates don't count, and if the cap leaves fewer than `top`, the best of the rest fill up.
    """
    names = ranked[0].overrides if ranked else {}
    varied = {name for name in names if len(_distinct([c.overrides[name] for c in ranked])) > 1}
    cap = max(1, -(-top // 2))
    chosen: list[Candidate] = []
    counts: dict[tuple, int] = {}
    for c in ranked:
        traits = [(name, type(c.overrides[name]), c.overrides[name]) for name in sorted(varied)]
        if len(chosen) < top and all(counts.get(t, 0) < cap for t in traits):
            chosen.append(c)
            for t in traits:
                counts[t] = counts.get(t, 0) + 1
    rest = [c for c in ranked if all(c is not d for d in chosen)]
    picked = {id(c) for c in [*chosen, *rest[: top - len(chosen)]]}
    return [c for c in ranked if id(c) in picked]


def _summarize(candidates: list[Candidate], limit: int = 8) -> str:
    """The rejection reasons, most common first, each with one example."""
    by_reason: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_reason.setdefault(c.error or "", []).append(c)
    lines = []
    for reason, cs in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))[:limit]:
        more = f" (and {len(cs) - 1} more)" if len(cs) > 1 else ""
        lines.append(f"{cs[0].overrides}{more}: {reason}")
    return "\n  ".join(lines)


# --------------------------------------------------------------------------- the search space


def grid(
    nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct], options: Options
) -> list[tuple[str, tuple[Any, ...]]]:
    """The names being tuned and the values to try for each: `options.tune` with "auto" resolved."""
    used = {n.id for n in ast.walk(nest.fdef) if isinstance(n, ast.Name)}
    computed = _prologue_names(nest) - set(tiles.declarations(nest.prologue, environment(fn).names))
    explicit = {name: values for name, values in options.tune if values != AUTO}
    for name, values in explicit.items():
        if not isinstance(values, tuple):
            raise AutoKernelError(
                f"{nest.fn_name}: tune[{name!r}] must be {AUTO!r} or a sequence of candidate values, got {values!r}"
            )
        if name in computed:
            raise AutoKernelError(
                f"{nest.fn_name}: tune[{name!r}]: `{name}` is computed in the function's prologue, so overriding it "
                f"has no effect; declare it as a block size to tune it: `{name} = autokernel.tile(extent, block)`"
            )
        if name not in used:
            raise AutoKernelError(f"{nest.fn_name}: tune[{name!r}]: `{name}` isn't used by the function")
    auto = [name for name, values in options.tune if values == AUTO]
    if not auto:
        return list(explicit.items())

    hw = hardware.target(options.target)
    sizes, why_not = block_sizes(nest, fn, specs, hw.lanes)
    generated: dict[str, tuple[int, ...]] = {}
    for name in auto:
        if name == ALL_NAMES:
            generated.update((n, v) for n, v in sizes.items() if n not in explicit)
        elif name in sizes:
            generated[name] = sizes[name]
        else:
            reason = why_not.get(name, "isn't the step of any loop")
            raise AutoKernelError(
                f"{nest.fn_name}: tune[{name!r}] is 'auto', but `{name}` {reason}; 'auto' tunes block sizes: "
                "loop steps that are int constants (module globals or closure variables)"
            )
    if not generated and not explicit:
        reasons = "".join(f"\n  `{n}` {r}" for n, r in why_not.items())
        raise AutoKernelError(
            f"{nest.fn_name}: tune='auto' found no block sizes to tune: no loop steps by an int constant from a "
            f"module global or closure variable{reasons}"
        )
    return sorted([*explicit.items(), *generated.items()])


def block_sizes(
    nest: LoopNest, fn: Callable | Environment, specs: Sequence[jax.ShapeDtypeStruct], lanes: int = 128
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Candidate values for every block size in the nest, and why other loop steps can't be tuned.

    A block size is a name used as the step of a loop (`range(0, M, TM)`) bound to an int constant in the
    function's module globals or closure, or declared in the prologue with `autokernel.tile`. Other steps
    computed in the prologue (from input shapes) and steps of 1 (loops that index a dim) aren't block sizes.
    The candidates are powers of two and three times powers of two between a minimum set by the tiles the
    loops cut (`lanes` on a tile's last axis, the dtype's sublane count on its second-to-last, else
    `AUTO_MIN`) and `AUTO_MAX`, plus the current value and the whole loop (up to `AUTO_WHOLE_MAX`); each must
    divide every loop it steps.
    """
    source = environment(fn)
    env = source.names
    declared = set(tiles.declarations(nest.prologue, env))
    computed = _prologue_names(nest) - declared
    shapes_env = {**env, **dict(zip(nest.params, specs, strict=True))}
    code = UserCode(nest, source.filename)
    code.run_prologue(shapes_env)

    loops_of: dict[str, list] = {}
    for lp in nest.loops:
        if isinstance(lp.step, ast.Name):
            loops_of.setdefault(lp.step.id, []).append(lp)
    minimums = _tile_minimums(nest, specs, shapes_env, code, lanes)

    sizes: dict[str, tuple[int, ...]] = {}
    why_not: dict[str, str] = {}
    for name, loops in loops_of.items():
        value = shapes_env.get(name) if name in declared else env.get(name)
        if name in computed:
            why_not[name] = (
                "is computed in the function's prologue (from the input shapes); declare it as a block size to "
                f"tune it: `{name} = autokernel.tile(extent, block)`"
            )
        elif name not in env and name not in declared:
            why_not[name] = "isn't a module global or closure variable"
        elif not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            why_not[name] = f"is {value!r}, not an int"
        elif value <= 1:
            why_not[name] = f"is {value}: loops stepping by 1 index a dim, they have no block size"
        else:
            spans = [
                (code.eval_int(lp.start, f"`range` start of loop `{lp.var}`", shapes_env),
                 code.eval_int(lp.stop, f"`range` stop of loop `{lp.var}`", shapes_env))
                for lp in loops
            ]  # fmt: skip
            low = max(minimums.get(lp.var, AUTO_MIN) for lp in loops)
            sizes[name] = _values(int(value), low, spans)
    return sizes, why_not


def _values(current: int, low: int, spans: list[tuple[int, int]]) -> tuple[int, ...]:
    extents = [stop - start for start, stop in spans]
    high = min(AUTO_MAX, *extents)
    values = {1 << e for e in range(AUTO_MAX.bit_length()) if low <= 1 << e <= high}
    values |= {3 << e for e in range(AUTO_MAX.bit_length()) if low <= 3 << e <= high and (3 << e) % low == 0}
    values |= {e for e in extents if 0 < e <= AUTO_WHOLE_MAX}  # one block for the whole loop
    values.add(current)
    fits = tuple(v for v in sorted(values) if all(stop > start and start % v == 0 == stop % v for start, stop in spans))
    return fits or (current,)  # nothing divides: keep the current value and let planning say why


def _prologue_names(nest: LoopNest) -> set[str]:
    return {
        n.id for s in nest.prologue for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _tile_minimums(
    nest: LoopNest, specs: Sequence[jax.ShapeDtypeStruct], env: dict[str, Any], code: UserCode, lanes: int
) -> dict[str, int]:
    """Per loop variable, the smallest block TPU layout rules allow on the tile dims it slices."""
    accesses = []  # (dtype, rank, index) of every tile read of an input and every output store
    params = dict(zip(nest.params, specs, strict=True))
    for tree in [*nest.tile_statements(), *(st.value for st in nest.stores)]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in params:
                spec = params[node.value.id]
                accesses.append((spec.dtype, len(spec.shape), node.slice))
    stores = {st.out: st for st in nest.stores}
    for output in nest.outputs:
        alloc = code.compile(output.alloc, "eval")
        try:
            out = jax.eval_shape(
                lambda *xs, alloc=alloc: eval(alloc, {**env, **dict(zip(nest.params, xs, strict=True))}), *specs
            )
        except Exception:  # planning reports it
            continue
        if isinstance(out, jax.ShapeDtypeStruct):
            accesses.append((out.dtype, len(out.shape), stores[output.var].index))

    loop_vars = {lp.var for lp in nest.loops}
    minimums: dict[str, int] = {}
    for dtype, rank, index in accesses:
        dims = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        dims += [ast.Slice()] * (rank - len(dims))
        kept = [d for d in dims if isinstance(d, ast.Slice)]  # integer-indexed dims are dropped from the tile
        for pos, d in enumerate(kept):
            if isinstance(d.lower, ast.Name) and d.lower.id in loop_vars:
                from_end = len(kept) - pos
                need = lanes if from_end == 1 else hardware.sublanes(dtype) if from_end == 2 else AUTO_MIN
                minimums[d.lower.id] = max(minimums.get(d.lower.id, AUTO_MIN), need)
    return minimums


# --------------------------------------------------------------------------- measuring


def measure(
    candidates: list[Candidate],
    args: Sequence[Any],
    top: int,
    check: Callable[[Lowered], str | None] | None = None,
) -> Lowered | None:
    """Time the `top` best-predicted block sizes on the attached TPU, then the other buffer depth of the
    winner; the fastest, or None if none ran.

    `check(lowered)` runs a candidate and says why its output is wrong (None if it is right): a candidate
    whose output disagrees with the reference is dropped before it is timed, as one that fails to run is,
    so a miscompiled block size can't win by being fast."""
    valid = [c for c in candidates if c.lowered is not None]
    shapes = list(dict.fromkeys(_shape(c) for c in valid))  # ranked order

    def time_candidate(c: Candidate) -> None:
        try:
            wrong = check(c.lowered) if check is not None else None
            if wrong is not None:
                c.error = f"wrong result: {wrong}"
                return
            c.measured_s = time_kernel(c.lowered, args)
        except Exception as e:  # e.g. Mosaic rejects it: drop the candidate
            c.error = f"failed to run: {type(e).__name__}"

    for shape in shapes[:top]:  # each shape at its best-predicted (default) depth
        time_candidate(next(c for c in valid if _shape(c) == shape))
    timed = [c for c in valid if c.measured_s is not None]
    if not timed:
        return None
    best = min(timed, key=lambda c: c.measured_s)
    for c in valid:  # the winner's other depths
        if _shape(c) == _shape(best) and c.measured_s is None and c.error is None:
            time_candidate(c)
    timed = [c for c in valid if c.measured_s is not None]
    return min(timed, key=lambda c: c.measured_s).lowered


def candidate_key(c: Candidate) -> str:
    """A candidate's tuned values and buffer depth as a string, the same in every process (the tuning cache's
    history of measurements is keyed by it)."""
    values = ", ".join(f"{k}={type(v).__name__}:{v!r}" for k, v in sorted(c.overrides.items()))
    return f"{values}; buffers={c.buffers}"


def _volume(c: Candidate) -> float:
    """The product of a candidate's int block sizes: bigger blocks mean fewer steps and fewer bytes per flop."""
    sizes = [v for v in c.overrides.values() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)]
    return float(np.prod(sizes)) if sizes else 0.0


GUIDED_SEED = 4  # candidates the cost model ranks best, measured before following the profile
GUIDED_SPREAD = 8  # without a direction to follow, try no further down the model's ranking than this


def guided(
    candidates: list[Candidate],
    args: Sequence[Any],
    budget: int,
    make: Callable[[Candidate], Lowered],
    seconds: Callable[[Lowered, Sequence[Any]], float],
    diagnose: Callable[[Lowered, Sequence[Any]], str | None],
    check: Callable[[Lowered], str | None] | None = None,
    history: dict[str, float] | None = None,
) -> Lowered | None:
    """Profile-guided tuning: measure the cost model's best `GUIDED_SEED`, then repeatedly profile the fastest
    so far and try the candidate its bottleneck points to, up to `budget` new measurements; the fastest, or
    None if nothing ran.

    - "dma" (copies not hidden behind compute): the same blocks with one more buffer, else bigger blocks;
    - "overhead" (fixed per-step costs dominate): bigger blocks (fewer steps);
    - "compute", or no diagnosis: the next candidates the model ranks best (no further than `GUIDED_SPREAD`).

    `candidates` are ranked by the model; their `lowered` is built by `make` when first measured. `seconds`
    measures device time, `diagnose` returns a bottleneck, `check` drops wrong results (see `measure`).
    `history` maps `candidate_key` to seconds measured by earlier runs: those aren't measured again, and new
    measurements are added to it, so successive runs keep improving on the pick until nothing is left to try.
    """
    history = history if history is not None else {}
    valid = [c for c in candidates if c.error is None]
    for c in valid:
        if candidate_key(c) in history:
            c.measured_s = history[candidate_key(c)]

    def run(c: Candidate) -> None:
        try:
            if c.lowered is None:
                c.lowered = make(c)
            wrong = check(c.lowered) if check is not None else None
            if wrong is not None:
                c.error = f"wrong result: {wrong}"
                return
            c.measured_s = seconds(c.lowered, args)
            history[candidate_key(c)] = c.measured_s
        except Exception as e:  # e.g. Mosaic rejects it: drop the candidate
            c.error = f"failed to run: {type(e).__name__}"

    def timed() -> list[Candidate]:
        return [c for c in valid if c.measured_s is not None and c.error is None]

    spent = 0
    for c in valid[:GUIDED_SEED]:
        if spent < budget and c.measured_s is None:
            run(c)
            spent += 1
    while spent < budget and timed():
        best = min(timed(), key=lambda c: c.measured_s)
        if best.lowered is None:
            best.lowered = make(best)
        best.bottleneck = diagnose(best.lowered, args)
        nxt = _next_candidate(valid, best, best.bottleneck)
        if nxt is None:
            break
        run(nxt)
        spent += 1
    if not timed():
        return None
    best = min(timed(), key=lambda c: c.measured_s)
    if best.lowered is None:
        best.lowered = make(best)
    return best.lowered


def _next_candidate(valid: list[Candidate], best: Candidate, bottleneck: str | None) -> Candidate | None:
    """The untried candidate `best`'s bottleneck points to (see `guided`), best-ranked by the model first."""
    untried = [c for c in valid if c.measured_s is None and c.error is None]
    if bottleneck == "dma":
        deeper = [c for c in untried if _shape(c) == _shape(best) and (c.buffers or 0) > (best.buffers or 0)]
        if deeper:
            return deeper[0]
    if bottleneck in ("dma", "overhead"):
        bigger = [c for c in untried if _volume(c) > _volume(best)]
        if bigger:
            return bigger[0]
    near = [c for c in valid[:GUIDED_SPREAD] if c in untried]
    return near[0] if near else None


def _shape(c: Candidate) -> tuple:
    """A candidate's tuned values as a key (`2` and `2.0` differ)."""
    return tuple((k, type(v), v) for k, v in sorted(c.overrides.items(), key=lambda kv: kv[0]))


def time_kernel(
    lowered: Lowered, args: Sequence[Any], runs: int = 10, repeats: int = 3, interpret: Any = False
) -> float:
    """Median seconds per call on the attached TPU, after a compiling warm-up (`interpret`: in the Pallas TPU
    interpreter instead, which says nothing about TPU time but runs anywhere)."""
    fn = jax.jit(functools.partial(lowered.entry, interpret=interpret))
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / runs)
    return float(np.median(samples))
