"""The order in which a kernel runs its steps.

A *step* is one (output tile, reduction position) pair; the kernel runs every step of a tile before
moving to the next tile. Within a tile, phase p (the p-th reduction loop) contributes one step per
reduction position its mask keeps, or a single *empty* step when the mask keeps none (so the units
around that loop still run).

Steps are decoded in one of three ways:

- `uniform`: no reduction masks and no runtime masks. Every tile has the same steps, so a step decodes
  arithmetically: tile = s // steps_per_tile, and so on. No tables.
- `table`: static reduction masks. Steps are listed in SMEM tables. When the masks only depend on the
  innermost parallel loops (causal attention ignores batch and head), the table holds one *pattern*
  that repeats for every combination of the outer loops, which keeps it small.
- `runtime`: a mask reads inputs. The same tables are built on device by `runtime_tables` (the kernel's
  wrapper calls it with jax.numpy), and the step count is read at run time.

Tiles are numbered by *ordinal*: their position among the tiles the kernel computes. With several
TensorCores each core takes a contiguous run of ordinals, cut so that the cores' step counts are as even
as possible (a tile can't be split: its accumulator lives on one core).
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from .analysis import Specialization, evaluate_mask
from .errors import AutoKernelError

__all__ = [
    "RUNTIME_TABLES",
    "Column",
    "Schedule",
    "build_tables",
    "check_smem",
    "column_mask",
    "core_bounds",
    "plan",
    "runtime_tables",
    "smem_tables",
    "step_tables",
]

# A table entry's column code: (column << 2) | (first step of its phase << 1) | (last step of its phase).
FIRST, LAST, COLUMN_SHIFT = 2, 1, 2

# Tables a runtime schedule builds on device, in kernel-argument order.
RUNTIME_TABLES = ("step_tile", "step_code", "tile_ids", "bounds")


@dataclass(frozen=True)
class Column:
    """A position within a tile's step pattern: a reduction position of a phase, or its empty step."""

    phase: int
    depth: int  # flattened index into the phase's reduction loops
    empty: bool


@dataclass
class Schedule:
    mode: str  # "uniform" | "table" | "runtime"
    phase_steps: list[int]  # reduction positions per phase
    columns: list[Column]  # the per-tile step pattern's columns (table and runtime modes)
    tiles: int  # tile ordinals (static) or candidate tiles (runtime)
    steps: int  # steps, or the most there can be (runtime)
    cores: int = 1
    pattern_tiles: int = 0  # table mode: tiles in the repeating pattern
    pattern_steps: int = 0  # table mode: steps in the repeating pattern
    tables: dict[str, np.ndarray] = field(default_factory=dict)  # static SMEM tables
    core_steps: list[tuple[int, int]] = field(default_factory=list)  # static: each core's [begin, end) steps
    core_tiles: list[tuple[int, int]] = field(default_factory=list)  # static: each core's [begin, end) ordinals

    @property
    def steps_per_tile(self) -> int:
        """Uniform mode: steps every tile runs (1 when there's no reduction)."""
        return max(1, sum(self.phase_steps))

    def column_offsets(self) -> list[int]:
        """Index of each phase's first column (its empty column follows its reduction columns)."""
        return list(itertools.accumulate([0, *(k + 1 for k in self.phase_steps[:-1])]))[: len(self.phase_steps)]


# --------------------------------------------------------------------------- planning


def plan(spec: Specialization, cores: int, smem_budget: int) -> Schedule:
    """Choose how the kernel's steps are ordered and decoded."""
    phase_steps = [p.steps for p in spec.phases]
    columns = [Column(p, d, d == k) for p, k in enumerate(phase_steps) for d in range(k + 1)]

    if spec.runtime_masks:
        candidates = len(spec.tiles)
        steps = candidates * max(1, len(columns))
        sched = Schedule("runtime", phase_steps, columns, candidates, steps, cores)
        check_smem(spec, sched, False, smem_budget)
        return sched

    if all(m is None for m in spec.phase_masks):
        per = max(1, sum(phase_steps))
        sched = Schedule("uniform", phase_steps, columns, len(spec.tiles), len(spec.tiles) * per, cores)
        tile_starts = np.arange(sched.tiles + 1) * per
    else:
        # Static reduction masks: list the steps of one repeating pattern of tiles.
        inner = _pattern_tiles(spec)
        keep = column_mask([m[:inner] if m is not None else None for m in spec.phase_masks], phase_steps, inner, np)
        step_tile, step_code, _ = build_tables(keep, phase_steps, np)
        reps = len(spec.tiles) // inner
        sched = Schedule("table", phase_steps, columns, len(spec.tiles), reps * len(step_tile), cores)
        sched.pattern_tiles, sched.pattern_steps = inner, len(step_tile)
        sched.tables = {"step_tile": step_tile, "step_code": step_code}
        starts = np.searchsorted(step_tile, np.arange(inner))  # first step of each pattern tile
        tile_starts = (np.arange(reps)[:, None] * len(step_tile) + starts[None, :]).reshape(-1)
        tile_starts = np.append(tile_starts, sched.steps)
    bounds = core_bounds(tile_starts, cores, np)
    sched.core_tiles = list(itertools.pairwise(bounds.tolist()))
    sched.core_steps = list(itertools.pairwise(tile_starts[bounds].tolist()))
    check_smem(spec, sched, False, smem_budget)
    return sched


def core_bounds(tile_starts, cores, xp):
    """Cut tile ordinals into `cores` contiguous runs, making the most steps any core runs as few as possible.

    `tile_starts[t]` is the first step of tile `t`, and `tile_starts[-1]` the total step count; the result
    has `cores + 1` ordinals, starting at 0 and ending at the tile count. Works with numpy and jax.numpy.
    """
    n_tiles = tile_starts.shape[0] - 1
    if cores <= 2:  # one cut, as near the middle as the tiles allow: no other cut runs fewer steps on a core
        targets = xp.arange(cores + 1) * tile_starts[-1] // cores
        after = xp.clip(xp.searchsorted(tile_starts, targets), 0, n_tiles)  # first tile starting at/after it
        before = xp.maximum(after - 1, 0)
        nearest = xp.where(targets - tile_starts[before] < tile_starts[after] - targets, before, after)
        ends = xp.asarray([0, n_tiles])
        cut = xp.arange(cores + 1)
        return xp.where(cut == 0, ends[0], xp.where(cut == cores, ends[1], nearest))

    def greedy(limit):
        # Each core takes tiles while its steps stay within `limit`: the runs' ends, and whether the last
        # core's run fits too.
        at, ends = 0, []
        for _ in range(cores - 1):
            at = xp.searchsorted(tile_starts, tile_starts[at] + limit, side="right") - 1
            ends.append(at)
        return ends, tile_starts[-1] - tile_starts[at] <= limit

    # Binary search for the least per-core step count the greedy runs can keep to (a fixed number of rounds
    # with jax.numpy, where the total is traced).
    lo, hi = xp.asarray(0), xp.asarray(tile_starts[-1])
    rounds = int(tile_starts[-1]).bit_length() + 1 if xp is np else 32
    for _ in range(rounds):
        mid = (lo + hi) // 2
        _, fits = greedy(mid)
        lo, hi = xp.where(fits, lo, mid + 1), xp.where(fits, mid, hi)
    ends, _ = greedy(hi)
    return xp.stack([xp.asarray(0), *[xp.asarray(e) for e in ends], xp.asarray(n_tiles)])


def _pattern_tiles(spec: Specialization) -> int:
    """How many trailing tiles the step pattern repeats over.

    If the reduction masks don't read the outer parallel loops (causal attention ignores batch and head),
    the tiles split into identical runs, one per value of those loops, and one run's steps repeat.
    """
    if spec.nest.guard is not None:  # a static tile mask: the tiles aren't a full product
        return len(spec.tiles)
    used: set[str] = set()
    for phase in spec.phases:
        if phase.reduction.guard is not None:
            used |= {n.id for n in ast.walk(phase.reduction.guard) if isinstance(n, ast.Name)}
    first_used = next((n for n, lb in enumerate(spec.parallel) if lb.var in used), len(spec.parallel))
    return int(np.prod([lb.count for lb in spec.parallel[first_used:]], dtype=np.int64))


def smem_tables(spec: Specialization, sched: Schedule, fills: bool) -> dict[str, int]:
    """Bytes of SMEM the kernel's tables and scalars take, by what they hold (see `codegen.Emitter`)."""
    word, tiles = 4, len(spec.tiles)
    used = {}
    if sched.mode == "table":
        used["step tables"] = sum(t.nbytes for t in sched.tables.values())
    elif sched.mode == "runtime":
        used["step tables"] = word * (2 * sched.steps + tiles)  # step_tile, step_code, tile_ids
    if sched.cores > 1 or sched.mode == "runtime":
        used["core bounds"] = word * 2 * (sched.cores + 1)
    if spec.nest.guard is not None and not spec.tile_mask_runtime:
        used["tile tables"] = word * len(spec.parallel) * tiles
    if fills:
        used["fill tables"] = word * (len(spec.parallel) * len(spec.constant_tiles) + tiles + 1)
    used["scalars"] = word * (len(spec.scalars) + sum(sv.smem for sv in spec.state.values()))
    return used


def check_smem(spec: Specialization, sched: Schedule, fills: bool, budget: int) -> None:
    """Raise AutoKernelError if the kernel's SMEM tables and scalars (`fills`: with the masked-out tiles'
    fill tables) don't fit in `budget` bytes."""
    used = smem_tables(spec, sched, fills)
    total = sum(used.values())
    if total > budget:
        parts = ", ".join(f"{what} {n / 1024:.0f} KiB" for what, n in used.items() if n)
        # How to shrink each table, biggest first.
        hints = {
            "tile tables": "bigger blocks of the parallel loops (the tile mask's tables list every kept tile)",
            "fill tables": "background_fill=False (skipped tiles then keep their allocation's values instead)",
            "step tables": "bigger blocks (the step tables list every step)",
        }
        ways = [hints[what] for what, _ in sorted(used.items(), key=lambda kv: -kv[1]) if what in hints and used[what]]
        raise AutoKernelError(
            f"{spec.nest.fn_name}: its tables need {total / 1024:.0f} KiB of SMEM ({parts}; budget "
            f"{budget // 1024} KiB); use " + ("; or ".join(ways) if ways else "bigger blocks")
        )


# --------------------------------------------------------------------------- tables (numpy or jax.numpy)
#
# Generated kernels with runtime masks carry the source of `step_tables` and the functions it calls, so
# these take numpy or jax.numpy as `xp`, read no module globals but `np` and `jnp`, and have no annotations.


def column_mask(phase_masks, phase_steps, tiles, xp):
    """keep[t, c]: tile t runs column c. A phase's empty column is kept when none of its positions are."""
    parts = []
    for m, steps in zip(phase_masks, phase_steps, strict=True):
        real = xp.ones((tiles, steps), bool) if m is None else xp.asarray(m, bool).reshape(tiles, steps)
        parts += [real, ~real.any(axis=1, keepdims=True)]
    if not parts:
        return xp.ones((tiles, 1), bool)
    return xp.concatenate(parts, axis=1)


def build_tables(keep, phase_steps, xp, tile_keep=None):
    """The steps a column mask keeps, tile-major: (step_tile ordinal, step_code, number of steps).

    With jax.numpy (runtime masks) the tables are padded to every possible step and the count is dynamic.
    """
    tiles, ncols = keep.shape
    if tile_keep is not None:
        keep = keep & tile_keep[:, None]
        ordinal = xp.cumsum(tile_keep) - 1  # position of each candidate among the kept tiles
    else:
        ordinal = xp.arange(tiles)
    # First / last kept column of each tile within each phase (each phase's columns are contiguous).
    firsts, lasts, at = [], [], 0
    for width in [steps + 1 for steps in phase_steps]:  # a phase's reduction positions and its empty column
        seg = keep[:, at : at + width]
        count = xp.cumsum(seg.astype(np.int32), axis=1)
        firsts.append(seg & (count == 1))
        lasts.append(seg & (count == count[:, -1:]))
        at += width
    first = xp.concatenate(firsts, axis=1) if firsts else keep
    last = xp.concatenate(lasts, axis=1) if lasts else keep
    col = xp.arange(ncols, dtype=np.int32)
    code = (col[None, :] << COLUMN_SHIFT) | (first.astype(np.int32) * FIRST) | (last.astype(np.int32) * LAST)
    flat_keep = keep.reshape(-1)
    if xp is np:
        idx = np.nonzero(flat_keep)[0]
        count = len(idx)
    else:
        count = flat_keep.sum()
        idx = xp.nonzero(flat_keep, size=flat_keep.shape[0], fill_value=0)[0]
    step_tile = ordinal[idx // ncols]
    step_code = code.reshape(-1)[idx]
    return step_tile.astype(np.int32), step_code.astype(np.int32), count


def runtime_tables(spec: Specialization, schedule: Schedule, inputs: dict[str, Any]) -> dict[str, Any]:
    """Build the step tables on device for masks that read inputs (as the kernel's wrapper does)."""
    tile_keep = evaluate_mask(spec, None, inputs) if spec.tile_mask_runtime else None
    phase_masks = []
    for p, phase in enumerate(spec.phases):
        if phase.reduction.guard is None:
            phase_masks.append(None)
        elif not spec.phase_runtime[p]:
            phase_masks.append(jnp.asarray(spec.phase_masks[p]))
        else:
            phase_masks.append(evaluate_mask(spec, p, inputs))
    return step_tables(tile_keep, phase_masks, schedule.phase_steps, len(spec.tiles), schedule.cores)


def step_tables(tile_keep, phase_masks, phase_steps, tiles, cores):
    """The tables of a runtime schedule (`RUNTIME_TABLES`), with jax.numpy, from the tile mask (None: every
    candidate tile runs) and each phase's (tiles, steps) mask (None: unmasked). Generated kernels carry this
    function's source (with `column_mask`, `build_tables` and `core_bounds`) to build their tables."""
    keep = column_mask(phase_masks, phase_steps, tiles, jnp)
    step_tile, step_code, count = build_tables(keep, phase_steps, jnp, tile_keep)
    n_tiles = tiles if tile_keep is None else tile_keep.sum()
    tile_ids = jnp.arange(tiles) if tile_keep is None else jnp.nonzero(tile_keep, size=tiles, fill_value=0)[0]
    # Each core takes a contiguous run of tile ordinals; cut the runs where the step counts come out even.
    valid_tile = jnp.where(jnp.arange(step_tile.shape[0]) < count, step_tile, n_tiles)  # padding sorts last
    tile_starts = jnp.searchsorted(valid_tile, jnp.arange(tiles + 1))  # tiles >= n_tiles start at `count`
    tile_starts = jnp.minimum(tile_starts, count)
    tile_bounds = jnp.minimum(core_bounds(tile_starts, cores, jnp), n_tiles)
    bounds = jnp.concatenate([tile_starts[tile_bounds], tile_bounds]).astype(jnp.int32)
    return {
        "step_tile": step_tile,
        "step_code": step_code,
        "tile_ids": tile_ids.astype(jnp.int32),
        "bounds": bounds,  # [step bound per core..., tile bound per core...]
    }
