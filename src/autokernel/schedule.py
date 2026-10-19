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
- `runtime`: a mask reads inputs. The same tables are built on device by `build_tables` (the kernel's
  wrapper calls it with jax.numpy), and the step count is read at run time.

Tiles are numbered by *ordinal*: their position among the tiles the kernel computes.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis import Specialization, evaluate_mask, phase_coords, tile_coords
from .errors import AutoKernelError

__all__ = ["Column", "Schedule", "build_tables", "plan"]

# A table entry's column code: (column << 2) | (first step of its phase << 1) | (last step of its phase).
FIRST, LAST, COLUMN_SHIFT = 2, 1, 2


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
    pattern_tiles: int = 0  # table mode: tiles in the repeating pattern
    pattern_steps: int = 0  # table mode: steps in the repeating pattern
    tables: dict[str, np.ndarray] = field(default_factory=dict)  # static SMEM tables
    core_steps: list[tuple[int, int]] = field(default_factory=list)  # static: each core's [begin, end) steps
    core_tiles: list[tuple[int, int]] = field(default_factory=list)  # static: each core's [begin, end) ordinals
    cores: int = 1

    @property
    def steps_per_tile(self) -> int:
        """Uniform mode: steps every tile runs (1 when there's no reduction)."""
        return max(1, sum(self.phase_steps))

    def column_offsets(self) -> list[int]:
        """Index of each phase's first column (its empty column follows its reduction columns)."""
        offsets, at = [], 0
        for k in self.phase_steps:
            offsets.append(at)
            at += k + 1
        return offsets


# --------------------------------------------------------------------------- planning


def plan(spec: Specialization, cores: int, smem_budget: int) -> Schedule:
    """Choose how the kernel's steps are ordered and decoded."""
    phase_steps = [p.steps for p in spec.phases]
    columns = [Column(p, d, False) for p, k in enumerate(phase_steps) for d in [*range(k), k]]
    columns = [Column(c.phase, c.depth, c.depth == phase_steps[c.phase]) for c in columns]
    masked = any(m is not None for m in spec.phase_masks)

    if spec.runtime_masks:
        candidates = len(spec.tiles)
        steps = candidates * max(1, len(columns))
        sched = Schedule("runtime", phase_steps, columns, candidates, steps, cores=cores)
        _check_smem(spec, 2 * steps * 4 + 4 * (2 * cores + 2), smem_budget)
        return sched

    if not masked:
        sched = Schedule("uniform", phase_steps, columns, len(spec.tiles), 0, cores=cores)
        sched.steps = sched.tiles * sched.steps_per_tile
        sched.core_tiles = _split(sched.tiles, cores)
        sched.core_steps = [(b * sched.steps_per_tile, e * sched.steps_per_tile) for b, e in sched.core_tiles]
        return sched

    # Static reduction masks: list the steps of one repeating pattern of tiles.
    inner = _pattern_tiles(spec)
    keep = _column_mask(spec, [m[:inner] if m is not None else None for m in spec.phase_masks], inner, np)
    step_tile, step_code, _ = build_tables(keep, columns, np)
    reps = len(spec.tiles) // inner
    sched = Schedule("table", phase_steps, columns, len(spec.tiles), reps * len(step_tile), cores=cores)
    sched.pattern_tiles, sched.pattern_steps = inner, len(step_tile)
    sched.tables = {"step_tile": step_tile.astype(np.int32), "step_code": step_code.astype(np.int32)}
    _check_smem(spec, sum(t.nbytes for t in sched.tables.values()), smem_budget)
    # Steps before each pattern tile, to find the step where each core's first tile starts.
    starts = np.searchsorted(step_tile, np.arange(inner + 1))
    sched.core_tiles = _split(sched.tiles, cores)
    sched.core_steps = [(_step_of(b, inner, starts), _step_of(e, inner, starts)) for b, e in sched.core_tiles]
    return sched


def _split(n: int, cores: int) -> list[tuple[int, int]]:
    bounds = [c * n // cores for c in range(cores + 1)]
    return list(itertools.pairwise(bounds))


def _step_of(ordinal: int, inner: int, starts: np.ndarray) -> int:
    rep, o = divmod(ordinal, inner)
    return int(rep * starts[-1] + starts[o])


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


def _check_smem(spec: Specialization, used: int, budget: int) -> None:
    if used > budget:
        raise AutoKernelError(
            f"{spec.nest.fn_name}: the step tables need {used / 1024:.0f} KiB of SMEM (budget {budget // 1024} KiB); "
            "use bigger blocks"
        )


# --------------------------------------------------------------------------- tables (numpy or jax.numpy)


def _column_mask(spec: Specialization, phase_masks: list[Any], tiles: int, xp: Any) -> Any:
    """keep[t, c]: tile t runs column c. A phase's empty column is kept when none of its positions are."""
    parts = []
    for p, phase in enumerate(spec.phases):
        m = phase_masks[p]
        real = xp.ones((tiles, phase.steps), bool) if m is None else xp.asarray(m, bool).reshape(tiles, phase.steps)
        parts += [real, ~real.any(axis=1, keepdims=True)]
    if not parts:
        return xp.ones((tiles, 1), bool)
    return xp.concatenate(parts, axis=1)


def build_tables(keep: Any, columns: list[Column], xp: Any, tile_keep: Any = None) -> tuple[Any, Any, Any]:
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
    for width in _phase_widths(columns):
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


def _phase_widths(columns: list[Column]) -> list[int]:
    widths: dict[int, int] = {}
    for c in columns:
        widths[c.phase] = widths.get(c.phase, 0) + 1
    return [widths[p] for p in sorted(widths)]


def runtime_tables(spec: Specialization, schedule: Schedule, inputs: dict[str, Any]) -> dict[str, Any]:
    """Build the step tables on device for masks that read inputs (called from the kernel's wrapper)."""
    import jax.numpy as jnp

    tiles = len(spec.tiles)
    tile_keep = None
    if spec.nest.guard is not None:
        tile_keep = jnp.broadcast_to(
            jnp.asarray(evaluate_mask(spec, spec.nest.guard, tile_coords(spec, jnp), jnp, inputs), bool), (tiles,)
        )
    phase_masks = []
    for p, phase in enumerate(spec.phases):
        guard = phase.reduction.guard
        if guard is None:
            phase_masks.append(None)
        elif not spec.phase_runtime[p]:
            phase_masks.append(jnp.asarray(spec.phase_masks[p]))
        else:
            coords = {
                **tile_coords(spec, jnp, column=True),
                **{k: jnp.asarray(v) for k, v in phase_coords(phase).items()},
            }
            m = evaluate_mask(spec, guard, coords, jnp, inputs)
            phase_masks.append(jnp.broadcast_to(jnp.asarray(m, bool), (tiles, phase.steps)))
    keep = _column_mask(spec, phase_masks, tiles, jnp)
    step_tile, step_code, count = build_tables(keep, schedule.columns, jnp, tile_keep)
    n_tiles = tiles if tile_keep is None else tile_keep.sum()
    tile_ids = jnp.arange(tiles) if tile_keep is None else jnp.nonzero(tile_keep, size=tiles, fill_value=0)[0]
    # Each core takes a contiguous run of tile ordinals; find the step where each run starts.
    cores = schedule.cores
    tile_bounds = (jnp.arange(cores + 1) * n_tiles) // cores
    step_bounds = jnp.searchsorted(jnp.where(jnp.arange(step_tile.shape[0]) < count, step_tile, n_tiles), tile_bounds)
    bounds = jnp.concatenate([step_bounds, tile_bounds]).astype(jnp.int32)
    return {
        "step_tile": step_tile,
        "step_code": step_code,
        "tile_ids": tile_ids.astype(jnp.int32),
        "bounds": bounds,  # [step bound per core..., tile bound per core...]
    }
