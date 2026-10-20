"""The order a kernel runs its steps in, and how a step finds its tile, phase and block.

A kernel is one loop over steps. Tiles run in the parallel loops' order (row-major, outermost slowest);
each tile's steps are its phases' steps, back to back: the first reduction loop's, then the second's, ...
Without masks a step's tile, phase and reduction block are arithmetic on the step index. Masks are
evaluated here, once per loop value they read, and the kernel walks tables (in SMEM) of only the tiles and
steps that run:

- step tables (a reduction mask that skips steps): for every step, its tile and a code packing its
  reduction block, its phase, and whether it is the phase's first and last step, and empty;
- tile tables (a tile mask that skips tiles): for every tile that runs, the block index of each parallel
  loop; and for every tile a mask skips, the same, so the kernel can write it with the output's fill.

The tables cover only the loops the masks read (the innermost parallel loops from the first one any mask
reads): causal attention's masks don't read the batch and head loops, so its tables hold one batch-head's
pattern, repeated, and stay small whatever the batch size.

A tile whose mask skips every block of a reduction loop still takes one step of it, an empty one: its body
doesn't run (as the loop's body doesn't), but the code after the loop does.

Masks that read index arrays (runtime masks, `if j >= lengths[b]: continue`) can only be evaluated when the
kernel runs: then the step tables are made on the device, just before the kernel, from every step that
could run (`runtime`); the schedule describes those steps, and its step count is the most there can be.
"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import v5e
from .plan import Plan, Range
from .tilecode import reads

__all__ = ["Schedule", "make_schedule"]

# Table names: steps' tiles and codes (and, made on the device, their count), and per inner loop `var`:
# TILE_TABLE % var, FILL_TABLE % var.
STEP_TILE, STEP_CODE, STEP_COUNT, TILE_TABLE, FILL_TABLE = "step_tile", "step_code", "step_count", "tile_%s", "fill_%s"
# A step's code: (block * phases + phase) << CODE_SHIFT | empty << 2 | first << 1 | last.
CODE_SHIFT = 3


@dataclass(frozen=True)
class Schedule:
    """Which tiles and steps a kernel runs, in order, and the tables that find them."""

    tiles: int  # output tiles that run
    total_tiles: int  # output tiles, skipped ones included
    steps: int
    phase_counts: tuple[int, ...]  # each phase's reduction blocks (its steps per tile, when nothing is masked)
    phase_steps: tuple[int, ...]  # each phase's steps over the whole run
    step_tables: bool  # are steps found through tables (a mask skips steps), or arithmetic
    outer: tuple[Range, ...]  # parallel loops no mask reads: arithmetic on the step or tile index
    inner: tuple[Range, ...]  # the parallel loops the tables cover
    pattern_tiles: int  # tiles that run per combination of the outer loops
    pattern_steps: int  # steps per combination of the outer loops
    pattern_fills: int  # tiles skipped per combination of the outer loops
    tile_tables: bool  # are tiles found through tables (a tile mask skips tiles), or arithmetic
    tables: dict[str, np.ndarray] = field(default_factory=dict)  # name -> int32 table, in SMEM
    empty: bool = False  # may a step be empty (a phase whose mask skips all of a tile's blocks)?
    # Runtime masks: the step tables are made on the device from every step that could run (`steps` is the
    # most there can be). For each of the pattern's tiles, the value of each inner loop; for each phase whose
    # mask reads no index array, which of each tile's blocks run (the others are evaluated on the device).
    runtime: bool = False
    tile_values: dict[str, np.ndarray] = field(default_factory=dict)
    static_runs: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def steps_per_tile(self) -> int:
        """A tile's steps when nothing is masked: every phase's blocks (one step if there are no phases)."""
        return sum(self.phase_counts) or 1

    @property
    def repeats(self) -> int:
        """How many times the tables' pattern repeats: once per combination of the outer loops."""
        return math.prod(r.count for r in self.outer)

    @property
    def fills(self) -> int:
        """Skipped output tiles: the kernel writes them with the outputs' fill values."""
        return self.pattern_fills * self.repeats

    def step_at(self, step: int) -> tuple[int, int, int]:
        """(the tile, the phase, the reduction block) of a step: what the kernel's `step` function computes."""
        if not self.step_tables:
            tile, within = divmod(step, self.steps_per_tile)
            phase = 0
            while phase + 1 < len(self.phase_counts) and within >= self.phase_counts[phase]:
                within -= self.phase_counts[phase]
                phase += 1
            return tile, phase, within
        rep, r = divmod(step, self.pattern_steps)
        tile = rep * self.pattern_tiles + int(self.tables[STEP_TILE][r])
        phase_block = int(self.tables[STEP_CODE][r]) >> CODE_SHIFT
        return tile, phase_block % len(self.phase_counts), phase_block // len(self.phase_counts)

    def empty_at(self, step: int) -> bool:
        """Is the step empty (its phase's body doesn't run)?"""
        return self.empty and bool(self.tables[STEP_CODE][step % self.pattern_steps] & 4)

    def blocks_at(self, tile: int) -> tuple[int, ...]:
        """The parallel loops' blocks of the `tile`-th tile that runs: what the kernel's `blocks` computes."""
        if not self.tile_tables:
            return _mixed_radix(tile, self.outer + self.inner)
        rep, t = divmod(tile, self.pattern_tiles)
        return _mixed_radix(rep, self.outer) + tuple(int(self.tables[TILE_TABLE % r.var][t]) for r in self.inner)


def make_schedule(plan: Plan) -> Schedule:
    """The steps `plan`'s kernel runs; raises KernelError if a mask leaves nothing to run, or tables too big
    for SMEM. Tables are made only for what the masks actually skip."""
    nest = plan.nest
    phases = [phase.range for phase in plan.phases]
    counts = tuple(r.count for r in phases)
    tile_masks = [r.mask for r in plan.parallel if r.mask is not None]
    step_masks = [r.mask for r in phases if r.mask is not None]
    if not tile_masks and not step_masks:
        _check_smem(plan, {})
        return _arithmetic(plan, counts)

    read = set().union(*(reads(m) for m in [*tile_masks, *step_masks]))
    split = next((i for i, r in enumerate(plan.parallel) if r.var in read), len(plan.parallel))
    outer, inner = plan.parallel[:split], plan.parallel[split:]

    active, skipped, steps = [], [], []  # steps: per active tile, each phase's reduction blocks
    for blocks in itertools.product(*(range(r.count) for r in inner)):
        values = {r.var: r.value(b) for r, b in zip(inner, blocks, strict=True)}
        if any(_holds(plan, mask, values) for mask in tile_masks):
            skipped.append(blocks)
            continue
        active.append(blocks)
        if not plan.runtime:
            steps.append([_blocks_run(plan, r, values) for r in phases])
    if not active:
        raise nest.error("the masks skip every output tile", nest.parallel[0].node)
    if plan.runtime:
        return _runtime(plan, counts, outer, inner, active, skipped)
    step_tables = any(ks != list(range(r.count)) for tile in steps for ks, r in zip(tile, phases, strict=True))
    if not skipped and not step_tables:
        _check_smem(plan, {})
        return _arithmetic(plan, counts)  # the masks skip nothing

    tables = _tile_tables(inner, active, skipped)
    empty = False
    if step_tables:
        tile_of, codes = [], []
        for t, tile in enumerate(steps):
            for p, ks in enumerate(tile):
                runs = ks or [0]  # a phase none of whose blocks run takes one empty step
                empty |= not ks
                tile_of += [t] * len(runs)
                codes += [_code(kb, p, len(phases), not ks, n == 0, n == len(runs) - 1) for n, kb in enumerate(runs)]
        tables[STEP_TILE] = np.array(tile_of, np.int32)
        tables[STEP_CODE] = np.array(codes, np.int32)
    _check_smem(plan, tables)
    repeats = math.prod(r.count for r in outer)
    pattern_steps = sum(max(len(ks), 1) for tile in steps for ks in tile) or len(active)
    tiles, fills = repeats * len(active), repeats * len(skipped)
    _check_tiles(plan, tiles, fills)
    return Schedule(
        tiles=tiles,
        total_tiles=math.prod(r.count for r in plan.parallel),
        steps=repeats * pattern_steps,
        phase_counts=counts,
        phase_steps=tuple(repeats * sum(max(len(tile[p]), 1) for tile in steps) for p in range(len(phases))),
        step_tables=step_tables,
        outer=outer,
        inner=inner,
        pattern_tiles=len(active),
        pattern_steps=pattern_steps,
        pattern_fills=len(skipped),
        tile_tables=bool(skipped),
        tables=tables,
        empty=empty,
    )


def _runtime(
    plan: Plan,
    counts: tuple[int, ...],
    outer: tuple[Range, ...],
    inner: tuple[Range, ...],
    active: list[tuple[int, ...]],
    skipped: list[tuple[int, ...]],
) -> Schedule:
    """A schedule whose step tables are made on the device (masks read index arrays): every step that could
    run, described by the pattern's tiles' loop values and the masks that can be evaluated now."""
    phases = [phase.range for phase in plan.phases]
    tables = _tile_tables(inner, active, skipped)
    capacity = len(active) * sum(counts)
    # The tables made on the device: each step's tile and code, and their count.
    _check_smem(plan, tables, 4 * (2 * capacity + 1))
    tile_values = {r.var: np.array([r.value(blocks[i]) for blocks in active], np.int32) for i, r in enumerate(inner)}
    static_runs = {}
    for p, r in enumerate(phases):
        if r.mask is not None and not r.runtime:
            static_runs[p] = np.array(
                [
                    [not _holds(plan, r.mask, {**_values(inner, blocks), r.var: r.value(kb)}) for kb in range(r.count)]
                    for blocks in active
                ]
            )
    repeats = math.prod(q.count for q in outer)
    tiles = repeats * len(active)
    _check_tiles(plan, tiles, repeats * len(skipped))
    return Schedule(
        tiles=tiles,
        total_tiles=math.prod(r.count for r in plan.parallel),
        steps=repeats * capacity,
        phase_counts=counts,
        phase_steps=tuple(tiles * count for count in counts),
        step_tables=True,
        outer=outer,
        inner=inner,
        pattern_tiles=len(active),
        pattern_steps=capacity,
        pattern_fills=len(skipped),
        tile_tables=bool(skipped),
        tables=tables,
        empty=True,
        runtime=True,
        tile_values=tile_values,
        static_runs=static_runs,
    )


def _code(block: int, phase: int, phases: int, empty: bool, first: bool, last: bool) -> int:
    return (block * phases + phase) << CODE_SHIFT | empty << 2 | first << 1 | last


def _values(inner: tuple[Range, ...], blocks: tuple[int, ...]) -> dict[str, int]:
    return {r.var: r.value(b) for r, b in zip(inner, blocks, strict=True)}


def _tile_tables(
    inner: tuple[Range, ...], active: list[tuple[int, ...]], skipped: list[tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """The inner loops' blocks of the tiles that run and of those skipped (none if nothing is skipped)."""
    tables: dict[str, np.ndarray] = {}
    if skipped:
        for i, r in enumerate(inner):
            tables[TILE_TABLE % r.var] = np.array([b[i] for b in active], np.int32)
            tables[FILL_TABLE % r.var] = np.array([b[i] for b in skipped], np.int32)
    return tables


def _check_smem(plan: Plan, tables: dict[str, np.ndarray], more: int = 0) -> None:
    """Tables, index arrays (and `more` bytes) must fit in what SMEM a kernel can use."""
    smem = sum(t.nbytes for t in tables.values()) + more
    smem += sum(v5e.padded_bytes(index.shape, index.dtype) for index in plan.indices)
    if smem > v5e.SMEM_BUDGET:
        raise plan.nest.error(
            f"the step tables and index arrays take {smem // 1024} KiB of SMEM, more than the "
            f"{v5e.SMEM_BUDGET // 1024} KiB a kernel can use: use bigger tiles or smaller index arrays",
            plan.nest.parallel[0].node,
        )


def _check_tiles(plan: Plan, tiles: int, fills: int) -> None:
    if tiles * max(fills, 1) >= 2**31:
        node = plan.nest.parallel[0].node
        raise plan.nest.error("too many output tiles for the kernel's int32 step arithmetic", node)


def _arithmetic(plan: Plan, counts: tuple[int, ...]) -> Schedule:
    """Every tile runs every step: steps, phases and tiles are arithmetic on the step index, no tables."""
    tiles = math.prod(r.count for r in plan.parallel)
    per_tile = sum(counts) or 1
    return Schedule(
        tiles=tiles,
        total_tiles=tiles,
        steps=tiles * per_tile,
        phase_counts=counts,
        phase_steps=tuple(tiles * count for count in counts),
        step_tables=False,
        outer=plan.parallel,
        inner=(),
        pattern_tiles=tiles,
        pattern_steps=tiles * per_tile,
        pattern_fills=0,
        tile_tables=False,
    )


def _blocks_run(plan: Plan, phase: Range, values: dict[str, int]) -> list[int]:
    """The reduction blocks of `phase` its mask runs for a tile at these loop values."""
    return [
        kb
        for kb in range(phase.count)
        if phase.mask is None or not _holds(plan, phase.mask, {**values, phase.var: phase.value(kb)})
    ]


def _holds(plan: Plan, mask: ast.expr, values: dict[str, int]) -> bool:
    """Does `mask` hold (skip) for these loop values?"""
    result = plan.eval(mask, values)
    try:
        return bool(result)
    except Exception as e:
        raise plan.nest.error(
            f"masks are Python conditions on loop variables and constants, evaluated when planning; this one "
            f"gave {type(result).__name__}",
            mask,
        ) from e


def _mixed_radix(index: int, ranges: tuple[Range, ...]) -> tuple[int, ...]:
    """The block of each loop in `ranges` at row-major position `index` (outermost first)."""
    blocks = []
    for r in reversed(ranges):
        index, block = divmod(index, r.count)
        blocks.append(block)
    return tuple(reversed(blocks))
