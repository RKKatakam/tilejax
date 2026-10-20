"""The order a kernel runs its steps in, and how a step finds its tile.

A kernel is one loop over steps. Tiles run in the parallel loops' order (row-major, outermost slowest),
each tile's reduction steps back to back. Without masks a step's tile and reduction block are arithmetic
on the step index. Masks are evaluated here, once per loop value they read, and the kernel walks tables
(in SMEM) of only the tiles and steps that run:

- step tables (a reduction mask): for every step, its tile and a code packing its reduction block and
  whether it is the tile's first and last step;
- tile tables (a tile mask): for every tile that runs, the block index of each parallel loop; and for
  every tile a mask skips, the same, so the kernel can write it with the output's fill.

The tables cover only the loops the masks read (the innermost parallel loops from the first one any mask
reads): causal attention's masks don't read the batch and head loops, so its tables hold one batch-head's
pattern, repeated, and stay small whatever the batch size.
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


@dataclass(frozen=True)
class Schedule:
    """Which tiles and steps a kernel runs, in order, and the tables that find them."""

    tiles: int  # output tiles that run
    total_tiles: int  # output tiles, skipped ones included
    steps: int
    steps_per_tile: int | None  # when every tile runs every reduction step; None: the step tables say
    outer: tuple[Range, ...]  # parallel loops no mask reads: arithmetic on the step or tile index
    inner: tuple[Range, ...]  # the parallel loops the tables cover
    pattern_tiles: int  # tiles that run per combination of the outer loops
    pattern_steps: int  # steps per combination of the outer loops
    pattern_fills: int  # tiles skipped per combination of the outer loops
    tile_tables: bool  # are tiles found through tables (a tile mask), or arithmetic
    tables: dict[str, np.ndarray] = field(default_factory=dict)  # name -> int32 table, in SMEM

    @property
    def repeats(self) -> int:
        """How many times the tables' pattern repeats: once per combination of the outer loops."""
        return math.prod(r.count for r in self.outer)

    @property
    def fills(self) -> int:
        """Skipped output tiles: the kernel writes them with the outputs' fill values."""
        return self.pattern_fills * self.repeats

    def step_at(self, step: int) -> tuple[int, int]:
        """(the tile, the reduction block) of a step: what the kernel's `step` function computes."""
        if self.steps_per_tile is not None:
            tile, block = divmod(step, self.steps_per_tile)
            return tile, block
        rep, r = divmod(step, self.pattern_steps)
        return rep * self.pattern_tiles + int(self.tables[STEP_TILE][r]), int(self.tables[STEP_CODE][r]) >> 2

    def blocks_at(self, tile: int) -> tuple[int, ...]:
        """The parallel loops' blocks of the `tile`-th tile that runs: what the kernel's `blocks` computes."""
        if not self.tile_tables:
            return _mixed_radix(tile, self.outer + self.inner)
        rep, t = divmod(tile, self.pattern_tiles)
        return _mixed_radix(rep, self.outer) + tuple(int(self.tables[TILE_TABLE % r.var][t]) for r in self.inner)


# Table names: steps' tiles and codes, and per inner loop `var`: TILE_TABLE % var, FILL_TABLE % var.
STEP_TILE, STEP_CODE, TILE_TABLE, FILL_TABLE = "step_tile", "step_code", "tile_%s", "fill_%s"


def make_schedule(plan: Plan) -> Schedule:
    """The steps `plan`'s kernel runs; raises KernelError if a mask leaves nothing to run, or a tile with no
    steps, or tables too big for SMEM. Tables are made only for what the masks actually skip."""
    nest = plan.nest
    tile_masks = [r.mask for r in plan.parallel if r.mask is not None]
    step_mask = plan.reduction.mask if plan.reduction else None
    steps_per_tile = plan.reduction.count if plan.reduction else 1
    if not tile_masks and step_mask is None:
        return _arithmetic(plan, steps_per_tile)

    read = set().union(*(reads(m) for m in [*tile_masks, step_mask] if m is not None))
    split = next((i for i, r in enumerate(plan.parallel) if r.var in read), len(plan.parallel))
    outer, inner = plan.parallel[:split], plan.parallel[split:]

    active, skipped, steps = [], [], []  # steps: the reduction blocks each active tile runs
    for blocks in itertools.product(*(range(r.count) for r in inner)):
        values = {r.var: r.value(b) for r, b in zip(inner, blocks, strict=True)}
        if any(_holds(plan, mask, values) for mask in tile_masks):
            skipped.append(blocks)
            continue
        active.append(blocks)
        steps.append([kb for kb in range(steps_per_tile) if step_mask is None or not _step_skipped(plan, values, kb)])
        if not steps[-1]:
            where = ", ".join(f"{var}={value}" for var, value in values.items())
            raise nest.error(
                f"the mask skips every reduction step of the tile at {where or 'every tile'}: every tile that "
                "runs takes at least one step (skip the whole tile with a mask on a parallel loop)",
                nest.reduction.node if nest.reduction else None,
            )
    if not active:
        raise nest.error("the masks skip every output tile", nest.parallel[0].node)
    step_tables = any(len(ks) < steps_per_tile for ks in steps)
    if not skipped and not step_tables:
        return _arithmetic(plan, steps_per_tile)  # the masks skip nothing

    tables: dict[str, np.ndarray] = {}
    if skipped:
        for i, r in enumerate(inner):
            tables[TILE_TABLE % r.var] = np.array([b[i] for b in active], np.int32)
            tables[FILL_TABLE % r.var] = np.array([b[i] for b in skipped], np.int32)
    if step_tables:
        tables[STEP_TILE] = np.array([t for t, ks in enumerate(steps) for _ in ks], np.int32)
        tables[STEP_CODE] = np.array(
            [kb << 2 | (n == 0) << 1 | (n == len(ks) - 1) for ks in steps for n, kb in enumerate(ks)], np.int32
        )

    smem = sum(t.nbytes for t in tables.values())
    if smem > v5e.SMEM_BUDGET:
        raise nest.error(
            f"the masks' step tables take {smem // 1024} KiB of SMEM, more than the {v5e.SMEM_BUDGET // 1024} KiB "
            "a kernel can use: use bigger tiles",
            nest.parallel[0].node,
        )
    repeats = math.prod(r.count for r in outer)
    pattern_steps = sum(len(ks) for ks in steps)
    tiles, fills = repeats * len(active), repeats * len(skipped)
    if tiles * max(fills, 1) >= 2**31:
        raise nest.error("too many output tiles for the kernel's int32 step arithmetic", nest.parallel[0].node)
    return Schedule(
        tiles=tiles,
        total_tiles=math.prod(r.count for r in plan.parallel),
        steps=repeats * pattern_steps,
        steps_per_tile=None if step_tables else steps_per_tile,
        outer=outer,
        inner=inner,
        pattern_tiles=len(active),
        pattern_steps=pattern_steps,
        pattern_fills=len(skipped),
        tile_tables=bool(skipped),
        tables=tables,
    )


def _arithmetic(plan: Plan, steps_per_tile: int) -> Schedule:
    """Every tile runs every step: steps and tiles are arithmetic on the step index, no tables."""
    tiles = math.prod(r.count for r in plan.parallel)
    return Schedule(
        tiles=tiles,
        total_tiles=tiles,
        steps=tiles * steps_per_tile,
        steps_per_tile=steps_per_tile,
        outer=plan.parallel,
        inner=(),
        pattern_tiles=tiles,
        pattern_steps=tiles * steps_per_tile,
        pattern_fills=0,
        tile_tables=False,
    )


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


def _step_skipped(plan: Plan, values: dict[str, int], kb: int) -> bool:
    reduction = plan.reduction
    assert reduction is not None and reduction.mask is not None
    return _holds(plan, reduction.mask, {**values, reduction.var: reduction.value(kb)})


def _mixed_radix(index: int, ranges: tuple[Range, ...]) -> tuple[int, ...]:
    """The block of each loop in `ranges` at row-major position `index` (outermost first)."""
    blocks = []
    for r in reversed(ranges):
        index, block = divmod(index, r.count)
        blocks.append(block)
    return tuple(reversed(blocks))
