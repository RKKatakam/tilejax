"""The manual backend: one loop over every step, with hand-written double-buffered DMAs and no grid.

A step is one (output tile, reduction index) pair. The kernel walks the output tiles in loop order (only
the active ones under a tile mask), and for each runs the reduction steps:

    for t in tiles:                       # from SMEM tables of block indices
        for k in range(K):
            prefetch step s+1's input blocks into the other slot
            wait for step s's input blocks
            init accumulators (k == 0), run the reduction step, and on k == K-1 write the output tile
            into one of two VMEM slots and copy it to HBM asynchronously

Inputs whose block changes every step (they're tiled by the reduction loop) alternate two slots by step.
Inputs that only change between tiles (e.g. a bias row) are re-fetched only when their block changes.
The loop carries the tile and reduction counters, so a step costs a handful of scalar ops instead of the
grid pipeliner's bookkeeping.
"""

from __future__ import annotations

import ast

import numpy as np

from .analysis import Block, Operand, Specialization
from .emit import Emitter, Options, dtype_src, indent
from .errors import AutoKernelError

# A step-table entry packs (tile << _TILE_SHIFT) | (reduction block << 2) | (first << 1) | last.
_TILE_SHIFT = 12
_MAX_RED_BLOCKS = 1 << (_TILE_SHIFT - 2)

__all__ = ["ManualEmitter"]


class ManualEmitter(Emitter):
    pipeline = "manual"

    def __init__(self, spec: Specialization, options: Options):
        super().__init__(spec, options)
        names = spec.names
        self.K = self.red.grid if self.red else 1
        self.n_tiles = len(self.tiles())
        # With a reduction mask, the loop walks a table of only the steps that run; otherwise every tile
        # runs all K reduction steps and the loop carries tile/reduction counters.
        self.step_table = spec.red_blocks is not None
        self.n_steps = sum(map(len, spec.red_blocks)) if spec.red_blocks is not None else self.n_tiles * self.K

        # SMEM tables: the absolute block index of each parallel loop, per computed tile.
        parallel = [lp.var for lp in self.nest.parallel]
        if self.mask is not None:
            self.block_tables = dict(self.mask.tables)
        else:
            self.block_tables = {v: names.fresh(f"{v}_blocks") for v in parallel}
        for n, v in enumerate(parallel):
            self.tables[self.block_tables[v]] = np.array([blocks[n] for blocks in self.tiles()], np.int32)
        if self.step_table:
            self.steps_name = names.fresh("steps")
            self.tables[self.steps_name] = self._pack_steps()
        self.tables.update(self._fill_tables())

        # Kernel-local names that can't collide with the user's.
        self.v = {
            key: names.fresh(key)
            for key in (
                "tile",
                "depth",
                "step",
                "carry",
                "last",
                "next_tile",
                "next_depth",
                "slot",
                "has_next",
                "safe_tile",
                "out_slot",
                "first",
                "entry",
                "next_entry",
            )
        }
        self.hbm = {}  # input param -> its HBM ref in the kernel
        for op in spec.operands:
            if op.param not in self.hbm:
                self.hbm[op.param] = names.fresh(f"{op.param}_hbm")
        self.out_hbm = names.fresh(f"{self.nest.out_var}_hbm")
        # Masked-out tiles with no known constant keep the allocation's values through an aliased input.
        self.out_init = (
            names.fresh(f"{self.nest.out_var}_init")
            if self.mask is not None and self.mask.constant and not self.fill
            else None
        )
        self.slots = {op.ref: names.fresh(f"{op.ref}_slots") for op in spec.operands}
        self.sems = {op.ref: names.fresh(f"{op.ref}_sems") for op in spec.operands}
        self.out_slots = names.fresh(f"{spec.out_ref}_slots")
        self.out_sems = names.fresh(f"{spec.out_ref}_sems")
        self.streamed = [op for op in spec.operands if self._streamed(op)]
        self.per_tile = [op for op in spec.operands if not self._streamed(op)]
        self.version = {op.ref: names.fresh(f"{op.ref}_version") for op in self.per_tile}
        self.pending = {op.ref: names.fresh(f"{op.ref}_pending") for op in self.per_tile}
        self.changed = {op.ref: names.fresh(f"{op.ref}_changed") for op in self.per_tile}

    def steps(self) -> int:
        return self.n_steps

    def _pack_steps(self) -> np.ndarray:
        assert self.spec.red_blocks is not None and self.red is not None
        if self.n_tiles >= 1 << (31 - _TILE_SHIFT) or self.K > _MAX_RED_BLOCKS:
            raise AutoKernelError(
                f"{self.nest.fn_name}: too many tiles ({self.n_tiles}) or reduction blocks ({self.K}) for the step "
                "table; use bigger tiles"
            )
        base = self.red.start // self.red.step
        entries = []
        for t, blocks in enumerate(self.spec.red_blocks):
            for n, kb in enumerate(blocks):
                first, last = n == 0, n == len(blocks) - 1
                entries.append((t << _TILE_SHIFT) | ((kb - base) << 2) | (first << 1) | last)
        return np.array(entries, np.int32)

    def _first_depth(self) -> int:
        """The reduction counter of the very first step."""
        if self.spec.red_blocks is None or self.red is None:
            return 0
        return self.spec.red_blocks[0][0] - self.red.start // self.red.step

    def _streamed(self, op: Operand) -> bool:
        """Does the operand's block change on every step? (It's tiled by the reduction loop.)"""
        if self.red is None or self.red.var not in op.block.loops:
            return False
        return self.step_table or self.K > 1

    # ------------------------------------------------------------------ DMA helpers

    def _slices(self, block: Block, tile: str, depth: str) -> str:
        """The HBM window of a block: `pl.ds` per tiled dim, a plain index per squeezed dim."""
        parts = []
        for d, (var, size, index) in enumerate(zip(block.loops, block.shape, block.index, strict=True)):
            if var is None:
                start = str(int(index) * size)
            elif self.red is not None and var == self.red.var:
                base = self.red.start // self.red.step
                start = f"({depth} + {base}) * {size}" if base else f"{depth} * {size}"
            else:
                start = (
                    f"{self.block_tables[var]}[{tile}] * {size}" if size != 1 else f"{self.block_tables[var]}[{tile}]"
                )
            parts.append(start if block.is_squeezed(d) else f"pl.ds({start}, {size})")
        return ", ".join(parts)

    def _copy_fn(self, op: Operand) -> str:
        return f"_{op.ref}_copy"

    def _helpers(self) -> list[str]:
        spec = self.spec
        lines = []
        for op in spec.operands:
            lines += [
                f"def {self._copy_fn(op)}(tile, depth, slot):  # {op.param}[{op.block.text}]",
                f"    src = {self.hbm[op.param]}.at[{self._slices(op.block, 'tile', 'depth')}]",
                f"    return pltpu.make_async_copy(src, {self.slots[op.ref]}.at[slot], {self.sems[op.ref]}.at[slot])",
                "",
            ]
        lines += [
            f"def _{spec.out_ref}_copy(tile, slot):  # {self.nest.out_var}[{spec.out_block.text}]",
            f"    dst = {self.out_hbm}.at[{self._slices(spec.out_block, 'tile', '0')}]",
            f"    return pltpu.make_async_copy({self.out_slots}.at[slot], dst, {self.out_sems}.at[slot])",
            "",
        ]
        return lines

    def _tile_changed(self, op: Operand) -> str:
        """Scalar condition: the operand's block differs between this tile and the next."""
        v = self.v
        parts = [
            f"({self.block_tables[var]}[{v['safe_tile']}] != {self.block_tables[var]}[{v['tile']}])"
            for var in dict.fromkeys(op.block.loops)
            if var is not None and (self.red is None or var != self.red.var)
        ]
        if not parts:
            return "jnp.bool_(False)"
        cond = " | ".join(parts)
        return f"{v['last']} & ({cond})" if self.step_table or self.K > 1 else cond

    # ------------------------------------------------------------------ kernel

    def _loop_values(self, used: set[str]) -> list[str]:
        v, lines = self.v, []
        for lp in self.nest.loops:
            if lp.var not in used:
                continue
            lb = self.spec.loops[lp.var]
            mul = f" * {lb.step}" if lb.step != 1 else ""
            if self.red is not None and lp.var == self.red.var:
                lines.append(f"{lp.var} = {v['depth']}{mul}" + (f" + {lb.start}" if lb.start else ""))
            else:
                lines.append(f"{lp.var} = {self.block_tables[lp.var]}[{v['tile']}]{mul}")
        return lines

    def _store(self, epilogue: list[str]) -> list[str]:
        """Write the finished output tile into a VMEM slot and start copying it to HBM."""
        v, out = self.v, self.spec.out_ref
        return [
            f"{v['out_slot']} = {v['tile']} & 1",
            "",
            f"@pl.when({v['tile']} >= 2)",
            "def _reuse_out_slot():  # the copy that last used this slot must have left",
            f"    _{out}_copy({v['tile']} - 2, {v['out_slot']}).wait()",
            "",
            f"{out} = {self.out_slots}.at[{v['out_slot']}]",
            *epilogue,
            f"_{out}_copy({v['tile']}, {v['out_slot']}).start()",
        ]

    def _step_body(self) -> list[str]:
        v, K, fill = self.v, self.K, self.fill
        temps, inits, step, epilogue = self.body()
        used = self.names_used(temps + inits + step + epilogue)

        lines = []
        if self.step_table:
            # Decode this step and the next from the step table: tile, reduction block, first/last flags.
            sh, mask = _TILE_SHIFT, _MAX_RED_BLOCKS - 1
            lines += [
                f"{v['entry']} = {self.steps_name}[{v['step']}]",
                f"{v['tile']} = {v['entry']} >> {sh}",
                f"{v['depth']} = ({v['entry']} >> 2) & {mask}",
                f"{v['first']} = (({v['entry']} >> 1) & 1) == 1",
                f"{v['last']} = ({v['entry']} & 1) == 1",
                f"{v['next_entry']} = {self.steps_name}[jnp.minimum({v['step']} + 1, {self.n_steps - 1})]",
                f"{v['next_tile']} = {v['next_entry']} >> {sh}",
                f"{v['next_depth']} = ({v['next_entry']} >> 2) & {mask}",
            ]
        elif K > 1:
            lines += [
                f"{v['last']} = {v['depth']} == {K - 1}",
                f"{v['next_tile']} = jnp.where({v['last']}, {v['tile']} + 1, {v['tile']})",
                f"{v['next_depth']} = jnp.where({v['last']}, 0, {v['depth']} + 1)",
            ]
        else:
            lines += [f"{v['next_tile']} = {v['tile']} + 1", f"{v['next_depth']} = {v['depth']}"]
        if self.streamed:
            lines.append(f"{v['slot']} = {v['step']} & 1")
        if self.per_tile:
            lines.append(f"{v['safe_tile']} = jnp.minimum({v['next_tile']}, {self.n_tiles - 1})")
            lines += [f"{self.changed[op.ref]} = {self._tile_changed(op)}" for op in self.per_tile]

        # 1. Start fetching the next step's blocks into the other slot.
        prefetch = [
            f"{self._copy_fn(op)}({v['next_tile']}, {v['next_depth']}, 1 - {v['slot']}).start()" for op in self.streamed
        ]
        for op in self.per_tile:
            prefetch += [
                f"@pl.when({self.changed[op.ref]})",
                f"def _fetch_{op.ref}():",
                f"    next_slot = 1 - ({self.version[op.ref]} & 1)",
                f"    {self._copy_fn(op)}({v['next_tile']}, {v['next_depth']}, next_slot).start()",
            ]
        if prefetch:
            lines += [
                "",
                f"{v['has_next']} = {v['step']} + 1 < {self.n_steps}",
                "",
                f"@pl.when({v['has_next']})",
                "def _prefetch_next_step():",
                *indent(prefetch),
            ]

        # 2. Wait for this step's blocks.
        lines.append("")
        lines += [f"{self._copy_fn(op)}({v['tile']}, {v['depth']}, {v['slot']}).wait()" for op in self.streamed]
        for op in self.per_tile:
            lines += [
                f"@pl.when({self.pending[op.ref]} == 1)",
                f"def _wait_{op.ref}():",
                f"    {self._copy_fn(op)}({v['tile']}, {v['depth']}, {self.version[op.ref]} & 1).wait()",
            ]
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{v['slot']}]" for op in self.streamed]
        lines += [f"{op.ref} = {self.slots[op.ref]}.at[{self.version[op.ref]} & 1]" for op in self.per_tile]

        # 3. The user's tile body.
        lines += self._loop_values(used) + temps
        starts_tile = v["first"] if self.step_table else f"{v['depth']} == 0"
        if fill:
            issue = self.fill_issue(v["tile"])
            many = self.step_table or K > 1
            lines += ["", f"@pl.when({starts_tile})", "def _fill():", *indent(issue)] if many else ["", *issue]
        if self.step_table or K > 1:
            lines += ["", f"@pl.when({starts_tile})", "def _init():", *indent(inits), "", *step, ""]
            lines += [f"@pl.when({v['last']})", "def _store():", *indent(self._store(epilogue))]
        else:
            lines += [*inits, *step, *self._store(epilogue)]

        carry = [] if self.step_table else [v["next_tile"], v["next_depth"]]
        for op in self.per_tile:
            carry += [
                f"{self.version[op.ref]} + {self.changed[op.ref]}.astype(jnp.int32)",
                f"{self.changed[op.ref]}.astype(jnp.int32)",
            ]
        if carry:
            lines += ["", f"return ({', '.join(carry)},)"]
        return lines

    def kernel(self, name: str) -> list[str]:
        spec, v, fill = self.spec, self.v, self.fill
        args = (
            list(self.tables)
            + list(self.hbm.values())
            + ([self.out_init] if self.out_init else [])
            + [self.out_hbm]
            + [s for op in spec.operands for s in (self.slots[op.ref], self.sems[op.ref])]
            + [self.out_slots, self.out_sems]
            + [acc.ref for acc in spec.accumulators]
            + ([fill.ref, fill.sem] if fill else [])
        )
        carry_names = [] if self.step_table else [v["tile"], v["depth"]]
        init = [] if self.step_table else ["jnp.int32(0)", "jnp.int32(0)"]
        for op in self.per_tile:
            carry_names += [self.version[op.ref], self.pending[op.ref]]
            init += ["jnp.int32(0)", "jnp.int32(1)"]
        if carry_names:
            loop = [
                f"@pl.loop(0, {self.n_steps}, init_carry=({', '.join(init)},))",
                f"def _steps({v['step']}, {v['carry']}):",
                f"    {', '.join(carry_names)}, = {v['carry']}",
            ]
        else:
            loop = [f"@pl.loop(0, {self.n_steps})", f"def _steps({v['step']}):"]

        lines = [f"def {name}({', '.join(args)}):"]
        lines += indent(self._helpers())
        if fill:
            lines += indent([*self.fill_helpers(self.out_hbm), "", *self.fill_tile_init(), ""])
        lines += indent(
            [
                "# Fetch the first step's blocks.",
                *[f"{self._copy_fn(op)}(0, {self._first_depth()}, 0).start()" for op in spec.operands],
                "",
                *loop,
                *indent(self._step_body()),
                "",
                "# Wait for the output copies still in flight.",
                *[f"_{spec.out_ref}_copy({t}, {t % 2}).wait()" for t in range(max(0, self.n_tiles - 2), self.n_tiles)],
            ]
        )
        if fill:
            lines += indent(self.fill_drain())
        return lines

    # ------------------------------------------------------------------ pallas_call wrapper

    def wrapper(self, kernel: str) -> list[str]:
        nest, spec, mask, opts = self.nest, self.spec, self.mask, self.options
        in_specs = ["pl.BlockSpec(memory_space=pltpu.SMEM)"] * len(self.tables)
        in_specs += ["pl.BlockSpec(memory_space=pl.ANY)"] * (len(self.hbm) + (1 if self.out_init else 0))
        scratch = []
        for op in spec.operands:
            scratch += [
                f"pltpu.VMEM({(2, *op.block.kernel_shape)}, {dtype_src(op.dtype)})",
                "pltpu.SemaphoreType.DMA((2,))",
            ]
        scratch += [
            f"pltpu.VMEM({(2, *spec.out_block.kernel_shape)}, {dtype_src(spec.out.dtype)})",
            "pltpu.SemaphoreType.DMA((2,))",
        ]
        scratch += [f"pltpu.VMEM({acc.scratch.shape}, {dtype_src(acc.scratch.dtype)})" for acc in spec.accumulators]
        if self.fill:
            scratch += [
                f"pltpu.VMEM({spec.out_block.kernel_shape}, {dtype_src(spec.out.dtype)})",
                "pltpu.SemaphoreType.DMA(())",
            ]

        lines = [f"def {nest.fn_name}_pallas({', '.join(nest.params)}, *, interpret=False):"]
        tiles = f"{self.n_tiles} output tiles" + (
            f" ({mask.total} without the tile mask `{ast.unparse(nest.guard)}`)" if mask else ""
        )
        if self.step_table:
            red_mask = ast.unparse(nest.red_guard)
            lines.append(
                f"    # No grid: one loop over the {self.n_steps} steps of {tiles} that the reduction mask "
                f"`{red_mask}` keeps (of {self.n_tiles * self.K}), double-buffered by hand."
            )
        else:
            lines.append(f"    # No grid: one loop over {tiles} x {self.K} reduction steps, double-buffered by hand.")
        args = [t.upper() for t in self.tables] + list(self.hbm)
        if self.out_init:
            lines += [
                "    # Masked-out tiles keep the initial value through the alias below.",
                f"    {self.out_init} = {ast.unparse(nest.out_alloc)}",
            ]
            args.append(self.out_init)
        lines += [
            "    return pl.pallas_call(",
            f"        {kernel},",
            f"        out_shape=jax.ShapeDtypeStruct({tuple(spec.out.shape)}, {dtype_src(spec.out.dtype)}),",
            "        in_specs=[" + ", ".join(in_specs) + "],",
            "        out_specs=pl.BlockSpec(memory_space=pl.ANY),",
            "        scratch_shapes=[",
            *[f"            {s}," for s in scratch],
            "        ],",
        ]
        if self.out_init:
            lines.append(f"        input_output_aliases={{{len(args) - 1}: 0}},")
        if opts.vmem_limit_bytes:
            lines.append(f"        compiler_params=pltpu.CompilerParams(vmem_limit_bytes={opts.vmem_limit_bytes}),")
        lines += [
            "        interpret=interpret,",
            f"        name={nest.fn_name!r},",
            f"    )({', '.join(args)})",
        ]
        return lines
