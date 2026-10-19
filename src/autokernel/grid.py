"""The grid backend: a Pallas grid with BlockSpecs, where Pallas's pipeliner moves the data.

Outer loops become grid axes and slices become BlockSpecs; with a tile mask, the parallel loops collapse
into one grid axis over the active tiles, read from scalar-prefetch tables. Every grid step pays Pallas's
pipeline bookkeeping (roughly 15-25 ns on a v5e), so the manual backend is faster for small tiles; this
one keeps "parallel" grid axes that chips with two TensorCores can split.
"""

from __future__ import annotations

import ast

import numpy as np

from .analysis import Block, Specialization
from .emit import Emitter, Options, dtype_src, indent, tuple_src
from .errors import AutoKernelError

__all__ = ["GridEmitter"]


def _block_shape_src(block: Block) -> str:
    """BlockSpec block shape: squeezed (integer-indexed) dims are `None`."""
    return tuple_src(["None" if block.is_squeezed(d) else str(n) for d, n in enumerate(block.shape)])


class GridEmitter(Emitter):
    pipeline = "grid"

    def __init__(self, spec: Specialization, options: Options):
        super().__init__(spec, options)
        if spec.red_blocks is not None:
            raise AutoKernelError(
                f"{self.nest.fn_name}: masks on the reduction loop need pipeline='manual' (the default); a Pallas "
                "grid can't skip individual reduction steps"
            )
        self.out_in = spec.names.fresh(f"{self.nest.out_var}_in") if self.mask else None  # aliased HBM output
        if self.mask is not None:
            for n, table in enumerate(self.mask.tables.values()):
                self.tables[table] = np.array([blocks[n] for blocks in self.mask.active], np.int32)
        self.tables.update(self._fill_tables())

    def grid(self) -> tuple[int, ...]:
        nest, spec, red, mask = self.nest, self.spec, self.red, self.mask
        if mask is None:
            return tuple(spec.loops[lp.var].grid for lp in nest.loops)
        return (len(mask.active),) + ((red.grid,) if red else ())

    def steps(self) -> int:
        return int(np.prod(self.grid()))

    def _index_map_args(self) -> list[str]:
        if self.mask is None:
            return [lp.var for lp in self.nest.loops]
        return [self.mask.grid_var] + ([self.red.var] if self.red else []) + list(self.tables)

    def _dimension_semantics(self) -> tuple[str, ...]:
        red = ("arbitrary",) if self.red else ()
        if self.mask is None:
            return tuple("parallel" for _ in self.nest.parallel) + red
        # Fill DMAs start on some grid steps and are drained on the last one, so steps run in order.
        return ("arbitrary" if self.fill else "parallel", *red)

    def _loop_values(self, used: set[str]) -> list[str]:
        """Loop variables the kernel still reads (not just in slices) get their value from the grid."""
        lines = []
        for lp in self.nest.loops:
            if lp.var not in used:
                continue
            lb = self.spec.loops[lp.var]
            mul = f" * {lb.step}" if lb.step != 1 else ""
            if self.mask is not None and lp.var in self.mask.tables:
                lines.append(f"{lp.var} = {self.mask.tables[lp.var]}[pl.program_id(0)]{mul}")
            else:
                lines.append(f"{lp.var} = pl.program_id({lb.axis}){mul}" + (f" + {lb.start}" if lb.start else ""))
        return lines

    def kernel(self, name: str) -> list[str]:
        spec, red, fill = self.spec, self.red, self.fill
        temps, inits, step, epilogue = self.body()
        used = self.names_used(temps + inits + step + epilogue)
        args = (
            list(self.tables)
            + [op.ref for op in spec.operands]
            + ([self.out_in] if self.out_in else [])
            + [spec.out_ref]
            + [acc.ref for acc in spec.accumulators]
            + ([fill.ref, fill.sem] if fill else [])
        )
        lines = [f"def {name}({', '.join(args)}):"]
        lines += indent(self._loop_values(used) + temps)
        if fill:
            first = "(pl.program_id(0) == 0)" + (" & (pl.program_id(1) == 0)" if red else "")
            lines += ["", *indent(self.fill_helpers(self.out_in)), ""]
            lines += [f"    @pl.when({first})", "    def _make_fill_tile():", *indent(self.fill_tile_init(), 2), ""]
            issue = self.fill_issue("pl.program_id(0)")
            if red:
                lines += [f"    @pl.when(pl.program_id({red.axis}) == 0)", "    def _fill():", *indent(issue, 2)]
            else:
                lines += indent(issue)
        if red:
            ax = red.axis
            lines += ["", f"    @pl.when(pl.program_id({ax}) == 0)", "    def _init():", *indent(inits, 2), ""]
            lines += indent(step)
            lines += ["", f"    @pl.when(pl.program_id({ax}) == pl.num_programs({ax}) - 1)", "    def _store():"]
            lines += indent(epilogue, 2)
        else:
            lines += indent(epilogue)
        if fill:
            last = "(pl.program_id(0) == pl.num_programs(0) - 1)" + (
                " & (pl.program_id(1) == pl.num_programs(1) - 1)" if red else ""
            )
            lines += ["", f"    @pl.when({last})", "    def _wait_fill_copies():", *indent(self.fill_drain(), 2)]
        return lines

    def wrapper(self, kernel: str) -> list[str]:
        nest, spec, mask, fill, opts = self.nest, self.spec, self.mask, self.fill, self.options
        lam = f"lambda {', '.join(self._index_map_args())}:"
        specs = ["in_specs=["]
        specs += [
            f"    pl.BlockSpec({_block_shape_src(op.block)}, {lam} {tuple_src(op.block.index)}),"
            f"  # {op.param}[{op.block.text}]"
            for op in spec.operands
        ]
        if mask:
            specs.append(f"    pl.BlockSpec(memory_space=pl.ANY),  # {self.out_in}: aliased to the output")
        specs += [
            "],",
            f"out_specs=pl.BlockSpec({_block_shape_src(spec.out_block)}, {lam} {tuple_src(spec.out_block.index)}),"
            f"  # {nest.out_var}[{spec.out_block.text}]",
        ]
        scratch = [f"pltpu.VMEM({acc.scratch.shape}, {dtype_src(acc.scratch.dtype)})" for acc in spec.accumulators]
        if fill:
            scratch += [
                f"pltpu.VMEM({spec.out_block.kernel_shape}, {dtype_src(spec.out.dtype)})",
                "pltpu.SemaphoreType.DMA(())",
            ]
        if scratch:
            specs.append(f"scratch_shapes=[{', '.join(scratch)}],")
        params = f"dimension_semantics={self._dimension_semantics()!r}"
        if opts.vmem_limit_bytes:
            params += f", vmem_limit_bytes={opts.vmem_limit_bytes}"

        grid = self.grid()
        lines = [f"def {nest.fn_name}_pallas({', '.join(nest.params)}, *, interpret=False):"]
        args = [op.param for op in spec.operands]
        if mask:
            lines.append(
                f"    # Tile mask `{ast.unparse(nest.guard)}` keeps {len(mask.active)} of {mask.total} "
                "output tiles; the grid walks only those."
            )
            if fill:
                lines += [
                    "    # The kernel writes every other tile too (fill DMAs), so the output starts uninitialised.",
                    f"    {self.out_in} = jax.lax.empty({tuple(spec.out.shape)}, {dtype_src(spec.out.dtype)})",
                ]
            else:
                lines += [
                    "    # The other tiles keep the initial value through the alias below.",
                    f"    {self.out_in} = {ast.unparse(nest.out_alloc)}",
                ]
            args = [t.upper() for t in self.tables] + args + [self.out_in]
        lines += [
            "    return pl.pallas_call(",
            f"        {kernel},",
            f"        out_shape=jax.ShapeDtypeStruct({tuple(spec.out.shape)}, {dtype_src(spec.out.dtype)}),",
        ]
        if mask:
            lines += [
                "        grid_spec=pltpu.PrefetchScalarGridSpec(",
                f"            num_scalar_prefetch={len(self.tables)},",
                f"            grid={grid},",
                *indent(specs, 3),
                "        ),",
                f"        input_output_aliases={{{len(args) - 1}: 0}},",
            ]
        else:
            lines += [f"        grid={grid},", *indent(specs, 2)]
        lines += [
            f"        compiler_params=pltpu.CompilerParams({params}),",
            "        interpret=interpret,",
            f"        name={nest.fn_name!r},",
            f"    )({', '.join(args)})",
        ]
        return lines
