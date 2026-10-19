"""Generate Pallas TPU source from a `Specialization`, then compile it into a callable.

The generated code is plain, readable Pallas: a kernel function plus a wrapper that calls
`pl.pallas_call`. `Lowered.source` holds it so users can inspect exactly what runs.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import hardware
from .analysis import Block, Operand, Specialization, specialize
from .errors import AutoKernelError
from .frontend import LoopNest

__all__ = ["FILL_DMA_DEPTH", "MATMUL_PRECISIONS", "Lowered", "Options", "lower"]

# Mosaic's default f32 matmul is a single bf16 pass; HIGHEST does the multi-pass f32 matmul the
# Python code actually describes.
MATMUL_PRECISIONS = {"highest": "jax.lax.Precision.HIGHEST", "default": "None"}

# Constant tiles are written with at most this many fill DMAs in flight.
FILL_DMA_DEPTH = 4

_PRELUDE = """\
def _mxu_dot(x, y):
    # `@` lowers to an MXU matmul that accumulates in f32 (or i32 for integer inputs).
    acc = jnp.int32 if jnp.issubdtype(x.dtype, jnp.integer) else jnp.float32
    # bf16/int8 operands are already exact on the MXU; only f32 operands take a precision.
    precision = {precision} if x.dtype == jnp.float32 or y.dtype == jnp.float32 else None
    return jnp.dot(x, y, preferred_element_type=acc, precision=precision)
"""


@dataclass(frozen=True)
class Options:
    """Code generation options; see `autokernel.tpu` for what each one means."""

    check_tiling: bool = True
    vmem_limit_bytes: int | None = None
    matmul_precision: str = "highest"
    background_fill: bool = True

    def __post_init__(self) -> None:
        if self.matmul_precision not in MATMUL_PRECISIONS:
            raise AutoKernelError(
                f"matmul_precision must be one of {sorted(MATMUL_PRECISIONS)}, got {self.matmul_precision!r}"
            )


@dataclass(frozen=True)
class Lowered:
    """A kernel specialised to one set of input shapes/dtypes."""

    source: str  # the generated Pallas code
    entry: Callable[..., jax.Array]  # entry(*args, interpret=...) -> output
    grid: tuple[int, ...]
    operands: list[Operand]
    out: jax.ShapeDtypeStruct
    out_block: Block
    scratch: dict[str, jax.ShapeDtypeStruct]  # accumulator name -> VMEM scratch
    vmem_bytes: int  # estimated VMEM use, double-buffered blocks included
    active_tiles: int | None = None  # with a tile mask: output tiles actually computed
    total_tiles: int | None = None
    fill: str | None = None  # how masked-out tiles get their value: "background" DMAs or "prefill"


def lower(
    nest: LoopNest, fn: Callable, specs: Sequence[jax.ShapeDtypeStruct], options: Options | None = None
) -> Lowered:
    """Specialise `nest` to `specs`, generate the Pallas kernel and compile its source."""
    return _Emitter(specialize(nest, fn, specs), options or Options()).lower()


@dataclass
class _FillPlan:
    """Masked-out tiles written by the kernel itself, from one VMEM tile, with async DMAs."""

    value: str  # Python source for the fill scalar
    tables: dict[str, str]  # parallel loop var -> scalar-prefetch table of the constant tiles' blocks
    start: str  # table: grid tile t issues fill copies [start[t], start[t + 1])
    ref: str
    sem: str


def _dtype_src(dtype) -> str:
    return f"jnp.{np.dtype(dtype).name}"


def _tuple_src(items: Sequence[str]) -> str:
    return f"({', '.join(items)}{',' if len(items) == 1 else ''})"


def _indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line if line else line for line in lines]


class _Emitter:
    def __init__(self, spec: Specialization, options: Options):
        self.spec, self.options, self.nest = spec, options, spec.nest
        self.red = spec.reduction
        self.mask = spec.mask
        self.out_in = spec.names.fresh(f"{self.nest.out_var}_in") if self.mask else None  # aliased HBM output
        self.fill = self._plan_fill()
        self.prefetch = self._prefetch_tables()

    # ------------------------------------------------------------------ planning

    def _plan_fill(self) -> _FillPlan | None:
        """Write the skipped tiles from the kernel when their value is a known constant.

        This overlaps those writes with the matmuls and lets the output start uninitialised, instead of
        filling the whole output in a separate pass before the kernel runs.
        """
        mask = self.mask
        if mask is None or not mask.constant or mask.fill is None or not self.options.background_fill:
            return None
        names = self.spec.names
        value = mask.fill
        return _FillPlan(
            value=repr(value) if np.isfinite(value) else f"float('{value}')",
            tables={lp.var: names.fresh(f"{lp.var}_fill_blocks") for lp in self.nest.parallel},
            start=names.fresh("fill_start"),
            ref=names.fresh("fill_ref"),
            sem=names.fresh("fill_sem"),
        )

    def _prefetch_tables(self) -> dict[str, np.ndarray]:
        """Scalar-prefetch tables (SMEM), in kernel-argument order."""
        tables: dict[str, np.ndarray] = {}
        if self.mask is None:
            return tables
        for n, table in enumerate(self.mask.tables.values()):
            tables[table] = np.array([blocks[n] for blocks in self.mask.active], np.int32)
        if self.fill is not None:
            for n, table in enumerate(self.fill.tables.values()):
                tables[table] = np.array([blocks[n] for blocks in self.mask.constant], np.int32)
            n_tiles, n_fill = len(self.mask.active), len(self.mask.constant)
            tables[self.fill.start] = np.array([t * n_fill // n_tiles for t in range(n_tiles + 1)], np.int32)
        return tables

    # ------------------------------------------------------------------ user code -> kernel code

    def _rewrite(self, node: ast.AST, refs: dict[str, str]) -> ast.AST:
        """Inputs become their tile refs, accumulators their scratch refs, `@` an MXU matmul."""
        spec, params = self.spec, self.nest.params

        class Rewriter(ast.NodeTransformer):
            def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
                if isinstance(n.value, ast.Name) and n.value.id in params:
                    ref = spec.operand(n).ref
                    return ast.copy_location(ast.parse(f"{ref}[...]", mode="eval").body, n)
                return self.generic_visit(n)

            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in refs:
                    return ast.copy_location(
                        ast.Subscript(ast.Name(refs[n.id], ast.Load()), ast.Constant(...), n.ctx), n
                    )
                return n

            def visit_BinOp(self, n: ast.BinOp) -> ast.AST:
                self.generic_visit(n)
                if isinstance(n.op, ast.MatMult):
                    return ast.copy_location(ast.Call(ast.Name("_mxu_dot", ast.Load()), [n.left, n.right], []), n)
                return n

        return Rewriter().visit(copy.deepcopy(node))

    def _stmt(self, stmt: ast.stmt, refs: dict[str, str], cast: dict[str, str] | None = None) -> str:
        out = self._rewrite(stmt, refs)
        assert isinstance(out, (ast.Assign, ast.AugAssign))
        target = out.targets[0] if isinstance(out, ast.Assign) else out.target
        if cast and isinstance(out, ast.Assign) and isinstance(target, ast.Subscript) and target.value.id in cast:  # type: ignore[attr-defined]
            out.value = ast.parse(f"({ast.unparse(out.value)}).astype({cast[target.value.id]})", mode="eval").body  # type: ignore[attr-defined]
        return ast.unparse(out)

    def _body(self) -> tuple[list[str], list[str], list[str], list[str]]:
        """(temporaries, accumulator inits, reduction step, epilogue) in the user's order."""
        nest, spec = self.nest, self.spec
        refs = {acc.name: acc.ref for acc in spec.accumulators}
        widened = {
            acc.ref: _dtype_src(acc.scratch.dtype) for acc in spec.accumulators if acc.init.dtype != acc.scratch.dtype
        }
        temps, inits = [], []
        for s in nest.pre:
            target = s.targets[0].id if isinstance(s, ast.Assign) else s.target.id  # type: ignore[attr-defined]
            if target in refs:
                inits.append(self._stmt(s, refs, widened))
            elif any(isinstance(n, ast.Name) and n.id in refs for n in ast.walk(s)):
                raise nest.error(s, f"`{target}` reads an accumulator before the reduction loop")
            else:
                temps.append(self._stmt(s, {}))
        step = [self._stmt(s, refs) for s in nest.body]
        epilogue = [self._stmt(s, refs) for s in nest.post]
        value = self._rewrite(nest.store_value, refs)
        value_src = ast.unparse(value)
        if not isinstance(value, (ast.Name, ast.Subscript, ast.Call, ast.Attribute)):
            value_src = f"({value_src})"
        epilogue.append(f"{spec.out_ref}[...] = {value_src}.astype({spec.out_ref}.dtype)")
        return temps, inits, step, epilogue

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

    # ------------------------------------------------------------------ kernel

    def _kernel(self, name: str) -> list[str]:
        spec, red, fill = self.spec, self.red, self.fill
        temps, inits, step, epilogue = self._body()
        used = {
            n.id for n in ast.walk(ast.parse("\n".join(temps + inits + step + epilogue))) if isinstance(n, ast.Name)
        }
        args = (
            list(self.prefetch)
            + [op.ref for op in spec.operands]
            + ([self.out_in] if self.out_in else [])
            + [spec.out_ref]
            + [acc.ref for acc in spec.accumulators]
            + ([fill.ref, fill.sem] if fill else [])
        )

        lines = [f"def {name}({', '.join(args)}):"]
        lines += _indent(self._loop_values(used) + temps)
        if fill:
            lines += _indent(self._fill_start())
        if red:
            ax = red.axis
            lines += ["", f"    @pl.when(pl.program_id({ax}) == 0)", "    def _init():", *_indent(inits, 2), ""]
            lines += _indent(step)
            lines += ["", f"    @pl.when(pl.program_id({ax}) == pl.num_programs({ax}) - 1)", "    def _store():"]
            lines += _indent(epilogue, 2)
        else:
            lines += _indent(epilogue)
        if fill:
            lines += _indent(self._fill_drain())
        return lines

    def _fill_start(self) -> list[str]:
        fill, spec, red = self.fill, self.spec, self.red
        assert fill is not None and self.mask is not None
        block = spec.out_block
        dst = [
            f"pl.ds({fill.tables[var]}[c] * {size}, {size})" if var else f"pl.ds({int(i) * size}, {size})"
            for var, size, i in zip(block.loops, block.shape, block.index, strict=True)
        ]
        first = "(pl.program_id(0) == 0)" + (" & (pl.program_id(1) == 0)" if red else "")
        lines = [
            "",
            f"# {len(self.mask.constant)} masked-out tiles are the constant {fill.value}: write them from one VMEM",
            f"# tile with async DMAs spread across the grid ({FILL_DMA_DEPTH} in flight at most), behind the matmuls.",
            "def _fill_copy(c):",
            f"    return pltpu.make_async_copy({fill.ref}, {self.out_in}.at[{', '.join(dst)}], {fill.sem})",
            "",
            f"@pl.when({first})",
            "def _make_fill_tile():",
            f"    {fill.ref}[...] = jnp.full({fill.ref}.shape, {fill.value}, {fill.ref}.dtype)",
            "",
        ]
        copies = [
            f"@pl.loop({fill.start}[pl.program_id(0)], {fill.start}[pl.program_id(0) + 1])",
            "def _start_fill_copies(c):",
            f"    @pl.when(c >= {FILL_DMA_DEPTH})",
            "    def _limit_in_flight():",
            "        _fill_copy(0).wait()",
            "    _fill_copy(c).start()",
        ]
        if red:
            return [*lines, f"@pl.when(pl.program_id({red.axis}) == 0)", "def _fill():", *_indent(copies)]
        return lines + copies

    def _fill_drain(self) -> list[str]:
        assert self.mask is not None
        last = "(pl.program_id(0) == pl.num_programs(0) - 1)" + (
            " & (pl.program_id(1) == pl.num_programs(1) - 1)" if self.red else ""
        )
        in_flight = min(len(self.mask.constant), FILL_DMA_DEPTH)
        return [
            "",
            f"@pl.when({last})",
            "def _wait_fill_copies():",
            f"    for _ in range({in_flight}):",
            "        _fill_copy(0).wait()",
        ]

    # ------------------------------------------------------------------ pallas_call wrapper

    def _grid(self) -> tuple[tuple[int, ...], list[str], tuple[str, ...]]:
        """(grid, index-map arguments, dimension semantics)."""
        nest, spec, red, mask = self.nest, self.spec, self.red, self.mask
        if mask is None:
            return (
                tuple(spec.loops[lp.var].grid for lp in nest.loops),
                [lp.var for lp in nest.loops],
                tuple("parallel" for _ in nest.parallel) + (("arbitrary",) if red else ()),
            )
        args = [mask.grid_var] + ([red.var] if red else []) + list(self.prefetch)
        # Fill DMAs start on some grid steps and are drained on the last one, so steps run in order.
        tile_axis = "arbitrary" if self.fill else "parallel"
        return (
            (len(mask.active),) + ((red.grid,) if red else ()),
            args,
            (tile_axis,) + (("arbitrary",) if red else ()),
        )

    def _wrapper(self, kernel: str, grid: tuple[int, ...], lam_args: list[str], sems: tuple[str, ...]) -> list[str]:
        nest, spec, mask, fill, opts = self.nest, self.spec, self.mask, self.fill, self.options
        lam = f"lambda {', '.join(lam_args)}:"
        specs = ["in_specs=["]
        specs += [
            f"    pl.BlockSpec({op.block.shape}, {lam} {_tuple_src(op.block.index)}),  # {op.param}[{op.block.text}]"
            for op in spec.operands
        ]
        if mask:
            specs.append(f"    pl.BlockSpec(memory_space=pl.ANY),  # {self.out_in}: aliased to the output")
        specs += [
            "],",
            f"out_specs=pl.BlockSpec({spec.out_block.shape}, {lam} {_tuple_src(spec.out_block.index)}),"
            f"  # {nest.out_var}[{spec.out_block.text}]",
        ]
        scratch = [f"pltpu.VMEM({acc.scratch.shape}, {_dtype_src(acc.scratch.dtype)})" for acc in spec.accumulators]
        if fill:
            scratch += [
                f"pltpu.VMEM({spec.out_block.shape}, {_dtype_src(spec.out.dtype)})",
                "pltpu.SemaphoreType.DMA(())",
            ]
        if scratch:
            specs.append(f"scratch_shapes=[{', '.join(scratch)}],")
        params = f"dimension_semantics={sems!r}"
        if opts.vmem_limit_bytes:
            params += f", vmem_limit_bytes={opts.vmem_limit_bytes}"

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
                    f"    {self.out_in} = jax.lax.empty({tuple(spec.out.shape)}, {_dtype_src(spec.out.dtype)})",
                ]
            else:
                lines += [
                    "    # The other tiles keep the initial value through the alias below.",
                    f"    {self.out_in} = {ast.unparse(nest.out_alloc)}",
                ]
            args = [t.upper() for t in self.prefetch] + args + [self.out_in]
        lines += [
            "    return pl.pallas_call(",
            f"        {kernel},",
            f"        out_shape=jax.ShapeDtypeStruct({tuple(spec.out.shape)}, {_dtype_src(spec.out.dtype)}),",
        ]
        if mask:
            lines += [
                "        grid_spec=pltpu.PrefetchScalarGridSpec(",
                f"            num_scalar_prefetch={len(self.prefetch)},",
                f"            grid={grid},",
                *_indent(specs, 3),
                "        ),",
                f"        input_output_aliases={{{len(args) - 1}: 0}},",
            ]
        else:
            lines += [f"        grid={grid},", *_indent(specs, 2)]
        lines += [
            f"        compiler_params=pltpu.CompilerParams({params}),",
            "        interpret=interpret,",
            f"        name={nest.fn_name!r},",
            f"    )({', '.join(args)})",
        ]
        return lines

    # ------------------------------------------------------------------ validation and assembly

    def _check_hardware(self) -> int:
        spec, fill = self.spec, self.fill
        problems = []
        for op in spec.operands:
            problems += hardware.check_block(
                f"input `{op.param}[{op.block.text}]` block {op.block.shape}", op.block.shape, op.full_shape, op.dtype
            )
        problems += hardware.check_block(
            f"output `{self.nest.out_var}[{spec.out_block.text}]` block {spec.out_block.shape}",
            spec.out_block.shape,
            tuple(spec.out.shape),
            spec.out.dtype,
        )
        if problems and self.options.check_tiling:
            raise AutoKernelError(
                f"{self.nest.fn_name}: tiles don't satisfy TPU (Mosaic) layout rules:\n  - "
                + "\n  - ".join(problems)
                + "\nPick tile sizes that fit, or pass check_tiling=False to try anyway."
            )
        pipelined = [(op.block.shape, op.dtype) for op in spec.operands] + [(spec.out_block.shape, spec.out.dtype)]
        scratch = [(acc.scratch.shape, acc.scratch.dtype) for acc in spec.accumulators]
        if fill:
            scratch.append((spec.out_block.shape, spec.out.dtype))
        return hardware.check_vmem(pipelined, scratch, self.options.vmem_limit_bytes)

    def lower(self) -> Lowered:
        spec, nest, mask = self.spec, self.nest, self.mask
        kernel_name = spec.names.fresh(f"{nest.fn_name}_kernel")
        kernel = self._kernel(kernel_name)
        vmem = self._check_hardware()
        grid, lam_args, sems = self._grid()
        wrapper = self._wrapper(kernel_name, grid, lam_args, sems)

        header = (
            f"# Generated by autokernel from `{nest.fn_name}` ({spec.filename}:{nest.first_line}) for "
            + ", ".join(
                f"{p}: {_dtype_src(s.dtype)}{tuple(s.shape)}" for p, s in zip(nest.params, spec.specs, strict=True)
            )
        )
        prelude = _PRELUDE.format(precision=MATMUL_PRECISIONS[self.options.matmul_precision])
        source = "\n".join([header, "", prelude, "", *kernel, "", "", *wrapper, ""])

        namespace: dict[str, Any] = dict(spec.consts)
        namespace.update(jax=jax, jnp=jnp, pl=pl, pltpu=pltpu)
        namespace.update({table.upper(): contents for table, contents in self.prefetch.items()})
        exec(compile(source, f"<autokernel:{nest.fn_name}>", "exec"), namespace)
        return Lowered(
            source=source,
            entry=namespace[f"{nest.fn_name}_pallas"],
            grid=grid,
            operands=spec.operands,
            out=spec.out,
            out_block=spec.out_block,
            scratch={acc.name: acc.scratch for acc in spec.accumulators},
            vmem_bytes=vmem,
            active_tiles=len(mask.active) if mask else None,
            total_tiles=mask.total if mask else None,
            fill=("background" if self.fill else "prefill") if mask and mask.constant else None,
        )
