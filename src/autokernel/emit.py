"""Shared pieces of the code generators: options, results, and the user-code rewriting both backends use.

A backend (`grid.GridEmitter`, `manual.ManualEmitter`) turns a `Specialization` into Pallas source: a
kernel function plus a wrapper that calls `pl.pallas_call`. The tile body (accumulator init, reduction
step, epilogue) is the same in both; only how tiles are visited and how data moves differs.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import hardware
from .analysis import Block, Operand, Specialization
from .errors import AutoKernelError

__all__ = ["FILL_DMA_DEPTH", "MATMUL_PRECISIONS", "PIPELINES", "Emitter", "Lowered", "Options"]

# Mosaic's default f32 matmul is a single bf16 pass; HIGHEST does the multi-pass f32 matmul the
# Python code actually describes.
MATMUL_PRECISIONS = {"highest": "jax.lax.Precision.HIGHEST", "default": "None"}

# How tiles are visited and data is moved:
#   "manual": one loop over every step with hand-written double-buffered DMAs (no grid). Fastest.
#   "grid":   a Pallas grid with BlockSpecs; Pallas's pipeliner moves the data.
PIPELINES = ("manual", "grid")

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
    pipeline: str = "manual"

    def __post_init__(self) -> None:
        if self.matmul_precision not in MATMUL_PRECISIONS:
            raise AutoKernelError(
                f"matmul_precision must be one of {sorted(MATMUL_PRECISIONS)}, got {self.matmul_precision!r}"
            )
        if self.pipeline not in PIPELINES:
            raise AutoKernelError(f"pipeline must be one of {list(PIPELINES)}, got {self.pipeline!r}")


@dataclass(frozen=True)
class Lowered:
    """A kernel specialised to one set of input shapes/dtypes."""

    source: str  # the generated Pallas code
    entry: Callable[..., jax.Array]  # entry(*args, interpret=...) -> output
    pipeline: str  # "manual" or "grid"
    grid: tuple[int, ...]  # the Pallas grid; () for manual kernels
    steps: int  # tile steps the kernel runs: grid points, or iterations of the manual loop
    operands: list[Operand]
    out: jax.ShapeDtypeStruct
    out_block: Block
    scratch: dict[str, jax.ShapeDtypeStruct]  # accumulator name -> VMEM scratch
    vmem_bytes: int  # estimated VMEM use, double-buffered blocks included
    active_tiles: int | None = None  # with a tile mask: output tiles actually computed
    total_tiles: int | None = None
    fill: str | None = None  # how masked-out tiles get their value: "background" DMAs or "prefill"


@dataclass
class FillPlan:
    """Masked-out tiles written by the kernel itself, from one VMEM tile, with async DMAs."""

    value: str  # Python source for the fill scalar
    tables: dict[str, str]  # parallel loop var -> scalar-prefetch table of the constant tiles' blocks
    start: str  # table: tile t issues fill copies [start[t], start[t + 1])
    ref: str
    sem: str


def dtype_src(dtype: Any) -> str:
    return f"jnp.{np.dtype(dtype).name}"


def tuple_src(items: Sequence[str]) -> str:
    return f"({', '.join(items)}{',' if len(items) == 1 else ''})"


def indent(lines: Sequence[str], depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line if line else line for line in lines]


class Emitter:
    """Base class for code generators. Subclasses implement `kernel` and `wrapper`."""

    pipeline: ClassVar[str]

    def __init__(self, spec: Specialization, options: Options):
        self.spec, self.options, self.nest = spec, options, spec.nest
        self.red = spec.reduction
        self.mask = spec.mask
        self.fill = self._plan_fill()
        # Scalar tables the kernel reads from SMEM, in kernel-argument order.
        self.tables: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ to implement

    def kernel(self, name: str) -> list[str]:
        raise NotImplementedError

    def wrapper(self, kernel: str) -> list[str]:
        raise NotImplementedError

    def grid(self) -> tuple[int, ...]:
        return ()

    def steps(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ tiles

    def tiles(self) -> list[tuple[int, ...]]:
        """Absolute block indices of the parallel loops for every output tile computed, in loop order."""
        if self.mask is not None:
            return list(self.mask.active)
        bounds = [self.spec.loops[lp.var] for lp in self.nest.parallel]
        return [tuple(b) for b in np.ndindex(*(lb.grid for lb in bounds))] if bounds else [()]

    def _fill_tables(self) -> dict[str, np.ndarray]:
        tables: dict[str, np.ndarray] = {}
        if self.fill is None or self.mask is None:
            return tables
        for n, table in enumerate(self.fill.tables.values()):
            tables[table] = np.array([blocks[n] for blocks in self.mask.constant], np.int32)
        n_tiles, n_fill = len(self.mask.active), len(self.mask.constant)
        tables[self.fill.start] = np.array([t * n_fill // n_tiles for t in range(n_tiles + 1)], np.int32)
        return tables

    # ------------------------------------------------------------------ masked-out tiles

    def _plan_fill(self) -> FillPlan | None:
        """Write the skipped tiles from the kernel when their value is a known constant.

        This overlaps those writes with the matmuls and lets the output start uninitialised, instead of
        filling the whole output in a separate pass before the kernel runs.
        """
        mask = self.mask
        if mask is None or not mask.constant or mask.fill is None or not self.options.background_fill:
            return None
        names = self.spec.names
        value = mask.fill
        return FillPlan(
            value=repr(value) if np.isfinite(value) else f"float('{value}')",
            tables={lp.var: names.fresh(f"{lp.var}_fill_blocks") for lp in self.nest.parallel},
            start=names.fresh("fill_start"),
            ref=names.fresh("fill_ref"),
            sem=names.fresh("fill_sem"),
        )

    def fill_helpers(self, dst: str) -> list[str]:
        """`_fill_copy(c)`: the DMA writing constant tile c of the output (ref `dst`) from the fill tile."""
        fill, block = self.fill, self.spec.out_block
        assert fill is not None and self.mask is not None
        dst_slices = [
            f"pl.ds({fill.tables[var]}[c] * {size}, {size})" if var else f"pl.ds({int(i) * size}, {size})"
            for var, size, i in zip(block.loops, block.shape, block.index, strict=True)
        ]
        return [
            f"# {len(self.mask.constant)} masked-out tiles are the constant {fill.value}: write them from one VMEM",
            f"# tile with async DMAs spread across the tiles ({FILL_DMA_DEPTH} in flight at most), behind the matmuls.",
            "def _fill_copy(c):",
            f"    return pltpu.make_async_copy({fill.ref}, {dst}.at[{', '.join(dst_slices)}], {fill.sem})",
        ]

    def fill_tile_init(self) -> list[str]:
        fill = self.fill
        assert fill is not None
        return [f"{fill.ref}[...] = jnp.full({fill.ref}.shape, {fill.value}, {fill.ref}.dtype)"]

    def fill_issue(self, tile: str) -> list[str]:
        """Start the fill copies assigned to output tile `tile`, keeping FILL_DMA_DEPTH in flight."""
        fill = self.fill
        assert fill is not None
        return [
            f"@pl.loop({fill.start}[{tile}], {fill.start}[{tile} + 1])",
            "def _start_fill_copies(c):",
            f"    @pl.when(c >= {FILL_DMA_DEPTH})",
            "    def _limit_in_flight():",
            "        _fill_copy(0).wait()",
            "    _fill_copy(c).start()",
        ]

    def fill_drain(self) -> list[str]:
        assert self.mask is not None
        return [f"for _ in range({min(len(self.mask.constant), FILL_DMA_DEPTH)}):", "    _fill_copy(0).wait()"]

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

    def body(self) -> tuple[list[str], list[str], list[str], list[str]]:
        """(temporaries, accumulator inits, reduction step, epilogue) in the user's order."""
        nest, spec = self.nest, self.spec
        refs = {acc.name: acc.ref for acc in spec.accumulators}
        widened = {
            acc.ref: dtype_src(acc.scratch.dtype) for acc in spec.accumulators if acc.init.dtype != acc.scratch.dtype
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

    @staticmethod
    def names_used(lines: Sequence[str]) -> set[str]:
        return {n.id for n in ast.walk(ast.parse("\n".join(lines))) if isinstance(n, ast.Name)}

    # ------------------------------------------------------------------ validation and assembly

    def check_hardware(self) -> int:
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
        hardware.check_smem(sum(t.nbytes for t in self.tables.values()))
        pipelined = [(op.block.shape, op.dtype) for op in spec.operands] + [(spec.out_block.shape, spec.out.dtype)]
        scratch = [(acc.scratch.shape, acc.scratch.dtype) for acc in spec.accumulators]
        if fill:
            scratch.append((spec.out_block.shape, spec.out.dtype))
        return hardware.check_vmem(pipelined, scratch, self.options.vmem_limit_bytes)

    def lower(self) -> Lowered:
        spec, nest, mask = self.spec, self.nest, self.mask
        kernel_name = spec.names.fresh(f"{nest.fn_name}_kernel")
        kernel = self.kernel(kernel_name)
        vmem = self.check_hardware()
        wrapper = self.wrapper(kernel_name)

        header = (
            f"# Generated by autokernel ({self.pipeline} pipeline) from `{nest.fn_name}` "
            f"({spec.filename}:{nest.first_line}) for "
            + ", ".join(
                f"{p}: {dtype_src(s.dtype)}{tuple(s.shape)}" for p, s in zip(nest.params, spec.specs, strict=True)
            )
        )
        prelude = _PRELUDE.format(precision=MATMUL_PRECISIONS[self.options.matmul_precision])
        source = "\n".join([header, "", prelude, "", *kernel, "", "", *wrapper, ""])

        namespace: dict[str, Any] = dict(spec.consts)
        namespace.update(jax=jax, jnp=jnp, pl=pl, pltpu=pltpu)
        namespace.update({table.upper(): contents for table, contents in self.tables.items()})
        exec(compile(source, f"<autokernel:{nest.fn_name}>", "exec"), namespace)
        return Lowered(
            source=source,
            entry=namespace[f"{nest.fn_name}_pallas"],
            pipeline=self.pipeline,
            grid=self.grid(),
            steps=self.steps(),
            operands=spec.operands,
            out=spec.out,
            out_block=spec.out_block,
            scratch={acc.name: acc.scratch for acc in spec.accumulators},
            vmem_bytes=vmem,
            active_tiles=len(mask.active) if mask else None,
            total_tiles=mask.total if mask else None,
            fill=("background" if self.fill else "prefill") if mask and mask.constant else None,
        )
