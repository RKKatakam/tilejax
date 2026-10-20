"""How deeply a kernel buffers its copies: the one thing about a kernel profiling chooses.

Block sizes are the user's (they are in the loops); how many VMEM slots each copy rotates through is the
kernel's. More input slots let fetches run further ahead of the compute; more output slots let more
tiles' copies to HBM be in flight. `autokernel.optimize` picks them from where a profile says the kernel
waits (profiling.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import v5e
from .errors import KernelError
from .plan import Kind, Plan
from .schedule import Schedule

__all__ = [
    "MAX_INPUT_BUFFERS",
    "MAX_OUTPUT_BUFFERS",
    "TILE_SLOTS",
    "Pipeline",
    "default_pipeline",
    "fits",
    "vmem_bytes",
]

MAX_INPUT_BUFFERS = 4
MAX_OUTPUT_BUFFERS = 3
TILE_SLOTS = 2  # an input that changes once per tile: this tile's and the next one's


@dataclass(frozen=True, order=True)
class Pipeline:
    """VMEM slots per copy: `input_buffers` for each input tile that changes every step (fetched that many
    steps ahead, less one), `output_buffers` for each output tile."""

    input_buffers: int = 3
    output_buffers: int = 2

    def __post_init__(self) -> None:
        if not (1 <= self.input_buffers <= MAX_INPUT_BUFFERS and 1 <= self.output_buffers <= MAX_OUTPUT_BUFFERS):
            raise KernelError(
                f"a pipeline has 1 to {MAX_INPUT_BUFFERS} input buffers and 1 to {MAX_OUTPUT_BUFFERS} output "
                f"buffers, not {self.input_buffers} and {self.output_buffers}"
            )

    def slots(self, kind: Kind) -> int:
        """VMEM slots for an input tile of this kind."""
        return {
            Kind.STREAM: self.input_buffers,
            Kind.ITERATION: self.input_buffers,
            Kind.TILE: TILE_SLOTS,
            Kind.ONCE: 1,
        }[kind]

    def __str__(self) -> str:
        return f"{self.input_buffers} input / {self.output_buffers} output buffers"


def vmem_bytes(plan: Plan, schedule: Schedule, pipeline: Pipeline) -> int:
    """VMEM the kernel's own buffers take: input slots, output slots and fills, carried values."""
    total = sum(pipeline.slots(op.kind) * v5e.padded_bytes(op.tile.shape, op.tile.dtype) for op in plan.operands)
    per_output = pipeline.output_buffers + (1 if schedule.fills else 0)
    total += sum(per_output * v5e.padded_bytes(out.tile.shape, out.dtype) for out in plan.outputs)
    total += sum(v5e.padded_bytes(c.shape, c.dtype) for c in plan.carried.values())
    return total


def fits(plan: Plan, schedule: Schedule, pipeline: Pipeline) -> bool:
    return vmem_bytes(plan, schedule, pipeline) <= v5e.VMEM_BUDGET


def default_pipeline(plan: Plan, schedule: Schedule) -> Pipeline:
    """Three input slots where they fit in VMEM (on a v5e, a 512^3-tile bf16 matmul ran 15% faster with
    three than with two), else two; two output slots, or one if each output is stored once (a kernel of one
    tile); raises KernelError if not even two input slots fit."""
    stores = max(schedule.tiles if out.phase is None else schedule.phase_steps[out.phase] for out in plan.outputs)
    outputs = min(2, stores)
    for pipeline in (Pipeline(3, outputs), Pipeline(2, outputs)):
        if fits(plan, schedule, pipeline):
            return pipeline
    need = vmem_bytes(plan, schedule, Pipeline(2, outputs))
    raise KernelError(
        f"{plan.nest.name}: its tiles need {need / 2**20:.1f} MiB of VMEM double-buffered, more than the "
        f"{v5e.VMEM_BUDGET / 2**20:.0f} MiB a kernel can use: use smaller tiles"
    )
