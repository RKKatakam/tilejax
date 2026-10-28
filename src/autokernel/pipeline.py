"""How deeply a kernel buffers its copies: the one thing about a kernel profiling chooses.

Block sizes are the user's (they are in the loops); how many VMEM slots each copy rotates through is the
kernel's. More input slots let fetches run further ahead of the compute; more output slots let more
tiles' copies to HBM be in flight. The tuner picks them from where a profile says the kernel waits
(profiling.py).

A kernel whose tiles take one step can also move its copies in bursts: the inputs of a burst of steps
together, then their outputs together, so at any time the copies in flight all read HBM or all write it.
A v5e's HBM loses ~6% to turning around between reads and writes: streaming a pool through VMEM and back
in 1 MiB copies moved 640-650 GB/s however deep the pipeline (reads alone: 731, writes alone: 639); in
alternating bursts of 8-16 MiB, 677-683 GB/s (XLA's elementwise pass: 673).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import v5e
from .errors import KernelError
from .plan import Kind, Plan
from .schedule import Schedule

__all__ = [
    "MAX_BURST",
    "MAX_INPUT_BUFFERS",
    "MAX_OUTPUT_BUFFERS",
    "TILE_SLOTS",
    "Pipeline",
    "default_pipeline",
    "fits",
    "rings",
    "vmem_bytes",
]

MAX_INPUT_BUFFERS = 4
MAX_OUTPUT_BUFFERS = 3
MAX_BURST = 16
TILE_SLOTS = 2  # an input that changes once per tile: this tile's and the next one's


@dataclass(frozen=True, order=True)
class Pipeline:
    """VMEM slots per copy: `input_buffers` for each input tile that changes every step (fetched that many
    steps ahead, less one), `output_buffers` for each output tile. With `burst` steps (`Pipeline.bursts`),
    the copies move in bursts instead: a burst's inputs are fetched while the burst before it computes, and
    its outputs are copied out once the next burst's inputs are in (2 * burst input slots, burst output
    slots)."""

    input_buffers: int = 3
    output_buffers: int = 2
    burst: int = 0

    @staticmethod
    def bursts(steps: int) -> Pipeline:
        """Copies in bursts of `steps` steps."""
        return Pipeline(2 * steps, steps, steps)

    def __post_init__(self) -> None:
        if self.burst:
            if not (
                1 <= self.burst <= MAX_BURST
                and (self.input_buffers, self.output_buffers) == (2 * self.burst, self.burst)
            ):
                raise KernelError(
                    f"a pipeline of bursts of 1 to {MAX_BURST} steps has 2 input buffers and 1 output buffer per "
                    f"step of a burst, not {self.input_buffers} and {self.output_buffers} for {self.burst}"
                )
            return
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
        if self.burst:
            return f"bursts of {self.burst} steps"
        return f"{self.input_buffers} input / {self.output_buffers} output buffers"


def rings(plan: Plan) -> dict[str, str]:
    """The operand whose ring of VMEM slots each operand uses (by name). Windows streamed by different
    phases with the same tile shape and dtype share one ring: a step only ever uses its own slot of a ring
    (the step's index modulo the ring's depth), and every earlier step that used it has run, so the windows
    of one phase never meet another's there. Every other operand has a ring of its own (a phase's windows
    are all in use at its steps; a once-per-iteration window lives across its phase)."""
    owner, shared, used = {}, {}, {}  # (shape, dtype) -> its rings' owners; (phase, shape, dtype) -> rings used
    for op in plan.operands:
        if op.kind is not Kind.STREAM or op.phase is None:
            owner[op.name] = op.name
            continue
        key = (op.tile.shape, op.tile.dtype)
        n = used.get((op.phase, *key), 0)
        used[(op.phase, *key)] = n + 1
        owners = shared.setdefault(key, [])
        if n == len(owners):
            owners.append(op.name)
        owner[op.name] = owners[n]
    return owner


def vmem_bytes(plan: Plan, schedule: Schedule, pipeline: Pipeline) -> int:
    """VMEM the kernel's own buffers take: input slots (a ring per `rings` owner), output slots and fills,
    carried values."""
    owner = rings(plan)
    total = sum(
        pipeline.slots(op.kind) * v5e.padded_bytes(op.tile.shape, op.tile.dtype)
        for op in plan.operands
        if owner[op.name] == op.name
    )
    total += sum(
        (pipeline.output_buffers + (1 if schedule.fills and out.fill is not None else 0))
        * v5e.padded_bytes(out.tile.shape, out.dtype)
        for out in plan.outputs
    )
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
