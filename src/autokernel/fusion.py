"""Fuse consecutive tile loop nests of a program into one gridless kernel (a "megakernel").

A program's nests run one after another. As separate `pallas_call`s each pays a launch, fills its input
pipeline from cold and drains its output copies before the next can start. Fused, one kernel runs every
nest's step loop in turn:

    early_0(); late_0()
    run_0(hook=early_1)     # once nest 0 has issued its last copies, nest 1 starts fetching its weights
    drain_0()               # nest 0's outputs are all in HBM
    late_1()                # nest 1's first copies of what nest 0 wrote
    run_1(hook=early_2)
    ...

Each nest is generated exactly as it would be on its own (`codegen.Emitter`), as a *part*: a function of
its refs that defines its copies and step loop. Its first copies are split in two: `early` fetches inputs
no earlier nest writes (weights, the program's inputs) and runs as soon as the previous nest has issued
its own last copies, so they stream in behind its final steps; `late` fetches what an earlier nest wrote,
after that nest's output copies have drained.

**VMEM.** Only neighbouring nests are live together, so each part's scratch is allocated with
`pl.run_scoped` for just as long as it is needed, and later parts reuse the memory of earlier ones. The
exception is a part's *early* scratch (the buffers its `early` copies fill while the previous part still
runs), which has to be allocated before the previous part's own scope opens; with scopes being nested,
it then stays allocated to the end of the kernel. So while part k runs, VMEM holds its own scratch plus the
early scratch of parts 1 .. k + 1: `fused_vmem` is that peak, and nests are fused while it fits.

**Where to cut.** A run of nests longer than `fuse` (or that wouldn't fit in VMEM, ...) is cut into
several kernels: `plan_segments` uses the fewest kernels, and among the ways to do that, cuts where the
least data passes from the nests before a cut to the nests after it. Each kernel says why it ends where it
does (`FusedLowered.cut`, `ProgramLowered.cuts`).

**Two TensorCores** (v4, v5p). Each part splits its tiles between the cores as it would on its own, so a
part may read what the other core wrote in an earlier part: before a part's `late` copies, the cores meet
at a barrier (each signals a semaphore on both cores, then waits for both signals), which they reach only
once their own output copies of the earlier parts have drained.

Arrays passed between fused nests are outputs of the fused `pallas_call` (in HBM), read back by the later
nests' copies. Nests aren't fused when a nest's step tables or aliased output allocation need the *values*
of an array an earlier nest of the same kernel writes (they're computed before the kernel starts).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import codegen, hardware
from . import cost as cost_lib
from . import schedule as schedule_lib
from .analysis import Namer
from .hardware import vmem_nbytes
from .lowering import Lowered

__all__ = [
    "MAX_PARTS",
    "FusedLowered",
    "NestPlan",
    "Part",
    "Segment",
    "fusable",
    "fuse",
    "fused_vmem",
    "plan_segments",
    "scratch_split",
    "smem_bytes",
    "unfusable",
    "wrapper_reads",
]

# Nests per fused kernel, by default. On a v5e (bf16 transformer layers, 512^3 tiles), fusing runs of up to
# three nests was 1-3% faster than one kernel per nest, but seven nests in one kernel 1.5% slower: a long
# kernel loses more than the launches and pipeline fills it saves (likely its code outgrowing the core's
# instruction memory).
MAX_PARTS = 3
# Fuse neighbouring nests only when the launch, fill and drain fusing hides are at least this part of their time.
FUSE_MIN_GAIN = 0.02


@dataclass(frozen=True)
class Part:
    """One nest of a fused kernel: its kernel, the names of its inputs, and where each input comes from:
    `sources[param]` is `("input", name)` (an input of the fused kernel) or `("part", j, var)` (output `var`
    of the j-th part)."""

    name: str
    lowered: Lowered
    params: tuple[str, ...]
    sources: dict[str, tuple]


@dataclass(frozen=True)
class FusedLowered:
    """Several nests fused into one gridless kernel, run one after another."""

    name: str
    source: str
    entry: Callable[..., Any]  # entry(*inputs, interpret=...) -> every part's outputs, in order
    parts: tuple[tuple[str, Lowered], ...]
    inputs: tuple[str, ...]  # the program names the kernel takes, in order
    outputs: tuple[tuple[int, str], ...]  # (part, output variable) of each result, in order
    vmem_limit_bytes: int
    estimate: cost_lib.Estimate
    cores: int = 1
    schedule: str = "fused"
    cut: str = ""  # why the kernel ends where it does, if the run of nests goes on after it

    @property
    def steps(self) -> int:
        return sum(lw.steps for _, lw in self.parts)

    @property
    def tiles(self) -> int:
        return sum(lw.tiles for _, lw in self.parts)


def wrapper_reads(lowered: Lowered) -> set[str]:
    """Inputs whose values the kernel's wrapper needs before it starts: those its runtime masks read (to
    list the steps) and those its aliased outputs' allocations read."""

    spec = lowered.plan.spec
    nest = spec.nest
    names: set[str] = set()
    if spec.runtime_masks:
        guards = [nest.guard, *(r.guard for r in nest.reductions)]
        names |= {n.id for g in guards if g is not None for n in ast.walk(g) if isinstance(n, ast.Name)}
    for out in spec.outputs:
        if out.var in lowered.plan.aliased:
            names |= {n.id for n in ast.walk(out.alloc) if isinstance(n, ast.Name)}
    return names & set(nest.params)


def unfusable(lowered: Lowered) -> str | None:
    """Why this kernel can't be a part of a fused kernel, or None if it can."""
    if lowered.plan is None:
        return "it isn't a generated kernel"
    return None


def fusable(lowered: Lowered) -> bool:
    """Can this kernel be a part of a fused kernel at all?"""
    return unfusable(lowered) is None


def vmem_bytes(lowered: Lowered) -> int:
    """The VMEM a nest's kernel uses on its own (scratch and the compiler's temporaries)."""
    return lowered.estimate.vmem_bytes


def scratch_split(lowered: Lowered, dependent: set[str] | frozenset[str]) -> tuple[int, int]:
    """A part's VMEM as (early, rest): the early scratch (the rings of inputs not in `dependent`, which its
    `early` copies fill while the previous part runs, and fill values) and everything else (rings of
    `dependent` inputs, output slots, state, temporaries), as the cost model counts them."""
    p = lowered.plan
    early = sum(
        cost_lib.slots_nbytes(p.buffers if op.phase is not None else 2, op.block.kernel_shape, op.dtype)
        for op in p.spec.operands
        if op.param not in dependent
    )
    early += sum(vmem_nbytes(o.block.kernel_shape, o.out.dtype) for o in p.spec.outputs if o.var in p.fills)
    return early, max(0, vmem_bytes(lowered) - early)


def vmem_budget(parts: Sequence[Lowered], hw: hardware.Hardware) -> int:
    """What a fused kernel may plan to use: the chip's budget, or the smallest VMEM limit a user set on a part."""
    limits = [p.user_vmem_limit for p in parts if p.user_vmem_limit]
    return min(min(limits), hw.vmem_bytes) if limits else hw.vmem_budget


def smem_bytes(lowered: Lowered) -> int:
    """The SMEM a nest's kernel takes for its tables and scalars; a fused kernel takes every part's."""
    p = lowered.plan
    return sum(schedule_lib.smem_tables(p.spec, p.sched, bool(p.fills)).values())


def fused_vmem(parts: Sequence[tuple[int, int]]) -> int:
    """The peak VMEM of a fused kernel whose parts use (early, rest) bytes each: while part k runs, its own
    scratch and the early scratch of parts 1 .. k + 1 are allocated (see the module docstring)."""
    n, peak = len(parts), 0
    for k in range(n):
        live = parts[k][1] + sum(parts[j][0] for j in range(1, min(k + 2, n)))
        if k == 0:
            live += parts[0][0]
        peak = max(peak, live)
    return peak


# --------------------------------------------------------------------------- cutting a run of nests


@dataclass(frozen=True)
class NestPlan:
    """What `plan_segments` needs to know about one nest of a run (indices are positions in the run):

    - `sources`: its inputs that are outputs of earlier nests of the run, read straight from their HBM
      buffers: param -> (nest, variable);
    - `blocking`: earlier nests whose outputs it needs before its kernel starts (a mask or an aliased
      allocation reads them, or they go through JAX code first), with why: nest -> reason;
    - `nbytes`: the size of each earlier output it reads: (nest, variable) -> bytes.
    """

    name: str
    lowered: Lowered
    sources: dict[str, tuple[int, str]] = field(default_factory=dict)
    blocking: dict[int, str] = field(default_factory=dict)
    nbytes: dict[tuple[int, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    """Nests [start, end) of a run, run as one kernel; `reason`: why it ends where it does."""

    start: int
    end: int
    reason: str


def _mib(n: int) -> str:
    return f"{n / 2**20:.1f} MiB"


def not_worth_it(before: Lowered, after: Lowered) -> str | None:
    """Why fusing two neighbouring nests isn't worth it (None: it is).

    Fusing hides the fixed cost between two kernels: a launch, the second one's pipeline fill and the first
    one's drain. That pays when the kernels are short (decode-sized MLPs on a v5e: 1.5-2.5% faster), but
    fusing long ones gained at most 0.4%, and a prefill attention kernel fused with its neighbours lost 1%.
    So nests are fused only when that fixed cost is a noticeable part (`FUSE_MIN_GAIN`) of their time.
    """
    hidden = cost_lib.LAUNCH_S + after.estimate.fill_s + before.estimate.drain_s
    total = before.estimate.in_program_s + after.estimate.in_program_s
    if hidden < FUSE_MIN_GAIN * total:
        return (
            f"they run long enough ({total * 1e3:.2f} ms) that fusing would hide only {hidden * 1e6:.0f} µs "
            f"of launch, pipeline fill and drain (under {FUSE_MIN_GAIN:.1%})"
        )
    return None


def plan_segments(nests: Sequence[NestPlan], max_parts: int, hw: hardware.Hardware) -> list[Segment]:
    """Cut a run of nests into kernels: the fewest kernels, and of those, the cuts that pass the least data
    from the nests before each cut to the nests after it."""
    n = len(nests)
    crossing = [0] * (n + 1)  # bytes of earlier nests' outputs read across a cut before nest b
    for b in range(1, n):
        arrays = {
            key: size for j in range(b, n) for key, size in nests[j].nbytes.items() if key[0] < b
        }  # fmt: skip
        crossing[b] = sum(arrays.values())

    def why_not(a: int, b: int) -> str | None:
        """Why nests [a, b) can't be one kernel (None: they can)."""
        if b - a <= 1:
            return None
        for j in range(a, b):
            if (r := unfusable(nests[j].lowered)) is not None:
                return f"`{nests[j].name}` can't be fused: {r}"
        if b - a > max_parts:
            return f"a fused kernel runs at most {max_parts} nests (`fuse={max_parts}`)"
        for j in range(a + 1, b):
            if (r := not_worth_it(nests[j - 1].lowered, nests[j].lowered)) is not None:
                return r
        if len({nests[j].lowered.cores for j in range(a, b)}) > 1:
            return "the nests split their work over different numbers of TensorCores"
        for j in range(a, b):
            for i, why in nests[j].blocking.items():
                if a <= i < j:
                    return f"`{nests[j].name}` needs what `{nests[i].name}` writes before its kernel starts ({why})"
        parts = []
        for j in range(a, b):
            dependent = {p for p, (i, _) in nests[j].sources.items() if a <= i < j}
            parts.append(scratch_split(nests[j].lowered, dependent))
        need, budget = fused_vmem(parts), vmem_budget([nests[j].lowered for j in range(a, b)], hw)
        if need > budget:
            return f"together they would need {_mib(need)} of VMEM, more than the {_mib(budget)} budget"
        smem = sum(smem_bytes(nests[j].lowered) for j in range(a, b))  # every part's tables are inputs
        if smem > hw.smem_budget:
            return (
                f"together their tables and scalars would need {smem / 1024:.0f} KiB of SMEM, more than the "
                f"{hw.smem_budget // 1024} KiB budget"
            )
        return None

    best: list[tuple[int, int, int] | None] = [None] * (n + 1)  # (kernels, bytes across cuts, start of last)
    best[0] = (0, 0, 0)
    for b in range(1, n + 1):
        for a in range(b - 1, max(-1, b - 1 - max(max_parts, 1)), -1):
            if best[a] is None or why_not(a, b) is not None:
                continue
            cand = (best[a][0] + 1, best[a][1] + crossing[a], a)
            if best[b] is None or cand[:2] < best[b][:2]:
                best[b] = cand
    starts, b = [], n
    while b > 0:
        a = best[b][2]
        starts.append((a, b))
        b = a
    segments = []
    for a, b in reversed(starts):
        if b == n:
            reason = "the end of the run of nests"
        else:
            # Why the kernel can't take the next nest too; when that isn't what decided the cut (another
            # cut would have done as well), also why the cut is here.
            limit = why_not(a, b + 1)
            least = (
                f"of the ways to run these {n} nests as {best[n][0]} kernels, this cut passes the least data "
                f"from one kernel to the next ({_mib(crossing[b])})"
            )
            if limit is None:
                reason = least
            elif limit.startswith("a fused kernel runs at most"):
                reason = f"{limit}, and {least}"
            else:
                reason = limit
            reason = f"cut before `{nests[b].name}`: {reason}"
        segments.append(Segment(a, b, reason))
    return segments


# --------------------------------------------------------------------------- the fused kernel


def fuse(name: str, parts: Sequence[Part], matmul_precision: str, hw: hardware.Hardware, cut: str = "") -> FusedLowered:
    """Generate one kernel running `parts` in order."""
    emitters = []
    for part in parts:
        p = part.lowered.plan
        dependent = {param for param, src in part.sources.items() if src[0] == "part"}
        emitters.append(
            codegen.Emitter(
                p.spec, p.sched, p.buffers, matmul_precision, p.fills, p.aliased, 0, None, dependent=dependent
            )
        )
    cores = parts[0].lowered.cores

    # The fused kernel's inputs: every part's SMEM inputs, then the arrays it reads from outside, then the
    # parts' aliased initial values. Its outputs: every part's outputs.
    inputs: list[str] = []
    for part in parts:
        for param in part.params:
            src = part.sources[param]
            if src[0] == "input" and src[1] not in inputs:
                inputs.append(src[1])
    groups = [e.arg_groups() for e in emitters]
    scratch = [e.scratch_groups() for e in emitters]  # per part: ((names, shapes) early, (names, shapes) rest)
    namer = Namer(set(inputs))
    smem_args = [[namer.fresh(f"p{k}_{a}") for a in g["smem"]] for k, g in enumerate(groups)]
    array_inputs = [
        n
        for n in inputs
        if any(src == ("input", n) for e, part in zip(emitters, parts, strict=True)
               for param, src in part.sources.items() if param in e.hbm)
    ]  # fmt: skip
    hbm_args = {n: namer.fresh(f"{n}_hbm") for n in array_inputs}
    alias_args = [[namer.fresh(f"p{k}_{a}") for a in g["aliased"]] for k, g in enumerate(groups)]
    out_args = [[namer.fresh(f"p{k}_{a}") for a in g["outputs"]] for k, g in enumerate(groups)]
    early_args = [[namer.fresh(f"p{k}_{a}") for a in s[0][0]] for k, s in enumerate(scratch)]
    rest_args = [[namer.fresh(f"p{k}_{a}") for a in s[1][0]] for k, s in enumerate(scratch)]
    barrier = namer.fresh("barrier_sem") if cores > 1 else None
    out_index: dict[tuple[int, str], int] = {}
    outputs: list[tuple[int, str]] = []
    for k, e in enumerate(emitters):
        for o in e.spec.outputs:
            out_index[(k, o.var)] = len(outputs)
            outputs.append((k, o.var))
    flat_out_args = [a for args in out_args for a in args]
    # Arrays a later part reads are written into a buffer passed in (uninitialized, aliased to the output)
    # and read back from it: on a v5e, copies from a buffer the kernel received as an input are much faster
    # than from one of its outputs (it measured ~0.75 us more per 256 KiB tile read back from an output).
    fed = list(dict.fromkeys((src[1], src[2]) for part in parts for src in part.sources.values() if src[0] == "part"))
    buffers: dict[tuple[int, str], str] = {}
    new_buffers: list[tuple[int, str]] = []
    for j, var in fed:
        if var in emitters[j].aliased:  # already passed in: its allocation
            buffers[(j, var)] = alias_args[j][list(emitters[j].aliased).index(var)]
        else:
            buffers[(j, var)] = namer.fresh(f"p{j}_{var}_buffer")
            new_buffers.append((j, var))

    # ---- the kernel
    kernel_name = f"{name}_kernel"
    kernel_params = [
        *[a for args in smem_args for a in args],
        *hbm_args.values(),
        *[a for args in alias_args for a in args],
        *[buffers[b] for b in new_buffers],
        *flat_out_args,
        *([barrier] if barrier else []),
    ]
    n = len(parts)

    def part_args(k: int) -> list[str]:
        part, e = parts[k], emitters[k]
        hbm = []
        for param in e.hbm:
            src = part.sources[param]
            hbm.append(hbm_args[src[1]] if src[0] == "input" else buffers[(src[1], src[2])])
        outs = [buffers.get((k, o.var), a) for o, a in zip(e.spec.outputs, out_args[k], strict=True)]
        return [*smem_args[k], *hbm, *alias_args[k], *outs, *early_args[k]]

    def define(k: int) -> str:
        return f"early_{k}, with_scratch_{k} = _part_{k}({', '.join(part_args(k))})"

    def scope(fn: str, params: list[str], shapes: list[str], body: list[str]) -> list[str]:
        """`body` with `params` allocated (`pl.run_scoped`) for its duration; inline if there are none."""
        if not params:
            return body
        return [f"def {fn}({', '.join(params)}):", *codegen._indent(body), "", "pl.run_scoped(",
                f"    {fn},", *codegen._indent(shapes), ")", ""]  # fmt: skip

    def run(k: int) -> list[str]:
        """Part k's late copies, steps and drain (its early copies are under way)."""
        lines = [f"late_{k}, run_{k}, drain_{k} = with_scratch_{k}({', '.join(rest_args[k])})"]
        if barrier and k and emitters[k].dependent:
            lines.append("_barrier()  # every core's copies of the earlier nests' outputs have landed")
        lines += [f"late_{k}()  # {parts[k].name}", f"run_{k}(early_{k + 1})" if k + 1 < n else f"run_{k}()"]
        return [*lines, f"drain_{k}()"]

    def rest(k: int) -> list[str]:
        """Parts k.. once part k's early scratch is allocated and its `early` has been called."""
        own = scope(f"_scope_{k}", rest_args[k], scratch[k][1][1], run(k))
        if k + 1 == n:
            return own
        body = [define(k + 1), *own, *rest(k + 1)]
        return scope(f"_scope_{k + 1}_early", early_args[k + 1], scratch[k + 1][0][1], body)

    first = [define(0), f"early_0()  # {parts[0].name}", *run(0)]
    first = scope("_scope_0", early_args[0] + rest_args[0], scratch[0][0][1] + scratch[0][1][1], first)
    body = []
    if barrier:
        signals = [f"pl.semaphore_signal({barrier}, 1, core_index={c})" for c in range(cores)]
        body += [*codegen._fn("_barrier", "", [*signals, f"pl.semaphore_wait({barrier}, {cores})"]), ""]
    body += scope("_scope_1_early", early_args[1], scratch[1][0][1], [define(1), *first, *rest(1)])
    kernel = [f"def {kernel_name}({', '.join(kernel_params)}):", *codegen._indent(body)]

    # ---- the wrapper
    smem_spec, any_spec = "pl.BlockSpec(memory_space=pltpu.SMEM)", "pl.BlockSpec(memory_space=pl.ANY)"
    n_smem = sum(len(a) for a in smem_args)
    n_alias = sum(len(a) for a in alias_args)
    in_specs = [smem_spec] * n_smem + [any_spec] * (len(array_inputs) + n_alias + len(new_buffers))
    out_shape = []
    for e in emitters:
        out_shape += [f"jax.ShapeDtypeStruct({tuple(o.out.shape)}, {codegen._dtype_src(o.out.dtype)})"
                      for o in e.spec.outputs]  # fmt: skip
    aliases, at = {}, n_smem + len(array_inputs)
    for k, e in enumerate(emitters):
        for var in e.aliased:
            aliases[at] = out_index[(k, var)]
            at += 1
    for b in new_buffers:
        aliases[at] = out_index[b]
        at += 1
    split = [scratch_split(p.lowered, {q for q, s in p.sources.items() if s[0] == "part"}) for p in parts]
    vmem = fused_vmem(split)
    limit = min(p.lowered.vmem_limit_bytes for p in parts)
    estimate = cost_lib.Estimate(
        sum(p.lowered.estimate.seconds for p in parts),
        sum(p.lowered.estimate.flops for p in parts),
        sum(p.lowered.estimate.transcendentals for p in parts),
        sum(p.lowered.estimate.hbm_bytes for p in parts),
        vmem,
    )
    wrapper = [f"def {name}_pallas({', '.join([*inputs, '*', 'interpret=False'])}):"]
    split_over = f", each split over {cores} TensorCores" if cores > 1 else ""
    wrapper.append(f"    # One gridless kernel running {len(parts)} nests in turn{split_over}:")
    for k, (part, e) in enumerate(zip(parts, emitters, strict=True)):
        wrapper.append(f"    #   {k}. {part.name}: {e.describe()}{e.buffering()}")
    for k, part in enumerate(parts):
        args = [part.sources[p][1] if part.sources[p][0] == "input" else "None" for p in part.params]
        wrapper.append(f"    smem_{k}, init_{k} = _prepare_{k}({', '.join(args)})")
    for j, var in new_buffers:
        o = next(o for o in emitters[j].spec.outputs if o.var == var)
        shape, dtype = tuple(o.out.shape), codegen._dtype_src(o.out.dtype)
        wrapper.append(f"    {buffers[(j, var)]} = jnp.empty({shape}, {dtype})  # `{var}`, read back by a later nest")
    call_args = [*[f"*smem_{k}" for k in range(len(parts))], *array_inputs, *[f"*init_{k}" for k in range(len(parts))]]
    call_args += [buffers[b] for b in new_buffers]
    params = [f"vmem_limit_bytes={limit}"] + (['dimension_semantics=("parallel",)'] if cores > 1 else [])
    wrapper += [
        "    return pl.pallas_call(",
        f"        {kernel_name},",
        f"        out_shape=({', '.join(out_shape)},),",
        *([f"        grid=({cores},),"] if cores > 1 else []),
        f"        in_specs=[{', '.join(in_specs)}],",
        f"        out_specs=({', '.join([any_spec] * len(out_shape))},),",
        # Each part's scratch is allocated inside the kernel, for as long as the part needs it.
        f"        scratch_shapes=[{'pltpu.SemaphoreType.REGULAR' if barrier else ''}],",
        *([f"        input_output_aliases={aliases},"] if aliases else []),
        f"        compiler_params=pltpu.CompilerParams({', '.join(params)}),",
        "        cost_estimate=_COST,",
        "        interpret=interpret,",
        f"        name={name!r},",
        f"    )({', '.join(call_args)})",
    ]

    # ---- assemble: each part runs in its own namespace (its own constants), like a kernel of its own
    prelude = codegen.prelude()
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp, "pl": pl, "pltpu": pltpu, "_COST": estimate.cost_estimate()}
    part_sources = []
    for k, (part, e) in enumerate(zip(parts, emitters, strict=True)):
        src = [*e.part(f"_part_{k}", k + 1 < len(parts)), "", *e.prepare(f"_prepare_{k}"), ""]
        part_ns = e.namespace()  # (with the user's functions, which `module_defs` defines from source)
        exec(compile(prelude, f"<autokernel:{part.name}>", "exec"), part_ns)
        exec(compile("\n".join(src), f"<autokernel:{part.name}>", "exec"), part_ns)
        namespace[f"_part_{k}"], namespace[f"_prepare_{k}"] = part_ns[f"_part_{k}"], part_ns[f"_prepare_{k}"]
        part_sources += [f"# {'-' * 20} part {k}: {part.name}", e.header(), "", *e.module_defs, *src, ""]
    header = f"# Generated by autokernel: {len(parts)} nests fused into one kernel: " + ", ".join(p.name for p in parts)
    if cut:
        header += f"\n# It ends here because: {cut}"
    main = [*kernel, "", "", *wrapper, ""]
    exec(compile("\n".join(main), f"<autokernel:{name}>", "exec"), namespace)
    source = "\n".join(
        [header, "", prelude, *codegen.assignments({"_COST": namespace["_COST"]}), "", *part_sources, *main]
    )
    return FusedLowered(
        name=name,
        source=source,
        entry=namespace[f"{name}_pallas"],
        parts=tuple((p.name, p.lowered) for p in parts),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        vmem_limit_bytes=limit,
        estimate=estimate,
        cores=cores,
        cut=cut,
    )
