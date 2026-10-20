# AGENTS.md

Guidance for anyone, human or agent, changing autokernel. Read this before writing code.

## What autokernel is, and isn't

autokernel compiles one Python loop nest over tiles into one **gridless** Pallas kernel for a **TPU
v5e**. The scope is deliberately small:

- **One chip.** Everything assumes a TPU v5e (`v5e.py`): 128 MiB VMEM, 1 MiB SMEM, one TensorCore, Mosaic's
  (8, 128) layout. Off a TPU, kernels run in the Pallas TPU interpreter. Don't add other chips.
- **One loop nest per kernel.** No programs, fusion, vmap/grad rules or sharding.
- **No tuning knobs.** Block sizes are the user's (they are in the loops). The only thing autokernel
  chooses about a kernel's performance is its buffering (`pipeline.py`), and it chooses it from
  `jax.named_scope` trace regions measured on the TPU (`profiling.py`). Don't add cost models, search
  spaces, caches or options; make the profile say what to do instead.
- **The examples are the spec**: matmul, causal matmul, causal attention (checked against JAX's splash
  attention). A feature earns its place by making one of them better or by a new example that needs it.

## The pipeline

```
fn --parse.py--> Nest --plan.py--> Plan --schedule.py--> Schedule --codegen.py--> source --exec--> kernel
                (syntax)          (shapes, types)       (step tables)           (+ Pipeline)
```

| Module | Responsibility | Never does |
| --- | --- | --- |
| `parse.py` | the language's syntax; every syntactic error | evaluate anything |
| `plan.py` | run the prologue; loop ranges; tile windows; operand kinds; carried values; type trace | order steps |
| `schedule.py` | step order; evaluate masks into tables; `step_at`/`blocks_at` | know about buffers |
| `pipeline.py` | `Pipeline` (slots per copy); VMEM accounting; default buffering | measure |
| `codegen.py` | emit the source from Plan + Schedule + Pipeline | make semantic decisions |
| `mxu.py` | the meaning of `@`, shared by kernel, type trace and reference | |
| `reference.py` | the function as plain JAX (`@` rewritten to `mxu`) | |
| `api.py` | `Kernel`: per-signature caching, `lower`/`source`/`reference`/`verify` | |
| `profiling.py` | read device traces, `KernelProfile`, `candidates`, `optimize` | |
| `tilecode.py` | name analysis of tile statements; recognising output stores | |

### Invariants

1. **The generated source is what runs.** `Lowered.source` is exec'd as is (registered with `linecache`,
   so tracebacks show it). It is standalone Pallas: constants, step tables and helper functions are written
   into it. Keep it readable: it is a product, not an intermediate.
2. **The reference is the user's function**, with only `@` changed (to `mxu._mxu_dot`). The type trace in
   `plan.py` and the kernel use the same `mxu` functions, so all three agree on types.
3. **Errors are `KernelError`s naming the user's line** (`nest.error(message, node)`), raised before
   anything runs. Never let a Python or JAX exception from user code escape planning unwrapped.
4. **Constants are read once per input signature**, when the kernel is first built (as `jax.jit` does).
5. **100% line and branch coverage**, enforced by `pyproject.toml`. Unreachable code is deleted, not
   excluded; no `pragma: no cover`.
6. Masks are evaluated at build time, on loop values and constants only (they can't read arrays).

### How a step runs (codegen.py)

Decode the step (arithmetic, or SMEM tables when masked) → start the fetch `input_buffers - 1` steps ahead
→ at a tile's first step, start the next tile's per-tile inputs and wait for this tile's → wait for this
step's inputs → at a tile's last step, wait for the output slot → pre (first step) / body / post (last
step, stores + output DMA starts). Masked-out tiles are filled from a constant VMEM tile, a share per
tile, at most 4 in flight. Trace regions wrap the waits, the output wait and the tile code.

## Measured facts that shape the code (TPU v5e, JAX 0.11.2)

Each of these was found on hardware; don't undo them without measuring again.

- Mosaic takes its internal scratch from the scoped VMEM limit: the limit is always 0.9 × VMEM
  (`v5e.VMEM_LIMIT`), not sized from the kernel's buffers (that was up to 2.7% slower).
- A semaphore `.wait()` is a scheduling barrier: the output-slot wait goes at the start of a tile's last
  step, before the tile code.
- A DMA window with a traced offset is slower than a constant one: loops of one block emit constant
  offsets (`codegen._scaled`).
- Only single-step tiles drop the accumulator: splitting a multi-step reduction's first step was slower.
- Three input slots beat two (8192³ bf16, 512³ tiles: 6.30 vs 7.39 ms); `optimize` finds when four help.
- `jax.named_scope` in a kernel shows up in the profiler as events on the "XLA TraceMe" line, named
  `<kernel>/<region>`, user scopes by their own name, only with
  `LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true` set before the backend starts. The kernel's
  call is on the "XLA Ops" line, named `<kernel>.N` (sometimes `<kernel>.N.<output>_hbm`).
- Mosaic on a v5e has no 8-bit integer vector arithmetic and no mixed-dtype matmul (hence
  `mxu._mxu_operands` promotes); the interpreter accepts both, so cross-lowering tests matter.
- Mosaic and XLA can differ by one bf16 ulp on fused elementwise math, and XLA may skip an
  f32 -> bf16 -> f32 round trip in the reference (excess precision) that the kernel performs: exact
  kernel-vs-reference checks only hold in the interpreter (mark such tests `cpu`). Plain `f32 @ f32` in
  XLA on a TPU is one bf16 pass: compare kernels against float64 NumPy, not against `x @ y`.

## Commands

```bash
uv sync
uv run pytest --cov=autokernel          # must stay at 100% (line + branch)
uv run ruff check . && uv run ruff format --check .
uv run python examples/matmul.py        # prints the generated source
```

Tests: `test_parse.py` (every syntax rule), `test_plan.py`, `test_schedule.py`, `test_codegen.py` (every
emission path runs in the interpreter with the race detector and cross-lowers to Mosaic for a v5e),
`test_api.py`, `test_profiling.py` (synthetic traces), `test_examples.py`, `test_tpu.py` (`tpu` marker:
real hardware only). `helpers.make_kernel(source, **globals)` builds a kernel from a source string, which
keeps error-case tests short.

## Running on a TPU v5e (Colab CLI)

The only TPU available is a Colab v5e through `~/.local/bin/colab` (one session at a time; an idle one
is reclaimed after ~30 minutes, so set up and run back to back). Sub-agents are not allowed to create
sessions; ask the main session to run `tools/colab_tpu.sh`.

```bash
tools/colab_tpu.sh                      # tests + benchmarks on a fresh v5e session, output in tpu-run.txt
tools/colab_tpu.sh "python -m pytest tests -q"   # or any command, run from the repo root on the VM
```

What the script does, in case you need it by hand: `colab new -s ak --tpu v5e1`; upload a tarball of the
working tree; `colab exec -s ak -f tools/colab_driver.py`; `colab stop -s ak`. The driver installs
`jax[tpu]==0.11.2` and runs everything **through `subprocess`**: never `import jax` inside the Colab
kernel itself, or it holds the TPU for every later process (`colab restart-kernel -s ak` frees it).

## Conventions

- Python 3.12, `ruff` (120 columns), type hints on every function, `from __future__ import annotations`.
- Docstrings say what a thing is or does; comments say why. Prose names things the way the README does
  (tile, step, reduction, carried value, operand, fill, trace region).
- Prefer data (`Plan`, `Schedule`, `Pipeline` are frozen dataclasses) flowing forward to objects
  reaching back. A module only imports the ones to its left in the pipeline.
- Generated names come from `codegen._Namer`, so they never collide with names in the user's code.
- Keep public API small: `autokernel.__all__`. New options need a measured reason; see "What autokernel
  isn't".
- When something is learned on hardware, add it to "Measured facts" above with the measurement.
- Commits: no `Co-Authored-By` trailers.
