# AGENTS.md

Guidance for anyone, human or agent, changing autokernel. Read this before writing code.

## What autokernel is, and isn't

autokernel compiles one Python loop nest over tiles, written as plain JAX, into one **gridless** Pallas
kernel for a **TPU v5e**, and tunes its schedule on the hardware. The scope is deliberately small:

- **One chip.** Everything assumes a TPU v5e (`v5e.py`): 128 MiB VMEM, 1 MiB SMEM, one TensorCore, Mosaic's
  (8, 128) layout. Off a TPU, kernels run in the Pallas TPU interpreter. Don't add other chips.
- **One loop nest per kernel.** No programs, fusion, vmap/grad rules or sharding.
- **One object to optimize over.** A kernel's only performance input is its tunable: the
  `@autokernel.tunable` default of its last parameter, a frozen pytree of `choice(...)` fields
  (`tunable.py`). The schedule autokernel searches is that tunable plus the buffering (`pipeline.py`).
  There are no options on `@autokernel.kernel`: called without its tunable, a kernel tunes; given one, it
  uses it.
- **Tuning from first principles** (`tune.py`): test every candidate against the plain-JAX reference, time
  it from a profiler trace, read its `jax.named_scope` regions (`profiling.py`), and let the bottleneck
  pick the next schedules. Don't add cost models, disk caches or search options; make the profile say what
  to do instead.
- **The examples are the spec**: every block of a modern decoder, each checked against an independent
  reference (float64 NumPy) and the strongest JAX baseline (XLA, splash, megablox, paged attention):
  - basics: matmul, causal matmul, causal attention, RMSNorm + projection, a fused attention block, a
    transformer (`transformer.py`);
  - Qwen: fused RMSNorm + QKV + QK-norm + RoPE, GQA attention (sliding window, output gate), SwiGLU
    (`qwen3.py`, `gqa_attention.py`, `swiglu.py`), whole dense and MoE models with KV-cache decoding
    (`qwen3_models.py`);
  - DeepSeek-V3 / Kimi K2: MLA prefill (`mla.py`), absorbed MLA decode, whole models (`deepseek.py`);
  - MoE: routers, dropless grouped SwiGLU experts over runtime masks and windows, shared experts (`moe.py`);
    a stack of MoE layers decoding in one kernel, each layer's routing computed inside it (`moe_megakernel.py`),
    and a whole MoE decode step in one kernel (`moe_decode_megakernel.py`);
  - linear attention: chunked Gated DeltaNet and KDA (`linear_attention.py`, `linear_attention_layer.py`),
    hybrid Qwen3-Next / Kimi Linear models (`hybrid_layers.py`, `hybrid_models.py`);
  - a real model: Gemma 4 E2B's text decoder with its weights, greedy generation against vLLM (`gemma4.py`,
    `gemma4_kernels.py`, `gemma4_megakernel.py`);
  - decoding: GQA (contiguous, paged) and MLA attention over KV caches with per-sequence lengths
    (`decode.py`, `kv_cache.py`, `decode_model.py`), a whole decode step as one kernel with cross-layer
    prefetch (`decode_megakernel.py`), a stack of layers in one kernel (`layer_stack.py`), KDA's decode step
    over paged state pools updated in place, bit for bit vLLM TorchTPU's fused kernel (`kda_decode.py`), ragged
    paged attention over a batch of decodes, prefills and chunks, with RPA v3's ABI (`ragged_paged_attention.py`).
  A feature earns its place by making one of them better or by a new example that needs it.

## The pipeline

```
fn --parse.py--> Nest --plan.py--> Plan --schedule.py--> Schedule --codegen.py--> source --exec--> kernel
                (syntax)       (shapes, tunable,      (step tables)           (+ Pipeline)
                                 types)
tune.py: for each candidate Config (tunable, Pipeline): build -> run -> compare with the reference ->
         profile (trace regions) -> proposals from the bottleneck -> ... -> the fastest Config
```

| Module | Responsibility | Never does |
| --- | --- | --- |
| `parse.py` | the language's syntax; every syntactic error | evaluate anything |
| `plan.py` | run the prologue; loop ranges; tile windows; operand kinds; carried values; type trace | order steps |
| `schedule.py` | step order; evaluate masks into tables (or describe the steps runtime masks choose from); `step_at`/`blocks_at` | know about buffers |
| `pipeline.py` | `Pipeline` (slots per copy); VMEM accounting; default buffering | measure |
| `codegen.py` | emit the source from Plan + Schedule + Pipeline | make semantic decisions |
| `mxu.py` | the meaning of `@`, shared by kernel, type trace and reference | |
| `reference.py` | the function as plain JAX (`@` rewritten to `mxu`) | |
| `tunable.py` | `@tunable`, `choice`, neighbours of a tunable | know about kernels |
| `tune.py` | `Config`, `search`, `proposals` (bottleneck -> next schedules), testing candidates, `tune` | parse or plan |
| `api.py` | `Kernel`: per-signature caching, tunable handling, `lower`/`source`/`reference`/`verify`/`tuning` | |
| `profiling.py` | read device traces into `KernelProfile`; `measure` a lowered kernel | choose anything |
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
6. Masks are evaluated at build time, on loop values and constants, except reduction masks that read
   elements of index arrays (runtime masks): the entry function evaluates those on the device, vectorized,
   into step tables, and the kernel's step loop runs to a count read from SMEM.

### How a step runs (codegen.py)

Decode the step into (tile, phase, block, first/last of the phase, empty) (arithmetic, or SMEM tables when
masked; with runtime masks, tables the entry function made on the device) → start the fetch of the step `input_buffers - 1` ahead (its phase's operands: prefetch crosses
phase and tile boundaries) → at a tile's first step, start the next tile's per-tile inputs and wait for
this tile's → wait for this phase's inputs → at a tile's last step, wait for the output slot → the code
before the phases (tile's first step) / the phase's body / the code after it (phase's last step; after the
last phase: stores + output DMA starts). Masked-out tiles are filled from a constant VMEM tile, a share per
tile, at most 4 in flight (an output updated in place isn't filled: it keeps the input's values). Trace
regions wrap the waits, the output wait and the tile code. In bursts (`Pipeline.bursts(g)`, single-step
kernels only), a burst's first step waits for the previous burst's output copies and starts the next
burst's fetches; its last step waits for them and starts its own output copies (the first burst's steps
wait for their own inputs, the last burst's copy their outputs out as they go).

A tile's reduction loops are its phases (`plan.Phase`); a reduction loop of one iteration is inlined into
the code around it by plan.py (its variable bound by a synthetic assignment), so it takes no step and the
windows it moves are fetched once. Values cross phases through VMEM scratch exactly as they cross steps.

A repeat loop (`plan.Repeat`, a loop around reduction loops) repeats its phases' steps per iteration
(`Plan.instances`); steps then always decode through tables whose codes hold the iteration (`rb`). Its own
code before its first phase is that phase's `before` (run at its first step), the code after its last phase
is `segments[last]` (every iteration), and the tile's code after the loop is `Repeat.after` (run when
`rb` is the last iteration). Windows only a repeat loop moves stream with the phase whose steps read them
(`Plan.read_phases`), so prefetch crosses iterations: this is cross-layer weight prefetch. Which values are
carried is decided on the code in the order it runs, repeat loops twice over (`plan._Sequence`).

## Measured facts that shape the code (TPU v5e, JAX 0.11.2)

Each of these was found on hardware; don't undo them without measuring again.

- Mosaic takes its internal scratch from the scoped VMEM limit: the limit is always 0.9 × VMEM
  (`v5e.VMEM_LIMIT`), not sized from the kernel's buffers (that was up to 2.7% slower).
- A semaphore `.wait()` is a scheduling barrier: the output-slot wait goes at the start of a tile's last
  step, before the tile code.
- A DMA window with a traced offset is slower than a constant one: loops of one block emit constant
  offsets (`codegen._scaled`).
- Only single-step tiles drop the accumulator: splitting a multi-step reduction's first step was slower.
- Three input slots beat two (8192³ bf16, 512³ tiles: 6.30 vs 7.39 ms); four helped where profiles showed
  exposed waits (8192³ bf16 with 256³ tiles: 13.54 -> 11.86 ms).
- Tuning must run its kernels outside any trace: when a kernel's first call is under `jax.jit`, JAX's
  (thread-local) trace would stage the random inputs and test runs. `jax.ensure_compile_time_eval` is not
  an option (it leaks into the Pallas kernel's own trace); tune.py runs tuning on a thread of its own.
- Tuning from the defaults on a v5e: 4096³ bf16 from 128³ tiles 7.72 -> 0.736 ms (1024³, 4 buffers);
  8192³ 6.35 -> 5.77 ms; causal 3.48 -> 3.24 ms; attention 700 -> 685 µs; RMSNorm + projection
  8192×4096→4096 1.66 -> 1.49 ms (XLA unfused 1.61), 16384×2048→2048 0.876 -> 0.778 ms (XLA 0.907).
  A kernel that clamps a tunable field to the array (`min(tiles.tm, M)`) makes some neighbours the same
  kernel: proposals skip neighbours whose loops step as the current schedule's do.
- The whole attention block (attention + RMSNorm + FFN, one kernel, two phases) beat splash (at its best
  blocks) + RMSNorm + FFN in XLA by 1.06-1.13x on six shapes (D 1024-4096, F 4096-16384, head dim 64 and
  128); tuning picked 256-512 query/key blocks and streamed FFN weights in 1024-wide blocks. Bigger blocks
  ran Mosaic out of VMEM (intermediates beyond the kernel's own buffers): the tuner rules those out.
- Whole transformer (examples/transformer.py), tokens to logits: 1.03-1.23x faster than plain JAX + splash
  at its best blocks on four model sizes. Fusing the FFN into the attention kernel wins at d_model 1024
  (1.24 ms fused vs 0.54 + 0.74 split); at d_model 2048-4096 with batch*seq 4096 the attention's query
  blocks are too few rows to reuse the FFN weights, and a separate FFN kernel (1024-row tiles) wins by
  ~2%. The split form's extra launch + HBM round trip cost < 1%: compare tuned device times directly.
- A matmul accumulating in under 32 bits (plain `bf16 @ bf16` in a helper) can't compile for the MXU:
  plan.py rejects it from the traced jaxpr (`_check_matmuls`), naming the line.
- 128³ matmul tiles are per-step-overhead bound (~0.2 us of fixed cost per step); 512³ bf16 tiles are
  DMA/MXU balanced. The tuner's proposals follow from these: overhead -> bigger tiles.
- `jax.named_scope` in a kernel shows up in the profiler as events on the "XLA TraceMe" line, named
  `<kernel>/<region>`, user scopes by their own name, only with
  `LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true` set before the backend starts. The kernel's
  call is on the "XLA Ops" line, named `<kernel>.N` (sometimes `<kernel>.N.<output>_hbm`).
- Mosaic on a v5e has no 8-bit integer vector arithmetic and no mixed-dtype matmul (hence
  `mxu._mxu_operands` promotes); the interpreter accepts both, so cross-lowering tests matter.
- Gemma 4 E2B (text, real weights), 128-token prompts + 256 greedy tokens: 1.06-1.16x vLLM 0.31's output
  tokens/s at batch 1-256 (B1 158 vs 148, B256 13604 vs 11713). The one-kernel decode step (all layers, cross-layer
  prefetch) is copy-bound at B1 (6.31 ms vs a 5.5 ms weight-read floor) and only beats three kernels per layer
  there (6.36 ms): its attention takes a step per sequence (B32: 3429 steps, 7.49 ms vs 6.18 at B1), so the
  per-layer kernels win from B8 (6.58 vs 6.70 ms) up. Tuning moved every batch by under 1%.
- MoE at full width (Qwen3-235B layer, 4096 tokens): fused `expert_ffn_blocks` 10.35 ms (119.5 TFLOP/s, MXU-bound)
  vs ragged_dot 15.00 and megablox 16.86, each baseline at its best of XLA's default scoped VMEM limit and
  `--xla_tpu_scoped_vmem_limit_kib` raised to 115 MiB (megablox's 256+-row tilings only compile with it; it went
  20.45 -> 16.86 ms). Compare baselines at both limits. Decode is bandwidth-bound for all (720-750 GB/s).
  An expert's whole weights per step can't fit VMEM at full width (Mixtral: 352 MB): `moe_decode` streams
  blocks of hidden units in a repeat loop over the chosen experts. `compare` checks outputs in chunks:
  float64 copies of a 16384-token layer's rows ran the TPU VM's host out of memory.
- Mosaic and XLA can differ by one bf16 ulp on fused elementwise math, and XLA may skip an
  f32 -> bf16 -> f32 round trip in the reference (excess precision) that the kernel performs: exact
  kernel-vs-reference checks only hold in the interpreter (mark such tests `cpu`). Plain `f32 @ f32` in
  XLA on a TPU is one bf16 pass: compare kernels against float64 NumPy, not against `x @ y`.

- HBM turns around between reads and writes: streaming a pool through VMEM and back in 1 MiB copies moved
  640-650 GB/s whatever the pipeline depth, copy size or split (reads alone 731, writes alone 639; XLA's
  elementwise pass 673); alternating bursts of reads and of writes, 590 (1 MiB) / 665 (4) / 677 (8) / 682
  (16 MiB) GB/s (`benchmarks/vs_vllm_kda/probe_*.py`). Hence `Pipeline.bursts` for single-step kernels;
  the first burst waits per step and the last writes per step (+2-4% at 64-256 bursts' worth of steps).
- An input updated in place and called undonated is copied by XLA first; KDA decode's 16-head overlapped
  schedule ran 3-5% faster on that fresh copy than in place (batch 8-128; 32-head tiles and bursts didn't
  care): tuning times such kernels in place (`profiling.measure` donates them).
- `@` is the MXU's full-precision f32 matmul only in tile code: in a helper function it is plain JAX's,
  which for f32 on Mosaic (as in XLA) is one bf16 pass. vLLM's KDA decode relies on that pass;
  `kda_decode.py` matches it bit for bit by doing the same (explicit `jax.lax.dot_general`).
- KDA decode against vLLM TorchTPU's `decode_kda` (Kimi Linear: 32 heads of 128, pools of 2B+1 slots,
  donated), device time: B1 10.7 -> 8.5 us, B8 55.0 -> 55.2, B32 215.5 -> 206.5, B256 1706 -> 1615 (658 ->
  695 GB/s); transposed pools B1 12.4 -> 8.6, B256 1711 -> 1618. Bit-identical outputs and states on a
  v5e. vLLM swaps transposed states (4.2 us of compute per sequence, more than its reads take, so bursts
  lost); reading them as transposed MXU weights (`dot_general` contracting their last dim) gives the same
  bits at ~2 us. The plain-JAX step's states are ~3e-3 off float64 on a v5e (not from matmul precision:
  the same under `jax.default_matmul_precision("highest")`).

- Ragged paged attention against vLLM TorchTPU's RPA v3, as its runner calls it (MIXED for every prefill;
  its PREFILL case is ~11 µs slower), device µs per layer step: decode 1.36-1.79x at 512-position contexts,
  0.94-1.02x at 2k-8k (both read 710-750 GB/s); prefill 1.16-1.50x, chunks 1.13-1.16x, mixed 1.04-1.10x.
  The prefill lead is RPA's q/output layout copies (~270 µs at 4096 tokens): its prefill kernel alone beats
  ours on short prompts (469 vs 618 µs at 4 × 1024). XLA's scatter of new tokens into the cache relaid the
  whole cache out twice (1 ms each at 75k positions): `rpa_write` writes them instead. TPU gathers and
  scatters of int tables cost ~7 ns an entry, which is why runtime ranges step with a cursor, not tables.

- Decode megakernel profile (`benchmarks/decode_megakernel_profile.py`): every shape copy-bound, 733 GB/s at
  B1 down to 643 at B64 x 512. Deeper pipelines didn't help (3 -> 6 input slots: 3126 -> 3085 us at B64;
  the code between phases is 44 us of 3.1 ms): the B64 loss was copying whole 256-row cache blocks past
  each sequence's length (a fifth of its attention bytes). Ragged reads fixed it (3.33 -> 2.97 ms, ~747
  GB/s of what must be read). At 64 sequences of ~32 positions the per-step cost dominates: attention steps
  over groups of 8 sequences, 1760 -> 1446 us; past ~512 positions groups read more than they save.
- A ragged copy along a tiled dim must be whole tiles, and Mosaic must see it: its size rounded up and its
  start and size wrapped in `pl.multiple_of` (else "Slice sizes / Offsets along tiled dimensions must be
  aligned to tiles"). A VMEM -> SMEM DMA of a staging buffer and per-element scalar extraction both work
  for turning a computed int32 vector into SMEM scalars (`benchmarks/probe_computed_indices.py`).
- Routing inside the kernel (`moe_megakernel.py`, two Qwen3-30B-A3B layers): 1.32x moe.py's per-layer
  decode at 1 token (0.225 vs 0.297 ms), 1.02-1.05x from 4 to 64 (both near 750 GB/s). Each layer's first
  expert blocks wait for the routing (no prefetch can cross it): 680 GB/s at 1 token.
- A whole MoE decode step in one kernel (two Qwen3-30B-A3B layers, contexts ~1k): 1.16x the per-layer
  kernels at B1 (0.546 vs 0.633 ms), 1.03x at B8, parity at B32 (both read the chosen experts, ~110 of 128).
- Tuning is per input signature, not per lengths: a short-context batch with the same cache shape reuses a
  long one's tuning (the attention group size most). Benchmarks give short contexts caches of their own.

## Commands

```bash
uv sync
uv run pytest --cov=autokernel          # must stay at 100% (line + branch)
uv run ruff check . && uv run ruff format --check .
uv run python examples/matmul.py        # prints the generated source
```

Tests: `test_parse.py` (every syntax rule), `test_plan.py`, `test_schedule.py`, `test_codegen.py` (every
emission path runs in the interpreter with the race detector and cross-lowers to Mosaic for a v5e),
`test_index_arrays.py` (SMEM index arrays, runtime masks and windows), `test_in_place.py` (outputs updated in place,
stores at index-array windows), `test_bursts.py` (copies in bursts), `test_repeat_loops.py`, `test_step_outputs.py` (outputs stored per step),
`test_runtime_ranges.py` (runtime ranges and tiles, ragged rows, gathered windows, pairs), `test_computed_indices.py`
(routing inside a kernel), `test_api.py`, `test_profiling.py` (synthetic traces), `test_tunable.py`, `test_tune.py` (the search and
proposals with synthetic profiles; tuning end to end with `tune._can_time` and `profiling.measure` faked,
kernels really running in the interpreter), `test_examples.py`, `test_tpu.py` (`tpu` marker: real hardware
only). `helpers.make_kernel(source, **globals)` builds a kernel from a source string, which
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
- A kernel's result must not depend on its tunable: tuning checks every candidate against the reference
  run at another tunable (the one with the fewest loop iterations, usually the largest blocks). Make a
  tunable that would change the math unbuildable (`assert S % bk == 0` in the prologue): the decode
  megakernel's caches flattened across sequences let a 2048-row block straddle two of them at S_max 7168,
  so its reference was wrong and every correct candidate was rejected.
- Keep public API small: `autokernel.__all__`. New options need a measured reason; see "What autokernel
  isn't".
- When something is learned on hardware, add it to "Measured facts" above with the measurement.
- Commits: no `Co-Authored-By` trailers.
