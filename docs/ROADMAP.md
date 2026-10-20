# autokernel vs Helion, and what to add next

## What Helion is (GPU Helion; its TPU backend is out of scope here)

Helion (`pytorch/helion`) is PyTorch's tile DSL. A `@helion.kernel` is host PyTorch code around
`for tile in hl.tile(...)` loops. The top-level tile loop becomes the launch grid (program ids), and inner
tile loops become sequential `tl.range` loops in **one Triton kernel**. A few points matter for us:

- **One kernel per decorated function, and no cross-kernel fusion.** Several top-level loops in one
  function are dispatched by program-id ranges inside one launch. Pointwise prologue/epilogue fusion only
  happens through Inductor (`torch_compile_fusion=True`) around a single kernel. Helion has no pass that
  decides how a module is split into kernels.
- **Implicit masks.** Every tile load/store is masked (`index < extent`, loads `other=0`), and masked lanes
  are set to the reduction's identity before a reduction. Masks are dropped where the extent is provably a
  multiple of the block. `tile.begin/end/index/count` and `hl.jagged_tile` cover ragged and jagged data.
- **Empirical autotuning.** The config space covers block sizes, warps/stages, loop orders, L2 grouping,
  pid type (flat/xyz/persistent), indexing (pointer/block_ptr/TMA), reduction loops, unroll factors and
  more. It is searched with surrogate-guided pattern search or differential evolution, about 10 minutes
  and about 1500 configs per kernel. Each candidate is checked against a baseline output (rtol/atol 1e-2,
  atol scaled by RMS) and benchmarked in a subprocess with a timeout. Results are cached on disk, keyed by
  the kernel's source hash, the specialization key (shapes/dtypes/strides), the hardware, the backend and
  the config-spec hash.
- **No autograd and no global layout or memory planning.**

## Where autokernel is meant to beat it on TPU

| | Helion | autokernel |
| --- | --- | --- |
| Kernel boundary | whatever the author wrote | found automatically: a whole function becomes a program of nests, consecutive nests fuse into one `pallas_call` (cost-gated), views between nests are read as DMA windows |
| Per-step overhead | grid program ids (on TPU a Pallas grid costs 15–25 ns/step, splash attention's grid ~150–450 ns/step) | **gridless**: one `pl.loop`, steps decoded with a few scalar ops; a grid only to split work across the TensorCores of a chip that has more than one (`grid=(cores,)`, `dimension_semantics=("parallel",)`) |
| Tuning cost | minutes, ~1500 timed configs | a calibrated roofline model ranks thousands of combinations in <1 s; only the best 8 (kept diverse) are timed |
| Tuning correctness | baseline-output check per candidate | the same check, against the Python loops themselves (independent `jnp.matmul` reference); plus block sizes that would make a loop ragged are rejected because they can change the result |
| Tuning cache | source hash + specialization + hardware + backend | nest AST + values of every constant it reads + input signature + chip + options + JAX/autokernel versions |
| Ragged edges | implicit masks per element | whole trailing tiles: inputs zero-padded and outputs cropped in the wrapper, so the kernel only sees aligned tiles; masks only where the user's math needs them |
| Autodiff, vmap, sharding | none (hand-written backward) | `jax.grad`/`jvp`/`vmap`/`shard_map` of kernels and whole programs |

**Grid policy, as you asked:** every kernel is gridless. A `grid=(cores,)` appears only when the target
chip has two or more TensorCores per device (v4, v5p megacore). There it splits the steps at a tile
boundary that balances step counts across the cores, masks included (`schedule.core_bounds`). One-core
chips (v5e, v6e) get no grid at all.

## Feature map: what to add, roughly in order of value per effort

Effort: **S** about a day, **M** a few days, **L** a week or more. Each item names the files it touches.

### Quick wins (S)

1. **Pinned configs (`config=`).** `@autokernel.tpu(config={"TM": 512, "TN": 1024, "buffers": 3})` skips
   the search entirely, like Helion's `config=`. These are just `overrides` plus `buffers` passed to
   `lowering.lower` (`api.TpuKernel._lower`).
2. **A cache CLI.** `python -m autokernel.cache list|clear|show KEY` over `cache.directory()`, plus
   `kernel.tuning_key(*args)` for debugging misses (`cache.py`).
3. **Ahead-of-time tuning.** `kernel.autotune(specs_list)` fills the cache for a list of shapes (a serving
   deployment's buckets) before traffic arrives. It loops over `TpuKernel._lower` with
   `ShapeDtypeStruct`s, plus a measured pass on a TPU (Helion's `aot_runner`).
4. **`AUTOKERNEL_PRINT_SOURCE=1`.** Print every generated kernel's source as it is built (Helion's
   `print_output_code`). This is a hook in `lowering.build`.
5. **`verify(..., races=True)`.** Run the interpreter with `detect_races=True`,
   `uninitialized_memory="nan"`, as `tests/helpers.run_checking_races` already does.
6. **An edge-mask helper for ragged reductions.** `autokernel.within(k, TK, K)` would return
   `k + iota < K` with the right lane-wide shape. It is a one-liner that saves users from getting the
   `jnp.where` shape wrong.
7. **`kernel.profile(*args)`.** Device time from the JAX profiler (as `benchmarks/transformer.py`'s
   `device_ops` does), returned next to the cost model's prediction, so you can recalibrate on new chips.
8. **CI.** Run `uv run pytest -n auto --cov=autokernel --cov-fail-under=100` and ruff on every push, plus
   an optional Colab v5e job for the Mosaic-only checks.

### Medium (M)

9. **Automatic identity fill for ragged reductions.** When a ragged reduction feeds `jnp.max/min/
   logsumexp`, insert the `-inf`/`+inf` mask yourself, as Helion's `_mask_to` does. It would be detected
   from the jaxpr of the reduction body (`analysis._trace_tile` already traces it).
10. **Shape buckets.** Ragged loops make dynamic batch/sequence sizes cheap: pad a runtime size up to the
    next bucket and compile one kernel per bucket (`tpu(buckets={"S": (512, 1024, 2048)})`). The ragged
    machinery and the tuning cache already do most of the work.
11. **Subprocess-isolated measuring.** Time candidates in a child process with a timeout, so a Mosaic
    crash or hang drops one candidate instead of the run (Helion's `autotune_benchmark_subprocess`,
    `tune.measure`).
12. **In-kernel ragged edges (no pad copy).** When the ragged dim isn't one of the last two (batch, heads,
    layer stacks), a partial DMA window is tile-aligned. Clamp it in the copy function and zero the
    slot's tail, and skip the `jnp.pad` HBM copy (`codegen._copies`). It needs a v5e run to confirm Mosaic
    accepts it.
13. **Jagged loops.** `for t in range(0, lens[b], T)` with a runtime stop per batch element. The
    runtime-mask step tables (`schedule.runtime_tables`) already build per-tile step counts on device;
    this generalises them to per-tile extents (Helion's `hl.jagged_tile`).
14. **fp8 matmuls** on chips that have them (`hardware` already carries `fp8_flops_per_s`; `mxu` needs
    the accumulator rules).
15. **Loop-order and core-split knobs in the search.** Add the parallel-loop order (which tile is
    outermost, for weight reuse) and the TensorCore split point as tunables next to the block sizes
    (`tune.Space`).

### Large (L): whole-module fusion, the thing Helion can't do

16. **Fuse the JAX code between nests.** Element-wise code between two nests (a residual add, a norm, an
    activation) currently runs as XLA and breaks a fusion run. Lowering it into the previous nest's
    epilogue, or the next one's prologue, would make a transformer layer one kernel.
    `program.ProgramGroup` already knows the data flow, and `views.py` already handles the
    reshape/transpose subset.
17. **Producer–consumer fusion inside one kernel.** Keep a tile of nest A's output in VMEM and feed it
    straight into nest B's step instead of round-tripping through HBM (today fused parts still exchange
    data through HBM). Doing this needs a fusion legality check (B reads A's tiles in A's order) and a
    VMEM budget across parts (`fusion.py`).
18. **Collective matmuls.** For sharded programs, overlap all-gather / reduce-scatter with the matmul
    using remote DMAs (`pltpu.make_async_remote_copy`) in the gridless loop. This is the TPU equivalent
    of Helion's symmetric-memory examples, but generated from the sharding plan (`sharding.py`).
19. **Scans as kernels.** `h = h.at[t].set(f(h[t - 1]))` loops (RNNs, SSM recurrences) run as Python/JAX
    today. A carried-state nest kind would put them on the TPU with the state in VMEM.
20. **A cost model for fusion decisions over the whole program.** Choose cut points with the same
    roofline model, across groups and not only runs of consecutive nests, and tune block sizes jointly
    for fused parts. Helion tunes each kernel alone.
