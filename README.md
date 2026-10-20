# autokernel

Write a TPU kernel as plain JAX: Python loops over tiles. Get a fast, **gridless**
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel for a **TPU v5e**, with its schedule
tuned on the hardware.

```python
import jax.numpy as jnp
import autokernel


@autokernel.tunable
class MatmulTiles:  # what autokernel may choose: a frozen pytree of choices
    tm: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tn: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)
    tk: int = autokernel.choice(128, 256, 512, 1024, 2048, default=512)


@autokernel.kernel
def matmul(x, y, tiles=MatmulTiles()):
    M, K = x.shape
    N = y.shape[1]
    tm, tn, tk = min(tiles.tm, M), min(tiles.tn, N), min(tiles.tk, K)
    out = jnp.zeros((M, N), x.dtype)
    for i in range(0, M, tm):
        for j in range(0, N, tn):
            acc = jnp.zeros((tm, tn), jnp.float32)
            for k in range(0, K, tk):
                acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
            out = out.at[i : i + tm, j : j + tn].set(acc.astype(x.dtype))
    return out


out = matmul(x, y)                     # tunes the schedule on the first call, then runs the kernel
print(matmul.tuning(x, y))             # what tuning tried and chose
out = matmul(x, y, MatmulTiles(tm=256))  # or choose it yourself
print(matmul.source(x, y))             # the generated Pallas source: plain, standalone, readable
matmul.verify(x, y)                    # checks the kernel against the loops run as plain JAX
```

The function is plain JAX and is the specification: `matmul.fn(x, y)` runs it as written (under
`jax.jit` too: a tunable is a pytree with no leaves), and `verify` compares the kernel with it. The one
thing a kernel does differently from plain JAX is `@`, which accumulates in f32 (i32 for integers) as the
MXU does, instead of rounding `bf16 @ bf16` to bf16; `verify` uses the same rule.

autokernel targets **one chip, the TPU v5e**. Without a TPU, kernels run in the Pallas TPU interpreter
with their default schedules, so you can develop and test them anywhere.

## How it works

Every kernel is **gridless**: one `pl.loop` over all of its steps (an output tile × a reduction block),
with hand-written, multi-buffered DMAs, the way you would write a performance-critical Pallas kernel by
hand. A Pallas grid adds pipeline bookkeeping to every step; this loop decodes each step with a few scalar
ops instead.

| Python | Generated kernel |
| --- | --- |
| outer loops whose tiles the outputs are stored at | the output tiles the step loop walks |
| code before the inner loops | runs at a tile's first step |
| each inner (reduction) loop | a phase: the tile's steps for it, one phase after another |
| code after a reduction loop (the stores, after the last) | runs at the phase's last step |
| a reduction loop of one iteration | no step: its body runs inline |
| `x[i : i + T, k : k + T]` that moves every step | a ring of VMEM slots, fetched steps ahead (across phases) |
| `q[b, h, i : i + T, :]` that moves once per tile | two VMEM slots, the next tile's fetched early |
| `bias[0:1, :]` that never moves | fetched once |
| values carried between steps (`acc`) | VMEM scratch |
| `out = out.at[...].set(value)` | output slots copied to HBM asynchronously |
| `a @ b`, `a @ b.T` | MXU matmuls accumulating in f32, `b.T` read transposed in place |
| `if cond: continue` | skipped tiles and steps: no copies, no compute, no step |

Generated code is what runs: `kernel.source(...)` is exactly the module autokernel executes.

## Performance on a TPU v5e

Measured with JAX 0.11.2, wall-clock per call under `jax.jit`, milliseconds; reproduce with
`benchmarks/`.

**Matmul**, 8192³ bf16 (`benchmarks/matmul.py`):

| Kernel | XLA | autokernel 512³ tiles | autokernel 1024×2048×1024 tiles |
| --- | --- | --- | --- |
| dense `x @ y` | 5.905 | 6.320 | **5.739** (192 TFLOP/s) |
| causal `tril(x @ y)` | 5.912 | **3.498** | 3.607 |

The causal kernel skips every tile above the diagonal, so it does half of XLA's work.

**Causal multi-head attention** against JAX's splash attention at the same block sizes, bf16, head dim
128, q pre-scaled for both (`benchmarks/attention.py`). Outputs are identical.

| B, H, S | block_q × block_kv | splash | autokernel | speedup |
| --- | --- | --- | --- | --- |
| 4, 16, 2048 | 256 × 256 | 2.205 | **1.386** | 1.59× |
| 4, 16, 2048 | 512 × 512 | 0.945 | **0.709** | 1.33× |
| 4, 16, 2048 | 1024 × 1024 | 0.872 | **0.803** | 1.09× |
| 1, 16, 4096 | 256 × 256 | 1.876 | **1.260** | 1.49× |
| 1, 16, 4096 | 512 × 512 | 0.778 | **0.606** | 1.28× |
| 1, 16, 4096 | 1024 × 1024 | 0.695 | **0.649** | 1.07× |

## Writing kernels

```python
@autokernel.kernel
def f(x, y, tiles=Tiles()):                   # the input arrays; optionally last, a tunable
    M, K = x.shape                            # prologue: Python on shapes, run when the kernel is built
    TM, TN, TK = tiles.tm, tiles.tn, tiles.tk # (the tunable's fields are plain Python values)
    out = jnp.zeros((M, N), jnp.float32)      # each output allocated once: jnp.zeros or jnp.full
    for i in range(0, M, TM):                 # parallel loops: each iteration stores its own tiles
        for j in range(0, N, TN):
            if j > i:                         # optional tile mask: skip this output tile
                continue
            acc = jnp.zeros((TM, TN))         # code before the reduction loops: once per tile
            for k in range(0, K, TK):         # any number of reduction loops, one after another
                if k > i:                     # optional step mask: skip this step
                    continue
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc)   # each output stored once per tile, at the end
    return out
```

- **Parameters** are the input arrays, then optionally a tunable with a default instance of an
  `@autokernel.tunable` class; tile code reads its fields (`tiles.tm`).
- **Loops** are `range(start, stop, step)` with ints known when the kernel is built (constants, closures,
  shapes, the tunable's fields), a whole number of steps, starting at a multiple of the step.
- **Inputs are read as tiles.** Each index is `v : v + size` or `v` for a loop variable `v`, `:`, a constant
  slice (`8:16`) or a constant int. Integer indices come before sliced dims (`q[b, h, i : i + T, :]`).
  Tiles follow the TPU layout: the last dim a multiple of 128, the second-to-last a multiple of 8 (16 for
  bf16, 32 for 8-bit), each unless it is the whole dim.
- **Outputs are stored as tiles**: every parallel loop places each output tile through exactly one dim
  (`v : v + step`, or `v` for a loop of step 1), covering the dim exactly; other dims are stored whole
  (`:`). Masked-out tiles keep the allocation's value (`jnp.zeros`, `jnp.full(..., -jnp.inf)`), written by
  DMAs that overlap the compute.
- **Tile code** is assignments, augmented assignments and `with jax.named_scope(name):` blocks; use
  `jnp.where` instead of `if`. Values carried from one reduction step to the next (`acc`) are found
  automatically and kept in VMEM, so each keeps one type, the one the loop body gives it: start it with
  that type (`jnp.zeros((TM, TN), jnp.float32)` for `acc += a @ b`, which is f32), or with a number
  (`m = -1e30`). Nothing carries over from one tile to the next. Every tile that runs takes at least one
  step of each reduction loop.
- **Reduction loops** run one after another, each a phase of the tile's steps: a whole transformer block
  (attention over key blocks, then a feed-forward layer over blocks of hidden units) is one kernel. Code
  between them runs once, at the end of the phase before; values it hands to the next phase are carried in
  VMEM like accumulators. A loop of one iteration (`range(0, F, F)`) takes no step: its body runs inline,
  and windows only it moved are fetched once, so a tunable block size covering a whole dim turns streamed
  weights into resident ones.
- **Repeat loops**: a loop around reduction loops (`for l in range(L):` holding a layer's reduction loops
  and code) runs their steps iteration after iteration, carrying values from one iteration to the next as
  from one step to the next. Windows it moves (`w_up[l, :, f : f + bf]`) stream with the phase that reads
  them, so the pipeline fetches the next iteration's (the next layer's weights) while this one runs: a
  whole stack of layers is one kernel, with prefetch across layers. The repeat loop's own code runs at its
  first phase's first step and its last phase's last step, every iteration; the tile's code after the loop,
  as its last iteration ends. Repeat loops don't nest; one of one iteration runs inline.
- **Loop variables** can be used as values (`i + jax.lax.broadcasted_iota(...)`); they behave as Python
  ints (`bf16_tile * i` stays bf16), as int32 in the kernel.
- **Defaults that don't fit** (block sizes too big for VMEM, or not dividing the arrays) are replaced by the
  nearest tunable that builds, and schedules Mosaic can't compile are ruled out while tuning.
- **Masks** are `if cond: continue` at the top of a loop. They are Python conditions on loop variables and
  constants, evaluated once per combination of the loop values they read when the kernel is built; the
  kernel walks SMEM tables of only the tiles and steps that run. Tables cover only the loops a mask reads:
  causal attention's masks don't read the batch and head loops, so its tables hold one batch-head pattern.
  A tile whose mask skips every block of a reduction loop takes one empty step of it (no copies, no body).
- **Index arrays** are int32 inputs read element by element, `lengths[b]` (integer indices only, loop
  variables and constants): the kernel keeps them in SMEM. Their elements are values in tile code
  (`cols < lengths[b]`), integer indices of tile reads (`w[expert[t], :, j : j + tn]`: the window is
  chosen when the kernel runs, clamped to the array as JAX clamps), and conditions in reduction masks
  (`if j >= lengths[b]: continue`). Such **runtime masks** are evaluated on the device just before the
  kernel, for every step at once (Python's `and`, `or`, `not` become their elementwise forms), and the
  kernel loops over only the steps that run: a KV-cache read stops at each sequence's length, a grouped
  matmul visits only the experts whose tokens meet a tile. The function itself branches on array values
  there, so it runs as plain JAX eagerly (not under `jax.jit`); the kernel runs anywhere.
- **Constants** the tile code reads (module globals, closure values, prologue names) are read when a kernel
  is first built for an input shape, as `jax.jit` does, and written into the generated source: numbers,
  strings, dtypes, tuples, modules, and functions (imported, or their source copied).

Anything else is rejected with a `KernelError` naming the line, before anything runs.

The [examples](examples/) are matmul, causal matmul, causal multi-head (splash) attention, RMSNorm fused
with the projection after it, and a whole attention block (attention, RMSNorm, feed-forward layer) in one
kernel, tested against NumPy, JAX's own splash attention, and plain JAX.

**A whole attention block in one kernel** ([examples/attention_block.py](examples/attention_block.py)):
causal multi-head attention, the post-attention RMSNorm and the feed-forward layer
(`gelu(h @ w_up) @ w_down`). The parallel loops walk (batch, query block); the tile's first reduction loop
is flash attention over the key blocks for every head at once (skipping future blocks); the code after it
puts the heads side by side and normalizes them; the second reduction loop is the feed-forward layer over
blocks of hidden units, streaming the weights (or keeping them in VMEM, if the tuner picks a block of all
of them). Nothing between attention and the block's output goes to HBM. Against JAX's splash attention
followed by RMSNorm and the feed-forward layer in XLA, tuned, on a v5e (bf16, ms; `benchmarks/attention_block.py`):

| B | H | S | head dim | F | splash (default blocks) + norm + FFN | splash (its best blocks) + norm + FFN | autokernel | vs default | vs best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 8 | 2048 | 128 | 4096 | 4.226 | 1.245 | **1.106** | 3.82× | 1.13× |
| 8 | 8 | 1024 | 128 | 4096 | 2.603 | 1.099 | **1.022** | 2.55× | 1.08× |
| 2 | 16 | 2048 | 128 | 8192 | 4.933 | 1.951 | **1.821** | 2.71× | 1.07× |
| 1 | 16 | 4096 | 128 | 8192 | 7.330 | 2.199 | **2.066** | 3.55× | 1.06× |
| 4 | 16 | 2048 | 64 | 4096 | 6.784 | 1.837 | **1.714** | 3.96× | 1.07× |
| 1 | 32 | 2048 | 128 | 16384 | 6.118 | 3.539 | **3.264** | 1.87× | 1.08× |

**A whole transformer** ([examples/transformer.py](examples/transformer.py)): a decoder-only model,
tokens to logits (embedding, L pre-norm layers of causal multi-head attention and a GELU feed-forward
layer, final norm, vocab projection), as autokernel kernels composed in plain JAX on (batch, seq, d_model)
activations, a head being a block of columns (no transposes). Per layer: the attention RMSNorm fused with
the QKV projection; then attention, output projection, residual, RMSNorm, feed-forward layer and residual
in one kernel of three phases. With big feed-forward weights the attention's query blocks are too few rows
to reuse them, so the feed-forward layer can run as a kernel of its own instead: each layer runs the form
its kernels' tuned device times say is faster. The QKV projection stays a kernel of its own: attention for
a query block needs every earlier token's keys and values. Against the same model in plain JAX with
splash attention, tuned, on a v5e (bf16, ms; `benchmarks/transformer.py`):

| layers | d_model | heads | FFN | batch | seq | splash default | splash best | autokernel | vs default | vs best | layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1024 | 8 | 4096 | 4 | 2048 | 24.200 | 11.380 | **9.225** | 2.62× | 1.23× | fused |
| 4 | 2048 | 16 | 8192 | 2 | 2048 | 24.048 | 14.902 | **13.902** | 1.73× | 1.07× | split |
| 2 | 2048 | 16 | 8192 | 1 | 4096 | 16.792 | 9.617 | **8.930** | 1.88× | 1.08× | fused |
| 2 | 4096 | 32 | 16384 | 1 | 2048 | 16.553 | 13.092 | **12.721** | 1.30× | 1.03× | split |

(Logits agree with the baseline to 0.03–0.05, one or two bf16 steps. Timings vary by 1–2% from run to
run, tuning included.)

**RMSNorm fused with a projection** ([examples/rmsnorm_matmul.py](examples/rmsnorm_matmul.py)): the row
scale commutes with the matmul, `rmsnorm(x) @ w = (1 / rms(x)) * ((x * gamma) @ w)`, so one reduction
over the model dimension accumulates `(x * gamma) @ w` on the MXU and each row's sum of squares, and the
tile is scaled once at the end. The normalized activations never go to HBM; tuned, it is 1.08–1.16×
faster than XLA's `rmsnorm(x) @ w` on large inputs, and even with it on 512 rows.

## Tuning: test, trace, change the schedule

A kernel's schedule is its tunable (the `@autokernel.tunable` default of its last parameter: block sizes,
or anything the loops read from it) and its buffering (VMEM slots per copy). Call a kernel without its
tunable and, the first time it sees inputs of a shape, autokernel tunes the schedule on the TPU, on random
inputs of that shape, from first principles:

1. **Test** each candidate: its output must match the loops run as plain JAX, or it's out (as is one that
   doesn't build, doesn't fit in VMEM, or Mosaic can't compile).
2. **Trace** it: run it under the JAX profiler. Its device time ranks it, and its trace regions say what
   limits it (below).
3. **Change the schedule** the way the bottleneck says, keeping what's faster: exposed input copies → more
   input buffers, then the neighbouring tiles that move fewer bytes; exposed output copies → more output
   buffers; per-step overhead → the neighbouring tiles with fewer, bigger steps; compute-bound → every
   neighbour, fewest steps first. A neighbour moves one field one step along its choices.

It climbs from the defaults until nothing proposed is faster (at most 12 candidates), and remembers the
result for inputs of that shape. Passing a tunable skips tuning; off a TPU the defaults are used. Tuning
also works when the first call is traced (`jax.jit(matmul)(x, y)`): it runs on a thread of its own.

On a v5e (bf16; device time of the chosen schedule; `benchmarks/`):

| Kernel | defaults | tuned | chosen |
| --- | --- | --- | --- |
| matmul 4096³ from 128³ tiles (`tests/test_tpu.py`) | 7.72 ms | **0.736 ms** (187 TFLOP/s) | 1024³, 4 input buffers |
| matmul 8192³ | 6.35 ms | **5.77 ms** (XLA: 5.90) | 2048 × 1024 × 512 |
| causal matmul 8192³ | 3.48 ms | **3.24 ms** (XLA: 5.91) | 512 × 512 × 1024 |
| attention 4 × 16 × 2048 | 0.700 ms | **0.685 ms** (splash at its best: 0.872) | 512 × 512, 4 input buffers |
| RMSNorm + projection 8192 × 4096 → 4096 | 1.66 ms | **1.49 ms** (XLA unfused: 1.61) | 2048 × 1024 × 512 |
| RMSNorm + projection 16384 × 2048 → 2048 | 0.876 ms | **0.778 ms** (XLA unfused: 0.907) | 2048 × 1024 × 512 |
| RMSNorm + projection 512 × 4096 → 4096 | 0.114 ms | **0.110 ms** (XLA unfused: 0.109) | 512 × 1024 × 512 |

The 128³ run, as `kernel.tuning(x, y)` prints it: the defaults were DMA-bound, so the first move was a
fourth input buffer; then each step of tiles (compute-bound, but with tiny steps) cut the time, until
1024³ tiles, where nothing proposed was faster:

```
chose Small(tm=1024, tn=1024, tk=1024), 4 input / 2 output buffers
   Small(tm=128, tn=128, tk=128), 3 input / 2 output buffers        7722.6 us  dma
   Small(tm=128, tn=128, tk=128), 4 input / 2 output buffers        6021.9 us  compute
   Small(tm=256, tn=128, tk=128), 4 input / 2 output buffers        3428.2 us  compute
   ...
 * Small(tm=1024, tn=1024, tk=1024), 4 input / 2 output buffers      736.1 us  compute
```

### Trace regions

```python
report = autokernel.profile(matmul, x, y, MatmulTiles(tm=256, tn=256, tk=256))  # on a TPU v5e
print(report)
```

```
matmul (MatmulTiles(tm=256, tn=256, tk=256), 3 input / 2 output buffers): 13541.7 us per call over 5 calls
  steps               32768     (0.41 us each)
  compute            9356.9 us
  input waits        2912.4 us  (+0.8 us filling the pipeline)
  output waits         35.9 us
  other              1235.7 us  (decoding steps, starting copies, draining)
  bottleneck: dma
```

(An 8192³ bf16 matmul with 256³ tiles: 22% of its time is copies the pipeline didn't hide.)

To profile, the kernel is generated with **trace regions**: in every step its input waits
(`<kernel>/wait`), its output-slot waits (`<kernel>/out_wait`) and the tile code (`<kernel>/compute`) are
`jax.named_scope`s, which Pallas lowers to Mosaic trace regions and libtpu records with
`--xla_enable_custom_call_region_trace=true`. Importing autokernel sets that flag (if JAX's backend hasn't
started yet; else call `autokernel.enable_profiling()` first thing). Scopes you open in tile code,
`with jax.named_scope("softmax"):`, are reported too. The trace stays in `report.logdir` for TensorBoard
or XProf.

## API

| | |
| --- | --- |
| `@autokernel.tunable`, `autokernel.choice(...)` | declare what a kernel's schedule may be |
| `@autokernel.kernel` | compile a loop nest; the result is called like the function, with or without its tunable |
| `kernel.tuning(*inputs)` | the `Tuning` for inputs like these: every trial, and the `config` chosen |
| `kernel.lower(*args)` | the `Lowered` kernel for inputs like these (arrays or `jax.ShapeDtypeStruct`s) and tunable: `source`, `plan`, `schedule`, `config`, `steps`, `tiles`, `vmem_bytes` |
| `kernel.source(*args)` | the generated Pallas source |
| `kernel.reference(*args)` | the loop nest run as plain JAX |
| `kernel.verify(*args, rtol=None, atol=None)` | run both and compare; the largest difference |
| `autokernel.profile(kernel, *args)` | a `KernelProfile` from the device trace |
| `autokernel.enable_profiling()` | have libtpu record trace regions (if autokernel was imported after JAX started) |

## Development

```bash
uv sync                                  # CPU: kernels run in the Pallas TPU interpreter
uv run pytest --cov=autokernel           # 100% line and branch coverage is required
uv run ruff check . && uv run ruff format --check .
uv run python examples/causal_attention.py
uv run python benchmarks/attention.py    # on a TPU v5e
```

On CPU the tests run every kernel in the TPU interpreter with its race detector, and cross-lower every
kind of kernel to Mosaic for a v5e, which catches what the interpreter can't. `tests/test_tpu.py` runs on a
v5e only: kernels natively, against splash attention, real profiles, and tuning. [AGENTS.md](AGENTS.md) describes
the architecture, the conventions, and how to run on a TPU.

```
src/autokernel/
  parse.py      the loop-nest language: Python source -> Nest (syntax only)
  plan.py       a Nest for concrete shapes: loops, tiles, operands, carried values, types
  schedule.py   step order; masks evaluated into SMEM step and tile tables
  tunable.py    @autokernel.tunable: the pytree of choices a kernel gives autokernel
  tune.py       the autotuner: test, trace, change the schedule
  pipeline.py   buffering (slots per copy) and VMEM accounting
  codegen.py    the gridless Pallas source
  mxu.py        what `@` means, for kernels and references alike
  reference.py  the function as plain JAX
  api.py        @autokernel.kernel, Kernel, Lowered, verify
  profiling.py  trace regions -> KernelProfile
  tilecode.py   names tile code reads and assigns; output stores
  v5e.py        the chip: memories and layout rules
  errors.py     KernelError
examples/       matmul, causal matmul, causal attention (+ JAX splash), RMSNorm fused with a projection,
                a whole attention block, a whole transformer
benchmarks/     timing and profiling scripts for a TPU v5e
tests/
```
