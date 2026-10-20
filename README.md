# autokernel

Write a TPU kernel as ordinary Python loops over tiles; get a fast, **gridless**
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel for a **TPU v5e**.

```python
import jax.numpy as jnp
import autokernel

TM, TN, TK = 512, 512, 512


@autokernel.kernel
def matmul(x, y):
    M, K = x.shape
    N = y.shape[1]
    dtype = x.dtype
    out = jnp.zeros((M, N), dtype)
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN), jnp.float32)
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc.astype(dtype))
    return out


out = matmul(x, y)          # runs the generated kernel
print(matmul.source(x, y))  # the generated Pallas source: plain, standalone, readable
matmul.verify(x, y)         # checks the kernel against the loops run as plain JAX
```

The Python function is the specification: `verify` runs it as ordinary JAX and compares. The one thing a
kernel does differently from plain JAX is `@`, which accumulates in f32 (i32 for integers) as the MXU does,
instead of rounding `bf16 @ bf16` to bf16; `verify` uses the same rule.

autokernel targets **one chip, the TPU v5e**. Without a TPU, kernels run in the Pallas TPU interpreter, so
you can develop and test them anywhere.

## How it works

Every kernel is **gridless**: one `pl.loop` over all of its steps (an output tile × a reduction block),
with hand-written, multi-buffered DMAs, the way you would write a performance-critical Pallas kernel by
hand. A Pallas grid adds pipeline bookkeeping to every step; this loop decodes each step with a few scalar
ops instead.

| Python | Generated kernel |
| --- | --- |
| outer loops whose tiles the outputs are stored at | the output tiles the step loop walks |
| code before the inner loop | runs at a tile's first step |
| the inner (reduction) loop | the tile's steps |
| code after it, and the stores | runs at a tile's last step |
| `x[i : i + T, k : k + T]` that moves every step | a ring of VMEM slots, fetched steps ahead |
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
def f(x, y):                                  # positional parameters: the input arrays
    M, K = x.shape                            # prologue: Python on shapes, run when the kernel is built
    out = jnp.zeros((M, N), jnp.float32)      # each output allocated once: jnp.zeros or jnp.full
    for i in range(0, M, TM):                 # parallel loops: each iteration stores its own tiles
        for j in range(0, N, TN):
            if j > i:                         # optional tile mask: skip this output tile
                continue
            acc = jnp.zeros((TM, TN))         # pre: runs once per tile
            for k in range(0, K, TK):         # at most one reduction loop
                if k > i:                     # optional step mask: skip this step
                    continue
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc)   # post: each output stored once per tile
    return out
```

- **Loops** are `range(start, stop, step)` with ints known when the kernel is built (constants, closures,
  shapes), a whole number of steps, starting at a multiple of the step.
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
  reduction step.
- **Loop variables** can be used as values (`i + jax.lax.broadcasted_iota(...)`); they behave as Python
  ints (`bf16_tile * i` stays bf16), as int32 in the kernel.
- **Masks** are `if cond: continue` at the top of a loop. They are Python conditions on loop variables and
  constants, evaluated once per combination of the loop values they read when the kernel is built; the
  kernel walks SMEM tables of only the tiles and steps that run. Tables cover only the loops a mask reads:
  causal attention's masks don't read the batch and head loops, so its tables hold one batch-head pattern.
- **Constants** the tile code reads (module globals, closure values, prologue names) are read when a kernel
  is first built for an input shape, as `jax.jit` does, and written into the generated source: numbers,
  strings, dtypes, tuples, modules, and functions (imported, or their source copied).

Anything else is rejected with a `KernelError` naming the line, before anything runs.

The [examples](examples/) are matmul, causal matmul, and causal multi-head (splash) attention, tested
against NumPy and against JAX's own splash attention.

## Profiling with named scopes

Profiling is the one way autokernel looks at performance. There are no tuning knobs: block sizes are what
you write in the loops, and buffering is chosen from profiles.

```python
import autokernel
autokernel.enable_profiling()              # first thing: before JAX starts its TPU backend

report = autokernel.profile(matmul, x, y)  # on a TPU v5e
print(report)
```

```
matmul (3 input / 2 output buffers): 13541.7 us per call over 5 calls
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
`--xla_enable_custom_call_region_trace=true` (what `enable_profiling` sets). Scopes you open in tile code,
`with jax.named_scope("softmax"):`, are reported too. The trace stays in `report.logdir` for TensorBoard
or XProf.

`autokernel.optimize` uses those regions to make the kernel faster:

```python
result = autokernel.optimize(matmul, x, y)  # the kernel keeps the result for inputs like these
print(result)
```

While a profile shows exposed input waits it adds an input slot (fetches start a step further ahead), and
while it shows output waits an output slot, keeping each change that makes the kernel faster. On a v5e
(device time, trace regions on):

| Kernel | before | after |
| --- | --- | --- |
| matmul 8192³, 256³ tiles | 13.54 ms | **11.86 ms** (4 input buffers) |
| causal matmul 8192³, 512³ tiles | 3.50 ms | **3.37 ms** (4 input buffers) |
| attention 4×16×2048, 512 × 512 blocks | 700 µs | **685 µs** (4 input buffers) |
| matmul 8192³, 512³ tiles | 6.36 ms | 6.36 ms (compute-bound: unchanged) |

## API

| | |
| --- | --- |
| `@autokernel.kernel` | compile a loop nest; the result is called like the function |
| `kernel.lower(*inputs)` | the `Lowered` kernel for inputs like these (arrays or `jax.ShapeDtypeStruct`s): `source`, `plan`, `schedule`, `pipeline`, `steps`, `tiles`, `vmem_bytes` |
| `kernel.source(*inputs)` | the generated Pallas source |
| `kernel.reference(*inputs)` | the loop nest run as plain JAX |
| `kernel.verify(*inputs, rtol=None, atol=None)` | run both and compare; the largest difference |
| `autokernel.enable_profiling()` | have libtpu record trace regions (before JAX starts) |
| `autokernel.profile(kernel, *inputs)` | a `KernelProfile` from the device trace |
| `autokernel.optimize(kernel, *inputs)` | buffering chosen from profiles: an `Optimization` |

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
v5e only: kernels natively, against splash attention, and real profiles. [AGENTS.md](AGENTS.md) describes
the architecture, the conventions, and how to run on a TPU.

```
src/autokernel/
  parse.py      the loop-nest language: Python source -> Nest (syntax only)
  plan.py       a Nest for concrete shapes: loops, tiles, operands, carried values, types
  schedule.py   step order; masks evaluated into SMEM step and tile tables
  pipeline.py   buffering (slots per copy) and VMEM accounting
  codegen.py    the gridless Pallas source
  mxu.py        what `@` means, for kernels and references alike
  reference.py  the function as plain JAX
  api.py        @autokernel.kernel, Kernel, Lowered, verify
  profiling.py  trace regions -> KernelProfile; optimize
  tilecode.py   names tile code reads and assigns; output stores
  v5e.py        the chip: memories and layout rules
  errors.py     KernelError
examples/       matmul, causal matmul, causal attention (+ JAX splash for comparison)
benchmarks/     timing and profiling scripts for a TPU v5e
tests/
```
