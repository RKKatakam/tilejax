# autokernel

Write a TPU kernel as ordinary Python tile loops; get a fast, gridless
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel.

```python
import jax.numpy as jnp
import autokernel

M, K, N = 4096, 4096, 4096
TM, TN, TK = 512, 512, 512


@autokernel.tpu
def matmul(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc)
    return out


out = matmul(x, y)  # runs the generated kernel
print(matmul.source(x, y))  # the generated Pallas code
matmul.verify(x, y)  # checks the kernel against the Python loops
```

The Python function is the specification: `verify` runs it eagerly and compares. The generated kernel
is plain Pallas you can read, copy and edit.

autokernel targets **TPU only**. Without a TPU it plans for a named chip (a v5e by default) and runs
kernels in the Pallas TPU interpreter, so you can develop, test and tune on a laptop.

## How it works

Every kernel is **gridless**: one `pl.loop` over all of its steps (an output tile × a reduction
position), with hand-written, multi-buffered DMAs, the way you'd write a performance-critical Pallas
kernel by hand. A Pallas grid would add 15–25 ns of pipeline bookkeeping to every step on a v5e; this
loop instead decodes each step with a few scalar ops.

| Python | Generated kernel |
| --- | --- |
| outer loops whose tiles the outputs are written in | the output tiles the loop walks |
| statements between those loops | run once per tile |
| inner loops that update values | reduction steps of each tile, one phase per loop (nest) |
| `x[i:i + T, k:k + T]`, `q[b, h, i:i + T, :]` | async copies into a ring of VMEM slots, prefetched ahead |
| inputs that only change between tiles (a bias row) | fetched once per tile, not per step |
| values carried between steps (accumulators) | VMEM scratch (SMEM for scalars) |
| `out = out.at[...].set(value)` | two VMEM slots per output, copied to HBM asynchronously |
| `a @ b`, `a @ b.T`, batched `@` | MXU matmuls accumulating in f32 (i32 for integers) |
| `if cond: continue` on loop coordinates | skipped tiles or steps: no copies, no compute, no step |

On chips with two TensorCores per device (v4, v5p) the work is split between them on tile boundaries.
Each kernel also carries a `pl.CostEstimate` for XLA and a VMEM limit sized from its own needs.

## Performance

Measured on a TPU v5e with autokernel 0.2, milliseconds, lower is better. Reproduce with `benchmarks/`.

### Matmul

8192³, median of 3 × 20 runs. "Hand-written" is a manual-DMA gridless Pallas kernel written by hand for
the same tiles.

| Kernel | Tiles | XLA | Hand-written | autokernel |
| --- | --- | --- | --- | --- |
| causal `tril(x @ y)`, bf16 | 128³ | 5.91 | 46.73 | **44.94** |
| causal `tril(x @ y)`, bf16 | 1024×1024×512 | 5.91 | 3.37 | **3.35** |
| causal `tril(x @ y)`, f32 | 512×512×1024 | 35.15 | 18.12 | **18.08** |
| dense `x @ y`, bf16 | 1024×1024×512 | 5.90 | 5.92 | **5.89** |
| dense `x @ y`, f32 | 512×512×1024 | 35.08 | 34.05 | **33.99** |

autokernel matches or beats the hand-written kernel at almost every tile size measured (by up to 4% with
small tiles, where per-step overhead dominates). The causal matmul is about 1.75× faster than XLA in bf16
and 1.9× in f32, because XLA computes the whole matrix.

### Causal multi-head attention

bf16, head dim 128, median of 3 × 10 runs. "JAX Splash" is
`jax.experimental.pallas.ops.tpu.splash_attention` with a causal mask and the same block sizes.
TFLOP/s counts the causal half of the work. Reproduce with `benchmarks/attention.py`.

Batch 4, 16 heads, sequence 4096. XLA (`jax.nn.dot_product_attention(is_causal=True)`): 32.32 ms.

| Blocks (q × kv) | JAX Splash | autokernel | Speedup |
| --- | --- | --- | --- |
| 256 × 256 | 8.31 (33 TFLOP/s) | **5.25** (52 TFLOP/s) | 1.58× |
| 512 × 256 | 5.28 (52 TFLOP/s) | **3.59** (77 TFLOP/s) | 1.47× |
| 512 × 512 | 3.38 (82 TFLOP/s) | **2.40** (115 TFLOP/s) | 1.41× |
| 1024 × 512 | 3.16 (87 TFLOP/s) | **2.51** (110 TFLOP/s) | 1.26× |

Batch 2, 16 heads, sequence 8192. XLA runs out of HBM (it materialises the 16 GB score matrix).

| Blocks (q × kv) | JAX Splash | autokernel | Speedup |
| --- | --- | --- | --- |
| 256 × 256 | 16.04 (34 TFLOP/s) | **9.94** (55 TFLOP/s) | 1.61× |
| 512 × 256 | 9.89 (56 TFLOP/s) | **6.58** (84 TFLOP/s) | 1.50× |
| 512 × 512 | 6.19 (89 TFLOP/s) | **4.36** (126 TFLOP/s) | 1.42× |
| 1024 × 512 | 5.53 (99 TFLOP/s) | **4.37** (126 TFLOP/s) | 1.27× |

Both kernels' outputs agree with XLA (or, at sequence 8192, with each other) to within one bf16 rounding
step.

### Tuning

`examples/tuned_matmul.py` (2048³ bf16, 27 block-size candidates) on a v5e: the cost model's pick,
512³, was also the fastest measured, and `autotune="measure"` then chose 3 input buffers over 2
(0.132 ms against 0.146 ms).

## Writing kernels

```python
@autokernel.tpu
def f(x, y, mask):  # positional array parameters
    M, K = x.shape  # optional prologue: shape-level constants
    out = jnp.zeros((M, N))  # one allocation per output
    stats = jnp.zeros((M, 128))
    for i in range(0, M, TM):  # parallel loops: the tiles of the outputs
        scale = i / M  # statements between them run once per tile
        for j in range(0, N, TN):
            if j > i + TM - 1:  # optional tile mask
                continue
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):  # any number of reduction loops, each possibly a perfect nest
                if mask[i // TM, k // TK] == 0:  # optional reduction mask, here read from an input
                    continue
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            out = out.at[i : i + TM, j : j + TN].set(acc * scale)  # one store per output, at the end
            stats = stats.at[i : i + TM, :].set(...)
    return out, stats  # one output or a tuple
```

- Loops are `range(start, stop, step)` with values known at trace time (constants, closures, `x.shape`).
- Inputs are read through tiles: each dimension is `v:v + T` (T equal to loop `v`'s step), a constant
  aligned block `c:c + T`, `:`, or an integer index: a loop variable stepping by 1 (`q[b, h, ...]`) or a
  constant. Integer-indexed dimensions are dropped from the tile, as in NumPy.
- Tile bodies are assignments (`name = expr`, `name op= expr`) and reduction loops. Values that must
  survive between steps are found automatically and kept in VMEM.
- Loop variables may be used as values, e.g. `jax.lax.broadcasted_iota(...) + i`.
- Every output element must be produced: each output's tiles cover it, apart from masked-out tiles.

Anything else is rejected with an `AutoKernelError` naming the line, before anything runs.

### Masks

A mask is `if cond: continue` at the top of a loop body, or `if cond:` around all of it.

- A **tile mask** (on the innermost parallel loop) skips output tiles. Skipped tiles keep the output
  allocation's value: when that's a known constant (`jnp.zeros`, `jnp.full(..., -inf)`, ...) the kernel
  writes them with async DMAs that overlap the compute; otherwise the output starts as the allocation.
- A **reduction mask** skips reduction steps. A tile whose mask skips every step still runs the code
  around the loop, just as the Python does.
- Masks over loop coordinates and constants are evaluated at trace time, and the kernel walks a table of
  only the steps that run. When the mask ignores the outer loops (causal attention ignores batch and
  head), the table holds one repeating pattern, so it stays small.
- A mask may also **read input arrays** (`block_mask[i // T, k // T]`): the step table is then built on
  device from the actual mask, and the kernel's step count is decided at run time.

[examples/splash_attention.py](examples/splash_attention.py) is causal multi-head splash attention written
this way.

### Writing fast kernels

Keep per-row statistics lane-wide: shape `(rows, 128)` with the value repeated across the 128 lanes of a
TPU vector register, widened with `jnp.tile` where they meet a wider block. A `(rows, 1)` column needs a
cross-lane broadcast every time it meets a `(rows, n)` block; in the attention kernel, switching the
running max and sum to the lane-wide form made it about 1.7× faster on a v5e.

## Tuning block sizes

Name the constants the loops use and the values to try:

```python
@autokernel.tpu(tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (128, 256, 512)})
def matmul(x, y): ...
```

autokernel builds every combination, drops those that break TPU layout rules, don't divide the loops or
don't fit in VMEM, and ranks the rest with a roofline cost model:

- per-unit FLOPs and transcendentals from JAX's own `cost_analysis()` on tile-shaped inputs, with the
  matmul FLOPs (counted from the jaxpr) on the MXU and the rest on the VPU;
- HBM traffic per step and per tile;
- the chip's numbers from `pltpu.get_tpu_info()`: HBM bandwidth, bf16 MXU throughput, VMEM and SMEM
  capacity and TensorCores;
- a measured fixed cost per step.

With `autotune="measure"` it also times the best few candidates on the TPU (on the first call with
concrete arrays) and tries 2 and 3 input buffers. `kernel.tuning(*args)` lists every candidate with its
predicted and measured time, or why it was rejected. [examples/tuned_matmul.py](examples/tuned_matmul.py)
shows it.

## Options

```python
@autokernel.tpu(
    interpret=None,              # None: native on TPU, interpreter elsewhere. False: require a TPU.
    verify=False,                # check against the Python loops on the first call per input shape
    tune=None,                   # {constant: candidate values} to search
    autotune="model",            # "model": cost model only. "measure": also time the best few on the TPU.
    buffers=None,                # VMEM slots per streamed input (2 = double buffering); default 2, or tuned
    target=None,                 # the TPU to plan for off-TPU: "v5e" (default), "v5p", "v6e", ...
    cores=None,                  # TensorCores to split across; default: all of the target's
    check_tiling=True,           # reject tiles that break TPU (8, 128) layout rules
    vmem_limit_bytes=None,       # Mosaic's VMEM limit; default sized from the kernel's estimated use
    matmul_precision="highest",  # "default" uses a single bf16 MXU pass for f32 (faster, less exact)
    background_fill=True,        # write masked-out tiles with in-kernel DMAs
)
```

`kernel.lower(*args)` returns a `Lowered` with the schedule, step and tile counts, operand tiles, state
variables, cost estimate, VMEM limit and generated `source`. Arguments can be arrays or
`jax.ShapeDtypeStruct`s.

## TPU rules it enforces

- Tile shapes: the last dimension must be a multiple of 128 or the whole array dimension; the
  second-to-last a multiple of 8 for 32-bit types (16 for bf16, 32 for 8-bit) or the whole dimension.
- f32 matmuls use `Precision.HIGHEST` by default: Mosaic's default is a single bf16 pass, about 0.2 off
  at K=384. bf16 inputs are exact in one pass, accumulated in f32.
- Narrow float accumulators (bf16, f16) are widened to f32 scratch.
- VMEM use is estimated (input rings, output slots, state, intermediates) and must fit the chip; the
  kernel's VMEM limit is set from the estimate.
- Step tables live in SMEM and must fit in half of it (512 KiB on v4 and newer). Kernels with no masks
  need no tables at all.

## Install

```bash
uv sync                      # CPU development: kernels run in the Pallas TPU interpreter
uv sync --extra tpu          # on a TPU VM
```

## Development

```bash
uv run pytest                           # all tests: on CPU via the TPU interpreter, or natively on a TPU
uv run ruff check . && uv run ruff format --check .
uv run python examples/splash_attention.py
uv run python benchmarks/matmul.py      # timings; meaningful only on a TPU
uv run python benchmarks/attention.py   # autokernel vs JAX splash attention vs XLA
```

Besides running kernels in the interpreter (with its race detector), the tests cross-lower every kind of
kernel to Mosaic for a v5e, and the two-TensorCore form for a v5p, which catches lowering errors the
interpreter can't see. Timing still needs real hardware.

## Layout

```
src/autokernel/
  frontend.py   parse the Python loop nest
  analysis.py   resolve it for concrete shapes: loops, tiles, operands, outputs, masks, state variables
  schedule.py   order and decode the steps: arithmetic, static step tables, or tables built on device
  codegen.py    emit the gridless Pallas kernel and its pallas_call wrapper
  cost.py       the roofline cost model (JAX cost analysis + TPU specs)
  lowering.py   options, and the pipeline from parsed loops to a compiled kernel
  tune.py       block-size search
  hardware.py   TPU specs (from pltpu.get_tpu_info) and layout rules
  api.py        the @autokernel.tpu decorator
examples/       runnable examples
benchmarks/     timing scripts for a TPU
tests/
```
