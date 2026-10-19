# autokernel

Write a TPU kernel as ordinary Python tile loops; get a [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
kernel that runs on TPU.

```python
import jax.numpy as jnp
import autokernel

M, K, N = 1024, 1024, 1024
TM, TN, TK = 256, 256, 256


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


out = matmul(x, y)  # runs the generated pl.pallas_call
print(matmul.source(x, y))  # the generated Pallas code
matmul.verify(x, y)  # checks the kernel against the Python loops
```

The Python function is the specification: `verify` runs it eagerly and compares. The generated kernel
is plain Pallas you can read, copy and edit.

autokernel targets **TPU only**. Without a TPU it runs kernels in the Pallas TPU interpreter, so you
can develop and test on a laptop.

## What it generates

By default autokernel generates a **gridless** kernel (`pipeline="manual"`): one loop over every step
(output tile × reduction index) with hand-written, double-buffered DMAs, the way you'd write a
performance-critical Pallas kernel by hand. A Pallas grid adds roughly 15–25 ns of pipeline bookkeeping
to every step on a v5e; the gridless loop carries its tile and reduction counters in registers instead.

| Python | Generated Pallas (manual pipeline) |
| --- | --- |
| outer `for` loops whose tiles the output is written in | the tiles one loop walks, from SMEM tables |
| an inner loop that updates an accumulator | the reduction steps of each tile |
| `x[i:i + T, k:k + T]` | async copies into a 2-slot VMEM buffer, prefetched one step ahead |
| `acc = jnp.zeros(...)` before the inner loop | a VMEM scratch buffer, zeroed on the first step |
| `out = out.at[...].set(value)` | a 2-slot VMEM output buffer, copied to HBM asynchronously |
| `a @ b` | an MXU matmul accumulating in f32 (i32 for integer inputs) |
| `if cond:` / `if cond: continue` on tile coordinates | the loop visits the active tiles only |

Inputs whose block only changes between tiles (a bias row, say) are fetched once per tile, not once per
step. `pipeline="grid"` generates the Pallas-grid version instead: outer loops become `"parallel"` grid
axes and slices become `BlockSpec`s, which chips with two TensorCores can split between them.

## Performance

TPU v5e, 8192³, milliseconds (median of 3 × 20 runs). "Hand-written" is a manual-DMA gridless Pallas
kernel written by hand for the same tiles.

| Kernel | Tiles | XLA | Hand-written | autokernel (manual) | autokernel (grid) |
| --- | --- | --- | --- | --- | --- |
| causal `tril(x @ y)`, bf16 | 128³ | 5.91 | 46.74 | **44.83** | 49.12 |
| causal `tril(x @ y)`, bf16 | 1024×1024×512 | 5.91 | 3.36 | **3.35** | 3.39 |
| causal `tril(x @ y)`, f32 | 512×512×1024 | 35.16 | 18.14 | **18.08** | 18.16 |
| dense `x @ y`, bf16 | 1024×1024×512 | 5.90 | 5.92 | **5.89** | 5.92 |
| dense `x @ y`, f32 | 512×512×1024 | 35.08 | 34.06 | **34.00** | 34.04 |

The gridless backend matches or beats the hand-written kernel at every tile size measured (by up to 4%
with small tiles, where per-step overhead dominates). A causal matmul with the tile mask is about 1.75×
faster than XLA in bf16 and 1.9× in f32, because XLA computes the whole matrix.

## Supported code

```python
@autokernel.tpu
def f(x, y):  # positional array parameters only
    M, K = x.shape  # optional prologue: shape-level constants
    out = jnp.zeros((M, N))  # the output allocation
    for i in range(0, M, TM):  # one or more parallel tile loops
        for j in range(0, N, TN):
            if j > i + TM - 1:  # optional tile mask (see below)
                continue
            acc = jnp.zeros((TM, TN))  # optional accumulators and temporaries
            for k in range(0, K, TK):  # at most one reduction loop
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            y_tile = jnp.maximum(acc, 0.0)  # optional epilogue
            out = out.at[i : i + TM, j : j + TN].set(y_tile)
    return out
```

- Loops are `range(start, stop, step)` with values known at trace time (constants, closures, `x.shape`).
- Inputs are only read through tile slices: each dimension is `v:v + T` (T equal to loop `v`'s step),
  a constant aligned block `c:c + T`, or `:`.
- Loop variables may also be used as values, e.g. `jax.lax.broadcasted_iota(...) + i`.
- Every output element must be produced: the loops must cover the output.

Anything else is rejected with an `AutoKernelError` naming the line, before anything runs.

### Tile masks

Guard the tile body with a condition on the parallel loop variables to skip tiles, for example a
causal (lower-triangular) matmul:

```python
@autokernel.tpu
def causal_matmul(x, y):
    out = jnp.zeros((M, N))
    for i in range(0, M, TM):
        for j in range(0, N, TN):
            if j > i + TM - 1:  # every entry of this tile is above the diagonal
                continue
            acc = jnp.zeros((TM, TN))
            for k in range(0, K, TK):
                acc += x[i : i + TM, k : k + TK] @ y[k : k + TK, j : j + TN]
            rows = i + jax.lax.broadcasted_iota(jnp.int32, (TM, TN), 0)
            cols = j + jax.lax.broadcasted_iota(jnp.int32, (TM, TN), 1)
            out = out.at[i : i + TM, j : j + TN].set(jnp.where(cols <= rows, acc, 0.0))
    return out
```

The mask is evaluated for every tile at trace time, and the kernel visits only the active tiles
(their coordinates live in SMEM tables), so skipped tiles cost no compute, no copies and no steps.
Skipped tiles keep the output allocation's value. When that value is a known constant (`jnp.zeros`,
`jnp.full(..., -inf)`, and so on), the kernel writes those tiles itself with async DMAs that overlap the
matmuls, so the output never needs a separate fill pass.

On a TPU v5e at 8192³ this runs 1.65–1.9× faster than the same loops without the mask.

## Options

```python
@autokernel.tpu(
    interpret=None,              # None: native on TPU, interpreter elsewhere. False: require a TPU.
    verify=False,                # check against the Python loops on the first call per input shape
    check_tiling=True,           # reject tiles that break TPU (8, 128) layout rules
    vmem_limit_bytes=None,       # raise Mosaic's scoped-VMEM limit for big tiles
    matmul_precision="highest",  # "default" uses a single bf16 MXU pass for f32 (faster, less exact)
    background_fill=True,        # write masked-out tiles with in-kernel DMAs (see Tile masks)
    pipeline="manual",           # "manual": gridless, hand-pipelined (fastest). "grid": Pallas grid.
)
```

`TpuKernel.lower(*args)` returns a `Lowered` with the pipeline, grid and step count, operand tiles, VMEM
estimate, tile-mask statistics and generated `source`. Arguments can be arrays or `jax.ShapeDtypeStruct`s.

## TPU rules it enforces

- Block shapes: the last dimension must be a multiple of 128 or the whole array dimension; the
  second-to-last a multiple of 8 for 32-bit types (16 for bf16, 32 for 8-bit) or the whole dimension.
- f32 matmuls use `Precision.HIGHEST` by default: Mosaic's default is a single bf16 pass, about
  0.2 off at K=384. bf16 inputs are exact in one pass, accumulated in f32.
- Narrow float accumulators (bf16, f16) are widened to f32 VMEM scratch.
- A VMEM estimate (double-buffered blocks plus scratch) warns when it exceeds the scoped budget.

## Install

```bash
uv sync                      # CPU development: kernels run in the Pallas TPU interpreter
uv sync --extra tpu          # on a TPU VM
```

## Development

```bash
uv run pytest                # all tests, on CPU via the TPU interpreter (or natively on a TPU)
uv run ruff check . && uv run ruff format --check .
uv run python examples/causal_matmul.py
uv run python benchmarks/matmul.py   # timings; meaningful only on a TPU
```

The test suite also cross-lowers kernels to Mosaic for a TPU v5e on CPU, which catches lowering
errors the interpreter can't see. Timing still needs real hardware.

## Layout

```
src/autokernel/
  frontend.py   parse the Python loop nest into a LoopNest
  analysis.py   resolve it for concrete input shapes: loops, tiles, operands, masks, accumulators
  emit.py       what both backends share: options, the tile body, fill DMAs, hardware checks
  manual.py     the gridless backend: one loop, hand-written double-buffered DMAs (default)
  grid.py       the grid backend: a Pallas grid with BlockSpecs
  codegen.py    pick a backend, generate and compile the kernel
  hardware.py   TPU layout and VMEM rules
  api.py        the @autokernel.tpu decorator
examples/       runnable examples
benchmarks/     timing scripts for a TPU
tests/
```

## Limitations

- One reduction loop per kernel, perfectly nested parallel loops, one output.
- Tile masks depend only on tile coordinates known at trace time: no runtime mask arrays, and no
  masks on the reduction loop.
- `@` on 3-D or higher tiles is lowered as `jnp.dot`, which isn't batched matmul semantics; only
  use it on 2-D tiles.
- Gridless kernels run on one TensorCore. On chips with two per device (v4, v5p), use
  `pipeline="grid"` (with `background_fill=False` for masked kernels) so the parallel grid axes can be
  split between them. v5e and v6e have one TensorCore per chip.
