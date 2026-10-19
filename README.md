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

The Python function is the specification: `verify` runs it eagerly and compares. The one thing the
kernel does differently from plain JAX is `@`: it accumulates in f32 (i32 for integers) like the MXU,
instead of rounding `bf16 @ bf16` to bf16 or wrapping `int8 @ int8`; `verify` compares against that. The
generated kernel is plain Pallas you can read, copy and edit: `source` carries its imports, constants and
tables, so it runs on its own.

The same decorator compiles a **whole model**: a function with many loop nests, JAX code between them and
Python loops over layers becomes a program of gridless kernels, fused where they run back to back, with
`jax.grad`, `jax.vmap`, `jax.jvp` and sharded inputs working on it. Block sizes can be left to the tuner
(`tune="auto"`).

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

On chips with two TensorCores per device (v4, v5p) the work is split between them on a tile boundary
chosen so that both cores run about the same number of steps, even when a mask gives tiles very different
step counts. Each kernel also carries a `pl.CostEstimate` for XLA, and the chip's whole scoped VMEM as its limit.

## Performance

Measured on a TPU v5e with autokernel 0.3 (JAX 0.11.2), milliseconds, lower is better. Reproduce with `benchmarks/`.

### Matmul

8192³, mean of 20 runs after a warm-up. autokernel picks its block sizes the way XLA does: `benchmarks/matmul.py` tunes
each kernel over a grid of 18 block sizes (the cost model ranks them, the best eight are timed), with 2
or 3 input buffers.

| Kernel | XLA | autokernel | Blocks chosen |
| --- | --- | --- | --- |
| dense `x @ y`, bf16 | 5.90 | **5.73** | 1024×2048×1024 |
| dense `x @ y`, f32 | 35.08 | **33.84** | 512×1024×1024 |
| causal `tril(x @ y)`, bf16 | 5.91 | **3.26** | 512×512×1024 |
| causal `tril(x @ y)`, f32 | 35.15 | **18.10** | 512×512×1024 |

The dense kernels run at 192 TFLOP/s (bf16, 97% of the chip's 197) and 32.5 TFLOP/s (f32 at
`Precision.HIGHEST`), 3–4% ahead of XLA; the causal ones are 1.8× faster than XLA, which computes the
whole matrix. At 4096³ the bf16 dense kernel is 3% behind XLA (0.75 against 0.73 ms): with only eight
output tiles the pipeline's first fetch and last write-back aren't hidden.

Block sizes matter more than anything in the kernel. Each step costs a fixed ~0.2 µs on top of its
DMAs and matmul, so a 512³ bf16 block (1.3 µs of DMA against 1.4 µs of MXU time) leaves no slack and runs
at 174 TFLOP/s, and 128³ blocks run at 18 TFLOP/s. The tuner avoids both; hand-picked tiles should be
checked with `kernel.tuning(x, y)`.

### Causal multi-head attention

bf16, head dim 128, causal, under `jax.jit`. Both kernels use **the same block sizes** (JAX splash with
`block_kv_compute = block_kv`); q is pre-scaled for both. Reproduce with `benchmarks/attention.py`.

Batch 4, 16 heads, sequence 2048: JAX splash / autokernel, ms (rows: block_q; columns: block_kv).

| block_q | splash 256 / 512 / 1024 | autokernel 256 / 512 / 1024 | splash ÷ autokernel |
| --- | --- | --- | --- |
| 256 | 2.212 / 1.271 / 1.103 | 1.409 / 0.854 / 0.915 | 1.57× / 1.49× / 1.21× |
| 512 | 1.454 / 0.952 / 0.945 | 1.049 / **0.725** / 0.837 | 1.39× / 1.31× / 1.13× |
| 1024 | 1.146 / 0.956 / 0.880 | 0.947 / 0.828 / 0.814 | 1.21× / 1.15× / 1.08× |

Batch 1, 16 heads, sequence 4096:

| block_q | splash 256 / 512 / 1024 | autokernel 256 / 512 / 1024 | splash ÷ autokernel |
| --- | --- | --- | --- |
| 256 | 1.883 / 1.063 / 0.887 | 1.297 / 0.735 / 0.729 | 1.45× / 1.45× / 1.22× |
| 512 | 1.211 / 0.790 / 0.750 | 0.926 / **0.625** / 0.676 | 1.31× / 1.26× / 1.11× |
| 1024 | 0.901 / 0.755 / 0.703 | 0.760 / 0.667 / 0.659 | 1.19× / 1.13× / 1.07× |

Splash's own best configuration (1024×1024 blocks computing 512 keys at a time) reaches 0.81 / 0.64 ms,
against autokernel's best of 0.725 / 0.625 ms at 512×512 (1.12× / 1.02×). The difference is a fixed cost
per step, not the block size: splash's Pallas grid pays roughly 150–450 ns more per step (pipeline
bookkeeping, and its running statistics passed as pipeline outputs rather than scratch), and its causal
grid also visits the (query block, future key block) pairs autokernel's schedule never runs. So the gap is
largest with small blocks and smallest with large ones. Outputs agree with XLA to one bf16 rounding step.

### A transformer stack

`benchmarks/transformer.py`: 4 layers of pre-norm causal attention (16 heads of 128) and a ReLU MLP, bf16,
D=2048, F=8192, written as five tile loop nests per layer (RMSNorm folded into the QKV and up projections;
a head is a block of columns, so no reshapes). The layer loop runs as a `fori_loop`; matmul block sizes come
from a sweep. "XLA + splash" is the same model in plain JAX with JAX's splash attention **at the same
attention block sizes** (512×512); "splash best" is splash at its own best configuration (1024×1024 blocks
computing 512 keys at a time). Milliseconds on a v5e, under `jax.jit`.

| Shape | autokernel | XLA + splash | XLA + splash best | XLA |
| --- | --- | --- | --- | --- |
| B=1, S=2048 | **5.51** (162 TFLOP/s) | 5.72 | 5.74 | 7.84 |
| B=4, S=2048 | **20.79** (172 TFLOP/s) | 23.17 | 22.88 | 31.41 |
| B=1, S=4096 | **11.58** (166 TFLOP/s) | 11.93 | 11.78 | 40.2 |

autokernel is 1.03–1.10× faster than XLA + splash at its best. Where that comes from at B=4 (profiler device
time): 0.98 ms of head transposes between (B, S, H, D) and (B, H, S, D) that XLA + splash needs and
autokernel doesn't (its attention reads a head as a block of columns), 0.55 ms of attention at the same
blocks, and the MLP, where XLA's matmul fusions slow down once the norm, ReLU and residual are fused in.

### Plain matmuls

bf16 with f32 accumulation, bf16 out, device time (profiler) on a v5e, ms. autokernel's nest is the plain
`acc = zeros; for k: acc += a @ b` loop.

| Shape (M×K×N) | XLA | autokernel | Blocks |
| --- | --- | --- | --- |
| 8192×2048×2048 | **0.357** | 0.365 | 1024×512×2048 |
| 8192×2048×8192 | **1.419** | 1.423 | 1024×1024×2048 |
| 8192×8192×2048 | **1.431** | 1.439 | 512×1024×8192 |

Within 0.3–2% of XLA; what's left is the pipeline fill and drain of a standalone kernel (the first tile's
inputs and the last tile's output, ~8 µs), which matters less the longer the kernel runs, and is hidden
between the kernels of a fused program. A reduction of one step (`TK = K`) compiles to no accumulator at
all: the tile is just `(a @ b).astype(...)`, the same code (checked against the jaxpr) as writing the nest
without the reduction loop.

### Decode

`benchmarks/decode.py`: 4 layers of a 2048→8192→2048 ReLU MLP on a few rows, where every layer streams its
67 MB of weights. Milliseconds on a v5e.

| Rows | autokernel, 512×512 blocks | autokernel, `tune="auto"` | XLA |
| --- | --- | --- | --- |
| 64 | 0.423 | 0.383 (TN=2048, TK=128) | **0.379** |
| 256 | 0.520 | **0.437** | 0.448 |

### Tuning

`examples/tuned_matmul.py` (2048³ bf16, 27 block-size candidates) on a v5e: the cost model's pick,
512³, was also the fastest measured, and `autotune="measure"` confirmed 3 input buffers over 2
(0.130 ms against 0.145 ms). Three buffers are the default where they fit in VMEM: on a v5e they run the
8192³ bf16 matmul with 512³ tiles in 6.31 ms against 7.37 ms with two.

## Writing kernels

```python
@autokernel.tpu
def f(x, y, mask):  # positional array parameters
    M, K = x.shape  # optional prologue: shape-level constants
    out = jnp.zeros((M, N))  # one allocation per output
    relu = jnp.zeros((M, N), jnp.bfloat16)
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
            relu = relu.at[i : i + TM, j : j + TN].set(jnp.maximum(acc, 0).astype(jnp.bfloat16))
    return out, relu  # one output or a tuple
```

- Loops are `range(stop)`, `range(start, stop)` or `range(start, stop, step)` with values known at trace
  time (constants, closures, `x.shape`).
- Every parallel loop tiles exactly one dim of every output, so no two tiles write the same block and
  the tiles cover the whole array.
- Inputs are read through tiles: each dimension is `v:v + T` (T equal to loop `v`'s step), a constant
  aligned block `c:c + T`, `:`, or an integer index: a loop variable stepping by 1 (`q[b, h, ...]`) or a
  constant. Integer-indexed dimensions are dropped from the tile, as in NumPy.
- Tile bodies are assignments (`name = expr`, `name op= expr`) and reduction loops. Values that must
  survive between steps are found automatically and kept in VMEM, with the type Python gives them.
  Every value a tile reads must be assigned in it first (nothing carries over from one tile to the next),
  and values assigned between parallel loops can't be reassigned further in. A value a masked reduction
  loop assigns must also be assigned before that loop if anything reads it afterwards (the mask may skip
  every step of a tile).
- Loop variables may be used as values, e.g. `jax.lax.broadcasted_iota(...) + i`. They are Python ints,
  weakly typed (`bf16_tile * i` stays bf16), and a reduction loop's variable keeps its last value after
  the loop. Python control flow (`if`, `int(...)`) can't depend on them outside a mask: use `jnp.where`.
  In the kernel they are int32: arithmetic on them wraps where Python ints wouldn't.
- Every output element must be produced: each output's tiles cover it, apart from masked-out tiles.
- `a @ b`, `a @ b.T` and batched `@` (NumPy's broadcasting rules) run on the MXU and accumulate in f32
  (i32 when both inputs are integers), whatever the inputs' dtype, and so does `a @= b` (a carried bf16
  value becomes f32); `reference` and `verify` use the same semantics, through their own `jnp.matmul`,
  not the kernel's code.
- A single kernel takes its arguments positionally or by name, with defaults. Global and closure arrays its
  tile body reads through slices are inputs, read with their current values on every call; changing a
  module or closure constant the nest reads gives a new kernel, as it changes what the Python computes.
  Integer scalar inputs narrower than 32 bits keep their dtype; other integer scalars act like Python ints.
- Output allocations are evaluated after the whole prologue, so the prologue can't reassign a name an
  allocation reads, and an output can't rebind a parameter of the same name (`x = jnp.zeros(...)`).
  `...` in tile indices and `:=` in the nest aren't supported.

Anything else is rejected with an `AutoKernelError` naming the line, before anything runs.

### Masks

A mask is `if cond: continue` at the top of a loop body, or `if cond:` around all of it.

- A **tile mask** (on the innermost parallel loop) skips output tiles. Skipped tiles keep the output
  allocation's value: when that's a known constant (`jnp.zeros`, `jnp.full(..., -inf)`, ...) the kernel
  writes them with async DMAs that overlap the compute; otherwise the output starts as the allocation.
- A **reduction mask** skips reduction steps. A tile whose mask skips every step still runs the code
  around the loop, just as the Python does.
- Masks over loop coordinates and constants are evaluated at trace time, as plain Python once per
  combination of the loop values they read (so `max(...)`, `np.sum(table[i // T])` or a helper function
  mean what they say), and the kernel walks a table of only the steps that run. When the mask ignores the
  outer loops (causal attention ignores batch and head), the table holds one repeating pattern, so it
  stays small.
- A mask may also **read input arrays** (`block_mask[i // T, k // T]`): the step table is then built on
  device from the actual mask, and the kernel's step count is decided at run time. Such a mask runs as
  jax.numpy under `jax.vmap`, one step at a time: `and`, `or`, `not`, `max`, `min`, `abs`, `bool` and
  `in (a, b)` are translated, but other Python control flow on the values it reads is rejected.
  `reference`, `verify` and derivatives evaluate such a mask as an array (`jnp.where`), so they also work
  under `jax.jit` and `jax.vmap`.

[examples/splash_attention.py](examples/splash_attention.py) is causal multi-head splash attention written
this way.

### Writing fast kernels

Keep per-row statistics lane-wide: shape `(rows, 128)` with the value repeated across the 128 lanes of a
TPU vector register, widened with `jnp.tile` where they meet a wider block. A `(rows, 1)` column needs a
cross-lane broadcast every time it meets a `(rows, n)` block; in the attention kernel, switching the
running max and sum to the lane-wide form made it about 1.7× faster on a v5e.

With few rows (decode), weights dominate: make column blocks wide (each column tile fetches the
activations again) and weight blocks around 0.5–2 MiB (bigger ones lengthen the pipeline fill).
`tune="auto"` picks TN=2048, TK=128 for a 64-row 2048→8192→2048 MLP.

Untuned kernels whose steps are too small to hide the fixed per-step cost (128³ matmul tiles, say) get a
warning on a TPU suggesting `tune="auto"`.

## Programs: whole models

A function that is more than one loop nest becomes a `TpuProgram`: its tile loop nests run as gridless
kernels, and the code around the nests is ordinary JAX and Python.

```python
@autokernel.tpu
def mlp(x, w1, w2):
    x2 = x.reshape(-1, D)  # plain JAX, compiled by XLA around the kernels
    for layer in range(L):  # runs as a jax.lax.fori_loop: each kernel is compiled once
        h = jnp.zeros((M, F))  # a nest's outputs are allocated just before it
        for i in range(0, M, TM):  # nest 1
            for j in range(0, F, TN):
                acc = jnp.zeros((TM, TN))
                for k in range(0, D, TK):
                    acc += x2[i : i + TM, k : k + TK] @ w1[layer, k : k + TK, j : j + TN]
                h = h.at[i : i + TM, j : j + TN].set(jnp.maximum(acc, 0))
        y = jnp.zeros((M, D))
        for i in range(0, M, TM):  # nest 2: fused with nest 1 into one kernel
            ...
        x2 = x2 + y
    return x2.reshape(x.shape)
```

- A tile loop nest is a `for` over `range` whose loops end in stores of tiles placed by the loop
  variables (`out = out.at[i : i + T, ...].set(...)`, or `out.at[h, i : i + T]`) and that doesn't read
  the arrays it stores to. Any other loop, such as a scan `h = h.at[t].set(f(h[t - 1]))` or collecting
  results `outs = outs.at[layer].set(h)`, is ordinary Python, run as JAX. Nests can sit inside plain
  Python `for`, `if` and `with` blocks.
- An output allocated just before its nest is allocated by the kernel; if the allocation computes values
  (`y = x * 2`), elements no tile writes keep them, while a constant fill (`jnp.zeros`, `jnp.full`) must be
  covered by the tiles. Otherwise the nest starts from the
  array's current value (an allocation before an `if` or loop, an argument, an earlier nest's output), and
  elements no tile writes keep it, as in Python.
- A nest's loop variables and temporaries don't exist after it: reading one later is an error naming the
  line.
- A nest reads the function's local values by name: arrays become the kernel's inputs. Numbers, dtypes and
  tuples are compiled into the kernel as constants, one kernel per distinct set, except **integers the
  nest only uses as indices of its inputs or as values in its tile body** (`w1[layer, ...]` above, or
  `x * layer`): those are *runtime scalars*, passed to the kernel in SMEM, so one kernel serves every
  value. A Python integer that takes part in integer arithmetic in the tile body (`p * p`, `layer + 1`)
  stays a runtime scalar only when that arithmetic, done in the kernel's int32, gives what Python gives for
  the current values; otherwise it is compiled in. Integers it needs at trace time (loop bounds, masks, slices, allocations) stay constants. An
  integer computed by JAX (a traced 0-d array, `idx = jnp.argmax(scores)`) can index inputs the same way,
  and a 0-d float computed by JAX (`scale = 1 / jnp.sqrt(d)`) used only as a value in the tile body is a
  runtime scalar too (an f32 in SMEM; a bf16 is read as an f32 that mixes with tiles like a Python float).
  Global and closure arrays a nest reads through slices are inputs too, with their current values on
  every call, and global and closure numbers a nest reads are compiled in with their current values.
- **Python loops around nests** (`for layer in range(L)`) run as a `jax.lax.fori_loop`, so a deep model
  traces and compiles each kernel once, whatever its depth. The loop variable is then a traced integer;
  if the body needs its value at trace time (a nest's loop bound, Python control flow on it), the loop is
  unrolled instead, as it is when the body changes Python state other than its local values (attribute or
  item assignments, `print`, methods of Python objects such as `list.append`), when a value it carries
  isn't an array (a Python counter), or when a function defined outside the loop reads a value the loop
  changes. `unroll=True` always unrolls; `program.unrolled` says why each loop was unrolled.
- **Consecutive nests are fused** into one gridless kernel when nothing but views runs between them: the
  kernel runs one nest's step loop after the other's, starts fetching the next nest's weights (and other
  inputs no earlier nest writes) while the current nest finishes its last steps, and reads what an earlier
  nest wrote once that nest's output copies have drained. One launch and one pipeline fill for the whole
  run. Each nest's VMEM scratch is allocated only while it runs (`pl.run_scoped`), so only neighbouring
  nests need to fit in VMEM together. Fusing only pays when the kernels are short: the launch, pipeline
  fill and drain it hides must be at least 2% of the nests' time (decode-sized MLPs on a v5e ran 1.5–2.5%
  faster fused; a prefill attention kernel fused with its neighbours ran 1% slower), so long nests keep
  kernels of their own. At most three nests share a kernel (`fuse=n` for up to n): on a v5e, longer fused
  kernels measured slower than shorter ones; a longer run is cut into the fewest kernels,
  where the least data passes between them, and `lowered.cuts` says why each kernel ends where it does. On
  chips with two TensorCores (v4, v5p) the cores meet at a barrier before a nest reads what an earlier nest
  wrote. Nests aren't fused when a nest's runtime mask or aliased output reads an array an earlier nest of
  the run writes. `fuse=False` gives every nest its own kernel.
- **Views between nests cost nothing.** A statement that only reshapes, transposes (`.T`, `jnp.swapaxes`,
  `jnp.moveaxis`) or indexes an array with integers and whole slices
  (`qkv = qkv.reshape(B, S, 3, H, DH).transpose(2, 0, 3, 1, 4)`, `x = stack[1]`) is a view: a nest reading
  it is given the original array and copies each tile straight out of it (`qkv[0, b, h, i:i + T, :]` is
  rows `b * S + i ...` and columns `h * DH ...` of `qkv`), so XLA copies nothing and the view doesn't keep
  the nests around it from fusing. Where a tile of the view isn't one rectangular, TPU-aligned window of
  the original, the nest reads a copy made by XLA instead.
- Calling the program jit-compiles it once per input signature. `lower` returns every kernel the program
  runs, as `(name, kernel)` pairs: a `Lowered` for a nest of its own, a `FusedLowered` (its `parts` are
  the nests' `Lowered`s) for fused nests; `lowered["nest name"]` finds a nest either way. `source`,
  `reference`, `verify` and `tuning` work as for one kernel, and every entry point takes keyword arguments
  and defaults. `verify` names the first nest whose kernel disagrees with its Python loops, and
  `verify_nests(*args)` (or `verify(*args, per_nest=True)`) checks each nest on the inputs the program
  actually gives it. `reference` runs the function's own control flow (Python loops as written), each nest
  as its Python loops, so `verify` also checks how the program was rewritten. All options apply to every kernel; `tune` names a nest doesn't use are ignored for it.

Runtime indices follow Python's rules: a negative index counts from the end. An index known when the
program is traced (a Python int) is checked against the array and raises `AutoKernelError` when it's out
of range. A traced index can't be checked before the kernel runs: the kernel clamps it into the array (it
never reads outside it, which on a TPU would halt the core), and the check runs as a
`jax.experimental.checkify` check: under `checkify.checkify(program)` an out-of-range index is an error
naming the input and the index's value; without checkify the kernel reads the clamped index.

[examples/transformer_block.py](examples/transformer_block.py) is a stack of transformer blocks (RMSNorm,
QKV projection, causal flash attention, output projection, MLP) written entirely as tile loops: seven nests,
three kernels after fusion (the reshapes and transposes around the attention are views, read in place), the
same for any depth. [benchmarks/transformer.py](benchmarks/transformer.py) writes a layer without the
reshapes (a head is a block of columns); at its tuned block sizes a layer's five nests run as two kernels.

### Runtime scalars in a single kernel

A 0-d integer input of a single-kernel function is a runtime scalar too: it can index dims of the other
inputs (`w[layer, i:i + T, :]`) and be used as a value in the tile body, read from SMEM.

## JAX transformations

Kernels and programs compose with JAX's transformations:

- **`jax.vmap`**: a batch runs as one kernel, generated from the loop nest with the batch as an extra
  outermost parallel loop (`for b in range(B)`, batched inputs read `x[b, ...]`, outputs written
  `out.at[b, ...]`), not one launch per element. Any `in_axes`, nested `vmap`s, and `vmap` of `jit` or
  `grad` work, masks included: a runtime mask reading a batched input is read per batch element. A batched
  runtime scalar instead runs the kernel once per element in a `jax.lax.map`, with a warning. In a program
  each nest is batched this way and the code between them by JAX (fused nests run as separate batched
  kernels under `vmap`).
- **Differentiation**: by default a kernel's derivative is that of its Python loops, run as plain JAX
  (computed by XLA, not by a generated kernel), while the primal still runs the kernel, once. Forward mode
  (`jax.jvp`, `jax.jacfwd`, `jax.linearize`), reverse mode (`jax.grad`, `jax.vjp`, `jax.jacrev`) and
  higher orders all work, through programs and fused kernels too. `backward=` supplies a reverse-mode rule
  instead, `backward(*inputs, cotangent) -> one gradient per input (or None)`, which may itself call
  autokernel kernels (`jax.jvp` is then unavailable).
- **Sharded inputs**: called with arrays sharded over a mesh (a `NamedSharding`; under `jax.jit`, a mesh
  with `Explicit` axes, the default of `jax.make_mesh`), a kernel runs under `jax.shard_map`, each device
  running a kernel generated for its shard, when every sharded dim is tiled by a parallel loop used only
  to index: batch, head and row loops usually. Outputs come out sharded the same way. A dim sharded any
  other way (a reduction dim, or one whose loop a mask or the tile body reads) is gathered first, with a
  warning. Inside `jax.jit` with `Auto` mesh axes the kernel can't see the shardings, and XLA runs it on
  gathered inputs.

## Tuning block sizes

Let autokernel choose them:

```python
@autokernel.tpu(tune="auto")
def matmul(x, y): ...  # the loops step by module constants TM, TN, TK
```

or name the constants and the values to try, with `"auto"` for any of them:

```python
@autokernel.tpu(tune={"TM": (128, 256, 512), "TN": "auto", "TK": (256, 512)})
def matmul(x, y): ...
```

`"auto"` tunes *block sizes*: names a loop steps by (`range(0, M, TM)`) that are int constants in the
module or a closure. Steps computed in the prologue from input shapes, and steps of 1 (loops that index a
dim), aren't tuned. The candidates are powers of two and three times powers of two from the smallest tile the
TPU layout allows for the dims the loop slices (128 on a tile's last axis, 8, 16 or 32 rows by dtype on its
second-to-last) up to 4096, plus the constant's own value and one block for the whole loop (up to 8192: a
single reduction step), keeping those that divide the loop. Combinations that can't fit in VMEM are pruned
without planning (VMEM grows with block sizes, except that a block covering a whole reduction loop runs
one step per tile, with its tiles in two slots and no accumulator, so it is compared only with itself); up to 128 combinations are ranked exhaustively,
larger spaces by coordinate descent over the cost model, which found the exhaustive best on every nest we
compared (an 8192³ bf16 matmul's 490 combinations in 75 plans, 0.3 s; four names of eight values, 4096
combinations, in under a second).

Candidates that break TPU layout rules, don't divide the loops or don't fit in VMEM are dropped, and the
rest are ranked by a roofline cost model:

- MXU work per dot at its own dtype's rate (bf16, f32 at HIGHEST, int8/fp8 at the chip's rate), and vector
  work counted per vreg from the jaxpr, with transcendentals on the EUP; MXU and vector work overlap;
- HBM traffic: every fetch, so a tile read again for each column (or row) tile counts again: blocks that
  re-stream big weights for many small row tiles rank low;
- VMEM: every buffer padded to Mosaic's (sublane, 128) tiles, temporaries at their peak live size;
- the chip's numbers from `pltpu.get_tpu_info()`: HBM bandwidth, MXU rates, VMEM and SMEM capacity and
  TensorCores;
- a measured fixed cost per step and per copy, the DMA latency left exposed by the buffer depth, and each
  kernel's launch, pipeline fill and drain.

Calibrated against the measurements above, its predictions are within about 10% (mostly 5%).

Ranking needs no code generation, and each unit of the loop body is only re-analysed when the tile shapes
it sees change, so only the best eight candidates (kept diverse: at most half of them share any one tuned
value) are generated as kernels. With `autotune="measure"` it also times those eight on the TPU (on the first call with concrete arrays), then 2 and 3 input buffers for
the winner; the model's predictions for good block sizes are often within a few percent of each other,
and only timing tells those apart. `kernel.tuning(*args)` lists every candidate with its predicted and
measured time, or why it was rejected (`lowered` is set only for the generated ones). [examples/tuned_matmul.py](examples/tuned_matmul.py)
shows it. `autotune="measure"` without `tune` times the kernel as written with 2 and 3 buffers.

## Options

```python
@autokernel.tpu(
    interpret=None,              # None: native on TPU, interpreter elsewhere. False: require a TPU.
    verify=False,                # check against the Python loops on the first call with arrays, per input shape
    tune=None,                   # "auto", or {constant: candidate values or "auto"} to search
    autotune="model",            # "model": cost model only. "measure": also time the best few on the TPU.
    buffers=None,                # VMEM slots per streamed input; default 3 (2 if 3 don't fit in VMEM), or tuned
    target=None,                 # the TPU to plan for off-TPU: "v5e" (default), "v5p", "v6e", ...
    cores=None,                  # TensorCores to split across; default: all of the target's
    check_tiling=True,           # reject tiles that break TPU (8, 128) layout rules
    vmem_limit_bytes=None,       # Mosaic's VMEM limit, and the budget kernels are planned in; default the chip's
    matmul_precision="highest",  # "default" uses a single bf16 MXU pass for f32 (faster, less exact)
    background_fill=True,        # write masked-out tiles with in-kernel DMAs
    fuse=True,                   # programs: fuse consecutive nests (only views between them) into one kernel
    unroll=False,                # programs: unroll Python loops around nests instead of a fori_loop
    backward=None,               # reverse-mode rule; default: the derivative of the Python loops (fwd and rev)
)
```

`vmem_limit_bytes` defaults to the chip's scoped VMEM (see "TPU rules it enforces").

`kernel.lower(*args)` returns a `Lowered` with the schedule, step and tile counts (per TensorCore too),
operand tiles, state variables, cost estimate, VMEM limit and generated `source`. Arguments can be arrays
or `jax.ShapeDtypeStruct`s.

## TPU rules it enforces

- Tile shapes: the last dimension must be a multiple of 128 or the whole array dimension; the
  second-to-last a multiple of 8 for 32-bit types (16 for bf16, 32 for 8-bit) or the whole dimension.
- f32 matmuls use `Precision.HIGHEST` by default: Mosaic's default is a single bf16 pass, about 0.2 off
  at K=384. bf16 inputs are exact in one pass, accumulated in f32.
- State keeps Python's types: a bf16 `acc += tile` rounds to bf16 every step, while `acc += a @ b` is
  f32 (the MXU's accumulator). Scalars live in SMEM, narrow ones in a 32-bit word.
- VMEM use is estimated (input rings, output slots, state and the peak of the intermediates, each padded
  to Mosaic's (sublane, 128) tiles) and must fit the chip. The kernel's VMEM limit is the chip's scoped VMEM,
  not the estimate: Mosaic takes its own scratch (spilled intermediates) from what the limit leaves, and
  limits sized from the estimate made attention-heavy kernels up to 2.7% slower on a v5e.
- Integer vector arithmetic only on 16- and 32-bit values: 8-bit elementwise math, comparisons and
  reductions (`x_i8 * 2`, `jnp.maximum(x_i8, 0)`) are rejected with an `AutoKernelError` naming the line,
  since Mosaic can't compile them on a v5e though the interpreter runs them; widen with
  `.astype(jnp.int32)`. 8-bit `@` runs on the MXU and is fine.
- Step, tile, fill and core tables and runtime scalars live in SMEM and must fit in half of it (512 KiB on
  v4 and newer). Kernels with no masks need no tables at all.

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
uv run python examples/transformer_block.py
uv run python benchmarks/matmul.py      # timings; meaningful only on a TPU
uv run python benchmarks/attention.py   # autokernel vs JAX splash attention at the same block sizes
uv run python benchmarks/transformer.py # a transformer stack: autokernel vs XLA + splash vs XLA (--breakdown)
uv run python benchmarks/decode.py      # decode-like MLP layers that stream their weights
```

Besides running kernels in the interpreter (with its race detector), the tests cross-lower every kind of
kernel to Mosaic for a v5e, and the two-TensorCore form (fused kernels with their barriers too) for a
v5p, which catches lowering errors the interpreter can't see. `tests/test_sharding.py` runs
`tests/sharding_cases.py` in a subprocess with four CPU devices. Timing still needs real hardware.

## Layout

```
src/autokernel/
  frontend.py   parse the Python loop nest
  analysis.py   resolve it for concrete shapes: loops, tiles, operands, outputs, masks, state variables
  schedule.py   order and decode the steps: arithmetic, static step tables, or tables built on device;
                split them across TensorCores
  codegen.py    emit the gridless Pallas kernel and its pallas_call wrapper
  mxu.py        what `@` means: MXU dots accumulating in f32 / i32 for the kernel and the trace; plain
                jnp.matmul with the same accumulators for reference and verify
  cost.py       the roofline cost model (jaxpr op counts + TPU specs)
  lowering.py   options, and the pipeline from parsed loops to a compiled kernel
  tune.py       block-size search
  hardware.py   TPU specs (from pltpu.get_tpu_info) and layout rules
  api.py        the @autokernel.tpu decorator and TpuKernel
  program.py    programs: several nests and JAX code in one function; runtime scalars, rolled layer loops
  fusion.py     fuse consecutive nests of a program into one kernel
  views.py      reshapes, transposes and indexing between nests, read as windows of the array they view
  autodiff.py   derivatives: a custom_jvp from the Python loops, or a user `backward`; the vmap hook
  batching.py   jax.vmap: the loop nest with an extra batch loop, as one kernel
  sharding.py   sharded inputs: which loops split across devices; the per-shard kernel under shard_map
  reference.py  the Python loops as a traceable reference (runtime masks as jnp.where)
  errors.py     AutoKernelError
examples/       runnable examples
benchmarks/     timing scripts for a TPU
tests/
```
