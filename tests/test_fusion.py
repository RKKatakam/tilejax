"""Programs: consecutive nests fused into one kernel, runtime integer indices (one kernel for every layer),
and Python loops around nests run as `jax.lax.fori_loop`."""

import importlib.util
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import AutoKernelError, FusedLowered

M, K, N = 256, 256, 256
T = 128
L = 3

V5E = jax.sharding.AbstractDevice(device_kind="TPU v5 lite", num_cores=1, platform="tpu")
V5P = jax.sharding.AbstractDevice(device_kind="TPU v5", num_cores=2, platform="tpu")
# Simulating a two-TensorCore v5p (one interpreter thread per core) needs the interpreter on the CPU: on a
# TPU host its callbacks can't reach the simulated shared memory, and the v5p kernels can't run natively.
interpreter_only = pytest.mark.skipif(
    jax.default_backend() == "tpu", reason="simulates a two-TensorCore v5p in the CPU interpreter"
)


def check_races(lowered, args, cores=1):
    """Run one kernel of a program in the TPU interpreter with the race detector on (simulating `cores`
    TensorCores); return its outputs."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan", num_cores_or_threads=cores)
    out = jax.block_until_ready(lowered.entry(*args, interpret=params))
    assert interpreter.races is not None and not interpreter.races.races_found, "the kernel has data races"
    return out


def random_inputs(specs, seed=0):
    """Arrays for a kernel's input specs: normal floats, zeros for integer scalars, ones for integer arrays."""
    out = []
    for n, s in enumerate(specs):
        if jnp.issubdtype(s.dtype, jnp.integer):
            out.append(jnp.zeros(s.shape, s.dtype) if s.shape == () else jnp.ones(s.shape, s.dtype))
        elif s.dtype == jnp.bool_:
            out.append(jnp.ones(s.shape, bool))
        else:
            out.append(rand(s.shape, seed + n, s.dtype))
    return out


def lower_for_v5e(lowered, specs, device=V5E):
    """Cross-lower a kernel to Mosaic for a v5e (or `device`), as `jax.jit(...).lower` does on a TPU."""
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(lambda *xs: lowered.entry(*xs, interpret=False)).trace(*specs)
        return traced.lower(lowering_platforms=("tpu",)).as_text()


def check_every_kernel(prog, *args):
    """Race-check every kernel the program runs, and cross-lower each to Mosaic for a v5e."""
    lowered = prog.lower(*args)
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        check_races(kernel, random_inputs(specs))
        assert "tpu_custom_call" in lower_for_v5e(kernel, specs)
    return lowered


# --------------------------------------------------------------------------- fusion


def mlp(x, w1, b1, w2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.maximum(acc + b1[:, j : j + T], 0))
    y = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, N, T):
                acc += h[i : i + T, k : k + T] @ w2[k : k + T, j : j + T]
            y = y.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + acc)
    return y


def mlp_inputs():
    return rand((M, K), 1), rand((K, N), 2) / 16, rand((1, N), 3), rand((N, N), 4) / 16


def test_consecutive_nests_run_as_one_kernel():
    prog = autokernel.tpu(mlp)
    args = mlp_inputs()
    lowered = prog.lower(*args)
    assert lowered.names == ["mlp_h+mlp_y"]
    fused = lowered[0][1]
    assert isinstance(fused, FusedLowered) and [n for n, _ in fused.parts] == ["mlp_h", "mlp_y"]
    assert fused.inputs == ("x", "w1", "b1", "w2") and fused.outputs == ((0, "h"), (1, "y"))
    assert fused.steps == 8 + 8 and lowered["mlp_y"].steps == 8  # a part by its nest's name
    assert fused.source.count("pl.pallas_call(") == 1 and "grid=" not in fused.source
    unfused = autokernel.tpu(mlp, fuse=False)
    assert unfused.lower(*args).names == ["mlp_h", "mlp_y"]
    np.testing.assert_allclose(prog(*args), unfused(*args), rtol=1e-6, atol=1e-6)
    assert prog.verify(*args) < 1e-4


def test_fused_kernel_has_no_races_and_lowers_to_mosaic():
    prog = autokernel.tpu(mlp)
    args = mlp_inputs()
    lowered = check_every_kernel(prog, *args)
    h, y = check_races(lowered[0][1], args)  # the fused kernel returns every nest's outputs
    want_h = jnp.maximum(jnp.dot(args[0], args[1], precision="highest") + args[2], 0)
    np.testing.assert_allclose(h, want_h, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y, prog.reference(*args), rtol=1e-4, atol=1e-4)


def test_weights_are_fetched_while_the_previous_nest_finishes():
    source = autokernel.tpu(mlp).source(*mlp_inputs())
    part1 = source[source.index("part 1: mlp_y") :]
    early = part1[part1.index("def _early():") : part1.index("def _with_scratch(")]
    late = part1[part1.index("def _late():") : part1.index("def _run(")]
    assert "_w2_ref_copy" in early and "_h_ref_copy" not in early  # weights: during nest 0's last steps
    assert "_h_ref_copy" in late  # what nest 0 wrote: once its copies have drained
    assert "run_0(early_1)" in source and source.index("drain_0()") < source.index("late_1()")


def test_fusion_keeps_python_semantics_of_names_rebound_between_nests():
    def f(x):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] + 1)
        x = jnp.zeros((M, N))  # rebinds `x`: the next nest's output
        for i in range(0, M, T):
            x = x.at[i : i + T, :].set(y[i : i + T, :] * 2)
        z = jnp.zeros((M, N))
        for i in range(0, M, T):
            z = z.at[i : i + T, :].set(x[i : i + T, :] - y[i : i + T, :])
        return x, y, z

    prog = autokernel.tpu(f)
    x = rand((M, N))
    assert prog.lower(x).names == ["f_y+f_x+f_z"]
    # At most two nests per kernel: two kernels either way, cut where less data passes between them (`y`
    # alone before `f_x`; `x` and `y` before `f_z`).
    two = autokernel.tpu(f, fuse=2).lower(x)
    assert two.names == ["f_y", "f_x+f_z"]
    assert two.cuts == ("cut before `f_x`: of the ways to run these 3 nests as 2 kernels, this cut passes the "
                        "least data from one kernel to the next (0.2 MiB)",)  # fmt: skip
    got = prog(x)
    for g, w in zip(got, ((x + 1) * 2, x + 1, x + 1), strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-6)
    check_every_kernel(prog, x)


def test_jax_code_between_nests_separates_kernels():
    def f(x):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] + 1)
        y = y * 3
        z = jnp.zeros((M, N))
        for i in range(0, M, T):
            z = z.at[i : i + T, :].set(y[i : i + T, :] * 2)
        return z

    prog = autokernel.tpu(f)
    assert prog.lower(rand((M, N))).names == ["f_y", "f_z"]
    np.testing.assert_allclose(prog(rand((M, N))), (rand((M, N)) + 1) * 6, rtol=1e-6)


def mixing(x, w1, w2):
    # The second nest reads every row block of `h` for each of its tiles: on two TensorCores (each taking
    # half of the row blocks), rows the other core wrote.
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.maximum(acc, 0))
    y = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, M, T):
                acc += w2[i : i + T, k : k + T] @ h[k : k + T, j : j + T]
            y = y.at[i : i + T, j : j + T].set(acc)
    return y


@interpreter_only
def test_fusion_across_two_tensorcores_meets_at_a_barrier():
    # On a v5p each nest splits its tiles between the two TensorCores; the cores meet at a barrier (semaphore
    # signals across cores) once their copies of `h` have drained, before `mixing_y` reads it.
    prog = autokernel.tpu(mixing, target="v5p")
    args = (rand((M, K), 1), rand((K, N), 2) / 16, rand((M, M), 4) / 16)
    lowered = prog.lower(*args)
    assert lowered.names == ["mixing_h+mixing_y"]
    fused = lowered[0][1]
    assert fused.cores == 2 and all(lw.cores == 2 for _, lw in fused.parts)
    assert "grid=(2,)" in fused.source and "pl.semaphore_wait(barrier_sem, 2)" in fused.source
    assert fused.source.index("drain_0()") < fused.source.index("_barrier()  #") < fused.source.index("late_1()")
    want = prog.reference(*args)
    _, y = check_races(fused, args, cores=2)  # each core a thread of the interpreter
    np.testing.assert_allclose(y, want, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(prog(*args), want, rtol=1e-4, atol=1e-4)
    assert "tpu_custom_call" in lower_for_v5e(fused, args, device=V5P)

    # Without the barrier, the race detector sees `mixing_y` read rows of `h` the other core is writing.
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    namespace = dict(fused.entry.__globals__)
    source = fused.source.replace("_barrier()  #", "pass  #")
    exec(compile(source[source.index(f"def {fused.name}_kernel(") :], "<no barrier>", "exec"), namespace)
    params = pltpu.InterpretParams(detect_races=True, num_cores_or_threads=2)
    jax.block_until_ready(namespace[f"{fused.name}_pallas"](*args, interpret=params))
    assert interpreter.races.races_found


@interpreter_only
def test_a_core_with_no_tiles_still_meets_the_other_at_the_barriers():
    def f(x, w1, w2):  # one tile per nest: the second TensorCore has nothing to do but meet the first
        h = jnp.zeros((T, T))
        for i in range(0, T, T):
            h = h.at[i : i + T, :].set(jnp.dot(x[i : i + T, :], w1[:, :]))
        y = jnp.zeros((T, T))
        for i in range(0, T, T):
            y = y.at[i : i + T, :].set(w2[i : i + T, :] @ h[:, :])
        return y

    prog = autokernel.tpu(f, target="v5p")
    args = [rand((T, T), s) for s in range(3)]
    fused = prog.lower(*args)[0][1]
    assert [lw.core_steps for _, lw in fused.parts] == [[(0, 0), (0, 1)]] * 2
    check_races(fused, args, cores=2)
    np.testing.assert_allclose(prog(*args), prog.reference(*args), rtol=1e-5, atol=1e-5)


def averaged(x, y):
    # `n` is a scalar carried between steps (SMEM scratch), allocated, like all of a part's scratch, in the
    # part's own scope (`pl.run_scoped`) of the fused kernel.
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            n = 0.0
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                n += 1.0
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(acc / n)
    z = jnp.zeros((M, N))
    for i in range(0, M, T):
        z = z.at[i : i + T, :].set(h[i : i + T, :] * 2)
    return z


def test_each_part_allocates_its_scratch_in_its_own_scope():
    prog = autokernel.tpu(averaged)
    x, y = rand((M, K), 1), rand((K, N), 2)
    lowered = check_every_kernel(prog, x, y)
    fused = lowered[0][1]
    assert lowered.names == ["averaged_h+averaged_z"] and "scratch_shapes=[]" in fused.source
    assert fused.source.count("pl.run_scoped(") == 2  # part 0's buffers, then part 1's (it has no early ones)
    assert "pltpu.SMEM((1,), jnp.float32),  # n" in fused.source
    np.testing.assert_allclose(prog(x, y), jnp.dot(x, y, precision="highest") / 2 * 2, rtol=1e-5, atol=1e-5)


def test_only_neighbouring_nests_need_to_fit_in_vmem_together(monkeypatch):
    from autokernel import fusion, hardware

    # Each nest needs 60% of the VMEM budget: together 120%. But the second nest's scratch is allocated once
    # the first's is released, except for the buffers its weights are fetched into early.
    budget = hardware.target("v5e").vmem_budget
    monkeypatch.setattr(fusion, "vmem_bytes", lambda lowered: int(0.6 * budget))
    lowered = autokernel.tpu(mlp).lower(*mlp_inputs())
    assert lowered.names == ["mlp_h+mlp_y"]
    early = fusion.scratch_split(lowered["mlp_y"], {"h"})[0]  # w2's ring and x's tiles
    assert lowered[0][1].estimate.vmem_bytes == int(0.6 * budget) + early
    assert fusion.fused_vmem([(10, 90), (20, 80), (30, 70)]) == max(10 + 90 + 20, 20 + 30 + 80, 20 + 30 + 70)


def test_a_layer_at_the_benchmark_block_sizes_fuses_into_two_kernels():
    # benchmarks/transformer.py's layer at its tuned block sizes on a v5e (128 MiB of VMEM): only the down
    # projection's weight ring is allocated while the up projection (1024 x 2048 x 2048 blocks) runs, so the
    # fused kernel needs less VMEM than the two nests together.
    path = pathlib.Path(__file__).parent.parent / "benchmarks" / "transformer.py"
    spec = importlib.util.spec_from_file_location("transformer_benchmark_for_fusion", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    blocks = dict(TQM=1024, TQN=1024, TQK=512, TOM=1024, TON=1024, TOK=1024, TUM=1024, TUN=2048, TUK=2048)
    bench.configure(B=1, S=2048, D=2048, H=16, F=8192, L=4, TQ=512, TKV=512, TDM=1024, TDN=1024, TDK=1024, **blocks)
    args = jax.eval_shape(lambda: bench.inputs("prefill"))
    lowered = autokernel.tpu(bench.transformer, target="v5e").lower(*args)
    assert [len(k.parts) for _, k in lowered] == [3, 2]
    budget = lowered[1][1].parts[0][1].plan.hw.vmem_budget
    up, down = (lw.estimate.vmem_bytes for _, lw in lowered[1][1].parts)
    assert up + down > lowered[1][1].estimate.vmem_bytes and budget >= lowered[1][1].estimate.vmem_bytes
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        assert "tpu_custom_call" in lower_for_v5e(kernel, specs)


def test_nests_that_dont_fit_in_vmem_together_are_not_fused(monkeypatch):
    from autokernel import fusion

    monkeypatch.setattr(fusion, "vmem_bytes", lambda lowered: 10**12)
    assert autokernel.tpu(mlp).lower(*mlp_inputs()).names == ["mlp_h", "mlp_y"]


def masks_and_fills(x, y, block_mask):
    # A static tile mask whose skipped tiles are written by fill DMAs (part 0), one whose skipped tiles
    # keep an aliased allocation (part 1), and a reduction mask read from an input (part 2).
    a = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            if j > i:
                continue
            a = a.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    b = jnp.full((M, N), 7.0) + y
    for i in range(0, M, T):
        for j in range(0, N, T):
            if j < i:
                continue
            b = b.at[i : i + T, j : j + T].set(a[i : i + T, j : j + T] + 1)
    c = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                if block_mask[i // T, k // T] == 0:
                    continue
                acc += b[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            c = c.at[i : i + T, j : j + T].set(acc)
    return a, b, c


def test_fused_masks_fills_and_aliased_outputs():
    prog = autokernel.tpu(masks_and_fills)
    x, y = rand((M, N), 1), rand((K, N), 2)
    block_mask = jnp.array([[1, 0], [1, 1]], jnp.int32)
    lowered = prog.lower(x, y, block_mask)
    assert lowered.names == ["masks_and_fills_a+masks_and_fills_b+masks_and_fills_c"]
    parts = [lw for _, lw in lowered[0][1].parts]
    assert [p.fill for p in parts] == ["background", "prefill", None] and parts[2].schedule == "runtime"
    got, want = prog(x, y, block_mask), autokernel.tpu(masks_and_fills, fuse=False)(x, y, block_mask)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
    assert prog.verify(x, y, block_mask) < 1e-4
    check_every_kernel(prog, x, y, block_mask)


def test_a_mask_reading_an_array_written_by_an_earlier_nest_starts_a_new_kernel():
    def f(x, flags):
        on = jnp.zeros((2, 128), jnp.int32)
        for i in range(2):
            on = on.at[i, :].set(flags[i, :] * 2)
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            if on[i // T, 0] == 0:  # read before the kernel starts: `on` must exist by then
                continue
            out = out.at[i : i + T, :].set(x[i : i + T, :] + 1)
        return out

    prog = autokernel.tpu(f)
    x, flags = rand((M, N)), jnp.array([[1] * 128, [0] * 128], jnp.int32)
    assert prog.lower(x, flags).names == ["f_on", "f_out"]
    np.testing.assert_allclose(prog(x, flags), jnp.where(jnp.arange(M)[:, None] < T, x + 1, 0))


# --------------------------------------------------------------------------- runtime indices and layer loops


def deep(x, w, b):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc = jnp.zeros((T, T))
                for k in range(0, K, T):
                    acc += x[i : i + T, k : k + T] @ w[layer, k : k + T, j : j + T]
                h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc + b[layer, :, j : j + T]))
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                y = y.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + h[i : i + T, j : j + T] * (layer + 1))
        x = y
    return x


def deep_inputs():
    return rand((M, K), 1), rand((L, K, N), 2) / 16, rand((L, 1, N), 3)


def deep_reference(x, w, b):
    for layer in range(L):
        x = x + jnp.tanh(jnp.dot(x, w[layer], precision="highest") + b[layer]) * (layer + 1)
    return x


@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("unroll", [False, True])
def test_layer_index_is_a_runtime_scalar(fuse, unroll):
    prog = autokernel.tpu(deep, fuse=fuse, unroll=unroll)
    args = deep_inputs()
    np.testing.assert_allclose(prog(*args), deep_reference(*args), rtol=1e-4, atol=1e-4)
    lowered = prog.lower(*args)
    # One kernel per nest whatever the depth: `layer` indexes `w` and `b`, and is a value in the second nest.
    assert lowered.names == (["deep_h+deep_y"] if fuse else ["deep_h", "deep_y"])
    assert lowered["deep_h"].scalars == ["layer"] and lowered["deep_y"].scalars == ["layer"]
    assert "w_index0 = _index(layer, 3)" in lowered.source
    assert prog.unrolled == []
    if not fuse:
        check_every_kernel(prog, *args)


def test_layer_loop_runs_as_a_fori_loop():
    def top_level(prog):
        return [e.primitive.name for e in jax.make_jaxpr(prog._program_fn())(*deep_inputs()).jaxpr.eqns]

    assert top_level(autokernel.tpu(deep)) == ["scan"]  # the kernel called once, in the loop body
    assert len(top_level(autokernel.tpu(deep, unroll=True))) == L  # the kernel called L times


def test_a_loop_variable_needed_at_trace_time_unrolls_the_loop():
    def f(x):
        for n in range(1, 3):
            y = jnp.zeros((M, n * T))  # a shape: `n` must be a Python int
            for i in range(0, M, T):
                for j in range(0, n * T, T):
                    y = y.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * n)
            x = jnp.concatenate([y, x[:, n * T :]], axis=1)
        return x

    prog = autokernel.tpu(f)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), prog.reference(x), rtol=1e-6)
    assert prog.unrolled  # tried as a fori_loop, unrolled instead
    assert len(prog.lower(x)) == 2  # the nest has a kernel per value of `n`


def gather_rows(x, w, scores):
    idx = jnp.argmax(scores)  # a traced integer computed by JAX
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + w[idx, i : i + T, :])
    return out


def test_traced_integer_from_jax_code_indexes_an_input():
    prog = autokernel.tpu(gather_rows)
    x, w = rand((M, N), 1), rand((4, M, N), 2)
    for best in range(4):
        scores = jnp.zeros(4).at[best].set(1)
        np.testing.assert_allclose(prog(x, w, scores), x + w[best], rtol=1e-6)
    assert len(prog.kernels[0].kernels) == 1


def test_python_int_index_is_checked_against_the_array(tmp_path):
    src = """
import jax.numpy as jnp
T = 128

def f(x, w, which):
    idx = which * 2 - 1
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + w[idx, i : i + T, :])
    return out
"""
    path = tmp_path / "mod.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = autokernel.tpu(mod.f)
    x, w = rand((M, N), 1), rand((3, M, N), 2)
    np.testing.assert_allclose(prog(x, w, 1), x + w[1], rtol=1e-6)
    np.testing.assert_allclose(prog(x, w, 0), x + w[-1], rtol=1e-6)  # -1: Python's rules, the last one
    with pytest.raises(AutoKernelError, match=r"f_out: `w\[idx, i:i \+ T, :\]`: index 3 \(`idx`\) in dim 0 is outside"):
        prog(x, w, 2)
    assert len(prog.kernels[0].kernels) == 1  # one kernel for every value


def test_traced_integer_needed_at_trace_time_is_an_error():
    def f(x, n):
        k = jnp.sum(n)
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, k, T):
                out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T])
        return out

    with pytest.raises(AutoKernelError, match="`k` is a traced integer"):
        autokernel.tpu(f)(rand((M, N)), jnp.array([N]))


def test_python_int_needing_its_value_in_the_tile_body_is_compiled_in():
    def f(x):
        reps = 2
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(jnp.tile(x[i : i + T, : N // reps], (1, reps)))  # a shape
        return out + 0  # (a program, not a single kernel)

    prog = autokernel.tpu(f)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), jnp.tile(x[:, : N // 2], (1, 2)))
    assert prog.lower(x)[0][1].scalars == []


def test_single_kernel_takes_runtime_scalars():
    @autokernel.tpu
    def pick(w, layer):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(w[layer, i : i + T, :] * (layer + 1))
        return out

    w = rand((3, M, N))
    for layer in (0, 2, -1):
        np.testing.assert_allclose(pick(w, jnp.int32(layer)), w[layer] * (layer + 1), rtol=1e-6)
    with pytest.raises(AutoKernelError, match="index 3"):
        pick(w, jnp.int32(3))
    assert "tpu_custom_call" in lower_for_v5e(pick.lower(w, jnp.int32(0)), [w, jax.ShapeDtypeStruct((), jnp.int32)])


# --------------------------------------------------------------------------- the transformer example


def load_example():
    path = pathlib.Path(__file__).parent.parent / "examples" / "transformer_block.py"
    spec = importlib.util.spec_from_file_location("transformer_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def example_inputs(mod):
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    D, F, Lx = mod.D, mod.F, mod.L
    return (
        jax.random.normal(keys[0], (mod.B, mod.S, D)),
        1 + 0.1 * jax.random.normal(keys[1], (Lx, 1, D)),
        jax.random.normal(keys[2], (Lx, D, 3 * D)) / math.sqrt(D),
        jax.random.normal(keys[3], (Lx, D, D)) / math.sqrt(D),
        1 + 0.1 * jax.random.normal(keys[4], (Lx, 1, D)),
        jax.random.normal(keys[5], (Lx, D, F)) / math.sqrt(D),
        jax.random.normal(keys[6], (Lx, F, D)) / math.sqrt(F),
    )


@interpreter_only
def test_transformer_example_on_two_tensorcores():
    mod = load_example()
    args = example_inputs(mod)
    prog = autokernel.tpu(mod.transformer.fn, target="v5p")
    lowered = prog.lower(*args)
    assert [len(getattr(k, "parts", [k])) for _, k in lowered] == [3, 1, 3] and all(k.cores == 2 for _, k in lowered)
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        check_races(kernel, random_inputs(specs), cores=2)
        assert "tpu_custom_call" in lower_for_v5e(kernel, specs, device=V5P)
    np.testing.assert_allclose(prog(*args), mod.plain_transformer(*args), rtol=1e-4, atol=1e-4)


def test_transformer_example_builds_each_kernel_once_for_any_depth():
    mod = load_example()
    args = example_inputs(mod)
    unfused = autokernel.tpu(mod.transformer.fn, fuse=False).lower(*args)
    assert len(unfused) == 7  # one kernel per nest, whatever L is
    lowered = mod.transformer.lower(*args)
    # Fused: nests separated by nothing but views (the reshapes and transposes of `qkv` and `attn`) share a
    # kernel, at most three to a kernel: three kernels instead of four with the views copied by XLA.
    assert [len(getattr(k, "parts", [k])) for _, k in lowered] == [3, 1, 3]
    assert len(lowered.cuts) == 2
    # The attention reads q, k and v as windows of the QKV projection's output, and the output projection
    # reads the attention's output as windows of the (batch, head, seq, head dim) array: no transposes.
    assert "qkv_hbm.at[pl.ds((b_blk * 256 + j_blk * 128), 128), pl.ds((h_blk * 128 + 512), 128)]" in lowered.source
    assert "attn_hbm.at[(i_blk >> 1), k_blk, pl.ds((i_blk & 1) * 128, 128), pl.ds(0, 128)]" in lowered.source
    assert "transpose" not in jax.jit(mod.transformer).lower(*args).as_text()
    out = mod.transformer(*args)
    np.testing.assert_allclose(out, mod.plain_transformer(*args), rtol=1e-4, atol=1e-4)
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        check_races(kernel, random_inputs(specs))
        assert "tpu_custom_call" in lower_for_v5e(kernel, specs)


def test_long_nests_are_not_fused_when_fusing_would_hide_too_little(monkeypatch):
    # Fusing hides a launch, a pipeline fill and a drain: worth it for short kernels, not for long ones.
    from autokernel import fusion

    monkeypatch.setattr(fusion, "FUSE_MIN_GAIN", 1.0)  # as if the nests ran far longer than what fusing hides
    lowered = autokernel.tpu(mlp).lower(*mlp_inputs())
    assert lowered.names == ["mlp_h", "mlp_y"]
    assert "fusing would hide only" in " ".join(str(c) for c in lowered.cuts)


def test_a_users_vmem_limit_is_the_fused_kernels_budget():
    args = mlp_inputs()
    fused = autokernel.tpu(mlp).lower(*args)[0][1]
    alone = max(lw.estimate.vmem_bytes for _, lw in fused.parts)
    assert alone < fused.estimate.vmem_bytes
    limit = (alone + fused.estimate.vmem_bytes) // 2  # each nest fits on its own, the two together don't
    lowered = autokernel.tpu(mlp, vmem_limit_bytes=limit).lower(*args)
    assert lowered.names == ["mlp_h", "mlp_y"] and any("of VMEM" in c for c in lowered.cuts)
    roomy = autokernel.tpu(mlp, vmem_limit_bytes=2 * fused.estimate.vmem_bytes).lower(*args)
    assert roomy.names == ["mlp_h+mlp_y"]
    assert f"vmem_limit_bytes={2 * fused.estimate.vmem_bytes}" in roomy[0][1].source


def test_the_fused_wrapper_says_how_each_part_buffers_its_inputs():
    source = autokernel.tpu(mlp).source(*mlp_inputs())
    assert "0. mlp_h: 8 steps of 4 tiles; streamed inputs 3-buffered, others fetched once per tile" in source
