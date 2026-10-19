"""Programs: several tile loop nests mixed with ordinary JAX code and Python control flow, each nest its own
kernel."""

import importlib.util
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError

M, K, N = 256, 256, 256
T = 128


# --------------------------------------------------------------------------- (a) chained nests


def matmul_bias_relu(x, w, b):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            acc = jnp.zeros((T, T))
            for k in range(0, K, T):
                acc += x[i : i + T, k : k + T] @ w[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(acc)
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        for j in range(0, N, T):
            out = out.at[i : i + T, j : j + T].set(jnp.maximum(h[i : i + T, j : j + T] + b[:, j : j + T], 0))
    return out


def test_chained_nests():
    prog = autokernel.tpu(matmul_bias_relu)
    assert isinstance(prog, autokernel.TpuProgram)
    assert [n.name for n in prog.kernels] == ["matmul_bias_relu_h", "matmul_bias_relu_out"]
    assert prog.kernels[0].inputs == ["x", "w"] and prog.kernels[1].inputs == ["h", "b"]
    x, w, b = rand((M, K), 1), rand((K, N), 2), rand((1, N), 3)
    out = prog(x, w, b)
    want = jnp.maximum(jnp.dot(x, w, precision="highest") + b, 0)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)
    assert prog.verify(x, w, b) < 1e-4


def test_single_nest_stays_a_tpu_kernel():
    def matmul(x, w):
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc = jnp.zeros((T, T))
                for k in range(0, K, T):
                    acc += x[i : i + T, k : k + T] @ w[k : k + T, j : j + T]
                out = out.at[i : i + T, j : j + T].set(acc)
        return out

    assert type(autokernel.tpu(matmul)) is autokernel.TpuKernel


# --------------------------------------------------------------------------- (b) JAX code between nests

B, S, D, F = 2, 128, 256, 512  # batch, sequence, model and hidden widths; M = B * S rows


def mlp(x, w1, w2):
    rows = x.shape[0] * x.shape[1]
    x2 = x.reshape(rows, D)  # plain JAX
    h = jnp.zeros((rows, F))
    for i in range(0, rows, T):
        for j in range(0, F, T):
            acc = jnp.zeros((T, T))
            for k in range(0, D, T):
                acc += x2[i : i + T, k : k + T] @ w1[k : k + T, j : j + T]
            h = h.at[i : i + T, j : j + T].set(jnp.maximum(acc, 0))
    h = h * 0.5  # plain JAX
    y = jnp.zeros((rows, D))
    for i in range(0, rows, T):
        for j in range(0, D, T):
            acc = jnp.zeros((T, T))
            for k in range(0, F, T):
                acc += h[i : i + T, k : k + T] @ w2[k : k + T, j : j + T]
            y = y.at[i : i + T, j : j + T].set(acc)
    return (x2 + y).reshape(x.shape)  # residual


def test_mlp_with_jax_between_nests():
    prog = autokernel.tpu(mlp)
    x, w1, w2 = rand((B, S, D), 1), rand((D, F), 2) / 16, rand((F, D), 3) / 16
    out = prog(x, w1, w2)
    hp = "highest"
    h = jnp.maximum(jnp.dot(x.reshape(-1, D), w1, precision=hp), 0) * 0.5
    want = x + jnp.dot(h, w2, precision=hp).reshape(x.shape)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)
    assert prog.verify(x, w1, w2) < 1e-4
    # `rows` is a Python int computed by the program: a constant compiled into both kernels.
    assert [n.inputs for n in prog.kernels] == [["rows", "x2", "w1"], ["rows", "h", "w2"]]


# --------------------------------------------------------------------------- (c) a Python loop over layers

L = 2


def deep(x, w):
    for layer in range(L):  # plain Python: unrolled when traced
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc = jnp.zeros((T, T))
                for k in range(0, K, T):
                    acc += x[i : i + T, k : k + T] @ w[layer, k : k + T, j : j + T]
                h = h.at[i : i + T, j : j + T].set(jnp.tanh(acc))
        x = x + h
    return x


def test_python_loop_around_nests():
    prog = autokernel.tpu(deep)
    x, w = rand((M, K), 1), rand((L, K, N), 2) / 16
    want = x
    for layer in range(L):
        want = want + jnp.tanh(jnp.dot(want, w[layer], precision="highest"))
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-4, atol=1e-4)
    assert prog.verify(x, w) < 1e-4
    lowered = prog.lower(x, w)
    # `layer` only indexes `w`: a runtime scalar of one kernel, and the loop runs as a fori_loop.
    assert lowered.names == ["deep_h"] and lowered["deep_h"].scalars == ["layer"]
    assert len(prog.kernels) == 1 and len(prog.kernels[0].kernels) == 1
    assert prog.unrolled == []


# --------------------------------------------------------------------------- (d) a transformer block

H, DH = 2, 128  # heads and head dim: D == H * DH
TQ = TK = 128
LANES = 128
EPS = 1e-6
MASK_VALUE = -1e30


def transformer_block(x, g, wqkv, wo):
    seq = x.shape[0]
    # RMSNorm: whole rows per tile, the weight as a (1, D) row.
    xn = jnp.zeros((seq, D))
    for i in range(0, seq, T):
        row = x[i : i + T, :]
        scale = jax.lax.rsqrt(jnp.mean(row * row, axis=1, keepdims=True) + EPS)
        xn = xn.at[i : i + T, :].set(row * scale * g[:, :])
    # Fused QKV projection.
    qkv = jnp.zeros((seq, 3 * D))
    for i in range(0, seq, T):
        for j in range(0, 3 * D, T):
            acc = jnp.zeros((T, T))
            for k in range(0, D, T):
                acc += xn[i : i + T, k : k + T] @ wqkv[k : k + T, j : j + T]
            qkv = qkv.at[i : i + T, j : j + T].set(acc)
    qkv = qkv.reshape(seq, 3, H, DH).transpose(1, 2, 0, 3)  # (3, H, S, DH)
    # Causal flash attention, one head and query block per tile.
    attn = jnp.zeros((H, seq, DH))
    for hd in range(H):
        for i in range(0, seq, TQ):
            row_max = jnp.full((TQ, LANES), MASK_VALUE)
            row_sum = jnp.zeros((TQ, LANES))
            acc = jnp.zeros((TQ, DH))
            for j in range(0, seq, TK):
                if j > i + TQ - 1:
                    continue
                s = (qkv[0, hd, i : i + TQ, :] @ qkv[1, hd, j : j + TK, :].T) * (1 / math.sqrt(DH))
                rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                s = jnp.where(cols <= rows, s, MASK_VALUE)
                m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                alpha = jnp.exp(row_max - m_new)
                row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                acc = jnp.tile(alpha, (1, DH // LANES)) * acc + p @ qkv[2, hd, j : j + TK, :]
                row_max = m_new
            attn = attn.at[hd, i : i + TQ, :].set(acc / jnp.tile(row_sum, (1, DH // LANES)))
    attn = attn.transpose(1, 0, 2).reshape(seq, D)
    # Output projection and residual.
    out = jnp.zeros((seq, D))
    for i in range(0, seq, T):
        for j in range(0, D, T):
            acc = jnp.zeros((T, T))
            for k in range(0, D, T):
                acc += attn[i : i + T, k : k + T] @ wo[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] + acc)
    return out


def plain_transformer_block(x, g, wqkv, wo):
    hp = "highest"
    xn = x * jax.lax.rsqrt(jnp.mean(x * x, axis=1, keepdims=True) + EPS) * g
    q, k, v = jnp.dot(xn, wqkv, precision=hp).reshape(-1, 3, H, DH).transpose(1, 2, 0, 3)
    s = jnp.einsum("hqd,hkd->hqk", q, k, precision=hp) / math.sqrt(DH)
    causal = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), bool))
    p = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
    attn = jnp.einsum("hqk,hkd->hqd", p, v, precision=hp).transpose(1, 0, 2).reshape(x.shape)
    return x + jnp.dot(attn, wo, precision=hp)


def block_inputs(seq=256):
    return rand((seq, D), 1), 1 + rand((1, D), 2) / 4, rand((D, 3 * D), 3) / 16, rand((D, D), 4) / 16


@pytest.mark.parametrize("fuse", [False, True])
def test_transformer_block(fuse):
    prog = autokernel.tpu(transformer_block, fuse=fuse)
    args = block_inputs()
    out = prog(*args)
    np.testing.assert_allclose(out, plain_transformer_block(*args), rtol=1e-4, atol=1e-4)
    assert prog.verify(*args) < 1e-4
    lowered = prog.lower(*args)
    names = ["transformer_block_xn", "transformer_block_qkv", "transformer_block_attn", "transformer_block_out"]
    # Fused: the reshapes and transposes between the nests are views (the attention reads q, k and v as
    # windows of `qkv`), so nothing separates the nests: three to a kernel.
    assert lowered.names == (["+".join(names[:3]), names[3]] if fuse else names)
    attention = lowered["transformer_block_attn"]
    assert attention.steps == H * 3  # causal: 1 + 2 key blocks for each head's two query blocks


def test_transformer_block_kernels_lower_to_mosaic():
    """The interpreter skips Mosaic: cross-lower every kernel of the program for a v5e."""
    device = jax.sharding.AbstractDevice(device_kind="TPU v5 lite", num_cores=1, platform="tpu")
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=device)
    lowered = autokernel.tpu(transformer_block).lower(*block_inputs())
    assert len(lowered.inputs) == len(lowered) == 2  # three nests fused (through views), then one
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        with jax.sharding.use_abstract_mesh(mesh):
            jax.jit(lambda *xs, k=kernel: k.entry(*xs, interpret=False)).trace(*specs).lower(
                lowering_platforms=("tpu",)
            )


# --------------------------------------------------------------------------- (e) under jax.jit


def test_program_under_jit():
    prog = autokernel.tpu(matmul_bias_relu)
    x, w, b = rand((M, K), 1), rand((K, N), 2), rand((1, N), 3)

    @jax.jit
    def f(x, w, b):
        return prog(x * 2, w, b) - 1

    np.testing.assert_allclose(f(x, w, b), prog.reference(x * 2, w, b) - 1, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- (f) lower and source


def test_lower_and_source_give_one_kernel_per_nest():
    prog = autokernel.tpu(matmul_bias_relu, fuse=False)
    specs = [jax.ShapeDtypeStruct(s, jnp.float32) for s in ((M, K), (K, N), (1, N))]
    lowered = prog.lower(*specs)
    assert len(lowered) == 2 and lowered.names == ["matmul_bias_relu_h", "matmul_bias_relu_out"]
    assert all(isinstance(lw, autokernel.Lowered) for _, lw in lowered)
    assert lowered[0][1].steps == 2 * 2 * 2 and lowered[1][1].steps == 2 * 2
    assert [op.param for op in lowered["matmul_bias_relu_out"].operands] == ["h", "b"]
    source = prog.source(*specs)
    assert "kernel 1/2: matmul_bias_relu_h" in source and "kernel 2/2: matmul_bias_relu_out" in source
    assert "def matmul_bias_relu_h_pallas(x, w, *, interpret=False):" in source
    assert "grid=" not in source  # gridless kernels


# --------------------------------------------------------------------------- control flow and options


def scaled(x, relu):
    if relu:  # a static (non-array) argument: plain Python control flow
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(jnp.maximum(x[i : i + T, :], 0))
    else:
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * 2)
    return out


def test_nests_under_python_if_with_static_arguments():
    prog = autokernel.tpu(scaled, verify=True)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x, True), jnp.maximum(x, 0))
    np.testing.assert_allclose(prog(x, False), x * 2)
    assert prog.lower(x, False).names == ["scaled_out_2"]


TA = TB = 128  # tuned


def two_tilings(x):
    y = jnp.zeros((M, N))
    for i in range(0, M, TA):
        y = y.at[i : i + TA, :].set(x[i : i + TA, :] * 2)
    z = jnp.zeros((M, N))
    for i in range(0, M, TB):
        z = z.at[i : i + TB, :].set(y[i : i + TB, :] + 1)
    return z


def test_tuning_only_uses_the_constants_each_nest_reads():
    prog = autokernel.tpu(two_tilings, tune={"TA": (64, 128), "TB": (32, 128)})
    x = rand((M, N))
    tuning = prog.tuning(x)
    assert sorted(tuning) == ["two_tilings_y", "two_tilings_z"]
    assert {tuple(c.overrides) for c in tuning["two_tilings_y"]} == {("TA",)}
    assert {tuple(c.overrides) for c in tuning["two_tilings_z"]} == {("TB",)}
    assert len(tuning["two_tilings_y"]) == len(tuning["two_tilings_z"]) == 2
    np.testing.assert_allclose(prog(x), x * 2 + 1)


TC = 128  # tuned to 256 below


def tile_granular(x):
    y = jnp.zeros((M, N))
    for i in range(0, M, TC):
        for j in range(0, N, T):
            if j > i:  # whole tiles: what it keeps depends on TC
                continue
            y = y.at[i : i + TC, j : j + T].set(x[i : i + TC, j : j + T])
    return y * 2


def test_reference_uses_each_kernels_tuned_constants():
    prog = autokernel.tpu(tile_granular, tune={"TC": (256,)})
    x = rand((M, N))
    assert prog.lower(x)[0][1].overrides == {"TC": 256}
    prog.verify(x)
    np.testing.assert_array_equal(prog.reference(x)[128:, 128:], 0)  # TC = 256: the tile at j = 128 is skipped


def test_nest_with_two_outputs():
    def f(x):
        y = jnp.zeros((M, N))
        z = jnp.zeros((M, N), jnp.int32)
        for i in range(0, M, T):
            y = y.at[i : i + T, :].set(x[i : i + T, :] * 2)
            z = z.at[i : i + T, :].set((x[i : i + T, :] > 0).astype(jnp.int32))
        return y * z

    prog = autokernel.tpu(f)
    assert prog.kernels[0].outputs == ["y", "z"]
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), jnp.where(x > 0, x * 2, 0))
    assert prog.verify(x) == 0


def test_closures_and_constants_from_the_function():
    eps = 0.5  # a closure variable

    def f(x):
        n = x.shape[0] // 2  # a local constant
        out = jnp.zeros(x.shape, x.dtype)
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * n + eps)
        return out - 1

    prog = autokernel.tpu(f)
    x = rand((M, N))
    np.testing.assert_allclose(prog(x), x * (M // 2) + eps - 1, rtol=1e-6)


# --------------------------------------------------------------------------- (g) errors


def _load(tmp_path, src):
    path = tmp_path / "mod.py"
    path.write_text("import jax.numpy as jnp\n" + src)
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.f


@pytest.mark.parametrize(
    "src, match",
    [
        (  # unsupported code inside a nest
            """
def f(a):
    b = a * 2
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        print(i)
        out = out.at[i:i + 8, :].set(b[i:i + 8, :])
    return out
""",
            r"f_out \(line 7\): loop bodies may only contain",
        ),
        (  # a temporary of a nest read after it
            """
def f(a):
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        t = a[i:i + 8, :] * 2
        out = out.at[i:i + 8, :].set(t)
    return out + t
""",
            r"f \(line 8\): `t` is read here, but its value may come from the tile loop nest at line 5, where it is "
            "a temporary",
        ),
        (
            """
def f(a):
    return a + 1
""",
            "found no tile loop nest",
        ),
    ],
)
def test_rejects_bad_programs_at_decoration_time(tmp_path, src, match):
    with pytest.raises(AutoKernelError, match=match):
        autokernel.tpu(_load(tmp_path, src))


def test_errors_inside_a_nest_name_its_line(tmp_path):
    src = """
def f(a):
    b = a * 2
    out = jnp.zeros((64, 128))
    for i in range(0, 64, 8):
        out = out.at[i:i + 8, :].set(b[i:i + 8, :] + b.sum())
    return out
"""
    prog = autokernel.tpu(_load(tmp_path, src))
    with pytest.raises(AutoKernelError, match=r"f_out \(line 7\): input `b` may only be read through a tile slice"):
        prog(rand((64, 128)))


def test_unsupported_constant_is_an_error():
    def f(x):
        sizes = [1, 2]
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * sizes[0])
        return out + 1

    with pytest.raises(AutoKernelError, match="`sizes` is a list"):
        autokernel.tpu(f)(rand((M, N)))
