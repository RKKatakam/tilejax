"""The cost model: VMEM accounting, MXU and vector pricing, and calibration against measured v5e timings."""

import math

import jax
import jax.numpy as jnp
import pytest

import autokernel
from autokernel import cost, hardware
from autokernel import tune as tune_lib

S = jax.ShapeDtypeStruct
V5E = dict(target="v5e")
MASK_VALUE = -1e30
LANES = 128


def matmul(size, tm, tn, tk, dtype=jnp.bfloat16):
    def dense(x, y):
        out = jnp.zeros((size, size), dtype)
        for i in range(0, size, tm):
            for j in range(0, size, tn):
                acc = jnp.zeros((tm, tn), jnp.float32)
                for k in range(0, size, tk):
                    acc += x[i : i + tm, k : k + tk] @ y[k : k + tk, j : j + tn]
                out = out.at[i : i + tm, j : j + tn].set(acc.astype(dtype))
        return out

    return dense, [S((size, size), dtype)] * 2


def attention(B, H, L, D, TQ, TK):
    """Causal flash attention as in benchmarks/attention.py (lane-wide row statistics)."""
    scale = 1.0 / math.sqrt(D)

    def causal_attention(q, k, v):
        dtype = q.dtype
        out = jnp.zeros((B, H, L, D), dtype)
        for b in range(0, B, 1):
            for h in range(0, H, 1):
                for i in range(0, L, TQ):
                    row_max = jnp.full((TQ, LANES), MASK_VALUE)
                    row_sum = jnp.zeros((TQ, LANES))
                    acc = jnp.zeros((TQ, D))
                    for j in range(0, L, TK):
                        if j > i + TQ - 1:
                            continue
                        s = (q[b, h, i : i + TQ, :] @ k[b, h, j : j + TK, :].T) * scale
                        rows = i + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 0)
                        cols = j + jax.lax.broadcasted_iota(jnp.int32, (TQ, TK), 1)
                        s = jnp.where(cols <= rows, s, MASK_VALUE)
                        m_new = jnp.maximum(row_max, jnp.max(s, axis=1, keepdims=True))
                        p = jnp.exp(s - jnp.tile(m_new, (1, TK // LANES)))
                        alpha = jnp.exp(row_max - m_new)
                        row_sum = alpha * row_sum + jnp.sum(p, axis=1, keepdims=True)
                        acc = jnp.tile(alpha, (1, D // LANES)) * acc + p.astype(dtype) @ v[b, h, j : j + TK, :]
                        row_max = m_new
                    out = out.at[b, h, i : i + TQ, :].set((acc / jnp.tile(row_sum, (1, D // LANES))).astype(dtype))
        return out

    return causal_attention, [S((B, H, L, D), jnp.bfloat16)] * 3


def lower(fn_specs, **options):
    fn, specs = fn_specs
    return autokernel.tpu(fn, **{**V5E, **options}).lower(*specs)


# --------------------------------------------------------------------------- VMEM


def test_vmem_buffers_are_padded_to_whole_tiles():
    M, C, T = 8192, 8, 1024

    def narrow(x):
        out = jnp.zeros((M, C))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
        return out

    lw = autokernel.tpu(narrow, **V5E).lower(S((M, C), jnp.float32))
    padded = hardware.vmem_nbytes((T, C), jnp.float32)
    assert padded == T * 128 * 4  # 8 lanes take a whole 128-lane row
    assert lw.estimate.vmem_bytes >= 2 * padded + 2 * padded  # two input slots, two output slots
    assert hardware.vmem_nbytes((3, 128), jnp.bfloat16) == 16 * 128 * 2  # bf16 rows pad to 16


def test_intermediates_count_their_peak_not_their_sum():
    """Attention's body has a dozen (TQ, TK) temporaries, but only a few are live at once."""
    lw = lower(attention(1, 1, 8192, 128, 1024, 1024))
    scores = 1024 * 1024 * 4
    assert lw.estimate.vmem_bytes < 6 * scores  # it was 51.5 MiB summing all of them
    big = lower(attention(1, 1, 8192, 128, 1024, 1024), target="v5p")  # 48 MiB budget: fits now
    assert big.estimate.vmem_bytes == lw.estimate.vmem_bytes


def test_accumulating_matmul_is_in_place():
    """`acc += x @ y`: Mosaic accumulates into the matmul's result, so one (TM, TN) temporary, not two."""
    lw = lower(matmul(4096, 2048, 2048, 512))
    est = lw.estimate
    tm_tn_f32 = 2048 * 2048 * 4
    rings = 3 * 2 * 2048 * 512 * 2
    outputs = 2 * 2048 * 2048 * 2
    assert est.vmem_bytes == rings + outputs + tm_tn_f32 + tm_tn_f32  # + the acc state + its update


# --------------------------------------------------------------------------- compute


def bf16_first(a, b, c, d):
    out = jnp.zeros((512, 512))
    for i in range(0, 512, 256):
        for j in range(0, 512, 256):
            acc = jnp.zeros((256, 256))
            for k in range(0, 512, 256):
                acc += a[i : i + 256, k : k + 256] @ b[k : k + 256, j : j + 256]
                acc += c[i : i + 256, k : k + 256] @ d[k : k + 256, j : j + 256]
            out = out.at[i : i + 256, j : j + 256].set(acc)
    return out


def f32_first(a, b, c, d):
    out = jnp.zeros((512, 512))
    for i in range(0, 512, 256):
        for j in range(0, 512, 256):
            acc = jnp.zeros((256, 256))
            for k in range(0, 512, 256):
                acc += c[i : i + 256, k : k + 256] @ d[k : k + 256, j : j + 256]
                acc += a[i : i + 256, k : k + 256] @ b[k : k + 256, j : j + 256]
            out = out.at[i : i + 256, j : j + 256].set(acc)
    return out


MIXED = [S((512, 512), jnp.bfloat16)] * 2 + [S((512, 512), jnp.float32)] * 2


def test_each_matmul_is_priced_at_its_own_dtype():
    """Each dot runs at its own dtype's MXU rate, whichever comes last (f32 at HIGHEST: six bf16 passes)."""
    a, b = lower((bf16_first, MIXED)).estimate, lower((f32_first, MIXED)).estimate
    assert a.seconds == b.seconds
    fast = lower((bf16_first, MIXED), matmul_precision="default").estimate  # f32 in one bf16 pass
    assert a.seconds > fast.seconds


def test_accumulation_overlaps_the_mxu():
    """The per-step `acc += dot` add runs beside the MXU: a longer K doesn't make steps cheaper per FLOP,
    so the model has no reason to prefer TK=2048 (it used to)."""
    tk1024 = lower(matmul(8192, 1024, 2048, 1024)).estimate
    tk2048 = lower(matmul(8192, 1024, 2048, 2048)).estimate
    assert tk2048.seconds == pytest.approx(tk1024.seconds, rel=0.01)


def test_tuning_builds_a_diverse_top_few():
    fn, specs = matmul(8192, 512, 512, 512)
    candidates = autokernel.tpu(fn, tune="auto", **V5E).tuning(*specs)
    built = [c for c in candidates if c.lowered is not None]
    assert len(built) == 8 and built[0] is candidates[0]
    for name in ("tm", "tn", "tk"):
        values = [c.overrides[name] for c in built]
        assert max(values.count(v) for v in values) <= 4
    best = candidates[0].estimate
    assert best.vmem_bytes < 0.75 * hardware.target("v5e").vmem_budget  # not at the budget any more


def test_diverse_caps_each_value_at_half():
    def ranked(*traits):
        return [tune_lib.Candidate({"T": t, "U": u}, 3, None) for t, u in traits]

    picked = tune_lib.diverse(ranked((1, 1), (1, 2), (1, 3), (2, 1), (2, 2)), 4)
    assert [(c.overrides["T"], c.overrides["U"]) for c in picked] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert len(tune_lib.diverse(ranked((1, 1), (1, 2), (1, 3), (1, 4)), 4)) == 4  # T has one value: not capped
    tight = ranked((1, 1), (1, 1), (1, 1), (2, 1), (1, 2))
    assert tune_lib.diverse(tight, 4) == tight[:4]  # the cap leaves two: the best of the rest fill up


def test_near_budget_vmem_is_penalised_in_the_score():
    small = lower(matmul(8192, 1024, 1024, 1024)).estimate
    huge = lower(matmul(8192, 2048, 2048, 2048), buffers=2).estimate
    assert small.score == small.seconds
    assert huge.vmem_bytes > 0.75 * hardware.target("v5e").vmem_budget and huge.score > huge.seconds


def test_overhead_bound_kernels_are_flagged():
    assert cost.overhead_bound(lower(matmul(2048, 128, 128, 128)).estimate)
    assert not cost.overhead_bound(lower(matmul(2048, 512, 512, 512)).estimate)
    assert not cost.overhead_bound(lower(matmul(256, 128, 128, 128)).estimate)  # too few steps to matter


# --------------------------------------------------------------------------- calibration (TPU v5e, README)

MEASURED = [  # (kernel, buffers, measured ms on a v5e)
    (matmul(8192, 1024, 2048, 1024), None, 5.73),
    (matmul(8192, 512, 512, 512), 3, 6.31),
    (matmul(8192, 512, 512, 512), 2, 7.37),
    (matmul(8192, 512, 1024, 1024, jnp.float32), None, 33.84),
    (attention(4, 16, 4096, 128, 256, 256), None, 5.13),
    (attention(4, 16, 4096, 128, 512, 256), None, 3.62),
    (attention(4, 16, 4096, 128, 512, 512), None, 2.41),
    (attention(4, 16, 4096, 128, 1024, 512), None, 2.51),
]


@pytest.fixture(scope="module")
def predicted_ms():
    return [lower(k, buffers=b).estimate.seconds * 1e3 for k, b, _ in MEASURED]


# How far each prediction may be from its measurement: the model's error when calibrated (+1.8%, +4.8%,
# -0.8%, +2.2%, -15.5%, +1.3%, +9.6%, +6.6%) with a margin of a few percent, so that a change to the model
# that loses accuracy on any one kernel fails here.
TOLERANCE = [0.06, 0.09, 0.05, 0.06, 0.20, 0.05, 0.14, 0.11]


def test_predictions_are_close_to_measured_times(predicted_ms):
    for (_, _, measured), predicted, rel in zip(MEASURED, predicted_ms, TOLERANCE, strict=True):
        assert predicted == pytest.approx(measured, rel=rel)


def test_attention_block_sizes_rank_like_the_hardware(predicted_ms):
    attn = predicted_ms[4:]
    measured = [m for _, _, m in MEASURED[4:]]
    assert sorted(range(4), key=attn.__getitem__) == sorted(range(4), key=measured.__getitem__)
    assert predicted_ms[1] < predicted_ms[2]  # three buffers beat two for 512^3 tiles


def test_streamed_tiles_count_every_time_they_are_fetched():
    """HBM traffic counts a tile once per fetch: x is read once per column tile, y once per row tile."""
    size, tm, tn, tk = 4096, 512, 1024, 256
    est = lower(matmul(size, tm, tn, tk)).estimate
    x = y = out = size * size * 2
    assert est.hbm_bytes == x * (size // tn) + y * (size // tm) + out
    assert est.hbm_s == pytest.approx(est.hbm_bytes / (hardware.target("v5e").hbm_bytes_per_s * cost.HBM_EFFICIENCY))


def test_deeper_buffers_hide_dma_latency():
    """Two buffers start a step's copies one step ahead, so their latency shows; three hide it."""
    compute, dma = 1.0e-6, 0.9e-6
    two, three = (cost.step_seconds(compute, dma, 2, b) for b in (2, 3))
    assert two == pytest.approx(cost.STEP_OVERHEAD_S + dma + cost.DMA_LATENCY_S)
    assert three == pytest.approx(cost.STEP_OVERHEAD_S + compute)
    assert cost.step_seconds(0.1e-6, 2e-6, 2, 3) == 2e-6  # DMA-bound: the transfer itself


# --------------------------------------------------------------------------- the transformer's projections

LANEseqT, EPS = 128, 1e-6


def qkv(B, seq, D, L, TM, TN, TK):
    """benchmarks/transformer.py's QKV projection with RMSNorm folded in (`layer` a runtime scalar)."""

    def transformer_q(x, norm1, wq, wk, wv, layer):
        q = jnp.zeros((B, seq, D), jnp.bfloat16)
        k = jnp.zeros((B, seq, D), jnp.bfloat16)
        v = jnp.zeros((B, seq, D), jnp.bfloat16)
        for b in range(B):
            for i in range(0, seq, TM):
                for j in range(0, D, TN):
                    aq = jnp.zeros((TM, TN))
                    ak = jnp.zeros((TM, TN))
                    av = jnp.zeros((TM, TN))
                    ss = jnp.zeros((TM, LANEseqT))
                    for kk in range(0, D, TK):
                        xf = x[b, i : i + TM, kk : kk + TK].astype(jnp.float32)
                        ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                        xt = (xf * norm1[layer, :, kk : kk + TK]).astype(jnp.bfloat16)
                        aq += xt @ wq[layer, kk : kk + TK, j : j + TN]
                        ak += xt @ wk[layer, kk : kk + TK, j : j + TN]
                        av += xt @ wv[layer, kk : kk + TK, j : j + TN]
                    inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TN // LANEseqT))
                    q = q.at[b, i : i + TM, j : j + TN].set((aq * inv).astype(jnp.bfloat16))
                    k = k.at[b, i : i + TM, j : j + TN].set((ak * inv).astype(jnp.bfloat16))
                    v = v.at[b, i : i + TM, j : j + TN].set((av * inv).astype(jnp.bfloat16))
        return q, k, v

    w = S((L, D, D), jnp.bfloat16)
    return transformer_q, [S((B, seq, D), jnp.bfloat16), S((L, 1, D), jnp.float32), w, w, w, S((), jnp.int32)]


def residual_proj(B, seq, K, N, L, TM, TN, TK):
    """The output and down projections: x + a @ w[layer]."""

    def transformer_x(a, w, x, layer):
        out = jnp.zeros((B, seq, N), jnp.bfloat16)
        for b in range(B):
            for i in range(0, seq, TM):
                for j in range(0, N, TN):
                    acc = jnp.zeros((TM, TN))
                    for kk in range(0, K, TK):
                        acc += a[b, i : i + TM, kk : kk + TK] @ w[layer, kk : kk + TK, j : j + TN]
                    out = out.at[b, i : i + TM, j : j + TN].set(
                        (x[b, i : i + TM, j : j + TN] + acc).astype(jnp.bfloat16)
                    )
        return out

    bf = jnp.bfloat16
    return transformer_x, [S((B, seq, K), bf), S((L, K, N), bf), S((B, seq, N), bf), S((), jnp.int32)]


def up(B, seq, D, F, L, TM, TN, TK):
    """The MLP's up projection with RMSNorm and ReLU folded in."""

    def transformer_up(x1, norm2, w_up, layer):
        out = jnp.zeros((B, seq, F), jnp.bfloat16)
        for b in range(B):
            for i in range(0, seq, TM):
                for j in range(0, F, TN):
                    acc = jnp.zeros((TM, TN))
                    ss = jnp.zeros((TM, LANEseqT))
                    for kk in range(0, D, TK):
                        xf = x1[b, i : i + TM, kk : kk + TK].astype(jnp.float32)
                        ss += jnp.sum(xf * xf, axis=1, keepdims=True)
                        xn = (xf * norm2[layer, :, kk : kk + TK]).astype(jnp.bfloat16)
                        acc += xn @ w_up[layer, kk : kk + TK, j : j + TN]
                    inv = jnp.tile(jax.lax.rsqrt(ss / D + EPS), (1, TN // LANEseqT))
                    out = out.at[b, i : i + TM, j : j + TN].set(jnp.maximum(acc * inv, 0).astype(jnp.bfloat16))
        return out

    bf = jnp.bfloat16
    return transformer_up, [S((B, seq, D), bf), S((L, 1, D), jnp.float32), S((L, D, F), bf), S((), jnp.int32)]


PROJECTIONS = {  # B=1, S=2048, D=2048, F=8192, 4 layers
    "qkv": lambda t: qkv(1, 2048, 2048, 4, *t),
    "out": lambda t: residual_proj(1, 2048, 2048, 2048, 4, *t),
    "up": lambda t: up(1, 2048, 2048, 8192, 4, *t),
    "down": lambda t: residual_proj(1, 2048, 8192, 2048, 4, *t),
}
# Each projection timed alone on a v5e (us; default buffers, or two where three didn't fit). The QKV
# projection's 256x2048x2048 is what `autotune="measure"` once picked: it fetches all three 2048x2048 weights
# again for each of the 8 row tiles (192 MiB of weight traffic, against 48 MiB for 1024x1024x512).
SWEPT = {
    "qkv": {(1024, 1024, 512): 306.5, (1024, 1024, 1024): 311.0, (512, 1024, 1024): 306.7, (1024, 512, 1024): 306.1,
            (2048, 512, 512): 307.4, (512, 2048, 512): 309.1, (256, 2048, 2048): 357.4},
    "out": {(1024, 1024, 1024): 110.6, (2048, 1024, 1024): 119.2, (1024, 2048, 1024): 118.7,
            (2048, 2048, 512): 128.7, (2048, 512, 2048): 128.7, (1024, 2048, 2048): 123.1},
    "up": {(1024, 2048, 2048): 392.0},
    "down": {(1024, 1024, 1024): 379.3, (2048, 1024, 1024): 385.6, (1024, 2048, 1024): 384.1,
             (2048, 2048, 512): 392.6, (512, 2048, 2048): 384.0, (1024, 2048, 2048): 390.4},
}  # fmt: skip


@pytest.fixture(scope="module")
def swept_us():
    return {
        nest: {tiles: lower(PROJECTIONS[nest](tiles)).estimate.seconds * 1e6 for tiles in sweep}
        for nest, sweep in SWEPT.items()
    }


def test_projections_are_predicted_within_ten_percent(swept_us):
    for nest, sweep in SWEPT.items():
        for tiles, measured in sweep.items():
            assert swept_us[nest][tiles] == pytest.approx(measured, rel=0.1), (nest, tiles)


def test_refetching_every_weight_per_row_tile_ranks_last(swept_us):
    qkv_us = swept_us["qkv"]
    bad = qkv_us.pop((256, 2048, 2048))
    assert bad > 1.08 * max(qkv_us.values())  # measured: 357 against at most 311
    for nest in ("out", "down"):  # the measured best of each sweep is predicted best
        assert min(swept_us[nest], key=swept_us[nest].get) == (1024, 1024, 1024)


@pytest.mark.parametrize("nest", sorted(PROJECTIONS))
def test_auto_tuning_beats_the_hand_swept_blocks(nest, swept_us):
    fn, specs = PROJECTIONS[nest]((1024, 1024, 512))
    best = autokernel.tpu(fn, tune="auto", **V5E).tuning(*specs)[0]
    assert best.predicted_s * 1e6 <= min(swept_us[nest].values()) * 1.002


# --------------------------------------------------------------------------- decode: weight streaming


def test_decode_mlp_streams_weights_at_xla_speed():
    """64 rows through the MLP (2048 -> 8192 -> 2048), as in a fori_loop over layers. Measured on a v5e with
    512x512 blocks: 103 us a layer, XLA 95 us. The model puts the gap in the activations fetched again
    for every column tile (8 MiB a layer) and in small blocks; the tuned blocks stream each weight once."""
    rows = 64
    fixed = [up(1, rows, 2048, 8192, 4, rows, 512, 512), residual_proj(1, rows, 8192, 2048, 4, rows, 512, 512)]
    fixed_us = sum(lower(k).estimate.in_program_s for k in fixed) * 1e6
    assert fixed_us == pytest.approx(103, rel=0.1)
    tuned = [autokernel.tpu(fn, tune="auto", **V5E).lower(*specs) for fn, specs in fixed]
    tuned_us = sum(lw.estimate.in_program_s for lw in tuned) * 1e6
    assert tuned_us <= 95 and all(lw.overrides["TN"] == 2048 for lw in tuned)
    weights = 2 * 2048 * 8192 * 2
    assert sum(lw.estimate.hbm_bytes for lw in tuned) < 1.06 * weights


def test_weight_streaming_kernels_match_measured_times():
    """256 rows, 512x512 blocks, in a fori_loop on a v5e: a = x @ w1[layer] (2048 -> 8192) 66 us, then
    a @ w2[layer] (8192 -> 2048) 61 us."""

    def proj(rows, K, N, T):
        def proj(x, w, layer):
            a = jnp.zeros((rows, N), jnp.bfloat16)
            for i in range(0, rows, rows):
                for j in range(0, N, T):
                    acc = jnp.zeros((rows, T))
                    for k in range(0, K, T):
                        acc += x[i : i + rows, k : k + T] @ w[layer, k : k + T, j : j + T]
                    a = a.at[i : i + rows, j : j + T].set(acc.astype(jnp.bfloat16))
            return a

        bf = jnp.bfloat16
        return lower((proj, [S((rows, K), bf), S((4, K, N), bf), S((), jnp.int32)])).estimate.in_program_s * 1e6

    assert proj(256, 2048, 8192, 512) == pytest.approx(66, rel=0.15)
    assert proj(256, 8192, 2048, 512) == pytest.approx(61, rel=0.15)


# --------------------------------------------------------------------------- narrow integers on the VPU


def int_kernel(dtype, body):
    def f(x):
        out = jnp.zeros((256, 256), jnp.int32)
        for i in range(0, 256, 128):
            y = body(x[i : i + 128, :])
            out = out.at[i : i + 128, :].set(y.astype(jnp.int32))
        return out

    return f, [S((256, 256), dtype)]


@pytest.mark.parametrize("body", [lambda t: t * 2, lambda t: jnp.maximum(t, 0), lambda t: t + t, lambda t: t > 3])
def test_int8_vector_arithmetic_is_rejected_at_planning(body):
    with pytest.raises(autokernel.AutoKernelError, match=r"\(line \d+\): `y = .*int8 vectors.*astype\(jnp.int32\)"):
        lower(int_kernel(jnp.int8, body))


def test_int8_loads_converts_and_matmuls_are_fine():
    lower(int_kernel(jnp.int8, lambda t: t.astype(jnp.int32) * 2))  # widened first
    lower(int_kernel(jnp.int16, lambda t: t * 2))  # 16-bit vectors are supported
    lower(int_kernel(jnp.int8, lambda t: jnp.where(t.astype(jnp.int32) > 0, t, 0)))  # a select of int8 values
    lower(int_kernel(jnp.int8, lambda t: t[:, :128] @ t[:128, :]))  # the MXU takes int8


def test_vpu_integer_widths_are_per_chip(monkeypatch):
    assert hardware.target("v5e").vpu_int_bits == (16, 32)
    assert all(hardware.target(chip).vpu_int_bits == (16, 32) for chip in ("v4", "v5p", "v6e"))
    with pytest.raises(autokernel.AutoKernelError, match="v6e's vector unit has no 8-bit"):
        lower(int_kernel(jnp.int8, lambda t: t * 2), target="v6e")
