"""Multi-head latent attention (DeepSeek-V3 / Kimi K2 MLA), prefill, bf16, on a TPU v5e: autokernel's kernels,
each tuned, against the same block in plain JAX (XLA) and with JAX's splash attention (q/k head dim 192, and
zero-padded to 256; at splash's default block sizes and at its best over a grid).

    uv run python benchmarks/mla.py
    uv run python benchmarks/mla.py --shape 1,4096,2048,16,768      # batch,seq,d_model,heads,q_rank

Per shape: the whole block (x -> x + attention), then its parts: the projections (RMSNorm + down projections
+ latent norms + RoPE, then the up projections), against XLA's; and the attention with the output projection
and residual, against splash attention on ready-made (B, H, S, d) inputs + the output projection in XLA.
(`--interpret` runs a tiny shape in the Pallas TPU interpreter, off a TPU, to check the script itself.)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from matmul import matmul
from mla import (
    LANES,
    NOPE,
    QK,
    ROPE,
    MlaConfig,
    baseline,
    init_params,
    latents,
    mla_attention,
    mla_block,
    mla_down,
    pack,
    rope_tables,
    splash_baseline,
)
from timing import require_tpu, time_ms

SHAPES = [  # batch, seq, d_model, heads, q_rank (kv_rank 512, head dims 128 + 64, v 128)
    "1,4096,2048,16,768",
    "2,4096,2048,16,1536",
    "4,2048,2048,16,1536",
    "1,4096,2048,32,1536",  # Kimi K2's ratio of heads to d_model
    "2,2048,4096,32,1536",
    "1,2048,4096,64,1536",
]
XLA_SCORES_LIMIT = 2**30  # bytes of f32 (B, H, S, S) scores the XLA baseline may materialize
BLOCKS = (512, 1024)  # splash's block grid (q and kv)


def flops(config: MlaConfig, batch: int, seq: int) -> dict[str, float]:
    """Model FLOPs of the parts (causal attention computing half the scores)."""
    t, d, h, rq, rkv = batch * seq, config.d_model, config.heads, config.q_rank, config.kv_rank
    projections = 2 * t * d * (rq + rkv + ROPE) + 2 * t * rq * h * QK + 2 * t * rkv * h * 2 * NOPE
    attention = batch * h * seq * seq * (QK + NOPE) + 2 * t * h * NOPE * d
    return {"projections": projections, "attention": attention, "block": projections + attention}


def projections(x: jax.Array, packed: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
    """mla_block's kernels before the attention: q, kv and the rope key."""
    b, s, _ = x.shape
    cos, sin = rope_tables(s)
    c_q, c_kv, k_rope = mla_down(
        x, packed["attn_norm"], packed["w_down"], packed["q_norm"], packed["kv_norm"], cos, sin
    )
    q = matmul(c_q.reshape(b * s, -1), packed["w_uq"]).reshape(b, s, -1)
    kv = matmul(c_kv.reshape(b * s, -1), packed["w_ukv"]).reshape(b, s, -1)
    return q, kv, k_rope


def xla_projections(x: jax.Array, params: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
    """The same in plain JAX: q_nope, q_rope, k_nope, v (B, S, H, d) and the rope key."""
    q_nope, q_rope, c_kv, k_rope = latents(x, params)
    k_nope, v = (jnp.einsum("bsr,rhe->bshe", c_kv, params[w]).astype(x.dtype) for w in ("w_uk", "w_uv"))
    return q_nope, q_rope, k_nope, v, k_rope


def splash_attention_part(heads: int, seq: int, blocks: tuple[int, int] | None, interpret: bool) -> Callable:
    """Splash attention on ready-made (B, H, S, d) q, k, v (q pre-scaled), the output projection and the
    residual in XLA."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib

    mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq, seq)) for _ in range(heads)])
    sizes = (
        None if blocks is None else splash.BlockSizes(block_q=blocks[0], block_kv=blocks[1], block_kv_compute=blocks[1])
    )
    attend = jax.vmap(splash.make_splash_mha_single_device(mask, block_sizes=sizes, interpret=interpret))

    def part(x, q, k, v, w_o):
        o = attend(q, k, v)
        return (x.astype(jnp.float32) + jnp.einsum("bhqe,hed->bqd", o, w_o, preferred_element_type=jnp.float32)).astype(
            x.dtype
        )

    return jax.jit(part)


def splash_inputs(x: jax.Array, params: dict[str, jax.Array], pad: bool) -> tuple[jax.Array, ...]:
    """q (pre-scaled) and k as [nope | rope] per head (192 wide, or 256 padded), v: (B, H, S, d)."""
    q_nope, q_rope, k_nope, v, k_rope = xla_projections(x, params)
    b, s, h, _ = q_nope.shape
    widths = [(0, 0)] * 3 + [(0, 2 * LANES - QK if pad else 0)]
    q = jnp.pad((jnp.concatenate([q_nope, q_rope], -1) * QK**-0.5).astype(x.dtype), widths)
    k = jnp.pad(jnp.concatenate([k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, s, h, ROPE))], -1), widths)
    return tuple(t.transpose(0, 2, 1, 3) for t in (q, k, v))


def best_of(candidates: dict[str, Callable[[], Callable]], *args: Any, **timing: int) -> tuple[float, str, Callable]:
    """The fastest of named candidate functions (built lazily; those that fail to build or run are reported
    and skipped): its time in ms, its name and the function."""
    best = (float("inf"), "none", None)
    for name, make in candidates.items():
        try:
            fn = make()
            t = time_ms(fn, *args, **timing)
        except Exception as error:  # a baseline variant Mosaic can't compile is skipped
            print(f"    {name}: failed ({type(error).__name__}: {str(error).splitlines()[0][:100]})")
            continue
        print(f"    {name:<40} {t:8.3f} ms")
        if t < best[0]:
            best = (t, name, fn)
    return best


def splash_variants(heads: int, seq: int, interpret: bool, part: bool) -> dict[str, Callable[[], Callable]]:
    blocks = [None] + [(min(bq, seq), min(bk, seq)) for bq in BLOCKS for bk in BLOCKS]
    variants = {}
    for pad in (False, True):
        for b in blocks:
            name = f"splash {'256 (padded)' if pad else '192'} {'default' if b is None else f'{b[0]} x {b[1]}'}"
            if part:
                variants[name] = lambda b=b: splash_attention_part(heads, seq, b, interpret)
            else:
                bq, bk = (None, None) if b is None else b
                variants[name] = lambda bq=bq, bk=bk, pad=pad: splash_baseline(heads, seq, bq, bk, pad, interpret)
    return variants


def diff(a: Any, b: Any) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="batch,seq,d_model,heads,q_rank (repeatable)")
    parser.add_argument("--interpret", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    shapes = args.shape or (["1,256,256,2,256"] if args.interpret else SHAPES)
    timing = {"runs": 1, "repeats": 1} if args.interpret else {}
    if not args.interpret:
        require_tpu()
    ours = jax.jit(mla_block)
    ours_projections = jax.jit(projections)
    summary = []
    for batch, seq, d_model, heads, q_rank in [tuple(map(int, shape.split(","))) for shape in shapes]:
        config = MlaConfig(d_model=d_model, heads=heads, q_rank=q_rank)
        params = init_params(config)
        packed = pack(params)
        x = jax.random.normal(jax.random.key(1), (batch, seq, d_model), jnp.bfloat16)
        work = flops(config, batch, seq)
        print(f"\n{config}, batch {batch}, seq {seq}: ms (TFLOP/s)")

        # The whole block.
        print("  baselines, whole block:")
        candidates = splash_variants(heads, seq, args.interpret, part=False)
        if batch * heads * seq * seq * 4 <= XLA_SCORES_LIMIT:
            candidates = {"plain JAX (XLA)": lambda: jax.jit(baseline), **candidates}
        else:
            print(f"    plain JAX (XLA): skipped, its (B, H, S, S) scores exceed {XLA_SCORES_LIMIT >> 20} MiB")
        t_base, base_name, base_fn = best_of(candidates, x, params, **timing)
        t_ours = time_ms(ours, x, packed, **timing)  # tunes each kernel on the first call
        error = diff(ours(x, packed), base_fn(x, params))
        print(f"  best baseline ({base_name}) {t_base:8.3f} ({work['block'] / t_base / 1e9:6.1f})")
        print(f"  autokernel, tuned {t_ours:8.3f} ({work['block'] / t_ours / 1e9:6.1f})")
        print(f"  speedup {t_base / t_ours:.2f}x; max |autokernel - baseline| {error:.4f}")

        # The projections.
        t_xla_proj = time_ms(jax.jit(xla_projections), x, params, **timing)
        t_proj = time_ms(ours_projections, x, packed, **timing)
        print(f"  projections: XLA {t_xla_proj:.3f}, autokernel {t_proj:.3f} ({t_xla_proj / t_proj:.2f}x)")

        # The attention with the output projection and residual, from ready-made inputs.
        q, kv, k_rope = ours_projections(x, packed)
        cos, sin = rope_tables(seq)
        attention_args = (x, q, kv, k_rope, cos, sin, packed["w_o"])
        t_attention = time_ms(jax.jit(mla_attention), *attention_args, **timing)
        t_splash, splash_name = float("inf"), "none"
        for pad in (False, True):
            inputs = jax.jit(splash_inputs, static_argnums=2)(x, params, pad)
            variants = splash_variants(heads, seq, args.interpret, part=True)
            variants = {name: make for name, make in variants.items() if ("padded" in name) == pad}
            t, name, _ = best_of(variants, x, *inputs, params["w_o"], **timing)
            t_splash, splash_name = min((t_splash, splash_name), (t, name))
        print(
            f"  attention + W_o + residual: {splash_name} {t_splash:.3f}, autokernel {t_attention:.3f}"
            f" ({t_splash / t_attention:.2f}x; {work['attention'] / t_attention / 1e9:.1f} TFLOP/s)"
        )
        down_args = (x, packed["attn_norm"], packed["w_down"], packed["q_norm"], packed["kv_norm"], cos, sin)
        rows = batch * seq
        print("  chosen: down     ", mla_down.lower(*down_args).config)
        print("          up q     ", matmul.lower(jax.ShapeDtypeStruct((rows, q_rank), x.dtype), packed["w_uq"]).config)
        c_kv = jax.ShapeDtypeStruct((rows, config.kv_rank), x.dtype)
        print("          up kv    ", matmul.lower(c_kv, packed["w_ukv"]).config)
        print("          attention", mla_attention.lower(*attention_args).config)
        times = (t_base, base_name, t_ours, error, t_proj, t_xla_proj, t_attention, t_splash)
        summary.append(((batch, seq, d_model, heads, q_rank), times))

    print(
        "\nB  S     D     H   Rq    best baseline (ms)                     autokernel  speedup  max diff"
        "  projections (XLA)  attention (splash)"
    )
    for (b, s, d, h, rq), (t_base, name, t_ours, error, t_proj, t_xla_proj, t_att, t_splash) in summary:
        print(
            f"{b:<2} {s:<5} {d:<5} {h:<3} {rq:<5} {t_base:8.3f} {name:<29} {t_ours:10.3f} {t_base / t_ours:7.2f}x"
            f" {error:9.4f}  {t_proj:7.3f} ({t_xla_proj:7.3f})  {t_att:7.3f} ({t_splash:7.3f})"
        )


if __name__ == "__main__":
    main()
