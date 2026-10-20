"""KV-cache decoding on a TPU v5e: autokernel's decode attention kernels, tuned, against XLA and JAX's own
Pallas TPU decode kernels (paged_attention, ragged_paged_attention), bf16, ragged cache lengths (uniform in
[mean / 2, 3 mean / 2)). Decode is bound by HBM bandwidth: each line also reports the bytes the math needs
(the valid cache positions, the weights) per second, against the v5e's ~819 GB/s.

1. GQA decode attention (H 32, Hkv 8, head_dim 128, one new token): `gqa_decode` (contiguous cache),
   `paged_gqa_decode` (pages of 128), XLA (dense over the whole cache, masked), paged_attention (at its
   best pages per compute block) and ragged_paged_attention (at its default and best block sizes).
2. Absorbed MLA decode attention (latent 512, RoPE 64, H 16-128): `mla_decode` against XLA.
3. A whole decode step of a small GQA model (embedding, layers of RMSNorm + QKV projection, RoPE, cache
   append, attention, output projection, SwiGLU FFN; final norm and logits), greedy, steps chained with
   the caches donated: autokernel's kernels against the same model in plain JAX with XLA's attention,
   paged_attention or ragged_paged_attention. Tokens per second.

    uv run python benchmarks/decode.py
    uv run python benchmarks/decode.py --gqa 32,2048 --mla 32,2048,64 --model 4,2048,16,8,6144,32768:32,1024
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from decode import (
    gqa_decode,
    jax_paged_attention,
    jax_ragged_paged_attention,
    mla_decode,
    paged_gqa_decode,
    xla_gqa_decode,
    xla_mla_decode,
)
from decode_model import Config, baseline_step, decode_step, init_params, random_caches
from kv_cache import jax_paged_layout, pad_lanes, page_tables, ragged_kv_pages, to_pages
from timing import require_tpu, time_ms

PEAK_GBS = 819.0  # v5e HBM bandwidth
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
PAGE = 128
GQA = ["8,1024", "32,1024", "64,1024", "16,2048", "64,2048", "32,4096", "8,8192"]  # batch, mean context
MLA = ["8,2048,16", "32,2048,16", "32,2048,64", "64,1024,128", "16,4096,128"]  # batch, mean context, heads
MODELS = [  # layers, d_model, heads, kv_heads, ffn, vocab : batch, mean context
    "4,2048,16,8,6144,32768:32,1024",
    "4,2048,16,8,6144,32768:8,4096",
    "4,1024,16,8,3072,32768:64,512",
]
SMALL = (["2,256"], ["2,256,16"], ["2,256,4,2,512,1024:2,128"])  # --small: a quick run, anywhere


def ragged_lengths(batch: int, mean: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).integers(mean // 2, 3 * mean // 2, batch), jnp.int32)


def cache_len(mean: int, extra: int = 0, multiple: int = 1024) -> int:
    """S_max: room for the longest sequence (and `extra` more tokens), rounded up."""
    return -(-(3 * mean // 2 + extra) // multiple) * multiple


def max_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def best_of(candidates: dict[str, Callable[[], float]]) -> tuple[float, str]:
    """The fastest of some named timings (those that fail to compile or run are skipped)."""
    best, name = float("inf"), "-"
    for label, timing in candidates.items():
        try:
            t = timing()
        except Exception as e:  # a block size the kernel rejects
            print(f"    ({label}: {type(e).__name__}: {str(e).splitlines()[0][:100]})")
            continue
        if t < best:
            best, name = t, label
    return best, name


def line(label: str, ms: float, gbs: float, extra: str = "") -> None:
    print(f"  {label:<44} {ms:8.3f} ms  {gbs:6.0f} GB/s ({gbs / PEAK_GBS:4.0%}) {extra}")


def bench_gqa(shape: str, on_tpu: bool) -> tuple:
    batch, mean = map(int, shape.split(","))
    s_max = cache_len(mean)
    lengths = ragged_lengths(batch, mean)
    keys = jax.random.split(jax.random.key(0), 3)
    q = (jax.random.normal(keys[0], (batch, 1, HEADS, HEAD_DIM)) * HEAD_DIM**-0.5).astype(jnp.bfloat16)
    k, v = (jax.random.normal(key, (batch, s_max, KV_HEADS * HEAD_DIM), jnp.bfloat16) for key in keys[1:])
    tables = jnp.asarray(page_tables(batch, s_max, PAGE))
    k_pages, v_pages = to_pages(k, tables), to_pages(v, tables)
    gbytes = float(jnp.sum(lengths)) * KV_HEADS * HEAD_DIM * 2 * 2 / 1e9
    gbs = lambda ms: gbytes / ms * 1e3  # noqa: E731
    print(f"GQA decode B={batch} mean context {mean} (S_max {s_max}, {gbytes * 1e3:.0f} MB of valid K/V)")

    want = jax.jit(xla_gqa_decode)(q, k, v, lengths)
    gqa_decode(q, k, v, lengths)  # eagerly: tunes on these lengths
    paged_gqa_decode(q, k_pages, v_pages, tables, lengths)
    t_ours = time_ms(jax.jit(gqa_decode), q, k, v, lengths)
    t_paged = time_ms(jax.jit(paged_gqa_decode), q, k_pages, v_pages, tables, lengths)
    t_xla = time_ms(jax.jit(xla_gqa_decode), q, k, v, lengths)
    diff = max(
        max_diff(gqa_decode(q, k, v, lengths), want),
        max_diff(paged_gqa_decode(q, k_pages, v_pages, tables, lengths), want),
    )
    line("autokernel gqa_decode (contiguous)", t_ours, gbs(t_ours), str(gqa_decode.tuning(q, k, v, lengths).config))
    line("autokernel paged_gqa_decode (pages of 128)", t_paged, gbs(t_paged))
    line("XLA (dense over S_max, masked)", t_xla, gbs(t_xla))
    t_jax_paged = t_ragged = float("inf")
    if on_tpu:
        jk, jv = jax_paged_layout(k_pages, KV_HEADS), jax_paged_layout(v_pages, KV_HEADS)
        per_seq = s_max // PAGE
        t_jax_paged, blocks = best_of(
            {
                f"{n} pages/block": lambda n=n: time_ms(
                    jax.jit(jax_paged_attention, static_argnums=5), q, jk, jv, tables, lengths, n
                )
                for n in (1, 2, 4, 8, 16)
                if per_seq % n == 0
            }
        )
        line(f"JAX paged_attention ({blocks})", t_jax_paged, gbs(t_jax_paged))
        diff = max(diff, max_diff(jax_paged_attention(q, jk, jv, tables, lengths, 4), want))
        del jk, jv
        kv = ragged_kv_pages(k_pages, v_pages, KV_HEADS)

        def time_ragged(**sizes: int) -> float:
            # (Its block sizes come from its tuned table unless both are given.)
            return time_ms(jax.jit(lambda *a: jax_ragged_paged_attention(*a, **sizes)), q, kv, tables, lengths)

        candidates = {"its tuned table's blocks": time_ragged}
        for pages_per_block in (2, 4, 8, 16):
            for queries in (8, 32):
                sizes = {"num_kv_pages_per_block": pages_per_block, "num_queries_per_block": queries}
                candidates[f"{pages_per_block} pages x {queries} queries"] = lambda s=sizes: time_ragged(**s)
        t_ragged, blocks = best_of(candidates)
        line(f"JAX ragged_paged_attention ({blocks})", t_ragged, gbs(t_ragged))
        diff = max(diff, max_diff(jax_ragged_paged_attention(q, kv, tables, lengths), want))
    best_base = min(t_xla, t_jax_paged, t_ragged)
    ours = min(t_ours, t_paged)
    print(f"  speedup over the best baseline {best_base / ours:.2f}x; max |out - XLA| {diff:.4f}")
    return (batch, mean, t_ours, t_paged, t_xla, t_jax_paged, t_ragged, gbs(ours), diff)


def bench_mla(shape: str) -> tuple:
    batch, mean, heads = map(int, shape.split(","))
    s_max = cache_len(mean)
    lengths = ragged_lengths(batch, mean)
    keys = jax.random.split(jax.random.key(1), 4)
    scale = 192**-0.5
    q_c = (jax.random.normal(keys[0], (batch, 1, heads, 512)) * scale * 0.1).astype(jnp.bfloat16)
    q_r = pad_lanes((jax.random.normal(keys[1], (batch, 1, heads, 64)) * scale).astype(jnp.bfloat16))
    c_kv = jax.random.normal(keys[2], (batch, s_max, 512), jnp.bfloat16)
    k_rope = pad_lanes(jax.random.normal(keys[3], (batch, s_max, 64), jnp.bfloat16))
    args = (q_c, q_r, c_kv, k_rope, lengths)
    tokens = float(jnp.sum(lengths))
    gbytes = tokens * 576 * 2 / 1e9  # what the math reads: 512 latents + 64 RoPE dims per position
    tflops = tokens * heads * (2 * 576 + 2 * 512) / 1e12
    print(f"MLA decode B={batch} mean context {mean} H={heads} (S_max {s_max}, {gbytes * 1e3:.0f} MB of valid cache)")
    mla_decode(*args)  # eagerly: tunes on these lengths
    t_ours = time_ms(jax.jit(mla_decode), *args)
    t_xla = time_ms(jax.jit(xla_mla_decode), *args)
    diff = max_diff(mla_decode(*args), jax.jit(xla_mla_decode)(*args))
    for label, t in (("autokernel mla_decode", t_ours), ("XLA (dense over S_max, masked)", t_xla)):
        line(label, t, gbytes / t * 1e3, f"{tflops / t * 1e3:6.1f} TFLOP/s")
    print(f"  {mla_decode.tuning(*args).config}; speedup {t_xla / t_ours:.2f}x; max |out - XLA| {diff:.4f}")
    return (batch, mean, heads, t_ours, t_xla, gbytes / t_ours * 1e3, diff)


def time_steps(step: Callable, tokens: jax.Array, caches: Any, lengths: jax.Array, params: Any, tables: Any, n: int):
    """Wall-clock ms per greedy decode step over n chained steps (caches donated), after 2 warm-up steps;
    and the first step's logits."""
    first = None
    for i in range(n + 2):
        if i == 2:
            jax.block_until_ready(tokens)
            start = time.perf_counter()
        logits, caches, lengths = step(tokens, caches, lengths, params, tables)
        first = logits if first is None else first
        tokens = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)[:, None]
    jax.block_until_ready(tokens)
    return (time.perf_counter() - start) / n * 1e3, first


def bench_model(spec: str, on_tpu: bool, steps: int) -> tuple:
    model, run = spec.split(":")
    config = Config(*map(int, model.split(",")))
    batch, mean = map(int, run.split(","))
    s_max = cache_len(mean, steps + 2)
    params = init_params(config)
    lengths = ragged_lengths(batch, mean)
    source = random_caches(config, np.asarray(lengths), s_max)
    tables = jnp.asarray(page_tables(batch, s_max, PAGE))
    tokens = jax.random.randint(jax.random.key(2), (batch, 1), 0, config.vocab)
    weights = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(params["layers"])) + params["w_vocab"].nbytes
    kv = config.layers * float(jnp.sum(lengths) + batch * (steps / 2 + 1)) * config.kv_heads * HEAD_DIM * 2 * 2
    gbs = lambda ms: (weights + kv) / ms / 1e6  # noqa: E731
    print(f"decode step {config}, batch {batch}, mean context {mean} (S_max {s_max}): ms per step, tokens/s")
    print(f"  per step: {weights / 1e6:.0f} MB of weights, ~{kv / 1e6:.0f} MB of valid K/V")

    def copy():
        return jax.tree.map(jnp.copy, source)

    def pages(convert=lambda x: x):
        return [{n: convert(to_pages(c[n], tables)) for n in c} for c in source]

    results = {}
    want = None
    # Ours: a first step eagerly, so the attention kernels tune on the real lengths (under jit they'd tune
    # on random ones); then chained jitted steps, the caches donated.
    for label, tbl in (("autokernel (contiguous cache)", None), ("autokernel (paged cache)", tables)):
        caches = copy() if tbl is None else pages()
        decode_step(tokens, caches, lengths, params, tbl)
        t, logits = time_steps(jax.jit(decode_step, donate_argnums=1), tokens, caches, lengths, params, tbl, steps)
        want = logits if want is None else want
        results[label] = (t, max_diff(logits, want))
        del caches
    baselines = [("plain JAX + XLA attention", "xla", copy)]
    if on_tpu:

        def jax_pages():
            return pages(lambda x: jax_paged_layout(x, config.kv_heads))

        def ragged_pages():
            return [{"kv": ragged_kv_pages(c["k"], c["v"], config.kv_heads)} for c in pages()]

        baselines += [("plain JAX + paged_attention", "paged", jax_pages)]
        baselines += [("plain JAX + ragged_paged_attention", "ragged", ragged_pages)]
    for label, attention, make in baselines:
        options = [{}] if attention != "paged" else [{"pages_per_compute_block": n} for n in (4, 8)]
        for opts in options:
            try:
                t, logits = time_steps(baseline_step(attention, **opts), tokens, make(), lengths, params, tables, steps)
            except Exception as e:
                print(f"    ({label} {opts}: {type(e).__name__}: {str(e).splitlines()[0][:100]})")
                continue
            name = f"{label} {opts}" if opts else label
            results[name] = (t, max_diff(logits, want))
    for name, (t, diff) in results.items():
        line(name, t, gbs(t), f"{batch / t * 1e3:9.0f} tokens/s  max |logits - ours| {diff:.3f}")
    ours = min(t for name, (t, _) in results.items() if name.startswith("autokernel"))
    base = min(t for name, (t, _) in results.items() if not name.startswith("autokernel"))
    print(f"  speedup over the best plain-JAX step {base / ours:.2f}x")
    return (model, batch, mean, base, ours, batch / ours * 1e3, gbs(ours))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa", action="append", help="batch,mean_context (repeatable)")
    parser.add_argument("--mla", action="append", help="batch,mean_context,heads (repeatable)")
    parser.add_argument("--model", action="append", help="layers,d_model,heads,kv_heads,ffn,vocab:batch,mean_context")
    parser.add_argument("--steps", type=int, default=32, help="decode steps timed per model")
    parser.add_argument("--small", action="store_true", help=argparse.SUPPRESS)  # tiny shapes, runs anywhere
    args = parser.parse_args()
    on_tpu = jax.default_backend() == "tpu"
    if not args.small:
        require_tpu()
    picked = args.gqa or args.mla or args.model
    gqa = args.gqa or ([] if picked else SMALL[0] if args.small else GQA)
    mla = args.mla or ([] if picked else SMALL[1] if args.small else MLA)
    models = args.model or ([] if picked else SMALL[2] if args.small else MODELS)
    steps = 2 if args.small else args.steps
    rows_gqa = [bench_gqa(shape, on_tpu) for shape in gqa]
    rows_mla = [bench_mla(shape) for shape in mla]
    rows_model = [bench_model(spec, on_tpu, steps) for spec in models]

    if rows_gqa:
        print("\nGQA decode attention (H 32, Hkv 8, Dh 128), ms")
        print("B   mean  ours   ours paged  XLA     paged_attn  ragged_paged  speedup  GB/s  max diff")
        for b, mean, t, tp, tx, tj, tr, rate, diff in rows_gqa:
            best = min(tx, tj, tr)
            print(
                f"{b:<3} {mean:<5} {t:6.3f} {tp:10.3f} {tx:7.3f} {tj:11.3f} {tr:13.3f}"
                f" {best / min(t, tp):7.2f}x {rate:5.0f} {diff:9.4f}"
            )
    if rows_mla:
        print("\nAbsorbed MLA decode attention (latent 512 + RoPE 64), ms")
        print("B   mean  H    ours    XLA     speedup  GB/s  max diff")
        for b, mean, h, t, tx, rate, diff in rows_mla:
            print(f"{b:<3} {mean:<5} {h:<4} {t:7.3f} {tx:7.3f} {tx / t:7.2f}x {rate:5.0f} {diff:9.4f}")
    if rows_model:
        print("\nDecode step, ms per step (best plain JAX baseline vs autokernel)")
        print("model (layers,d,H,Hkv,ffn,vocab)  B   mean  plain JAX  autokernel  speedup  tokens/s  GB/s")
        for model, b, mean, base, ours, rate, bw in rows_model:
            print(f"{model:<33} {b:<3} {mean:<5} {base:9.3f} {ours:11.3f} {base / ours:7.2f}x {rate:9.0f} {bw:5.0f}")


if __name__ == "__main__":
    main()
