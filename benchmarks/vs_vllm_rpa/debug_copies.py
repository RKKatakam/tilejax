"""Debug: the copies XLA adds around autokernel's RPA step (compiled HLO), and the prefill kernel's profile.

python benchmarks/vs_vllm_rpa/debug_copies.py
"""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent.parent / "examples")]

import jax
from ragged_paged_attention import (
    HEAD_DIM,
    args_of,
    autokernel_rpa,
    make_inputs,
    prefill_kernel,
    prefill_tiles,
    tune,
    write_new_tokens,
)
from run import workloads

import autokernel

SCALE = HEAD_DIM**-0.5
for name in ("mixed32+2x512", "prefill4x1024"):
    q_lens, kv_lens = workloads()[name]
    inputs = make_inputs(q_lens, kv_lens, max_tokens=-(-sum(q_lens) // 16) * 16)
    tune(inputs, SCALE)  # (on these inputs: under jax.jit the kernels would tune on random ones)
    for donate in ((3,), (0, 1, 2, 3)):
        fn = jax.jit(lambda *a: autokernel_rpa(*a, sm_scale=SCALE), donate_argnums=donate)
        hlo = fn.lower(*args_of(inputs)).compile().as_text()
        copies = [line.strip()[:220] for line in hlo.splitlines() if " copy(" in line or "copy-start" in line]
        print(f"{name}, donating {donate}: {len(copies)} copies")
        for c in copies:
            print("   ", c)
    q, k, v, cache, kv, pages, cu, dist = args_of(inputs)
    cache = write_new_tokens(cache, k, v, kv, pages, cu, dist)
    tiles = prefill_tiles(kv, pages, cu, dist, max_tokens=q.shape[0])
    print(autokernel.profile(prefill_kernel(SCALE), q, cache, pages, *tiles.values()))
