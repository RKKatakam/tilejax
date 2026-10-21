# Ragged paged attention: autokernel vs vLLM TorchTPU

`vllm_torchtpu/` is vendored, unmodified, from [vllm-torchtpu](https://github.com/vllm-project/tpu-inference)
v0.29.0 (Apache-2.0, `LICENSE.vllm-torchtpu`; the copyright headers are the files' own):

- `kernels/ragged_paged_attention/v3/kernel.py`: RPA v3, `ragged_paged_attention`;
- `kernels/ragged_paged_attention/v3/util.py`: its helpers.

`envs.py` is a stub for the one setting the kernel reads, and the `__init__.py` files are package markers.

- `run.py`: the benchmark: vLLM's kernel as its runner calls it (`vllm`, the MIXED case for prefills), with
  its PREFILL case (`vllm-prefill`), and autokernel's `examples/ragged_paged_attention.py`;
- `plot.py`: the README's chart, from `benchmarks/results/rpa_vs_vllm_v5e.txt`;
- `debug_copies.py`: the copies XLA adds around each step (compiled HLO);
- `probe_layout.py`, `probe_page.py`, `probe_prefill.py`, `probe_prefill_variants.py`: probes behind the
  kernels' design.
