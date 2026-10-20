# KDA decode: autokernel vs vLLM TorchTPU

`vllm_torchtpu/` is vendored, unmodified, from [vllm-torchtpu](https://github.com/vllm-project/tpu-inference)
v0.29.0 (Apache-2.0, `LICENSE.vllm-torchtpu`; the copyright headers are the files' own):

- `kernels/kimi_k3/decode_kda.py`: the fused single-token KDA decode kernel (`decode_kda`);
- `kernels/gdn/v3/compute_gdn.py`: its one helper, `fused_transpose_broadcast`;
- `kernels/gdn/v3/config.py`: imported by `compute_gdn.py`.

The `__init__.py` files are empty package markers (the originals import the rest of vLLM). It is plain
JAX/Pallas and runs on our JAX (0.11.2, the same as vllm-torchtpu's).

- `run.py`: the benchmark (vLLM's kernel, autokernel's `examples/kda_decode.py`, the plain-JAX step).
- `probe_math.py`, `probe_dma.py`, `probe_burst.py`: hand-written Pallas probes behind the kernel's design.
