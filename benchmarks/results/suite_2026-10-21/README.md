# Benchmark rerun, TPU v5e (Colab), 2026-10-21, commit 080fc71

Raw output of a partial rerun of the benchmark suite; the README's tables are not updated from it yet.
Each file's first line names its command and whether it finished.

- complete: matmul, attention, profile, attention_block, rmsnorm_matmul, qwen3, mla, KDA vs vLLM (both pool
  layouts).
- `vs_vllm_rpa_run_py_check.txt`: killed by the driver's 3600 s limit per command after the decode workloads
  and 7 of 8 prefills (autokernel's prefill16x256, the chunked and the mixed workloads are missing).
- `transformer_py.txt`: the VM was lost after the first model size (twice, in earlier attempts too).
- `linear_attention_py.txt`: stopped by hand during the Kimi Delta Attention kernel's tuning.
- not run yet: layer_stack, moe (all configs), moe_megakernel, moe_decode_megakernel, qwen3_models, deepseek,
  hybrid_models, the megakernel and DMA probes, Gemma 4 and vLLM. decode.py and decode_megakernel.py ran at
  this commit already (`../decode_v5e.txt`, `../decode_megakernel_v5e.txt`).
