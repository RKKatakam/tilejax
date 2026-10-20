#!/usr/bin/env bash
# On the TPU VM: vLLM's TPU build in a venv of its own (it pins its own JAX), and Gemma 4 E2B's checkpoint.
set -euo pipefail
uv venv -q /content/vllm --python 3.12 --python-preference only-managed
/content/vllm/bin/python --version
uv pip install -q --python /content/vllm/bin/python vllm-tpu huggingface_hub
/content/vllm/bin/python -c "import vllm, jax; print('vllm', vllm.__version__, 'jax', jax.__version__, jax.devices())"
/content/vllm/bin/python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("google/gemma-4-E2B", local_dir="/content/gemma-4-E2B"))
PY
ls -la /content/gemma-4-E2B
