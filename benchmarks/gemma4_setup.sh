#!/usr/bin/env bash
# On the TPU VM: what benchmarks/gemma4*.py need besides JAX, and Gemma 4 E2B's checkpoint (text weights are
# read from model.safetensors; the tokenizer only decodes text for show).
set -euo pipefail
uv pip install --system -q huggingface_hub tokenizers ml_dtypes
python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("google/gemma-4-E2B", local_dir="/content/gemma-4-E2B",
                        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]))
PY
ls -la /content/gemma-4-E2B
