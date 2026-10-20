#!/usr/bin/env bash
# Run commands on a fresh Colab TPU v5e session; the output goes to stdout and tpu-run.txt.
#
#   tools/colab_tpu.sh                                   # the TPU tests and the benchmarks
#   tools/colab_tpu.sh "python -m pytest tests -q" ...   # any commands, run from the repo root on the VM
set -euo pipefail
cd "$(dirname "$0")/.."
COLAB=${COLAB:-$HOME/.local/bin/colab}
SESSION=${SESSION:-ak}

commands=("$@")
if [ ${#commands[@]} -eq 0 ]; then
  commands=(
    "python -m pytest tests -q -p no:cacheprovider"
    "python benchmarks/matmul.py"
    "python benchmarks/attention.py"
    "python benchmarks/profile.py"
  )
fi

work=$(mktemp -d)
printf '%s\n' "${commands[@]}" > "$work/commands.txt"
COPYFILE_DISABLE=1 tar czf "$work/ak.tar.gz" src tests examples benchmarks pyproject.toml -C "$work" commands.txt

"$COLAB" new -s "$SESSION" --tpu v5e1
trap '"$COLAB" stop -s "$SESSION" || true' EXIT
"$COLAB" upload -s "$SESSION" "$work/ak.tar.gz" /content/ak.tar.gz
"$COLAB" exec -s "$SESSION" -f tools/colab_driver.py --timeout ${TIMEOUT:-14400} | tee tpu-run.txt
