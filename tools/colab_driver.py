"""Runs on a Colab TPU v5e VM (through `colab exec`): unpack the repo, install JAX for TPUs, run commands.

Everything runs in subprocesses: importing JAX in the Colab kernel itself would hold the TPU for every
later process. The commands come from /content/ak/commands.txt, one per line, run from the repo root.
"""

import subprocess

REPO = "/content/ak"


def sh(command: str, timeout: int = 5400) -> None:
    print(f"$ {command}", flush=True)
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=timeout)
    print(result.stdout[-30000:], result.stderr[-8000:], f"[exit {result.returncode}]", flush=True)


sh(
    f"mkdir -p {REPO} && tar xzf /content/ak.tar.gz -C {REPO} 2>/dev/null && "
    "uv pip install --system -q 'jax[tpu]==0.11.2' 'numpy>=2.5.4' 'pytest>=9' pytest-cov"
)
with open(f"{REPO}/commands.txt") as f:
    for line in f.read().splitlines():
        if line.strip():
            sh(f"cd {REPO} && PYTHONPATH={REPO}/src {line}")
