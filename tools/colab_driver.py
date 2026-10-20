"""Runs on a Colab TPU v5e VM (through `colab exec`): unpack the repo, install JAX for TPUs, run commands.

Everything runs in subprocesses: importing JAX in the Colab kernel itself would hold the TPU for every
later process. The commands come from /content/ak/commands.txt, one per line, run from the repo root. Their
output streams as it comes; a command still running after `TIMEOUT` seconds is killed (a hung kernel
shouldn't cost the rest of the run).
"""

import subprocess
import time

REPO = "/content/ak"
TIMEOUT = 3600


def sh(command: str, timeout: int = TIMEOUT) -> None:
    print(f"$ {command}", flush=True)
    start = time.monotonic()
    process = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if time.monotonic() - start > timeout:
            process.kill()
            print(f"[killed after {timeout} s]", flush=True)
            break
    print(f"[exit {process.wait()}]", flush=True)


sh(
    f"mkdir -p {REPO} && tar xzf /content/ak.tar.gz -C {REPO} 2>/dev/null && "
    "uv pip install --system -q 'jax[tpu]==0.11.2' 'numpy>=2.5.4' 'pytest>=9' pytest-cov"
)
with open(f"{REPO}/commands.txt") as f:
    for line in f.read().splitlines():
        if line.strip():
            sh(f"cd {REPO} && PYTHONPATH={REPO}/src PYTHONUNBUFFERED=1 {line}")
