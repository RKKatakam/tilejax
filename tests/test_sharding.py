"""Kernels and programs on inputs sharded over a mesh of devices: run per shard under `jax.shard_map`.

These tests need several devices, and JAX fixes its device count when it first initialises, so the cases
(`sharding_cases.py`) run with pytest in a fresh Python process with four CPU devices
(`--xla_force_host_platform_device_count=4`), where the kernels run in the Pallas TPU interpreter.
"""

import os
import subprocess
import sys


def test_sharded_inputs_on_four_devices():
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    here = os.path.dirname(os.path.abspath(__file__))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(here, "sharding_cases.py")],
        env=env,
        cwd=os.path.dirname(here),
        capture_output=True,
        text=True,
        timeout=900,
    )
    summary = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    assert run.returncode == 0 and " passed" in summary and "skipped" not in summary, (
        run.stdout[-6000:] + run.stderr[-3000:]
    )
