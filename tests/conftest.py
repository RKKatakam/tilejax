"""Test-wide setup: trace regions on (a TPU records them only if told before JAX starts), 8 simulated
chips off a TPU, the examples importable, and `tpu` tests run only on a TPU v5e."""

import os
import sys
from pathlib import Path

REGION_FLAG = "--xla_enable_custom_call_region_trace=true"  # autokernel.profiling.REGION_FLAG
os.environ["LIBTPU_INIT_ARGS"] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} {REGION_FLAG}".strip()
# Off a TPU, 8 CPU devices: kernels gathering across chips run on a simulated mesh of them.
CHIPS_FLAG = "--xla_force_host_platform_device_count=8"
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {CHIPS_FLAG}".strip()

import jax  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

ON_TPU = jax.default_backend() == "tpu"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "tpu" in item.keywords and not ON_TPU:
            item.add_marker(pytest.mark.skip(reason="needs a TPU v5e"))
        if "cpu" in item.keywords and ON_TPU:
            item.add_marker(pytest.mark.skip(reason="relies on the interpreter or a simulated TPU"))
