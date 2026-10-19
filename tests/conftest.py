import pytest

import autokernel.codegen


@pytest.fixture(params=autokernel.codegen.PIPELINES)
def pipeline(request):
    """Run a test against every backend: "manual" (no grid) and "grid"."""
    return request.param
