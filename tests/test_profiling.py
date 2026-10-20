"""Profiling from trace regions, and buffering chosen from profiles.

The trace reading and the choices are tested on synthetic device traces; `test_tpu.py` runs them on a
TPU v5e. Off a TPU the drivers run the kernel in the interpreter under the JAX profiler, with the device
trace replaced by synthetic events."""

import os

import jax
import pytest
from helpers import rand
from matmul import make_matmul

import autokernel
from autokernel import KernelError, Pipeline, profiling
from autokernel.profiling import Event, candidates, summarize

US = 1e-6


def trace(name="matmul", calls=2, steps=4, wait=1.0, first_wait=5.0, out_wait=0.0, compute=10.0, scopes=()):
    """Device events for `calls` calls of a kernel: per step a wait, an output wait and compute regions (us),
    and inside compute, each named scope for a microsecond."""
    events, t = [], 0.0
    for _ in range(calls):
        start = t
        t += 2
        for step in range(steps):
            w = first_wait if step == 0 else wait
            events.append(Event("XLA TraceMe", f"{name}/wait", t * US, (t + w) * US))
            t += w
            events.append(Event("XLA TraceMe", f"{name}/out_wait", t * US, (t + out_wait) * US))
            t += out_wait
            events.append(Event("XLA TraceMe", f"{name}/compute", t * US, (t + compute) * US))
            events.append(Event("XLA TraceMe", "someone else's region", t * US, t * US))
            for i, scope in enumerate(scopes):
                events.append(Event("XLA TraceMe", f"outer/{scope}", (t + i) * US, (t + i + 1) * US))
            t += compute
        t += 3
        events.append(Event("XLA Ops", f"{name}.1", start * US, t * US))
        events.append(Event("XLA Ops", "fusion.2", t * US, (t + 50) * US))  # another op: not this kernel
        events.append(Event("XLA TraceMe", "unrelated", t * US, (t + 1) * US))
        t += 60
    return events


def test_summarize_a_trace():
    p = summarize(trace(scopes=("mxu",)), "matmul", Pipeline(), scopes=("mxu", "absent"))
    assert p.calls == 2 and p.regions and p.steps == 4
    assert p.device_s == pytest.approx((2 + 5 + 3 * 1 + 4 * 10 + 3) * US)
    assert p.wait_s == pytest.approx(8 * US) and p.fill_s == pytest.approx(5 * US)
    assert p.exposed_wait_s == pytest.approx(3 * US) and p.compute_s == pytest.approx(40 * US)
    assert p.other_s == pytest.approx(5 * US) and p.bottleneck == "compute"
    assert p.scopes == {"mxu": pytest.approx(4 * US), "absent": 0.0}
    text = str(p)
    assert "steps                   4" in text and "scope 'mxu'" in text and "bottleneck: compute" in text


def test_bottlenecks():
    assert summarize(trace(wait=10.0), "matmul", Pipeline()).bottleneck == "dma"
    assert summarize(trace(compute=0.5, first_wait=0.5, wait=0.1), "matmul", Pipeline()).bottleneck == "overhead"
    without_regions = [e for e in trace() if e.line == "XLA Ops"]
    p = summarize(without_regions, "matmul", Pipeline())
    assert not p.regions and p.bottleneck is None and "no trace regions" in str(p)


def test_a_trace_without_the_kernel():
    with pytest.raises(KernelError, match="no call of kernel 'other'"):
        summarize(trace(), "other", Pipeline())


def test_kernel_ops_named_after_their_outputs_match():
    events = [Event("XLA Ops", "matmul.3.out_hbm", 0.0, 1.0), Event("XLA Ops", "matmul_kernel.1", 0.0, 9.0)]
    assert summarize(events, "matmul", Pipeline()).device_s == 1.0


def profile_with(**regions):
    return summarize(trace(**regions), "matmul", Pipeline())


def test_candidates():
    fits = lambda p: True  # noqa: E731
    assert candidates(profile_with(wait=10.0), fits) == [Pipeline(4, 2)]
    assert candidates(profile_with(out_wait=5.0, wait=0.0), fits) == [Pipeline(3, 3)]
    both = profile_with(wait=10.0, out_wait=5.0)
    assert candidates(both, fits) == [Pipeline(4, 2), Pipeline(3, 3)]
    assert candidates(both, lambda p: p.output_buffers == 3) == [Pipeline(3, 3)]
    assert candidates(profile_with(wait=0.0), fits) == []
    both.pipeline = Pipeline(4, 3)
    assert candidates(both, fits) == []
    both.regions = False
    assert candidates(both, fits) == []


def test_optimization_report():
    first, second = profile_with(wait=10.0), profile_with()
    second.pipeline = Pipeline(4, 2)
    result = autokernel.Optimization(second, [first, second])
    assert result.pipeline == Pipeline(4, 2)
    assert str(result).splitlines()[0] == "matmul: chose 4 input / 2 output buffers"


def test_enable_profiling(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--other")
    monkeypatch.setattr(profiling, "_backend_started", lambda: False)
    autokernel.enable_profiling()
    assert os.environ["LIBTPU_INIT_ARGS"] == f"--other {profiling.REGION_FLAG}" and profiling.regions_enabled()
    autokernel.enable_profiling()  # already on
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    monkeypatch.setattr(profiling, "_backend_started", lambda: True)
    with pytest.raises(KernelError, match="must run before JAX starts its backend"):
        autokernel.enable_profiling()


def test_the_backend_has_started_in_tests():
    jax.devices()
    assert profiling._backend_started()


def test_device_events_read_a_profiler_trace(tmp_path):
    jax.profiler.start_trace(str(tmp_path))
    jax.block_until_ready(jax.numpy.ones(8) * 2)
    jax.profiler.stop_trace()
    events = profiling.device_events(str(tmp_path), plane="/host:CPU")
    assert events and all(isinstance(e, Event) for e in events)
    assert profiling.device_events(str(tmp_path)) == [] or jax.default_backend() == "tpu"
    with pytest.raises(KernelError, match="wrote no trace"):
        profiling.device_events(str(tmp_path / "empty"))


@pytest.fixture
def fake_tpu(monkeypatch):
    """Profile off a TPU: kernels run in the interpreter, and each trace read returns the next of `traces`."""
    traces: list = []
    monkeypatch.setattr(profiling, "_require_tpu", lambda: None)
    monkeypatch.setattr(profiling, "device_events", lambda logdir: traces.pop(0))
    monkeypatch.setenv("LIBTPU_INIT_ARGS", profiling.REGION_FLAG)
    return traces


@pytest.mark.cpu
def test_profile(fake_tpu):
    kernel = make_matmul(128, 128, 128)
    x, y = rand((256, 256)), rand((256, 256), 1)
    fake_tpu.append(trace())
    p = autokernel.profile(kernel, x, y, runs=1)
    assert p.regions and p.logdir and p.pipeline == Pipeline(3, 2) and f"trace: {p.logdir}" in str(p)


@pytest.mark.cpu
def test_optimize_keeps_what_helps(fake_tpu):
    kernel = make_matmul(128, 128, 128)
    x, y = rand((256, 256)), rand((256, 256), 1)
    fake_tpu += [trace(wait=10.0), trace(wait=10.0, compute=5.0), trace(wait=10.0, compute=4.9)]
    result = autokernel.optimize(kernel, x, y, runs=1)
    # 3 -> 4 input buffers helped; 4 is the most there are, so it stops there.
    assert [p.pipeline for p in result.tried] == [Pipeline(3, 2), Pipeline(4, 2)]
    assert result.pipeline == Pipeline(4, 2) and kernel.lower(x, y).pipeline == Pipeline(4, 2)
    assert len(fake_tpu) == 1


@pytest.mark.cpu
def test_optimize_tries_every_suggestion_then_stops(fake_tpu):
    kernel = make_matmul(128, 128, 128)
    x, y = rand((256, 256)), rand((256, 256), 1)
    fake_tpu += [trace(wait=10.0, out_wait=5.0)] * 3
    result = autokernel.optimize(kernel, x, y, runs=1)
    assert [p.pipeline for p in result.tried] == [Pipeline(3, 2), Pipeline(4, 2), Pipeline(3, 3)]
    assert result.pipeline == Pipeline(3, 2) and not fake_tpu


def test_profiling_needs_a_tpu_and_regions(monkeypatch):
    kernel = make_matmul(128, 128, 128)
    x, y = rand((256, 256)), rand((256, 256), 1)
    monkeypatch.setattr(profiling.v5e, "on_tpu", lambda: False)
    with pytest.raises(KernelError, match="needs a TPU v5e"):
        autokernel.profile(kernel, x, y)
    monkeypatch.setattr(profiling.v5e, "on_tpu", lambda: True)
    profiling._require_tpu()
    monkeypatch.setattr(profiling, "_require_tpu", lambda: None)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    with pytest.raises(KernelError, match="reads trace regions"):
        autokernel.optimize(kernel, x, y)
