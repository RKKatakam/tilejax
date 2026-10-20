"""Profiling: kernels built with trace regions, and reading a profiler trace into per-kernel breakdowns.

Reading real traces needs a TPU (`tpu_profile_check.py` runs that on a v5e); here the regions' code is checked
in the interpreter and the trace reader on synthetic events.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import AutoKernelError, hardware, lowering, profiling

T = 128


def matmul(x, y):
    out = jnp.zeros((512, 256))
    for i in range(0, 512, T):
        for j in range(0, 256, T):
            acc = jnp.zeros((T, T))
            for k in range(0, 512, T):
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc)
    return out


def two_nests(x):
    h = jnp.zeros((512, 128))
    for i in range(0, 512, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 2)
    out = jnp.zeros((512, 128))
    for i in range(0, 512, T):
        out = out.at[i : i + T, :].set(h[i : i + T, :] + 1)
    return out


def traced(lowered, options):
    return lowering.build(dataclasses.replace(lowered.plan, trace=True), options)


def test_trace_regions_mark_waits_and_compute():
    kern = autokernel.tpu(matmul)
    x, y = rand((512, 512)), rand((512, 256), 1)
    lw = kern.lower(x, y)
    assert "named_scope" not in lw.source  # off by default: the kernel is unchanged
    t = traced(lw, kern.options)
    for what in ("wait", "compute", "out_wait"):
        assert f"jax.named_scope('matmul/{what}')" in t.source
    got = t.entry(x, y, interpret=pltpu.InterpretParams())
    np.testing.assert_allclose(got, kern(x, y), rtol=1e-6)


def test_options_trace_flows_into_programs_and_fused_kernels():
    prog = autokernel.TpuProgram(two_nests, options=lowering.Options(trace=True))
    x = rand((512, 128))
    lowered = prog.lower(x)
    ((name, fused),) = lowered.kernels
    assert "+" in name and "jax.named_scope('two_nests_h/compute')" in fused.source
    assert "jax.named_scope('two_nests_out/wait')" in fused.source
    np.testing.assert_allclose(prog(x), 2 * np.asarray(x) + 1, rtol=1e-6)


def test_kernel_names_for_the_trace():
    kern = autokernel.tpu(matmul)
    lw = kern.lower(rand((512, 512)), rand((512, 256), 1))
    assert profiling._kernel_names(lw) == ("matmul", ["matmul"], lw.estimate.seconds)
    ((_, fused),) = autokernel.tpu(two_nests).lower(rand((512, 128))).kernels
    name, parts, _ = profiling._kernel_names(fused)
    assert name == fused.name and parts == ["two_nests_h", "two_nests_out"]


def test_op_names_match_with_and_without_regions():
    p = profiling._op_pattern("matmul")
    assert p.match("matmul.1") and p.match("matmul") and p.match("%matmul.12 = f32[512,256] custom-call(...)")
    assert p.match("matmul.1.out_hbm")  # seen on a v5e: XLA appends the output's name
    assert not p.match("matmul_2.1") and not p.match("other")


def _fake_trace(monkeypatch, events):
    monkeypatch.setattr(profiling, "_events", lambda logdir: events)


def test_reading_a_trace(monkeypatch):
    us = 1e-6
    events = [
        ("XLA Ops", "matmul.1", 0.0, 10 * us),
        ("XLA TraceMe", "matmul/wait", 0.0, 2 * us),  # the fill
        ("XLA TraceMe", "matmul/compute", 2 * us, 5 * us),
        ("XLA TraceMe", "matmul/wait", 5 * us, 6 * us),  # exposed DMA
        ("XLA TraceMe", "matmul/compute", 6 * us, 9 * us),
        ("XLA TraceMe", "matmul/out_wait", 7 * us, 8 * us),  # inside compute
        ("XLA TraceMe", "unrelated/compute", 20 * us, 21 * us),
        ("XLA Ops", "fusion.3", 12 * us, 14 * us),  # an XLA op between kernels
        ("XLA TraceMe", "no_slash", 0.0, 1 * us),
    ]
    _fake_trace(monkeypatch, events)
    (k,), xla, top = profiling.read_trace("unused", [("matmul", ["matmul"], 9 * us)], runs=1)
    assert k.calls == 1 and k.regions and k.steps == 2
    assert k.device_s == pytest.approx(10 * us) and k.wait_s == pytest.approx(3 * us)
    assert k.fill_s == pytest.approx(2 * us) and k.out_wait_s == pytest.approx(1 * us)
    assert k.compute_s == pytest.approx(5 * us)  # 6 us of compute regions less the output wait inside
    assert k.exposed_dma_s == pytest.approx(2 * us) and k.other_s == pytest.approx(1 * us)
    assert k.overlap == pytest.approx(0.6) and k.bottleneck == "dma"
    assert xla == pytest.approx(2 * us) and top == [("fusion.3", pytest.approx(2 * us))]
    assert "matmul" in k.row() and "dma" in k.row()


def test_bottlenecks():
    base = dict(name="k", calls=1, device_s=10.0, regions=True)
    assert profiling.KernelProfile(**base, compute_s=9.0).bottleneck == "compute"
    assert profiling.KernelProfile(**base, compute_s=6.0).bottleneck == "overhead"
    assert profiling.KernelProfile(**base, compute_s=5.0, wait_s=3.0).bottleneck == "dma"
    blind = profiling.KernelProfile("k", 1, 10.0)
    assert blind.bottleneck is None and blind.overlap is None and "(no regions)" in blind.row()


def test_fused_kernels_report_their_parts(monkeypatch):
    events = [
        ("XLA Ops", "f.1", 0.0, 10.0),
        ("XLA TraceMe", "a/compute", 1.0, 3.0),
        ("XLA TraceMe", "b/wait", 3.0, 4.0),
        ("XLA TraceMe", "b/compute", 4.0, 8.0),
        ("XLA TraceMe", "b/bogus", 8.0, 9.0),
    ]
    _fake_trace(monkeypatch, events)
    (k,), _, _ = profiling.read_trace("unused", [("f", ["a", "b"], None)], runs=1)
    assert k.parts["a"]["compute"] == 2.0 and k.parts["b"]["wait"] == 1.0 and k.compute_s == 6.0
    assert k.fill_s == 0.0  # the first wait came after a compute region


def test_a_profile_report(monkeypatch):
    events = [("XLA Ops", "%k.1 = f32[] custom-call()", 0.0, 4e-6), ("XLA Ops", "%copy.2 = f32[8] copy()", 5e-6, 6e-6)]
    _fake_trace(monkeypatch, events)
    kernels, xla, top = profiling.read_trace("unused", [("k", ["k"], None), ("never_ran", ["never_ran"], None)], 2)
    report = profiling.Profile(kernels, xla, 2, "/tmp/somewhere", top)
    text = str(report)
    assert top == [("copy.2", pytest.approx(0.5e-6))] and "copy.2" in text
    assert "no trace regions" in text and "/tmp/somewhere" in text and report["k"].calls == 1
    assert report["never_ran"].calls == 0 and report["never_ran"].device_s == 0.0
    with pytest.raises(KeyError, match="profiled"):
        report["missing"]


def test_reading_a_directory_without_a_trace(tmp_path):
    with pytest.raises(AutoKernelError, match="wrote no trace"):
        profiling._events(str(tmp_path))


def test_reading_a_real_xspace(tmp_path):
    """The profiler's own files: device events on a TPU, none on the CPU backend (no TPU planes)."""
    jax.profiler.start_trace(str(tmp_path))
    (jnp.ones(8) * 2).block_until_ready()
    jax.profiler.stop_trace()
    events = profiling._events(str(tmp_path))
    assert bool(events) == hardware.on_tpu()


def test_profile_needs_a_tpu(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: False)
    with pytest.raises(AutoKernelError, match="needs a TPU"):
        autokernel.profile(autokernel.tpu(matmul), rand((512, 512)), rand((512, 256), 1))
    with pytest.raises(AutoKernelError, match="takes a kernel or program"):
        profiling._traced(lambda x: x)


def test_profile_runs_the_traced_kernel(monkeypatch, tmp_path):
    """On a (pretend) TPU: the traced twin runs, the trace is read for the kernel's name."""
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    monkeypatch.setattr(profiling, "regions_enabled", lambda: True)
    kern = autokernel.tpu(matmul, interpret=True, target="v5e")
    x, y = rand((512, 512)), rand((512, 256), 1)
    kern.lower(x, y)
    seen = []

    def run_traced(fn):
        _run, names = original(fn)

        def interpreted(*args):  # the twin is built for the chip: run its interpreter form instead
            ctx, inputs = fn._bind(args, {})
            lw = profiling_build(fn._lower(ctx, inputs))
            seen.append(lw)
            return lw.entry(*inputs, interpret=pltpu.InterpretParams())

        return interpreted, names

    original = profiling._traced
    profiling_build = lambda lw: traced(lw, kern.options)  # noqa: E731
    monkeypatch.setattr(profiling, "_traced", run_traced)
    monkeypatch.setattr(profiling, "_events", lambda logdir: [("XLA Ops", "matmul.1", 0.0, 1e-5)] * 3)
    report = autokernel.profile(kern, x, y, runs=3, logdir=str(tmp_path))
    assert seen and "named_scope" in seen[0].source
    assert report["matmul"].calls == 3 and report.logdir == str(tmp_path) and report.runs == 3


def test_the_traced_runner_of_a_kernel_and_a_program(monkeypatch):
    kern = autokernel.tpu(matmul, interpret=True)
    x, y = rand((512, 512)), rand((512, 256), 1)
    _run, names = profiling._traced(kern)
    assert names(x, y)[0][0] == "matmul"
    _run_p, names_p = profiling._traced(autokernel.tpu(two_nests, interpret=True))
    assert names_p(rand((512, 128)))[0][1] == ["two_nests_h", "two_nests_out"]


def test_enable_profiling(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--other=1")
    monkeypatch.setattr(profiling, "_backend_started", lambda: False)
    autokernel.enable_profiling()
    assert profiling.regions_enabled() and "--other=1" in profiling.os.environ["LIBTPU_INIT_ARGS"]
    autokernel.enable_profiling()  # already on: nothing to do
    assert profiling.os.environ["LIBTPU_INIT_ARGS"].count(profiling.REGION_FLAG) == 1
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    monkeypatch.setattr(profiling, "_backend_started", lambda: True)
    with pytest.raises(AutoKernelError, match="before JAX starts its backend"):
        autokernel.enable_profiling()


def test_backend_started_is_true_once_jax_ran():
    jnp.ones(1).block_until_ready()
    assert profiling._backend_started()


def test_profiling_warns_without_regions(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    monkeypatch.setattr(profiling, "regions_enabled", lambda: False)
    monkeypatch.setattr(profiling, "_traced", lambda fn: (lambda *a: jnp.ones(1), lambda *a: []))
    monkeypatch.setattr(profiling, "read_trace", lambda logdir, names, runs: ([], 0.0, []))
    with pytest.warns(UserWarning, match="trace regions are off"):
        report = autokernel.profile(object(), runs=1)
    assert report.kernels == []


@pytest.mark.skipif(not hardware.on_tpu(), reason="reads a real TPU trace")
def test_profiling_on_a_tpu():
    kern = autokernel.tpu(matmul)
    x, y = rand((512, 512)), rand((512, 256), 1)
    report = autokernel.profile(kern, x, y, runs=2)
    k = report["matmul"]
    assert k.calls == 2 and k.device_s > 0
    if profiling.regions_enabled():
        assert k.regions and k.steps == kern.lower(x, y).steps and k.compute_s > 0
    single = profiling.profile_lowered(kern.lower(x, y), (x, y), runs=2)
    assert single.calls == 2 and single.device_s > 0


# --------------------------------------------------------------------------- TPU-only paths, on the CPU


class _Interpreted:
    """`functools` for `profiling`, with `partial(entry, interpret=False)` running in the interpreter."""

    @staticmethod
    def partial(fn, **kwargs):
        if kwargs.get("interpret") is False:
            kwargs["interpret"] = pltpu.InterpretParams()
        import functools

        return functools.partial(fn, **kwargs)


def test_trace_planes_are_read(monkeypatch, tmp_path):
    (tmp_path / "x.xplane.pb").write_bytes(b"")
    ev = type("Ev", (), {"name": "k.1", "start_ns": 1000.0, "duration_ns": 500.0})
    line = type("Line", (), {"name": "XLA Ops", "events": [ev]})
    planes = [type("P", (), {"name": "/device:TPU:0", "lines": [line]}), type("P", (), {"name": "/host:CPU"})]
    monkeypatch.setattr(
        jax.profiler.ProfileData, "from_file", staticmethod(lambda path: type("PD", (), {"planes": planes}))
    )
    ((line_name, name, start, end),) = profiling._events(str(tmp_path))
    assert (line_name, name) == ("XLA Ops", "k.1") and start == pytest.approx(1e-6) and end == pytest.approx(1.5e-6)


def test_the_traced_runner_builds_once_per_kernel(monkeypatch):
    monkeypatch.setattr(profiling, "functools", _Interpreted)
    kern = autokernel.tpu(matmul, interpret=True)
    x, y = rand((512, 512)), rand((512, 256), 1)
    run, _ = profiling._traced(kern)
    np.testing.assert_allclose(run(x, y), kern(x, y), rtol=1e-6)
    np.testing.assert_allclose(run(x, y), kern(x, y), rtol=1e-6)  # the built twin is reused


def test_profile_lowered(monkeypatch):
    monkeypatch.setattr(profiling, "functools", _Interpreted)
    kern = autokernel.tpu(matmul, interpret=True)
    x, y = rand((512, 512)), rand((512, 256), 1)
    lw = kern.lower(x, y)
    events = [("XLA Ops", "matmul.1", 0.0, 2e-6), ("XLA TraceMe", "matmul/compute", 0.0, 1e-6)]
    monkeypatch.setattr(profiling, "_events", lambda logdir: events)
    plain = profiling.profile_lowered(lw, (x, y), runs=1)
    assert plain.device_s == pytest.approx(2e-6)
    traced = profiling.profile_lowered(lw, (x, y), runs=1, trace=True, options=kern.options)
    assert traced.regions and traced.compute_s == pytest.approx(1e-6)
    monkeypatch.setattr(profiling, "_events", lambda logdir: [])
    with pytest.raises(AutoKernelError, match="no call of kernel 'matmul'"):
        profiling.profile_lowered(lw, (x, y), runs=1)
