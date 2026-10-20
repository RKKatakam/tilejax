"""The autotuner: the search, the proposals each bottleneck leads to, testing candidates, and tuning a
kernel end to end. Timings come from a synthetic `profiling.measure`: off a TPU, kernels run in the
interpreter (so testing candidates is real) but can't be timed."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import f64, make_kernel, rand

import autokernel
from autokernel import Config, KernelError, KernelProfile, Pipeline, Trial, profiling, tune
from autokernel.tune import _measure, _reference_config, proposals, random_inputs, search


@autokernel.tunable
class Tiles:
    tm: int = autokernel.choice(64, 128, 256, 512, default=128)
    tk: int = autokernel.choice(128, 256, 512, default=128)


SOURCE = """
def f(x, y, t=Tiles()):
    M, K = x.shape
    out = jnp.zeros((M, 128))
    for i in range(0, M, t.tm):
        acc = jnp.zeros((t.tm, 128))
        for k in range(0, K, t.tk):
            acc += x[i : i + t.tm, k : k + t.tk] @ y[k : k + t.tk, :]
        out = out.at[i : i + t.tm, :].set(acc)
    return out
"""


def kernel():
    return make_kernel(SOURCE, Tiles=Tiles)


X, Y = rand((256, 256)), rand((256, 128), 1)


def profile(config, seconds=10e-6, steps=4, wait=0.0, out_wait=0.0, compute=None, regions=True):
    """A KernelProfile with these times (`compute` defaults to all of the time not waiting)."""
    compute = seconds - wait - out_wait if compute is None else compute
    return KernelProfile("f", config, 1, seconds, regions, steps, wait, 0.0, out_wait, compute)


# --------------------------------------------------------------------------- search


def test_search_climbs_to_the_fastest_and_stops():
    times = {1: 9.0, 2: 7.0, 3: 8.0, 4: 6.0}

    def measure(config):
        return Trial(config, profile(config, seconds=times[config]))

    tuning = search(1, measure, lambda trial: [trial.config + 1, trial.config + 2] if trial.config < 4 else [], 10)
    assert tuning.config == 4 and [t.config for t in tuning.trials] == [1, 2, 3, 4]
    assert "chose 4" in str(tuning) and str(tuning).count("*") == 1


def test_search_skips_what_it_tried_ignores_failures_and_keeps_to_its_budget():
    def measure(config):
        if config == "broken":
            return Trial(config, error="doesn't build")
        return Trial(config, profile(config, seconds={"a": 5.0, "b": 4.0, "c": 3.0}[config]))

    proposals_ = {"a": ["broken", "a", "b"], "b": ["a", "c"], "c": []}
    tuning = search("a", measure, lambda t: proposals_[t.config], 3)
    assert [t.config for t in tuning.trials] == ["a", "broken", "b"] and tuning.config == "b"
    assert "doesn't build" in str(tuning)


def test_search_needs_a_working_start():
    with pytest.raises(KernelError, match=r"the default schedule \(start\) fails: no"):
        search("start", lambda c: Trial(c, error="no"), lambda t: [], 5)


def test_a_small_gain_isnt_a_gain():
    def measure(config):
        return Trial(config, profile(config, seconds=1.0 if config == "a" else 0.995))

    assert search("a", measure, lambda t: ["b"], 5).config == "a"


# --------------------------------------------------------------------------- proposals


def test_proposals_follow_the_bottleneck():
    k = kernel()
    sig = k.signature(X, Y)
    start = Config(Tiles(), Pipeline(3, 2))
    # Per-step overhead: bigger tiles only, fewest steps first. tm=512 doesn't divide the 256 rows.
    overhead = Trial(start, profile(start, seconds=10.0, compute=1.0))
    assert [c.tunable for c in proposals(k, sig, overhead)] == [Tiles(tm=256), Tiles(tk=256)]
    # Exposed copies: more input buffers first, then schedules that move fewer bytes.
    dma = Trial(start, profile(start, seconds=10.0, wait=5.0))
    assert proposals(k, sig, dma)[0] == Config(Tiles(), Pipeline(4, 2))
    # (a deeper tk moves as many bytes: only a taller tm reuses y)
    assert [c.tunable for c in proposals(k, sig, dma)[1:]] == [Tiles(tm=256)]
    # Exposed output copies: one more output buffer.
    outs = Trial(start, profile(start, seconds=10.0, out_wait=5.0))
    assert proposals(k, sig, outs)[0] == Config(Tiles(), Pipeline(3, 3))
    # Compute-bound, or no regions: every neighbour, fewest steps first.
    compute = Trial(start, profile(start, seconds=10.0))
    assert [c.tunable for c in proposals(k, sig, compute)] == [Tiles(tm=256), Tiles(tk=256), Tiles(tm=64)]
    blind = Trial(start, profile(start, seconds=10.0, wait=5.0, regions=False))
    assert proposals(k, sig, blind) == proposals(k, sig, compute)
    # At the most buffers there are, none are proposed.
    most = Config(Tiles(), Pipeline(4, 3))
    both = Trial(most, profile(most, seconds=10.0, wait=3.0, out_wait=3.0))
    assert all(c.pipeline == most.pipeline for c in proposals(k, sig, both))


WIDE = """
def f(x, y, t=Tiles()):
    M, K = x.shape
    out = jnp.zeros((M, 24448))
    for i in range(0, M, t.tm):
        acc = jnp.zeros((t.tm, 24448))
        for k in range(0, K, t.tk):
            acc += x[i : i + t.tm, k : k + t.tk] @ y[k : k + t.tk, :]
        out = out.at[i : i + t.tm, :].set(acc)
    return out
"""


def test_proposals_change_the_buffering_when_tiles_dont_fit_it():
    wide = make_kernel(WIDE, Tiles=Tiles)
    sig = wide.signature(rand((1024, 512)), rand((512, 24448), 1))
    start = wide.default_config(sig)
    proposed = proposals(wide, sig, Trial(start, profile(start, seconds=10.0)))  # (a 24448-wide f32 tile)
    by_tunable = {c.tunable: c.pipeline for c in proposed}
    assert start.pipeline == Pipeline(3, 2) and by_tunable[Tiles(tm=256)] == Pipeline(2, 2)
    assert Tiles(tm=512) not in by_tunable  # not even two buffers fit


def test_kernels_without_a_tunable_tune_their_buffering():
    k = make_kernel(SOURCE.replace("t=Tiles()", "").replace("t.tm", "128").replace("t.tk", "128"))
    sig = k.signature(X, Y)
    start = k.default_config(sig)
    assert start.tunable is None and str(start) == "3 input / 2 output buffers"
    dma = Trial(start, profile(start, seconds=10.0, wait=5.0))
    assert proposals(k, sig, dma) == [Config(None, Pipeline(4, 2))]
    assert _reference_config(k, sig, start) is None


# --------------------------------------------------------------------------- testing candidates


def test_the_reference_runs_the_fewest_iterations():
    k = kernel()
    sig = k.signature(X, Y)
    assert _reference_config(k, sig, k.default_config(sig)) == Tiles()  # tm=512 doesn't divide 256 rows
    sig = k.signature(rand((1024, 512)), Y if False else rand((512, 128)))
    assert _reference_config(k, sig, k.default_config(sig)) == Tiles(tm=512, tk=512)


def test_random_inputs():
    a, b, c = random_inputs((((8, 128), np.dtype(jnp.bfloat16)), ((4,), np.dtype(np.int8)), ((2,), np.dtype(bool))))
    assert a.dtype == jnp.bfloat16 and b.dtype == jnp.int8 and c.dtype == jnp.bool_
    assert int(jnp.abs(b).max()) <= 8 and float(jnp.std(a.astype(jnp.float32))) > 0.5


def test_measure_tests_each_candidate(monkeypatch):
    monkeypatch.setattr(profiling, "measure", lambda lowered, inputs, runs: profile(lowered.config))
    k = kernel()
    sig = k.signature(X, Y)
    want = k.reference(X, Y, Tiles())
    good = _measure(k, sig, Config(Tiles(tm=64), Pipeline()), (X, Y), want)
    assert good.profile is not None and good.error is None
    wrong = _measure(k, sig, Config(Tiles(), Pipeline()), (X, Y), want + 1)
    assert wrong.profile is None and wrong.error.startswith("wrong result: output 'out'")
    broken = _measure(k, sig, Config(Tiles(tm=512), Pipeline()), (X, Y), want)
    assert broken.seconds == np.inf and broken.error.startswith("KernelError: <test source")


# --------------------------------------------------------------------------- tuning a kernel


@pytest.mark.cpu
def test_off_a_tpu_tuning_uses_the_defaults():
    k = kernel()
    tuning = k.tuning(X, Y)
    assert tuning.config == Config(Tiles(), Pipeline(3, 2)) and tuning.trials == []
    assert str(tuning).endswith("(the defaults: tuning needs a TPU)")
    assert k.tuning(X, Y) is tuning  # once per signature
    np.testing.assert_allclose(f64(k(X, Y)), f64(X) @ f64(Y), rtol=1e-4, atol=1e-4)


@pytest.mark.cpu
def test_tuning_climbs_to_bigger_tiles_when_steps_are_overhead(monkeypatch):
    def measure(lowered, inputs, runs):
        steps = lowered.steps
        return profile(lowered.config, seconds=steps * 10e-6, steps=steps, compute=steps * 1e-6)

    monkeypatch.setattr(tune, "_can_time", lambda: True)
    monkeypatch.setattr(profiling, "measure", measure)
    k = kernel()
    tuning = k.tuning(X, Y)
    assert tuning.config.tunable == Tiles(tm=256, tk=256)  # one step: the least overhead
    assert [t.config.tunable for t in tuning.trials] == [Tiles(), Tiles(tm=256), Tiles(tm=256, tk=256)]
    assert all(t.profile is not None for t in tuning.trials)
    np.testing.assert_allclose(f64(k(X, Y)), f64(X) @ f64(Y), rtol=1e-4, atol=1e-4)
    assert k.lower(X, Y).config == tuning.config and k.verify(X, Y) < 1e-4
    assert "Tiles(tm=256, tk=256)" in str(tuning)


@pytest.mark.cpu
def test_tuning_when_the_first_call_is_traced(monkeypatch):
    monkeypatch.setattr(tune, "_can_time", lambda: True)
    monkeypatch.setattr(profiling, "measure", lambda lowered, inputs, runs: profile(lowered.config))
    k = kernel()
    out = jax.jit(k)(X, Y)  # tunes, running and timing kernels while jit traces the call
    np.testing.assert_allclose(f64(out), f64(X) @ f64(Y), rtol=1e-4, atol=1e-4)
    assert all(t.profile is not None for t in k.tuning(X, Y).trials)


def test_the_reference_keeps_the_defaults_when_they_run_fewest():
    @autokernel.tunable
    class Biggest:
        tm: int = autokernel.choice(128, 256, default=256)
        tk: int = autokernel.choice(128, 256, default=256)

    k = make_kernel(SOURCE.replace("t=Tiles()", "t=Biggest()"), Biggest=Biggest)
    sig = k.signature(X, Y)
    assert _reference_config(k, sig, k.default_config(sig)) == Biggest()


def test_a_kernel_and_its_tunable_can_be_local():
    @autokernel.tunable
    class Local:
        tm: int = autokernel.choice(8, 16)

    @autokernel.kernel
    def double(x, tiles=Local()):
        out = jnp.zeros(x.shape)
        for i in range(0, x.shape[0], tiles.tm):
            out = out.at[i : i + tiles.tm, :].set(x[i : i + tiles.tm, :] * 2)
        return out

    x = rand((32, 128))
    assert double.verify(x) == 0.0 and double.verify(x, Local(tm=16)) == 0.0
