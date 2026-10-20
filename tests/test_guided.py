"""Profile-guided tuning (`autotune="profile"`, `tune.guided`): the search follows each profile's bottleneck,
and its measurements carry over between runs through the tuning cache."""

import json

import jax.numpy as jnp
import pytest
from helpers import rand

import autokernel
from autokernel import cache, hardware, profiling
from autokernel import tune as tune_lib
from autokernel.tune import Candidate

S = 256
TM = TN = TK = 128


def matmul(a, b):
    c = jnp.zeros((S, S))
    for i in range(0, S, TM):
        for j in range(0, S, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, S, TK):
                acc += a[i : i + TM, k : k + TK] @ b[k : k + TK, j : j + TN]
            c = c.at[i : i + TM, j : j + TN].set(acc)
    return c


TUNE = {"TM": (128, 256), "TN": (128, 256), "TK": (128, 256)}


def cand(tm, tn, buffers=2, error=None):
    return Candidate({"TM": tm, "TN": tn}, buffers, None, error)


def made(c):
    return f"kernel {c.overrides} b={c.buffers}"


def run_guided(cands, times, bottlenecks, budget=10, history=None, check=None):
    """`tune.guided` with fake kernels: `times[key]` seconds, `bottlenecks[key]` diagnoses."""
    order = []

    def make(c):
        return made(c)

    def seconds(lw, args):
        order.append(lw)
        if times.get(lw) == "boom":
            raise RuntimeError("mosaic")
        return times[lw]

    best = tune_lib.guided(cands, (), budget, make, seconds, lambda lw, args: bottlenecks.get(lw), check, history)
    return best, order


def test_a_dma_bound_pick_tries_more_buffers_then_bigger_blocks(monkeypatch):
    monkeypatch.setattr(tune_lib, "GUIDED_SEED", 1)
    a2, a3, big, small = cand(128, 128, 2), cand(128, 128, 3), cand(256, 256), cand(128, 64)
    times = {made(a2): 3.0, made(a3): 2.0, made(big): 1.0, made(small): 5.0}
    bottlenecks = {made(a2): "dma", made(a3): "dma", made(big): "compute"}
    best, order = run_guided([a2, small, a3, big], times, bottlenecks, budget=3)
    assert order == [made(a2), made(a3), made(big)] and best == made(big)
    assert a2.bottleneck == "dma" and a3.bottleneck == "dma"


def test_an_overhead_bound_pick_tries_bigger_blocks(monkeypatch):
    monkeypatch.setattr(tune_lib, "GUIDED_SEED", 1)
    a, b, big = cand(128, 128), cand(128, 64), cand(256, 128)
    times = {made(a): 2.0, made(b): 3.0, made(big): 1.0}
    best, order = run_guided([a, b, big], times, {made(a): "overhead", made(big): "compute"}, budget=2)
    assert order == [made(a), made(big)] and best == made(big)


def test_a_compute_bound_pick_tries_the_models_next_until_the_spread(monkeypatch):
    monkeypatch.setattr(tune_lib, "GUIDED_SEED", 1)
    monkeypatch.setattr(tune_lib, "GUIDED_SPREAD", 2)
    cands = [cand(128, 128), cand(256, 128), cand(128, 256)]
    times = {made(c): t for c, t in zip(cands, (1.0, 2.0, 0.5), strict=True)}
    best, order = run_guided(cands, times, {made(cands[0]): "compute"}, budget=10)
    assert order == [made(cands[0]), made(cands[1])] and best == made(cands[0])  # the third is past the spread


def test_history_is_reused_and_extended():
    a, b = cand(128, 128), cand(256, 256)
    history = {tune_lib.candidate_key(a): 1.0}
    best, order = run_guided([a, b], {made(b): 2.0}, {made(a): "compute"}, history=history)
    assert order == [made(b)] and best == made(a) and a.measured_s == 1.0
    assert history[tune_lib.candidate_key(b)] == 2.0


def test_failures_and_wrong_results_are_dropped():
    a, b, c = cand(128, 128), cand(256, 256), cand(256, 128, error="too big")
    best, _ = run_guided([a, b, c], {made(a): "boom", made(b): 1.0}, {}, check=None)
    assert a.error == "failed to run: RuntimeError" and best == made(b)
    a2, b2 = cand(128, 128), cand(256, 256)
    best, _ = run_guided([a2, b2], {made(a2): 1.0, made(b2): 2.0}, {}, check=lambda lw: "nan" if "128" in lw else None)
    assert a2.error == "wrong result: nan" and best == made(b2)


def test_nothing_ran():
    a = cand(128, 128)
    assert run_guided([a], {made(a): "boom"}, {})[0] is None


def test_a_history_only_best_is_built():
    a = cand(128, 128)
    best, order = run_guided([a], {}, {made(a): "compute"}, budget=0, history={tune_lib.candidate_key(a): 1.0})
    assert best == made(a) and order == []


def test_keys_and_volumes():
    c = Candidate({"TM": 128, "flag": True, "s": 0.5}, 3, None)
    assert tune_lib.candidate_key(c) == "TM=int:128, flag=bool:True, s=float:0.5; buffers=3"
    assert tune_lib._volume(c) == 128.0 and tune_lib._volume(Candidate({"s": 0.5}, 2, None)) == 0.0


# --------------------------------------------------------------------------- through a kernel


@pytest.fixture
def on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOKERNEL_CACHE", "1")
    monkeypatch.setenv("AUTOKERNEL_CACHE_DIR", str(tmp_path))
    return tmp_path / "tuning"


def test_profile_mode_tunes_from_device_times_and_remembers(on_disk, monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    fastest = {"TK": 256, "TM": 256, "TN": 128}

    def fake_profile(lw, args, runs=10, trace=False, options=None):
        t = 1.0 if lw.overrides == fastest else 2.0
        return profiling.KernelProfile("matmul", runs, t, regions=trace, compute_s=t * 0.9)

    monkeypatch.setattr(profiling, "profile_lowered", fake_profile)
    a, b = rand((S, S)), rand((S, S), 1)
    first = autokernel.tpu(matmul, tune=TUNE, autotune="profile", interpret=True, target="v5e")
    assert first.lower(a, b).overrides == fastest
    timed = [c for c in first.tuning(a, b) if c.measured_s is not None]
    assert len(timed) >= tune_lib.GUIDED_SEED and any(c.bottleneck == "compute" for c in timed)
    (entry,) = [json.loads(p.read_text()) for p in on_disk.glob("*.json")]
    assert entry["measured_s"] == 1.0 and len(entry["history"]) == len(timed)

    # The next run starts from the cached pick and the history: nothing measured twice.
    measured = []
    monkeypatch.setattr(
        profiling,
        "profile_lowered",
        lambda lw, args, runs=10, trace=False, options=None: (
            measured.append(trace) or fake_profile(lw, args, runs, trace)
        ),
    )
    second = autokernel.tpu(matmul, tune=TUNE, autotune="profile", interpret=True, target="v5e")
    assert second.lower(a, b).overrides == fastest
    # Only what the first run didn't measure is timed; the rest are diagnoses (traced profiles).
    assert measured.count(False) <= tune_lib.GUIDED_SPREAD - len(timed) + 2
    cache_entry = cache.load(next(iter(on_disk.glob("*.json"))).stem)
    assert len(cache_entry.history) >= len(timed)
    second.verify(a, b)


def test_profile_mode_builds_candidates_the_search_left_unbuilt(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    real_search = tune_lib.search

    def few_built(*args, **kwargs):
        candidates = real_search(*args, **kwargs)
        for c in candidates[1:]:
            c.lowered = None
        return candidates

    monkeypatch.setattr(tune_lib, "search", few_built)
    monkeypatch.setattr(
        profiling,
        "profile_lowered",
        lambda lw, args, runs=10, trace=False, options=None: profiling.KernelProfile("matmul", runs, 1.0),
    )
    kern = autokernel.tpu(matmul, tune=TUNE, autotune="profile", interpret=True, target="v5e")
    a, b = rand((S, S)), rand((S, S), 1)
    kern.lower(a, b)
    built = [c for c in kern.tuning(a, b) if c.measured_s is not None]
    assert len(built) >= 2 and all(c.lowered is not None for c in built)
