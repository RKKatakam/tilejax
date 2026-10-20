"""The tuning cache (`autokernel.cache`) and the accuracy check of measured candidates."""

import dataclasses
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import cache, hardware, lowering
from autokernel import tune as tune_lib

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


@pytest.fixture
def on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOKERNEL_CACHE", "1")
    monkeypatch.setenv("AUTOKERNEL_CACHE_DIR", str(tmp_path))
    return tmp_path / "tuning"


def args():
    return rand((S, S)), rand((S, S), 1)


# --------------------------------------------------------------------------- the store


def test_entries_round_trip(on_disk):
    entry = cache.Entry({"TM": 128, "flag": True, "scale": 0.5, "mode": "x"}, 3, 1.5e-3)
    assert cache.store("k", entry)
    assert cache.load("k") == entry
    assert type(cache.load("k").overrides["TM"]) is int and type(cache.load("k").overrides["flag"]) is bool
    assert json.loads((on_disk / "k.json").read_text())["buffers"] == 3
    assert cache.store("model", cache.Entry({"np": np.int32(64), "f": np.float32(2)}, 2, None))
    assert cache.load("model") == cache.Entry({"np": 64, "f": 2.0}, 2, None)


def test_missing_and_unreadable_entries_are_misses(on_disk):
    assert cache.load("nothing") is None
    on_disk.mkdir(parents=True)
    (on_disk / "bad.json").write_text("{not json")
    assert cache.load("bad") is None
    (on_disk / "partial.json").write_text(json.dumps({"overrides": {}}))
    assert cache.load("partial") is None
    (on_disk / "kind.json").write_text(json.dumps({"overrides": {"T": {"tuple": 1}}, "buffers": 2, "measured_s": None}))
    assert cache.load("kind") is None


def test_values_that_are_not_plain_are_not_stored(on_disk):
    assert not cache.store("k", cache.Entry({"T": (1, 2)}, 2, None))
    assert cache.load("k") is None


def test_the_cache_can_be_turned_off(on_disk, monkeypatch):
    monkeypatch.setenv("AUTOKERNEL_CACHE", "0")
    assert not cache.enabled()
    assert not cache.store("k", cache.Entry({"T": 1}, 2, None))
    monkeypatch.setenv("AUTOKERNEL_CACHE", "1")
    assert cache.enabled() and cache.load("k") is None


def test_an_unwritable_cache_is_skipped(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("AUTOKERNEL_CACHE", "1")
    monkeypatch.setenv("AUTOKERNEL_CACHE_DIR", str(blocker))  # `tuning/` can't be made under a file
    assert not cache.store("k", cache.Entry({"T": 1}, 2, None))


def test_the_default_directory_is_under_the_home_cache(monkeypatch):
    monkeypatch.delenv("AUTOKERNEL_CACHE_DIR", raising=False)
    assert cache.directory() == cache.Path(os.path.expanduser("~")) / ".cache" / "autokernel" / "tuning"


# --------------------------------------------------------------------------- keys


def helper(x):
    return x + 1


def other_helper(x):
    return x + 2


class Thing:
    pass


def test_fingerprints_are_stable_and_tell_values_apart():
    fp = cache.fingerprint
    assert fp(3) != fp(3.0) and fp(True) != fp(1) and fp("a") == "a" and fp(None) is None
    assert fp((1, 2)) != fp([1, 2]) and fp({"a": 1}) == fp({"a": 1})
    assert fp(np.int32(3)) == ["int32", "np.int32(3)"]
    assert fp(np.dtype(np.float32)) == fp(np.float32) == ["dtype", "float32"]
    small = np.arange(4)
    assert fp(small) == fp(small.copy()) and fp(small) != fp(small + 1)
    assert fp(jnp.arange(4)) == fp(np.arange(4, dtype=np.int32))
    big = np.zeros(cache._SMALL_ARRAY // 4 + 1, np.float32)
    assert fp(big)[3] is None  # keyed by shape and dtype only
    assert fp(math) == ["module", "math"]
    assert fp(helper) != fp(other_helper) and fp(helper) == fp(helper)
    assert fp(Thing()) == ["object", __name__, "Thing"]
    assert fp(len)[0] == "callable"  # a builtin: no Python source
    nested = []
    for _ in range(12):
        nested = [nested]
    assert "..." in json.dumps(fp(nested))


def test_keys_change_with_everything_that_decides_the_pick():
    nest = autokernel.tpu(matmul).nest
    spec = [jax.ShapeDtypeStruct((S, S), jnp.float32)] * 2
    options = lowering.Options(tune=lowering.Options.make_tune(TUNE))
    base = cache.key(nest.fdef, {"S": S}, spec, "v5e", options)
    assert base == cache.key(nest.fdef, {"S": S}, spec, "v5e", options)
    assert base != cache.key(nest.fdef, {"S": S + 1}, spec, "v5e", options)
    assert base != cache.key(nest.fdef, {"S": S}, [jax.ShapeDtypeStruct((S, S), jnp.bfloat16)] * 2, "v5e", options)
    assert base != cache.key(nest.fdef, {"S": S}, spec, "v5p", options)
    assert base != cache.key(nest.fdef, {"S": S}, spec, "v5e", dataclasses.replace(options, autotune="measure"))
    other = autokernel.tpu(_matmul_t).nest
    assert base != cache.key(other.fdef, {"S": S}, spec, "v5e", options)


def _matmul_t(a, b):
    c = jnp.zeros((S, S))
    for i in range(0, S, TM):
        for j in range(0, S, TN):
            acc = jnp.zeros((TM, TN))
            for k in range(0, S, TK):
                acc += a[i : i + TM, k : k + TK] @ b[j : j + TN, k : k + TK].T
            c = c.at[i : i + TM, j : j + TN].set(acc)
    return c


# --------------------------------------------------------------------------- kernels using it


def test_a_search_is_reused_by_a_new_kernel(on_disk, monkeypatch):
    first = autokernel.tpu(matmul, tune=TUNE)
    a, b = args()
    picked = first.lower(a, b)
    assert len(first.tuning(a, b)) == 8 and not any(c.cached for c in first.tuning(a, b))
    assert len(list(on_disk.glob("*.json"))) == 1

    searches = []
    monkeypatch.setattr(tune_lib, "search", lambda *a, **k: searches.append(1))
    second = autokernel.tpu(matmul, tune=TUNE)
    again = second.lower(a, b)
    assert not searches and again.overrides == picked.overrides and again.buffers == picked.buffers
    (only,) = second.tuning(a, b)
    assert only.cached and only.measured_s is None and only.lowered is again
    second.verify(a, b)


def test_a_stale_entry_is_searched_again(on_disk):
    kern = autokernel.tpu(matmul, tune=TUNE)
    a, b = args()
    kern.lower(a, b)
    (path,) = on_disk.glob("*.json")
    data = json.loads(path.read_text())
    data["overrides"]["TM"] = {"int": 96}  # no longer a valid pick (a ragged loop, say)
    data["overrides"]["TK"] = {"int": 100}
    path.write_text(json.dumps(data))
    fresh = autokernel.tpu(matmul, tune=TUNE)
    lw = fresh.lower(a, b)
    assert len(fresh.tuning(a, b)) == 8 and lw.overrides["TK"] in (128, 256)


def test_model_picks_are_measured_again_and_measured_picks_are_kept(on_disk, monkeypatch):
    a, b = args()
    autokernel.tpu(matmul, tune=TUNE, target="v5e").lower(a, b)  # a model pick in the cache
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    timed = []

    def fake_time(lw, xs):
        timed.append(dict(lw.overrides))
        return 0.5 if lw.overrides == {"TM": 256, "TN": 128, "TK": 128} else 1.0

    monkeypatch.setattr(tune_lib, "time_kernel", fake_time)
    kern = autokernel.tpu(matmul, tune=TUNE, autotune="measure", target="v5e", interpret=True)
    best = kern.lower(a, b)
    assert timed and best.overrides == {"TM": 256, "TN": 128, "TK": 128}
    assert not any(c.cached for c in kern.tuning(a, b))
    entries = [json.loads(p.read_text()) for p in on_disk.glob("*.json")]
    assert any(e["measured_s"] == 0.5 for e in entries)

    timed.clear()
    reuse = autokernel.tpu(matmul, tune=TUNE, autotune="measure", target="v5e", interpret=True)
    assert reuse.lower(a, b).overrides == best.overrides and not timed  # measured once, ever
    (only,) = reuse.tuning(a, b)
    assert only.cached and only.measured_s == 0.5


# --------------------------------------------------------------------------- checking measured candidates


def test_measure_drops_candidates_with_wrong_results(monkeypatch):
    # Model mode: on a TPU, a measured kernel would already have timed its candidates in `tuning`.
    kern = autokernel.tpu(matmul, tune=TUNE, interpret=True)
    a, b = args()
    candidates = [c for c in kern.tuning(a, b) if c.lowered is not None]
    fastest = candidates[0].lowered
    monkeypatch.setattr(tune_lib, "time_kernel", lambda lw, xs: 0.1 if lw is fastest else 1.0)
    best = tune_lib.measure(candidates, (a, b), 8, lambda lw: "off by one" if lw is fastest else None)
    assert best is not fastest
    assert candidates[0].error == "wrong result: off by one" and candidates[0].measured_s is None


def test_the_accuracy_check_compares_with_the_python_loops():
    kern = autokernel.tpu(matmul, tune=TUNE, autotune="measure", interpret=True)
    a, b = args()
    ctx, inputs = kern._bind((a, b), {})
    lowered = kern.lower(a, b)
    check = kern._accuracy_check(ctx, inputs)
    assert check(lowered) is None
    broken = dataclasses.replace(lowered, entry=lambda *xs, interpret=None: lowered.entry(*xs, interpret=interpret) + 1)
    assert "disagrees with the Python reference" in check(broken)


def test_a_measured_search_checks_every_candidate(monkeypatch):
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    monkeypatch.setattr(tune_lib, "time_kernel", lambda lw, xs: 1.0)
    checked = []
    real = autokernel.TpuKernel._accuracy_check

    def spy(self, ctx, arrays):
        check = real(self, ctx, arrays)
        return lambda lw: checked.append(lw) or check(lw)

    monkeypatch.setattr(autokernel.TpuKernel, "_accuracy_check", spy)
    kern = autokernel.tpu(matmul, tune=TUNE, autotune="measure", interpret=True, target="v5e")
    a, b = args()
    kern(a, b)
    timed = [c for c in kern.tuning(a, b) if c.measured_s is not None]
    assert timed and len(checked) == len(timed) and all(c.error is None for c in timed)
