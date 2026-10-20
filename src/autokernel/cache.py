"""Tuning results on disk, so a block-size search (and its measurements on the TPU) runs once per kernel
rather than once per process.

An entry is the tuned constants and buffer depth a search picked, keyed by everything that decides that
pick: the nest's source, the values of the constants it reads, the input signature, the chip, the options,
and the versions of JAX and autokernel. Entries live in `$AUTOKERNEL_CACHE_DIR` (default
`~/.cache/autokernel`), one JSON file each under `tuning/`; `AUTOKERNEL_CACHE=0` turns the cache off.

A measured pick (`autotune="measure"`) is only reused by a measured search, and a model pick by either (a
measured search then measures again, and replaces it). A profile-guided search (`autotune="profile"`) also
keeps every candidate it measured, so each run carries on from what earlier runs learned. Unreadable or
stale entries are ignored, and a cache that can't be written is skipped: the cache never changes what a
kernel computes, only how long finding its block sizes takes.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np

__all__ = ["Entry", "directory", "enabled", "fingerprint", "key", "load", "store"]

_VERSION = 1  # of the entry format and key: bump to drop every entry
_SMALL_ARRAY = 1 << 20  # constant arrays up to this many bytes are keyed by their contents, larger by shape


@dataclasses.dataclass(frozen=True)
class Entry:
    overrides: dict[str, Any]
    buffers: int
    measured_s: float | None  # None: the cost model's pick, not measured
    # Every candidate measured so far (`autotune="profile"`), by `tune.candidate_key`: device seconds.
    history: dict[str, float] = dataclasses.field(default_factory=dict)


def enabled() -> bool:
    return os.environ.get("AUTOKERNEL_CACHE", "1").lower() not in ("0", "false", "no", "off", "")


def directory() -> Path:
    base = os.environ.get("AUTOKERNEL_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "autokernel")
    return Path(base) / "tuning"


def fingerprint(value: Any, depth: int = 0) -> Any:
    """A JSON-able stand-in for a constant the kernel reads, the same in every process for the same value:
    scalars and strings as themselves, containers element by element, small arrays by their contents,
    functions by their source, modules by name, anything else by its type."""
    if depth > 8:
        return "..."
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, complex, np.number)):
        return [type(value).__name__, repr(value)]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [fingerprint(v, depth + 1) for v in value]]
    if isinstance(value, Mapping):
        items = sorted(((repr(k), fingerprint(v, depth + 1)) for k, v in value.items()), key=lambda kv: kv[0])
        return ["dict", items]
    if isinstance(value, np.dtype) or (isinstance(value, type) and issubclass(value, np.generic)):
        return ["dtype", str(np.dtype(value))]
    if isinstance(value, (np.ndarray, jax.Array)) and not isinstance(value, jax.core.Tracer):
        a = np.asarray(value)
        contents = hashlib.sha256(a.tobytes()).hexdigest() if a.nbytes <= _SMALL_ARRAY else None
        return ["array", list(a.shape), str(a.dtype), contents]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    if callable(value):
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            code = getattr(value, "__code__", None)
            source = code.co_code.hex() if code is not None else ""
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return ["callable", getattr(value, "__module__", None), name, hashlib.sha256(source.encode()).hexdigest()]
    return ["object", type(value).__module__, type(value).__qualname__]


def key(
    fdef: ast.FunctionDef,
    constants: Mapping[str, Any],
    specs: Sequence[jax.ShapeDtypeStruct],
    chip: str,
    options: Any,
) -> str:
    """The cache key of a search: a hash of the nest's code, its constants' values, the input signature, the
    chip, the options that shape the search and the library versions."""
    from . import __version__  # the package imports this module

    parts = {
        "format": _VERSION,
        "source": ast.dump(fdef, include_attributes=False),
        "constants": sorted((name, fingerprint(v)) for name, v in constants.items()),
        "inputs": [[list(s.shape), str(np.dtype(s.dtype)), bool(getattr(s, "weak_type", False))] for s in specs],
        "chip": chip,
        "options": fingerprint(dataclasses.asdict(options)),
        "jax": jax.__version__,
        "autokernel": __version__,
    }
    blob = json.dumps(parts, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return {"bool": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"int": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"float": float(value)}
    if isinstance(value, str):
        return {"str": value}
    raise TypeError(type(value).__name__)


def _decode(value: Mapping[str, Any]) -> Any:
    ((kind, v),) = value.items()
    return {"bool": bool, "int": int, "float": float, "str": str}[kind](v)


def load(k: str) -> Entry | None:
    """The entry stored under `k`, or None (no entry, the cache off, or an entry that can't be read)."""
    if not enabled():
        return None
    try:
        data = json.loads((directory() / f"{k}.json").read_text())
        overrides = {name: _decode(v) for name, v in data["overrides"].items()}
        measured = data["measured_s"]
        history = {str(k): float(v) for k, v in data.get("history", {}).items()}
        return Entry(overrides, int(data["buffers"]), None if measured is None else float(measured), history)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def store(k: str, entry: Entry) -> bool:
    """Write `entry` under `k`, atomically; False if the cache is off, can't be written, or the tuned values
    aren't plain numbers or strings."""
    if not enabled():
        return False
    try:
        data = {
            "overrides": {name: _encode(v) for name, v in entry.overrides.items()},
            "buffers": int(entry.buffers),
            "measured_s": entry.measured_s,
            "history": {k: float(v) for k, v in entry.history.items()},
        }
    except TypeError:
        return False
    folder = directory()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, folder / f"{k}.json")
        return True
    except OSError:
        return False
