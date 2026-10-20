"""Random matmul nests against NumPy: the kernel *and* the reference are each compared with an einsum in
float64, so a mistake shared by both (they are built from the same parsed nest) can't hide.

Each case draws a dtype (f32, bf16, int8), block sizes, sizes that are or aren't whole numbers of blocks
(ragged loops), a transposed or batched operand, an epilogue and a static tile mask, writes the nest as
Python source in a module of its own (the frontend reads functions' source) and compiles it.
"""

import importlib.util
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

import autokernel

CASES = 40

_TEMPLATE = """
import jax.numpy as jnp

def kernel({params}):
    out = jnp.zeros({out_shape}, {out_dtype})
{loops}
    return out
"""


def _case(seed):
    rng = np.random.default_rng(seed)
    dtype = rng.choice(["float32", "bfloat16", "int8"])
    sub = {"float32": 8, "bfloat16": 16, "int8": 32}[dtype]
    tm = int(rng.choice([1, 2, 4])) * sub
    tn, tk = int(rng.choice([128, 256])), int(rng.choice([128, 256]))
    ragged = rng.random() < 0.4

    def size(t):
        n = t * int(rng.integers(1, 4))
        return n + int(rng.integers(1, t)) if ragged and rng.random() < 0.7 else n

    m, n, k = size(tm), size(tn), size(tk)
    batch = int(rng.integers(2, 4)) if rng.random() < 0.3 else 0
    transposed = rng.random() < 0.3
    epilogue = str(rng.choice(["", "relu", "bias", "scale", "cast"]))
    if dtype == "int8" and epilogue == "cast":
        epilogue = "scale"
    masked = not batch and rng.random() < 0.25
    return dict(dtype=dtype, tm=tm, tn=tn, tk=tk, m=m, n=n, k=k, batch=batch, transposed=transposed,
                epilogue=epilogue, masked=masked)  # fmt: skip


def _source(c):
    b = "b, " if c["batch"] else ""
    x = f"x[{b}i : i + {c['tm']}, k : k + {c['tk']}]"
    y = f"y[j : j + {c['tn']}, k : k + {c['tk']}].T" if c["transposed"] else f"y[k : k + {c['tk']}, j : j + {c['tn']}]"
    acc_dtype = "jnp.int32" if c["dtype"] == "int8" else "jnp.float32"
    value = {
        "": "acc",
        "relu": "jnp.maximum(acc, 0)",
        "bias": f"acc + bias[:, j : j + {c['tn']}]",
        "scale": "acc * 2",
        "cast": "acc.astype(jnp.bfloat16)",
    }[c["epilogue"]]
    out_dtype = "jnp.bfloat16" if c["epilogue"] == "cast" else acc_dtype
    body = [
        f"for i in range(0, {c['m']}, {c['tm']}):",
        f"    for j in range(0, {c['n']}, {c['tn']}):",
        *(["        if j > i:", "            continue"] if c["masked"] else []),
        f"        acc = jnp.zeros(({c['tm']}, {c['tn']}), {acc_dtype})",
        f"        for k in range(0, {c['k']}, {c['tk']}):",
        f"            acc += {x} @ {y}",
        f"        out = out.at[{b}i : i + {c['tm']}, j : j + {c['tn']}].set({value})",
    ]
    if c["batch"]:
        body = [f"for b in range({c['batch']}):", *("    " + line for line in body)]
    params = "x, y, bias" if c["epilogue"] == "bias" else "x, y"
    shape = (c["batch"], c["m"], c["n"]) if c["batch"] else (c["m"], c["n"])
    return _TEMPLATE.format(
        params=params, out_shape=shape, out_dtype=out_dtype, loops=textwrap.indent("\n".join(body), "    ")
    )


def _inputs(c, rng):
    dt = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "int8": jnp.int8}[c["dtype"]]
    xs = (c["batch"], c["m"], c["k"]) if c["batch"] else (c["m"], c["k"])
    ys = (c["n"], c["k"]) if c["transposed"] else (c["k"], c["n"])
    if c["dtype"] == "int8":
        x, y = rng.integers(-8, 8, xs), rng.integers(-8, 8, ys)
    else:
        x, y = rng.standard_normal(xs), rng.standard_normal(ys)
    args = [jnp.asarray(x, dt), jnp.asarray(y, dt)]
    if c["epilogue"] == "bias":
        bias = rng.integers(-4, 4, (1, c["n"])) if c["dtype"] == "int8" else rng.standard_normal((1, c["n"]))
        args.append(jnp.asarray(bias, jnp.int32 if c["dtype"] == "int8" else jnp.float32))
    return args


def _expected(c, args):
    x, y = (np.asarray(a.astype(jnp.float32), np.float64) for a in args[:2])
    want = np.einsum("...mk,nk->...mn" if c["transposed"] else "...mk,kn->...mn", x, y)
    if c["epilogue"] == "relu":
        want = np.maximum(want, 0)
    elif c["epilogue"] == "bias":
        want = want + np.asarray(args[2], np.float64)
    elif c["epilogue"] == "scale":
        want = want * 2
    if c["masked"]:
        rows = (np.arange(c["m"]) // c["tm"]) * c["tm"]
        cols = (np.arange(c["n"]) // c["tn"]) * c["tn"]
        want = np.where(cols[None, :] > rows[:, None], 0.0, want)
    return want


def _compile(c, tmp_path, seed):
    path = tmp_path / f"fuzz_case_{seed}.py"
    path.write_text(_source(c))
    spec = importlib.util.spec_from_file_location(f"fuzz_case_{seed}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return autokernel.tpu(module.kernel)


def _tolerance(c):
    if c["dtype"] == "int8":
        return 0.0, 0.0
    if c["epilogue"] == "cast":
        return 1e-2, 1e-2 * np.sqrt(c["k"])
    if c["dtype"] == "bfloat16":
        return 1e-4, 1e-4 * np.sqrt(c["k"])  # exact bf16 products, f32 sums
    return 1e-4, 1e-4 * np.sqrt(c["k"])


@pytest.mark.parametrize("seed", range(CASES))
def test_random_nest_matches_numpy(seed, tmp_path):
    c = _case(seed)
    kern = _compile(c, tmp_path, seed)
    args = _inputs(c, np.random.default_rng(1000 + seed))
    want = _expected(c, args)
    rtol, atol = _tolerance(c)
    got = kern(*args)
    assert got.shape == want.shape, (c, got.shape)
    np.testing.assert_allclose(np.asarray(got.astype(jnp.float32), np.float64), want, rtol=rtol, atol=atol,
                               err_msg=f"kernel, case {c}")  # fmt: skip
    ref = kern.reference(*args)
    np.testing.assert_allclose(np.asarray(ref.astype(jnp.float32), np.float64), want, rtol=rtol, atol=atol,
                               err_msg=f"reference, case {c}")  # fmt: skip


def test_the_cases_cover_every_feature():
    cases = [_case(s) for s in range(CASES)]
    assert {c["dtype"] for c in cases} == {"float32", "bfloat16", "int8"}
    assert {c["epilogue"] for c in cases} == {"", "relu", "bias", "scale", "cast"}
    for feature in ("batch", "transposed", "masked"):
        assert any(c[feature] for c in cases) and not all(c[feature] for c in cases), feature
    ragged = [c for c in cases if c["m"] % c["tm"] or c["n"] % c["tn"] or c["k"] % c["tk"]]
    assert 0 < len(ragged) < CASES
