"""Planning: loop ranges, tile windows, operand kinds, carried values and their types, and the errors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import make_kernel, spec

from autokernel import KernelError
from autokernel.plan import Kind, Window

X = spec(256, 256)


def plan(source: str, *inputs, **namespace):
    kernel = make_kernel(source, **namespace)
    return kernel.build(kernel.signature(*(inputs or (X,))), None)[0]


def plan_error(source: str, *inputs, **namespace) -> str:
    with pytest.raises(KernelError) as info:
        plan(source, *inputs, **namespace)
    return str(info.value)


def nest_with(tile: str, prologue: str = "out = jnp.zeros((256, 256))", loops: str = "for i in range(0, 256, 128):"):
    """A one-input kernel `f(x)`: the prologue, then `loops` around the tile lines."""
    lines = ["def f(x):", f"    {prologue}", f"    {loops}"]
    lines += [f"        {line}" for line in tile.strip().splitlines()]
    lines += ["    return out"]
    return "\n".join(lines)


def test_matmul_plan():
    p = plan(
        """
        def f(x, y):
            M, K = x.shape
            out = jnp.zeros((M, y.shape[1]), x.dtype)
            for i in range(0, M, 128):
                for j in range(0, 256, 128):
                    acc = jnp.zeros((128, 128))
                    for k in range(0, K, 128):
                        acc += x[i : i + 128, k : k + 128] @ y[k : k + 128, j : j + 128]
                    out = out.at[i : i + 128, j : j + 128].set(acc + x[i : i + 128, 0:128])
            return out
        """,
        spec(256, 384),
        spec(384, 256, dtype=jnp.bfloat16),
    )
    assert [(r.var, r.count) for r in p.loops] == [("i", 2), ("j", 2), ("k", 3)] and not p.single_step
    assert [(op.name, op.kind) for op in p.operands] == [("x", Kind.STREAM), ("y", Kind.STREAM), ("x2", Kind.TILE)]
    assert p.operands[0].tile.windows == (Window("i", 0, 128), Window("k", 0, 128))
    assert p.operands[2].tile.windows == (Window("i", 0, 128), Window(None, 0, 128))
    assert p.carried["acc"].dtype == jnp.float32 and p.carried["acc"].shape == (128, 128)
    out = p.outputs[0]
    assert (out.shape, out.dtype, out.fill, out.value.dtype) == ((256, 256), np.float32, 0, jnp.float32)
    assert [phase.range.var for phase in p.phases] == ["k"] and p.eval(p.nest.parallel[0].bounds[1]) == 256


def test_operand_kinds_and_windows():
    p = plan(
        """
        def f(x, b, s):
            out = jnp.zeros((2, 256, 128))
            for h in range(2):
                for i in range(0, 256, 128):
                    out = out.at[h, i : i + 128, :].set(x[h, i : i + 128, :] + b[:, 128:].sum(0) + s[1, 8:16, :].sum(0))
            return out
        """,
        spec(2, 256, 128),
        spec(8, 256),
        spec(2, 16, 128),
    )
    assert p.single_step
    assert [op.kind for op in p.operands] == [Kind.STREAM, Kind.ONCE, Kind.ONCE]
    assert p.operands[0].tile.windows[0] == Window("h", 0, None) and p.operands[0].tile.shape == (128, 128)
    assert p.operands[1].tile.windows == (Window(None, 0, 8), Window(None, 128, 128))
    assert p.operands[2].tile.windows == (Window(None, 1, None), Window(None, 8, 8), Window(None, 0, 128))


def test_tile_operands_and_constants():
    p = plan(
        """
        def f(x):
            out = jnp.full((256, 256), -1.0, jnp.bfloat16)
            for i in range(0, 256, 128):
                with jax.named_scope("pre"):
                    t = x[i : i + 128, :] * SCALE
                for k in range(0, 256, 128):
                    t = t + x[k : k + 128, :]
                out = out.at[i : i + 128, :].set(t)
            return out
        """,
        SCALE=2.0,
    )
    assert [op.kind for op in p.operands] == [Kind.TILE, Kind.STREAM]
    assert p.constants == {"SCALE": 2.0, "jax": jax} and p.scopes == ("pre",)
    assert p.outputs[0].fill == -1.0 and set(p.carried) == {"t"}


def test_full_with_keyword_fill():
    p = plan(
        nest_with("out = out.at[i : i + 128, :].set(x[i : i + 128, :])", "out = jnp.full((256, 256), fill_value=3)")
    )
    assert p.outputs[0].fill == 3 and p.outputs[0].dtype == jnp.int32


@pytest.mark.parametrize(
    ("prologue", "message"),
    [
        ("out = x", "allocate output `out` with jnp.zeros"),
        ("out = jnp.ones((256, 256))", "allocate output `out` with jnp.zeros"),
        ("out = jnp.zeros((256, 256), 'nope')", "allocate output `out` with jnp.zeros"),
        ("out = jnp.full((256, 256), jnp.ones(()))", "filled with a scalar constant"),
        ("out = jnp.zeros(undefined)", "can't evaluate `undefined`"),
    ],
)
def test_allocations(prologue, message):
    assert message in plan_error(nest_with("out = out.at[i : i + 128, :].set(1.0)", prologue))


def test_the_prologue_runs_on_shapes():
    source = """
        def f(x):
            n = x.sum()
            out = jnp.zeros((256, 256))
            for i in range(0, 256, 128):
                out = out.at[i : i + 128, :].set(1.0)
            return out
        """
    assert "the prologue runs on the inputs' shapes" in plan_error(source)


@pytest.mark.parametrize(
    ("loops", "message"),
    [
        ("for i in range(0, 256.0, 128):", "loop bounds are ints"),
        ("for i in range(0, 256, 0):", "steps by 0"),
        ("for i in range(256, 0):", "which is empty"),
        ("for i in range(0, 250, 128):", "whole number of steps"),
        ("for i in range(64, 256, 128):", "whole number of steps"),
        ("for i in range(64, 320, 128):", "start at a multiple of its step"),
    ],
)
def test_loop_ranges(loops, message):
    assert message in plan_error(nest_with("out = out.at[i : i + 128, :].set(1.0)", loops=loops))


@pytest.mark.parametrize(
    ("read", "message"),
    [
        ("x[i : i + 128 : 2, :]", "each index is"),
        ("x[i : 128, :]", "each index is"),
        ("x[i : i + i, :]", "each index is"),
        ("x[0 : i + 128, :]", "each index is"),
        ("x[i + 1, :]", "each index is"),
        ("x[i : i + 128.0, :]", "tile bounds are ints"),
        ("x[i : i + 0, :]", "at least one element"),
        ("x[i : i + 256, :]", "reads elements 0 to 383"),
        ("x[-1, :]", "reads elements -1 to -1"),
        ("x[i : i + 128]", "index every one of x's 2 dims"),
        ("x[0, 0]", "keeps at least one dim"),
        ("x[i : i + 128, 0]", "integer indices go before every sliced dim"),
        ("x[i : i + 128, 0:100]", "multiple of 128 or all 256"),
        ("x[i : i + 4, :]", "multiple of 8 for float32"),
    ],
)
def test_input_windows(read, message):
    assert message in plan_error(nest_with(f"out = out.at[i : i + 128, :].set({read}.sum())"))


@pytest.mark.parametrize(
    ("store", "message"),
    [
        ("out = out.at[:, :].set(1.0)", "`i` indexes exactly one dim"),
        ("out = out.at[i : i + 128, 0:128].set(1.0)", "stored whole, with `:`"),
        ("out = out.at[i : i + 64, :].set(1.0)", "must cover the dim of 256 exactly"),
        ("out = out.at[i : i + 128, :].set((1, 2))", "must be an array or a number"),
        ("out = out.at[i : i + 128, :].set(jnp.ones(3))", "can't store a value of shape (3,)"),
        ("out = out.at[i : i + 128, :].set(jnp.ones((128, 256, 2)))", "can't store a value of shape (128, 256, 2)"),
    ],
)
def test_output_stores(store, message):
    assert message in plan_error(nest_with(store))


def test_output_loops_cover_from_zero():
    source = nest_with("out = out.at[i : i + 128, :].set(1.0)", loops="for i in range(128, 256, 128):")
    assert "must cover the dim of 256 exactly" in plan_error(source)


def _reduction(pre: str, body: str, post: str = "out = out.at[i : i + 128, :].set(acc)") -> str:
    return nest_with(f"{pre}\nfor k in range(0, 256, 128):\n    {body}\n{post}")


def test_a_scalar_start_broadcasts():
    p = plan(_reduction("acc = 0.0", "acc = acc + x[k : k + 128, :]"))
    assert p.carried["acc"].shape == (128, 256)


def test_carried_values():
    p = plan(_reduction("acc = -1e30\nscale = 2.0", "acc = jnp.maximum(acc, x[k : k + 128, :].astype(jnp.bfloat16))"))
    assert p.carried["acc"].dtype == jnp.bfloat16  # a Python number takes the tile's type, as in Python
    p = plan(
        _reduction(
            "acc = jnp.zeros((128, 256))\nb = acc", "c = x[k : k + 128, :]", "out = out.at[i : i + 128, :].set(c + b)"
        )
    )
    assert set(p.carried) == {"b"}  # read after the loop, assigned before it; `c` is the last step's


@pytest.mark.parametrize(
    ("pre", "body", "message"),
    [
        ("acc = 0.0", "acc = acc + 1.0", "so it must be an array"),
        ("acc = jnp.zeros((128, 256))", "acc = jnp.concatenate([acc, acc])", "as float32[256, 256], but"),
        ("acc = jnp.zeros((128, 256), jnp.int32)", "acc = (acc + 1).astype(jnp.int8)", "keeps one type"),
        ("acc = jnp.zeros((128, 256), jnp.bfloat16)", "acc = acc + x[k : k + 128, :]", "as float32[128, 256], but"),
        ("acc = jnp.zeros((128, 256))", "acc = acc + 1\n    acc = acc.astype(jnp.bfloat16)", "keeps one type"),
        ("acc = jnp.ones((128, 1))", "acc = jnp.sum(acc, 1, keepdims=True) + x[k : k + 128, :]", "keeps one type"),
        ("acc = jnp.zeros((128, 256))\nacc = acc.astype(jnp.int8)", "acc = acc + 1.0", "keeps one type"),
        ("acc = None\nacc = jnp.zeros((128, 256))", "acc = acc + 1.0", "makes it not an array"),
        ("acc = jnp.zeros((128, 256))\nn = acc", "acc, n = acc + 1, n", "one at a time, not by unpacking"),
        ("acc = jnp.zeros((128, 256))", "acc = acc + undefined", "tile code fails for these input shapes"),
        ("acc = jnp.zeros((128, 256))", "acc = acc @ x[k : k + 128, :]", "tile code fails for these input shapes"),
    ],
)
def test_carried_value_errors(pre, body, message):
    assert message in plan_error(_reduction(pre, body))


def test_nothing_carries_over_between_tiles():
    error = plan_error(nest_with("v = w + 1\nw = x[i : i + 128, :]\nout = out.at[i : i + 128, :].set(v)"))
    assert "`w` is read before the tile assigns it" in error


def test_tile_code_errors_name_the_line():
    error = plan_error(_reduction("acc = jnp.zeros((128, 256))", "acc = acc + undefined"))
    assert error.splitlines()[1].strip() == "acc = acc + undefined"


def test_named_scopes_must_be_named():
    for opener in ("jax.named_scope(NAME_NUMBER)", "jax.named_scope('a', 'b')", "jax.default_device(None)"):
        source = nest_with(f"with {opener}:\n    v = 1.0\nout = out.at[i : i + 128, :].set(v)")
        assert "only `with jax.named_scope(name):`" in plan_error(source, NAME_NUMBER=3)
