"""The loop-nest language: what parses, and the error each rule gives, naming the line."""

import ast

import jax
import jax.numpy as jnp
import pytest

import autokernel
from autokernel import KernelError
from autokernel.parse import parse
from autokernel.tilecode import first_reads, reads


def error(fn) -> str:
    with pytest.raises(KernelError) as info:
        autokernel.kernel(fn)
    return str(info.value)


def test_a_nest_parses_into_its_parts():
    def f(x, y, /):
        """A docstring."""
        M = x.shape[0]
        out = jnp.zeros((M, 128))
        for i in range(0, M, 8):
            for j in range(0, 128, 128):
                if i > 8:
                    continue
                acc = jnp.zeros((8, 128))
                for k in range(0, 128, 128):
                    if k > 0:
                        continue
                    with jax.named_scope("mm"):
                        acc @= y[k : k + 128, j : j + 128]
                out = out.at[i : i + 8, j : j + 128].set(acc + x[i : i + 8, j : j + 128])
        return out

    nest = parse(f)
    assert nest.params == ("x", "y") and nest.outputs == ("out",)
    assert [loop.var for loop in nest.parallel] == ["i", "j"] and [loop.var for loop in nest.reductions] == ["k"]
    assert nest.parallel[0].mask is None and nest.parallel[1].mask is not None and nest.reductions[0].mask is not None
    assert len(nest.prologue) == 2 and [len(s) for s in nest.segments] == [1, 1] and len(nest.bodies[0]) == 1
    assert nest.store(nest.segments[1][0]) is nest.stores["out"] and nest.store(nest.segments[0][0]) is None
    assert [loop.var for loop in nest.loops] == ["i", "j", "k"]
    assert nest.error("boom", nest.segments[1][0]).args[0].startswith(f"{__file__}:")


def test_a_nest_without_a_reduction_is_all_post():
    def f(x):
        out = jnp.zeros(x.shape)
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :])
        return out

    nest = parse(f)
    assert (
        nest.reductions == nest.bodies == () and [len(s) for s in nest.segments] == [1] and nest.loops == nest.parallel
    )


def test_the_kernel_must_be_python_source():
    assert "can't read its source" in error(len)
    assert "defined with `def`" in error(lambda x: x)


def test_parameters_are_positional():
    def f(x, *rest):
        pass

    def g(x, y=1, z=2):
        pass

    def h(x, *, y):
        pass

    for fn in (f, g, h):
        assert "its input arrays, then optionally a tunable" in error(fn)


def test_the_function_shape():
    def no_loop(x):
        return x

    def bad_prologue(x):
        if x:
            pass
        for i in range(1):
            pass
        return x

    def nothing_after(x):
        for i in range(1):
            pass

    def code_after(x):
        out = jnp.zeros(1)
        for i in range(1):
            pass
        out = out + 1
        return out

    assert "no `for` loop" in error(no_loop)
    assert "assignments and asserts" in error(bad_prologue)
    assert "followed by `return`" in error(nothing_after)
    assert "followed by `return`" in error(code_after)


def test_returns_and_allocations():
    def returns_expression(x):
        for i in range(1):
            pass
        return x + 1

    def returns_twice(x):
        out = jnp.zeros(1)
        for i in range(1):
            pass
        return out, out

    def returns_nothing(x):
        for i in range(1):
            pass
        return

    def unallocated(x):
        for i in range(1):
            pass
        return out  # noqa: F821

    def allocated_twice(x):
        out = jnp.zeros(1)
        out = jnp.zeros(2)
        for i in range(1):
            pass
        return out

    def unpacked(x):
        out, other = jnp.zeros(1), 2
        for i in range(1):
            pass
        return out

    def augmented(x):
        out = jnp.zeros(1)
        out += 1
        for i in range(1):
            pass
        return out

    assert "return the outputs by name" in error(returns_expression)
    assert "return the outputs by name" in error(returns_nothing)
    assert "returned once" in error(returns_twice)
    assert "must be allocated before the loops" in error(unallocated)
    for fn in (allocated_twice, unpacked, augmented):
        assert "allocate output `out` once" in error(fn)


def test_loops():
    def not_range(x):
        out = jnp.zeros(1)
        for i in [0, 1]:
            pass
        return out

    def keywords(x):
        out = jnp.zeros(1)
        for i in range(stop=1):
            pass
        return out

    def unpacking(x):
        out = jnp.zeros(1)
        for i, j in range(1):
            pass
        return out

    def with_else(x):
        out = jnp.zeros(1)
        for i in range(1):
            pass
        else:
            pass
        return out

    def only_a_mask(x):
        out = jnp.zeros(1)
        for i in range(1):
            if i:
                continue
        return out

    def two_reductions(x):
        out = jnp.zeros(1)
        for i in range(1):
            for k in range(1):
                a = 1
            for k2 in range(1):
                b = 2
            out = out.at[i].set(1)
        return out

    def nested_repeats(x):
        out = jnp.zeros(1)
        for i in range(1):
            for r in range(1):
                for k in range(1):
                    for k2 in range(1):
                        a = 1
            out = out.at[i].set(1)
        return out

    def masked_repeat(x):
        out = jnp.zeros(1)
        for i in range(1):
            for r in range(2):
                if r:
                    continue
                for k in range(1):
                    a = 1
            out = out.at[i].set(1)
        return out

    def loop_in_a_scope(x):
        out = jnp.zeros(1)
        for i in range(1):
            with jax.named_scope("s"):
                for k in range(1):
                    a = 1
            out = out.at[i].set(1)
        return out

    for fn in (not_range, keywords, unpacking):
        assert "`for v in range(start, stop, step)`" in error(fn)
    assert "`else` block" in error(with_else)
    assert "nothing to run" in error(only_a_mask)
    nest = parse(two_reductions)  # reduction loops run one after another: phases
    assert [loop.var for loop in nest.reductions] == ["k", "k2"] and [len(s) for s in nest.segments] == [0, 0, 1]
    assert "repeat loops don't nest" in error(nested_repeats)
    assert "a repeat loop (around reduction loops) can't be masked" in error(masked_repeat)
    assert "loops nest only as parallel loops around reduction (and repeat) loops" in error(loop_in_a_scope)


def _tile(*lines: str) -> str:
    """The error for a nest whose tile runs `lines` before its store."""
    source = "\n".join(
        [
            "def f(x):",
            "    out = jnp.zeros((8, 128))",
            "    for i in range(0, 8, 8):",
            *(f"        {line}" for line in lines),
            "        out = out.at[i : i + 8, :].set(x[i : i + 8, :])",
            "    return out",
        ]
    )
    namespace = {"jnp": jnp, "jax": jax}
    code = compile(source, "<tile>", "exec")
    exec(code, namespace)
    fn = namespace["f"]
    import linecache

    linecache.cache["<tile>"] = (len(source), None, source.splitlines(keepends=True), "<tile>")
    return error(fn)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["a = b = x[i : i + 8, :]"], "tile code assigns names"),
        (["a = jnp.zeros(8)", "a[0] = 1"], "tile code assigns names"),
        (["a = jnp.zeros(8)", "a.b += 1"], "tile code augments names"),
        (["with jax.named_scope('s') as s:", "    pass"], "only `with jax.named_scope(name):`"),
        (["with open:", "    pass"], "only `with jax.named_scope(name):`"),
        (["if i:", "    a = 1"], "use jnp.where instead of `if`"),
        (["print(i)"], "use jnp.where instead of `if`"),
        (["a = (b := 1)"], "can't assign inside expressions"),
        (["a = x"], "`x` is an input"),
        (["x = jnp.zeros(8)"], "`x` is an input"),
        (["a = out"], "are written as `out = out.at[...].set(...)`"),
        (["i = 1"], "loop variable `i` can't be assigned"),
    ],
)
def test_tile_statements(lines, message):
    assert message in _tile(*lines)


def test_the_tunable_is_read_by_its_fields():
    def f(x, tiles=None):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            t = tiles
            out = out.at[i : i + 8, :].set(x[i : i + 8, :])
        return out

    assert "`tiles` is the kernel's tunable: tile code reads its fields" in error(f)


def test_stores():
    def store_in_pre(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(1.0)
            for k in range(1):
                a = 1
        return out

    def store_in_a_repeat_loop(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            a = 1
            for r in range(2):
                for k in range(1):
                    out = out.at[i : i + 8, :].set(1.0)
        return out

    def stored_twice(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(1.0)
            out = out.at[i : i + 8, :].set(2.0)
        return out

    def never_stored(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            a = 1
        return out

    def reads_an_output(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(out[i : i + 8, :])
        return out

    def reads_the_reduction_variable(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            for k in range(1):
                a = 1
            out = out.at[i : i + 8, :].set(k)
        return out

    assert "after the reduction loop" in error(store_in_pre)
    assert "in the body of a reduction loop outside repeat loops" in error(store_in_a_repeat_loop)
    assert "stored twice" in error(stored_twice)
    assert "no tile stores output `out`" in error(never_stored)
    assert "tiles can't be read" in error(reads_an_output)
    assert "this one reads `x`" in _tile("if x.shape[0] > 8:", "    continue")

    def reads_an_inner_loop(x):
        out = jnp.zeros((8, 128))
        for i in range(0, 8, 8):
            if j > 0:  # noqa: F821
                continue
            for j in range(0, 128, 128):
                out = out.at[i : i + 8, j : j + 128].set(1.0)
        return out

    def asserts_shapes(x):
        assert x.shape[0] == 8
        out = jnp.zeros((8, 128))
        pass
        for i in range(0, 8, 8):
            out = out.at[i : i + 8, :].set(1.0)
        return out

    assert "not `j`" in error(reads_an_inner_loop)
    assert parse(asserts_shapes).prologue[0].__class__ is ast.Assert
    assert "only defined inside its reduction loop" in error(reads_the_reduction_variable)


def test_name_analysis():
    stmts = ast.parse(
        "a = b + c\n"
        "c = a\n"
        "d += [e for e in c if f]\n"
        "g = lambda h, q=r: h + q + s\n"
        "with t():\n"
        "    u = {v: w for v in a}\n"
    ).body
    assert first_reads(stmts) == {"b", "c", "d", "f", "r", "s", "t", "w"}
    assert reads(ast.parse("{x for x in y}", mode="eval").body) == {"y"}
    assert reads(ast.parse("(z for z in y)", mode="eval").body) == {"y"}
