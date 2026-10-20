"""Programs: the corners of `program.py` the other tests don't reach, through the public API where it can get
there (Python control flow around and after nests, views between nests, loops that can't roll, static
arguments, lowering and tuning), and through its helpers where it can't (interval arithmetic, receivers)."""

import ast
import functools
import heapq
import textwrap
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand

import autokernel
from autokernel import AutoKernelError, program

M, N, T, L = 16, 128, 8, 2


def fdef_of(src: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(src)).body[0]


def regions_of(src: str) -> list:
    fdef = fdef_of(src)
    return program._find(fdef, fdef.name, 1)


# --------------------------------------------------------------------------- recognising single kernels and nests


def test_single_nest_form_rejects_other_prologues_and_epilogues():
    nest = """
        for i in range(0, 16, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :])
    """
    body = textwrap.indent(textwrap.dedent(nest), "    ")
    ok = "def f(x):\n    out = jnp.zeros((16, 128))\n" + body + "    return out\n"
    assert program.is_single_nest(fdef_of(ok))
    # An `if` in the prologue isn't a constant or an allocation.
    pro = "def f(x):\n    out = jnp.zeros((16, 128))\n    if x is None:\n        pass\n" + body + "    return out\n"
    assert not program.is_single_nest(fdef_of(pro))
    # What follows the nest isn't `return` of outputs.
    after = "def f(x):\n    out = jnp.zeros((16, 128))\n" + body + "    y = out\n"
    assert not program.is_single_nest(fdef_of(after))
    bare = "def f(x):\n    out = jnp.zeros((16, 128))\n" + body + "    return\n"
    assert not program.is_single_nest(fdef_of(bare))


def test_is_fill():
    expr = lambda s: ast.parse(s, mode="eval").body  # noqa: E731
    assert program._is_fill(expr("jnp.zeros((8, 8)).astype(jnp.bfloat16)"))
    assert program._is_fill(expr("full((8, 8), 1.0)"))
    assert not program._is_fill(expr("x"))  # not a call at all
    assert not program._is_fill(expr("x * 2"))
    assert not program._is_fill(expr("jnp.dot(x, y)"))


def test_a_loop_between_a_nests_loops_makes_it_ordinary_python():
    loop = fdef_of(
        """
        def f(x):
            for i in range(0, 16, 8):
                k = 0
                while k < 2:
                    k += 1
                for j in range(0, 128, 128):
                    out = out.at[i : i + 8, j : j + 128].set(x[i : i + 8, j : j + 128])
        """
    ).body[0]
    assert program._nest_stores(loop) == []


def test_blocks_of_compound_statements():
    fdef = fdef_of(
        """
        def f(x):
            with ctx() as c:
                a = 1
            try:
                b = 1
            except E:
                b = 2
            else:
                b = 3
            finally:
                b = 4
            match x:
                case 1:
                    d = 1
                case _:
                    d = 2
            e = 5
        """
    )
    w, t, m, e = fdef.body
    assert [[ast.unparse(s) for s in b] for b in program._blocks(w)] == [["a = 1"]]
    assert [[ast.unparse(s) for s in b] for b in program._blocks(t)] == [["b = 1"], ["b = 2"], ["b = 3"], ["b = 4"]]
    assert [[ast.unparse(s) for s in b] for b in program._blocks(m)] == [["d = 1"], ["d = 2"]]
    assert program._blocks(e) == []


def test_scope_loads_of_comprehensions_and_lambdas():
    def loads(src: str) -> list[str]:
        return sorted({n.id for n in program._scope_loads(ast.parse(src, mode="eval").body)})

    # The comprehension's own variables aren't read from the scope; its filters' other names are.
    assert loads("[a + b for a in xs if a > c for b in ys(a)]") == ["c", "xs", "ys"]
    assert loads("{k: v for k, v in d.items() if v}") == ["d"]
    # A lambda's defaults are read where it is written; its body reads the scope except its parameters.
    assert loads("lambda a, b=dflt, *, c=kw: a + b + c + z") == ["dflt", "kw", "z"]


# --------------------------------------------------------------------------- reads after a nest, through control flow

NEST_THEN = """
def f(x, flag):
    out = jnp.zeros((16, 128))
    for i in range(0, 16, 8):
        t = x[i : i + 8, :] * 2
        out = out.at[i : i + 8, :].set(t)
{after}
    return out
"""


@pytest.mark.parametrize(
    "after, name",
    [
        ("    while flag:\n        flag = i", "i"),
        ("    with ctx(t):\n        pass", "t"),
        ("    with ctx() as c:\n        y = c + i", "i"),
        ("    try:\n        y = 1\n    except E:\n        y = t\n    else:\n        y = 2", "t"),
        ("    try:\n        y = 1\n    finally:\n        y = i", "i"),
        ("    match flag:\n        case 1 if i:\n            pass", "i"),
        ("    match flag:\n        case 1:\n            y = t", "t"),
    ],
)
def test_reads_after_a_nest_through_control_flow(after, name):
    fdef = fdef_of(NEST_THEN.format(after=after))
    with pytest.raises(AutoKernelError, match=f"`{name}` is read here, but its value may come from the tile loop"):
        program._program_ast(fdef, "f", 1)


@pytest.mark.parametrize(
    "after",
    [
        "    with ctx() as i:\n        y = i",  # rebound by `as` before it is read
        "    try:\n        t = 1\n    except E:\n        pass\n    y = 2",
        "    match flag:\n        case 1:\n            y = 1",
    ],
)
def test_no_read_after_a_nest(after):
    program._program_ast(fdef_of(NEST_THEN.format(after=after)), "f", 1)


def test_read_after_a_nest_in_try_body_reaches_else_and_finally():
    src = """
    def f(x):
        try:
            out = jnp.zeros((16, 128))
            for i in range(0, 16, 8):
                out = out.at[i : i + 8, :].set(x[i : i + 8, :])
        except E:
            pass
        else:
            y = i
        return out
    """
    with pytest.raises(AutoKernelError, match=r"`i` is read here.*where it is a loop variable"):
        program._program_ast(fdef_of(src), "f", 1)


# --------------------------------------------------------------------------- view statements between nests


def test_view_stmt_forms():
    def view(src: str, local=frozenset({"x", "y", "s"})):
        return program._view_stmt(ast.parse(src).body[0], set(local))

    v = view("v = jnp.reshape(x, (4, 4))")  # the function form
    assert (v.target, v.base, v.reads) == ("v", "x", ("x",))
    assert view("v = jnp.swapaxes(x.T, 0, 1)[0]").base == "x"
    assert view("v = x.reshape(s)").reads == ("x", "s")
    assert view("v = x[1:]") is None  # a slice that isn't whole copies
    assert view("v = x.reshape(*s)") is None  # starred arguments
    assert view("v = jnp.reshape()") is None  # no array to view
    assert view("v = jnp.sum(x)") is None  # not a view function
    assert view("v = x.sum()") is None
    assert view("v = z.T") is None  # not a local
    assert view("v = x") is None  # no view operation
    assert view("v = (lambda: x)()") is None


# --------------------------------------------------------------------------- loops around nests: what they read and do


def stmts(src: str) -> list[ast.stmt]:
    return ast.parse(textwrap.dedent(src)).body


def test_escapes():
    assert program._escapes(stmts("return x"))
    assert program._escapes(stmts("del x"))
    assert program._escapes(stmts("if a:\n    break"))
    assert program._escapes(stmts("if a:\n    continue"))
    # Leaving an inner loop, or a nested function returning, doesn't leave this body.
    assert not program._escapes(stmts("for j in r:\n    break\ndef g():\n    return 1\nwhile a:\n    continue"))


def test_exposed_through_while_with_and_try():
    exposed = program._upward_exposed
    assert exposed(stmts("while a:\n    b = c\nb = 1")) == {"a", "c"}
    # `with` binds its `as` names, then its body runs.
    assert exposed(stmts("with ctx(m) as c:\n    d = c + e\ny = d")) == {"ctx", "m", "e"}
    # Nothing a `try` or `match` binds is bound for certain: every read in it may come first.
    assert exposed(stmts("try:\n    q = 1\nexcept E:\n    pass\ny = q")) == {"E", "q"}
    assert exposed(stmts("match s:\n    case 1:\n        q = 1\ny = q")) == {"s", "q"}


def test_chain_of_attributes_and_subscripts():
    expr = lambda s: ast.parse(s, mode="eval").body  # noqa: E731
    assert program._chain(expr("kv['k'].at[i]")) == ("kv", (("item", "k"), ("attr", "at"), ("item", None)))
    assert program._chain(expr("outs[-1].buf")) == ("outs", (("item", -1), ("attr", "buf")))
    assert program._chain(expr("outs[-1.5]")) == ("outs", (("item", None),))  # not an int key
    assert program._chain(expr("f().x")) is None


def test_lazy_loads():
    expr = lambda s: ast.parse(s, mode="eval").body  # noqa: E731
    # A generator's first iterable is evaluated where it is written; the rest when it runs.
    assert program._lazy_loads(expr("(a * h for a in xs if a > c)")) == {"h", "c"}
    assert program._lazy_loads(expr("lambda a=d: a + h")) == {"h"}


def test_side_effects():
    def why(src):
        return program._side_effect(stmts(src), lambda n: n.lineno)

    assert "`obj.n` is assigned" in why("obj.n = 1")
    assert "`print(...)` has an effect" in why("print(x)")
    assert "`cache().append(...)` may change" in why("cache().append(x)")
    assert why("y = x + 1") is None


# --------------------------------------------------------------------------- interval arithmetic of runtime scalars


def test_interval_ops():
    op = program._interval_op
    assert op(ast.Add(), (2, 2), (3, 3)) == (5, 5)
    assert op(ast.Add(), (0, 3), (1, 2)) == (1, 5)
    assert op(ast.Sub(), (0, 3), (1, 2)) == (-2, 2)
    assert op(ast.Mult(), (-2, 3), (4, 5)) == (-10, 15)
    assert op(ast.FloorDiv(), (-8, 8), (2, 4)) == (-4, 4)
    assert op(ast.LShift(), (1, 2), (0, 3)) == (1, 16)
    assert op(ast.RShift(), (16, 32), (1, 2)) == (4, 16)
    assert op(ast.Mod(), (-50, 50), (3, 7)) == (0, 6)
    assert op(ast.Mod(), (-50, 50), (-7, -3)) == (-6, 0)
    assert op(ast.Pow(), (-2, 3), (2, 3)) == (-8, 27)
    assert op(ast.Pow(), (1, 3), (0, 1)) == (1, 3)
    assert op(ast.BitAnd(), (0, 5), (0, 12)) == (0, 5)
    assert op(ast.BitOr(), (0, 5), (0, 12)) == (0, 31)
    assert op(ast.BitXor(), (-5, 5), (0, 12)) == (-32, 31)
    for bad, a, b in [
        (ast.Pow(), (2, 2), (65, 65)),
        (ast.LShift(), (1, 1), (-1, -1)),
        (ast.FloorDiv(), (1, 4), (-1, 1)),
        (ast.LShift(), (1, 4), (-1, 1)),
        (ast.LShift(), (1, 4), (0, 65)),
        (ast.RShift(), (1, 4), (-1, 1)),
        (ast.Mod(), (1, 4), (-1, 1)),
        (ast.Pow(), (1, 4), (-1, 1)),
        (ast.Pow(), (1, 4), (0, 65)),
    ]:
        with pytest.raises(OverflowError):
            op(bad, a, b)


def int_arith(src: str) -> program._IntArith:
    loop = stmts(src)[0]
    return program._int_arith(program._Region("f", [], [], loop, [], [], frozenset()))


def test_int_arith_temporaries_and_unary_ops():
    arith = int_arith(
        """
        for i in range(0, 16, 8):
            k = -i
            m = k * 2
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * (m * n) + (-n))
        """
    )
    assert set(arith.temps) == {"k", "m"}
    assert arith.fits("n", {"n": 3}.get)
    assert not arith.fits("n", {"n": 2**30}.get)  # m * n leaves int32
    neg = int_arith(
        """
        for i in range(0, 16, 8):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * (~n) * (-n) * (+n))
        """
    )
    assert neg.fits("n", {"n": 3}.get)
    assert not neg.fits("n", {"n": -(2**31)}.get)  # ~n fits in int32, -n doesn't
    assert not neg.fits("n", {"n": 2**31}.get)


def test_int_arith_loop_bounds():
    def fits(bounds, values, expr="i * n"):
        src = f"""
        for i in range({bounds}):
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * ({expr}))
        """
        return int_arith(src).fits("n", {"n": 4096, **values}.get)

    shaped = jax.ShapeDtypeStruct((2**21, 8), jnp.float32)
    assert fits("0, x.shape[0], 8", {"x": jax.ShapeDtypeStruct((16, 8), jnp.float32)})
    assert not fits("0, x.shape[0], 8", {"x": shaped})  # i * n reaches 2**31
    assert not fits("0, x.shape[k], 8", {"x": shaped})  # an index it can't evaluate: assume the worst
    assert not fits("0, y.shape[0], 8", {"x": shaped})  # an unbound name
    assert not fits("0, 16, s", {"s": 0})  # a zero step
    assert not fits("0, 16, s", {"s": jnp.int32(2)})  # an array step: no value known
    assert fits("16, 0, -8", {})
    assert not fits("0, -f(), 8", {})  # unary of something it can't evaluate
    assert not fits("0, f() + 1, 8", {})
    assert not fits("0, (1, 2), 8", {})


def test_int_arith_unknown_ranges():
    # A loop variable also assigned in its body has no single range.
    arith = int_arith(
        """
        for i in range(0, 16, 8):
            i = i + 1
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * (i * n))
        """
    )
    assert "i" in arith.unknown and "i" not in arith.ranges
    assert not arith.fits("n", {"n": 3}.get)
    # A loop bound defined through the loop's own variable has no bound.
    arith = int_arith(
        """
        for i in range(0, 16, 8):
            for j in range(0, k, 8):
                k = j + 1
                out = out.at[i : i + 8, j : j + 8].set(x[i : i + 8, j : j + 8] * (j * n))
        """
    )
    assert not arith.fits("n", {"n": 3}.get)


def test_int_arith_temporaries_of_the_scalar():
    # `p` is computed from `n`: its own arithmetic carries `n`'s value.
    arith = int_arith(
        """
        for i in range(0, 16, 8):
            p = n + 1
            out = out.at[i : i + 8, :].set(x[i : i + 8, :] * (p * i))
        """
    )
    assert arith.fits("n", {"n": 3}.get)
    assert not arith.fits("n", {"n": 2**29}.get)  # (n + 1) * 8 leaves int32


def test_uses_of_a_name_in_a_computed_index():
    (region,) = regions_of(
        """
        def f(x, w, ids, layer):
            out = jnp.zeros((16, 128))
            for i in range(0, 16, 8):
                out = out.at[i : i + 8, :].set(w[ids[layer], i : i + 8, :] + w[layer, i : i + 8, :])
            return out
        """
    )
    uses = program._uses(region)
    # `layer` indexes `w` directly (a runtime scalar can), but also looks up `ids` (it needs its value).
    assert uses["layer"] == {("index", "w"), ("static",)}
    assert uses["ids"] == {("static",)}


def test_is_int():
    assert program._is_int(3) and program._is_int(np.int8(3)) and program._is_int(jnp.int32(3))
    assert not program._is_int(True) and not program._is_int(np.bool_(True))
    assert not program._is_int(3.0) and not program._is_int(jnp.zeros((2,), jnp.int32))


# --------------------------------------------------------------------------- what a rolled loop's calls may change


def test_changes_state():
    rng = np.random.default_rng(0)
    assert program._changes_state(functools.partial(np.random.normal, 0.0)) is not None
    assert "RandomState" in program._changes_state(np.random.normal)
    assert "heapq`)" in program._changes_state(heapq.heappush)
    assert "(`numpy`)" in program._changes_state(np.copyto)
    assert "a method of a Python list" in program._changes_state([].append)
    assert program._changes_state("abc".upper) is None  # a method of an inert value
    assert program._changes_state({}.get) is None  # a container's reader
    assert "Generator" in program._changes_state(rng.normal)
    assert program._changes_state(jnp.tanh) is None

    class Holder:
        pass

    weird = Holder()
    weird.__self__ = [1]  # has a `__self__` but isn't a method
    assert program._changes_state(weird) is None


def test_reach():
    unknown = program._UNKNOWN
    cfg = {"a": [1, 2], "b": jnp.zeros(3)}
    assert program._reach(cfg, (("item", "a"),)) == [[1, 2]]
    assert program._reach(cfg, (("item", None),))[0] == [1, 2]  # every value of a mapping
    assert program._reach([4, 5], (("item", None),)) == [4, 5]  # every element of a list
    assert program._reach(object(), (("item", None),)) == [unknown]  # can't tell
    assert program._reach(cfg, (("item", "zz"),)) == [unknown]  # a lookup that fails
    assert program._reach(cfg, (("item", "b"), ("attr", "at"))) == []  # past an array: nothing
    assert program._reach(object(), (("item", None), ("attr", "x"))) == [unknown]
    assert program._reach(list(range(2000)), (("item", None),)) == [unknown]  # too many to look at


def test_receiver_problem():
    problem = program._receiver_problem
    unknown = program._UNKNOWN
    assert problem(jnp.zeros(2), "x.at", "at", False) is None
    assert "calls `f().g(...)` on a value it can't look at" in problem(unknown, "f().g", "g", True)
    assert "takes `f().g` on a value" in problem(unknown, "f().g", "g", False)
    assert problem(len, "len", None, True) is None
    assert "it calls `push(...)`" in problem([].append, "push", None, True)

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError(name)

    # An attribute it can't get: not a method of the object, and not called.
    assert problem(Broken(), "b.x", "x", False) is None
    assert problem(np.random, "np.random.seed", "seed", True).startswith("it calls `np.random.seed(...)`")
    assert problem(np, "np.tanh", "tanh", True) is None

    class Cfg:
        act = staticmethod(jnp.tanh)
        log = [].append

    assert problem(Cfg(), "cfg.act", "act", True) is None  # a value it holds, pure
    assert "it calls `cfg.log(...)`" in problem(Cfg(), "cfg.log", "log", True)
    assert problem((1, 2), "t.count", "count", True) is None
    outs = []
    assert "on a Python list" in problem(outs, "outs.append", "append", True)
    assert "it takes `outs.append`, a method" in problem(outs, "outs.append", "append", False)

    class Model:
        def apply(self):
            pass

    assert problem(Model(), "model.apply", "apply", False) is None


def unbound_cell():
    def body(layer, carry):
        return later(carry)

    cell = body
    if cell is None:  # never: `later` stays an empty cell
        later = None
    return body


def test_empty_closure_cells():
    body = unbound_cell()
    assert program._closure_values(body) == {}
    assert program._python_values(body, (), (), (("later", "later", (), None, True),)) is None
    # A carried Python int can't be a traced value; a carried float can, unless it is read after the loop.
    assert "`n`, a Python int" in program._python_values(body, (1,), ("n",), ())
    assert program._python_values(body, (1.0,), ("s",), ()) is None
    assert "`s`, a Python float" in program._python_values(body, (1.0,), ("s",), (), after=("s",))


def test_kernel_names():
    nest = type("Nest", (), {"name": "f_h"})()
    a, b = ((), (), (("n", int, 1),)), ((), ("n",), (("n", int, 1),))
    assert program._kernel_name(nest, a, [a]) == "f_h"
    assert program._kernel_name(nest, b, [a, b]) == "f_h[1]"  # no constant tells them apart
    c = ((), (), (("n", int, 2),))
    assert program._kernel_name(nest, c, [a, c]) == "f_h[n=2]"
    assert program._unique_names([("k", 1, ()), ("k", 2, ()), ("j", 3, ())]) == [
        ("k#1", 1, ()),
        ("k#2", 2, ()),
        ("j", 3, ()),
    ]


# --------------------------------------------------------------------------- programs: views between nests


def view_then_nest(x):
    xr = x.reshape(M, N)  # a view: the nest reads windows of `x` itself
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(xr[i : i + T, :] * 2.0)
    return out


def test_a_nest_reading_a_view_differentiates_and_batches():
    prog = autokernel.tpu(view_then_nest)
    x = rand((2, M // 2, N), 1)
    np.testing.assert_allclose(prog(x), 2 * x.reshape(M, N), rtol=1e-6)
    (call,) = [nest for nest in prog.kernels]
    assert repr(call) == f"<ProgramNest view_then_nest_out (line {call.line}): out>"
    assert repr(prog.groups[0]) == "<ProgramGroup view_then_nest_out>"
    # Under jax.grad the nest's Python loops are differentiated; under jax.vmap it runs on the view's array.
    grad = jax.grad(lambda x: (prog(x) * x.reshape(M, N)).sum())(x)
    np.testing.assert_allclose(grad, 4 * x, rtol=1e-5)
    xs = jnp.stack([x, -x])
    np.testing.assert_allclose(jax.vmap(prog)(xs), 2 * xs.reshape(2, M, N), rtol=1e-6)


def nest_then_view(x):
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(x[i : i + T, :] + 1.0)
    t = out.T  # after the last nest: plain JAX, no group
    return t


def test_a_view_after_the_last_nest_is_plain_jax():
    prog = autokernel.tpu(nest_then_view)
    assert prog.groups == []
    x = rand((M, N), 2)
    np.testing.assert_allclose(prog(x), (x + 1).T, rtol=1e-6)


def glue_between(x, z):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 3.0)
    h2 = h.reshape(z.shape)  # reads another array: JAX code, not a view
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(h2[i : i + T, :] + z[i : i + T, :])
    return out


@pytest.mark.parametrize("fuse", [True, False])
def test_glue_between_nests_cuts_the_run(fuse):
    prog = autokernel.tpu(glue_between, fuse=fuse)
    x, z = rand((M, N), 3), rand((M, N), 4)
    np.testing.assert_allclose(prog(x, z), 3 * x + z, rtol=1e-5, atol=1e-6)
    lowered = prog.lower(x, z)
    if fuse:
        (cut,) = lowered.cuts
        assert cut.startswith("cut before `glue_between_out`") and "(`h2` is computed from `h` by JAX code)" in cut
    assert lowered.names == ["glue_between_h", "glue_between_out"]
    assert lowered["glue_between_out"] is lowered[1][1]
    with pytest.raises(KeyError, match="no kernel named 'nope'; the kernels are"):
        lowered["nope"]
    # The kernels that aren't fused have their own tuning candidates.
    assert set(prog.tuning(x, z)) == {"glue_between_h", "glue_between_out"}


def fused_view_of_input(y, x3):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(y[i : i + T, :] * 2.0)
    xr = x3.reshape(M, N)  # a view of an input, read in place by the fused kernel
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(h[i : i + T, :] + xr[i : i + T, :])
    return out


def test_a_fused_kernel_reads_a_view_of_an_input_in_place():
    prog = autokernel.tpu(fused_view_of_input)
    y, x3 = rand((M, N), 5), rand((2, M // 2, N), 6)
    np.testing.assert_allclose(prog(y, x3), 2 * y + x3.reshape(M, N), rtol=1e-5, atol=1e-6)
    lowered = prog.lower(y, x3)
    assert lowered.names == ["fused_view_of_input_h+fused_view_of_input_out"]
    # The fused kernel takes `x3` itself, not a copy of its reshape.
    assert [tuple(s.shape) for s in lowered.inputs[0]] == [(M, N), (2, M // 2, N)]
    assert lowered["fused_view_of_input_out"] is lowered[0][1].parts[1][1]


def colliding_input_name(x, z, _in1_x2):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * _in1_x2[i : i + T, :])
    x2 = x.reshape(z.shape)  # JAX code on inputs only: the fused kernel takes its value
    out = jnp.zeros((M, N))
    for i in range(0, M, T):
        out = out.at[i : i + T, :].set(h[i : i + T, :] + x2[i : i + T, :])
    return out


def test_a_fused_kernels_input_named_apart_from_the_programs_names():
    prog = autokernel.tpu(colliding_input_name)
    x, z, w = rand((M, N), 7), rand((M, N), 8), rand((M, N), 9)
    np.testing.assert_allclose(prog(x, z, w), x * w + x, rtol=1e-5, atol=1e-6)
    (fused,) = [k for _, k in prog.lower(x, z, w)]
    assert "_in1_x2_" in fused.inputs and "_in1_x2" in fused.inputs


# --------------------------------------------------------------------------- programs: Python loops around nests


def over_a_tuple(a, b):
    total = 0.0
    for x in (a, b):  # not a `range` loop: unrolled, the same nest at two shapes
        h = jnp.zeros(x.shape)
        for i in range(0, x.shape[0], T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
        total = total + h.sum()
    return total


def test_a_loop_over_a_tuple_unrolls_and_names_kernels_apart():
    prog = autokernel.tpu(over_a_tuple)
    assert len(prog.unrolled) == 1 and "it isn't a `for v in range(...)` loop" in prog.unrolled[0]
    a, b = rand((M, N), 1), rand((2 * M, N), 2)
    np.testing.assert_allclose(prog(a, b), 2 * (a.sum() + b.sum()), rtol=1e-4)
    assert prog.lower(a, b).names == ["over_a_tuple_h#1", "over_a_tuple_h#2"]


def returns_early(x, w):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        if layer == 1:
            return h
        x = h
    return x


def reassigns_its_variable(x, w):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = h
    y = x * layer
    return y


def carries_a_name_with_no_value(x, w):
    for layer in range(L):
        try:
            x = x + 0.0
        finally:
            pass
        if layer > 0:
            x = x + prev  # noqa: F821
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        prev = h  # noqa: F841 (read on the next iteration)
        x = h
    return x


@pytest.mark.parametrize(
    "fn, why",
    [
        (returns_early, "control leaves its body early"),
        (reassigns_its_variable, "its body assigns `layer`, or `layer` is read after it"),
        (carries_a_name_with_no_value, "`prev` changes in it but may have no value before it"),
    ],
)
def test_loops_that_dont_roll(fn, why):
    prog = autokernel.tpu(fn)
    assert len(prog.unrolled) == 1 and why in prog.unrolled[0]
    x, w = rand((M, N), 1), rand((L, M, N), 2)
    xn, wn = np.asarray(x), np.asarray(w)
    if fn is returns_early:
        want = xn * wn[0] * wn[1]
    elif fn is reassigns_its_variable:
        want = xn * wn[0] * wn[1] * (L - 1)
    else:
        h0 = xn * wn[0]
        want = (h0 + h0) * wn[1]
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-6)


def two_rolled_loops(x, w):
    for layer in range(L):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :])
        x = h
    for step in range(L):  # the first loop's body function is defined before this one: not a closure of it
        g = jnp.zeros((M, N))
        for i in range(0, M, T):
            g = g.at[i : i + T, :].set(x[i : i + T, :] + w[step, i : i + T, :])
        x = g
    return x


def test_consecutive_loops_both_roll():
    prog = autokernel.tpu(two_rolled_loops)
    assert prog.unrolled == []
    x, w = rand((M, N), 1), rand((L, M, N), 2)
    xn, wn = np.asarray(x), np.asarray(w)
    np.testing.assert_allclose(prog(x, w), xn * wn[0] * wn[1] + wn[0] + wn[1], rtol=1e-5, atol=1e-5)
    prims = [e.primitive.name for e in jax.make_jaxpr(prog._program_fn())(x, w).jaxpr.eqns]
    assert sum(p in ("scan", "while") for p in prims) == 2


class Ctx:
    def __enter__(self):
        return 2.0

    def __exit__(self, *exc):
        return False


def loops_in_with_and_try(x, w):
    with Ctx() as scale:
        for layer in range(L):
            h = jnp.zeros((M, N))
            for i in range(0, M, T):
                h = h.at[i : i + T, :].set(x[i : i + T, :] * w[layer, i : i + T, :] * scale)
            x = h
    try:
        for step in range(L):
            g = jnp.zeros((M, N))
            for i in range(0, M, T):
                g = g.at[i : i + T, :].set(x[i : i + T, :] + w[step, i : i + T, :])
            x = g
    except ValueError as err:
        raise RuntimeError("unreachable") from err
    finally:
        x = x + 1.0
    return x


def test_loops_inside_with_and_try_roll():
    prog = autokernel.tpu(loops_in_with_and_try)
    assert prog.unrolled == []
    x, w = rand((M, N), 1), rand((L, M, N), 2)
    xn, wn = np.asarray(x), np.asarray(w)
    want = xn * wn[0] * wn[1] * 4 + wn[0] + wn[1] + 1
    np.testing.assert_allclose(prog(x, w), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prog._reference_fn()(x, w), want, rtol=1e-5, atol=1e-5)


def scaled_layers(x):
    for s in (1, 2):  # unrolled: `s` sets the nest's tile, a constant of its kernel
        h = jnp.zeros((M, N))
        for i in range(0, M, T * s):
            h = h.at[i : i + T * s, :].set(x[i : i + T * s, :] * 2.0)
        out = jnp.zeros((M, N))
        for i in range(0, M, T):
            out = out.at[i : i + T, :].set(h[i : i + T, :] + 1.0)
        x = out
    return x


def test_fused_kernels_of_nests_with_several_kernels_are_named_apart():
    prog = autokernel.tpu(scaled_layers)
    x = rand((M, N), 3)
    np.testing.assert_allclose(prog(x), (x * 2 + 1) * 2 + 1, rtol=1e-5)
    lowered = prog.lower(x)
    assert lowered.names == ["scaled_layers_h[s=1]+scaled_layers_out", "scaled_layers_h[s=2]+scaled_layers_out"]
    assert [n for n, _ in lowered[1][1].parts] == ["scaled_layers_h[s=2]", "scaled_layers_out"]


# --------------------------------------------------------------------------- programs: constants and globals

TILES = [8, 8, 8, 8, 8, 8, 8, 8]
G = jnp.ones((M, N))
counter = 0


def reads_globals(x):
    global counter
    counter += 1
    h = jnp.zeros((M, N))
    for i in range(0, G.shape[0], len(TILES)):  # an array's shape, and a list's length
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
    return h + 1.0


def test_globals_read_by_a_nest():
    prog = autokernel.tpu(reads_globals)
    x = rand((M, N), 4)
    calls = counter
    np.testing.assert_allclose(prog(x), 2 * x + 1, rtol=1e-6)
    assert counter == calls + 1  # it assigns a global: it sees the module's globals, live
    (nest,) = prog.kernels
    assert {"G", "TILES"} <= set(nest.outer_names)
    # Outside any input signature, an unhashable global (`TILES`) has no identity to key kernels by.
    consts = {name for name, _, _ in nest._outer_consts()}
    assert "M" in consts and "TILES" not in consts and "G" not in consts


def body_needs_the_int(x, n):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] + jnp.zeros((T, n)))  # a shape: needs `n`'s value
    return h + 1.0


def test_an_int_the_body_needs_is_compiled_in():
    prog = autokernel.tpu(body_needs_the_int)
    x = rand((M, N), 5)
    np.testing.assert_allclose(prog(x, N), x + 1, rtol=1e-6)
    assert prog.kernels[0].compiled_in == {"n"}


def with_options(x, opts):
    h = jnp.zeros((M, N))
    for i in range(0, M, T):
        h = h.at[i : i + T, :].set(x[i : i + T, :] * 2.0)
    return h + len(opts)


def test_an_unhashable_static_argument():
    prog = autokernel.tpu(with_options)
    x = rand((M, N), 6)
    # The reference runs with it (it isn't a key of anything there); the program can't compile per value.
    np.testing.assert_allclose(prog.reference(x, {1, 2}), 2 * x + 2, rtol=1e-6)
    with pytest.raises(AutoKernelError, match=r"a static argument .* must be hashable"):
        prog(x, {1, 2})


def make_closure_program():
    def uses_scale(x):
        h = jnp.zeros((M, N))
        for i in range(0, M, T):
            h = h.at[i : i + T, :].set(x[i : i + T, :] * scale)
        return h + 1.0

    prog = autokernel.tpu(uses_scale)
    before = prog._outer_value("scale", live=True)
    scale = 3.0
    return prog, before


def test_a_closure_variable_bound_after_decoration():
    prog, before = make_closure_program()
    assert before is None  # an empty cell
    x = rand((M, N), 7)
    np.testing.assert_allclose(prog(x), 3 * x + 1, rtol=1e-6)


def test_check_explains_a_disagreement():
    prog = autokernel.tpu(nest_then_view)
    x = rand((M, N), 8)
    got = prog(x)
    # Every nest agrees with its Python loops: the difference builds up across them.
    with pytest.raises(AssertionError, match="every nest's kernel agrees with its Python loops"):
        prog._check(got + 1.0, got, (x,), {})
    # The diagnosis itself fails: the original error is kept.
    with pytest.raises(AssertionError, match="couldn't check the nests one by one: TypeError"):
        prog._check(got + 1.0, got, (x, x), {})


def test_measured_tuning_runs_the_first_call_eagerly(monkeypatch):
    from autokernel import hardware

    prog = autokernel.tpu(glue_between, tune={"T": (8, 16)}, autotune="measure")
    # The program thinks it is on a TPU (its kernels know better: they don't time anything here).
    monkeypatch.setattr(program, "hardware", types.SimpleNamespace(on_tpu=lambda: True, target=hardware.target))
    x, z = rand((M, N), 3), rand((M, N), 4)
    np.testing.assert_allclose(prog(x, z), 3 * x + z, rtol=1e-5, atol=1e-6)
    assert all(s.jitted is None for s in prog._snapshots.values())  # traced again, with the winners
    np.testing.assert_allclose(prog(x, z), 3 * x + z, rtol=1e-5, atol=1e-6)
