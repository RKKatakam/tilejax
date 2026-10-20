"""Corners of code generation, views, cost, fusion, schedules and hardware the other tests don't reach:
narrow SMEM scalars, broadcast stores, runtime indices into views, runtime masks over several reduction
phases, user functions spread over modules, the view tracer's refusals and window errors, and helpers."""

import dataclasses
import importlib.util
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from jax.experimental.pallas import tpu as pltpu

import autokernel
from autokernel import codegen, cost, fusion, hardware, schedule, views
from autokernel.analysis import specialize
from autokernel.errors import AutoKernelError

T = 128
INTERPRET = pltpu.InterpretParams()


def _specs(*args):
    return [jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)) for a in args]


# --------------------------------------------------------------------------- narrow SMEM state


def running_bf16_max(x):
    out = jnp.zeros((256, T))
    for i in range(0, 256, T):
        m = jnp.bfloat16(0)
        acc = jnp.zeros((T, T))
        for k in range(0, 256, T):
            t = x[i : i + T, k : k + T]
            m = jnp.maximum(m, jnp.max(t).astype(jnp.bfloat16))
            acc += t
        out = out.at[i : i + T, :].set(acc * m.astype(jnp.float32))
    return out


def test_a_bf16_scalar_carried_between_steps_lives_in_a_32_bit_smem_word():
    x = rand((256, 256))
    lowered = autokernel.tpu(running_bf16_max).lower(x)
    assert "pltpu.SMEM((1,), jnp.float32)" in lowered.source  # its scratch: a 32-bit word
    assert "m_ref[0].astype(jnp.bfloat16)" in lowered.source  # read back as a bf16
    assert "jnp.max(t).astype(jnp.bfloat16)), jnp.float32)" in lowered.source  # stored widened
    got = lowered.entry(x, interpret=INTERPRET)
    xn = np.asarray(x)
    want = []
    for i in (0, T):
        rows = xn[i : i + T]
        m = float(jnp.bfloat16(max(0.0, rows.max())))
        want.append((rows[:, :T] + rows[:, T:]) * m)
    np.testing.assert_allclose(got, np.concatenate(want), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- stores and view indices


def column_sums(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T].sum(axis=0, keepdims=True))
    return out


def test_a_stored_row_is_broadcast_to_its_tile():
    x = rand((256, 256))
    lowered = autokernel.tpu(column_sums).lower(x)
    assert "out_ref[...] = jnp.broadcast_to(" in lowered.source
    xn = np.asarray(x).reshape(2, T, 2, T).sum(axis=1, keepdims=True)  # per (i, j) tile: its column sums
    want = np.broadcast_to(xn, (2, T, 2, T)).reshape(256, 256)
    np.testing.assert_allclose(lowered.entry(x, interpret=INTERPRET), want, rtol=1e-5, atol=1e-4)


def layer_of_a_view(stack, w, layer):
    x = stack.transpose(1, 0, 2)  # (layer, row, col): a view of `stack`
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            out = out.at[i : i + T, j : j + T].set(x[layer, i : i + T, j : j + T] * w[i : i + T, j : j + T])
    return out


def test_a_runtime_index_into_a_view_is_a_window_of_its_base():
    stack, w = rand((256, 3, 256)), rand((256, 256), 1)
    prog = autokernel.tpu(layer_of_a_view)
    lowered = prog.lower(stack, w, 2)
    assert "x_hbm.at[pl.ds(i_blk * 128, 128), x_index0, pl.ds(j_blk * 128, 128)]" in lowered.source
    for layer in (0, 2):
        np.testing.assert_allclose(prog(stack, w, layer), stack[:, layer] * w, rtol=1e-6)


# --------------------------------------------------------------------------- runtime masks


def four_phases(x, y, keep):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            acc = jnp.zeros((T, T))
            for k in range(0, 256, T):  # a mask reading an input
                if keep[i // T, k // T] == 0:
                    continue
                acc += x[i : i + T, k : k + T] @ y[k : k + T, j : j + T]
            s = jnp.zeros((T, T))
            for k2 in range(0, 256, T):  # a static mask
                if k2 > i:
                    continue
                s += x[i : i + T, k2 : k2 + T]
            u = jnp.zeros((T, T))
            for k3 in range(0, 256, T):  # an input read at the first phase's last `k`, and no loop of its own
                if keep[k // T, 0] == 0:
                    continue
                u += y[k3 : k3 + T, j : j + T]
            v = jnp.zeros((T, T))
            for k4 in range(0, 256, T):  # no mask
                v += y[k4 : k4 + T, j : j + T]
            out = out.at[i : i + T, j : j + T].set(acc + s + u + v)
    return out


def four_phases_reference(x, y, keep):
    x, y, keep = map(np.asarray, (x, y, keep))
    out = np.zeros((256, 256), np.float32)
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            tile = sum(
                x[i : i + T, k : k + T] @ y[k : k + T, j : j + T] for k in range(0, 256, T) if keep[i // T, k // T]
            )
            tile = tile + sum(x[i : i + T, k : k + T] for k in range(0, 256, T) if k <= i)
            if keep[1, 0]:  # the first phase's last `k` is 128
                tile = tile + y[:T, j : j + T] + y[T:, j : j + T]
            out[i : i + T, j : j + T] = tile + y[:T, j : j + T] + y[T:, j : j + T]
    return out


@pytest.mark.parametrize("keep", [[[1, 0], [1, 1]], [[0, 1], [0, 0]]])
def test_runtime_masks_over_phases_with_static_masks_and_none(keep):
    x, y = rand((256, 256)), rand((256, 256), 1)
    keep = jnp.asarray(keep, jnp.int32)
    lowered = autokernel.tpu(four_phases).lower(x, y, keep)
    assert "jnp.asarray(phase1_mask)" in lowered.source  # the static mask's value
    assert "(lambda k=128: " in lowered.source  # the third phase's mask reads only the finished `k`
    assert "keep2, None], [2, 2, 2, 2], 4, 1)" in lowered.source  # the last phase has no mask
    got = lowered.entry(x, y, keep, interpret=INTERPRET)
    np.testing.assert_allclose(got, four_phases_reference(x, y, keep), rtol=1e-4, atol=1e-3)

    # The schedule module's own tables (as the generated wrapper builds them) agree.
    spec = specialize(autokernel.tpu(four_phases).nest, four_phases, _specs(x, y, keep))
    sched = schedule.plan(spec, 1, hardware.target("v5e").smem_budget)
    tables = schedule.runtime_tables(spec, sched, {"x": x, "y": y, "keep": keep})
    count = int(tables["bounds"][1])
    assert 0 < count <= sched.steps


def flagged(x, flag):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            if flag[0] == 0:  # reads an input, but no loop variable
                continue
            out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    return out


@pytest.mark.parametrize("on", [0, 1])
def test_a_tile_mask_reading_no_loop_variable(on):
    x, flag = rand((256, 256)), jnp.asarray([on], jnp.int32)
    lowered = autokernel.tpu(flagged).lower(x, flag)
    assert "tile_keep = jnp.broadcast_to(jnp.asarray(xp.logical_not(flag[0] == 0)).astype(bool), (4,))" in (
        lowered.source
    )
    got = np.asarray(autokernel.tpu(flagged)(x, flag))
    np.testing.assert_allclose(got, np.asarray(x) * 2 if on else np.zeros((256, 256)), rtol=1e-6)

    spec = specialize(autokernel.tpu(flagged).nest, flagged, _specs(x, flag))
    sched = schedule.plan(spec, 1, hardware.target("v5e").smem_budget)
    tables = schedule.runtime_tables(spec, sched, {"x": x, "flag": flag})  # no phases: one column per tile
    assert int(tables["bounds"][1]) == (4 if on else 0)
    assert np.asarray(tables["step_tile"])[: 4 if on else 0].tolist() == ([0, 1, 2, 3] if on else [])


# --------------------------------------------------------------------------- user functions over modules


KERNEL_MODULE = """
import jax.numpy as jnp

T = 128
SCALE = 2.0
SENTINEL = object()


def base(t):
    return t * SCALE


alias = base


def h1(t):
    return base(t) + SCALE


def h2(t):
    if SENTINEL is None:
        return t
    return alias(t) + SCALE


def back(t):
    return t + 1.0


def h3(t):
    return other(t)


def f(x):
    out = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            t = x[i:i + T, j:j + T]
            out = out.at[i:i + T, j:j + T].set(h1(t) + h2(t) + h3(t))
    return out
"""

OTHER_MODULE = """
def other(t):
    return cb(t) * 3.0
"""


def _load(tmp_path, src, name):
    """A module from a file, not importable by name (so the kernel source carries its functions)."""
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_functions_shared_aliased_sourceless_and_calling_back_across_modules(tmp_path):
    a = _load(tmp_path, KERNEL_MODULE, "akcov_kernel")
    b = _load(tmp_path, OTHER_MODULE, "akcov_other")
    a.other, b.cb = b.other, a.back  # the two modules call each other's functions
    x = rand((256, 256))
    lowered = autokernel.tpu(a.f).lower(x)
    src = lowered.source
    assert src.count("def base(t):") == 1  # read by `h1` and (as `alias`) by `h2`: defined once
    assert "alias = base" in src
    assert "# `SENTINEL` (a object) has no source here: define it to run this alone" in src
    assert "# `cb`: its module calls back into this one: define it to run this alone" in src
    assert "_akcov_kernel_functions(SCALE=2.0, other=_other)" in src  # `SCALE` passed once
    xn = np.asarray(x)
    want = (2 * xn + 2) + (2 * xn + 2) + (xn + 1) * 3
    np.testing.assert_allclose(lowered.entry(x, interpret=INTERPRET), want, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- source helpers


def plain(t):
    return t * 2


def uses_global(t):
    global T
    return t * T


_REGISTRY = []


def _register(fn):
    _REGISTRY.append(fn)
    return fn


@_register
def registered(t: jax.Array) -> jax.Array:
    return t + 1


def test_literals_of_containers():
    assert codegen._literal({1: 2.5, "a": (1,)}) == "{1: 2.5, 'a': (1,)}"
    assert codegen._literal({1: object()}) is None
    assert codegen._literal({object(): 1}) is None
    assert codegen._literal([1, object()]) is None
    assert codegen._literal(frozenset([3])) == "frozenset([3])"


def test_functions_that_have_no_source_of_their_own(monkeypatch):
    assert codegen._function_source(plain).source.startswith("def plain(t):")
    # Its module binds its name to something else.
    original = plain
    monkeypatch.setitem(globals(), "plain", 3)
    assert codegen._function_source(original) is None
    monkeypatch.undo()
    # Defined by `exec`: no source file.
    namespace = {}
    exec("def made(t):\n    return t\n", namespace)
    assert codegen._function_source(namespace["made"]) is None
    # Its source defines another name.
    renamed = types.FunctionType(plain.__code__, plain.__globals__, "renamed_copy")
    assert codegen._function_source(renamed) is None
    # A `global` statement.
    assert codegen._function_source(uses_global) is None


def test_a_decorator_that_returns_the_function_itself_is_dropped():
    fn = codegen._function_source(registered)
    assert fn.source == "def registered(t):\n    return t + 1"  # no decorator, no annotations
    assert fn.reads == {}


# --------------------------------------------------------------------------- views: what isn't one


F32 = jnp.float32


@pytest.mark.parametrize(
    "fn, shape",
    [
        (lambda a: a.reshape(7, 7), (256, 128)),  # doesn't trace
        (lambda a: (a, a), (256, 128)),  # two outputs
        (lambda a: jnp.float32(1), (256, 128)),  # not computed from the base
        (lambda a: jnp.negative(a), (256, 128)),  # arithmetic
        (lambda a: a[::2], (256, 128)),  # a strided slice
        (lambda a: jax.lax.reshape(a, (128, 256), dimensions=(1, 0)), (256, 128)),  # a reshape of a transpose
        (lambda a: jax.lax.broadcast_in_dim(a, (128, 256), (1, 0)), (256, 128)),  # a transposing broadcast
        (lambda a: jnp.broadcast_to(a, (256, 128)), (1, 128)),  # a real broadcast of a size-1 dim
        (lambda a: jnp.broadcast_to(a, (2, 256, 128)), (256, 128)),  # a new dim bigger than 1
        (lambda a: a.reshape(3, 2, 128), (2, 3, 128)),  # factors that don't divide each other
        (lambda a: a.reshape(4, 192), (6, 128)),
    ],
)
def test_what_the_view_tracer_refuses(fn, shape):
    assert views.ViewMap.trace(fn, jax.ShapeDtypeStruct(shape, F32)) is None


def test_copies_and_size_1_dims_are_views():
    vmap = views.ViewMap.trace(lambda a: jnp.copy(a).reshape(256, 128), jax.ShapeDtypeStruct((1, 256, 128), F32))
    assert vmap.shape == (256, 128) and vmap.factors[0] == ((1, 0),)  # the size-1 dim is fixed at 0


# --------------------------------------------------------------------------- views: windows


def _window(fn, base, tile, squeezed, starts):
    vmap = views.ViewMap.trace(fn, jax.ShapeDtypeStruct(base, F32))
    assert vmap is not None
    block = types.SimpleNamespace(shape=tile, squeezed=squeezed)

    def check(what, b, full):
        return hardware.check_block(what, b, full, F32)

    return vmap.window("v", block, "...", 128, check).render(starts)


@pytest.mark.parametrize(
    "fn, base, tile, squeezed, why",
    [
        (lambda a: a.reshape(2, 128, 256), (256, 256), (1, 128, 128), (False,) * 3, "a size-1 tile of view dim 0"),
        (lambda a: a.reshape(256, 128), (2, 128, 128), (192, 128), (False,) * 2, "doesn't fall in whole blocks"),
        # The tile covers half of the inner factor of a base dim but all of its outer factor.
        (lambda a: a.reshape(8, 2, 128).transpose(0, 2, 1), (8, 256), (8, 64, 2), (False,) * 3, "rectangular"),
        (lambda a: a.reshape(8, 2, 128).transpose(0, 2, 1), (8, 256), (8, 128, 2), (False,) * 3, "rectangular"),
        (
            lambda a: a.reshape(4, 2, 128).transpose(1, 0, 2).reshape(2, 512),
            (4, 256),
            (2, 128),
            (False, False),
            "two dims of the tile fall in base dim 1",
        ),
        (lambda a: a[None], (256, 128), (1, 128, 128), (False,) * 3, "doesn't map onto a window"),
        (lambda a: a[:, :, 1], (256, 128, 4), (128, 128), (False,) * 2, "picks single elements of the base's last"),
    ],
)
def test_tiles_of_views_that_arent_windows(fn, base, tile, squeezed, why):
    starts = [(f"s{d}", t, 0) for d, t in enumerate(tile)]
    with pytest.raises(views.ViewError, match=why):
        _window(fn, base, tile, squeezed, starts)


def test_a_window_of_a_1d_base():
    got = _window(lambda a: a.reshape(2, 128), (256,), (1, 128), (True, False), [("b", 1, 0), (None, 0, 0)])
    assert got == ["pl.ds(b * 128, 128)"]


def test_affine_index_arithmetic():
    A = views._Affine
    assert A("x", 3, 1).floordiv(2, "_div").render() == "((x * 3 + 1) >> 1)"  # a power of two: a shift
    assert A("x", 2, 1).floordiv(3, "_div").render() == "_div((x * 2 + 1), 3)"
    assert A("x", 4, 3).mod(2, "_rem").render() == "1"  # x * 4 is even
    assert A("x", 3, 1).mod(4, "_rem").render() == "((x * 3 + 1) & 3)"
    assert A("x", 3, 1).mod(5, "_rem").render() == "_rem((x * 3 + 1), 5)"


def test_a_view_at_run_time():
    base = jax.ShapeDtypeStruct((4, 256), F32)
    view = views.make_view(lambda a: a.reshape(8, 128), base)
    assert view.ndim == 2 and repr(view) == "<View (8, 128) of (4, 256)>"
    assert view.materialise() == jax.ShapeDtypeStruct((8, 128), F32)  # a spec's view is a spec
    x = rand((4, 256))
    np.testing.assert_array_equal(view.on(x).materialise(), x.reshape(8, 128))
    assert views.make_view(lambda a: a, 3.0) is None  # no array to view


# --------------------------------------------------------------------------- cost


def _counts(fn, *shapes):
    counts = cost._Counts(False)
    counts.add(jax.make_jaxpr(fn)(*(jax.ShapeDtypeStruct(s, F32) for s in shapes)).jaxpr)
    return counts


def test_reductions_across_lanes_sublanes_and_of_scalars():
    # A (16, 256) f32 array is 2 x 2 vregs: combined elementwise (4 ops), then within each vreg.
    assert _counts(lambda a: a.sum(axis=1), (16, 256)).valu == 4 + 4 / 2 * cost._LANE_REDUCE_OPS
    assert _counts(lambda a: a.sum(axis=0), (16, 256)).valu == 4 + 4 / 2 * cost._SUBLANE_REDUCE_OPS
    scalar = _counts(lambda s: jax.lax.reduce_max(s, axes=()), ())
    assert scalar.valu == scalar.elementwise == 0  # on the scalar unit


def test_nested_jaxprs_are_counted():
    plain_exp = _counts(jnp.exp, (16, 256))
    nested = _counts(jax.jit(jnp.exp), (16, 256))
    assert nested.eup == plain_exp.eup > 0 and nested.transcendentals == plain_exp.transcendentals
    assert not nested.unlowerable


def test_primitives_pallas_cant_lower_are_recorded():
    # Found on a v5e: `jax.checkpoint` (remat2) in a tile body fails in Mosaic, not in the interpreter.
    remat = _counts(jax.checkpoint(jnp.exp), (16, 256))
    assert [name for name, _ in remat.unlowerable] == ["remat2"] and remat.eup == 0


def test_constants_in_cost_keys():
    assert cost._freeze((1, 2.0, ("a",))) == ((int, 1), (float, 2.0), ((str, "a"),))


# --------------------------------------------------------------------------- fusion


def doubled_then_added(x, y):
    a = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            a = a.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
    b = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            b = b.at[i : i + T, j : j + T].set(a[i : i + T, j : j + T] + y[i : i + T, j : j + T])
    return b


def iota_then_added(y):
    a = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            a = a.at[i : i + T, j : j + T].set(jnp.full((T, T), 1.0) * (i + j))  # reads no array
    b = jnp.zeros((256, 256))
    for i in range(0, 256, T):
        for j in range(0, 256, T):
            b = b.at[i : i + T, j : j + T].set(a[i : i + T, j : j + T] + y[i : i + T, j : j + T])
    return b


def test_a_fused_kernels_tiles_and_a_part_reading_no_arrays():
    y = rand((256, 256))
    lowered = autokernel.tpu(iota_then_added).lower(y)
    fused = lowered[0][1]
    assert len(fused.parts) == 2 and fused.tiles == 8
    assert "#   0. iota_then_added_a: " in fused.source
    line = next(ln for ln in fused.source.splitlines() if "#   0. iota_then_added_a" in ln)
    assert "buffered" not in line  # no inputs, so nothing to say about their buffers
    offsets = np.add.outer(np.repeat([0, T], T), np.repeat([0, T], T)).astype(np.float32)
    np.testing.assert_allclose(autokernel.tpu(iota_then_added)(y), offsets + np.asarray(y), rtol=1e-6)
    # On its own, the first nest's kernel takes no inputs (its wrapper used to be `def ..._pallas(, *, ...)`).
    alone = autokernel.tpu(iota_then_added, fuse=False).lower(y)["iota_then_added_a"]
    assert "def iota_then_added_a_pallas(*, interpret=False):" in alone.source
    np.testing.assert_allclose(alone.entry(interpret=INTERPRET), offsets, rtol=1e-6)


def _single(fn, target, *args):
    return autokernel.tpu(fn, fuse=False, target=target).lower(*args)[0][1]


def test_which_nests_fuse(monkeypatch):
    monkeypatch.setattr(fusion, "not_worth_it", lambda before, after: None)
    x = jax.ShapeDtypeStruct((256, 256), F32)
    one, two = _single(doubled_then_added, "v5e", x, x), _single(doubled_then_added, "v5p", x, x)
    assert fusion.fusable(one) and one.cores == 1 and two.cores == 2
    hw = hardware.target("v5e")
    segments = fusion.plan_segments([fusion.NestPlan("a", one), fusion.NestPlan("b", two)], 3, hw)
    assert [(s.start, s.end) for s in segments] == [(0, 1), (1, 2)]
    assert "different numbers of TensorCores" in segments[0].reason
    handwritten = dataclasses.replace(one, plan=None)
    assert fusion.unfusable(handwritten) == "it isn't a generated kernel"
    segments = fusion.plan_segments([fusion.NestPlan("a", one), fusion.NestPlan("b", handwritten)], 3, hw)
    assert "`b` can't be fused: it isn't a generated kernel" in segments[0].reason


# --------------------------------------------------------------------------- hardware


def test_the_attached_tpu_is_the_target(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(hardware._tpu_info.ChipVersion("v6e"), 1)
    monkeypatch.setattr(hardware, "on_tpu", lambda: True)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    hardware.target.cache_clear()
    try:
        hw = hardware.target()
        assert hw.name == "v6e" and hw.vmem_bytes == info.vmem_capacity_bytes and hw.vpu_bf16
    finally:
        hardware.target.cache_clear()
    monkeypatch.undo()
    assert hardware.target().name == hardware.DEFAULT_TARGET


def test_layout_rules_of_scalars():
    assert hardware.check_block("x", (), (), F32) == []
    assert hardware.vmem_nbytes((), jnp.bfloat16) == 2  # a scalar counts one value
    with pytest.raises(AutoKernelError, match="unknown TPU target"):
        hardware.target("v1")
