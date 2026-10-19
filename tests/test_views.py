"""Views: reshapes, transposes and integer indexing between nests, read as windows of the array they view
(no XLA copy, and nothing separating the nests from being fused)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand
from test_fusion import V5E, check_every_kernel, check_races, lower_for_v5e

import autokernel
from autokernel import hardware, views

M = N = 256
T = 128
B, S, H, DH = 2, 128, 2, 128
D = H * DH


def window(vmap, shape, squeezed, starts, dtype=jnp.float32):
    """The base window of a tile (`shape`, `squeezed` per view dim) of a view, rendered for `starts`."""
    block = SimpleNamespace(shape=shape, squeezed=squeezed)

    def check(what, b, full):
        return hardware.check_block(what, b, full, dtype)

    return vmap.window("v", block, "...", 128, check).render(starts)


# --------------------------------------------------------------------------- the structure of a view


def test_a_head_of_q_k_or_v_is_a_window_of_the_qkv_projection():
    base = jax.ShapeDtypeStruct((B * S, 3 * D), jnp.float32)
    vmap = views.ViewMap.trace(lambda a: a.reshape(B, S, 3, H, DH).transpose(2, 0, 3, 1, 4), base)
    assert vmap.shape == (3, B, H, S, DH)
    # qkv[1, b, h, i:i + 128, :]: rows b * S + i, columns D + h * DH of the base
    got = window(vmap, (1, 1, 1, 128, DH), (True, True, True, False, False), [(None, 0, 1), ("b", 1, 0),
                 ("h", 1, 0), ("i", 128, 0), (None, 0, 0)])  # fmt: skip
    assert got == ["pl.ds((b * 128 + i * 128), 128)", "pl.ds((h * 128 + 256), 128)"]


def test_merged_dims_split_into_their_factors():
    base = jax.ShapeDtypeStruct((B, H, S, DH), jnp.float32)
    vmap = views.ViewMap.trace(lambda a: a.transpose(0, 2, 1, 3).reshape(B * S, D), base)
    # rows i:i + 128 of (batch, seq): batch i // S, seq i % S; columns k:k + 128 of (head, dim): head k // DH
    got = window(vmap, (128, 128), (False, False), [("i", 128, 0), ("k", 128, 0)])
    assert got == ["i", "k", "pl.ds(0, 128)", "pl.ds(0, 128)"]


@pytest.mark.parametrize(
    "fn, base, tile, why",
    [
        (lambda a: a.T, (256, 256), (128, 128), "another order"),  # a transpose of the tile itself
        (lambda a: a.reshape(512, 128), (8, 64, 128), (128, 128), "base's order"),  # straddles two batches
        (lambda a: a.reshape(256, 2, 64), (256, 128), None, None),  # fine as a view, see below
    ],
)
def test_tiles_that_are_not_one_window_of_the_base(fn, base, tile, why):
    vmap = views.ViewMap.trace(fn, jax.ShapeDtypeStruct(base, jnp.float32))
    if tile is None:  # columns 64:128 of the base: lane-misaligned
        with pytest.raises(views.ViewError, match=r"isn't a multiple of 128|must be a multiple of 128"):
            window(vmap, (128, 1, 64), (False, True, False), [("i", 128, 0), (None, 0, 1), (None, 0, 0)])
        return
    with pytest.raises(views.ViewError, match=why):
        window(vmap, tile, (False, False), [("i", tile[0], 0), (None, 0, 0)])


def test_what_is_not_a_view():
    base = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    assert views.ViewMap.trace(lambda a: a + 1, base) is None
    assert views.ViewMap.trace(lambda a: a[:128], base) is None  # a partial slice
    assert views.ViewMap.trace(lambda a: a.astype(jnp.bfloat16), base) is None
    assert views.make_view(lambda a: a.reshape(256, 256), base) is None  # the array itself
    assert views.ViewMap.trace(lambda a: jnp.swapaxes(a[None], 0, 1)[:, 0], base).identity


# --------------------------------------------------------------------------- programs


def heads(x, w):
    # A projection, its output viewed as heads, and a nest reading one head at a time.
    p = jnp.zeros((B * S, D))
    for i in range(0, B * S, T):
        for j in range(0, D, T):
            acc = jnp.zeros((T, T))
            for k in range(0, D, T):
                acc += x[i : i + T, k : k + T] @ w[k : k + T, j : j + T]
            p = p.at[i : i + T, j : j + T].set(acc)
    p = p.reshape(B, S, H, DH).transpose(0, 2, 1, 3)  # (batch, head, seq, dim): a view of `p`
    out = jnp.zeros((B, H, S, DH))
    for b in range(B):
        for h in range(H):
            for i in range(0, S, T):
                out = out.at[b, h, i : i + T, :].set(jnp.tanh(p[b, h, i : i + T, :]) * (h + 1))
    return out, p


def heads_reference(x, w):
    p = jnp.dot(x, w, precision="highest").reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    return jnp.tanh(p) * (jnp.arange(H) + 1)[None, :, None, None], p


def test_a_view_between_nests_is_read_in_place_and_fused():
    prog = autokernel.tpu(heads)
    x, w = rand((B * S, D), 1), rand((D, D), 2) / 16
    lowered = check_every_kernel(prog, x, w)
    assert lowered.names == ["heads_p+heads_out"]  # nothing separates the nests
    assert "p_hbm.at[pl.ds((b_blk * 128 + i_blk * 128), 128), pl.ds(h_blk * 128, 128)]" in lowered.source
    out, p = prog(x, w)  # `p`, the view, is still what the code says (materialised: it is returned)
    want_out, want_p = heads_reference(x, w)
    np.testing.assert_allclose(out, want_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(p, want_p, rtol=1e-4, atol=1e-4)
    assert prog.verify(x, w) < 1e-4


def heads_out(x, w):  # `heads` without returning the view
    p = jnp.zeros((B * S, D))
    for i in range(0, B * S, T):
        for j in range(0, D, T):
            acc = jnp.zeros((T, T))
            for k in range(0, D, T):
                acc += x[i : i + T, k : k + T] @ w[k : k + T, j : j + T]
            p = p.at[i : i + T, j : j + T].set(acc)
    p = p.reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    out = jnp.zeros((B, H, S, DH))
    for b in range(B):
        for h in range(H):
            for i in range(0, S, T):
                out = out.at[b, h, i : i + T, :].set(jnp.tanh(p[b, h, i : i + T, :]) * (h + 1))
    return out


def test_an_unused_view_is_never_copied():
    prog = autokernel.tpu(heads_out)
    x, w = rand((B * S, D), 1), rand((D, D), 2) / 16
    assert "transpose" not in jax.jit(prog).lower(x, w).as_text()
    np.testing.assert_allclose(prog(x, w), heads_reference(x, w)[0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("fuse", [True, False])
def test_a_view_that_isnt_a_window_is_copied_first(fuse):
    def f(x, w):
        y = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                y = y.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * 2)
        y = y.T  # a transpose of every tile: not a window of `y`
        z = jnp.zeros((M, N))
        for i in range(0, M, T):
            for j in range(0, N, T):
                z = z.at[i : i + T, j : j + T].set(y[i : i + T, j : j + T] + w[i : i + T, j : j + T])
        return z

    prog = autokernel.tpu(f, fuse=fuse)
    x, w = rand((M, N), 1), rand((M, N), 2)
    np.testing.assert_allclose(prog(x, w), (x * 2).T + w, rtol=1e-6)
    lowered = prog.lower(x, w)
    assert lowered.names == ["f_y", "f_z"]
    if fuse:
        assert "XLA copies it first" in lowered.cuts[0]
    check_every_kernel(prog, x, w)


def test_views_of_inputs_chained_and_indexed():
    def f(stack, w):
        x = stack[1]  # an integer index: a view of `stack`
        x = x.reshape(2, 128, D).reshape(256, D)  # views of views
        out = jnp.zeros((256, D))
        for i in range(0, 256, T):
            for j in range(0, D, T):
                out = out.at[i : i + T, j : j + T].set(x[i : i + T, j : j + T] * w[i : i + T, j : j + T])
        return out

    prog = autokernel.tpu(f)
    stack, w = rand((3, 256, D), 1), rand((256, D), 2)
    np.testing.assert_allclose(prog(stack, w), stack[1] * w, rtol=1e-6)
    lowered = check_every_kernel(prog, stack, w)
    assert "x_hbm.at[1, pl.ds(i_blk * 128, 128), pl.ds(j_blk * 128, 128)]" in lowered.source  # `stack`
    assert lowered.inputs[0][0].shape == stack.shape  # the kernel is given `stack` itself: no copy of `stack[1]`


def test_gradients_through_fused_views():
    prog = autokernel.tpu(heads)
    x, w = rand((B * S, D), 1), rand((D, D), 2) / 16

    def loss(f):
        return lambda x, w: jnp.sum(f(x, w)[0] ** 2)

    got = jax.grad(loss(prog), argnums=(0, 1))(x, w)
    want = jax.grad(loss(heads_reference), argnums=(0, 1))(x, w)
    for g, wt in zip(got, want, strict=True):
        np.testing.assert_allclose(g, wt, rtol=1e-3, atol=1e-3)


def test_view_kernels_race_free_and_lower_for_v5e_at_bf16():
    prog = autokernel.tpu(heads_out)
    x, w = rand((B * S, D), 1, jnp.bfloat16), rand((D, D), 2, jnp.bfloat16)
    lowered = prog.lower(x, w)
    for (_, kernel), specs in zip(lowered, lowered.inputs, strict=True):
        check_races(kernel, [rand(s.shape, 3, s.dtype) for s in specs])
        assert "tpu_custom_call" in lower_for_v5e(kernel, specs, device=V5E)
