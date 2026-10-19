"""The `@autokernel.tpu` decorator and the `TpuKernel` it returns for a single tile loop nest.

A function that is more than one loop nest (nests mixed with ordinary JAX code) becomes a `TpuProgram`
instead; see `program.py`.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from . import analysis, autodiff, batching, frontend, hardware, mxu, sharding
from . import cost as cost_lib
from . import reference as reference_lib
from . import tune as tune_lib
from .errors import AutoKernelError
from .lowering import Lowered, Options, lower

__all__ = ["TpuKernel", "check_outputs", "tpu"]

_Signature = tuple[tuple[tuple[int, ...], np.dtype], ...]


def _spec(x: Any) -> jax.ShapeDtypeStruct:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return jax.ShapeDtypeStruct((), jnp.int32)  # a runtime scalar
    if not hasattr(x, "shape") or not hasattr(x, "dtype"):
        raise AutoKernelError(f"kernel arguments must be arrays, got {type(x).__name__}")
    return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)


def _signature(args: tuple[Any, ...]) -> _Signature:
    return tuple((tuple(s.shape), np.dtype(s.dtype)) for s in map(_spec, args))


def _concrete(args: tuple[Any, ...]) -> bool:
    """Do the arguments hold values (jax or numpy arrays), not tracers or specs?"""
    return all(not isinstance(a, (jax.core.Tracer, jax.ShapeDtypeStruct)) for a in args)


_warned_interpreting = False


def _warn_interpreting() -> None:
    global _warned_interpreting
    if not _warned_interpreting:
        _warned_interpreting = True
        warnings.warn(
            f"autokernel: no TPU found (backend={jax.default_backend()!r}); running kernels in the Pallas TPU "
            "interpreter",
            stacklevel=4,
        )


class TpuKernel:
    """A Python tile loop nest compiled to a Pallas TPU kernel.

    Calling it runs the generated kernel. Other entry points:

    - `lower(*args)`: the `Lowered` kernel (schedule, tiles, cost estimate, generated `.source`)
    - `source(*args)`: the generated Pallas source
    - `reference(*args)`: run the original Python loop nest, the semantic ground truth
    - `verify(*args)`: run both and raise if they disagree; returns the max abs error
    - `tuning(*args)`: every tuning candidate tried, with predicted (and measured) times

    `args` for `lower`, `source` and `tuning` may be arrays or `jax.ShapeDtypeStruct`s. Kernels are
    generated, and tuned, once per input signature (shapes and dtypes).

    Calls compose with `jax.jit`, `jax.vmap` (a batch runs as one kernel generated with an extra batch
    loop), forward- and reverse-mode differentiation, and inputs sharded over a mesh (each device runs a
    kernel for its shard); see `autokernel.tpu`.
    """

    def __init__(
        self,
        fn: Callable | analysis.Environment,
        *,
        interpret: Any = None,
        verify: bool = False,
        options: Options | None = None,
        nest: frontend.LoopNest | None = None,
        backward: Callable | None = None,
    ):
        """`fn` is the Python function, or, for a `nest` parsed from elsewhere (one nest of a program), the
        `analysis.Environment` its code runs in. `backward` is the kernel's VJP (see `autokernel.tpu`)."""
        self.fn = fn
        self.nest = nest if nest is not None else frontend.parse(fn)  # fail at decoration time on unsupported code
        self.interpret = interpret
        self.verify_on_first_call = verify
        self.options = options or Options()
        self.backward = backward
        self._lowered: dict[_Signature, Lowered] = {}
        self._tuning: dict[_Signature, list[tune_lib.Candidate]] = {}
        self._measured: set[_Signature] = set()
        self._compiled: dict[tuple, tuple[Lowered, Callable[..., Any], Callable[..., Any]]] = {}
        self._verified: set[_Signature] = set()
        self._reference: dict[tuple, Callable] = {}  # by tuned constants
        self._batched: dict[tuple, TpuKernel | Exception] = {}  # (in_batched, size) -> kernel for jax.vmap
        self._local: dict[tuple, TpuKernel] = {}  # per-shard kernels for sharded inputs
        self._shard_plans: dict[tuple, sharding.ShardPlan] = {}
        self._warned: set[str] = set()
        if isinstance(fn, analysis.Environment):
            self.__name__ = self.__qualname__ = self.nest.fn_name
        else:
            functools.update_wrapper(self, fn)

    def _interpret_mode(self) -> Any:
        if self.interpret is None:
            if hardware.on_tpu():
                return False
            _warn_interpreting()
            return pltpu.InterpretParams()
        if self.interpret is True:
            return pltpu.InterpretParams()
        if self.interpret is False and not hardware.on_tpu():
            raise AutoKernelError(
                f"autokernel only targets TPU, but the default backend is {jax.default_backend()!r}; "
                "pass interpret=True to simulate it"
            )
        return self.interpret

    def lower(self, *args: Any) -> Lowered:
        key = _signature(args)
        if key not in self._lowered:
            specs = [_spec(a) for a in args]
            if self.options.tune:
                self._tuning[key] = tune_lib.search(self.nest, self.fn, specs, self.options)
                self._lowered[key] = self._tuning[key][0].lowered  # the cost model's pick
            else:
                self._lowered[key] = lower(self.nest, self.fn, specs, self.options)
                if hardware.on_tpu() and cost_lib.overhead_bound(self._lowered[key].estimate):
                    warnings.warn(
                        f"autokernel: {self.__name__}'s steps are too small to hide their fixed per-step cost "
                        f'({self._lowered[key].steps} steps); larger blocks would run faster: try tune="auto"',
                        stacklevel=3,
                    )
        # Measuring needs real arrays on a real TPU: do it on the first call that has them.
        measure = self.options.tune and self.options.autotune == "measure" and key not in self._measured
        if measure and _concrete(args) and hardware.on_tpu():
            self._measured.add(key)
            arrays = tuple(jax.numpy.asarray(a) for a in args)
            best = tune_lib.measure(self._tuning[key], arrays, self.options.measure_top)
            if best is not None:
                self._lowered[key] = best
        return self._lowered[key]

    def tuning(self, *args: Any) -> list[tune_lib.Candidate]:
        self.lower(*args)
        return self._tuning.get(_signature(args), [])

    def source(self, *args: Any) -> str:
        return self.lower(*args).source

    def reference(self, *args: Any) -> Any:
        """Run the Python loop nest with the kernel's `@` semantics: MXU dots accumulating in f32 / i32.

        Plain JAX would round `bf16 @ bf16` to bf16 at every step and wrap `int8 @ int8`; the kernel keeps
        the MXU's wide accumulator, so the reference does too.
        """
        # A tuned kernel runs with the tuned constants: so must the loops it is compared with.
        overrides = self.lower(*args).overrides if self.options.tune else {}
        key = tuple(sorted(overrides.items()))
        reference = self._reference.get(key)
        if reference is None:
            reference = self._reference[key] = self._build_reference(overrides)
        if self.options.matmul_precision == "default":
            return reference(*args)
        with jax.default_matmul_precision("highest"):
            return reference(*args)

    def _build_reference(self, overrides: dict[str, Any]) -> Callable:
        fdef = mxu.rewrite_matmuls(reference_lib.traceable_masks(self.nest))
        fdef.decorator_list = []
        module = ast.Module(body=[fdef], type_ignores=[])
        ast.increment_lineno(module, self.nest.first_line - 1)
        env = analysis.environment(self.fn)
        namespace = dict(env.names)
        namespace.update(overrides)
        namespace.update(mxu.helpers(self.options.matmul_precision))
        namespace.update(reference_lib.helpers())
        exec(compile(module, env.filename, "exec"), namespace)
        return namespace[self.nest.fn_name]

    def __call__(self, *args: Any) -> Any:
        lowered = self.lower(*args)
        if lowered.scalars:  # runtime indices whose values are known now: check them against the inputs
            known = {p: a for p, a in zip(self.nest.params, args, strict=True) if p in lowered.scalars}
            lowered.check_indices(known)  # traced ones: a checkify check (the kernel clamps them)
        key = _signature(args)
        shard_plan = self._shard_plan(lowered, args)
        ckey = (key, shard_plan.key if shard_plan is not None else None)
        compiled = self._compiled.get(ckey)
        if compiled is None or compiled[0] is not lowered:  # first call, or measuring changed the pick
            if shard_plan is None:
                forward = jax.jit(functools.partial(lowered.entry, interpret=self._interpret_mode()))
                reference = self.reference
            else:  # each device runs its shard's kernel, and differentiates its shard's loops
                local = self._local_kernel(lowered, shard_plan)
                forward = sharding.run(shard_plan, local, self.nest.returns_tuple)
                reference = sharding.run(shard_plan, local.reference, self.nest.returns_tuple)
            entry = autodiff.differentiable(forward, reference, self.backward, vmap_rule=self._vmap_rule)
            # jit: one cached trace of the wrappers rather than one per eager call
            compiled = self._compiled[ckey] = (lowered, jax.jit(entry), reference)
        out = compiled[1](*args)
        if self.verify_on_first_call and key not in self._verified and _concrete(args):
            self._verified.add(key)
            self._check(out, compiled[2](*args))
        return out

    # ------------------------------------------------------------------ jax.vmap

    def _vmap_rule(self, axis_size: int, in_batched: list[bool], *args: Any) -> tuple[Any, Any]:
        """custom_vmap rule: one kernel over the whole batch (`batching.batched_nest`), or one call per batch
        element when the nest can't be batched that way."""
        kernel = self._batched_kernel(tuple(in_batched), axis_size, args)
        out = kernel(*args) if kernel is not None else batching.sequential(self, in_batched, args)
        return out, jax.tree.map(lambda _: True, out)

    def _batched_kernel(self, in_batched: tuple[bool, ...], size: int, args: tuple[Any, ...]) -> TpuKernel | None:
        for p, a, b in zip(self.nest.params, args, in_batched, strict=True):
            if b and analysis.is_scalar_spec(jax.ShapeDtypeStruct(tuple(a.shape)[1:], a.dtype)):
                self._warn_sequential(f"its runtime scalar `{p}` is batched")
                return None
        bkey = (in_batched, size)
        kernel = self._batched.get(bkey)
        if kernel is None:
            try:
                nest = batching.batched_nest(self.nest, in_batched, size)
            except batching.BatchingError as e:
                self._batched[bkey] = e
                self._warn_sequential(str(e))
                return None
            env = analysis.environment(self.fn)
            names = {**env.names, batching.ALLOC: batching.batched_alloc}
            kernel = self._batched[bkey] = TpuKernel(
                analysis.Environment(names, env.filename), nest=nest, interpret=self.interpret, options=self.options
            )
        if isinstance(kernel, Exception):
            return None
        try:
            kernel.lower(*args)
        except AutoKernelError as e:
            self._warn_sequential(str(e))
            return None
        return kernel

    def _warn_sequential(self, why: str) -> None:
        if why not in self._warned:
            self._warned.add(why)
            warnings.warn(
                f"autokernel: {self.__name__}: can't batch the kernel into one kernel under jax.vmap ({why}); running "
                "it once per batch element",
                stacklevel=5,
            )

    # ------------------------------------------------------------------ sharded inputs

    def _shard_plan(self, lowered: Lowered, args: tuple[Any, ...]) -> sharding.ShardPlan | None:
        """How to run on `args` when some are sharded over a mesh (None when none is)."""
        found = sharding.find_mesh(args)
        if found is None or lowered.plan is None:
            return None
        key = (id(lowered), found[0], tuple(found[1]))
        if key in self._shard_plans:
            return self._shard_plans[key]
        shard_plan = self._shard_plans[key] = sharding.plan(lowered.plan.spec, *found)
        if shard_plan.loops:
            try:  # the per-shard kernel must exist for the per-shard shapes
                self._local_kernel(lowered, shard_plan).lower(*sharding.local_specs(shard_plan, args))
            except AutoKernelError as e:
                shard_plan = sharding.gather_all(lowered.plan.spec, shard_plan, f"its per-shard kernel fails: {e}")
                self._shard_plans[key] = shard_plan
        return shard_plan

    def _local_kernel(self, lowered: Lowered, shard_plan: sharding.ShardPlan) -> TpuKernel:
        """The kernel each device runs on its shards (this kernel itself when nothing is split)."""
        if not shard_plan.loops:
            return self
        key = (id(lowered), shard_plan.loops)
        kernel = self._local.get(key)
        if kernel is None:
            nest, extra = sharding.local_nest(lowered.plan.spec, shard_plan)
            env = analysis.environment(self.fn)
            names = {**env.names, **lowered.overrides, **extra}
            options = dataclasses.replace(self.options, tune=())  # the shard's kernel keeps the tuned tiles
            kernel = self._local[key] = TpuKernel(
                analysis.Environment(names, env.filename), nest=nest, interpret=self.interpret, options=options
            )
        return kernel

    def verify(self, *args: Any, rtol: float | None = None, atol: float | None = None) -> float:
        return self._check(self(*args), self.reference(*args), rtol, atol)

    def _check(self, got: Any, want: Any, rtol: float | None = None, atol: float | None = None) -> float:
        return check_outputs(self.__name__, got, want, self.options.matmul_precision, rtol, atol)


def check_outputs(
    name: str, got: Any, want: Any, matmul_precision: str, rtol: float | None = None, atol: float | None = None
) -> float:
    """Raise AssertionError unless `got` (from kernels) matches `want` (the reference); the max abs error.

    Both are arrays or matching pytrees of arrays. Tolerances default to what the dtypes allow.
    """
    got_leaves, got_tree = jax.tree.flatten(got)
    want_leaves, want_tree = jax.tree.flatten(want)
    if got_tree != want_tree:
        raise AssertionError(f"{name}: kernels returned {got_tree}, the reference returned {want_tree}")
    errs = [_check_one(name, g, w, matmul_precision, rtol, atol) for g, w in zip(got_leaves, want_leaves, strict=True)]
    return max(errs, default=0.0)


def _check_one(
    name: str, got: jax.Array, want: jax.Array, matmul_precision: str, rtol: float | None, atol: float | None
) -> float:
    if got.shape != want.shape or got.dtype != want.dtype:
        raise AssertionError(
            f"{name}: kernel returned {got.dtype}{got.shape}, reference returned {want.dtype}{want.shape}"
        )
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    dtype = np.dtype(want.dtype)
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        tol = 0.0  # integer results are exact on both sides
    elif dtype.itemsize <= 2 or (matmul_precision == "default" and hardware.on_tpu()):
        tol = 2e-2  # rounded at every step, or a single bf16 MXU pass for f32
    else:
        tol = 1e-4  # f32 on both sides: only the summation order differs
    if rtol is None:
        rtol = tol
    if atol is None:
        # Low-precision results round at every step, so their error grows with the output's magnitude:
        # compare against that instead of per element (its finite part: an inf would excuse anything).
        finite = np.abs(w[np.isfinite(w)])
        scale = float(np.max(finite)) if tol >= 2e-2 and finite.size else 1.0
        atol = tol * max(scale, 1.0)
    # Equal values, infinities and NaNs included, are exact.
    same = (g == w) | (np.isnan(g) & np.isnan(w))
    with np.errstate(invalid="ignore"):  # inf - inf
        err = float(np.max(np.abs(np.where(same, 0.0, g - w)))) if g.size else 0.0
    if not np.allclose(g, w, rtol=rtol, atol=atol, equal_nan=True):
        bad = int(np.sum(~np.isclose(g, w, rtol=rtol, atol=atol, equal_nan=True)))
        raise AssertionError(
            f"{name}: Pallas kernel disagrees with the Python reference in {bad}/{g.size} elements "
            f"(max abs err {err:.3g}, rtol={rtol}, atol={atol})"
        )
    return err


def tpu(
    fn: Callable | None = None,
    *,
    interpret: Any = None,
    verify: bool = False,
    tune: Mapping[str, Sequence[Any] | str] | str | None = None,
    autotune: str = "model",
    buffers: int | None = None,
    target: str | None = None,
    cores: int | None = None,
    check_tiling: bool = True,
    vmem_limit_bytes: int | None = None,
    matmul_precision: str = "highest",
    background_fill: bool = True,
    fuse: bool | int = True,
    unroll: bool = False,
    backward: Callable | None = None,
) -> Any:
    """Compile a Python tile loop nest into a Pallas TPU kernel, or a program of them.

    A function that is one tile loop nest (a prologue of constants and output allocations, the nest, and
    `return`) becomes a `TpuKernel`. Anything else, such as several nests mixed with ordinary JAX code and
    Python control flow, becomes a `TpuProgram`: each nest is its own kernel and the rest runs as JAX.
    Options apply to every kernel.

    Use it bare or with options::

        @autokernel.tpu
        def matmul(a, b): ...

        @autokernel.tpu(tune={"TM": (128, 256, 512), "TN": (128, 256, 512), "TK": (256, 512)})
        def matmul(a, b): ...  # TM, TN, TK are module constants the loops use

    Args:
        interpret: None (default) runs natively on TPU and in the Pallas TPU interpreter elsewhere;
            False requires a real TPU; True or a `pltpu.InterpretParams` always interprets.
        verify: on the first call per input signature with concrete arrays (not under `jax.jit`), compare
            against the original Python code.
        tune: constants to tune (tile sizes, usually) mapped to candidate values, or to "auto" to generate
            them; `tune="auto"` tunes every block size (a loop step that is an int constant). Every
            combination is tried; invalid ones (misaligned tiles, too much VMEM) are dropped and the rest
            ranked by a cost model built from the TPU's specs and JAX's cost analysis.
        autotune: "model" picks the best predicted candidate; "measure" also times the best eight block
            sizes on the TPU on the first call with concrete arrays, then 2 and 3 buffers for the winner.
        buffers: VMEM slots per streamed input (2 = double buffering). Default: 3 where they fit in VMEM,
            else 2; `autotune="measure"` tries both.
        target: the TPU to plan for when not running on one, e.g. "v5e" (default) or "v5p".
        cores: TensorCores to split the work across (default: all of the target's, 2 on v4 and v5p).
        check_tiling: reject tiles that break TPU (8, 128) layout rules before Mosaic does.
        vmem_limit_bytes: Mosaic's scoped-VMEM limit; by default sized from the kernel's estimated use.
        matmul_precision: "highest" (default) makes f32 `@` a full-precision multi-pass MXU matmul;
            "default" uses Mosaic's faster single bf16 pass (verify then uses looser tolerances).
        background_fill: with a static tile mask, write the skipped tiles' constant value with async DMAs
            from the kernel instead of filling the whole output before the kernel runs.
        fuse: (programs) run consecutive nests (no code between them) as one fused kernel, so each nest's
            first input copies overlap the previous nest's last steps, with one launch for all of them: up to
            three nests per kernel (`fusion.MAX_PARTS`), or up to `fuse` nests when it is an int.
        unroll: (programs) unroll Python `for` loops around nests when tracing, instead of running them as a
            `jax.lax.fori_loop` where the loop body allows it.
        backward: how `jax.grad` differentiates the kernel (or the whole program):
            `backward(*inputs, cotangent)` returns one gradient per input, or None for no gradient. It can
            itself call autokernel kernels. By default the gradient is that of the Python loops, run as plain
            JAX: always correct, but computed by XLA rather than a generated kernel.

    JAX transformations:

    - Differentiation: without `backward`, the kernel's JVP is its Python loops' JVP (run as plain JAX, the
      primal output still the kernel's), so forward mode (`jax.jvp`, `jax.jacfwd`, `jax.linearize`) and
      reverse mode (`jax.grad`, `jax.vjp`, `jax.jacrev`) both work, running the kernel once. With
      `backward`, only reverse mode is defined.
    - `jax.vmap`: the batched kernel is one kernel, generated from the loop nest with the batch as an
      extra outermost parallel loop (`for b in range(B)`, batched inputs read `x[b, ...]`, outputs written
      `out.at[b, ...]`), so a batch is one launch. Any `in_axes`, nested `vmap`s, `vmap` of `jit` or
      `grad` and the reverse all work, masks included (a runtime mask reading a batched input is evaluated
      per batch element). A batched runtime scalar (a 0-d integer input) instead runs the kernel once per
      batch element (`jax.lax.map`), with a warning. In a program, each nest is batched like this, and
      fused nests run as separate batched kernels.
    - Sharded inputs (a `NamedSharding` over a mesh: concrete arrays, or traced values under `jax.jit`
      with `Explicit` mesh axes, the default of `jax.make_mesh`): the kernel runs under `jax.shard_map`,
      each device running a kernel generated for its shard, when the sharded dims are tiled by parallel
      loops used only to index (batch, head and row loops usually; not a loop a mask or the tile body
      reads, or a reduction loop). Outputs come out sharded along the same loops. Input dims sharded any
      other way are gathered first, with a one-time warning. Under `jax.jit` with `Auto` mesh axes the
      kernel can't see the sharding and XLA runs it on gathered inputs.
    """
    options = Options(
        check_tiling=check_tiling,
        vmem_limit_bytes=vmem_limit_bytes,
        matmul_precision=matmul_precision,
        background_fill=background_fill,
        buffers=buffers,
        target=target,
        cores=cores,
        tune=Options.make_tune(tune),
        autotune=autotune,
    )

    def wrap(f: Callable) -> Any:
        from .program import TpuProgram, is_single_nest  # program builds on TpuKernel

        fdef = frontend.function_ast(f)
        if is_single_nest(fdef):
            return TpuKernel(f, interpret=interpret, verify=verify, options=options, backward=backward)
        return TpuProgram(
            f,
            interpret=interpret,
            verify=verify,
            options=options,
            fdef=fdef,
            fuse=fuse,
            unroll=unroll,
            backward=backward,
        )

    return wrap(fn) if fn is not None else wrap
