"""The `@autokernel.tpu` decorator and the `TpuKernel` it returns for a single tile loop nest.

A function that is more than one loop nest (nests mixed with ordinary JAX code) becomes a `TpuProgram`
instead; see `program.py`.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import inspect
import threading
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from . import analysis, autodiff, batching, frontend, hardware, mxu, scopes, sharding, tiles
from . import cache as cache_lib
from . import cost as cost_lib
from . import reference as reference_lib
from . import tune as tune_lib
from .errors import AutoKernelError
from .lowering import Lowered, Options, lower

__all__ = ["TpuKernel", "check_outputs", "tpu"]

_Signature = tuple[tuple[tuple[int, ...], np.dtype, bool], ...]


def _weak_type(x: Any) -> bool:
    return bool(getattr(getattr(x, "aval", x), "weak_type", False))


def _spec(x: Any) -> jax.ShapeDtypeStruct:
    if isinstance(x, jax.ShapeDtypeStruct):
        return x
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return jax.ShapeDtypeStruct((), jnp.int32)  # a runtime scalar
    if not hasattr(x, "shape") or not hasattr(x, "dtype"):
        raise AutoKernelError(f"kernel arguments must be arrays, got {type(x).__name__}")
    shape = tuple(x.shape)
    # A 0-d float mixes with tiles as its weak type says (`bf16_tile * s` stays bf16 for a weak `s`).
    weak = shape == () and jnp.issubdtype(x.dtype, jnp.floating) and _weak_type(x)
    return jax.ShapeDtypeStruct(shape, x.dtype, weak_type=weak)


def _arg_signature(x: Any) -> tuple[tuple[int, ...], np.dtype, bool]:
    if isinstance(x, jax.Array) and x.ndim:  # the common case, without building a spec
        return x.shape, x.dtype, False
    s = _spec(x)
    return tuple(s.shape), np.dtype(s.dtype), bool(s.weak_type)


def _signature(args: tuple[Any, ...]) -> _Signature:
    return tuple(map(_arg_signature, args))


def _concrete(args: tuple[Any, ...]) -> bool:
    """Do the arguments hold values (jax or numpy arrays), not tracers or specs?"""
    return all(not isinstance(a, (jax.core.Tracer, jax.ShapeDtypeStruct)) for a in args)


def _typed(overrides: Mapping[str, Any]) -> tuple:
    """Tuned constants as a cache key that tells `2` from `2.0`."""
    return tuple((k, type(v), v) for k, v in sorted(overrides.items(), key=lambda kv: kv[0]))


def _is_outer_array(value: Any) -> bool:
    """A global or closure value the kernel reads as an input array (rather than as a constant)."""
    return isinstance(value, (jax.Array, np.ndarray)) and value.ndim >= 1


# --------------------------------------------------------------------------- the names a nest reads


def _fn_bound(nest: frontend.LoopNest) -> set[str]:
    return set(nest.params) | scopes.binds(list(nest.fdef.body), False)


def _free_names(nest: frontend.LoopNest) -> tuple[str, ...]:
    """Names the nest's code reads that it doesn't bind: its module globals and closure variables."""
    a = nest.fdef.args
    annotations = [p.annotation for p in [*a.posonlyargs, *a.args] if p.annotation is not None]
    nodes = [*annotations, *nest.fdef.body]
    return tuple(dict.fromkeys(n.id for n in scopes.global_loads(nodes, _fn_bound(nest))))


def _sliced_names(nest: frontend.LoopNest) -> set[str]:
    """Free names the tile body reads through slices or loop variables (`BIAS[i : i + T, :]`): if they are
    global arrays, the kernel reads them as inputs, as a program's nests do."""
    loop_vars = {lp.var for lp in nest.loops}
    bound = _fn_bound(nest)
    found = set()
    for tree in [*nest.tile_statements(), *(st.value for st in nest.stores)]:
        free = {id(n) for n in scopes.global_loads([tree], bound)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and id(n.value) in free:
                dims = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
                if any(isinstance(d, ast.Slice) or (isinstance(d, ast.Name) and d.id in loop_vars) for d in dims):
                    found.add(n.value.id)
    return found


class _Reader:
    """Reads the current values of a function's module globals and closure variables by name."""

    def __init__(self, fn: Callable):
        fn = fn.__func__ if inspect.ismethod(fn) else fn
        self.name = fn.__name__
        self.cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True))
        self.globals = fn.__globals__

    def get(self, name: str, default: Any = None) -> Any:
        cell = self.cells.get(name)
        if cell is None:
            return self.globals.get(name, default)
        try:
            return cell.cell_contents
        except ValueError:  # an empty cell
            return default

    def snapshot(self, names: Iterable[str]) -> dict[str, Any]:
        """The current values of `names` (unbound ones, builtins say, left out)."""
        out = {}
        for n in names:
            if n in self.cells:
                try:
                    out[n] = self.cells[n].cell_contents
                except ValueError:
                    raise AutoKernelError(
                        f"{self.name}: closure variable `{n}` has no value yet; assign it before calling the kernel"
                    ) from None
            elif n in self.globals:
                out[n] = self.globals[n]
        return out


@dataclasses.dataclass(frozen=True, eq=False)
class _Context:
    """What a kernel's code sees for one input signature, fixed for every artefact built for it (kernel,
    tuning, reference and derivative, batched and per-shard kernels), as `jax.jit` reads a function's
    globals once per signature: the values its module and closure constants had when it was first called
    with that signature, and the nest with the global arrays it reads as extra parameters."""

    env: analysis.Environment
    nest: frontend.LoopNest
    outer: tuple[str, ...]  # global and closure arrays passed as inputs after the function's own


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

    Every entry point takes the arguments positionally or by name (but see `verify` for parameters named
    `rtol` or `atol`). `args` for `lower`, `source` and `tuning` may be arrays or `jax.ShapeDtypeStruct`s.

    Kernels are generated, and tuned, once per input signature (shapes and dtypes). As `jax.jit` does,
    the kernel reads the module and closure constants its code uses once per signature, when it is first
    built for it: the kernel, `reference`, `verify`, the derivative and batched and per-shard kernels for
    that signature all use those values, and changing such a constant afterwards changes none of them.
    Global and closure arrays the tile body reads through slices (`BIAS[i : i + T, :]`) are inputs
    instead, read with their current values on every call (under `jax.jit`, when it traces).

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
        self._lock = threading.RLock()  # concurrent first calls generate (and tune) a kernel once
        # Caches by input signature, through its `_Context`.
        self._contexts: dict[tuple, _Context] = {}  # by (global arrays read as inputs, signature)
        self._lowered: dict[_Context, Lowered] = {}
        self._tuning: dict[_Context, list[tune_lib.Candidate]] = {}
        self._measured: set[_Context] = set()
        self._cache_keys: dict[_Context, str] = {}  # tuning cache keys of the signatures searched
        self._cached_history: dict[_Context, dict[str, float]] = {}  # measurements earlier runs cached
        self._history: dict[_Context, dict[str, float]] = {}  # measurements, this run's included
        self._compiled: dict[tuple, tuple[Lowered, Callable[..., Any], Callable[..., Any]]] = {}
        self._verified: set[_Context] = set()
        self._reference: dict[tuple, Callable] = {}  # by context and tuned constants
        self._batched: dict[tuple, TpuKernel | Exception] = {}  # (context, in_batched, size) -> kernel for jax.vmap
        self._local: dict[tuple, TpuKernel] = {}  # per-shard kernels for sharded inputs
        self._shard_plans: dict[tuple, sharding.ShardPlan] = {}
        self._warned: set[str] = set()
        # The frontend only takes plain positional-or-keyword array parameters (no defaults, `*args` or
        # keyword-only ones): arguments bind to them by position or by name.
        params = [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in self.nest.params]
        self._parameters = inspect.Signature(params)
        if isinstance(fn, analysis.Environment):
            self.__name__ = self.__qualname__ = self.nest.fn_name
            self._reader: _Reader | None = None
            self._read: tuple[str, ...] = ()
            self._sliced: tuple[str, ...] = ()
        else:
            functools.update_wrapper(self, fn)
            self._reader = _Reader(fn)
            self._filename = inspect.getsourcefile(fn) or "<autokernel>"
            self._read = _free_names(self.nest)  # the constants a context snapshots (tuned ones are among them)
            self._sliced = tuple(sorted(_sliced_names(self.nest)))

    # ------------------------------------------------------------------ arguments and contexts

    def _bind(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[_Context, tuple[Any, ...]]:
        """The context for these arguments' signature, and the kernel's inputs: the arguments bound to the
        function's parameters, then the global arrays the nest reads."""
        if kwargs or len(args) != len(self.nest.params):
            try:
                args = tuple(self._parameters.bind(*args, **kwargs).arguments.values())
            except TypeError as e:
                raise AutoKernelError(
                    f"{self.__name__} takes {len(self.nest.params)} arrays, got {len(args) + len(kwargs)}: {e}"
                ) from None
        outer: tuple[str, ...] = ()
        if self._sliced:  # global arrays: their current values are inputs
            values = [self._reader.get(n) for n in self._sliced]
            outer = tuple(n for n, v in zip(self._sliced, values, strict=True) if _is_outer_array(v))
            args = (*args, *(v for v in values if _is_outer_array(v)))
        key = (outer, _signature(args))
        ctx = self._contexts.get(key)
        if ctx is None:
            with self._lock:
                ctx = self._contexts.get(key)
                if ctx is None:
                    ctx = self._contexts[key] = self._make_context(outer)
        return ctx, args

    def _make_context(self, outer: tuple[str, ...]) -> _Context:
        """A new signature's context: the current values of the constants the code reads, and the nest
        taking the global arrays it reads as extra parameters."""
        if self._reader is None:
            return _Context(self.fn, self.nest, ())
        names = self._reader.snapshot(n for n in self._read if n not in outer)
        nest = self.nest
        if outer:
            fdef = copy.deepcopy(nest.fdef)
            fdef.args.args += [ast.arg(n) for n in outer]
            nest = frontend.parse_function(fdef, nest.fn_name, nest.first_line)
        return _Context(analysis.Environment(names, self._filename), nest, outer)

    # ------------------------------------------------------------------ entry points

    def lower(self, *args: Any, **kwargs: Any) -> Lowered:
        return self._lower(*self._bind(args, kwargs))

    def tuning(self, *args: Any, **kwargs: Any) -> list[tune_lib.Candidate]:
        ctx, inputs = self._bind(args, kwargs)
        self._lower(ctx, inputs)
        return self._tuning.get(ctx, [])

    def source(self, *args: Any, **kwargs: Any) -> str:
        return self.lower(*args, **kwargs).source

    def reference(self, *args: Any, **kwargs: Any) -> Any:
        """Run the Python loop nest with the kernel's `@` semantics: MXU dots accumulating in f32 / i32.

        Plain JAX would round `bf16 @ bf16` to bf16 at every step and wrap `int8 @ int8`; the kernel keeps
        the MXU's wide accumulator, so the reference does too.
        """
        return self._run_reference(*self._bind(args, kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*self._bind(args, kwargs))

    def verify(self, *args: Any, rtol: float | None = None, atol: float | None = None, **kwargs: Any) -> float:
        """Run the kernel and `reference` and raise if they disagree; returns the max abs error.

        `rtol` and `atol` override the tolerances the dtypes allow, unless the kernel has a parameter of
        that name: the keyword then passes that argument, and the tolerance keeps its default.
        """
        for name, value in (("rtol", rtol), ("atol", atol)):
            if name in self.nest.params and value is not None:
                kwargs[name] = value
        rtol = None if "rtol" in self.nest.params else rtol
        atol = None if "atol" in self.nest.params else atol
        ctx, inputs = self._bind(args, kwargs)
        return self._check(self._call(ctx, inputs), self._run_reference(ctx, inputs), rtol, atol)

    def _check(self, got: Any, want: Any, rtol: float | None = None, atol: float | None = None) -> float:
        return check_outputs(self.__name__, got, want, self.options.matmul_precision, rtol, atol)

    # ------------------------------------------------------------------ building

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

    def _lower(self, ctx: _Context, inputs: tuple[Any, ...]) -> Lowered:
        options = self.options
        # Measuring without tuned names still picks the buffer depth: the search then has one combination.
        measuring = options.autotune in ("measure", "profile")
        search = bool(options.tune) or measuring
        lowered = self._lowered.get(ctx)
        if lowered is not None and (not measuring or ctx in self._measured):
            return lowered
        with self._lock:
            if ctx not in self._lowered:
                specs = [_spec(a) for a in inputs]
                if search:
                    self._search(ctx, specs)
                else:
                    self._lowered[ctx] = lower(ctx.nest, ctx.env, specs, options)
                    if hardware.on_tpu() and cost_lib.overhead_bound(self._lowered[ctx].estimate):
                        warnings.warn(
                            f"autokernel: {self.__name__}'s steps are too small to hide their fixed per-step cost "
                            f'({self._lowered[ctx].steps} steps); larger blocks would run faster: try tune="auto"',
                            stacklevel=4,
                        )
            # Measuring needs real arrays on a real TPU: do it on the first call that has them.
            measure = search and measuring and ctx not in self._measured
            if measure and _concrete(inputs) and hardware.on_tpu():
                self._measured.add(ctx)
                arrays = tuple(jax.numpy.asarray(a) for a in inputs)
                check = self._accuracy_check(ctx, arrays)
                if options.autotune == "profile":
                    best = self._profile_guided(ctx, inputs, arrays, check)
                else:
                    best = tune_lib.measure(self._tuning[ctx], arrays, options.measure_top, check)
                if best is not None:
                    self._lowered[ctx] = best
                    seconds = next(c.measured_s for c in self._tuning[ctx] if c.lowered is best)
                    self._remember(ctx, best, seconds, self._history.get(ctx, {}))
            return self._lowered[ctx]

    def _profile_guided(
        self, ctx: _Context, inputs: tuple[Any, ...], arrays: tuple[Any, ...], check: Callable
    ) -> Lowered | None:
        """`autotune="profile"`: `tune.guided` over this signature's candidates, timed and diagnosed from the
        device's own trace, continuing from the measurements earlier runs left in the tuning cache."""
        from . import profiling

        options, specs = self.options, [_spec(a) for a in inputs]
        if self._tuning[ctx] and self._tuning[ctx][0].cached:  # a cached pick: search again to explore from it
            self._tuning[ctx] = tune_lib.search(ctx.nest, ctx.env, specs, options)
        history = self._history.setdefault(ctx, dict(self._cached_history.get(ctx, {})))

        def make(c: tune_lib.Candidate) -> Lowered:
            return lower(ctx.nest, ctx.env, specs, options, c.overrides, c.buffers)

        return tune_lib.guided(
            self._tuning[ctx],
            arrays,
            options.measure_top,
            make,
            lambda lw, xs: profiling.profile_lowered(lw, xs).device_s,
            lambda lw, xs: profiling.profile_lowered(lw, xs, runs=3, trace=True, options=options).bottleneck,
            check,
            history,
        )

    def _search(self, ctx: _Context, specs: list[jax.ShapeDtypeStruct]) -> None:
        """Tune the kernel for a new signature: the pick of an earlier search from the tuning cache when there
        is one (a measured one, if this search measures), else a search, whose pick is cached."""
        options = self.options
        key = self._cache_keys[ctx] = self._cache_key(ctx, specs)
        entry = cache_lib.load(key)
        if entry is not None:
            self._cached_history[ctx] = dict(entry.history)
        if entry is not None and (entry.measured_s is not None or options.autotune == "model"):
            try:
                lowered = lower(ctx.nest, ctx.env, specs, options, entry.overrides, entry.buffers)
            except AutoKernelError:  # not a valid pick any more: search again
                lowered = None
            if lowered is not None:
                candidate = tune_lib.Candidate(
                    dict(entry.overrides), entry.buffers, lowered, None, entry.measured_s, lowered.estimate, True
                )
                self._tuning[ctx], self._lowered[ctx] = [candidate], lowered
                # A measured pick is final for "measure"; "profile" carries on exploring from it when it runs.
                if entry.measured_s is not None and options.autotune == "measure":
                    self._measured.add(ctx)
                return
        self._tuning[ctx] = tune_lib.search(ctx.nest, ctx.env, specs, options)
        self._lowered[ctx] = self._tuning[ctx][0].lowered  # the cost model's pick
        self._remember(ctx, self._lowered[ctx], None)

    def _cache_key(self, ctx: _Context, specs: list[jax.ShapeDtypeStruct]) -> str:
        """The tuning cache's key for this signature: the nest's code, the constants it reads and their values
        (tuned ones included), the inputs, the chip and the options."""
        used = {n.id for n in ast.walk(ctx.nest.fdef) if isinstance(n, ast.Name)}
        constants = {n: v for n, v in ctx.env.names.items() if n in used}
        constants["<views>"] = sorted(ctx.env.views)
        hw = hardware.target(self.options.target)
        chip = f"{hw.name}:{jax.devices()[0].device_kind}" if hardware.on_tpu() else f"{hw.name}:planned"
        return cache_lib.key(ctx.nest.fdef, constants, specs, chip, self.options)

    def _remember(
        self, ctx: _Context, lowered: Lowered, measured_s: float | None, history: dict[str, float] | None = None
    ) -> None:
        key = self._cache_keys.get(ctx)
        if key is not None:
            entry = cache_lib.Entry(dict(lowered.overrides), lowered.buffers, measured_s, dict(history or {}))
            cache_lib.store(key, entry)

    def _accuracy_check(self, ctx: _Context, arrays: tuple[Any, ...]) -> Callable[[Lowered], str | None]:
        """`check(lowered)` for `tune_lib.measure`: run a candidate on `arrays` and compare it with the Python
        loops (with the constants as written, run once), as `verify` does."""
        want: list[Any] = []
        interpret = self._interpret_mode()

        def check(lowered: Lowered) -> str | None:
            if not want:
                want.append(self._reference_call(self._build_reference(ctx, {}), arrays))
            got = jax.jit(functools.partial(lowered.entry, interpret=interpret))(*arrays)
            try:
                self._check(got, want[0])
            except AssertionError as e:
                return str(e)
            return None

        return check

    def _run_reference(self, ctx: _Context, inputs: tuple[Any, ...]) -> Any:
        # A tuned kernel runs with the tuned constants: so must the loops it is compared with. Without
        # tuning there is nothing to lower: the loops run even where no kernel can be planned.
        overrides = self._lower(ctx, inputs).overrides if self.options.tune else {}
        key = (ctx, _typed(overrides))
        reference = self._reference.get(key)
        if reference is None:
            with self._lock:
                reference = self._reference.get(key)
                if reference is None:
                    reference = self._reference[key] = self._build_reference(ctx, overrides)
        return self._reference_call(reference, inputs)

    def _reference_call(self, reference: Callable, inputs: tuple[Any, ...]) -> Any:
        """Run a reference built by `_build_reference` at the kernel's matmul precision."""
        if self.options.matmul_precision == "default":
            return reference(*inputs)
        with jax.default_matmul_precision("highest"):
            return reference(*inputs)

    def _build_reference(self, ctx: _Context, overrides: dict[str, Any]) -> Callable:
        arrays = {*ctx.nest.params, *ctx.nest.out_vars}
        fdef = mxu.rewrite_matmuls(reference_lib.ragged_tiles(reference_lib.traceable_masks(ctx.nest), arrays))
        fdef.decorator_list = []
        namespace = dict(ctx.env.names)
        fixed = {*overrides, *ctx.env.tuned}
        tuned = {id(s) for name, s in tiles.declarations(fdef.body, namespace).items() if name in fixed}
        fdef.body = [s for s in fdef.body if id(s) not in tuned]  # tuned block sizes keep the tuned value
        module = ast.Module(body=[fdef], type_ignores=[])
        ast.increment_lineno(module, ctx.nest.first_line - 1)
        namespace.update(overrides)
        namespace.update(mxu.reference_helpers())  # `@` as jnp.matmul, independent of the kernel's helpers
        namespace.update(reference_lib.helpers())
        exec(compile(module, ctx.env.filename, "exec"), namespace)
        return namespace[ctx.nest.fn_name]

    def _call(self, ctx: _Context, inputs: tuple[Any, ...]) -> Any:
        lowered = self._lower(ctx, inputs)
        if lowered.scalars:  # runtime indices whose values are known now: check them against the inputs
            known = {p: a for p, a in zip(ctx.nest.params, inputs, strict=True) if p in lowered.scalars}
            lowered.check_indices(known)  # traced ones: a checkify check (the kernel clamps them)
        found = sharding.find_mesh(inputs)
        if found is None:
            ckey: tuple = (ctx, None)
            compiled = self._compiled.get(ckey)
        else:
            compiled = None
        if compiled is None or compiled[0] is not lowered:  # first call, or measuring changed the pick
            with self._lock:
                shard_plan = self._shard_plan(ctx, lowered, inputs, found)
                ckey = (ctx, shard_plan.key if shard_plan is not None else None)
                compiled = self._compiled.get(ckey)
                if compiled is None or compiled[0] is not lowered:
                    compiled = self._compiled[ckey] = self._compile(ctx, lowered, shard_plan)
        out = compiled[1](*inputs)
        if self.verify_on_first_call and ctx not in self._verified and _concrete(inputs):
            self._check(out, compiled[2](*inputs))
            self._verified.add(ctx)  # only once it agrees: a mismatch is reported on every call
        return out

    def _compile(
        self, ctx: _Context, lowered: Lowered, shard_plan: sharding.ShardPlan | None
    ) -> tuple[Lowered, Callable[..., Any], Callable[..., Any]]:
        """The kernel with its derivative and batching rule, jitted, and the reference it is checked with."""
        if shard_plan is None:
            forward = jax.jit(functools.partial(lowered.entry, interpret=self._interpret_mode()))
            reference = functools.partial(self._reference_of, ctx)
        else:  # each device runs its shard's kernel, and differentiates its shard's loops
            local = self._local_kernel(ctx, lowered, shard_plan)
            if local is None:  # nothing is split: every device runs this kernel on gathered inputs
                run, run_reference = self._runner(ctx), functools.partial(self._reference_of, ctx)
            else:
                run, run_reference = local, local.reference
            forward = sharding.run(shard_plan, run, self.nest.returns_tuple)
            reference = sharding.run(shard_plan, run_reference, self.nest.returns_tuple)
        backward = self._backward(ctx)
        vmap_rule = functools.partial(self._vmap_rule, ctx)
        entry = autodiff.differentiable(forward, reference, backward, vmap_rule=vmap_rule)
        # jit: one cached trace of the wrappers rather than one per eager call
        return lowered, jax.jit(entry), reference

    def _runner(self, ctx: _Context) -> Callable[..., Any]:
        return lambda *inputs: self._call(ctx, inputs)

    def _reference_of(self, ctx: _Context, *inputs: Any) -> Any:
        return self._run_reference(ctx, inputs)

    def _backward(self, ctx: _Context) -> Callable | None:
        """The user's `backward` for the kernel's inputs: no gradients for the global arrays it also reads."""
        if self.backward is None or not ctx.outer:
            return self.backward
        user, backward = len(self.nest.params), self.backward

        def with_outer(*args: Any) -> Any:
            grads = backward(*args[:user], args[-1])
            grads = tuple(grads) if isinstance(grads, (tuple, list)) else (grads,)
            return (*grads, *(None for _ in ctx.outer))

        return with_outer

    # ------------------------------------------------------------------ jax.vmap

    def _vmap_rule(self, ctx: _Context, axis_size: int, in_batched: list[bool], *args: Any) -> tuple[Any, Any]:
        """custom_vmap rule: one kernel over the whole batch (`batching.batched_nest`), or one call per batch
        element when the nest can't be batched that way."""
        kernel = self._batched_kernel(ctx, tuple(in_batched), axis_size, args)
        if kernel is not None:
            out = kernel(*args)
        else:
            out = batching.sequential(self._runner(ctx), in_batched, args)
        return out, jax.tree.map(lambda _: True, out)

    def _batched_kernel(
        self, ctx: _Context, in_batched: tuple[bool, ...], size: int, args: tuple[Any, ...]
    ) -> TpuKernel | None:
        for p, a, b in zip(ctx.nest.params, args, in_batched, strict=True):
            if b and analysis.is_scalar_spec(jax.ShapeDtypeStruct(tuple(a.shape)[1:], a.dtype)):
                self._warn_sequential(f"its runtime scalar `{p}` is batched")
                return None
        bkey = (ctx, in_batched, size)
        with self._lock:
            kernel = self._batched.get(bkey)
            if kernel is None:
                try:
                    nest = batching.batched_nest(ctx.nest, in_batched, size)
                except batching.BatchingError as e:
                    self._batched[bkey] = e
                    self._warn_sequential(str(e))
                    return None
                names = {**ctx.env.names, batching.ALLOC: batching.batched_alloc}
                kernel = self._batched[bkey] = TpuKernel(
                    analysis.Environment(names, ctx.env.filename, tuned=ctx.env.tuned),
                    nest=nest,
                    interpret=self.interpret,
                    options=self.options,
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

    def _shard_plan(
        self, ctx: _Context, lowered: Lowered, args: tuple[Any, ...], found: tuple | None
    ) -> sharding.ShardPlan | None:
        """How to run on `args` when some are sharded over a mesh (`found`: `sharding.find_mesh(args)`;
        None when none is)."""
        if found is None or lowered.plan is None:
            return None
        key = (id(lowered), found[0], tuple(found[1]))
        if key in self._shard_plans:
            return self._shard_plans[key]
        shard_plan = self._shard_plans[key] = sharding.plan(lowered.plan.spec, *found)
        if shard_plan.loops:
            try:  # the per-shard kernel must exist for the per-shard shapes
                local = self._local_kernel(ctx, lowered, shard_plan)
                assert local is not None
                local.lower(*sharding.local_specs(shard_plan, args))
            except AutoKernelError as e:
                shard_plan = sharding.gather_all(lowered.plan.spec, shard_plan, f"its per-shard kernel fails: {e}")
                self._shard_plans[key] = shard_plan
        return shard_plan

    def _local_kernel(self, ctx: _Context, lowered: Lowered, shard_plan: sharding.ShardPlan) -> TpuKernel | None:
        """The kernel each device runs on its shards (None when nothing is split: this kernel itself)."""
        if not shard_plan.loops:
            return None
        key = (id(lowered), shard_plan.loops)
        kernel = self._local.get(key)
        if kernel is None:
            nest, extra = sharding.local_nest(lowered.plan.spec, shard_plan)
            names = {**ctx.env.names, **lowered.overrides, **extra}
            # The shard's kernel keeps the tuned tiles and buffer depth: nothing to search or measure.
            options = dataclasses.replace(self.options, tune=(), autotune="model", buffers=lowered.buffers)
            env = analysis.Environment(names, ctx.env.filename, tuned=ctx.env.tuned | frozenset(lowered.overrides))
            kernel = self._local[key] = TpuKernel(env, nest=nest, interpret=self.interpret, options=options)
        return kernel


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
            sizes on the TPU on the first call with concrete arrays, then 2 and 3 buffers for the winner
            (without `tune`, just the 2 and 3 buffers of the kernel as written); "profile" times candidates by
            their device time in a profiler trace and follows the bottleneck it finds (exposed DMA, per-step
            overhead, compute) to the next one to try, keeping every measurement in the tuning cache so each
            run continues from the last (see `tune.guided`; best with `autokernel.enable_profiling()`).
        buffers: VMEM slots per streamed input (2 = double buffering). Default: 3 where they fit in VMEM,
            else 2; `autotune="measure"` tries both.
        target: the TPU to plan for when not running on one, e.g. "v5e" (default) or "v5p".
        cores: TensorCores to split the work across (default: all of the target's, 2 on v4 and v5p).
        check_tiling: reject tiles that break TPU (8, 128) layout rules before Mosaic does.
        vmem_limit_bytes: Mosaic's scoped-VMEM limit; by default the target chip's (`Hardware.vmem_limit`,
            90% of its VMEM), not the kernel's estimated use, which must fit in 75% of it.
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
    if isinstance(tune, Mapping):
        for name, values in tune.items():
            if not isinstance(values, (str, Iterable)):
                raise AutoKernelError(
                    f"tune[{name!r}] must be 'auto' or a sequence of candidate values, got {values!r}"
                )
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
