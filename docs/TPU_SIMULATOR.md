# Should autokernel have a TPU simulator? Feasibility, risks and a plan

*Research report, 2026-10-19 (JAX 0.11.2). Research only: nothing here has been implemented yet.*

## Summary

**The short answer:** most of the bugs we call "TPU-only" can be caught without a TPU. The way to do it is not
a simulator. It is to run the real compiler, libtpu, in compile-only mode on a CPU host. That is the cheapest
and highest-value step. Real timing simulation is possible but expensive, and only worth building after a
proper calibration harness shows where the current roofline model is wrong.

Three findings shape everything below. All were checked against the installed JAX.

1. **Today's offline cross-lowering catches none of the known TPU-only bugs.** Six minimal Pallas kernels
   were lowered for "TPU v5 lite" with the AbstractMesh trick `tests/test_mosaic_lowering.py` uses. Every one
   lowered without error:
   - f32 @ bf16 `dot_general`
   - int8 vector add
   - i16 scalar multiply
   - int32 @ int32 matmul
   - a 256 MiB VMEM scratch buffer
   - a 4 MiB SMEM scratch buffer

   On a real v5e every one of them fails. The JAX-side lowering only runs the Pallas lowering rules and the
   open-source TPU dialect verifiers (`jax/_src/pallas/mosaic/lowering.py` ~1616, 1662). The checks that
   reject these kernels live elsewhere.
2. **The checks that reject them ship only inside libtpu.** JAX commit `43d14afa` (2025-11-19), "[Mosaic TPU]
   Remove the Mosaic TPU passes from the JAX repository", moved canonicalize-mosaic, infer-memref/vector-layout
   and apply-vector-layout into libtpu. jaxlib 0.11.2 registers only the TPU dialect and `mosaic-serde`. The
   last open snapshot of `canonicalize_mosaic.cc` (`cff5c5f6`) has the exact rules we hit:
   - "Only i1 and i32 scalars are supported" / "Only i32 addition…"
   - "Only vector<i16> and vector<i32>…"
   - "vector<i16> addition is only supported on hardware generation v6+" (unless compatibility mode is on)
   - "Mosaic matmul invoked with mixed element types"
   - "Mix int/int - NYI"
3. **libtpu can compile for a TPU on a machine that has no TPU.**
   `jax.experimental.topologies.get_topology_desc(...)` (`jax/_src/xla_bridge.py` ~1184) loads the libtpu
   PJRT plugin and builds compile-only devices. MaxText documents AOT compilation for a TPU topology on a
   CPU-only host, and says it flags out-of-memory errors.
   - libtpu wheels exist only for `manylinux x86_64`. So this runs on a Linux x86 CI runner, a Colab CPU
     runtime, or a container, not natively on an arm64 Mac.
   - **Not yet verified for Pallas `tpu_custom_call`s on a v5e topology.** That is milestone M0.

Related: the exact pre-layout Mosaic module libtpu receives can already be pulled out offline on the Mac. The
steps are: base64-decode `custom_call_config.body`, then parse it with the jaxlib TPU dialect and
`mosaic-serde{serialize=false}`. The result shows e.g. `tpu.matmul … vector<128x128xf32>, vector<128x128xbf16>`
and `arith.muli (i16, i16)`, which makes a Mac-native Mosaic IR linter possible as a fallback.

On the cost model: its "about 10%" accuracy is measured on the same points the constants were fitted to
(`tests/test_cost.py`: +1.8, +4.8, −0.8, +2.2, **−15.5**, +1.3, **+9.6**, +6.6%). There is no held-out validation.
The first performance task is therefore a calibration harness with a held-out set, not a simulator.

---

## 1. Feasibility

### 1(a) Correctness

| Layer | Runs off-TPU today? | What it checks |
|---|---|---|
| autokernel planning (`analysis`, `cost`, `hardware`) | yes | Hand-coded rules: (8,128) tiling, integer widths the vector unit computes on, 32-bit scalars, SMEM budget, VMEM estimate, primitives Pallas can't lower |
| Pallas → Mosaic lowering | yes (`test_mosaic_lowering.py`) | Unsupported primitives, ref transforms, some casts, dialect verifiers |
| `mosaic-serde` | yes | Serialization only |
| **libtpu: canonicalize, layout inference, apply-vector-layout, LLO, scheduling, VMEM/SMEM allocation** | **no** | i32-only scalar unit, mixed-dtype matmul, 8/16-bit vector arithmetic per generation, relayouts, VMEM/SMEM out-of-memory including Mosaic's internal scratch |
| libtpu runtime | needs a TPU | On-device bounds and semaphore checks, numerics, timing |

The Pallas TPU interpreter works at the jaxpr level, before Mosaic:

- A primitive without a TPU-specific implementation runs with `prim.bind` on CPU. So f32@bf16, int8 adds and
  i16 scalar multiplies just run.
- It models no memory capacities.
- Padding is optional (`buffer_bounds="padded"`).
- A semaphore left non-zero at exit is only **printed** (`_validate`, "TODO: make this raise").
- DMAs, semaphores, races (vector clocks), `dma_execution_mode="on_wait"` and multiple cores are modelled well.

| Bug class | Static rules | Mosaic IR lint (Mac) | **libtpu compile-only (x86)** | Stricter interpreter | Needs hardware |
|---|---|---|---|---|---|
| i16/i8 scalar arithmetic | yes (already) | yes | **yes, authoritative** | awkward | no |
| Mixed-dtype `tpu.matmul` | yes (promotion) | yes | **yes** | possible | no |
| 8/16-bit vector integer arithmetic | partly (`VPU_INT_BITS` is a guess) | yes, with a rule table | **yes** | possible | no |
| int32×int32 matmul on v5e | yes (rule) | yes | **yes** | possible | no |
| SMEM overflow | approximately | no | **yes, exact** | no | no |
| VMEM overflow incl. internal scratch and spills | approximately | no | **yes, exact** (+ `memory_analysis()`) | no | no |
| Unsupported relayouts / "Not implemented" in apply-vector-layout | no | no | **yes** | no | no |
| DMA shape/alignment legality | partly | partly | **yes** | partly | runtime out-of-bounds only |
| Semaphore balance | partly | no | no | **yes** (make `_validate` fatal) | on-device checks |
| Races, megacore cross-core races | no | no | no | **yes** (already) | no |
| Megacore split compiles (v5p) | no | no | **yes** | functional only | real 2-TensorCore runs |
| bf16 1-ulp differences, EUP approximations | no | no | no | no | **yes** |
| Barrier and dynamic-offset slowdowns | no | maybe | visible in LLO, if dumps work offline | no | timing |
| Runtime hangs / out-of-bounds halts | partly | no | no | yes | yes |

**Rule tables drift.** JAX's own `tpu_info.is_matmul_supported` claims v5e accepts f32 @ bf16, but Mosaic on a
v5e rejected exactly that. Old Mosaic source only allows i16 vector addition on v6+, while autokernel assumes
v5e computes on 16-bit vectors. **Only libtpu is authoritative.**

**Still needs hardware:**
- numerics
- timing
- runtime behaviour
- real two-TensorCore execution (Colab offers v5e only)
- ICI/remote DMAs

### 1(b) Performance

- `Compiled.cost_analysis()` returns the `pl.CostEstimate` that autokernel writes into the kernel, so for
  Mosaic kernels it is circular.
- `Compiled.memory_analysis()` gives real temp sizes once compiled.
- XLA's TPU analytical model covers XLA fusions, not Mosaic bodies. Google's published numbers (Kaufman et
  al., MLSys'21) are 6.1% mean error on tile size and 31.1% fusion MAPE for the production analytic model.
- **LLO dumps** (`--xla_jf_dump_llo_text`, `--xla_mosaic_dump_to`) are produced at compile time, so possibly
  also by a compile-only libtpu on a CPU host (unverified, M0b). A TensorCore is in-order, statically
  scheduled VLIW, so bundle count × clock is close to a step's compute time. This would be the most valuable
  timing signal, but the flags are undocumented.
- **No public functional or timing TPU simulator exists.** Academic ones model the wrong layer:
  - SCALE-Sim (systolic arrays)
  - LLMCompass (analytic, ~10.9% operator error, the same class as our roofline)
  - Timeloop, ONNXim, Ramulator

  GPU cycle simulators with public ISAs (Accel-Sim) still show ~36% MAPE on Hopper. Helion's answer to
  performance is measurement, not simulation.

**A useful simulator here would be an event model at the `Schedule`/`Specialization` level.** It would replay
the exact step sequence with these resources:
- **Scalar issue core:** decode and descriptor-issue costs.
- **DMA engines:** K streams sharing HBM bandwidth, a latency, and an efficiency that depends on row length,
  alignment and static vs dynamic offsets.
- **Compute per unit:** from jaxpr counts or LLO bundles.
- **Semaphore waits:** treated as barriers with a fitted spill penalty.
- **Pipeline:** fill, drain and launch costs.
- **Megacore:** two cores sharing HBM.

Python is fast enough. Keep the closed form for pruning and simulate the top ~32 candidates.

**Calibration on Colab v5e:**
- **Microbenchmark families:** DMA-only, empty steps with 0–4 copies, MXU-only, VPU/EUP-only, wait-barrier,
  fill/drain.
- **Fit:** robust log-least-squares.
- **Validate on a held-out set:** MAPE, max error, Kendall τ per sweep, and top-1 regret (the metric the tuner
  cares about).
- **Budget:** ~30–45 min per session, run resumably.

**Expected accuracy (estimates):**

| Model | Matmul | Attention (VPU/EUP-heavy) | Masked/fused | Top-1 regret |
|---|---|---|---|---|
| Current roofline, in-sample | 1–5% | up to 15.5% | unknown | good on sampled sweeps |
| Roofline refit on microbenchmarks + held-out set | 2–5% | 8–15% | 8–15% | ≤3–5% |
| Event simulator + jaxpr compute | 2–4% | 7–12% | 4–8% | ≤2–3% |
| Event simulator + offline LLO bundles | 2–3% | 3–6% | 3–6% | ≤2% |

---

## 2. Private API dependencies

The repo pins `jax<0.12`, and JAX 0.12.0 was released 2026-10-19.

| API | Status | Risk |
|---|---|---|
| `jax.experimental.topologies.get_topology_desc`, `.lower().compile()`, `memory_analysis()` | experimental-public / public, used by MaxText | **Low** |
| libtpu ↔ jax version | lockstep with `jax[tpu]==X` | Low if pinned together |
| `LIBTPU_INIT_ARGS` dump flags, LLO text format | undocumented | **High**: tooling only |
| jaxlib TPU dialect + `mosaic-serde` (IR linter) | semi-internal, versioned serde; Mosaic is moving into openxla/xla | Medium |
| Custom-call body format | internal JSON | Medium |
| `jax._src.pallas.mosaic.interpret.*` (already used by tests) | private, ~140 commits in two years, race semantics changed 2026-10 | **High** |
| `jax._src.tpu_info` | private, moved 2026-07 | High |
| Mosaic passes from Python | gone since 2025-11 | not an option |

---

## 3. Options

| # | Option | Effort (days) | TPU-only bugs caught | Timing error after | Main risks |
|---|---|---|---|---|---|
| **i** | **libtpu compile-only check** (`compile_for(chip, kernel, *specs)` on x86 Linux; plus `memory_analysis()`; optional dump capture) | **2–4** (spike: 0.5) | **~75–85%** (every compile-time class, exact SMEM/VMEM, relayouts, v5p megacore compile) | unchanged | Unverified for Pallas on v5e; needs x86; libtpu download size; seconds per compile |
| i-b | Mosaic IR linter on the Mac (decode the custom-call body, per-generation rule table ported from `canonicalize_mosaic.cc`) | 3–5 | ~45–55% | — | Rule drift, false positives |
| ii | Strict interpreter (fatal semaphore imbalance, padded bounds, NaN-initialised memory, races, both DMA modes) | 1 for the useful part | +5–10% beyond (i) | — | Most-churned private code; leave dtype rules to (i) |
| iii | Event simulator of the gridless loop | MVP 10–15, +8–15 with LLO | 0 | 4–8% (3–6% with LLO) on structural cases | Fragile formats; over-engineering |
| **iv** | **Calibration harness** on Colab v5e (microbenchmarks + held-out kernels, resumable, refit `cost.py`, report) | **4–7** | 0, but re-checks correctness on hardware | refit: matmul 2–5%, attention 8–15%, and for the first time a known held-out number | Colab availability; noise |
| v | Status quo | 0 | 0 | unknown outside the fitted points | — |

---

## 4. Recommendation

**Is it a good idea?**
- A cycle-accurate executing simulator: **no.** There is no public ISA, the Mosaic passes are closed, and
  even public-ISA GPU simulators are far off.
- What we actually want, *catching TPU-only bugs offline* and *trustworthy estimates*, is achievable:
  - the first by **running the real compiler offline**;
  - the second by **honest held-out calibration**, then a `Schedule`-level event simulator only if the data
    justifies it.

**The order to do it in:**
1. **M0 spike (½ day):** on x86 Linux (a Colab CPU runtime or GitHub Actions), install `jax[tpu]==0.11.2`,
   then `get_topology_desc` for v5e/v5p and compile `test_mosaic_lowering.CASES` plus the six failure
   fixtures. **M0b:** the same with the dump flags. Go/no-go:
   - if it works, do (i);
   - otherwise do i-b plus tighter static rules.
2. **(i) offline compile check (2–4 days):**
   - a `mosaic_compile` pytest marker over v5e/v5p in CI;
   - `kernel.compile_check(*specs, target=...)`, which the tuner uses to drop candidates Mosaic would reject
     before measuring;
   - replace the guessed `SMEM_RESERVE`, VMEM budget and `VPU_INT_BITS` with compile-verified facts.
3. **(ii)-lite (1 day):** fail on semaphore imbalance in test helpers; fuzz with `buffer_bounds="padded"` and
   both DMA modes.
4. **(iv) calibration harness (4–7 days):** held-out set of ≥30 kernels, refit, report, auto-generated
   regression bounds, re-run on each JAX/libtpu bump.
5. **Decision point:**
   - if held-out MAPE ≤ 8% and top-1 regret ≤ 3%, **stop** (refit roofline + `autotune="measure"`);
   - otherwise, and before roadmap items 17/18/20, build (iii), adding LLO ingestion only if M0b succeeded.

**What not to do:**
- Don't build a cycle-accurate TensorCore/MXU simulator, or adopt SCALE-Sim, Timeloop or Ramulator.
- Don't re-host the frozen open Mosaic passes.
- Don't monkeypatch the interpreter's evaluation loop for dtype rules.
- Don't make undocumented dumps or LLO parsing a hard dependency.
- Don't chase bit-exact numerics.
- Don't quote in-sample accuracy: restate the README's "within about 10%" once (iv) exists.
- Don't let a simulator replace measuring the top few candidates.

**Acceptance tests:**
- **A0:** on TPU-less x86 Linux, all `test_mosaic_lowering.CASES` compile for v5e and the split cases for v5p.
- **A1:** six fixtures must **fail** compile-only with hardware-matching messages:
  - i16 scalar `muli`
  - f32@bf16 matmul
  - int8 vector add
  - int32@int32 on v5e
  - ~1026 KiB of SMEM tables
  - VMEM above the limit

  961 KiB of SMEM must **pass**.
- **A2:** for ≥50 `test_fuzz` kernels, the offline compile verdict equals the real-v5e verdict 100% of the
  time.
- **A3:** an unbalanced-semaphore kernel fails the test helper.
- **A4:** report over ≥30 held-out kernels with MAPE ≤ 8%, max ≤ 20%, regret ≤ 3%; `test_cost.py` bounds
  generated from it.
- **A5 (only if iii is built):** ≥30% relative MAPE cut on the attention and masked/fused families, matmul not
  worse; otherwise delete it.

**Limits:** nothing offline will show real numerics, runtime hangs or megacore behaviour. The libtpu
compile-only path is strongly suggested by JAX's API and MaxText's documentation but unconfirmed for Pallas
on v5e until M0.

### Sources
- MaxText, AOT compile on CPU: https://maxtext.readthedocs.io/en/latest/guides/monitoring_and_debugging/features_and_diagnostics.html
- libtpu wheels: https://pypi.org/pypi/libtpu/json
- JAX `tests/aot_test.py`: https://github.com/jax-ml/jax/blob/main/tests/aot_test.py
- Mosaic passes removed from JAX: https://github.com/jax-ml/jax/commit/43d14afad3e6bc321309054c512c5ffe1d1bca86
- Mosaic TPU dialect in XLA: https://github.com/openxla/xla/tree/main/xla/mosaic/dialect/tpu
- Pallas changelog: https://github.com/jax-ml/jax/blob/main/docs/pallas/CHANGELOG.md
- LLO dump flags: https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation
- XProf deep kernel profiling: https://opensource.googleblog.com/2026/06/unlocking-tpu-performance-deep-kernel-profiling-with-xprof.html
- Kaufman et al., A Learned Performance Model for TPUs: https://arxiv.org/abs/2008.01040
- SCALE-Sim v3: https://arxiv.org/abs/2504.15377 · LLMCompass: https://arxiv.org/abs/2312.03134 · ONNXim: https://arxiv.org/abs/2406.08051
- Accel-Sim (ISCA'20): https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf
