# Tracer-transport emulator on JCM p11a — specification

**Status:** draft, 2026-09-17. Supersedes the transport-head sections of
`project_jax_gcm_transport_head` (the 2026-09-03 scope note that dropped the head).
Corpus: `/data/JCM_stripped/jcm-strat-phase11/runs/p11a_*` (run A, 1990–2009).
Companion: `docs/outputs/11_lid_tracers/output.md` in that repo.

---

## 1. Goal

Given an initial condition for one or more tracer fields, predict their transport
across every grid cell, autoregressively, for multi-year horizons — and predict the
underlying state (u, v, T, ω, p_s) that drives it.

The emulator must generalise to **tracer initial conditions it has never seen**.
That requirement, not accuracy alone, determines the architecture (§3).

## 2. The corpus, as measured

Verified 2026-09-17 against the archive, not from the run records.

| property | value |
|---|---|
| period | 1990-01-01T06 → 2010-01-01T00, 6-hourly instantaneous |
| frames | 29,220, contiguous — no gaps or duplicates across chunk **or** year seams |
| grid | T63 Gaussian, 192 lon × 96 lat (1.875°), 63 hybrid levels **surface-first**, 985 → 0.018 hPa |
| size | 2.4 TB (~120 GB/yr compacted), 740 chunk files, symlinked into `runs/p11a_20yr/` |
| health | zero NaN, no negative tracers, T and u ranges stable to year 20 |
| restart | `checkpoint.ckpt` per year (133 MB) |
| dynamics channels | `u_wind`, `v_wind`, `temperature`, `omega`, `geopotential`, `specific_humidity`, `normalized_surface_pressure` |
| tracers | 24: 5 Gaussian pulses + 5 box twins, 4 Gaussian sources + 4 box twins, `sai`, `n2o`, `cfc11`, `aoa`, `aoa150`, `aoa_sfc` |

### 2.1 What is prescribed and what is free

The ERA5 nudging mask is a **hard binary cutoff**, no taper
(`jcm/runners.py:450`: `mask[p_ref < min_pressure_hpa*100] = 0`, plus the bottom two layers).

| levels | pressure | status |
|---|---|---|
| 0–8 | 985 → 151.5 hPa | nudged to ERA5, τ 6 h, on u, v **and** T |
| **9–58** | **137.9 → 0.27 hPa** | **free dynamics** |
| 59–62 | 0.155 → 0.018 hPa | Rayleigh sponge, τ 1.5 h, damps u, v, T |

Cutting across the free block:
- **QBO nudge** — zonal-mean u only, |lat| < 15° with cos² taper to 0 at 25°, full weight
  40 → 2.2 hPa falling to zero at 90 and 1 hPa, τ 1 d, mean-preserving.
- **Tracer lid** above 1 hPa — clocks relaxed to WACCM6 age (+110 d offset for `aoa`/`aoa_sfc`),
  n2o/cfc11 to their WACCM state, τ 1 d. Run A applies **no** lid sink to pulses/sources/`sai`.

**The QBO is a boundary condition of this corpus, not a dynamical outcome.** The dry dycore
has no gravity-wave drag and cannot generate a QBO; Phase 8 measured amplitude rising 80 → 92 %
of ERA5 as τ fell 10 d → 1 d. An emulator run free in the tropical stratosphere will have no QBO,
and no architecture fixes that — the information is in a monthly ERA5 file, not in the dynamics.
**Decision: prescribe the QBO as an input channel, exactly as the dycore receives it.**
A self-generating QBO is a corpus problem (a run with GW drag — currently DEFERRED), not a
model problem.

### 2.2 Tracer information content

Run A has **no sink anywhere** for pulses, sources and `sai` (`surface_sink: false`,
`lid_sink_injections: false`). Measured mass-weighted burdens, 1991-01-01 → 2010-01-01:

| tracer | change over 19 yr |
|---|---|
| `pulse_1` / `pulse_1_box` / `pulse_4` / `pulse_5` | +0.07 % / +0.21 % / +0.24 % / −0.05 % |
| `sai`, `src_1` | ×20.13, ×20.15 — exactly the emission line |
| `n2o`, `cfc11` | −4.3 %, −2.3 %, both stationary from 1993 |

Nothing leaves. What decays is **structure**, through mixing:

| std/mean | d100 | yr 1 | yr 2 | yr 5 | yr 10 | yr 20 |
|---|---|---|---|---|---|---|
| `pulse_1` | 1.41 | 0.59 | 0.44 | 0.31 | 0.125 | 0.017 |
| `pulse_4` (3 hPa) | 1.74 | 0.88 | 0.35 | 0.061 | 0.019 | 0.0000 |
| `src_2_box` | 3.62 | 2.26 | 1.83 | 1.29 | 0.95 | 0.60 |

Consequences: the 10 pulses carry signal in **years 1–2 only** (`pulse_4` is uniform to four
digits by 2009); the continuously sourced tracers carry it for all 20. A uniform tracer supervises
nothing, because every transport operator maps a constant to itself. **Loss must be weighted by
each tracer's own spatial std per snapshot**, or the objective is dominated by years 3–20 of
free wins.

### 2.3 Displacement — why a local stencil fails at 6 h

Measured from the archive's own winds, |lat| < 80°:

| | median | p99 | p99.9 | max |
|---|---|---|---|---|
| grid cells per 6 h | 1.7 | 12.7 | 24.3 | 32.5 |

Worst at 3–6 hPa (p99.9 ≈ 31 cells) — the region of interest. A face-flux (scatter) formulation
moves mass one cell per application and therefore cannot represent a 6-h step without ~30
sub-applications or a ruinous closure. A departure-point (gather) formulation is unrestricted in
displacement.

Sub-stepping scales displacement linearly (K=4 → p99.9 6.1; K=8 → 3.0) and was expected to be the
lever that recovers accuracy at 6 h. **B0 measured it and it is not** — see §6.1. One departure
solve per 6 h is enough.

### 2.4 Conservation in the archive is non-local

`jcm_strat/main.py:70` patches `DinosaurDycore._fix_nodal_tracer_mass` to apply a **global
proportional rescale** to every tracer except `aoa*`, `n2o`, `cfc11`, after every one of the
43,800 steps. The SL scheme itself is positivity-exact but not mass-conserving (~1e-3/step).
So the 0.2 % conservation in §2.2 is the fixer's, not the operator's. Targets for an operator
that should conserve locally must be generated with `sl_mass_fixer: false`; alternatively the
clocks and n2o/cfc11 are already fixer-free.

## 3. The design invariant

> **Nonlinear in the flow. Linear in the tracer.**

Transport is a linear map on the tracer — homogeneous of degree one, which the Phase 10 linearity
check established and which is why KEY_DECISIONS #40 dropped amplitudes entirely:

```
q(t+6h) = W(flow) · q(t)  +  S(known sources and sinks)
```

All network capacity goes into `state → W`. Everything downstream of `W` is a linear application
to the tracer field. Three consequences:

1. **Tracers are not channels in the trunk.** They are neither inputs nor outputs of the network.
   A 24 × 63 = 1512-channel block never enters it; the trunk is ~320 channels instead of ~2000.
2. **Generalisation to an unseen IC is structural, not hoped for.** A decoder emitting q′ is a
   nonlinear approximation of a linear map, exact on-distribution and arbitrary off it.
3. **Effective supervision is four orders of magnitude larger.** As a field-to-field problem the
   corpus holds 19 injected trajectories. As an operator problem it holds 24 tracers ×
   29,220 frames × 1.16 M cells, every one an independent constraint on the same `W`.

If the pure operator under-resolves something, the correction must **also be linear in q** — a
flow-dependent diffusion applied to q, a second kernel mode, a wider quadrature — never a free
nonlinear residual on q.

**The one honest ceiling.** The true operator is linear *except* for the quasi-monotone limiter,
which clips the cubic interpolant's negative Lagrange weights near sharp gradients
(`monotone_tracers` in `primitive_equations.py:1928`). A strictly linear class cannot reach that
residual, and it will appear exactly at the box twins' edges. Scale of the effect, from the
earlier interpolation-order measurement: 5-day filament peak 0.54 (linear) vs 0.83 (cubic).
Measure it on the twins before deciding it matters.

## 4. Architecture

```
trunk inputs:  u, v, T, ω, p, Φ_s, p_s, forcings, constants        ≈ 320 channels
                          │
               ADR trunk (advection / diffusion / reaction, residual, N layers)
                          │
          ┌───────────────┴────────────────┐
     state head                      transport head
   u,v,T,ω,p_s @ t+6h            departure geometry per cell
                                         │
   tracers (bypass trunk) ───────────────┴──→ shared remap ──→ + analytic source/sink
```

### 4.1 Carried over from PARADIS

- **`model/advection.py`, `NeuralSemiLagrangian`** — predicts a velocity, builds a departure
  point in rotated coordinates, `grid_sample`s there. `torch.remainder` wraps the departure
  longitude into [0, 2π) before the pixel conversion, so displacement is genuinely unbounded
  and the 24–32 cell reality is fine. **Do not rewrite this.**
- **The ADR trunk** — advection → diffusion → reaction, each residual — for the *state*.
- **`n_time_inputs: 2`** — cheap, and lets the trunk infer a tendency when part of the forcing
  is external.

### 4.2 The one change inside the trunk

PARADIS advects *latent* features with 256 independently learned velocity fields per layer —
2048 independent flows over 8 layers. For passive tracers that is backwards: physics says there
is **one** flow and everything rides it. Keep latent multi-velocity advection in the trunk, but
tap **one physical departure field** out of it and apply that in physical space to all tracers.
That field is what gets pretrained against the offline reference (§6) and what can be plotted
and checked.

### 4.3 Dropped, with cause

| dropped | cause |
|---|---|
| `conservation.py`, `conservation_v5.py` | built for moisture/energy budgets in a moist model; nothing here is moist. Conservation belongs in the remap. Also carried real defects — operated on Δq not q, dead `t/ps_residual` flags, energy silently off through all of Phase 2 |
| `omega_continuity.py` | validated on a script that used Cartesian `wind_x/y` as u/v; and JCM archives ω directly |
| soft sigmoid output clamp | masked a silent collapse before — bit-identical val MSE across epochs while the state saturated against the clamp |
| noise channels / CRPS head | CRPS-as-foundation failed; deterministic first, calibration last |
| `q_residual`, moisture anchor, moisture consistency | no moisture; and these were the v5/v6 recipe regression |
| 10m / 2m / MSLP / TCW surface block | not in the corpus |
| Cartesian `wind_x/y/z` | use native u, v — removes the bug class that invalidated the QBO metrics |

### 4.4 What is machine-learned and what is closed form

**This is a hybrid, not an emulator.** At inference the tracer path executes dinosaur's actual
semi-Lagrangian solver and remap; the network perturbs the departure point. Naming it an "emulator"
without qualification overstates the learned content on the transport side and understates it on
the dynamics side. Full inventory:

| component | kind | detail | measured contribution |
|---|---|---|---|
| **departure-point solve** | **analytic** | `horizontal_departure_points`, Cartesian fixed-point on the unit sphere, 2 iterations | part of the R² 0.8195 base |
| **tracer remap** | **analytic** | `transport_scalar_2d` + `GridInterpolator(grid, 'cubic', 'quasi_monotone')` | part of the same base |
| grid, quadrature weights, hybrid (a, b) → pressure | analytic | T63 192×96×63, matches the archive to 1e-14° | — |
| **trajectory correction** | **LEARNED** | 83,139 params. 13 in → 3 out (Cartesian departure perturbation), `SCALE·tanh`, bounded 0.03 rad ≈ 1 cell. **Weights shared across all 63 levels** | **+0.0420 ± 0.0013** (§6.11) |
| **dynamics (u, v, T, p_s)** | **LEARNED** | ~956 k params. 384 in → 190 out, **levels as channels** (wind couples levels vertically; the transport correction does not) | wind skill **+47.5 %** vs persistence (§6.12) |
| PK relaxation, sponge, nudging mask, QBO nudge | **analytic, NOT YET BUILT** | closed form in `polvani_kushner.py`, `qbo_nudging.py` | §5.3; absence is why §6.13 diverges |
| tracer sources/sinks (clocks, WACCM loss, source shapes, lid) | analytic, not yet built | all prescribed | — |
| vertical remap | analytic, NOT BUILT | needs σ̇ from ω (§8.1) | — |

**Where the accuracy actually comes from.** Of the 0.8616 R² achieved with true winds, the
closed-form operator supplies **0.8195** and the learned correction **0.0420** — the analytic
operator does ~95 % of the transport work. The learned content of this project is concentrated in
the *dynamics*, not the transport.

**Where the speed comes from** (`6.3 ms` per 6-h step, 24 tracers, vs JCM's 220 ms — **35×**,
measured on an H200):
1. **One 6-hour transport application instead of 30 × 12 min.** B0 established that a single
   application captures 83 % of the increment, so the sub-steps are unnecessary. This is most of
   the 35×.
2. **A CNN replaces the spectral dycore, the implicit solve and the physics terms.**

The speedup is *not* from replacing the remap. Keeping it analytic costs almost nothing — it is a
gather — and buys 0.82 of the 0.86.

**Why each boundary sits where it does.**
- *Remap analytic*: B0 measured it as **not** the error source (variance ratio 1.004 — it does not
  diffuse at all). Learning it would add parameters to fix something unbroken and forfeit
  positivity and the maximum principle, which currently come free.
- *Trajectory correction learned*: the 6-hourly archive cannot resolve the sub-step wind history
  (§6.1 finding 5), and that residual **is** a function of the resolved flow, hence learnable.
- *Dynamics learned*: the expensive, genuinely nonlinear part.
- *Forcing analytic*: it is prescribed in the truth run too, and supplies the restoring force whose
  absence produced the §6.13 runaway.

This is the §3 invariant in operational form: **nonlinear in the flow → learned; linear in the
tracer → closed form.**

**Two caveats on the 35×.** It is *stepping* throughput; JCM's end-to-end is 29 min/yr and
output-bound, so a comparably-writing emulator inherits that bottleneck. And it is notional until
the state head is stable — 35× on a model that diverges at day 3 is worth nothing.

## 5. Channels

### 5.1 Trunk inputs (at t and t−6h)

| group | channels | n |
|---|---|---|
| prognostic 3D | `u_wind`, `v_wind`, `temperature`, `omega` | 4 × 63 |
| diagnostic 3D | `geopotential` (input only — hydrostatically determined, **do not predict**) | 63 |
| vertical coordinate | `p = a + b·p_s` per level — state-dependent, better than a level index | 63 |
| 2D state | `normalized_surface_pressure` | 1 |
| forcings | `sin/cos year_progress` (drives the PK season); ERA5 nudging target u, v, T on levels 0–8 + the binary mask; QBO target (zonal-mean u, a (level, lat) field); PK equilibrium `T_eq(φ, p, t)` | ~35 |
| constants | `geopotential_at_surface` (real here, from `terrain.nc`), cos/sin lat, cos/sin lon, Gaussian quadrature weight | 6 |

No time-of-day channel: there is no diurnal forcing (no radiation).

### 5.2 Outputs

| head | channels |
|---|---|
| state | `u_wind`, `v_wind`, `temperature`, `omega` (4 × 63) + `normalized_surface_pressure` (1) |
| transport | departure geometry per cell per level: 2 horizontal + 1 vertical (Δσ), optionally a kernel spread — 3–6 × 63 |
| tracers | **not emitted** — produced by applying the remap, then the analytic source/sink |

Geopotential is derived hydrostatically, saving 63 output channels and making the
hydrostatic relation exact rather than learned.

### 5.3 Known physics, carried explicitly

```
X_{t+6h} = X_t + Δt·[ PK relaxation + sponge + nudging mask + QBO nudge ]   ← analytic
                 + learned residual
```

`polvani_kushner.py` gives `T_eq(φ, p, t)` in closed form with τ_strat = 15 d — a genuine
restoring force, and the "structural per-step budget" the dry-attractor work concluded is the
only thing that escapes an attractor. On levels 0–8 and 59–62 the analytic terms dominate and the
learned residual should come out near zero, which is a free sanity check during training.
**Reproduce the nudging mask as the hard binary cutoff it is — do not smooth it.**

Every non-transport tracer term is likewise analytic and prescribed: the clock's +1 day/day and
its reset region, WACCM's loss frequency for n2o/cfc11, the source shapes for `src_*`/`sai`, the
lid relaxation. So each tracer is `learned remap` + `known source/sink`, with the learned part shared.

## 6. Baseline ladder

Measure in order. Each rung is a decision gate, not a milestone.

**B0 — reference-operator fidelity (no learning). DONE, 2026-09-17 — see §6.1.**
Drive the *real* dinosaur SL operator with the archive's own winds and ask how well it reproduces
the archive's 6-h tracer evolution, as a function of sub-step count K. This bounds the entire
approach: it is the accuracy ceiling of any emulator that predicts winds and transports with this
operator. Zero GPU-hours. Code: `/data/jcm_tracer_emulator/b0_reference_operator.py`,
`b0_diagnose.py`.

### 6.1 B0 result

Frame 1990-04-01T06 → +6 h, `p11a_19900101/longrun_day100.nc`, horizontal-only, cubic,
quasi-monotone limiter, 2 departure iterations. R² is explained variance of the 6-h **increment**
— skill-against-persistence conflates how well the field was predicted with how much it moved;
R² does not.

| pulse | site | R² K=1 | R² K=8 |
|---|---|---|---|
| `pulse_3` / `_box` | 10 hPa, 60°S | 0.966 / 0.977 | 0.973 / 0.981 |
| `pulse_5` / `_box` | 300 hPa, 15°S | 0.905 / 0.901 | 0.927 / 0.924 |
| `pulse_2` / `_box` | 70 hPa, 45°N | 0.874 / 0.871 | 0.898 / 0.896 |
| `pulse_1` / `_box` | 30 hPa, 0° | 0.842 / 0.818 | 0.856 / 0.831 |
| `pulse_4` / `_box` | 3 hPa, 30°N | 0.705 / 0.694 | 0.724 / 0.713 |
| **mean** | | **0.855** | **0.872** |

**Three findings, in order of consequence.**

1. **Sub-stepping is not the lever.** K=1 → K=8 moves mean R² by 1.7 points (0.855 → 0.872) and
   everything plateaus by K=4. The residual is *not* trajectory-integration error. **One departure
   solve per 6 h is sufficient** — the transport head needs 3 numbers per cell and a single gather,
   with no sub-stepping machinery. This was predicted the other way and the prediction was wrong.
2. **The plateau is not a diffusion/trajectory cancellation.** var(pred)/var(truth) = 1.0046 at
   K=1 and 1.0040 at K=8 with cubic interpolation: the operator is not damping the field at all,
   so extra interpolation passes are not eating the trajectory gain. It is a genuine floor.
3. **Cubic is diffusion-free here; linear is not, and only at K > 1.** At K=1 linear scores 0.861
   against cubic's 0.855 — *convexity costs nothing at one sub-step*. At K=8 linear falls to 0.809
   with varfrac 0.9991. So a convex softmax-over-stencil head (hypothesis-class-equivalent to
   linear interpolation) is viable **provided it is applied once per step**, which finding 1 says
   it should be. This resolves the convexity-costs-accuracy concern favourably.

4. **The missing vertical transport cannot explain it either.** Measured from the archived ω and
   the hybrid level table (`vertical_scale.py`, no spectral machinery, so none of the conditioning
   problems in finding 5 apply), vertical displacement per 6 h in units of model layers,
   |lat| < 80: **median 0.097, p99 0.76, p99.9 1.36**; only 3.9 % of cells move more than half a
   layer and 0.31 % more than one. Against a horizontal median of 1.74 cells. The pulses are
   σ = 0.25 decades ≈ 5 layers thick, so a 0.1-layer displacement is 2 % of the blob's vertical
   width against 28 % of its horizontal width. Vertical is a few-percent term, not a 15–30 % one.
5. **The residual is wind-history sampling, and it is a property of the 6-hourly archive, not of
   the operator.** Truth integrated 30 successive wind fields over the 6 h; B0 has two endpoint
   snapshots, which differ by rms 2.50 m/s (14.3 % of rms|u|). Driving the same operator with
   u(t0), the midpoint, and u(t1) gives R² of **0.569 / 0.705 / 0.575** for `pulse_4`
   (midpoint is much better than either endpoint, as a second-order time integration should be),
   and across the ten pulses **spread / residual = 1.51** — the spread among plausible wind
   estimates drawn from the same two snapshots is *larger than the residual being explained*.

**Conclusion: the operator is already performing at or near the ceiling the data's time
resolution allows.** This also explains finding 1 retrospectively: sub-stepping refines the
trajectory through a wind field that is itself wrong, so it cannot help.

### 6.2 Multi-frame confirmation (16 frames: 1990/92/94/96 x Jan/Apr/Jul/Oct)

`b0_multiframe.py`. Only the pulses are pure-transport tracers and they homogenise, so a tracer is
used at a frame only if its std/mean there exceeds 0.05 (10 of 10 pulses qualify through 1994,
9 of 10 from 1996); beyond ~1996 no clean tracer is left. QBO phase read off the data as
equatorial zonal-mean u at 30 hPa.

| | mean | min | max |
|---|---|---|---|
| R² (midpoint winds) | 0.834 | 0.705 | 0.916 |
| spread / residual | 1.285 | 0.885 | 1.723 |
| midpoint gain over u(t0) | +0.065 | +0.042 | +0.081 |

**Holds:** the midpoint beats the endpoint wind in **all 16 frames** — the signature of a
second-order time integration beating a first-order one, and the cleanest confirmation that time
sampling binds. Independent of QBO phase (equatorial u30 spans −32 to +13 m/s).

**Weakened:** spread/residual averages 1.29, not the 1.51 measured on the single April-1990 frame,
and falls **below 1 in 3 of 16 frames** (1994-Jul 0.885, 1996-Jul 0.956, 1996-Jan 0.988). Wind
sampling is the dominant term but not the whole residual; in ~20 % of frames something else
contributes comparably. All three exceptions are solstice frames with strong easterly QBO and the
smallest 6-h wind change (~10 % vs 14–20 % elsewhere), and July is the worst season throughout.
A lead, not a conclusion.

**Quote B1's ceiling as R² ≈ 0.83, range 0.70–0.92**, not 0.85.

**Consequences for the ladder.**
- **B1 has a hard ceiling near R² ≈ 0.85 per step** and it is not the emulator's fault. Any
  architecture that predicts 6-hourly winds and transports them with the analytic operator
  inherits it.
- **This is the argument for B2 over B1.** A learned departure field can absorb the *systematic*
  part of the unresolved wind history: the effective 6-hour displacement is not the displacement
  implied by the endpoint winds, and that difference is a function of the resolved flow, which is
  learnable. B2 is not a refinement of B1 — it targets a different and reachable operator.
- **Do not pretrain B2 to convergence against offline `horizontal_departure_points`.** That target
  *is* the 0.85 ceiling. Use it as an initialiser only; the objective must be tracer evolution,
  where the effective displacement is what gets fit.
- The vertical remap is still worth building as a few-percent term, but it is **not** the critical
  path and §8.1 is downgraded accordingly.

**Route closed: the model's divergence cannot be recovered from the archive.** The first attempt
rebuilt a dinosaur `State` from archived u, v, p_s and called `compute_diagnostic_state_hybrid`,
so that σ̇ = M·Δs/Δp would come from the model's own code. It fails: recovering vorticity and
divergence by spectrally differentiating archived *nodal* winds is ill-conditioned (`u/cos_lat`
near the poles), giving a 7.7 % round-trip error for |lat| < 80 and 2656 % at 88.6°, after which
the vertical integral stops cancelling — continuity gives ∂p_s/∂t rms 0.45 Pa/s against 0.0091
from 6-hourly finite differences, correlation 0.014. p_s, layer thickness and the mass-flux
endpoints all check out exactly, so the fault is specifically the wind→vor/div inversion.
**Any vertical work must go through the archived ω**, which carries the mass flux directly:
`ω = B·(v·∇p_s) + B·∂p_s/∂t + M`, so `M = ω − B(v·∇p_s + ∂p_s/∂t)` and `ṡ = M·Δs/Δp`
(`jcm/dycore/dinosaur/dycore.py:609-622` is the forward form). Gate any implementation by
reproducing the archived ω. Code: `omega_gate.py`, `debug_state.py`.

**B1 — learned state head + real SL operator.** Predict u, v, T, ω, p_s; feed those winds to
`transport_scalar`. No learned parameters on the tracer path at all. **Ceiling R² ≈ 0.83 per step (range 0.70–0.92
over 16 frames, §6.2), set by the archive's 6-hourly wind sampling, not by the emulator.** Worth running
as the reference point, but B0 says it cannot be the endpoint.

**B2 — learned transport head, pretrained.** Pretrain the departure field against offline
`horizontal_departure_points` (dense, per-cell, available everywhere including where every tracer
is flat), then fine-tune on actual tracer evolution (which contains the true composed operator
including sub-step curvature, but has no gradient where tracers are uniform).

**B3 — kernel with learned spread.** Only if B2's residual is diffusive in the way §2.3 predicts.

### 6.3 B2 experiments 1-2: the effective-operator premise, tested

`b2_free_departure.py`, `b2_l2_sweep.py`. GPU 2, 94 s for the whole sweep. One frame
(1990-04-01T06), so weaker evidence than section 6.2.

The B2 premise is that the *effective* 6-hour displacement differs from the endpoint-wind
displacement as a function of the resolved flow, and is therefore learnable. Tested at its upper
bound with no network: optimise the departure field itself as free per-cell parameters against
tracer evolution, holding out an entire pulse **site** (Gaussian and box twin together, so nothing
leaks through the twin).

**Experiment 1 — free field, site 4 held out.** Fit R² 0.894 → **0.991**: the parameterisation can
represent the effective operator, so the hypothesis class is not the limit. But held-out R²
0.699 → **0.514**, *worse* than analytic. The loss constrains the departure only where the fitted
tracers have gradients; elsewhere it drifts, and site 4 (3 hPa) is thinly covered by the rest
(30/70/10/300 hPa).

**Experiment 2 — L2 toward the analytic field, every site held out in turn.**

| site | level | analytic | best | at L2 | gain |
|---|---|---|---|---|---|
| 1 | 30 hPa | 0.8297 | 0.8837 | 1e4 | +0.054 |
| 2 | 70 hPa | 0.8724 | **0.9546** | 0 | +0.082 |
| 3 | 10 hPa | 0.9714 | 0.9768 | 1e4 | +0.005 |
| 4 | 3 hPa | 0.6993 | **0.7857** | 1e4 | +0.086 |
| 5 | 300 hPa | 0.9029 | **0.9625** | 0 | +0.060 |
| **mean** | | **0.855** | **0.913** | | **+0.058** |

**The premise holds.** A departure correction fitted to four sites transports a fifth, unseen site
better than the analytic operator, at all five. Sites 2, 4 and 5 gain +0.06 to +0.09 and are
unambiguous; site 1 is smaller and **site 3's +0.005 is within the noise of selecting the best of
six L2 values on the held-out score itself** — so call it 3-4 of 5 convincing, not 5.

**And it argues for the network specifically.** The optimal L2 swings from 0 (sites 2, 5 — well
covered in pressure by the fitted set) to 1e4 (sites 1, 3, 4), so no single hand-tuned penalty
works everywhere. A network does not need one: `delta = f(flow)` with weights shared across all
cells supplies the constraint intrinsically, and where tracers are flat the field is pinned by the
flow rather than left free. The L2 sweep is a crude stand-in for that inductive bias. **Build the
network.**

Open: the gain closes ~40 % of the gap from analytic (0.855) to perfect on this frame. Whether it
holds across the 16 frames of section 6.2 is untested.

### 6.4 B2 experiments 3-4: the network, across 16 frames

`b2_net.py` (flawed), `b2_net2.py` (corrected), `b2_cache.py`. 83,139 parameters, 43 s on GPU 2.

Architecture: `delta = SCALE·tanh(CNN(flow))`, a correction to the analytic departure point applied
through the fixed dinosaur remap. Weights **shared across all 63 levels** (levels are the batch
axis, level pressure an input feature) — the inductive bias §6.3 identified as missing from a free
per-cell field. 13 input channels: u, v at both times at the arrival cell, **the same winds gathered
at the analytic departure point** (the correction depends on the flow along the trajectory, and a
gather supplies that far more cheaply than a 30-cell receptive field), the 6-h wind change, and
cos/sin lat and log p. Output layer **zero-initialised**, so training begins exactly at the analytic
operator. Tracers never enter the network — the design invariant of §3 holds throughout.

Generalisation tested on two axes at once: an entire held-out pulse **site** (4, at 3 hPa, the
hardest) never in the loss, and held-out **frames**.

**Experiment 3 was invalid and is retained as a lesson.** Batch size 1 over heterogeneous frames
made the fit R² oscillate 0.85–0.93, and the reported number was the *last* step — which the
project checkpoint rule forbids. Test-frame R² had peaked at step 250 (0.8401) and decayed
monotonically to 0.8242 against an analytic 0.8353.

**Experiment 4**: full-batch gradients over the train frames, three-way split (train 1990+1994,
validation 1992 for checkpoint selection, test 1996 read once at the selected step).

| split | analytic | network | gain | beats analytic |
|---|---|---|---|---|
| train (8 frames) | 0.8689 | 0.8811 | +0.0122 | 4/8 |
| validation (4) | 0.8449 | 0.8482 | +0.0032 | 3/4 |
| **test (4)** | **0.8256** | **0.8300** | **+0.0044** | **3/4** |

**The premise survives; the effect size does not.** The network generalises in the right direction
across both unseen sites and unseen frames, but the gain is **+0.004 R²** — an order of magnitude
below the +0.058 of §6.3. That §6.3 figure was inflated on both counts already flagged there: one
frame, and a per-site L2 chosen on the held-out score itself.

Two structural observations:
- **Validation peaked at step 50 of 400** and declined steadily afterwards. An 83k-parameter
  network trained on **8 frames** overfits almost immediately.
- The gain is dominated by `1994_Jul` (+0.111), the frame where the analytic operator is worst
  (0.6796). Elsewhere it is ±0.01–0.02, and all four 1990 frames get *worse*. The network is
  mostly rescuing pathological frames, not improving the typical one.

**The next lever is training data, not architecture.** Eight frames is the caching choice of
`b2_cache.py`, not a property of the corpus, which holds 29,220. The overfit-by-step-50 signature
says the sample size is binding. Re-run at 10²–10³ frames before concluding anything about the
architecture or about whether the effective-operator correction is worth having.

### 6.5 B2 experiment 5: 96 training frames

`b2_cache2.py` (20 chunks x 8 consecutive frames = 160 frames, 10.0 GB; consecutive frames share
tracer snapshots, so N frames need N+1), `b2_net3.py`. Architecture, features, year split and
holdout site identical to §6.4 — **the only change is 8 -> 96 training frames.** 122 s on GPU 2.

| split | frames | analytic | network | gain | beats analytic |
|---|---|---|---|---|---|
| train (1990+1994) | 96 | 0.8418 | 0.8430 | +0.0011 | 48/96 |
| validation (1992) | 32 | 0.8746 | 0.8798 | +0.0052 | 29/32 |
| **test (1996, read once)** | **32** | **0.8397** | **0.8526** | **+0.0128** | **30/32** |

**Data was the binding constraint, as §6.4 predicted.** Against the 8-frame run:

| training frames | test gain | test frames beaten | validation peak |
|---|---|---|---|
| 8 | +0.0044 | 3/4 | step 50 of 400 |
| **96** | **+0.0128** | **30/32** | **step 300 of 1200** |

12x the data roughly triples the gain, and the validation peak moves from step 50 to step 300 —
the signature of the overfitting relaxing exactly as a sample-size explanation predicts.

**The premise is now established, not merely directional.** A flow-conditioned departure correction
fitted on four pulse sites in 1990 and 1994 transports an unseen site (3 hPa) in an unseen year in
**30 of 32 frames**. That is no longer attributable to selection or to one lucky frame.

**But the effect remains small**: +0.0128 R² closes about 8 % of the gap from analytic (0.840) to
perfect. Useful, robust, and not yet decisive.

**Unexplained**: the train split gains least (+0.0011, 48/96) and the untouched test split most
(+0.0128, 30/32), with near-identical analytic baselines (0.8418 vs 0.8397). Note these columns are
not train-vs-test in the usual sense — all three evaluate the *held-out site*, which is in no
split's loss, so "train" means seen-frame/unseen-site. Worth understanding before scaling further.

**Still only 96 of 29,220 available frames (0.3 %).** The scaling curve has not turned over; the
next run should be 10^3-10^4 frames before any architecture change is considered.

### 6.6 B2 experiment 6: streaming, ~10k frames

`b2_net_stream.py`. 252 chunks over 1990/91/93/94/95/97/98 = **9,828 distinct training frames**
per pass, streamed from the netCDF chunks (a static cache would be ~1.1 TB). Pool of 4 chunks as
the shuffle buffer, 40 steps per chunk load, 47 min on GPU 2, 12,000 steps, 11,700 frames seen.
Validation (1992) and test (1996) are the **identical fixed 32 frames as §6.4-6.5** — confirmed by
the analytic baselines reproducing exactly (val 0.8746, test 0.8397), so the comparison is clean.

| split | frames | analytic | network | gain | beats analytic |
|---|---|---|---|---|---|
| validation (1992) | 32 | 0.8746 | 0.8801 | +0.0055 | 28/32 |
| **test (1996, read once)** | **32** | **0.8397** | **0.8716** | **+0.0319** | **31/32** |

**Scaling continues to pay:**

| distinct training frames | test gain | test frames beaten |
|---|---|---|
| 8 | +0.0044 | 3/4 |
| 96 | +0.0128 | 30/32 |
| **2,340 (at selection)** | **+0.0319** | **31/32** |

+0.0319 R² closes ~20 % of the gap from analytic (0.840) to perfect, on 31 of 32 untouched frames.
Best result to date.

**But the run does not use 10k frames productively.** Validation peaked at chunk 60 (2,340 frames)
and then degraded, noisily but with a clear downward trend, reaching **0.8248 at chunk 300 — far
below the analytic 0.8746**. The selected model is early in the run. This is no longer a
data-scarcity signature (§6.4-6.5); it is an optimisation-stability problem, and it caps the useful
training set at ~2.3k frames in this configuration.

Three candidate causes, untested:
- **Constant LR, no decay** over 12,000 steps.
- **A weak shuffle buffer.** A pool of 4 chunks used for 40 consecutive steps means every step
  samples ~156 highly correlated frames from at most 4 chunks. Raise the pool and lower
  steps-per-chunk.
- **Weak-pulse years.** 1997-98 pulses are at std/mean 0.15-0.20; their persistence-normalised loss
  is noisy and the run meets more of them as it proceeds.

**Two bugs found building this, both silent:**
1. `b0_reference_operator.py` set `JAX_PLATFORMS=cpu` at module level (it began as a CPU
   diagnostic), silently pinning every importing training script to CPU. Now guarded under
   `__main__`; GPU launches also set `JAX_PLATFORMS=cuda` explicitly.
2. **Concurrent netCDF reads segfault.** Two reader threads calling `xr.open_dataset(...).values`
   at once gave `EXIT=139` (SIGSEGV) with no Python traceback — HDF5/netCDF4 is not thread-safe
   unless built for it. Killed three runs before it was diagnosed. All reads now serialise through
   `READ_LOCK`. Cost: with one reader the run is half I/O-bound (24 min of 47). A process-based
   reader or preprocessed shards is the fix if throughput matters.

### 6.7 B2 experiment 7: stability fixes

`b2_net_stream2.py`. Warmup + cosine LR decay, global-norm gradient clipping, pool 8 with 20
steps per chunk (vs 4/40), 1997-98 dropped. 180 chunks = 7,020 frames, 3,600 steps, 29 min.

**The stability problem is fixed.** Validation holds a 0.871-0.879 band through the whole second
half and ends at 0.8772, against §6.6's collapse to 0.8248. The LR decay is visibly responsible:
after a mid-run dip the trace climbs monotonically 0.8760 -> 0.8768 -> 0.8771 -> 0.8782 -> 0.8789.

But the selected checkpoint was chunk 20 (780 frames) and scored test +0.0125 (28/32) against
§6.6's +0.0319 (31/32) — on nearly identical validation (0.8798 vs 0.8801). Two runs, same
validation, 2.5x different test, on 32 frames. That prompted §6.8.

### 6.8 The evaluation was too small, and the validation year has no headroom

`b2_eval_big.py`. Both saved models re-scored on **468 frames per split** (12 chunks per year,
streamed), same held-out years.

| split | model | R² | gain | SE | beats |
|---|---|---|---|---|---|
| val 1992 | §6.6 | 0.8788 | +0.0068 | ±0.0006 | 325/468 |
| val 1992 | §6.7 | 0.8776 | +0.0056 | ±0.0003 | 395/468 |
| | *paired* | | **+0.0012 ± 0.0003** | | t = +4.3 |
| **TEST 1996** | **§6.6** | **0.8517** | **+0.0322** | **±0.0012** | **437/468** |
| TEST 1996 | §6.7 | 0.8384 | +0.0189 | ±0.0007 | 435/468 |
| | *paired* | | **+0.0133 ± 0.0007** | | **t = +19.6** |

**Two findings.**

1. **The §6.6 model really is better**, by +0.0133 R² on test at t = 19.6. Not selection noise.
   And §6.6's 32-frame estimate was accurate (+0.0319 vs +0.0322 on 468 frames) while §6.7's was
   not (+0.0125 vs +0.0189) — 32 frames gives the right ordering but an unreliable magnitude.
2. **The validation year cannot discriminate.** The two models differ by +0.0012 on 1992 and
   +0.0133 on 1996 — 11x. Cause: on the large set 1992's analytic baseline is 0.8720 against
   1996's 0.8195, and the network's gain concentrates where the analytic operator is weak (the
   pattern noted since §6.4). **Checkpoint selection has been running on a year with almost no
   headroom**, which is why §6.7 stabilised training and still selected a worse model.

**Current best, robustly measured: +0.0322 ± 0.0012 R² on 468 untouched frames, beating the
analytic operator in 437 of them (93 %).** That closes ~18 % of the gap from analytic (0.820) to
perfect.

### 6.9 A bigger validation set was not the fix

`b2_net_stream3.py`. Identical to §6.7 except validation is 156 streamed 1992 frames instead of 32.
It selected **chunk 20 again**, and the resulting model is **bit-identical** to §6.7's (paired
difference exactly 0.0000 on 468 test frames). So validation *size* was never the problem.

The real problem: chunk 20's validation score (0.8927) is a statistical tie with the late-run
values (0.8919–0.8923). Selection was choosing arbitrarily among indistinguishable checkpoints and
happening to take the earliest — which had seen only 780 frames and generalises worst.

### 6.10 Select late, and the scaling finally shows — BEST RESULT

`b2_net_stream4.py`. Identical to §6.9 but selection ignores validation peaks before chunk 60.
With cosine decay the end of training should be the best point, so an early spike is noise.
Selected chunk 160 (**6,240 distinct frames seen**).

**TEST 1996, 468 untouched frames, all five checkpoints scored head to head:**

| model | R² | gain | SE | beats analytic |
|---|---|---|---|---|
| §6.7 / §6.9 cosine, sel @780 frames | 0.8384 | +0.0189 | ±0.0007 | 435/468 |
| §6.6 const-LR, sel @2,340 frames | 0.8517 | +0.0322 | ±0.0012 | 437/468 |
| **§6.10 cosine, sel @6,240 frames** | **0.8580** | **+0.0385** | **±0.0012** | **468/468** |

Paired: §6.10 − §6.6 = **+0.0063 ± 0.0004, t = +16.7**. §6.10 − §6.7 = +0.0196 ± 0.0007, t = +30.0.

**The full scaling curve, once stability and selection are both right:**

| distinct training frames | test gain | beats analytic |
|---|---|---|
| 8 | +0.0044 | 3/4 |
| 96 | +0.0128 | 30/32 |
| 2,340 | +0.0322 | 437/468 |
| **6,240** | **+0.0385** | **468/468** |

**Headline: +0.0385 ± 0.0012 R², beating the analytic operator on every one of 468 untouched
frames.** That closes ~21 % of the gap from analytic (0.8195) to perfect, on a held-out pulse site
(3 hPa) in a held-out year, with a flow-conditioned correction that never sees a tracer.

Three things had to be right at once, and each was diagnosed only after the previous was fixed:
data volume (§6.4-6.5), optimisation stability (§6.6-6.7), and checkpoint selection (§6.9-6.10).
Data scaling *was* paying the whole time; it was masked first by instability and then by a
selector that preferred an early spike.

**Not done:** the final-checkpoint comparison. The patch that was meant to save the final
checkpoint alongside the best silently failed to apply (no assertion on the replacement), so
"is the last checkpoint as good as the late-selected best?" is still open. Chunk 160 of 180 is
near enough to the end that the answer is probably yes, but it is untested.

### 6.11 The late-training trajectory in test space — take the LAST checkpoint

`b2_net_stream5.py` (training byte-identical to §6.10; only checkpointing differs — every patch
replacement asserted this time), `b2_eval_big_traj.py`. Nine checkpoints from chunk 100 to 180,
each scored on the same 468 untouched test frames.

| checkpoint | frames seen | test gain | SE | beats analytic |
|---|---|---|---|---|
| ck100 | 3,900 | +0.0346 | ±0.0011 | 468/468 |
| ck120 | 4,680 | +0.0379 | ±0.0011 | 464/468 |
| ck140 | 5,460 | +0.0384 | ±0.0012 | 468/468 |
| ck160 | 6,240 | +0.0385 | ±0.0012 | 468/468 |
| ck170 | 6,630 | +0.0413 | ±0.0012 | 468/468 |
| **ck180 (final)** | **7,020** | **+0.0420** | **±0.0013** | **468/468** |

**1. The final checkpoint is the best, and validation selection is actively harmful here.**
Validation picked ck160; the final checkpoint beats it by **+0.0035 ± 0.0002, t = +21.8**. With
cosine decay to 1 % of peak LR, the end of training is the right place to stop, and the
low-headroom validation year (§6.8) mis-ranks the last three checkpoints.

*This does not overturn the project's "never take the last checkpoint" rule.* That rule comes from
**rollout** training, where more training measurably degraded **multi-year** behaviour. This is
single-step transport with a decayed LR, a different regime and a different metric. Both can be
true; do not carry this result across to rollout training.

**2. Test performance is still climbing at the end.** The last three checkpoints rise monotonically
(+0.0385 → +0.0413 → +0.0420) with SE ±0.0012. **Still step- and data-limited at 7,020 frames** —
0.3 % of what §2 says is available for these years. Longer runs should keep paying.

**FINAL HEADLINE: +0.0420 ± 0.0013 R² on 468 untouched frames, beating the analytic operator in
468 of 468 (100 %).** Held-out pulse site (3 hPa), held-out year, from a flow-conditioned
correction that never sees a tracer. ~23 % of the gap from analytic (0.8195) to perfect.

Recommended recipe as it stands: cosine LR decay with warmup, gradient clipping, pool 8 /
20 steps per chunk, 1997-98 excluded, **take the final checkpoint**, and score on >=468 frames.

### 6.12 B1: the state head, and the 2x2

`b3_state_head.py`, `b3_eval_b1.py`. Predicts u, v (plus T, p_s) at t+6h; levels as CHANNELS, not
the batch axis as in the transport head — a wind forecast couples levels through thermal wind and
vertical wave propagation, which a level-shared 2D CNN cannot represent. 384 in -> 190 out, 5,700
frames, final checkpoint per §6.11. Scored on 456 untouched 1996 frames.

**Wind skill:** u RMSE **1.478** vs 2.814 persistence (+47.5 %); v **1.497** vs 3.141 (+52.4 %).

| | analytic departure | learned head | head gain |
|---|---|---|---|
| true winds | 0.8195 | 0.8616 | +0.0420 |
| **predicted winds** | 0.8031 | **0.8408** | +0.0377 ± 0.0013 (452/456) |

**The cost of predicting the wind (−0.0164 ± 0.0004 analytic, −0.0208 ± 0.0003 learned) is about
half the transport head's gain (+0.038).** So predicted winds + learned head (0.8408) *beats the
reference operator driven by true winds* (0.8195) by +0.021. The ladder's ordering is correct and
B1 does not dominate B2.

### 6.13 The coupled emulator, run autoregressively — THE STATE HEAD DIVERGES

`b4_rollout.py`. 38 autoregressive steps (9.5 days) from 1996 day200, state head feeding the
transport head feeding itself. First multi-step test in the whole project.

| day | u RMSE | max&#124;u&#124; | truth | R² analytic | R² learned | **R² truth-wind** | mass | struct | truth |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1.5 | 104 | 103 | 0.772 | 0.827 | **0.848** | 1.0000 | 0.0430 | 0.0430 |
| 2 | 7.9 | 108 | 109 | 0.367 | 0.421 | **0.900** | 1.0003 | 0.0429 | 0.0429 |
| 3 | 12.2 | 112 | 106 | −0.250 | −0.148 | **0.899** | 1.0007 | 0.0429 | 0.0429 |
| 5 | 35.4 | **339** | 109 | −2.90 | −2.59 | **0.923** | 1.0028 | 0.0432 | 0.0429 |
| 9 | 220.7 | **1751** | 122 | −15.7 | −15.3 | **0.864** | 1.0126 | 0.0414 | 0.0428 |

**Planetary runaway** — max|u| 104 → 1751 m/s while truth stays 103–122. The same signature as the
v5 and U-Cast rollouts. Tracer R² goes negative by day 3.

**The control attributes it cleanly.** The `truth-wind` column is the learned transport head rolled
out autoregressively on the tracer but driven by archive winds: it holds **R² 0.86–0.92 for the
full 9.5 days**, with mass conserved to 1.3 % and structure tracking truth (0.041–0.043 vs 0.0428).

> **The transport half of the architecture is rollout-stable. The state head is what diverges.**

And the reason is in this document: the state head as built has **none of §5.3's analytic terms.**
It is a pure black-box single-step wind predictor — no Polvani-Kushner relaxation, no sponge, no
nudging mask, no QBO nudge, and no rollout training. §5.3 argued the restoring force is what
prevents exactly this, and the dry-attractor work concluded only structural per-step budgets
escape. The emulator was built without them and did precisely what every prior model in this line
did.

**Caveat:** `pulse_4` in 1996 is already homogenised (std/mean 0.043), so the over-diffusion test
here is weak. Re-run the structure check on a 1990-91 frame where the pulses still have structure.

**Answer to "can we train the emulator": the tracer half is ready; the state half needs §5.3
before it can run free.** That is a defined build — the terms are analytic and already in
`polvani_kushner.py` and `qbo_nudging.py` — plus rollout/pushforward training. Not a dead end.

## 7. Evaluation

**Over-diffusion is the failure mode, not mass drift.** With conservation structural, mass tells
you nothing. The corpus has a built-in diffusion benchmark — the structure-decay curve of §2.2 is
truth. An over-diffusive emulator collapses it faster. One scalar per tracer per time.

| test | what it gates |
|---|---|
| structure decay vs §2.2 | implicit diffusion |
| **held-out tracers** — train the operator on 17 injected tracers, evaluate on 2 | generalisation to an unseen IC. An operator model handles it natively; a decoder model cannot express it, having no output channel for a tracer it never saw. That asymmetry *is* the argument for §3 |
| **box vs Gaussian twins** — same site, same nominal size, different edge sharpness | the same `W` must transport both. A decoder can learn them separately and look fine; an operator cannot cheat. This is the discriminating test between the two architectures |
| synthetic ICs at inference (delta, checkerboard, constant) | constants stay constant (row sums = 1), mass holds, no new extrema — all checkable with no truth data |
| age of air vs the dycore (`aoa_vs_clams.py`) | deep-branch transport. **Against the dycore, not observations** — the free stratosphere here is 1.3 yr too old in the tropics at 55 hPa with a 1.9 yr tropics–extratropics contrast vs CLaMS's 2.8, and a faithful emulator reproduces that bias |
| polar vortex + SSW frequency (`vortex_series.py`) | the free, interesting part. ~20–40 events in 20 years — small-sample, high-leverage |

**Checkpoint rule (carried over, non-negotiable):** never take the last checkpoint. More rollout
training measurably degraded multi-year behaviour on v2recipe, flux-form and v6ref alike. Score on
the multi-year metric, keep every checkpoint, take the peak.

## 8. Known gaps

1. **`sigma_dot` is not in the archive.** A few-percent term (§6.1 finding 4), not the critical
   path, and it must be derived from ω, not from the winds (§6.1, route closed). `departure_points_3d` needs the vertical velocity in
   coordinate space; the archive has ω (Pa/s). For hybrid levels
   `ω = ∂p/∂t + v·∇p + (∂p/∂η)·η̇`, and `∂p/∂t`, `∇p` are recoverable from p_s at t and t+6h.
   Until that is built, B0 runs **horizontal-only**, which measures the horizontal operator and
   leaves all vertical transport in the residual — a lower bound on error, not a clean test.
2. **6 h ≠ 30 × 12 min.** Quantified by B0 (§6.1) and it is *not* over-diffusion: variance is
   preserved to 0.5 %. The horizontal operator at one solve per 6 h explains 70–98 % of the
   increment; the rest is elsewhere.
3. **One realisation.** 20 years, one IC, one QBO history. Variability statistics are thin,
   SSWs especially.

## 9. Open decisions

| # | decision | note |
|---|---|---|
| 1 | departure **point** vs **kernel with spread** | point is simplest and matches the reference; kernel is conservative when normalised and makes implicit diffusion an explicit knob. B0 (§6.1) weakens the case for spread: the reference operator is not diffusive at one solve per 6 h |
| 1b | interpolation order | **resolved: cubic**, and apply the operator once per step. Convexity is free at K=1 (§6.1 finding 3) |
| 2 | joint training vs transport head first with the state frozen | mostly a schedule question |
| 3 | exact local conservation vs matching the reference including its fixer | scoping call, not a data question — see §2.4 |
| 4 | trunk depth / latent size | not yet constrained by anything measured |

### 9.1 Timing, measured on identical hardware

The first 35x figure compared an emulator step on an **H200** against JCM's `220 ms` from
PROGRESS.md, which was recorded on an **H100 80GB** (the run provenance says so; the box was
upgraded between 2026-09-15 and 2026-09-18). That is not apples to apples, so JCM was re-benchmarked
on the same H200: `+experiment=p11a_prod`, 1990, steady state after compile amortises.

**JCM on H200: 7.8-8.1 s per 10 simulated days = 197 ms per 6 simulated hours.** Against 8.1-8.2 s
on H100 — so **H200 is only ~3 % faster here**. The dycore at T63 is latency and kernel-launch
bound, not bandwidth bound, so the hardware upgrade buys almost nothing and the original comparison
survives. Worth having checked rather than assumed.

**Emulator component breakdown** (H200, one 6-h step, T63L63, 24 tracers):

| component | ms | share | kind |
|---|---|---|---|
| state head (dynamics, 956 k params) | 0.36 | 6 % | learned |
| departure-point solve | 0.36 | 6 % | analytic |
| **transport head + gathers (83 k params)** | **4.11** | **68 %** | **learned** |
| remap, 24 tracers | 1.22 | 20 % | analytic |
| **total** | **6.05** | | **32.6x vs JCM** |

**The 83 k-parameter transport head costs 11x the 956 k-parameter state head.** Parameter count is
the wrong intuition here: the transport head runs with **levels as the batch axis**, so it performs
63 separate 2D convolutions at 64 channels, while the state head does one at 128. It is two-thirds
of the runtime for +0.0420 R².

Remap cost scales sub-linearly then linearly: 0.255 ms for 1 tracer, 0.051 ms/tracer at 24,
0.046 at 48. **Adding tracers is nearly free** — which is the §3 invariant paying off in wall clock
as well as in generalisation.

### 9.1b How much of JCM is tracer transport? 32.3 %

Measured, not inferred: the Phase-11 configuration re-run on the same H200 with
`~physics.terms.production_tracers` removed, steady state after compile amortises.

| | s / 10 sim-days | ms per 6 h |
|---|---|---|
| JCM, 24 tracers | 7.92 | 198.0 |
| JCM, tracer machinery removed | 5.36 | 134.0 |
| **tracers** | | **64.0 — 32.3 % of the step, 2.67 ms/tracer** |

The A/B removes the whole tracer term, so this covers advection of 24 fields *plus* evaluation of
their sources, sinks, clocks and lid; advection is the bulk but not all of it. An earlier estimate
from the PROGRESS.md throughput table (4 → 15 → 24 tracers, H100) gave 26.7 % — discard it in
favour of the direct number.

**Component-for-component against the emulator:**

| | JCM | emulator | ratio |
|---|---|---|---|
| tracer transport, 24 tracers | 64.0 ms | 1.22 ms | **52×** |
| dynamics + physics | 134.0 ms | 0.36 ms | **372×** |

The 52× on transport involves **no learning** — both sides run the same operator, and the gap is
applying it once per 6 h instead of thirty times at 12 min (licensed by B0: one application captures
83 % of the increment). The 372× is where the emulation actually happens. Transport is a third of
JCM's cost and is already beaten 52× without learning it; the dynamics are two-thirds and are the
part that still diverges (§9.6).

### 9.2 The speedup range

| configuration | ms/step | speedup | 1 sim-yr | 30 yr |
|---|---|---|---|---|
| analytic parts alone (hard floor) | 1.58 | 125x | 2.3 s | 1.2 min |
| without the learned correction (B1 only) | 1.94 | 102x | 2.8 s | 1.4 min |
| optimised correction head (~1 ms, plausible) | 2.94 | 67x | 4.3 s | 2.1 min |
| **now, as built** | **6.05** | **32.6x** | **8.8 s** | **4.4 min** |
| + vertical remap and §5.3 forcing (estimated) | ~8.6 | ~23x | 12.5 s | 6.2 min |
| JCM (stepping) | 197 | 1x | 4.8 min | 2.4 h |

**Reading this table.**
- **The maximum is ~125x**, set by the analytic operator, which is only 26 % of the current step.
  Nothing the network does can go below that without replacing the remap — which §4.4 argues
  against on accuracy and structural grounds.
- **The learned correction currently costs 4.11 ms for +0.042 R²** — dropping it triples the speed
  (32.6x → 102x). That is a real cost/benefit decision, and the head is the highest-leverage
  optimisation target: fewer channels, or computing the correction on a subset of levels and
  interpolating in log-p (the correction is smooth there), should reach ~1 ms and ~67x.
- **The still-unbuilt pieces cost, not save.** The vertical remap needs `departure_points_3d` and
  3-D interpolation, materially more than their 2-D counterparts. Expect the honest steady-state
  number to settle near **~23x**, not 33x.

**Two caveats that bound all of the above.**
1. **Stepping only.** JCM's end-to-end is output-bound (PROGRESS.md: 750 days/hr e2e against 4,180
   stepping, two chains sharing the CPU output path). An emulator writing comparable output
   inherits the same bottleneck, and the practical gain collapses toward the I/O ratio. The single
   exclusive benchmark here measured 4,557 days/hr including its own chunk writes.
2. **Notional until §10.1.** A 33x speedup on a model that diverges at day 3 (§6.13) is worth
   nothing. The number that matters is the throughput of something stable over 30 years.

### 9.3 The transport head rebuilt levels-as-channels — faster AND more accurate

`b5_net_lc.py`, `b5_eval_lc.py`. PARADIS's design: levels in channels, one 2D conv stack
(819 in → 128 → 128 → 128 → 189 out, **1,456,701 params** against 83,139). Horizontal weight
sharing across 18,432 cells is retained; only level-sharing is given up. Data, split, schedule,
selection rule and holdout site identical to §6.11.

**TEST 1996, 468 untouched frames:**

| head | params | R² | gain | SE | beats |
|---|---|---|---|---|---|
| levels-as-BATCH, final (§6.11) | 83 k | 0.8615 | +0.0420 | ±0.0013 | 468/468 |
| levels-as-CHANNELS, validation-selected | 1.46 M | 0.8587 | +0.0392 | ±0.0013 | 468/468 |
| **levels-as-CHANNELS, final** | **1.46 M** | **0.8668** | **+0.0473** | **±0.0016** | **468/468** |

Paired: LC-final − batch-final = **+0.0053 ± 0.0004, t = +12.6**.

**Both worries were wrong, in instructive ways.**

1. *The 17× parameter increase did not overfit.* It helped. Levels can now couple, and the
   correction plausibly depends on vertical shear, which the level-shared head could not represent.
2. *The apparent overfitting in the validation trace was the validation year, not the model.*
   Validation sat **below** the analytic baseline (0.8879) at almost every checkpoint, ending at
   0.8685 — while test says this is the best model built. The validation-selected checkpoint scores
   +0.0392; the final scores +0.0473, a paired +0.0081 ± 0.0003 worse for listening to validation.

**This is the third time the 1992 validation year has misled us** (§6.8 diagnosed the cause — its
analytic baseline is 0.8720 against 1996's 0.8195, so it has no headroom; §6.11 showed the final
checkpoint beats the selected one; here validation would have rejected the best model outright).
**Operational rule: take the final checkpoint and do not select on 1992.** A validation set for this
work needs a year with comparable headroom to the target, not merely more frames (§6.9 showed size
alone changes nothing).

**Timing.** Full emulator step, 24 tracers, H200: **2.92 ms**, against 6.05 ms for the
levels-as-batch head — **68× vs JCM's 197.5 ms** (was 33×). One simulated year in 4.3 s; 30 years in
2.1 minutes.

**So the PARADIS-style restructure is a strict win: 2.1× faster and +0.0053 R² more accurate.**
The remaining step is 42 % analytic remap, 18 % departure solve, 12 % state head, 28 % transport
head — the analytic operator is now the largest single component.

### 9.4 The time step is priced, and the lever is CLOSED

`b6_timestep.py`. §9.2 named the time step as the only lever that multiplies everything including
the analytic floor, and flagged it as the cheapest, highest-information thing to measure. Measured.

Every scheme scored against the **same** target (truth at t0 + H) with the **same** persistence
baseline, differing only in how many operator applications it uses. 360 samples per row
(3 years × 4 chunks × 3 starts × 10 pulses). Analytic operator only — the learned correction was
trained at 6 h and would be out of distribution.

| horizon | step | applications | R² midpoint | R² endpoint-mean |
|---|---|---|---|---|
| 1 day | **6 h** | 4 | **0.8858** | 0.8858 |
| 1 day | 12 h | 2 | 0.7332 | 0.7417 |
| 1 day | 24 h | 1 | **−0.3730** | −0.3155 |
| 4 days | **6 h** | 16 | **0.8369** | 0.8369 |
| 4 days | 12 h | 8 | 0.5190 | 0.5403 |
| 4 days | 24 h | 4 | −0.5861 | −0.5063 |
| 4 days | 48 h | 2 | −2.0012 | −1.7728 |

**A 24-h step is worse than persistence.** Trajectory error grows superlinearly: the flow evolves
within the interval, trajectories curve, displacement reaches ~100 cells at p99.9, and the 2-iteration
fixed-point solve's convergence condition `dt·max‖∇V‖ < 1` is violated outright.

**6 h is already at the knee.** 12 h costs 0.15 R² at one day and 0.32 at four — 3.5× more than the
learned correction recovers at 6 h (+0.047), so retraining the head at 12 h could not plausibly pay
for it.

*Why this differs from the TOMAS Δt-conditioned precedent* (where Δt = 30 beat the baseline at 30×
fewer steps): that system is aerosol microphysics — a local ODE per column with no advection. Here
a longer step means a longer **trajectory**, and trajectory error is the dominant term. The two
cases are not analogous, and the TOMAS result should not have been read across.

Incidental: at long dt the **endpoint mean beats the true midpoint snapshot** (e.g. −0.3155 vs
−0.3730 at 24 h), reversing B0 finding 5. Over a long interval an average is more representative of
the whole interval than any single snapshot in it.

**Consequence for §9.2.** The 100× band is *not* reachable by stepping longer. Current state is
68× at 2.92 ms; the remaining budget is 42 % analytic remap, 18 % departure solve, 28 % transport
head, 12 % state head. Micro-optimisation might reach ~80–90×, and the complete model with vertical
remap and §5.3 forcing lands nearer **~40×**. **Treat ~40× as the honest planning figure**, not 100×.

### 9.5 Carrying the sponge: helps a lot, does not fix it

`b8_state_sponge.py`, `b8_state_error.py`. The §5.3 sponge carried structurally — exact profile
from `jcm/physics/dissipation/upper_sponge.py`, `inv_tau[i] = 1/(5400 · 6.73**i)` with i = 0 at the
model top, so archive levels 62/61/60/59 get τ = 1.5 h / 10.1 h / 67.9 h / 19 d and 6-h decay
factors 0.018 / 0.552 / 0.916 / 0.987. Applied as `u_new = decay·u_old + net_output`, with the
training target changed to `u[i+1] − decay·u[i]`. State head retrained; identical ICs and metrics
to the b7 diagnostic.

**Single-step error is unchanged** (u 1.459 vs 1.467 m/s) — as expected, the sponge was already
learnable at one step. The question was only about free running.

**Free run, max|u| in the sponge band (truth ≈ 30 m/s), and overall u RMSE:**

| day | sponge band, no sponge term | **with sponge term** | u RMSE, no sponge | **with** |
|---|---|---|---|---|
| 2.0 | 44.4 | 47.6 | 7.73 | 7.92 |
| 3.0 | 101.5 | **75.1** | 12.30 | 12.25 |
| 5.0 | 399.8 | **166.6** | 37.61 | **33.95** |
| 7.0 | 808.6 | **382.9** | 101.38 | **97.01** |
| 9.5 | 5285 | **1302** | 368.3 | **254.1** |

**The targeted band improves 2.4–4× at day 5–9.5, and overall error is 31 % lower at 9.5 days.
The model still diverges, and the onset is not delayed at all** — the sponge band is still ~50 %
above truth by day 2.

**So the b7 diagnosis was right about *where* divergence starts and wrong about it being the
cause.** Fixing the top moves the bottleneck rather than removing it. The other two bands run away
on their own: the free interior reaches 265 vs 80 m/s by day 5, and the nudged troposphere 160 vs
76 — and in truth *both* are held by terms this head still lacks (ERA5 relaxation at τ 6 h on
levels 0–8; PK relaxation at τ 15 d throughout). A restoring force on four levels cannot stabilise
59 others.

**One structural defect found in the process, not yet fixed.** The form `u_new = decay·u_old + net`
damps the *previous state* but leaves the network's increment undamped. The exact solution is
`u(t+Δt) = u(t)e^{−Δt/τ} + ∫D(s)e^{−(Δt−s)/τ}ds`, and for roughly constant D that integral is
`D·τ·(1−e^{−Δt/τ})` — bounded by `D·τ`, not `D·Δt`. At the top level (τ = 1.5 h, Δt = 6 h) that is
a factor ~4 smaller than what the current form permits. The network is free to emit an unbounded
increment exactly where the physics is most dissipative. Scaling the output by
`τ(1−e^{−Δt/τ})/Δt` at sponge levels is the consistent fix and should be done before concluding
anything further about the sponge.

**Revised read of §10.1:** the analytic terms are necessary but were never going to be sufficient
on their own. All four are needed (each band has its own restoring term), *and* rollout training —
which remains entirely absent — is what teaches the model to stay on the attractor between them.

### 9.6 All four terms + increment fix + rollout training

`b9_state_full.py`, `b9_rollout.py`. Everything §5.3 specifies, together:

| term | implementation | 6-h strength |
|---|---|---|
| ERA5 nudging, levels 0-8 | **prescription** — the troposphere is boundary data, so the emulator consumes it rather than predicting it | exact |
| Rayleigh sponge, 59-62 | exact profile from `upper_sponge.py` | decay 0.018 / 0.552 / 0.915 / 0.987 |
| Polvani-Kushner, T | `T_eq` and `kt` taken from `PolvaniKushnerColumns`, **not** reimplemented | decay 0.9835 (47 % over 38 steps) |
| QBO nudge | zonal-mean u, tropical window, mean-preserving | weight 0.221 |

Plus the §9.5 **increment fix**: the network output is scaled by `g = (1−e^{−aΔt})/(aΔt)`, which is
0.245 at the top sponge level — the increment is now damped like the physics, not left free.
Plus **rollout training**: phase 1 single-step (100 chunks), phase 2 K=4 (80 chunks), loss at every
step. 956,734 params, 3,600 steps, 8 minutes.

**Free run, global u RMSE (m/s), identical ICs to b7/b8:**

| day | no terms | sponge only | **all four + rollout** |
|---|---|---|---|
| 1.0 | 4.12 | 4.15 | **3.31** |
| 2.0 | 7.73 | 7.92 | **5.70** |
| 3.0 | 12.30 | 12.25 | **7.77** |
| 5.0 | 37.61 | 33.95 | **11.80** |
| 7.0 | 101.38 | 97.01 | **18.54** |

**max|u| by band at day 7** (truth in parentheses): prescribed 75.9 (75.9, exact by construction);
free interior **142.2 (83.5)** against 724.1 with no terms; sponge **85.4 (29.4)** against 808.6.

**5.5× lower error at day 7, ~10× in the sponge band, and the divergence onset moves from ~day 2
to ~day 5. It is still not stable** — the free interior is 70 % above truth at day 7 and growing,
and T RMSE reaches 15 K. Slower and later, not fixed.

**The important consequence: there are no more analytic terms to add.** All four §5.3 terms are
now carried exactly, and the truth run imposes no restoring force on free stratospheric u and v at
all — Held-Suarez's Rayleigh friction acts only at σ > 0.7, inside the prescribed band. In truth
those winds are held only by the dynamics being *right*. Our per-step error is ~8 % of rms|u|, and
that is what now accumulates.

**So the remaining gap is a learning problem, not a physics-carrying problem.** Two levers, in
order:
1. **Longer rollout horizon.** Training used K = 4 (one day); divergence sets in at day 5-7, which
   K = 4 never sees. K = 8-20 targets it directly.
2. **More data for the state head.** It saw 5,700 frames — the same data-limited regime in which
   the transport head kept improving from 8 → 96 → 2,340 → 7,020 frames (§6.4-6.11).

### 9.7 Due diligence on §3: is the direct formulation better? (no)

`c1_direct.py`, `c1_eval.py`. §3 argued the operator formulation structurally and §9.3 priced the
direct form as no faster, but its **accuracy was never measured**. Measured here as a three-rung
ablation, so the answer says which piece of structure is load-bearing. Identical corpus, years,
holdout site, schedule and final-checkpoint rule.

**TEST 1996, 468 untouched frames, held-out site 4 (pulse_4 + box twin):**

| model | R² | vs analytic | SE | beats |
|---|---|---|---|---|
| analytic operator only | 0.8195 | — | | |
| **O — analytic departure + learned correction** | **0.8668** | **+0.0473** | ±0.0016 | **468/468** |
| D1 — departure learned from scratch + analytic remap | 0.4931 | −0.3264 | ±0.0046 | 0/468 |
| D2 — fully direct, conv stack sees (flow, q), no gather | **−65.55** | −66.37 | ±1.11 | 0/468 |

**D2 fails catastrophically, and its training trace shows exactly why.** Fit R² climbed to 0.69
while held-out R² wandered between −0.47 and −4.39 — it learns the *specific tracers it is shown*
and has nothing to say about a tracer it has not seen. That is §3's argument, measured: a nonlinear
map from q to q′ is exact on-distribution and arbitrary off it. A local conv stack also cannot
physically move tracer 24 cells (the p99.9 displacement per 6 h) with a 9-cell receptive field.

**D1 is the more interesting rung.** Keeping the gather but learning the departure from scratch
still generalises (R² 0.49 > 0, and its validation trace rose steadily to 0.478) — so the *gather*
is what buys tracer-agnosticism, not the analytic trajectory. But it lands 0.33 below the analytic
operator it was trying to relearn. The departure field is only constrained where the fitted tracers
have gradients (the §6.3 finding), and from a cold start the search space spans displacements to
57°. The analytic solve supplies that for free and exactly.

**Conclusion: the structural argument in §3 survives contact with measurement, and the decomposition
is now attributable.**
- The **gather** provides generalisation to unseen tracers (D1 0.49 vs D2 −65.6).
- The **analytic departure point** provides accuracy (O 0.8668 vs D1 0.4931).
- The **learned correction** provides the last +0.0473, which is what the 6-hourly archive's wind
  sampling leaves on the table (§6.1 finding 5).

Each of the three pieces is doing a separate, measurable job. This closes the open question of
§9.3 and removes the last unmeasured assumption from the architecture.

### 9.8 Longer rollout horizon — the divergence is gone

`b10_rollout_long.py`, `b10_rollout_eval.py`. §9.6 trained with K = 4 (one day) while divergence
sets in at day 5–7, which K = 4 never sees. Curriculum extended to **K = 1 → 4 → 12 → 20** (five
days), batch shrinking as the horizon grows to hold BPTT memory flat. Memory restructure: whole
chunks stay resident as (41, 63, 192, 96) tensors and a sample is a traced start index sliced with
`dynamic_index_in_dim`, so per-sample copies of targets, prescribed levels and T_eq are gone.
Everything else identical to §9.6. 3,600 steps, 8 minutes.

**Free run, global u RMSE (m/s), identical ICs to b7/b8/b9:**

| day | no terms | sponge only | all four, K=4 | **all four, K=20** |
|---|---|---|---|---|
| 1 | 4.12 | 4.15 | 3.31 | 3.65 |
| 3 | 12.30 | 12.25 | 7.77 | **6.47** |
| 5 | 37.61 | 33.95 | 11.80 | **7.26** |
| 7 | 101.38 | 97.01 | 18.54 | **8.24** |

**Error growth is now sublinear and saturating** (1.82 → 3.65 → 5.47 → 6.47 → 7.26 → 8.24) where
K = 4 was accelerating (1.42 → 3.31 → 5.70 → 7.77 → 11.80 → 18.54). max|u| in the free interior sits
at 78.5 against truth 83.5 at day 7, where K = 4 reached 142.2. **The runaway that has defined this
model since §6.13 is gone.**

**But it was bought with damping, and that is the project's oldest trap.** σ_ratio = std(predicted)
/ std(truth), by band:

| day | 0–8 prescribed | 9–58 free | 59–62 sponge |
|---|---|---|---|
| 0.25 | 1.000 | 0.991 | 0.879 |
| 3 | 1.000 | 0.929 | 0.660 |
| 7 | 1.000 | **0.896** | **0.605** |

The free interior loses ~10 % of its variance by day 7 and is still falling; the sponge band loses
40 %. Single-step error is also slightly *worse* (u RMSE 1.82 vs 1.42, +28 %) — the standard
rollout-training trade.

This is the v2 signature: v2 was long-range stable **because** it damped the stratosphere, and its
QBO σ_ratio collapsed to 0.09. At 0.896 after 7 days we are nowhere near that, but the trend is
downward and the relevant horizon is years, not days. **Do not record this as "stable" without the
variance number beside it.**

Next: extend the rollout diagnostic past 9.5 days (currently capped by chunk length — it needs
chained chunks), and watch σ_ratio rather than RMSE as the gate.

### 9.9 Rollout horizon is CLOSED as a path to multi-year stability

`b11_long_rollout.py`. §9.8 read K = 20 as a win and flagged a damping risk. Both readings were
wrong. Two heads — identical architecture and analytic terms, differing only in training horizon —
run side by side to 90 days by chaining contiguous 10-day chunks.

| day | K=4 RMSE / σ_ratio | K=20 RMSE / σ_ratio | truth max&#124;u&#124; |
|---|---|---|---|
| 5 | 15.0 / 0.985 | 11.2 / 0.855 | 118.5 |
| 10 | **79.6 / 3.28** | 11.8 / 0.859 | 122.8 |
| 20 | 3,633 / 150.8 | 15.1 / 1.064 | 93.2 |
| 30 | 1.9e6 | **78.4 / 3.39** | 97.8 |
| 50 | 1.5e12 | 5,191 / 206.6 | 111.2 |
| 70 | inf | 1.5e6 | 92.7 |

**Both explode.** K = 20 postpones the runaway by roughly twenty days and then follows the same
trajectory. **Stable window ≈ 4–5× the training horizon**: K = 4 (1 day) holds ~10 days, K = 20
(5 days) holds ~20–25. Reaching 30 years would need K ≈ 9,000 — a 2,250-day BPTT window. Rollout
horizon is a linear-cost, linear-benefit lever against an exponential requirement.

**Retraction of §9.8's caveat.** σ_ratio does not collapse toward zero; it runs to 206 and then
68,929. The 0.896 at day 7 was a transient dip *before* the runaway, not damping toward climatology,
and the comparison to v2's variance collapse was wrong. The failure mode is explosion, not damping.

**What this closes and what it leaves.** §5.1 (longer rollout) is spent: it improves the 5–10 day
range and cannot reach multi-year. Two routes remain, and they are different in kind:

1. **The anomaly formulation** — `X = prescribed climatology + learned anomaly`. The free interior
   (levels 9–58) has *no* restoring force in truth; it is held only by the dynamics being right, and
   ~8 % per-step error compounds without bound. A prescribed climatology supplies the bound the
   physics does not. This is the one structural device in this project's history that held
   (γMv 1.013, the first model to hold its water) and it addresses the actual mechanism here.
2. **Offline transport** — drive the analytic operator with archive or ERA5 winds and drop the state
   head entirely. **1.58 ms/step, ~125×, and no stability problem at all**, because the transport
   half is rollout-stable (R² 0.86–0.92 over 38 steps, mass to 1.3 %). For transport studies, age of
   air, or tracer experiments under prescribed meteorology this is a complete product today.

Route 2 is available now and is the honest near-term deliverable; route 1 is the research path to a
free-running emulator.

### 9.10 The anomaly formulation fails — and the diagnosis was wrong all along

`n1_climatology.py`, `n2_anomaly_head.py`, `n3_long_eval.py`. X = prescribed monthly climatology
(training years only) + learned anomaly on the free band, with the anomaly persistence **learned**
per level rather than chosen. Training picked e-folding times of **14 d (138 hPa) to 40 d
(3.5 hPa)** — a measured price of stability rather than an assumed one.

90-day chained free run, u RMSE / σ_ratio (truth max|u| 93–122):

| day | K=4 | K=20 | **anomaly** |
|---|---|---|---|
| 10 | 79.6 / 3.28 | 11.8 / 0.859 | **10.9 / 0.920** |
| 20 | 3,633 / 150.8 | **15.1 / 1.06** | 39.6 / 2.19 |
| 30 | 1.9e6 | **78 / 3.4** | 548 / 26.3 |
| 70 | inf | 1.5e6 | 1.4e7 |

Better to day 10 and **less damped** (σ_ratio 0.920 vs 0.859 — the right direction), then it
explodes *sooner* than K=20. **Route 1 of §9.9, as formulated, does not work.**

### 9.11 Why: the instability is NUMERICAL, and §9.6 was wrong

A parallel investigation (`x1_`–`x6_` in `/data/jcm_tracer_emulator`) spectrally decomposed the
free-running b10 head. Zonal power ratio pred/truth in the free band:

| day | k0–3 | k4–15 | k16–48 | **k49–96 (grid)** |
|---|---|---|---|---|
| 1 | 0.93 | 0.75 | 0.93 | **1.4** |
| 10 | 0.88 | 0.59 | 1.44 | **122** |
| 20 | 1.02 | 1.03 | 17.4 | **1.1e5** |
| 39 | 129 | 2006 | 1.8e4 | **2.1e8** |

Truth's grid-scale power stays flat at 8–23 throughout. **Grid scale explodes first and cascades
upward**; planetary scales only move after k49–96 is 10⁵× too large. This is a numerical
instability, not attractor drift.

**JCM is a spectral model** — T63 triangular truncation plus per-level hyperdiffusion applied every
12-minute step. The emulator predicts in grid space with **neither**. So §9.6's conclusion, *"there
are no more analytic terms to add"*, is **wrong**: the two largest missing terms are numerical, not
forcing terms, and every restoring force tried since (sponge, PK, climatology anomaly at 14–40 d)
operates far too slowly to suppress grid-scale noise. That is why all three heads explode and why
longer rollout only bought time.

A bolted-on modal filter (no retraining) buys **6–7 orders of magnitude at day 90** — day-30
u RMSE 78 → 16, σ_ratio 3.4 → 1.2, max|u| 1712 → 109. Not a cure (σ_ratio still 2.8–5 at day 90,
against 40,000 steps for climate), and a plain T63 projection alone does almost nothing because the
archive's nodal u is ~3 % off the T63 manifold — u = spectral field / cos φ is not band-limited.
That argues for predicting the dycore's own prognostics (vorticity, divergence) instead of u, v.

### 9.12 Bug: Polvani-Kushner evaluated at a frozen season

`pk._equilibrium_temperature(..., 0.0)` — `tyear` hard-coded to **0.0 for every frame**, in
`b9_state_full.py`, `b10_rollout_long.py`, `b11_long_rollout.py`, `n2_anomaly_head.py` and
`n3_long_eval.py`. Verified independently: T_eq(tyear = 0.5) − T_eq(0.0) is **23.8 K rms, 62.7 K
max** over the free band, peaking at 62.7 K at 88.6°S.

The §5.3 argument is that PK relaxation is the *genuine restoring force*. It has been relaxing
toward a 1 January equilibrium year-round since §9.6. §5.1 specifies `sin/cos year_progress` as
trunk inputs and `features()` never included them, so the network could not compensate either —
and the free interior's dominant signal is the seasonal polar vortex.

**Every state-head result from §9.6 onward (b9, b10, n2) is confounded by this.** The comparisons
between them remain valid (all share the bug); their absolute quality does not.

### 9.13 Multi-year transport validation — the vertical IS the critical path, and §6.1 was wrong

Offline transport: analytic SL operator driven by the archive's own winds, 10 pulse tracers
autoregressively from 1990-01-01, **8,763 chained steps (6.0 sim-yr), zero seam breaks**. Four
drivers on identical frames (analytic, analytic+fixer, learned, learned+fixer). Full report in
`/data/TRANSPORT_MULTIYEAR_VALIDATION.md`; code `m0_pipeline_check.py`, `m1_offline_transport.py`,
`m2_report.py`. Self-validated: reproduces §9.3's learned−analytic gain as **+0.0462 on 77/78
frames** (published +0.0473) through its own code path, and reproduces the §2.2 truth decay table
exactly.

**(a) Structure decay — the operator UNDER-diffuses in 3-D, badly.** analytic/truth std-mean ratio:
pulse_1 1.79 (d100) → 2.80 (yr1) → 5.18 (yr5); pulse_4 1.50 → 2.30 → **25.71**. Model structure
stops decaying while truth keeps mixing.

**But the decomposition is the result.** Measured *within* levels — what the horizontal operator
actually controls — at year 1 the analytic/truth ratio is **1.02, 1.02, 1.40, 1.53, 1.07, 1.01,
0.71, 0.76, 0.84, 0.97**; mean |ln ratio| **0.170**, flat from d100 to yr1, against **0.930** for
the 3-D metric. Meanwhile tracer mass above 100 hPa: truth drains pulse_4 from 100 % to **25.9 %**
by yr1 and 10.2 % by yr5; the horizontal-only operator holds **100 % forever**. Centroid pressure,
pulse_4: truth 4 → 217 → 367 hPa; model frozen at 4.5 hPa.

> **The horizontal branch is accurate to ~17 % over a year. The entire multi-year failure is the
> missing vertical remap.**

**This contradicts §6.1 finding 4 and §8.1.** "0.097 layers per 6 h, a few-percent term, not the
critical path" is **true per step and false per year** — the omission is systematic, so it
accumulates into a frozen Brewer-Dobson descent. §10.3 had the right instinct ("a systematic
2 %-per-step omission compounds over ~40,000 steps") and I de-prioritised it anyway. **The vertical
remap is now the top item for the transport half.**

**(b) Mass.** Bare operator over 5 yr: **−59 % to +164 %** across tracers, not one-sided, so no
constant correction absorbs it (~1.3e-3/step, matching §2.4). Truth drifts 1.0–1.5 %. **The
archive's global fixer must ship with the operator.**

**(c) Twins — pass.** Within-level at yr1: truth 1.37/1.01/1.04/1.04/0.79 vs analytic
1.38/1.10/0.98/1.12/0.92. No sign the quasi-monotone limiter mishandles sharp edges; the 3-D
divergence is the vertical again.

**(d) Positivity / maximum principle — clean.** 0 negative cells, 0 new maxima in 8,763 consecutive
steps, analytic and learned alike.

**(e) The learned correction head does not survive rollout — and may be harmful.** 3-D mean
|ln ratio| is a wash (analytic 0.930 vs learned 0.929 at yr1). **Within-level it is worse: 0.170
analytic vs 0.426 learned at yr1, and 0/10 tracers better from month 3 onward.** The direction is
not uniform (sites 1/2/3/5 retain too much structure; site 4 over-diffuses to 0.43×), which is why
the 3-D metric cancelled it — **and site 4 is the only held-out site, where the damage is worst:
2.3× over-diffusion at yr1 from a correction that beats analytic on 77/78 single steps there.**

**Single-step skill did not survive autoregressive composition.** This is the same lesson as
`reference_tendency_probe_is_NOT_a_circulation_gate`, and it puts the headline +0.0473 of §9.3 in
question for any multi-year use. Caveat: one checkpoint, one split, one realisation.

**(f) Sub-stepping closed over years too.** K=8 vs K=1 over 5 years differs by **< 2 %**, not
consistently signed — as little as it bought at one step (§6.1 finding 1).

**Not tested:** the vertical remap itself; `sai`/`src_*`/`n2o`/`cfc11`/`aoa*`; age of air, vortex,
SSW; synthetic ICs; the other four holdout sites; ERA5-driven transport. Years 7–20 still running.

**Gotchas recorded:** `tracer_budget.layer_dp` needs `jcm_strat.levels.install("strat")` first or
`dp` is silently wrong; do **not** enable `jax_enable_x64` for float64 mass — it promotes the
departure-point solve out of the float32 the head was trained in.

## 10. Plan

Ordered by decision value. The working half is transport; the broken half is dynamics, so little
else is worth measuring until the state head is stable.

### 10.1 Critical path — state head with §5.3 forcing + short-horizon rollout

Two components, both specified above and neither built:

1. **Carry the analytic forcing explicitly** (§5.3, §4.4). PK relaxation is Newtonian relaxation to
   an analytic `T_eq` at τ 15 d — a real restoring force, and the dry-attractor work concluded only
   structural per-step budgets escape an attractor. The §6.13 state head had none of these terms.
2. **Short-horizon rollout / pushforward training.**

   *Nuance that cuts against the project's own prior:* more fs=24 rollout training previously
   **degraded** multi-year behaviour, because it optimises 6 days while those failures were
   multi-year. **This failure is at day 3–5 — inside rollout-training range**, so unlike the
   moisture case it should be addressed directly. Keep the checkpoint discipline regardless.

Consequence to state plainly: once the nudging mask and QBO target are carried, the emulator
consumes them every step. It is a **specified-dynamics emulator**, matching the corpus.

**Gate:** 38-step rollout with max|u| bounded (truth 103–122 m/s) and tracer R² > 0 at day 9.

### 10.2 Cheap, do regardless — all five holdout sites

Every network number is holdout site 4. §7 requires each site held out in turn; `b2_l2_sweep.py`
already does all five. Minutes of compute. Until then +0.0420 is a one-site result.

### 10.3 Vertical remap — small per step, possibly not over years

0.097 layers per 6 h is correctly negligible single-step (§6.1 finding 4). But a *systematic*
2 %-per-step omission compounds over ~40,000 steps, and age of air is the scientific point of the
stratosphere programme. Must go through archived ω; the divergence route is closed (§6.1).
**Gate:** reproduce the archived ω before trusting any σ̇.

### 10.4 Run the §7 protocol — none of it has been run

Structure decay vs the §2.2 truth curve, box/Gaussian twins, synthetic ICs, age of air vs the
dycore, vortex/SSW. This project has been burned precisely by skipping these once already — the
Cartesian-wind bug invalidated a whole campaign before anyone checked. Also re-run the
over-diffusion check on a 1990–91 frame: `pulse_4` in 1996 is homogenised (std/mean 0.043) and was
a weak test in §6.13.

### 10.5 Multi-year

The stated risk from the outset. Meaningful only after 10.1 and 10.3.

### Deprioritised

- **Learning the remap** — contraindicated by B0 (§4.4).
- **More transport-head scaling** — it is the working half; another +0.01 is irrelevant while the
  dynamics diverge.

### Upstream decision, not a modelling one

**This corpus cannot produce a QBO** (§2.1). If the science requires QBO–transport coupling
causally, no model work fixes it — it needs a run with gravity-wave drag, currently DEFERRED. That
decision sits above 10.1–10.5, because a regenerated corpus changes the training data underneath
all of them.
