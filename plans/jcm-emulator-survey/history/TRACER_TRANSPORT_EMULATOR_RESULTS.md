# Tracer-transport emulator — results

**Status 2026-09-21.** Companion to `/data/TRACER_TRANSPORT_EMULATOR_SPEC.md` (1171 lines, the
chronological working record including failed routes). This document is organised by *conclusion*
rather than by experiment order, and states what is established, what is not, and what is next.

Corpus: `jcm-strat` phase 11 run A, `/data/JCM_stripped/jcm-strat-phase11/runs/p11a_*`.
Code: `/data/jcm_tracer_emulator/`.

---

## 1. Headline

| | result | evidence |
|---|---|---|
| **Transport accuracy** | R² **0.8668**, beating the analytic operator on **468 of 468** untouched frames (+0.0473 ± 0.0016) | §9.3 |
| **Speed** | **2.92 ms** per 6-h step, 24 tracers — **68×** JCM's 198.0 ms on the same H200 | §9.1–9.3 |
| **Single-step dynamics** | u RMSE **1.46 m/s** vs 2.82 persistence (+48 % skill); T 0.70 K (+46 %) | §6.12 |
| **Coupled free run** | **diverges at day 5–7.** Not solved. | §9.6 |

One simulated year in 4.3 s; thirty years in 2.1 minutes, against JCM's 2.4 hours — *if* the
dynamics can be made stable, which they cannot yet.

---

## 2. What the thing is

**A hybrid, not an emulator.** Transport executes the real semi-Lagrangian operator; a network
corrects its trajectory. Dynamics are learned outright, wrapped in four closed-form forcing terms
lifted from JCM itself.

| component | kind | detail | cost |
|---|---|---|---|
| departure-point solve | closed form | Cartesian fixed point, 2 iterations | 0.36 ms |
| tracer remap | closed form | cubic Lagrange gather + quasi-monotone limiter | 1.22 ms |
| trajectory correction | **learned** | 1.46 M params, bounded ~1 grid cell, levels as channels | 0.98 ms |
| dynamics (u, v, T, p_s) | **learned** | 957 k params, levels as channels, residual | 0.36 ms |
| PK · sponge · ERA5 nudge · QBO | closed form | exact, from `upper_sponge.py`, `polvani_kushner.py`, `qbo_nudging.py` | ~0 |

Of the 0.8668 R² reached on transport, the closed-form operator supplies **0.8195** and the network
**0.0473** — the analytic part does ~95 % of the transport work. The learned content is concentrated
in the dynamics.

**Design invariant: nonlinear in the flow, linear in the tracer.** Tracers are neither input nor
output of any network; they are carried by an operator the network parameterises. That is what
allows an unseen tracer field to be transported at all.

---

## 3. What is established

### 3.1 The corpus, and three limits that were not in the run records

20 years, 29,220 contiguous 6-hourly frames, T63 × 63 levels, 2.4 TB, zero NaN.

- **Only levels 9–58 (137.9 → 0.27 hPa) are free dynamics.** Levels 0–8 are relaxed to ERA5 on a
  *hard binary mask* (τ 6 h, no taper); 59–62 are Rayleigh-sponged at τ 1.5 h–19 d.
- **The QBO is a boundary condition, not a dynamical outcome.** The dry dycore has no gravity-wave
  drag and cannot generate one. No emulator trained on this corpus can produce a QBO.
- **Nothing leaves, but structure dies.** Pulse mass is conserved to 0.2 % over 19 years; what
  decays is spatial structure — `pulse_4` is uniform to four digits by 2009. Clean pure-transport
  signal exists only in roughly 1990–1996.

Archive conservation is enforced by a **global proportional mass fixer**, not by the local operator.

### 3.2 The transport residual is wind-history sampling, not operator error

Chain of eliminations (§6.1–6.2):

- Air moves **1.7 cells (median), 24.3 (p99.9), 32.5 (max)** per 6 h. A face-flux formulation cannot
  represent that; a departure-point formulation handles it natively.
- The analytic operator with true winds explains R² **0.83** (range 0.70–0.92 over 16 frames).
- **Not trajectory error** — sub-stepping K = 1 → 8 buys +0.017 and plateaus by K = 4.
- **Not interpolation diffusion** — variance ratio 1.004; the operator does not damp at all.
- **Not the missing vertical** — vertical displacement is 0.097 layers per 6 h, 2 % of a pulse's
  vertical width against 28 % horizontally.
- **It is wind sampling.** Truth integrated thirty wind fields; the archive has two endpoints
  differing by 14 % of rms|u|. The midpoint wind beats either endpoint in **16 of 16 frames**.

So R² 0.83 is the ceiling of *the archive's time resolution*, not of the operator — which is
precisely what the learned correction recovers.

### 3.3 The architecture decomposes, and each piece is separately attributable

Three-rung ablation, identical corpus and protocol, held-out site at 3 hPa (§9.7):

| variant | keeps | R² |
|---|---|---|
| analytic only | departure + remap | 0.8195 |
| **O — shipped** | both + learned correction | **0.8668** |
| D1 | remap only; departure learned cold | 0.4931 |
| D2 | nothing; CNN reads q, emits q′ | **−65.55** |

- The **gather** buys generalisation to unseen tracers (D1 0.49 vs D2 −65.6).
- The **analytic departure point** buys accuracy (O 0.867 vs D1 0.493).
- The **learned correction** buys the last +0.047.

*Caveat, stated plainly:* D2's four-layer 3×3 stack has a nine-cell receptive field against a
24-cell displacement, so its collapse partly demonstrates a known limitation rather than a verdict
on learned-vs-analytic. D1's 0.4931 also reflects a 30-minute cold-start budget. The narrow, durable
claim is that the *gather* is what buys tracer-agnosticism and that correcting the analytic
departure is far easier than relearning it.

### 3.4 Data scaling paid throughout, once stability and selection were right

| distinct training frames | test gain | beats analytic |
|---|---|---|
| 8 | +0.0044 | 3/4 |
| 96 | +0.0128 | 30/32 |
| 2,340 | +0.0322 | 437/468 |
| 7,020 | +0.0420 | 468/468 |
| 7,020, levels-as-channels | **+0.0473** | **468/468** |

Three things had to be right simultaneously — data volume, optimisation stability (cosine decay +
gradient clipping), and checkpoint selection — and each was diagnosed only after the previous was
fixed. The curve never turned over: 7,020 frames is 0.3 % of what is available.

### 3.5 Speed, measured on identical hardware

JCM re-benchmarked on the same H200 (the PROGRESS.md figure was H100; the box was upgraded
mid-project — H200 turns out to be only ~3 % faster here, so the comparison survives).

| | JCM | emulator | ratio |
|---|---|---|---|
| tracer transport, 24 tracers | 64.0 ms (**32.3 %** of the step) | 1.22 ms | 52× |
| dynamics + physics | 134.0 ms | 0.36 ms | 372× |
| **whole 6-h step** | **198.0 ms** | **2.92 ms** | **68×** |

The 52× on transport involves **no learning** — both sides run the same operator; the gap is one
application per 6 h instead of thirty at 12 min.

Restructuring the transport head levels-as-channels (PARADIS's design) was a **strict win**: 2.1×
faster *and* +0.0053 R² more accurate, because levels can now couple and the correction depends on
vertical shear.

### 3.6 Two levers that are closed

- **Longer time steps do not work.** At a fixed 1-day horizon: four 6-h steps R² 0.886; two 12-h
  steps 0.733; one 24-h step **−0.373** — worse than leaving the tracer where it started. 6 h is
  already at the knee, so the 100× band is not reachable by stepping longer.
- **Learning the remap is contraindicated.** It is not the error source (variance ratio 1.004), it
  is only 42 % of a 2.92 ms step, and replacing it forfeits positivity and the maximum principle.

---

## 4. What does not work

**The coupled free run.** With all four analytic terms carried exactly, the increment fix, and K = 4
rollout training:

| day | no terms | sponge only | all four + rollout |
|---|---|---|---|
| 3 | 12.30 | 12.25 | **7.77** |
| 5 | 37.61 | 33.95 | **11.80** |
| 7 | 101.38 | 97.01 | **18.54** |

*(global u RMSE, m/s)* — 5.5× better at day 7 and onset delayed from ~day 2 to ~day 5, but still
divergent. The free interior reaches 142 m/s against truth 83 by day 7.

**The transport half is rollout-stable**: driven by archive winds it holds R² 0.86–0.92 across 38
autoregressive steps with mass conserved to 1.3 %. The state head is what diverges.

**There are no more analytic terms to add.** The truth run imposes *no* restoring force on free
stratospheric u and v — Held-Suarez friction acts only below σ = 0.7, inside the prescribed band.
Those winds are held only by the dynamics being right. Per-step error is ~8 % of rms|u|, and that is
what accumulates. **The remaining gap is a learning problem, not a physics-carrying problem.**

---

## 4b. Update 2026-09-22 — longer rollout horizon fixed the divergence

`b10_rollout_long.py`. Curriculum extended K = 1 → 4 → 12 → 20 (five days), so the loss finally
covers the day 5–7 regime that was failing.

| day | all four terms, K=4 | **all four, K=20** |
|---|---|---|
| 3 | 7.77 | **6.47** |
| 5 | 11.80 | **7.26** |
| 7 | 18.54 | **8.24** |

*(global u RMSE, m/s)* — error growth is now **sublinear and saturating**; max|u| in the free
interior is 78.5 against truth 83.5 at day 7, where K = 4 reached 142.2. **The runaway is gone.**

**With one caveat that must travel with the result.** σ_ratio (std predicted / std truth) in the
free interior falls 0.991 → 0.929 → **0.896** by day 7 and is still declining; the sponge band
reaches 0.605. Single-step error is 28 % worse (1.82 vs 1.42 m/s). Stability was bought partly with
damping — the exact mechanism by which v2 was long-range stable and ended with a QBO σ_ratio of
0.09. At 0.896 over 7 days this is not that, but the relevant horizon is years.

**The gate for all further work is σ_ratio, not RMSE.** A model can be made arbitrarily stable by
damping it to climatology, and RMSE will look excellent while the science is gone.

---

## 4c. Update 2026-09-22 (later) — rollout horizon is closed

§4b read K = 20 as fixing the divergence. A 90-day chained rollout comparing both heads shows it
does not: **both explode**, K = 20 simply postpones it by ~20 days.

**Stable window ≈ 4–5× the training horizon.** K = 4 holds ~10 days; K = 20 holds ~20–25. Thirty
years would need K ≈ 9,000. This is a linear lever against an exponential need, and it is spent.

*Retraction:* §4b warned the stability was "bought with damping" and invoked v2's variance collapse.
Wrong — σ_ratio runs to 206 and then 68,929. The dip to 0.896 at day 7 was a transient before the
runaway. The failure is explosion, not damping.

**Two routes remain.** (a) The **anomaly formulation** — `X = prescribed climatology + learned
anomaly` — because the free interior has no restoring force in truth and ~8 % per-step error
compounds without bound; this is the one structural device in the project's history that held.
(b) **Offline transport** — drive the analytic operator with archive or ERA5 winds, drop the state
head: 1.58 ms/step, ~125×, no stability problem, and a complete product today for transport studies,
age of air and tracer experiments under prescribed meteorology.

---

## 4d. Update 2026-09-22 (evening) — three defects found, all actionable

Three independent findings landed, and together they redraw the priorities. All confirmed against
code or reproduced independently.

**1. The state-head divergence is a NUMERICAL instability, not attractor drift.** Spectral
decomposition of the free-running head: grid-scale (k49–96) power reaches 122× truth by day 10 and
2.1e8× by day 39, and cascades upward — planetary scales only move afterwards. JCM is a spectral
model with T63 truncation and per-level hyperdiffusion every 12-min step; the emulator has
**neither**. A bolted-on modal filter buys 6–7 orders of magnitude at day 90 without retraining.
**§4b/§4c's framing was wrong**: the two biggest missing terms are numerical, not forcing terms,
and every restoring force tried (sponge 1.5 h, PK 15 d, learned anomaly 14–40 d) is far too slow to
suppress grid-scale noise.

**2. Polvani-Kushner was evaluated at a frozen season.** `tyear` hard-coded to 0.0 in five scripts.
T_eq(July) − T_eq(January) is **23.8 K rms, 62.7 K max** over the free band (verified
independently), and the network had no seasonal input channel to compensate. Every state-head
result from §4a onward is confounded; comparisons between them survive, absolute quality does not.

**3. The missing vertical remap is the whole multi-year transport failure.** Within levels — what
the horizontal operator controls — the analytic operator is accurate to **~17 % at year 1**, with
exact positivity and an exact maximum principle over 8,763 consecutive steps. But tracer mass above
100 hPa: truth drains pulse_4 to 25.9 % by year 1; the horizontal-only operator holds **100 %
forever**, centroid pressure frozen at 4.5 hPa against truth's 217 hPa. **This contradicts the
earlier "0.097 layers per 6 h, not the critical path" reading** — true per step, false per year.

**And the learned correction may be harmful over rollout.** Within-level at year 1: analytic 0.170
vs learned 0.426, **0/10 tracers better from month 3 onward**, worst on the only held-out site
(2.3× over-diffusion) — from a correction that beats analytic on 77/78 single steps there. Single-step
skill did not survive autoregressive composition. One checkpoint, one split.

**Revised priorities:** (i) build the vertical remap — now top of the transport half; (ii) add
spectral truncation + hyperdiffusion to the state head; (iii) fix the PK season and add the seasonal
channel; (iv) re-price the learned correction on a rollout metric before carrying it anywhere.

---

## 5. What else needs to be done

Ordered by what unblocks the most.

### 5.1 Longer rollout horizon — the critical path
Training used K = 4 (one day). Divergence sets in at day 5–7, which K = 4 never sees. K = 8–20
targets it directly, at the same cost. **Gate:** 38-step rollout with max|u| bounded (truth
103–122 m/s) and tracer R² > 0 at day 9.

### 5.2 More data for the state head
It has seen 5,700 frames — the same data-limited regime in which the transport head improved
monotonically from 8 to 7,020 frames. Cheap, and the streaming pipeline already exists.

### 5.3 All five holdout sites
Every transport number is holdout site 4. The §7 protocol requires each site held out in turn, and
`b2_l2_sweep.py` already does all five. Minutes of compute. Until then +0.0473 is a one-site result.

### 5.4 The vertical remap
Never built. A few-percent term per step (0.097 layers), but a *systematic* 2 %-per-step omission
compounds over ~40,000 steps, and age of air is the scientific point of the stratosphere programme.
Must go through the archived ω — recovering the model's divergence from archived nodal winds is
ill-conditioned (7.7 % round-trip error interior, continuity off by 50×) and that route is closed.
**Gate:** reproduce the archived ω before trusting any σ̇.

### 5.5 Run the §7 evaluation protocol — none of it has been run
Structure-decay against the measured truth curve, box/Gaussian twins, synthetic ICs (constants stay
constant, mass holds, no new extrema), age of air vs the dycore, polar vortex and SSW frequency.
These are the metrics a reviewer would ask for, and this project has been burned before by skipping
them — the Cartesian-wind bug invalidated a whole campaign before anyone checked. Also re-run the
over-diffusion check on a 1990–91 frame: `pulse_4` in 1996 is already homogenised (std/mean 0.043)
and was a weak test.

### 5.6 Multi-year
The stated risk from the outset, and still entirely untested — every evaluation so far is
single-step or ≤ 9.5 days. Meaningful only after 5.1 and 5.4.

### 5.7 Upstream, and not a modelling decision
**This corpus cannot produce a QBO.** If the science requires QBO–transport coupling causally, no
model work fixes it; it needs a run with gravity-wave drag, currently deferred. That decision sits
above all of the above, because a regenerated corpus changes the training data underneath it.

---

## 6. Method notes that cost time to learn

- **Score on ≥ 468 frames.** A 32-frame estimate gives the right ordering but an unreliable
  magnitude (one model's 32-frame gain was +0.0125 against a true +0.0189).
- **Take the final checkpoint; do not select on 1992.** Its analytic baseline is 0.8720 against
  1996's 0.8195, so it has no headroom and mis-ranks models. It misled us three times — on the last
  occasion it would have rejected the best model outright, at a cost of +0.0081. *This does not
  overturn the project's "never take the last checkpoint" rule, which comes from rollout training
  and multi-year metrics — a different regime.*
- **Concurrent netCDF reads segfault.** HDF5 is not thread-safe unless built for it; two reader
  threads give `EXIT=139` with no Python traceback. Killed three runs before diagnosis.
- **`JAX_PLATFORMS` set at module level silently pins training to CPU.** Set `JAX_PLATFORMS=cuda`
  explicitly on GPU launches, alongside
  `LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu`.
- **Assert every string replacement in patch scripts.** One unasserted replacement silently failed
  and lost a checkpoint save, costing a rerun.

---

## 7. Corrections to earlier claims in this project

Recorded because they were stated confidently before being measured.

| claim | corrected to |
|---|---|
| sub-stepping is the lever that recovers 6-h accuracy | it is not — K = 1 → 8 buys +0.017 |
| the missing vertical transport is the residual, and the critical path | a few-percent term; 0.097 layers per 6 h |
| §6.3's +0.058 transport gain | inflated by selecting L2 on the held-out score; honest value ~10× smaller at that stage |
| best transport model is +0.0385 | +0.0420, then +0.0473 with levels-as-channels |
| the TOMAS Δt-conditioning precedent supports longer steps | not analogous — that system is a local ODE with no advection |
| fixing the sponge will stabilise the free run | it improves the target band 2.4–4× and does not prevent divergence |
| the 35× speedup | valid, but the original comparison mixed H100 and H200; direct re-benchmark gives 68× for the current model |

---

## 8. Where everything is

| | |
|---|---|
| Spec / working record | `/data/TRACER_TRANSPORT_EMULATOR_SPEC.md` |
| This document | `/data/TRACER_TRANSPORT_EMULATOR_RESULTS.md` |
| Code, logs, checkpoints | `/data/jcm_tracer_emulator/` |
| Corpus | `/data/JCM_stripped/jcm-strat-phase11/runs/p11a_*` (1990–2009, 2.4 TB) |
| Explainer (published) | https://claude.ai/artifact/FjPdEk5CzznasreJz9FDVT |

Key scripts: `b0_reference_operator.py` (operator fidelity), `b2_net_stream*.py` (transport head),
`b5_net_lc.py` (levels-as-channels, current best), `b3_state_head.py` / `b9_state_full.py` (dynamics),
`b9_rollout.py` (coupled free run), `c1_direct.py` (the D1/D2 ablation),
`b6_timestep.py` (time-step pricing).

*Note: the `jcm-strat` repo is owned by `susanne`; everything here lives outside it and was written
to `/data`, following the existing `RADIATION_MODULE_SPEC` / `RADIATION_MODULE_RESULTS` convention.*
