# PARADIS on JCM — results

**Status 2026-09-30.** (5.1 QBO claim RETRACTED -- see that section.) A 25 M-parameter PARADIS port trained on jcm-strat phase-11 run A, plus the
campaign that preceded it. Organised by conclusion, not by experiment order. Code:
`/data/paradis_jcm` (git-tracked, v2 snapshot as the baseline commit). Corpus:
`/data/jcm_paradis_c1`. Companion to `TRACER_TRANSPORT_EMULATOR_{SPEC,RESULTS}.md`.

---

> **Active experiment:** the nudge-free arm ("emulate the dycore by itself") is registered in
> `/data/JCM_NUDGEFREE_HYPOTHESIS.md` with its prediction and falsifiers stated in advance.
> It tests whether the free-running failure is an ILL-POSED learning problem (levels 0-8 are
> 63%-per-step ERA5, not a function of state) rather than architecture or capacity.

## 1. Headline

| | result |
|---|---|
| **Prescribed emulator** | 32/32 full-year free runs; sigma_ratio 1.00; **vortex corr +0.927 vs a 0.727 climatology floor** — the first learned model here to beat climatology |
| **Non-prescribed emulator** | runs **10 years free with no truth at any step**, bounded (max\|u\| 67–104 vs truth 93–123). The **QBO claim is RETRACTED** (§5.1): measured against truth it has **0 sign reversals, 3% easterly, spectral peak at 2 months** — it locks permanently westerly. Stable but **QBO-dead** |
| **Nudge-free corpus** | obtainable and stable, but **no QBO at any Hines amplitude**, and a markedly weaker troposphere |
| **Not yet** | the non-prescribed model's vortex correlation is **below** the 0.727 climatology floor (12-yr scale-up: **0.238**); and it has **no QBO at all**, not merely a weak one |
| **12-yr scale-up** | 2.4x data, 4.5x presentations: sigma_ratio **1.02–1.04** (best measured here, beats the 0.811 floor) but vortex **0.238**, RMSE d30 15.09 vs climatology 11.4 and persistence 14.80. **Scale did not move the vortex.** anomR 1.17, so the failure is PHASE, not variance collapse |

---

## 2. What was built

Vendored the v2 snapshot (`cesm_run_2/code_snapshot`) verbatim, then changed only what JCM needs.
v2 deliberately, not v5/v6 — the measured finding is that the v5/v6 recipe is the regression.

**Pruned** (dry dycore, no radiation): moisture, precipitation, the surface block, TOA/time-of-day
forcings, `conservation*.py`, `omega_continuity`, CRPS heads, the sigmoid clamp. Native u, v rather
than Cartesian wind_x/y/z — the representation whose metric bug invalidated an earlier campaign.

**Added**: JCM's analytic forcing carried structurally — Polvani-Kushner relaxation, Rayleigh sponge,
QBO nudge, prescription of levels 0–8, with the increment fix `g = (1-e^{-a dt})/(a dt)`.

**Verified before training** — each independently, none assumed:

| component | check | result |
|---|---|---|
| data adapter | vs source netCDF | **bit-exact**, 12 random (chunk, frame, var, level) + surface |
| time axis | gaps | 1460 frames/yr, strictly 6-hourly, year seams included |
| grid | lat→row at nodes | machine-exact on Gaussian, equiangular and non-uniform grids |
| loss weights | vs Gauss-Legendre | 5.4e-08 relative (the original hard-failed on a Gaussian grid) |
| T_eq torch port | vs JAX original | **6.1e-05 K** over four seasons on real p_s |
| prescription | vs truth | exact to float32 round-off |
| sponge | decay factors | 0.0183 / 0.5519 / 0.9155 / 0.9870, exact |
| overfit one batch | loss | 1.307 → 0.00086 |

Training: 15 000 steps, 5.5 h on 4 H200s. train 0.0712 → **0.0127**, val 0.0302 → **0.0128** on
held-out 1995, still improving at the end.

---

## 3. The prescribed emulator

8 checkpoints x 4 initial conditions, 360-day free runs on held-out 1996.

| checkpoint | survival | sr d30 | sr d180 | vortex corr | vortex std |
|---|---|---|---|---|---|
| **climatology floor** | — | 0.811 | 0.811 | **0.727** | 1.00 |
| **best** (val-selected) | 360 d | 1.001 | 1.019 | **+0.927** | 0.93 |
| epoch=09 | 360 d | 1.025 | 1.000 | +0.917 | 0.77 |
| epoch=15 | 360 d | 1.723 | 1.775 | +0.847 | 1.40 |
| epoch=00 | 360 d | 0.910 | 0.798 | +0.359 | 0.78 |

**32/32 (run, IC) pairs survived the full year.** The best prior configuration managed 2 of 6.
The val-loss-selected checkpoint is also best on the free-run metrics — which was NOT true of the
small head, where free-run quality and training loss were nearly uncorrelated.

**Controls.** With the network increment zeroed, `D_UV = 1` outside the sponge so the free band is
exactly frozen: **vortex std 0.00**, correlation undefined. The vortex skill is the network's, not
the prescribed boundary leaking through the analytic terms. That control also shows **sigma_ratio
~1 is achievable by persistence** (frozen state scores 0.94–1.24), so it is a weak metric on its own
and must be quoted with the vortex std ratio.

---

## 4. How much it leans on truth — the decomposition

| variant | prescription | QBO nudge | sr d30 | vortex corr |
|---|---|---|---|---|
| (a) as trained | yes | yes | 1.001 | **+0.927** |
| (b) | yes | no | 0.91 / 0.89 | +0.874 / +0.437 |
| (c) | no | yes | 0.68 / 0.81 | +0.960 / +0.905 |
| (d) fully free | no | no | 0.68 / 0.49 | +0.816 / +0.394 |

Dropping the **whole troposphere** (c) costs variance, not vortex skill. Dropping the **QBO nudge**
(b) costs more vortex skill than dropping the troposphere does — consistent with the QBO modulating
the extratropical vortex through wave filtering.

---

## 5. The non-prescribed emulator

Retrained with prescription and the QBO nudge **off during training**, so the network predicts all
63 levels and sustains the tropical wind itself. PK relaxation and the sponge stay on — closed-form
in (lat, p, season), needing no truth, so a deployed model keeps them. Loss still scored against the
nudged truth, so the nudge's effect is the target.

**Training for the job eliminated the damping:**

| fully free | sr d30 / d180 | vortex corr |
|---|---|---|
| prescribed-trained, prescription stripped at inference | 0.68 / 0.49 | +0.816 / +0.394 |
| **free-trained** | **1.15 / 1.06**, 0.94 / 0.97 | +0.686 / +0.475 |

sigma_ratio recovered from 0.49–0.68 to 0.94–1.15. Damping was the failure mode that disqualified
every other arm in this campaign. But vortex correlation (~0.58 median) is **below the 0.727
climatology floor**: it holds its variance and loses its skill.

### 5.1 The 10-year free climate run — RETRACTED as a QBO result (2026-09-30)

No truth at any step: an initial condition, calendar forcings it computes itself, static geography.
It runs 3652 of 3652 days without diverging, and that part stands:

```
max|u| by year:  104 -> 70 -> 67 -> 69 -> 72 -> 81 m/s     (truth 93-123)
survived 3652 of 3652 days
```

**The QBO claim previously recorded here was WRONG and is retracted.** It read: "the emulator
sustains an oscillation its own training system cannot generate ... spans -11.8 to +22.9 and
reverses." That was inferred from the *annually-smoothed range* of equatorial ubar at 30 hPa. A
large smoothed range does not imply an oscillation -- it is equally produced by a one-off spin-up
transient plus a persistent bias, which is what actually happened.

Measured 2026-09-30 against truth extracted from the same source (p11a 1997-2006, cos-weighted
over |lat| <= 7.5 deg), equatorial zonal-mean u at 30 hPa:

| | std (m/s) | dominant period | % of time easterly | sign reversals, 6-mo smoothed |
|---|---|---|---|---|
| truth (nudged JCM) | 14.45 | **30.0 months** | 57.3 | 8 over 10 yr |
| **free emulator** | 5.73 | **2.0 months** | **3.0** | **0** |
| free dycore, Hines 3.0 | 5.36 | 12.0 | 49.6 | - (noise) |

Splitting off the spin-up makes it unambiguous:

```
emulator yr 0-10 : mean +12.93  std 5.73   reversals 0
emulator yr 2-10 : mean +13.51  std 3.06   min +3.64  max +21.52   easterly 0.0% of the time
truth    yr 2-10 : mean  -7.25  std 14.46  reversals 6             easterly 56.4%
```

**After year 2 the emulator never goes easterly again.** Its one sign change is the initial
transient. Its spectral peak sits at 2 months (high-frequency wobble), not in the QBO band. It also
has the wrong SIGN of mean equatorial wind: +13.5 m/s against truth's -7.3.

Figure: `/data/paradis_jcm/qbo_comparison.png` (truth / free emulator / free dycore, time-height
plus the 30 hPa series). The truth panel shows a textbook descending QBO; the emulator panel is
uniformly westerly after year 1.5.

What this means:
- The emulator is **stable but QBO-dead**, which is the same failure mode as the v2/grid17 line
  in the CESM campaign -- not a new success.
- The signature is specifically a **missing easterly phase**, not a weak oscillation. Nothing ever
  drives the tropics westward. That is consistent with the measured loss of the westward
  inertia-gravity branch (<1% of truth retained) in the sibling CESM diagnostic.
- Prescribed climatology (the anomaly formulation) **cannot** fix this: the QBO is ~28-month and
  not phase-locked to the calendar, so it averages out of any day-of-year climatology.

Reproduce: `/data/paradis_jcm/qbo_figure.py`, `/data/paradis_jcm/qbo_descent.py`.

---

## 6. The corpus route — nudge-free GWD

Ten simulated years each, T63 L95, Hines drag + Lott-Miller SSO, `nudging.enabled=false` and
`held_suarez.qbo=null` (both verified in the run's own log; zero ERA5 reads). ~2 h and 478 GB per run.

| `rms_launch_wind` | equatorial ubar @30 hPa mean | deseasonalised std | annually-smoothed range |
|---|---|---|---|
| 1.0 | +0.7 | 0.3 | -0.2 .. +1.1 |
| 2.0 | -0.2 | 1.2 | -1.3 .. +0.7 |
| **3.0** | -0.0 | **5.0** | -1.3 .. +1.3 |
| dry model, no drag | 12–14 | 4 | — |
| ERA5 | — | 17 | spans both signs |

Launch amplitude drives tropical variability monotonically (0.3 → 1.2 → 5.0), so the knob works —
but it produces **stronger noise, not an oscillation**. 3.0 m/s is already the top of the
operational range.

**The run is not broken.** The polar vortex survives intact (10 hPa/60S July: **+41.0** free vs
+38.4 nudged). What collapses is everything below: 50 hPa 60S +8.7 vs +24.7, and the 500 hPa jet
+9.6 vs +17.2. Without ERA5 nudging the troposphere runs at about half strength, generating weak
waves, hence weak wave-driving of the lower stratosphere and nothing to sustain a QBO.

Consistent with `qbo_nudging.py`'s own documentation: a QBO needs gravity-wave drag (now present),
fine tropical vertical spacing (L95 provides it), and **tropical waves from convection** — which a
dry model does not have. Two of three; the third is the wall.

**Every existing jcm-strat corpus, phases 11 and 13 alike, has `nudging.enabled=true`.** The ERA5
nudging is what makes these runs correspond to actual years. A run can track reality or run free,
not both.

---

## 7. What is established from the earlier small-head campaign

Kept because it constrains what is worth trying next.

- **The blow-up was a grid-scale numerical instability.** Power at zonal wavenumber 49–96 grows
  1.4x → 1e5x truth while planetary scales follow only afterwards. JCM is spectral with per-level
  hyperdiffusion every 12-min step; the emulator had neither.
- **Spectral dissipation is the single biggest lever** (25 d → ~200 d median survival), but its
  FORM does not matter: JCM's own ECHAM hyperdiffusion, a tuned exponential filter, and JCM's
  per-level profile at tuned strength are statistically indistinguishable over 3 seeds each.
- **Survival time is a very noisy metric.** Byte-identical config *and seed* gave 132 d and 40 d;
  one checkpoint spans 40–169 d across four start dates. Use medians over >= 4 ICs and >= 3 runs.
- **Metric floors.** A frozen climatology scores sigma_ratio **0.811** and vortex correlation
  **0.727** (0.888 truncated at day 130). No small-head configuration ever beat the vortex floor.
- **Error saturation is ~16 m/s** (same day-of-year, different years; the seasonal-cycle figures
  of 31.9 and 25.8 that preceded it were both wrong). Trajectory RMSE discriminates to ~day 60.
- **These were all nulls**: spectral zeta/delta prognostics (1.18x, RMSE flat), a zeta/delta head
  (0.46x, decisive regression), PV advect-and-invert as a backbone (15x worse; it is an
  unconstrained energy source), a global energy-budget loss, filter tuning.
- **Dycore distillation is the one learned intervention that worked.** The exposure gap is
  measured: the operator error is flat at ~2–3.6 m/s at truth states all year but **27x worse** at
  the states the model visits. Against a matched archive-only control, distillation gave **0.67 vs
  6.33 divergence events per 1000 steps**, identical across every seed, and took a typical
  checkpoint from diverging at day 100 to bounded at day 200.

### Offline, deterministic results (all independently verified)

| | |
|---|---|
| the real dycore reproduces JCM's 6-h step from archive fields | **1.3 % of a persistence increment**; free-runs a year at sigma_ratio 1.000, vortex corr 0.997 |
| thermal wind from the prescribed-band anchor | **96 %** of the extratropical 3-D zonal wind, verified across six years |
| heating response operator | **~90 %** of the zonal-mean wind response at day 10, across 8 geometries, both signs, 0.5–20 K |
| PV under SL advection | +6–8 % vs persistence on u; **-13 %** on T |

**Design rule for the coupled system:** apply radiative heating as a **tendency, not a step**. It
does not shorten the ~2-day geostrophic adjustment window (that is forcing-independent) but it cuts
the spurious wind injected during it by **24–86x** (0.99 → 0.023 m/s).

---

## 8. Bugs found, all silent

Recorded because each would have produced a plausible-looking wrong answer.

| bug | effect |
|---|---|
| `jcm_to_zarr.py` ran `main()` on import | a test's *import statement* launched a second full conversion, corrupting a year mid-verification |
| `open_mfdataset` coordinate union | constants from `terrain.nc` differed from the data by 2.8e-14, so 96 latitudes silently became 144 |
| dataset `__len__` = full frame count | last samples return an EMPTY target; only manifests on a full epoch |
| a one-year gap in the corpus | copied v2's year list, which skipped 1992 (CESM's val year); samples straddling the seam paired frames a YEAR apart |
| validation overlapped training | val was 1993, inside the 1990–1995 training range |
| zarr chunking `(5,24,48,64)` | 5.1x read amplification, 94 MB moved to use 19 MB; 30x I/O bottleneck |
| PyTorch silently on CPU | needs `LD_LIBRARY_PATH=/run/nvidia/driver/...` exactly as JAX does |
| gate RMSE missing the nlon factor | inflated by sqrt(192) = 13.9x |
| `pgrep -f <pattern>` in wait loops | matches the wait loop's OWN command line; loops that can never exit, and a `pkill` that killed the command issuing it. **Cost two days of wall time.** Use PID files. |

---

## 9. Where it stands

A non-prescribed emulator is **viable** — stable for a decade, bounded, with a learned QBO and a
polar vortex, needing no truth at any step — and **not yet accurate**: vortex correlation below the
climatology floor, QBO period and amplitude wrong.

The two routes to free-running have separated cleanly: **the corpus route failed to produce a QBO,
and the emulator route produced one anyway.**
