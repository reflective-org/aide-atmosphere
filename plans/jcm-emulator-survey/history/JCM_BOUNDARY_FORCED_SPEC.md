# JCM boundary-forced stratospheric emulator — experiment spec

**For the second 8x H200 machine mounting the same `/data`. Written 2026-10-01.**

Self-contained. Needs `/data/jcm_paradis_c1` (corpus), `/data/paradis_jcm` (code), and GPUs.
Independent of the main box.

**Evidence convention.** Every number is tagged **[M]** = measured by a command run while writing
this spec (the command is given), **[C]** = cited from a named document, or **[E]** = ESTIMATE.
Nothing else is quoted. Where a figure handed to this spec did not reproduce, the measured value is
given and the discrepancy is stated (§10.6, §10.7, §13).

Python used throughout: `/home/ubuntu/atmospheric_scale/paradis_model/paradis_venv/bin/python`
(call it `$P`). GPU jobs need
`LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu` or torch silently runs on CPU.

---

## 1. Objective

Train and gate a free-running emulator of **model levels 9–62 only** (137.92 hPa → 0.018 hPa),
with **levels 0–8 (985.34 → 151.53 hPa) prescribed every step from a tropospheric trajectory
replayed from a different year** than the one being simulated.

The target is the one metric two formulations and ~155 GPU-h have failed to move: **polar-vortex
correlation above the 0.727 climatology floor**, without the excess uncorrelated internal variance
(anomR 1.6–2.4) that the last run diagnosed. [C: `JCM_ANOMALY_FORMULATION_RESULTS.md` §1, §2]

---

## 2. Established facts — each re-verified here

### 2.1 Levels 0–8 of the training corpus are ERA5 played back, not dynamics

The corpus is jcm-strat phase-11 run **p11a**. Its own composed config, archived in every output
chunk's provenance JSON, contains [M]:

```
nudging:  enabled: true   source: era5   tau_hours: 6.0   pbl_levels: 2
          nudge_temperature: true   min_pressure_hpa: 150.0   freq: 6h
```
> `grep -n -i "nudg" /data/JCM_stripped/jcm-strat-phase11/runs/p11a_19970101/longrun_day10.nc.provenance.json`

Winds **and** temperature are nudged. The design decision is recorded verbatim [C:
`/data/JCM_stripped/jcm-strat/KEY_DECISIONS.md` rows 7–8]: *"ERA5 nudging of winds + temperature,
τ = 6 h, `pbl_levels: 2`"* and *"Nudging cutoff 150 hPa hard mask"*.

The 150 hPa mask lands exactly between model levels 8 and 9 [M]:

| level | 0 | 7 | **8** | **9** | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|
| p (hPa) | 985.34 | 184.46 | **151.53** | **137.92** | 104.01 | 86.12 | 78.35 |
| hybrid b | 0.9721 | 0.03228 | 0.012521 | 0.007397 | 0.000723 | 0.000023 | **0.0** |

> `$P -c "import numpy as np; z=np.load('/data/paradis_jcm/jcm_analytic.npz'); print(z['NUDGE_TOP'], z['plev'][:16])"`
> → `NUDGE_TOP = 9`, confirming the emulator's own prescription boundary is levels 0–8.
> b from `hybrid_b_full` in `runs/p11a_19970101/longrun_day10.nc` [M].

A τ = 6 h relaxation applied over one 6-h output interval removes `1 − e^(−6/6) = 63.2%` of the
deviation from ERA5 per saved step [M, arithmetic].

### 2.2 Therefore the troposphere is not learnable as a function of state

It is set by an exogenous stream the network cannot see. Any loss term on levels 0–8 asks the
network to predict the output of an ERA5 lookup. This is not a conjecture about model capacity; it
is a property of the data-generating process in §2.1.

### 2.3 The all-63-level free-running emulator blows up there, and the current gate cannot see it

`/data/jcm_emulated_winds/1990` is a 360-day fully free rollout (`prescribe=False, qbo=False`,
written by `emulate_winds.py` on 2026-10-01 15:50; 1458 frames). Scored against truth
`/data/jcm_paradis_c1/1990` [M]:

| day | emul max\|u\| lev 9–58 | truth max\|u\| lev 9–58 | emul max\|u\| lev 0–8 | emul global-mean T | truth global-mean T |
|---|---|---|---|---|---|
| 100 | 91.6 | 61.9 | 92.5 | 233.5 | 234.3 |
| 180 | 143.8 | 119.2 | 103.5 | 235.6 | 233.8 |
| 200 | 100.4 | 126.2 | 129.2 | 247.2 | 233.7 |
| **210** | **370.1** | 142.1 | **420.9** | **291.8** | 233.7 |
| 250 | 323.1 | 97.9 | 420.0 | 283.0 | 234.0 |
| 360 | 308.6 | 110.6 | 392.0 | 281.5 | 233.6 |

At day 250 exactly: max|u| **420.0** at levels 0–8 and **323.1** in the free band; median |u| over
levels 9–58 **88.30** vs truth **11.34**; cos-weighted global-mean T **283.0 K** vs truth
**234.0 K** (**+49.0 K**). Levels 0–8 are *cold* (205.9 K vs 251.9 K) while the free band is *hot*
(302.1 K vs 229.7 K) — the column has inverted, not merely warmed.

**The gate cannot see any of this.** Reading `/data/paradis_jcm/gate_paradis.py` [M]:
* `LO, HI = 9, 59` — the divergence check and sigma_ratio both ignore levels 0–8 **and** the 4
  sponge levels. The 420 m/s band is outside the metric entirely.
* `FAIL = 500.0` — never tripped: the run settles at ~310–325 m/s and stays there for 150 days.
* Line `for m in (30, 90, 180, 270, 359)` computes sigma_ratio and anomR at five horizons; the
  per-IC print and the SUMMARY table emit only 30/90/180. **270 and 359 are computed and thrown
  away.** The transition here happens between day 200 and day 210 — after the last reported column.

### 2.4 A calendar climatology at the boundary is wave-starved — but by 38%, not by ~100%

Seasonal fraction of variance `1 − var_anom/var_tot` over levels 0–8 from
`/data/paradis_jcm/meta/clim_harmonics_1997_2008.npz` (K=3 annual harmonics, n=17532 frames,
1997–2008) [M]:

| | u_wind | v_wind | temperature | omega |
|---|---|---|---|---|
| unweighted grid-point sum, lev 0–8 | **0.213** | **0.029** | **0.617** | 0.031 |
| Gauss-weighted in latitude, lev 0–8 | 0.273 | 0.034 | 0.554 | 0.042 |
| Gauss-weighted, lev 9–62 | 0.558 | 0.100 | 0.569 | 0.045 |

The figures 0.213 / 0.029 / 0.617 handed to this spec are the **unweighted** variant; both are given
because the loss is latitude-weighted and the weighted numbers are the physically relevant ones.
97% of v at the boundary is non-seasonal either way.

**But the correct statement about wave flux is weaker than "almost none".** Measured eddy heat flux
`[v* T*]` (deviations from the zonal mean, zonally averaged), DJF 1996, daily sampling [M]:

| level | lat | total | stationary part a day-of-year climatology retains | transient | cross |
|---|---|---|---|---|---|
| 8 (151.53 hPa) | 56.9 N | 22.19 | 12.15 (55%) | 8.99 | 1.05 |
| 9 (137.92 hPa) | 55.0 N | 24.14 | **14.63 (61%)** | **9.26 (38%)** | 0.25 |

units K m/s. A per-grid-point day-of-year climatology keeps the **stationary** waves (it is fitted
point by point, not zonally), so it supplies ~61% of the DJF flux and loses the transient 38% —
plus **all** intraseasonal and interannual variability, i.e. every wave burst that precedes a
sudden warming. That is the real defect, and it is enough to disqualify the climatology route, but
the "almost NO upward wave flux" framing is overstated and should not be repeated.

**A replay year keeps essentially all of it** [M]: DJF `[v*T*]` peak at level 9 is 24.14 (1996) vs
**21.17** (1997) — 88%; at level 8, 22.19 vs 17.86 — 80%.

### 2.5 A replay year is not truth for the simulated year

Pointwise correlation between 1996 and 1997 on the same calendar date, anomalies taken from each
year's own time mean, daily, full globe [M]:

| field | level 0 (985 hPa) | level 5 (354 hPa) | level 8 (151.5 hPa) |
|---|---|---|---|
| v_wind | +0.028 | −0.002 | **+0.030** |
| u_wind | +0.081 | +0.147 | +0.314 |
| temperature | +0.391 | +0.670 | +0.705 |

The residual correlation is the shared seasonal cycle (which is what we *want* at the boundary);
the synoptic realisation is independent (v: r = 0.03). This is precisely the status of a prescribed
SST field in an AMIP run: right climate, right statistics, wrong weather.

### 2.6 Prescribing `ps` from the replay year is harmless above 90 hPa

`normalized_surface_pressure` is `p_s/101325` (measured range 0.5112–1.0860, mean 0.9660) [M].
RMS |Δps| between 1996 and 1997 on matched indices is 0.01084 = **1099 Pa** (p99 3586 Pa) [M].
Since `b = 0` for levels ≥ 15 (p ≤ 78.35 hPa), those levels' pressures are unaffected. The worst
affected free-band level is 9, where `b = 0.007397` gives an RMS pressure shift of **8.1 Pa =
0.081 hPa on 137.92 hPa = 0.059%** [M]. Negligible; state it in the results doc rather than
guarding against it.

---

## 3. The proposal, and what it does not claim

Emulate levels 9–62 free-running. Each step, overwrite levels 0–8 of `u, v, T, omega` and the
single `normalized_surface_pressure` channel from a **replay trajectory**: the same calendar date
from a year that is **never** the year being simulated.

Why this and not the alternatives:
* **vs. prescribing truth** (the current `prescribe: true` path, vortex corr +0.927 [C:
  `PARADIS_JCM_RESULTS.md` §3]): that run is given the simulated year's own troposphere. Replay is
  not (§2.5), so the vortex skill it earns is skill, not leakage.
* **vs. prescribing a seasonal climatology** (the anomaly formulation, vortex corr +0.047 [C:
  `JCM_ANOMALY_FORMULATION_RESULTS.md` §1]): loses the transient flux and all interannual
  variability (§2.4).
* **vs. a stochastic synthesis of the boundary**: needs a generative model with correct `v'T'`
  cross-spectra. Replay gets that for free, exactly.

### 3.1 Limitation, stated plainly

**This does NOT satisfy a strict "no external data stream" requirement.** A deployed
boundary-forced emulator still needs 1460 frames/year of tropospheric `u, v, T, omega, ps` from
somewhere, and in this corpus that troposphere is itself ERA5-nudged. The honest claim is:

> The stratospheric evolution is learned and free-running; the troposphere is a prescribed boundary
> condition of the same logical status as prescribed SSTs in an AMIP run.

If the project's requirement is a model that needs **nothing** at inference, this experiment does
not deliver it and should be scoped as a diagnostic: it isolates *whether the stratospheric
dynamics are learnable at all when the boundary is right*, which no run so far has tested. If that
answer is yes, the boundary becomes a separate, smaller modelling problem. If it is no, the
stratospheric operator itself is the defect and no amount of boundary work will help. Both outcomes
are decision-grade; say so up front rather than presenting it as a deployable model.

---

## 4. BLOCKING: gate fixes required before any verdict

> **STATUS 2026-10-01: these fixes are APPLIED** in `/data/paradis_jcm/gate_paradis.py` (backup
> `gate_paradis.py.bak2`), on the shared `/data`, so this machine inherits them: survival now
> distinguishes `died` / `DATA-END` / `ok` instead of reporting data exhaustion as 360 d; `FAIL`
> lowered 500 -> 200 m/s; the dataset spans into `year+1` so late ICs have truth (IC 1092 + 360 d
> needs 92 days beyond 1996); `sr270`/`sr359` and a global-mean-T drift column are reported.
> A re-score of both existing ladders is running; do NOT compare against the old published numbers.


`gate_paradis.py` must be fixed **first**, and the fixed gate must be re-run on an existing
checkpoint before the new run launches. Do not score the new experiment with the current tool.

**Every "survived 360 d" claim in this campaign is suspect until re-scored.** Two independent
reasons:

### 4.1 `surv` conflates "survived" with "ran out of data" — a real bug

`gate_paradis.py` rolls inside **one year's** dataset and contains:

```python
j = ic + t
if j + 1 >= len(ds): break        # <- no record that this was truncation
...
surv = a.days if dead is None else dead
```

`dead` stays `None`, so the run is reported as `surv 360 d`. With year 1996 (1464 frames) and the
default `--ics 0,364,728,1092`:

| IC | frames available | days actually rolled | reported |
|---|---|---|---|
| 0 | ~1462 | 360 (capped) | 360 |
| 364 | ~1098 | ~274 | **360** |
| 728 | ~734 | ~183 | **360** |
| 1092 | ~370 | **~92** | **360** |

Corroborated in the logs: `sr180` is `nan` for IC 1092 in **every** row of
`/data/paradis_jcm/gate_big.log` and `gate_base_g0.log` [M]. The published medians at d90/d180 are
`nanmedian` over 3 of 4 ICs, and three quarters of the "360 d" survivals never happened.

**Fix:** distinguish `DIVERGED`, `TRUNCATED` and `COMPLETED`; never report a horizon an IC could
not reach; refuse to median a column over fewer than 3 valid ICs.

**Fix (data):** years 1990–1995 are converted and are in **no** split (train 1997–2008, val 2009,
test 1996). Run the gate over a multi-year window (e.g. `1994-01-01 .. 1996-12-31`) so all four ICs
can complete 360 days. Caveat: **1990 is JCM's own spin-up year** — truth global-mean T rises
218.0 → 234.4 K over the first ~110 days, while 1991/1996/1997 sit flat at 233.2–234.4 K [M]. Use
1991+ for any climate metric.

### 4.2 Required new columns

| change | why | measured basis |
|---|---|---|
| report `sr270` and `sr359` | already computed and discarded (§2.3); the failure is at day 200–210 | [M] |
| add `anomR270`, `anomR359` | same line of code | — |
| **add a temperature-drift column**: cos-φ-weighted global-mean T over all 63 levels, pred − truth, at d30/90/180/270/359 | the free run's signature is +49 K, invisible to every current column | [M] §2.3 |
| **lower `FAIL` from 500 to 200 m/s** | truth free-band max\|u\| at daily sampling never exceeds **142.1** (1990) or **133.9** (1996); 200 is 1.4x the annual maximum, 500 is 3.5x | [M] |
| add a soft `excursion_day`: first day max\|u\| (levels 9–58) > 150 m/s | records the onset even when the run does not hard-fail | — |
| **score levels 0–8 and the sponge (59–62) too**, as separate reported columns | the current window `LO,HI = 9,59` hid a 420 m/s band | [M] §2.3 |
| sigma_ratio is an unweighted `std()` over (lev, lat, lon) | not a bug, but state it; it is not area-weighted | [M] code read |
| `--start/--end` instead of the hardcoded `f"{year}-01-01"` | needed for §4.1 | — |

Re-run the fixed gate on `gate_ladder_big/lad*.ckpt` (6 checkpoints, symlinks already in place) and
publish a corrected version of the table in `PARADIS_JCM_RESULTS.md` §3 / §5 **before** the new
experiment reports anything.

---

## 5. Success criteria

Scored with the **fixed** `gate_paradis.py`, median over ≥ 4 ICs that all complete, on the test
year, boundary replayed (§7).

Reference values, all [C: `gate_paradis.py` docstring + `JCM_ANOMALY_FORMULATION_RESULTS.md` §1,
and reproduced in `/data/paradis_jcm/gate_big.log` [M]]:

| | sr30/90/180 | vortex corr | vortex std | rm30 | anomR90 |
|---|---|---|---|---|---|
| **climatology floor** | 0.811 | **0.727** | 1.00 | 11.4 | **0.000** |
| persistence (rm30) | — | — | — | 14.80 | — |
| baseline ep123 (12-yr, free) | 1.115 / 1.022 / 1.037 | **+0.238** | 2.63 | 15.09 | 1.619 |
| baseline ep24 | 0.957 / 0.727 / 0.672 | +0.390 | 1.32 | 26.25 | 1.930 |
| anomaly ep123 | 1.056 / 0.988 / 1.006 | +0.047 | 1.18 | 19.44 | 1.731 |

### 5.1 What `anomR` is and why it is the column that matters

`anomR(m) = std(pred − clim(t)) / std(truth − clim(t))` over the free band at day *m*, where
`clim` is the K=3 annual-harmonic field from `meta/clim_harmonics_1997_2008.npz`, evaluated at the
rollout's own fractional year. It is enabled by `--clim`.

It exists because **`sigma_ratio` and `vortex corr` both reward collapsing onto climatology.** A
model frozen exactly at `clim(t)` scores `sigma_ratio 0.811` and `vortex corr 0.727` — both look
like improvements over the learned runs — while its anomaly is identically zero, so `anomR = 0.000`
exactly. Conversely:

> `sigma_ratio ≈ 1.02–1.04` with `anomR ≈ 1.6–2.4` is a **cancellation, not correctness**:
> `var(field) = var(stationary) + var(anomaly) + cross`. The model has too little forced structure
> and too much internal variance and the two offset. [C: `JCM_ANOMALY_FORMULATION_RESULTS.md` §2]

So a result is only real if it moves vortex correlation **and** keeps anomR near 1. The pass bar:

| gate | threshold | rationale |
|---|---|---|
| **G1 vortex correlation** | **> 0.727** median over ICs at the full horizon | the floor nothing learned has beaten [C] |
| **G2 anomR90 and anomR180** | in **(0.7, 1.4)** | excludes both climatology collapse (0) and the current failure (1.6–2.4) |
| **G3 vortex std ratio** | in (0.8, 1.25) | truth = 1.00; baseline ep123 is 2.63 |
| **G4 temperature drift** | \|ΔT_global\| < **2 K** at d359 | truth sits in 233.2–234.4 K all year [M]; the free run reached +49 K |
| **G5 no divergence** | max\|u\| (lev 9–58) < 200 m/s for 360 d, all ICs | truth annual max 133.9–142.1 [M] |
| **G6 sr270, sr359** | in (0.8, 1.25) | the horizons the old gate discarded |
| G7 rm30 | < 11.4 would beat climatology; < 14.80 beats persistence | both currently failed [C] |

**G1 + G2 together are the experiment.** G4–G6 are the falsifiers the old gate lacked. Report G7
but do not select on it.

A clean negative — boundary correct, vortex still below 0.727 — is a publishable result: it says
the stratospheric operator, not the boundary, is the defect, and it kills the entire
boundary-forcing branch for ~70 GPU-h.

---

## 6. Implementation

### 6.1 Channel layout — read it, never assume it

253 output features, verified by reading the corpus coordinate [M]:

```
index   0.. 62  u_wind_h0      .. u_wind_h62
index  63..125  v_wind_h0      .. v_wind_h62
index 126..188  temperature_h0 .. temperature_h62
index 189..251  omega_h0       .. omega_h62
index 252       normalized_surface_pressure
```
> `$P -c "import xarray as xr; d=xr.open_zarr('/data/jcm_paradis_c1/1997'); print([str(s) for s in d.features.values])"`

and `[str(f) for f in dm.dataset.dyn_output_features]` reproduces this order exactly (verified: the
features coordinate written by `emulate_winds.py`, which writes `ds.dyn_output_features`, is
element-for-element equal to the corpus coordinate [M]).

**Always build indices by name**, as `trainer.py` already does:
```python
feats = [str(f) for f in datamodule.dataset.dyn_output_features]
idx_u = [feats.index(f"u_wind_h{k}") for k in range(63)]
```
Here `num_common_features == num_dyn_inputs_single == n_out == 253`, so the two layouts coincide
and a positional shortcut *would* work — which is exactly how this class of bug stays silent until
it does not. The network input is **516 channels** = 2 × 253 (two time inputs) + 2 × 2 forcings
(`sin/cos_year_progress`) + 6 constants [M: `bench_inference.py` prints `input (3, 516, 96, 192)`].

### 6.2 Where the boundary is imposed today

`model/jcm_physics.py :: JCMPhysics.forward` already does exactly the overwrite, gated on
`prescribe`, with `n = self.nudge_top = 9` [M]:

```python
if prescribe:
    u1 = torch.cat([true_p["u"][:, :n], u1[:, n:]], dim=1)
    v1 = torch.cat([true_p["v"][:, :n], v1[:, n:]], dim=1)
    T1 = torch.cat([true_p["T"][:, :n], T1[:, n:]], dim=1)
ps1 = true_p["ps"] if prescribe else out_p["ps"]
return {"u": u1, "v": v1, "T": T1, "omega": out_p["omega"], "ps": ps1}
```

**Defect to fix: `omega` is never prescribed.** Levels 0–8 of omega stay whatever the network
emitted. That is tolerable for the dynamics metrics (omega carries loss weight 0.02, §6.4) but
**not** for the transport coupling: the operator in `/data/jcm_transport_operator` reads
`u, v, omega, ps` at t and t+6h at all 63 levels [C: `jcm_transport_operator/RESULTS.md` §2.3], and
tropospheric tracers would be advected by an unconstrained vertical velocity. Add omega to the
prescribed set.

### 6.3 The change: replay instead of truth

The cleanest route keeps `jcm_physics.py` untouched and changes **what is handed in as `true_p`**.

* **Training, `forecast_steps: 1` (the current regime).** Input is truth at t−6h and t; target is
  truth at t+6h; the only use of `true_p` is the boundary overwrite of the *output*, which is then
  masked out of the loss. Replay would be a no-op. **So at K=1, train with `prescribe: true`
  exactly as now, plus the loss mask (§6.4), and introduce replay only at inference.** This is the
  minimum-change arm.
* **Training, K ≥ 2 (rollout).** Here the overwritten output becomes the next input, so the
  boundary does bite. At inference the model's stratosphere and the replay troposphere are
  *mutually inconsistent* — a state the K=1 training distribution never contains. **Arm B trains
  under that condition**: supply the replay boundary from step 2 onward. Implement as a third batch
  tensor:

```python
# data/era5_dataset.py __getitem__ : return (x, y, bnd)
# bnd[s] = frame from the REPLAY year at the same calendar date as y[s], same 253 features,
#          normalised with the SAME train-only stats (do not refit).
# trainer.py training_step:
out = self._apply_jcm_physics(out, x_s, true=y_s if step == 0 else bnd_s, prescribe=True)
```
Keep the collate/`num_workers` path identical; `bnd` doubles the read volume per sample, so re-check
§10.3 before launching.

* **Inference / gate.** Add `--replay-year` (or `--replay-offset N`). The rollout loop in
  `gate_paradis.py` and `emulate_winds.py` already reads `y = ds[j][1]` purely as the prescription
  source; swap in a second dataset handle opened on the replay year and index it by matching
  `(month, day, hour)`.

### 6.4 Masking levels 0–8 out of the loss — and renormalising

`utils/loss.py :: ParadisLoss.forward` is
`(loss_elementwise * feature_weights[1,F,1,1] * lat_weights[1,1,H,1]).mean()`. Masking is a
one-line edit to `self.feature_weights`: zero the 28 channels
`u/v/T_h0..h8` (27) + `normalized_surface_pressure` (1), and also `omega_h0..h8` (9) once omega is
prescribed — **37 channels of 253**.

Measured current weight vector [M], by instantiating `ParadisLoss` with this config's arguments:

```
every atmospheric channel : 0.20     (u, v, T — uniform across all 63 levels)
every omega channel       : 0.02
normalized_surface_pressure: 1.00
sum of feature_weights    : 40.06
```
> `$P -c "...ParadisLoss('reversed_huber', lat, torch.arange(63.), 253, 1, vw, feats, 1.0, True, 'uniform', None)..."`

Two things fall out, both worth recording:

1. **`pressure_weight_mode: uniform` is not an implemented mode.** `loss.py` tests only
   `== "log"` and otherwise takes the `linear_floor` branch `max(p/1000, 0.2)`. Because this
   config's `features.pressure_levels` are **level indices 0–62, not pressures**, every level
   evaluates to the 0.2 floor — so the intent (uniform) is met **by coincidence**. Do not "fix"
   the mode string without re-measuring; it would change every weight.
2. **`normalized_surface_pressure` carries 5× the weight of any wind/temperature channel and 50×
   any omega channel.** Masking it removes `1.00` of `40.06`.

Mask total: `3 vars × 9 lev × 0.20 + 9 × 0.02 + 1.00 = 6.58` of `40.06` = **16.4%** of the loss
weight.

**Renormalise the surviving weights by `40.06 / (40.06 − 6.58) = 1.197`.** The reduction is not
harmless: the final reduction is `.mean()` over all elements, so a 16.4% weight cut is a global
loss rescale, and with `gradient_clip_val: 1.0` and decoupled `weight_decay: 0.01` a global loss
scale is **not** neutral even under Adam. (The sibling campaign measured an 8.48× unintended loss
reweighting from a comparable change [C: memory `project_anomaly_arm_first_result`].) Print the
weight vector at startup and assert the sum is unchanged.

### 6.5 Keep levels 0–8 in the INPUT

The network must see the troposphere — that is the whole mechanism by which the boundary supplies
wave flux. Mask the **loss** and overwrite the **output**; never touch the input channels. Also
keep the architecture, analytic forcing (PK relaxation + sponge, which need no truth), and the
`g = (1 − e^{−a dt})/(a dt)` increment fix untouched: they are verified components [C:
`PARADIS_JCM_RESULTS.md` §2].

### 6.6 Config

Copy, never edit in place:
`config/config_jcm_free_big_fast.yaml` → `config/config_jcm_bforce.yaml`, changing only:

```yaml
jcm_physics:
  prescribe: true          # boundary levels 0-8 come from outside
  prescribe_omega: true    # NEW, see 6.2
  qbo: false               # stays OFF: the QBO nudge is truth-dependent and is not the subject
boundary:
  enabled: true
  mode: replay             # replay | truth (control) | climatology (control)
  offset_years: 1          # see section 7
loss_mask:
  zero_levels: [0,1,2,3,4,5,6,7,8]
  zero_surface: true
  renormalize: true        # 6.4
training:
  experiment_name: jcm_paradis_bforce
compute:
  num_devices: 6
  num_workers: 12
```

Leave `qbo: false`. Turning it on reintroduces a truth-dependent term and would confound the
result; the QBO is a separate option (D) in `JCM_ANOMALY_FORMULATION_RESULTS.md` §6.

---

## 7. Boundary replay source and year map

Source: the same corpus, `/data/jcm_paradis_c1/<year>` — zarr with dims
`(time, latitude, longitude, features)` = `(1460|1464, 96, 192, 253)`, chunked
`(1, 96, 192, 253)` (one frame per chunk), 21 GB/year, 415 GB total for 1990–2009 [M].

Frame-count audit [M]:

| year | frames | first | last |
|---|---|---|---|
| 1990 | 1460 | 1990-01-01T06 | 1991-01-01T00 |
| **1996** | **1464** (leap) | 1996-01-01T06 | 1997-01-01T00 |
| 1997 | 1460 | 1997-01-01T06 | 1998-01-01T00 |
| 2000 / 2004 / 2008 | 1464 (leap) | …T06 | …T00 |
| 2009 | 1460 | 2009-01-01T06 | 2010-01-01T00 |

**Never index the replay by frame number.** Leap years have 4 extra frames, so a frame-index map
drifts by one day after 29 February. Match on `(month, day, hour)`; when the simulated year is a
leap year and the replay year is not, repeat 28 February for the four 29 February frames and log
that you did. Do not interpolate (it would smooth exactly the transient eddies the design is for).

Year assignment — the replay year must never equal the simulated year, and must never be a year
whose *stratosphere* the model is being scored against:

| phase | simulated year(s) | replay source | note |
|---|---|---|---|
| train | 1997–2008 | `year − 1` within 1997–2008, wrapping 1997 → 2008 | all inside the train split; no leakage |
| val | 2009 | **1995** | 1995 is in no split |
| **test / gate** | **1996** | **1994** | both outside train/val; 1994 is in no split |
| long-horizon gate (§4.1) | 1994-01-01 .. 1996-12-31 | 1991–1993 offset by 3 | 1990 excluded (spin-up, §4.1) |
| control A (upper bound) | 1996 | **1996 itself** = current `prescribe: true` | should reproduce vortex +0.927 [C] |
| control B (lower bound) | 1996 | seasonal climatology from `meta/clim_harmonics_1997_2008.npz` | isolates the transient-flux contribution measured in §2.4 |

Controls A and B are not optional. Without them the replay number has no scale: A is what a perfect
boundary buys, B is what the already-refuted calendar climatology buys, and the experiment's whole
claim is that replay sits near A rather than near B.

Normalisation: reuse `meta/stats_train_1997_2008.npz` for the replay frames. **Do not refit.** A
pooled accumulator that had swallowed the test year has already bitten this project once [C:
`JCM_TRACER_TRANSPORT_SPEC.md` §5].

---

## 8. Splits

Matching the existing runs exactly, so results compose with `PARADIS_JCM_RESULTS.md` and the
transport operator:

```
train : 1997-01-01T06:00:00  ..  2008-12-31      (12 contiguous years, ~5 QBO cycles)
val   : 2009-01-01           ..  2009-12-31
test  : 1996
```

The `T06:00:00` start is **required, not cosmetic**: the 1997 zarr begins at 1997-01-01T06:00 and
ends at 1998-01-01T00:00, so the 1997-01-01T00:00 frame physically lives in the **1996 (test)**
zarr [M]. Starting at T00 pulls a test frame into training.

---

## 9. Training arms and the checkpoint ladder

| arm | K | boundary in training | cost | what it answers |
|---|---|---|---|---|
| **A** | 1 | truth (= replay is a no-op at K=1, §6.3) | ~70 GPU-h | Is the stratosphere learnable at all with a correct boundary? |
| **B** | curriculum 1→4→8 | replay from step 2 | ~100–200 GPU-h [E] | Does matching the train-time boundary to the inference condition close the exposure gap (measured at 27× [C: `JCM_ANOMALY_FORMULATION_RESULTS.md` §6B])? |

**Run A first, gate it, and only launch B if A clears G4–G6.** If A diverges or drifts, B's extra
cost buys nothing.

Checkpointing: `every_n_train_steps: 500`, **keep every checkpoint** (the 12-yr run kept 129, 38 GB
[M]), and score a ladder of ≥ 6 across training. Never take the last.

---

## 10. Known traps, each with its evidence

### 10.1 `val_loss` is decoupled from climate skill — three demonstrations
| run | best val | vortex corr (final ckpt) |
|---|---|---|
| baseline | 0.01231 | +0.238 |
| anomaly | **0.01222** (better) | **+0.047** (5× worse) |

[C: `JCM_ANOMALY_FORMULATION_RESULTS.md` §3]. Also: the baseline's val_loss fell monotonically to
its best at the final step while its ep99 survived 36 days. **Never select a checkpoint or a
formulation on val_loss in this project.** Score the gate.

### 10.2 Never take the last checkpoint
`/data/paradis_jcm/gate_big.log` [M], median over 4 ICs:
`lad24 vortex +0.390` > `lad123 (final) +0.238` > `lad99 +0.323 but surv 36 d`. Mid-training
collapse and recovery is the norm here, not the exception. Score a ladder; report the peak **and**
the final, never only one.

### 10.3 Training is dataloader-bound
`config_jcm_free_big_fast.yaml` records [C, config comments]: 4 workers deliver **4.84**
samples/s/rank, 12 workers **15.29**; the live 6-GPU run had been at **4.12**. `bench_io.py`'s
docstring [C] gives a compute-only ceiling of **7.75** samples/s at `batch_size 1`; the config
comment gives **11.3** at `batch_size 3`. The two ceilings are for different batch sizes — quote
the 11.3 one, since this config runs `batch_size: 3`. **Use `num_workers: 12`.**

Hard constraint: **8 workers/rank × 6 ranks once wedged the node** via NFS+DDP contention [C:
`bench_io.py` docstring, memory `feedback_training_nfs_ddp_contention`]. 12 × 6 = 72 processes on
104 cores was run successfully for 12 h on 2026-09-28 [M: `run_big.log` start/end]. If arm B adds a
third tensor per sample, re-measure with `bench_io.py` before launching, not after.

### 10.4 Gradient checkpointing costs 1.28× for nothing
MEASURED 0.1293 → 0.1010 s/pass while using 4.5 GB [C: config comment + `bench_config.py`]. Keep
`gradient_checkpointing: false`.

### 10.5 The cards are H200 140 GB, not H100 80 GB
```
0..7, NVIDIA H200, 143771 MiB each            [M: sudo chroot /run/nvidia/driver nvidia-smi]
```
`/data/CLAUDE.md` line 5 says *"8× NVIDIA H100 80 GB"* and is **wrong**. Any memory-based sizing
decision taken from that file should be revisited — there is ~1.8× more memory than assumed.

### 10.6 `torch.compile` is ~1.03×, NOT 3.4× — measured twice today
Re-ran `bench_inference.py` on an idle H200, twice, GPUs 4 and 5 [M]:

| mode | s/step | yr/hour |
|---|---|---|
| fp32 weights / bf16 autocast, bs=1 | **0.0306** | 80.7 |
| bf16 autocast, bs=1 | 0.0271 | 90.9 |
| bs=16 (per member) | 0.0263 | 93.8 |
| **torch.compile (`max-autotune-no-cudagraphs`), bs=1** | **0.0295 / 0.0296** | 83.5 |

The compiled figure **0.0296** reproduces the number handed to this spec exactly; the uncompiled
baseline of 0.101 s/step does **not** reproduce — today's idle-box value is 0.0306, so the speedup
is **1.03×**, not 3.4×. The 0.101 was most plausibly measured while the box was running a 6-GPU
training job. **Do not budget a 3.4× compile win.** `compile: false` is a defensible default; if
you turn it on, re-measure on the actual run conditions.

Deployment context for the same measurement: 0.0306 s per 6-h step = **80.7 model-years/hour**,
~9.8× the dycore's ~0.30 s/step [C: `bench_inference.py` reference line]. A 360-day rollout is
1440 steps ≈ **44 s of GPU compute**; gate wall time is dominated by truth reads, not the network.

### 10.7 The free-run "QBO" is RETRACTED
Do not build on it. Measured against truth: 0 sign reversals, 3.0% of time easterly, spectral peak
at 2 months, mean +13.5 m/s vs truth −7.3 [C: `PARADIS_JCM_RESULTS.md` §5.1]. `qbo: false` stays
off here and the QBO is explicitly **not** a success criterion for this experiment.

### 10.8 Rollout metrics are noisy; `sigma_ratio ≈ 1` is achievable by a frozen field
A frozen state scores sigma_ratio 0.94–1.24 [C: `PARADIS_JCM_RESULTS.md` §3], and byte-identical
config *and seed* gave 132 d vs 40 d survival in the small-head campaign [C: §7]. Median over ≥ 4
ICs; quote sigma_ratio only alongside vortex std and anomR.

### 10.9 `pgrep -f <pattern>` in wait loops matches the wait loop itself
Cost two days of wall time in this campaign [C: `PARADIS_JCM_RESULTS.md` §8]. Use PID files. Launch
long jobs under `nohup`.

### 10.10 Use checkpoint mtimes, not the progress bar, for step counts
`progress_bar: false` is set because the bar floods the log with `\r` frames [C: config comment].

---

## 11. Compute budget

Reference measurements from the two completed 12-yr runs, 25.0 M parameters (24,954,109 trainable
[M: `run_big.log`]):

| run | GPUs | eff. batch | steps | wall | s/step | GPU-h |
|---|---|---|---|---|---|---|
| `jcm_paradis_free_big` | 6 | 36 | 60 000 | 2026-09-28 19:08:47 → 09-29 07:06:17 = **11.96 h** | **0.717** | **71.8** |
| `jcm_paradis_anom` | 5 | 30 | 72 000 | 2026-09-30 06:08:28 → 19:38:17 = **13.50 h** | **0.675** | **67.5** |

[M: start/end timestamps in `/data/paradis_jcm/run_big.log` and `run_anom.log`.]

Budget for this experiment:

| item | cost |
|---|---|
| gate fixes + re-score the 6-ckpt ladder | ~2 GPU-h [E], mostly I/O |
| **arm A**, 60 k steps, 6 GPUs, eff. batch 36 | **~12 h wall, ~72 GPU-h** [E, from the matched run above] |
| gate arm A: 6 ckpts × 4 ICs × 360 d + 2 controls | ~4–8 GPU-h [E] |
| arm B (K-curriculum 1→8), conditional | ~100–200 GPU-h [E]; K-step BPTT is superlinear per step |
| **total to first verdict (A only)** | **~80 GPU-h, ~1.5 days wall** |

Disk: the run's checkpoints will be ~38 GB (129 × ~300 MB) [M, from the existing run]. Corpus reads
are from the shared `/data`; the second machine adds read load the main box also sees — check `/data`
is < 90% before launching.

---

## 12. Deliverables

1. **`gate_paradis.py` fixed** per §4, with the `surv` truncation bug closed, `FAIL = 200`, d270 and
   d359 reported, a temperature-drift column, and boundary/sponge bands reported separately.
2. **A corrected re-score** of `gate_ladder_big/lad{00,24,49,74,99,123}.ckpt` with the fixed gate,
   and an erratum paragraph for `PARADIS_JCM_RESULTS.md` §3/§5 stating which published "360 d"
   survivals were truncations.
3. **`config/config_jcm_bforce.yaml`** (a copy, not an edit) + a snapshot committed with the run.
4. **Loss mask + renormalisation** in `trainer.py`/`utils/loss.py`, with the startup assertion on
   the weight sum (§6.4).
5. **`prescribe_omega`** in `model/jcm_physics.py` (§6.2).
6. **Replay boundary loader**: calendar-matched, leap-safe, train-only normalisation, with a unit
   check that `bnd` frames are bit-equal to the replay year's zarr at the matched date.
7. **Arm A trained**, all checkpoints retained, ladder gated, with controls A (truth boundary) and
   B (climatology boundary).
8. **`/data/JCM_BOUNDARY_FORCED_RESULTS.md`** per §13.
9. **Emulated-wind archive** for the best checkpoint, written by `emulate_winds.py` in the
   `/data/jcm_paradis_c1` layout, so the transport operator can consume it unchanged — this is the
   integration the transport spec §11 asks for [C: `JCM_TRACER_TRANSPORT_SPEC.md`].

---

## 13. Results-doc template

`/data/JCM_BOUNDARY_FORCED_RESULTS.md`. **Every number must name the command that produced it.**
No exceptions, including numbers carried over from this spec.

```markdown
# JCM boundary-forced emulator — results
**Status YYYY-MM-DD.** Box: <hostname>, N x H200. Nothing running / <what is running>.
Every number below was produced by a command named next to it. [M] measured, [C] cited, [E] estimate.

## 1. Headline
| | result | command / log |
|---|---|---|

## 2. Gate corrections applied first
- what changed in gate_paradis.py, with the diff
- RE-SCORE of the published ladder; which "360 d" entries were truncations
- the erratum sent to PARADIS_JCM_RESULTS.md

## 3. Pre-flight (the rungs below a full run)
- feature-order assertion output
- loss weight vector before/after the mask; sum before, sum after, renorm factor
- replay alignment check: N matched dates, leap handling, bit-equality spot checks
- overfit one batch: loss start -> end
- tiny run: s/step, samples/s/rank, projected GPU-h   <- report BEFORE launching

## 4. Arm A
| ckpt | surv | sr30/90/180/270/359 | vx corr | vx std | anomR90/180 | dT_global | max|u| | rm30 |
(median over >=4 ICs, every IC completing; truncated ICs excluded and SAID SO)

## 5. Controls
| boundary | vx corr | anomR90 | dT | note |
| truth (upper bound) | | | | should reproduce +0.927 |
| replay (the experiment) | | | | |
| climatology (lower bound) | | | | |

## 6. Verdict against section 5 gates
G1..G7, each PASS/FAIL with the number.

## 7. What this does NOT show
Restate the external-data-stream limitation (spec section 3.1) verbatim.

## 8. Bugs found (all silent ones get a row)
| bug | effect | how it was caught |

## 9. Cost
wall, GPU-h, from timestamps; name the log.
```

---

## 14. Facts this spec could NOT verify

1. **`torch.compile` 3.4× / 0.101 s uncompiled baseline** — contradicted (§10.6). Re-measured
   twice today: 0.0306 uncompiled vs 0.0295–0.0296 compiled = **1.03×**. The compiled value matches;
   the 0.101 baseline does not reproduce on an idle box.
2. **Which checkpoint produced `/data/jcm_emulated_winds/1990`** — the zarr's `.zattrs` is `{}` and
   no launch script references it. All blow-up numbers in §2.3 are measured from that archive and
   are correct *for that archive*; the attribution to the baseline run's final checkpoint is
   inferred from file timing (15:50 on 2026-10-01, immediately before the anomaly run's
   `emul_anom.log` at 15:54 which used `epoch=123.ckpt`), not verified. **Re-run `emulate_winds.py`
   with the checkpoint recorded in the zarr attrs.**
3. **"a calendar climatology supplies almost NO upward wave flux"** — overstated. Measured: it
   supplies 61% of the DJF eddy heat flux at 137.92 hPa (the stationary part) and loses the 38%
   transient part plus all interannual variability (§2.4). The conclusion (climatology boundary is
   wave-deficient) holds; the magnitude does not.
4. **Seasonal-fraction figures 0.213 / 0.029 / 0.617** — reproduced exactly as *unweighted*
   grid-point sums; the latitude-weighted values are 0.273 / 0.034 / 0.554 (§2.4).
5. **"4.84 samples/s/rank against a 7.75 compute ceiling"** — the two figures come from different
   batch sizes: 7.75 is `bench_io.py`'s `batch_size 1` ceiling, 11.3 is the config's `batch_size 3`
   ceiling. Both are config/docstring citations, neither was re-measured here (§10.3).
6. **Arm B cost (~100–200 GPU-h)** — ESTIMATE. No K-step BPTT run exists for this model at this
   scale; the only anchors are the two K=1 runs in §11.
