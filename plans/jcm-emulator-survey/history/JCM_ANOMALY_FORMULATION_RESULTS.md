# Anomaly formulation on JCM — result and where to go next

**2026-10-01.** Two matched 12-year runs, gated identically. Every number here came from a command
run in this session; the gate logs are `/data/paradis_jcm/gate_{anom,base}_g*.log`.

---

## 1. Verdict

**The hypothesis is refuted. Prescribing the seasonal climatology did not fix the vortex.**

It was predicted to deliver the 0.727 climatology floor "by construction". It delivered 0.047.

| | surv | sr30 | sr90 | sr180 | **vx corr** | vx std | rm30 | **anomR90** |
|---|---|---|---|---|---|---|---|---|
| **CLIMATOLOGY FLOOR** | — | 0.811 | 0.811 | 0.811 | **0.727** | 1.00 | 11.4 | **0.000** |
| baseline ep24 | 360 | 0.957 | 0.727 | 0.672 | **+0.390** | 1.32 | 26.25 | 1.930 |
| baseline ep123 (final) | 360 | 1.115 | 1.022 | 1.037 | +0.238 | 2.63 | 15.09 | 1.619 |
| anomaly ep24 | 360 | 0.812 | 0.728 | 0.695 | +0.320 | 1.05 | 20.41 | 1.658 |
| anomaly ep123 (final) | 360 | 1.056 | 0.988 | 1.006 | **+0.047** | 1.18 | 19.44 | 1.731 |

6 checkpoints x 4 ICs x 360-day fully-free rollouts, test year 1996, both runs scored with the
same tool. Matched at 2.16M presentations (60k x 36 vs 72k x 30). Cost ~155 GPU-h total.

### What improved
- **Stability.** The baseline's mid-training collapses are gone: ep99 survives **173 d vs 36 d**,
  ep74 **360 d vs 144 d**.
- **Vortex amplitude.** vx std **2.63 -> 1.18** at the final checkpoint (target 1.00). The vortex
  variability is now the right size; its phase is still wrong.
- sigma_ratio stays near 1 (1.006 at d180), marginally closer to 1 than baseline.

### What got worse
- **Vortex correlation** 0.238 -> 0.047 (final), 0.390 -> 0.320 (best checkpoint).
- **rm30** 15.09 -> 19.44. Both runs remain worse than climatology (11.4) AND persistence (14.80).

---

## 2. Why it failed — the opposite of the risk that was guarded against

The `anomR` column was added to catch the anomaly **collapsing to zero** (a model that "improves"
the vortex by silently becoming climatology). That did not happen. The opposite did:

**anomR = 1.6–2.4 in every run and every checkpoint.** The learned anomaly carries roughly twice
the variance of truth's.

So the prescribed climatology is correct, and then a large uncorrelated anomaly is added on top of
it, which dilutes rather than preserves the correlation. Prescribing the seasonal cycle only buys
the floor if the anomaly is small or skillful. Ours is neither.

### This also retracts the campaign's one apparent win

`sigma_ratio ~= 1.02–1.04` was reported as the best variance result this project has produced.
Read together with anomR ~= 1.6–2.4, that near-unity is a **cancellation**, not correctness:

    spatial var(field) = var(climatological/stationary part) + var(anomaly) + cross

The model has **too much internal variance and too little forced/stationary structure**, and the
two errors offset in the total. Without the anomR column this would have continued to be banked as
a success. **sigma_ratio alone cannot distinguish a correct field from two compensating errors.**

---

## 3. The third independent demonstration that val_loss is useless here

| run | final train | best val | vx corr (final ckpt) |
|---|---|---|---|
| baseline | 0.01234 | 0.01231 | +0.238 |
| anomaly | **0.01217** | **0.01222** | **+0.047** |

The anomaly run has the **better** val_loss and **5x worse** vortex correlation. Earlier in the
same campaign: the baseline's val_loss fell monotonically to its best at the final step while
ep99 survived only 36 days. **Never select a checkpoint or a formulation on val_loss in this
project.** Score the climate gate.

---

## 4. Also established this week

- **The free-run "QBO" is retracted** (`PARADIS_JCM_RESULTS.md` §5.1). Measured against truth:
  **0 sign reversals, 3.0% of time easterly, spectral peak at 2 months**, mean +13.5 m/s vs truth
  -7.3. It locks permanently westerly after spin-up. The original claim misread an
  annually-smoothed *range* as an oscillation. Figure: `/data/paradis_jcm/qbo_comparison.png`.
- The failure signature is a **missing easterly phase**, not a weak oscillation — nothing ever
  drives the tropics westward. Consistent with the measured loss of the westward inertia-gravity
  branch (<1% of truth retained) in the sibling CESM diagnostic.
- **Prescribed climatology cannot ever fix the QBO**: it is ~28-month and not phase-locked to the
  calendar, so it averages out of any day-of-year climatology.

---

## 5. Where this leaves the dynamics

Stable, with realistic total variance that is realistic for the wrong reason, a vortex whose phase
is wrong, and no QBO. Two formulations and ~155 GPU-h have not moved vortex correlation off
0.2–0.4 against a 0.727 floor. **Scale did not move it. Prescribing the seasonal cycle did not
move it.**

The new information is the decomposition: the problem is **excess uncorrelated internal variance**,
not missing variance and not a missing seasonal cycle. That is a different target from the one the
last two runs aimed at.

---

## 6. Options

Ranked by information per GPU-hour.

### A. Spectral filter in the rollout loop — *cheapest, biggest known lever, do this first*
The single largest effect ever measured in this campaign: a filter in the loop took free-run
survival from **day 25 to day 369**, and the finding was that filter **width** is the lever while
its form does not matter (ECHAM hyperdiffusion, a tuned exponential filter, and JCM's own per-level
profile were statistically indistinguishable). **It was never ported to PARADIS** — the only SHT in
the tree is inside `utils/amse_loss.py`, which is not the active loss.

It directly targets the diagnosed failure: excess small-scale variance is exactly what a spectral
filter removes, and anomR ~2 says that is what we have.

**Testable at inference with no retraining**: truncate the state each step during the free rollout
and re-gate an existing checkpoint. Sweep the width. **Cost: a few GPU-hours.** If the width sweep
moves anomR toward 1 and vortex upward, it becomes a retrain-time change.

### B. Rollout / curriculum training — *the most-evidenced untried training change*
Both runs are pure single-step (`forecast_steps: 1`, curriculum disabled). The exposure gap is
**measured at 27x**: operator error is flat at ~2–3.6 m/s at truth states all year but 27x worse at
the states the model actually visits. For JCM specifically, single-step training was measured to
*degrade* free-run stability (35 -> 12 d), recovered by rollout.
**Cost: one training run, ~70–150 GPU-h** (K-step BPTT is more expensive per step).
Caveat: in the CESM line *more* rollout training degraded multi-year behaviour — but that was fs=24
optimising a 6-day horizon, a different regime.

### C. Dycore distillation at PARADIS scale — *the one learned intervention that has ever worked*
In the small-head campaign it gave **0.67 vs 6.33 divergence events per 1000 steps** against a
matched archive-only control, identical across every seed, and took a typical checkpoint from
diverging at day 100 to bounded at day 200. It attacks the exposure gap at its root by training on
the states the model visits. Never tried at PARADIS scale. The dycore costs 3.1 h per 20 years, so
generating DAgger data is affordable.
**Cost: dycore time + one training run.**

### D. Give the model the QBO momentum it cannot see — *targets the QBO-dead failure directly*
The emulator learned the QBO as phenomenology because in the training data the **nudging term
supplies the momentum invisibly**. Expose that tendency as a target and the model can learn the
momentum forcing as a function of state, which a free run can then apply itself.
**CHECKED 2026-10-01 — it is exactly reconstructible, no JCM re-run needed.** The nudging is
analytically simple (`jcm_strat/qbo_nudging.py`): only the ZONAL MEAN of u is relaxed, with the
same tendency at every longitude of a latitude row,

    dU/dt|nudge = -(ubar_model - ubar_ERA5) / tau,    tau = 1 day

Both terms are archived. `ubar_model` is the training corpus itself; `ubar_ERA5` is in
`cache/era5_ref/era5_zm_monthly_{year}.nc` (`uzm`, 25 levels x 721 lat x 12 months), and **every
year 1989-2020 is present** — covering train 1997-2008, val 2009 and test 1996. The monthly values
are interpolated mean-preservingly to continuous time by `mean_preserving_nodes()` in that same
file, so the exact tendency the model saw can be rebuilt offline.

This makes D **cheap**: a derived-channel computation over the existing corpus, not a new
simulation. It is the only option on this list that addresses *why* there is no easterly phase,
and the momentum it supplies is precisely what the free-running model has no source for.

### E. Change the metric to match the actual goal — *cheap, and arguably overdue*
Everything has been scored on zonal wind: sigma_ratio, vortex correlation, u RMSE. **The stated
goal is tracer transport with a temperature feedback.** What governs transport is the **residual
circulation** (w*, v*) and age of air — and **neither has ever been measured on this emulator.**
It is entirely possible that the dynamics are already adequate for transport while failing the
vortex metric, or the reverse. Right now we do not know.
**Cost: a diagnostic script plus one existing rollout. No training.**

### F. Start tracer transport on the second machine — *parallel, unblocked*
`/data/JCM_TRACER_TRANSPORT_SPEC.md` is written and self-contained. The transport operator trains
against **truth winds**, so it is completely decoupled from fixing the dynamics and needs nothing
from this box. The second 8x H200 machine is idle for this purpose.

### Recommendation
**A + E now** (both cheap, neither needs training, and together they test the diagnosed cause and
tell us whether we have been optimising the right quantity at all), **F in parallel on the second
machine**, then **B or C** as the next training run depending on what A shows. **D** is the only
route to a real QBO, and the blocking question has now been answered: the tendency is exactly
reconstructible from archived data, so it costs a derived-channel pass rather than a simulation.

What is *not* worth doing: another scale increase, or another formulation change aimed at the
seasonal cycle. Both have now been tested and neither moved the metric.
