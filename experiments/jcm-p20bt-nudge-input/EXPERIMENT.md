---
id: jcm-p20bt-nudge-input
title: "Option D: p20bt emulator with the ERA5 nudging target as forcing input (toward 1 % of the 6-h change)"
kind: experiment
owner: shreyas
issue: null
created: 2026-10-24
updated: 2026-10-24
retrofitted: false

artifacts: []

state: running
next_action: >
  When jcm_p20bt_D finishes, score: (1) one-step error as % of the 6-h change per variable/band (onestep_rel_err_p20bt.py
  --cfg D --qbo); (2) 2-yr D free runs, stratospheric scorecard (freerun_zm_p20bt.py + strat_metrics_p20bt.py) vs the step-0
  model and with/without --vfilt-m0; (3) pick the checkpoint on the multi-year stratospheric metrics, not the last.
wake_after: null

hypothesis: >
  With the ERA5 nudging target and the dated QBO target supplied as forcing, the p20bt step is a deterministic function of
  (state, forcing). An emulator given them, with the truth's nudges applied structurally, cuts the one-step error well below
  the free model's 15-40 % of the 6-h change, most of all at h0-h15. Its multi-year free runs track the truth's forced
  stratosphere (QBO phase, vortex) instead of drifting.

discriminating_diagnostic: >
  (a) rms one-step error / rms truth 6-h change, cos-weighted, by variable and level band, 200 states of 1996 (target 1 %).
  (b) 2-yr free runs (state never reset, ERA5 forcing only): equatorial u at 10/30 hPa and 60N/60S u at 10 hPa, day-by-day
  correlation, std ratio and bias against truth; polar-cap T bias; h16-h73 time-mean zonal-mean u/T bias and its drift.

expected_effect:
  quantity: one-step error at h0-h15, % of 6-h change
  magnitude: -15
  units: percentage points vs the free base (25-30 % -> 10-15 % or lower)
  basis: ERA5 6-h-stale substitution moves the step by 64 % at h0-15 (dycgen probe_era5_lag); the base cannot see ERA5

noise_floor:
  quantity: one-step error, % of 6-h change
  estimate: 0.1
  units: "% (generator reproduces the archive to 0.06-0.07 % u/v/T, 0.5 % omega)"
  source: /data/paradis_jcm/dycgen/VALIDATION.md
  ensemble_needed: 200 states; free runs 4 ICs x 2 years (one seed)

falsifier: >
  The D one-step error at h0-h15 is not clearly below the base's (25-30 %); or D free runs drift from the truth's forced
  stratosphere no better than the step-0 model (base network + structural nudges, no learning).

prior_work:
  - ref: experiments/jcm-p20bt-free-base
    delta: same network warm-started; adds ERA5-target + year inputs and the structural ERA5/QBO nudges
adjacency:
  checked: 2026-10-24
  conflicts: []             # reads susanne's p20bt run and ERA5 cache read-only; GPU 7 left to her running job

compute:
  machine: voltagepark
  budget_gpu_hours: 150
  per_run_cap_gpu_hours: 40
  spent_gpu_hours: 4

smoke:
  config: configs/config_jcm_p20bt_D_smoke.yaml
  criteria: ["warm start equals base network (float64)", "overfit one batch to the base's absolute floor (<1e-3 MSE)",
             "300-step smoke completes, loss decreases, s/step measured"]
  passed: true
  record: jcm_p20bt_D_smoke

blocked_on:
  question: null
  options: []
  recommendation: null
  default_action: null
  deadline: null

verdict: null
verdict_basis: null
kill_reason: null
---

# Option D: emulator with the ERA5 nudging target as forcing input

See `decisions/2026-10-24-reframe-option-D.md` for why this design was chosen.

## Ladder record

1. **Signal.**
   - Given the state alone, the step depends on ERA5: a 6-h-stale target moves it 64 / 18 / 5-10 % (h0-15 / h16-73 / above).
   - Given state plus target, the generator is deterministic (bitwise).
2. **Falsifiers.**
   - The ERA5-target zarr is bitwise equal to the model's NudgingTarget on 6 frames. Flipping its levels or latitude makes it
     disagree 2.6-47×.
   - The physics nudge closed form matches a direct Euler integration to 1e-12. A level-flipped target fails, as it should.
3. **Shapes and alignment.** The ERA5 input channels equal the zarr at the newest input time (4e-6, float rounding). The year
   channel is exact. 821 input channels. The gate's FastIO is bit-identical to the dataset, including the new channels.
4. **Overfit.** Warm-started network = base network to 5e-13 in float64 (the 1e-2 seen on the GPU is TF32). One-batch MSE
   0.019 → 7.4e-4 at lr 3e-4, the same absolute floor as the base's passing overfit (7.8e-4). lr 1e-3 is too large for a
   warm start: the loss jumps to 25 at step 1.
5. **Tiny run.** 300 steps, 1.24 s/step on 6 GPUs, loss 0.0153 → 0.0142.

## Run

- **Config and launch.** `jcm_p20bt_D`, configs/config_jcm_p20bt_D.yaml, seed 42, 15k steps, lr 1e-4 WSD (warmup 500).
- **GPUs.** GPUs 2-6, effective batch 30. susanne started a Held-Suarez job holding 108 GB on GPU 7, so the 6-GPU launch was
  stopped during setup and relaunched on 5 GPUs.
- **Speed.** Measured 0.93 s/step, so about 3.9 h × 5 GPUs ≈ 19 GPU-h.
- **Output.** /data/paradis_logs/lightning_logs/jcm_p20bt_D.
