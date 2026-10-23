---
id: jcm-p20bt-free-base
title: PARADIS 25M free base on the p20bt (L95, Jucker + Hines) corpus, and why it dies
kind: experiment
owner: shreyas
issue: null
created: 2026-10-22
updated: 2026-10-23
retrofitted: false

artifacts: []

state: analysis
next_action: >
  Decide how to carry the m=0 damping into training (a fixed m=0 vector-SH filter on the output, applied in the train step as
  well as at inference), then a tiny run and a matched gate against this base with --vfilt-m0 36,4. Also score 1995 IC 1092
  and the remaining ladder checkpoints with the filter. Ask before any loss change or run over 10 minutes.
wake_after: null

hypothesis: >
  The 25M PARADIS recipe that free-runs the p11a corpus (with fine-tuning) learns the p20bt corpus well enough to free-run
  for a year from the base checkpoint alone. p20bt nudges only h0-h15 (p >= 400 hPa) toward ERA5, so h16-h94 must be
  emulated with no restoring force except the analytic forcing carried in the model.

discriminating_diagnostic: >
  gate_p20bt.py free rollouts: 4 ICs (0/364/728/1092), 360 d, test year 1996 (and 1995), free band h16-h93. Scored on survival
  (|u| < 300 m/s), sigma-ratio, vortex correlation at h54 (9.73 hPa) and 60.6N, rm30, Tprof and bias against the train-years
  climatology, all against the climatology floors in /data/paradis_jcm/meta/p20bt_floors.json.

expected_effect:
  quantity: median 360-d vortex correlation (1996, 4 ICs)
  magnitude: 0.932
  units: correlation (climatology floor; the model must beat it to add skill)
  basis: p20bt climatology floor, meta/p20bt_floors.json

noise_floor:
  quantity: median 360-d vortex correlation
  estimate: 0.05
  units: correlation; spread across the 4 ICs of one checkpoint (0.835-0.885 for the last ckpt with --vfilt-m0)
  source: logs_p20bt/gate_vf/g96vf_last_*.json
  ensemble_needed: 4 ICs x 2 test years; one training seed only (seed spread unmeasured)

falsifier: >
  No checkpoint survives 360 d on all 4 ICs without "bad" (died / Tprof > 4 K / bias > 15 m/s). That happened: the base
  hypothesis is refuted as stated. The mechanism and an inference-time constraint that fixes survival are in RESULTS.md.

prior_work:
  - ref: experiments/jcm-paradis-free
    delta: same architecture and recipe, new corpus (L95, nudging only below 400 hPa, top-level anomaly-only sponge, Hines GWD)
  - ref: experiments/jcm-finetune-stabilization
    delta: no fine-tuning here; this is the base the fine-tuning recipe would start from
adjacency:
  checked: 2026-10-22
  conflicts: []            # consumes susanne's p20bt run read-only; no overlapping work

compute:
  machine: voltagepark
  budget_gpu_hours: 120
  per_run_cap_gpu_hours: 90
  spent_gpu_hours: 84

smoke:
  config: configs/config_jcm_p20bt_smoke.yaml
  criteria: ["completes 300 steps", "loss decreases", "GPU utilisation and s/step measured", "overfit one batch to <1e-3 MSE"]
  passed: true
  record: jcm_p20bt_smoke

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

# PARADIS 25M free base on the p20bt corpus

## Question

Can the existing 25M PARADIS recipe free-run the new p20bt jcm-strat corpus (L95, Jucker forcing, Hines GWD, ERA5 nudging only
below 400 hPa) for a year from the base checkpoint? If not, where does the free run fail first, and why?

## Design

- Corpus /mnt/nvme_stage2/jcm_p20bt_c1: 1990-2009 continuous chain; train 1997-2008; stats on train years only; test 1995/1996.
- Analytic forcing in the loop (model/jcm_physics_p20bt.py, verified against JAX to 1.5e-5 m/s per 6-h step), free mode
  (prescribe=false, qbo=false; the QBO nudge needs the calendar year plumbed through the batch first).
- Config configs/config_jcm_p20bt_free.yaml: seed 42, 60k steps, 6x H200, batch 3 x accum 2, lr 2e-4 WSD, uniform pressure
  weighting. Output /data/paradis_logs/lightning_logs/jcm_p20bt_free/checkpoints (every epoch kept).
- Ladder gate over checkpoints 09, 25, 41, 57, 73, 89, 105, 121, last. Death diagnostics in analysis/.

## Ladder record

1. Signal: floors computed first (climatology vortex corr 0.932 / 0.880; persistence and climatology baselines in the floors file).
2. Falsifier: gate validated on pseudo-models (truth / clim / persistence) through its own data path.
3. Shapes: tyear alignment and physics checked against JAX; Gregorian year progress.
4. Overfit one batch: MSE 7.8e-4 (physics on, AMP off).
5. Tiny run: 300 steps, 0.83-0.93 s/step on 6 GPUs, projected 84 GPU-h; actual 13.99 h x 6 = 84 GPU-h.
