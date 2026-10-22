---
id: jcm-finetune-stabilization
title: 'Stabilizing the 25M emulator by fine-tuning: DAgger, rollout loss, masked dycore term, T anchor, prescribed QBO'
kind: experiment
owner: shreyas
issue: null
created: '2026-10-07'
updated: '2026-10-22'
retrofitted: true
artifacts: []
state: done
next_action: null
wake_after: null
hypothesis: null
discriminating_diagnostic: null
expected_effect:
  quantity: null
  magnitude: null
  units: null
  basis: null
noise_floor:
  quantity: null
  estimate: null
  units: null
  source: null
  ensemble_needed: null
falsifier: null
prior_work: []
adjacency:
  checked: null
  conflicts: []
compute:
  machine: null
  budget_gpu_hours: null
  per_run_cap_gpu_hours: null
  spent_gpu_hours: 0
smoke:
  config: null
  criteria: []
  passed: false
  record: null
blocked_on:
  question: null
  options: []
  recommendation: null
  default_action: null
  deadline: null
verdict: supported
verdict_basis: K=4 rollout + masked 48-h dycore term + global-mean-T anchor (G300_step200) gives 8/8 test runs with 0 bad
  and bias 6.7/7.0 m/s vs base 2/8 surviving; the anchor cuts bad runs 6x (4/80 vs 24/80, p<1e-4). But the vortex stays below
  the climatology floor and the learned correction does NOT generalise to the model's own drift (~0%).
kill_reason: null
---

# Stabilizing the 25M emulator by fine-tuning: DAgger, rollout loss, masked dycore term, T anchor, prescribed QBO

## Question

Starting from epoch=123, which fine-tuning ingredients make the free emulator stable and balanced for 1-2 years without losing variance?

## Design

Fixed harness (eff. batch 36, archive 1997-2008, warmup). Selection set 1998x4 + 2009 IC0, test 1996/1995 x 4 ICs; bad = died OR Tprof>4K OR bias>15 m/s; strict selection rule; consolidated ranking over 671 runs.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

![f_correction_G300own.png](figures/f_correction_G300own.png)

![f_correction_models.png](figures/f_correction_models.png)

![forecast_skill.png](figures/forecast_skill.png)

![qbo_vortex_tracking.png](figures/qbo_vortex_tracking.png)

![zm_u_error_1996.png](figures/zm_u_error_1996.png)

## Status

Generated from frontmatter. Do not hand edit.
