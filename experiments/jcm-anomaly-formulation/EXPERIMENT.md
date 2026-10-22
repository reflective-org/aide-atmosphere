---
id: jcm-anomaly-formulation
title: 'Anomaly formulation: prescribed seasonal climatology + learned anomaly'
kind: experiment
owner: shreyas
issue: null
created: '2026-09-30'
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
verdict: refuted
verdict_basis: At ep123 the anomaly model is damped (fixed gate vortex -0.117, std 0.46 vs base +0.482, 0.83) and its free
  rollout diverges at step 31; better val loss did not mean better climate. The ep24 ranking flip under the fixed gate is
  unexplored.
kill_reason: null
---

# Anomaly formulation: prescribed seasonal climatology + learned anomaly

## Question

Does predicting the anomaly about a prescribed day-of-year climatology (the formulation that rescued moisture in the CESM line) improve the free JCM emulator?

## Design

config_jcm_anom.yaml, 72k steps x eff. batch 30, matched against the 12-yr baseline; old gate and fixed-gate rescore.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

## Status

Generated from frontmatter. Do not hand edit.
