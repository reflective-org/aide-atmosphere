---
id: jcm-dagger-own-drift
title: DAgger round 2 on G300's own drifted states
kind: experiment
owner: shreyas
issue: 2
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: 'Write the preregistration (falsifier: out-of-sample fraction of the JCM correction stays ~0%) and harvest G300
  drifted states from training years only.'
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
verdict: null
verdict_basis: null
kill_reason: null
---

# DAgger round 2 on G300's own drifted states

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

From: the dycore term only corrects the drift it saw.

- [ ] Harvest G300's own drifted states from **training years only**
- [ ] Integrate the free JCM 2–10 d from them (z23 pipeline, each year's own p11a config)
- [ ] Retrain from epoch=123 on rounds 1+2
- [ ] Re-run the out-of-sample correction test on a held-out year/seed. Falsifier: fraction reproduced stays ~0 %

## Status

Generated from frontmatter. Do not hand edit.
