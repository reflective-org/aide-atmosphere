---
id: jcm-transport-vertical-remap
title: Vertical remap for the analytic SL transport operator
kind: experiment
owner: shreyas
issue: 6
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: Implement the vertical remap and rerun the m-series 5-yr validation; regenerate the stale tables with m2_report.py.
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

# Vertical remap for the analytic SL transport operator

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

From: pulse_4 is frozen at 4.5 hPa vs truth's 217. The horizontal operator is accurate to ~17 %/yr.

- [ ] Implement a vertical remap
- [ ] Rerun the m-series 5-yr validation
- [ ] Regenerate the stale tables with `m2_report.py`

## Status

Generated from frontmatter. Do not hand edit.
