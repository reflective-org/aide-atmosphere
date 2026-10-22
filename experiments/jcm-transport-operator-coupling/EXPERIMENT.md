---
id: jcm-transport-operator-coupling
title: Couple the learned 3-D transport operator to emulated winds
kind: experiment
owner: ali
issue: 7
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: 'Coordinate with Ali (roster: jcm-transport-operator) before touching his code; reconcile the pulse_4 truth-height
  metric (216.5 vs 397 hPa).'
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

# Couple the learned 3-D transport operator to emulated winds

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

ali's operator (`/data/jcm_transport_operator`, r3 step 7545) passes 10-yr age-of-air (−1.6 %) at the cost of 2.9× pulse variance. It has only been driven by archive winds.

`/data/jcm_emulated_winds` exists for this. **Owner: ali.** Confirm before touching that code.

Also unresolved: the pulse_4 truth height at yr 1 is 216.5 hPa (ours) vs 397 hPa (ali's). The metric definitions probably differ.

## Status

Generated from frontmatter. Do not hand edit.
