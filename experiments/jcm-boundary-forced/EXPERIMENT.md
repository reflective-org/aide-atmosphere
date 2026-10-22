---
id: jcm-boundary-forced
title: 'Boundary-forced arm: levels 9-62 with a replay-year troposphere (run or drop?)'
kind: experiment
owner: shreyas
issue: 8
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: Decide whether to run or drop now that fine-tuning stabilises the full column; if run, first update the spec's
  reference table to fixed-gate numbers.
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

# Boundary-forced arm: levels 9-62 with a replay-year troposphere (run or drop?)

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

A spec exists (`docs/history/JCM_BOUNDARY_FORCED_SPEC.md`, gates G1–G7) but nothing was ever run: no config, run or log.

- Its §5 reference table uses old-gate numbers that its own §4 warns against.
- Decide whether it is still relevant now that M5 stabilises the full-column model. If yes, update §5 to fixed-gate numbers first.

## Status

Generated from frontmatter. Do not hand edit.
