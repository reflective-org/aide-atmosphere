---
id: jcm-from-scratch-recipe
title: Train from scratch with rollout + dycore term + anchor
kind: experiment
owner: shreyas
issue: 5
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: Fix the >=5-yr loader, then the ladder (shapes, overfit, tiny run with s/step and a GPU-h projection) before
  any full run.
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

# Train from scratch with rollout + dycore term + anchor

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

Every result so far is a 100–200-step fine-tune of ep123.

- [ ] Fix the loader for ≥5-yr runs
- [ ] Ladder: shapes → overfit → tiny run with s/step and a GPU-h projection **before** any full run
- [ ] Understand the two-regime (bimodal bias) flip first? (p5 REPORT)

## Status

Generated from frontmatter. Do not hand edit.
