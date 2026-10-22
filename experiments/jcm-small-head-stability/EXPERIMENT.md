---
id: jcm-small-head-stability
title: 'Small state head + analytic forcing: can it free-run JCM stably?'
kind: experiment
owner: shreyas
issue: null
created: '2026-09-17'
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
verdict_basis: No small-head configuration beat the 0.727 climatology vortex floor; the blow-ups are grid-scale numerical
  instability that a filter delays (25 d -> ~200 d median) but does not cure, and the only full-year variants (anchor bound)
  diverge behind their clamp.
kill_reason: null
---

# Small state head + analytic forcing: can it free-run JCM stably?

## Question

Can a ~1M-parameter learned state head plus JCM's analytic forcing terms (PK relaxation, sponge, nudge, QBO) run JCM's free band stably for a year and beat climatology?

## Design

Test year 1996, 360-d free runs, medians over ICs and seeds; floors from a frozen climatology. Arms: filter in the loop, spectral zeta/delta, thermal-wind anchor, anchor bound, dycore distillation, real-dycore reference.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

## Status

Generated from frontmatter. Do not hand edit.
