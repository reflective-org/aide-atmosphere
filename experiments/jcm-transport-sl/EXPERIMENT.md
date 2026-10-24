---
id: jcm-transport-sl
title: Hybrid semi-Lagrangian tracer transport emulator and multi-year validation
kind: experiment
owner: shreyas
issue: null
created: '2026-09-10'
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
verdict_basis: The learned trajectory correction beats the analytic operator on 468/468 single steps (+0.047 R2) but is better
  on 0/10 tracers from month 3 with 2.3x over-diffusion; the analytic operator's horizontal part is sound but it has no vertical
  remap (pulse_4 centroid 4.5 vs 217 hPa).
kill_reason: null
---

# Hybrid semi-Lagrangian tracer transport emulator and multi-year validation

## Question

Can JCM tracer transport be emulated for unseen tracer ICs ('nonlinear in flow, linear in tracer'), and does single-step skill carry to multi-year rollouts?

## Design

Analytic departure solve + cubic-Lagrange gather + limiter, plus a 1.46M learned correction; test 1996, held-out site 4; offline 5- and 20-yr rollouts driven by archive winds.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

## Status

Generated from frontmatter. Do not hand edit.
