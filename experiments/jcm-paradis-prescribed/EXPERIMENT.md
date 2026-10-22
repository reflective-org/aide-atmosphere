---
id: jcm-paradis-prescribed
title: PARADIS 25M with levels 0-8 and the QBO prescribed from truth
kind: experiment
owner: shreyas
issue: null
created: '2026-09-23'
updated: '2026-10-22'
retrofitted: true
artifacts: []
state: killed
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
verdict: null
verdict_basis: null
kill_reason: 'Result void: trained and gated with the scrambled W_QBO nudge (permute bug, 882b1fb -> fixed 2026-10-21), which
  nudged the polar vortex toward truth, so the +0.927 vortex correlation was handed in, not learned.'
---

# PARADIS 25M with levels 0-8 and the QBO prescribed from truth

## Question

Does the 25M PARADIS port, given the ERA5-nudged troposphere (levels 0-8) and the QBO nudge, emulate the free stratosphere for a year?

## Design

config_jcm.yaml (no qbo key -> trainer default qbo=True); train 1990-94, val 1995, 15k steps; gate 1996 x 4 ICs.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

## Status

Generated from frontmatter. Do not hand edit.
