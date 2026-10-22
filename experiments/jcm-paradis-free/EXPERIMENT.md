---
id: jcm-paradis-free
title: 'PARADIS 25M fully free (no prescription, no QBO nudge): 5-yr, 10-yr climate, 12-yr scale-up'
kind: experiment
owner: shreyas
issue: null
created: '2026-09-26'
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
verdict_basis: Runs 10 years bounded with no truth at any step, but the vortex correlation (+0.58; 12-yr scale-up +0.482 fixed
  gate) stays below the 0.727 climatology floor and there is no QBO (0 reversals, permanently westerly).
kill_reason: null
---

# PARADIS 25M fully free (no prescription, no QBO nudge): 5-yr, 10-yr climate, 12-yr scale-up

## Question

Can the network learn what prescription and the QBO nudge supplied, so that a deployable emulator needs no truth at any step?

## Design

config_jcm_free.yaml (1990-94) and config_jcm_free_big_fast.yaml (1997-2008, 60k steps x eff. batch 36 -> epoch=123, the base for all fine-tuning). Fixed gate, 1996 x 4 ICs; 10-yr free climate run; filter transfer test.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

![qbo_comparison.png](figures/qbo_comparison.png)

## Status

Generated from frontmatter. Do not hand edit.
