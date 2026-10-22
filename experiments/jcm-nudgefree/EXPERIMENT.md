---
id: jcm-nudgefree
title: Is the free-run failure ill-posed? Nudge-free corpus, emulator and filter sweep
kind: experiment
owner: shreyas
issue: null
created: '2026-10-05'
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
verdict_basis: 'The registered falsifier was met: the nudge-free emulator also runs away (ep59 died d51/d69, ep147 died d56;
  anomR90 grows 7->12->22 with training); and no free JCM corpus produces a QBO at any Hines amplitude.'
kill_reason: null
---

# Is the free-run failure ill-posed? Nudge-free corpus, emulator and filter sweep

## Question

Is the free-running failure caused by levels 0-8 being 63%-per-step ERA5 (not a function of state)? If so, an emulator of a nudge-free JCM should not run away.

## Design

Registered in advance in history/JCM_NUDGEFREE_HYPOTHESIS.md. Nudge-free corpus /data/jcm_nf_c1 (train 2009-18, val 2019, test 2020); Hines GWD sweep; filter sweep on ep147.

## Results

See [`RESULTS.md`](RESULTS.md). This folder predates the preregistration schema (`retrofitted: true`): it was recorded from the cluster logs and write-ups, not preregistered. Provenance: every number cites a file on the Voltage Park box; see [`INVENTORY`](../../plans/jcm-emulator-survey/INVENTORY.md). Withdrawn claims: [`RETRACTIONS`](../../plans/jcm-emulator-survey/RETRACTIONS.md).

## Status

Generated from frontmatter. Do not hand edit.
