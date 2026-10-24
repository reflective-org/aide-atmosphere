---
id: jcm-qf-qbo-anchor-eval
title: 'Anchor + corrected prescribed QBO in training (QF2/QF3): finish evals, strict selection'
kind: experiment
owner: shreyas
issue: 1
created: '2026-10-22'
updated: '2026-10-22'
retrofitted: false
artifacts: []
state: proposed
next_action: Re-queue the 87 RAM-killed eval tasks (p6/status) behind the RAM guard after checking ps -u susanne; score QF3
  tests and QF2/QF3 steps 100-200; apply the strict selection rule against G300+QBO-at-inference.
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

# Anchor + corrected prescribed QBO in training (QF2/QF3): finish evals, strict selection

## Question

To be written at `scoping` (hypothesis + falsifier before any GPU time).

## Context and task list

## Status
- Training finished 10-21 19:10–19:13. Ckpts: `/mnt/nvme_stage2/jcm_dagger_p7/fix/ck/QF{2,3}_step{50..200}.pt`.
- 87 eval tasks were killed twice by the RAM guard (≥ 80 % host RAM from other users' jobs): 10-21 19:50 and 10-22 07:23–07:30. See `p6/status`, `ram.log`.

## Partial results (agent-computed medians from `p7/fix/diag/*.json`)
| run | result |
|---|---|
| QF2_s50 1996 | 4/4, bias 5.7, vortex .76, Tprof 1.5, zmV 7.4 |
| QF2_s50 1995 (3 ICs) | 3/3, bias 6.2, vortex .78, Tprof 2.8 |
| QF3_s50 selection | bias 5.4, Tprof 1.15 |

## To do
- [ ] Re-queue evals with the RAM guard (`ram_gate_requeue.sh`); check `ps -u susanne` first
- [ ] Score QF3 tests and steps 100–200 for both seeds
- [ ] Apply the strict rule (every selection run good) and compare against G300+QBO-at-inference 
- [ ] If QF wins, replicate across ≥2 more seeds

## Status

Generated from frontmatter. Do not hand edit.
