---
# ---------------------------------------------------------------------------
# Identity. `id` must equal the directory name.
# ---------------------------------------------------------------------------
id: template-experiment
title: One line, human readable
kind: experiment            # experiment | tooling | plan
owner: shreyas              # shreyas | agent | <collaborator name>
issue: null                 # GitHub issue number, or null
created: 2026-01-01
updated: 2026-01-01
retrofitted: false          # true only for folders that predate this schema

# ---------------------------------------------------------------------------
# Committed binary deliverables. Run output never belongs in git; a binary
# that IS the deliverable (a trained model, a lookup table) may be tracked
# only if listed here — paths relative to this folder, a directory entry
# covering everything under it. Undeclared .npz/.nc/... anywhere in the
# folder fails validation. Figures need no declaration.
# ---------------------------------------------------------------------------
artifacts: []

# ---------------------------------------------------------------------------
# State machine. See CLAUDE.md for the gate required to enter each state.
# proposed -> scoping -> smoke -> ready -> queued -> running -> analysis
#          -> writeup -> done
# blocked_human and killed can be entered from anywhere.
# ---------------------------------------------------------------------------
state: proposed
next_action: >
  One sentence. What the next agent session should do on wake.
wake_after: null            # ISO timestamp, or null for "as soon as possible"

# ---------------------------------------------------------------------------
# Preregistration. Required before any production compute is spent.
# ---------------------------------------------------------------------------
hypothesis: >
  A falsifiable claim about the physical system or the model, not a task
  description. "Increasing X shifts Y by Z through mechanism M", not
  "run a sweep over X".

discriminating_diagnostic: >
  The single quantity whose value separates the hypothesis from its
  alternatives, and how it is computed. Name the variable, the region, the
  averaging period, and the reduction.

expected_effect:
  quantity: null            # e.g. change in TOA net flux
  magnitude: null           # signed number
  units: null
  basis: null               # literature value, scaling argument, prior run

noise_floor:
  quantity: null            # same quantity as expected_effect
  estimate: null            # 1 sigma of internal variability
  units: null
  source: null              # run id or manifest ref the estimate came from
  ensemble_needed: null     # members needed to resolve expected_effect at 2 sigma

falsifier: >
  The result that would kill this experiment. If the experiment cannot
  produce such a result, it is not an experiment.

# ---------------------------------------------------------------------------
# Prior work and collision avoidance. Both gates block entry to `smoke`.
# ---------------------------------------------------------------------------
prior_work:
  - ref: null               # DOI or arXiv id (resolve via Semantic Scholar)
    delta: null             # one sentence: what this experiment does that the ref did not
adjacency:
  checked: null             # date roster.yaml was consulted
  conflicts: []             # roster ids that overlap, empty list if none

# ---------------------------------------------------------------------------
# Compute. Budgets are enforced by the pre-submission hook, not by good faith.
# Units are GPU-hours: the scarce resource on every machine is GPUs.
# ---------------------------------------------------------------------------
compute:
  machine: null             # voltagepark (see CLAUDE.md Machines)
  budget_gpu_hours: null
  per_run_cap_gpu_hours: null
  spent_gpu_hours: 0

smoke:
  config: null              # path under configs/
  criteria: []              # e.g. ["completes", "no NaNs", "energy closes to 0.1 W/m2", "diagnostic nonzero, correct sign"]
  passed: false
  record: null              # manifest job_id of the passing smoke run

# ---------------------------------------------------------------------------
# Escalation. Required whenever state is blocked_human.
# A block without a default_action and a deadline is a stall, not a question.
# ---------------------------------------------------------------------------
blocked_on:
  question: null
  options: []
  recommendation: null
  default_action: null
  deadline: null            # date after which the daemon proceeds with default_action

# ---------------------------------------------------------------------------
# Outcome. Required to enter writeup.
# ---------------------------------------------------------------------------
verdict: null               # supported | refuted | inconclusive
verdict_basis: null         # one sentence tying the result to falsifier
kill_reason: null           # required if state is killed
---

# {title}

## Question

What is being asked and why it matters. Two or three sentences. This is the
part a human reads first.

## Design

How the diagnostic is computed, what the control and perturbation are, the
ensemble size and its justification against `noise_floor`, and what is held
fixed. Anything a reader would need to judge whether the design can answer
the question.

## Status

Generated from frontmatter. Do not hand edit. Run `tools/generate_readme.py`.

## Notes

Free text. Anything that does not fit the schema and is not a log entry.
