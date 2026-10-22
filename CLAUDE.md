# aide-atmosphere

The durable record of ML atmosphere-emulator experiments run on the Voltage Park box. The first line of work is emulators
of JCM (`jcm-strat`). The structure is adapted from `climate-analytics-lab/jcm-experiments`.

## The one rule

**State lives in files. Issues are for talking to people.**

`experiments/<id>/EXPERIMENT.md` frontmatter is the single source of truth for what an experiment is and where it stands.
GitHub issues carry the conversation, the questions and the open tasks.
Sync is one-directional: write state to the file, then reflect a summary into the issue. Never read state back out of an issue.

## Layout

```
experiments/<id>/          # one folder per experiment. Must validate.
  EXPERIMENT.md            # frontmatter state machine + preregistration
  LOG.md                   # append only, one entry per session
  RESULTS.md               # numbers, each citing the file it came from
  configs/                 # immutable after submission
  runs/manifest.jsonl      # one JSON object per submitted job
  analysis/                # scripts only
  figures/                 # committed, small
  decisions/               # one file per escalated question and its answer
  draft.md
plans/                     # planning documents and cross-experiment surveys
  jcm-emulator-survey/     # status of every JCM emulator line + RETRACTIONS + INVENTORY
templates/                 # EXPERIMENT.md template
tools/                     # validator, README generator
roster.yaml                # who else is working on adjacent things
```

Never commit run output. Logs, checkpoints, `.npz`, `.nc` and zarr stay on the cluster and are referenced by path
(see `plans/jcm-emulator-survey/INVENTORY.md`). Figures are the exception.

## States and their gates

`tools/validate_experiments.py` enforces these, and it runs in CI. The gate table is the same as in jcm-experiments:
`proposed` → `scoping` (hypothesis) → `smoke` (diagnostic, falsifier, expected effect, prior work, adjacency, compute, smoke config)
→ `ready` (smoke passed) → `queued` (noise floor) → `running` (manifest entry) → `analysis` → `writeup` (verdict) → `done` (draft.md).
`blocked_human` needs a complete `blocked_on` block. `killed` needs a `kill_reason`.

Folders that predate this schema carry `retrofitted: true` and may only be `done`, `killed`, `proposed` or `blocked_human`.
Never set `retrofitted` on new work.

## Integrity rules specific to this line of work

- Every number comes from a command or file that ran or exists. Estimates are labelled as estimates.
- **Quote every free-run result against the climatology floors.** The JCM floors are vortex corr **0.727** and σ-ratio **0.811**
  (nudged corpus); on the nudge-free corpus they are 0.998 and ~1.01.
  Report medians over ≥4 ICs, with the bad-rate (died / Tprof > 4 K / bias > 15 m/s).
- **Survival time alone is not a result.** It is noisy (the same seed gave 132 vs 40 d) and invalid under any clamp or bound.
- **When a result is withdrawn,** add a row to `plans/jcm-emulator-survey/RETRACTIONS.md`, set the experiment's verdict or state
  accordingly, and log it. Don't delete history.
- **Never edit a config another run depends on.** Copy it. Every run gets a seed, a config snapshot and a named output dir.

## Machines

`compute.machine: voltagepark`: bare metal, 8× H200, 1 TB RAM, no scheduler.

- Check `ps -u susanne` before using GPUs 0/1.
- Host RAM is shared, and other users' jobs have pushed it to 99 %. Long eval chains need the RAM guard (`ram_gate_requeue.sh`).
- After any reboot, check that `ls /dev/nvidia0` exists, or torch silently runs on CPU.
- Use PID files, never `pgrep -f` / `pkill -f`.

## Git

- Work on `jcm/<experiment-id>` branches, one experiment per branch. Open a PR into `main`.
- Never force-push, and never commit to `main` directly.
- Regenerate the README with `python tools/generate_readme.py --root .` after any frontmatter change. CI checks it.
