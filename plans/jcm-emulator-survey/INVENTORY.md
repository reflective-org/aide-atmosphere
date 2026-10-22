# Inventory — what to put in the repo, and what stays on disk

Owner is `ubuntu` unless noted. The "code" column counts `.py/.sh/.yaml/.md` files at depth ≤ 3, measured 2026-10-22.

| path | contents | code size | git | include in repo? |
|---|---|---|---|---|
| `/data/paradis_jcm` | PARADIS-JCM model, trainer, gate, configs, converters, nf/anom/filter scripts + logs | 96 files, 455 K | yes; last commit 09-25, **164 uncommitted changes** (W_QBO fix, trainer, gate, anom/nf configs) | yes: commit the working tree first |
| `/data/jcm_dagger` | Phase 0 – q18 fine-tune scripts (`d1_roll.py`, `d2_target*.py`, `d3_train*.py`), logs, summaries | 64 files, 283 K | no | scripts + summaries + logs; not `.pt` / `.npz` (67 G) |
| `/mnt/nvme_stage2/jcm_dagger_p5` | SD3 campaign: PLAN_LOG, REPORT, tasks.txt, ckpts | 5 files | no | docs |
| `/mnt/nvme_stage2/jcm_dagger_p6` | anchor/replication: tasks.txt, `build_page.py`, `jcm_emulator_findings.html`, ckpts (`ck/G300_step200.pt` = current best) | 10 files | no | docs + page |
| `/mnt/nvme_stage2/jcm_dagger_p7` | QBO redo (`fix/`); `diag/` is retracted | 3 files | no | `fix/` diag JSONs |
| `/mnt/nvme_stage2/jcm_analysis` | ranking (collect/tables/attrib + REPORT, all_runs.json over 671 runs), correction, tracking | 18 files, 120 K | no | yes, all |
| `/data/jcm_tracer_emulator` | small-head (x/y/z) + hybrid transport (b/c/m) scripts, JSON results, logs | 161 files, 901 K | no | yes |
| `/data/jcm_transport_operator` (**ali**) | learned 3-D operator, gates, g6_dycore | 30 files, 194 K | no | ask ali |
| `/data/jcm_nf63`, `/data/jcm_nudgefree` | nudge-free chain scripts + Hines sweep | 66 files | no | scripts only |
| `/data/JCM_stripped/jcm-strat*` (**susanne**) | truth model, phases 1–23 | — | — | reference by link only |
| top-level docs in `/data` | `PARADIS_JCM_RESULTS.md`, `JCM_ANOMALY_FORMULATION_RESULTS.md`, `JCM_FILTER_SWEEP_RESULT.md`, `JCM_NUDGEFREE_HYPOTHESIS.md`, `JCM_BOUNDARY_FORCED_SPEC.md`, `JCM_TRACER_TRANSPORT_SPEC.md`, `TRACER_TRANSPORT_EMULATOR_{SPEC,RESULTS}.md`, `TRANSPORT_MULTIYEAR_VALIDATION.md` | ~230 K | no | yes, as `history/` (original text, with RETRACTIONS cross-referenced) |

## Checkpoints (do not commit; record paths and hashes)

| ckpt | path |
|---|---|
| base epoch=123 | `/data/paradis_logs/lightning_logs/jcm_paradis_free_big/` |
| prescribed (void), free, anom, nudgefree | `/data/paradis_logs/lightning_logs/jcm_paradis_{v1,free,anom,nudgefree}/` |
| SD3 | `/mnt/nvme_stage2/jcm_dagger_p5/ck/` |
| G300 / DTG300 / DT / E2 / E3 / LR1 / r3–r5 | `/mnt/nvme_stage2/jcm_dagger_p6/ck/` |
| QF2 / QF3 | `/mnt/nvme_stage2/jcm_dagger_p7/fix/ck/` |

Old checkpoints carry the scrambled W_QBO buffer. The fixed `JCMPhysics._load_from_state_dict` drops it on load, so load them only with post-10-21 code.

## Not JCM; excluded from this survey

- `/data/RADIATION_MODULE_*`: trained on CESM v6.
- `/data/cesm-jax-dycore` (susanne): a truth-model alternative.
