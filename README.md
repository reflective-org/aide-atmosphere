# aide-atmosphere

Experiments with ML emulators of atmospheric models. The first line of work is emulators of JCM (`jcm-strat`, the JAX spectral dry GCM).

`experiments/<id>/EXPERIMENT.md` frontmatter is the single source of truth for each experiment. See [`CLAUDE.md`](CLAUDE.md) for the state
machine and the rules. The cross-experiment status of the JCM work, including every retracted claim, is in
[`plans/jcm-emulator-survey/`](plans/jcm-emulator-survey/SUMMARY.md).

The sections below are generated from experiment frontmatter and `roster.yaml` by `python tools/generate_readme.py --root .`.
Do not hand-edit them.

## Experiments

<!-- BEGIN:experiments -->
| experiment | state | owner | issue | next action |
| --- | --- | --- | --- | --- |
| [PARADIS 25M free base on the p20bt (L95, Jucker + Hines) corpus, and why it dies](experiments/jcm-p20bt-free-base/EXPERIMENT.md) | analysis | shreyas |  | Decide how to carry the m=0 damping into training (a fixed m=0 vector-SH filt... |
| [Boundary-forced arm: levels 9-62 with a replay-year troposphere (run or drop?)](experiments/jcm-boundary-forced/EXPERIMENT.md) | proposed | shreyas | [#8](../../issues/8) | Decide whether to run or drop now that fine-tuning stabilises the full column... |
| [DAgger round 2 on G300's own drifted states](experiments/jcm-dagger-own-drift/EXPERIMENT.md) | proposed | shreyas | [#2](../../issues/2) | Write the preregistration (falsifier: out-of-sample fraction of the JCM corre... |
| [Longer dycore-correction horizon (4-10 d) and cross-seed soups](experiments/jcm-dycore-horizon-soups/EXPERIMENT.md) | proposed | shreyas | [#4](../../issues/4) | Write the preregistration; price a 4-10 d dycore horizon with the p5 harness ... |
| [Train from scratch with rollout + dycore term + anchor](experiments/jcm-from-scratch-recipe/EXPERIMENT.md) | proposed | shreyas | [#5](../../issues/5) | Fix the >=5-yr loader, then the ladder (shapes, overfit, tiny run with s/step... |
| [Anchor + corrected prescribed QBO in training (QF2/QF3): finish evals, strict selection](experiments/jcm-qf-qbo-anchor-eval/EXPERIMENT.md) | proposed | shreyas | [#1](../../issues/1) | Re-queue the 87 RAM-killed eval tasks (p6/status) behind the RAM guard after ... |
| [Mean-state anchors beyond global-mean T: hemispheric/polar-cap T, zonal-mean u](experiments/jcm-regional-anchors/EXPERIMENT.md) | proposed | shreyas | [#3](../../issues/3) | Write the preregistration and the matched comparison against G300 on the sele... |
| [Couple the learned 3-D transport operator to emulated winds](experiments/jcm-transport-operator-coupling/EXPERIMENT.md) | proposed | ali | [#7](../../issues/7) | Coordinate with Ali (roster: jcm-transport-operator) before touching his code... |
| [Vertical remap for the analytic SL transport operator](experiments/jcm-transport-vertical-remap/EXPERIMENT.md) | proposed | shreyas | [#6](../../issues/6) | Implement the vertical remap and rerun the m-series 5-yr validation; regenera... |
| [Anomaly formulation: prescribed seasonal climatology + learned anomaly](experiments/jcm-anomaly-formulation/EXPERIMENT.md) | done | shreyas |  |  |
| [Stabilizing the 25M emulator by fine-tuning: DAgger, rollout loss, masked dycore term, T anchor, prescribed QBO](experiments/jcm-finetune-stabilization/EXPERIMENT.md) | done | shreyas |  |  |
| [Is the free-run failure ill-posed? Nudge-free corpus, emulator and filter sweep](experiments/jcm-nudgefree/EXPERIMENT.md) | done | shreyas |  |  |
| [PARADIS 25M fully free (no prescription, no QBO nudge): 5-yr, 10-yr climate, 12-yr scale-up](experiments/jcm-paradis-free/EXPERIMENT.md) | done | shreyas |  |  |
| [Small state head + analytic forcing: can it free-run JCM stably?](experiments/jcm-small-head-stability/EXPERIMENT.md) | done | shreyas |  |  |
| [Hybrid semi-Lagrangian tracer transport emulator and multi-year validation](experiments/jcm-transport-sl/EXPERIMENT.md) | done | shreyas |  |  |
| [PARADIS 25M with levels 0-8 and the QBO prescribed from truth](experiments/jcm-paradis-prescribed/EXPERIMENT.md) | killed | shreyas |  |  |
<!-- END:experiments -->

## Waiting on Shreyas

<!-- BEGIN:blocked -->
_Nothing is waiting on a decision._
<!-- END:blocked -->

## Adjacent work

<!-- BEGIN:roster -->
| who | topic | status | notes |
| --- | --- | --- | --- |
| Susanne | jcm-strat truth model (phases 1-23) and the JAX CAM-FV dycore port | active | Every JCM corpus here comes from her runs (p11a, p13). A change in the truth model invalidates corpus-level comparisons. Her jobs share GPUs 0/1 and host RAM on the same box. |
| Ali | Learned 3-D tracer transport operator and the jcm_tracers_c1 corpus | active | /data/jcm_transport_operator and /data/jcm_tracers_c1 are his. Anything coupling transport to emulated winds overlaps here. |
| Duncan | Neural-JAM (emulated radiation / aerosol optics / microphysics inside JAM), climate-analytics-lab/jcm-experiments#17 | active | Any learned radiation or aerosol component overlaps. Also see the roster in jcm-experiments for the cloud-emulator and NN-bias-correction efforts. |
<!-- END:roster -->
