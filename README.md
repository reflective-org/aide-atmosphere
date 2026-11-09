# aide-atmosphere

Experiments with ML emulators of atmospheric models. The first line of work is emulators of JCM (`jcm-strat`, the JAX spectral dry GCM).

**Released models:** [`models/jcm-p2-hl/`](models/jcm-p2-hl/README.md) is the P2@200 stochastic emulator of a
stratosphere-resolving JCM dycore (T63, 95 levels to 0.01 hPa, 6-h step) that produced the 1997-1998 winds and temperatures
for the AIDE SAI campaign. It contains inference code, a smoke test and a rollout script; the weights are a release asset.

`experiments/<id>/EXPERIMENT.md` frontmatter is the single source of truth for each experiment. See [`CLAUDE.md`](CLAUDE.md) for the state
machine and the rules. The cross-experiment status of the JCM work, including every retracted claim, is in
[`plans/jcm-emulator-survey/`](plans/jcm-emulator-survey/SUMMARY.md).

The sections below are generated from experiment frontmatter and `roster.yaml` by `python tools/generate_readme.py --root .`.
Do not hand-edit them.

## Experiments

<!-- BEGIN:experiments -->
_No experiments yet._
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
