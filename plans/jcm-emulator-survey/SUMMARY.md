# JCM emulator survey — status of every line

Status as of **2026-10-22**. Survey of every ML-emulator line trained on JCM output on this machine (`jcm-strat`, the
JAX spectral dry GCM: T63 192×96, L63, Polvani–Kushner forcing, Rayleigh sponge, prescribed-QBO nudge, levels 0–8
nudged to ERA5). Every number below cites the file it came from. Retractions are listed in [RETRACTIONS.md](RETRACTIONS.md),
and nothing retracted is quoted as a result anywhere in this repo.

| doc | line | period | status |
|---|---|---|---|
| [00_truth_and_corpora](CORPORA.md) | jcm-strat truth runs and the zarr corpora built from them | Sep–Oct | reference |
| [01_small_head_campaign](../../experiments/jcm-small-head-stability/RESULTS.md) | ~1 M-param state head: filters, ζ/δ, thermal wind, anchor bound, dycore distillation | Sep 17 – Oct 1 | closed |
| [02_paradis_25m](../../experiments/jcm-paradis-free/RESULTS.md) | 25 M PARADIS port: prescribed, non-prescribed, 12-yr scale-up, anomaly formulation | Sep 23 – Oct 7 | closed; base ckpt for 04 |
| [03_nudge_free](../../experiments/jcm-nudgefree/RESULTS.md) | nudge-free corpus + emulator ("is the problem ill-posed?") and the filter sweep | Oct 5 – Oct 7 | closed, hypothesis falsified |
| [04_dagger_stabilization](../../experiments/jcm-finetune-stabilization/RESULTS.md) | fine-tuning epoch=123: DAgger, rollout loss, masked dycore term, T anchor, prescribed QBO | Oct 7 – Oct 22 | **active front** |
| [05_tracer_transport](../../experiments/jcm-transport-sl/RESULTS.md) | tracer transport: hybrid SL emulator, multi-year validation, learned 3-D operator (ali) | Sep 10 – Sep 30 | dormant |
| [RETRACTIONS](RETRACTIONS.md) | every claim that was later withdrawn, and why | | |
| [INVENTORY](INVENTORY.md) | code, logs, checkpoints, corpora: paths, owners, sizes, git state | | |

## Where it stands, in five lines

1. **No learned JCM emulator yet beats climatology on polar-vortex timing, and none has a QBO.** The climatology floors are vortex corr
   **0.727**, σ-ratio **0.811** (`PARADIS_JCM_RESULTS.md` §7). The only learned model that cleared them, the prescribed emulator at +0.927,
   ran with the scrambled W_QBO nudge on. That nudge pushed the polar vortex toward truth, so the result is void (RETRACTIONS R1).
2. **Stability is largely solved for 1–2 years, by fine-tuning.** `G300_step200` survives 8/8 test runs with 0 bad (bias 6.7/7.0 m/s,
   Tprof 1.6/1.7 K). The recipe is a K=4 archive rollout, a masked 48-h dycore-correction term, and a global-mean-T anchor.
   The anchor cuts bad runs **6×** (4/80 vs 24/80 pooled over 4 seeds, p<1e-4) (`/mnt/nvme_stage2/jcm_analysis/ranking/REPORT.md`).
3. **The correction it learned is not general.** On drift it was trained on it reproduces 72–79 % of the JCM's 10-day zonal-mean
   correction. On its *own* drifted states it reproduces ~0 % (`/mnt/nvme_stage2/jcm_analysis/correction/REPORT.md`).
4. **The free-running problem is not caused by the ERA5 nudging.** A nudge-free emulator trained on a nudge-free corpus also runs away.
   It dies at day 51–69, and its anomaly ratio grows with training (7 → 12 → 22). The registered falsifier was met .
5. **Transport works single-step but not over years.** The learned correction is better on 468/468 single steps and worse on 0/10 tracers
   from month 3. The analytic operator lacks a vertical remap. ali's learned 3-D operator passes the 10-yr age-of-air gate (−1.6 %) at a
   cost of 2.9× pulse variance .

## Current best

| use | checkpoint | evidence |
|---|---|---|
| overall (AMIP-style, needs observed QBO) | `/mnt/nvme_stage2/jcm_dagger_p6/ck/G300_step200.pt` + `--qbo-presc` (fixed W_QBO) | 8/8 survive, 0 bad, bias 5.3/5.9, tropical u err 1.7 (vs 9.4/10.7 without QBO). One seed, inference-only QBO |
| fully free (no truth at any step) | same ckpt without QBO | 8/8, 0 bad, bias 6.7/7.0, vortex .62/.79 |
| best vortex | `/mnt/nvme_stage2/jcm_dagger_p5/ck/SD3s1_step100.pt` | vortex .79/.87; the recipe without the anchor falls into the bad regime in ~30 % of checkpoint-runs |
| pending | `/mnt/nvme_stage2/jcm_dagger_p7/fix/ck/QF{2,3}_step*.pt` (anchor + fixed QBO **in training**) | QF2_s50 1996: 4/4, bias 5.7, vortex .76. Evals were RAM-killed 10-21/10-22. QF3 and steps 100–200 are unscored |

All figures are on test years 1996/1995, 4 ICs each. The source is `ranking/REPORT.md` unless noted. "Bad" means died OR Tprof > 4 K OR bias > 15 m/s.

## Lessons that held up across lines

- **Survival time is a near-useless metric on its own.** The same config and seed gave 132 vs 40 d. One checkpoint spans 40–169 d across ICs.
  Under any clamp or bound the metric is invalid. Use medians over ≥4 ICs × ≥3 runs, plus balance metrics (Tprof, zmV, bias).
- **Metric floors first.** A frozen climatology scores σ-ratio 0.811 and vortex corr 0.727, and a frozen state scores σ-ratio 0.94–1.24.
  σ-ratio ≈ 1 alone proves nothing.
- **Validation loss does not track free-run climate.** Two examples: the anomaly run has better val loss and a worse vortex; DAgger
  improves stability and worsens held-out one-step loss.
- **Single-step gates do not predict rollouts.** This held for transport (+0.047 R² single step, worse at month 3) and for state heads.
- **Fine-tune harness matters more than the loss.** On the old harness (batch 1, no warmup), archive-only fine-tuning broke balance by itself
  (+86 K at level 8). Every "DAgger works" result from that harness is void.
- **The early blow-ups were grid-scale numerical instability.** The model had no spectral truncation or hyperdiffusion. A filter fixes
  *death*, not *wrongness*: the remaining error is planetary (l < 10).
- **The real JCM dycore reproduces a 6-h archive step to 1.3 % of persistence.** A dycore + learned-correction hybrid saturates at 3–4× speedup.

## Open next steps (as written in the source docs, not yet done)

1. Finish the QF2/QF3 evals with the strict selection rule, and replicate across seeds (ranking §4.1).
2. DAgger round 2 on G300's **own** drifted states (training years only), then re-run the out-of-sample correction test on a held-out year (correction REPORT).
3. More mean-state anchors: hemispheric/polar-cap T and zonal-mean u (ranking §4.2).
4. Longer dycore-correction horizon (4–10 d); cross-seed soups of accepted anchor checkpoints (ranking §4.3).
5. Train from scratch with rollout + dycore term + anchor. Fix the loader for ≥5-yr runs (p5 REPORT).
6. Transport: add a vertical remap to the analytic operator; couple ali's operator to emulated winds.
7. Never executed: the boundary-forced arm (`/data/JCM_BOUNDARY_FORCED_SPEC.md`, levels 9–62 with a replay troposphere, gates G1–G7).

## Open work → experiments and issues

- `jcm-qf-qbo-anchor-eval` — Anchor + corrected prescribed QBO in training (QF2/QF3): finish evals, strict selection (#1)
- `jcm-dagger-own-drift` — DAgger round 2 on G300's own drifted states (#2)
- `jcm-regional-anchors` — Mean-state anchors beyond global-mean T: hemispheric/polar-cap T, zonal-mean u (#3)
- `jcm-dycore-horizon-soups` — Longer dycore-correction horizon (4-10 d) and cross-seed soups (#4)
- `jcm-from-scratch-recipe` — Train from scratch with rollout + dycore term + anchor (#5)
- `jcm-transport-vertical-remap` — Vertical remap for the analytic SL transport operator (#6)
- `jcm-transport-operator-coupling` — Couple the learned 3-D transport operator to emulated winds (#7)
- `jcm-boundary-forced` — Boundary-forced arm: levels 9-62 with a replay-year troposphere (run or drop?) (#8)
- Commit the `/data/paradis_jcm` working tree (#9)
- Correct stale claims in the /data source docs (#10)
