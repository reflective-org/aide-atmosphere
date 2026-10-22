# RESULTS — jcm-finetune-stabilization

# 04 — Stabilizing the 25 M emulator by fine-tuning (Oct 7 – Oct 22) — ACTIVE

Every arm fine-tunes the base **`epoch=123`** (`jcm_paradis_free_big`, 02/B3) and is gated **fully free** (qbo=False).
The exception is phase 14, which uses the corrected QBO.

**Where things are:**

| what | path |
|---|---|
| Phase 0 – q18 | `/data/jcm_dagger/` |
| p5, p6, p7 | `/mnt/nvme_stage2/jcm_dagger_p{5,6,7}/` (moved to NVMe when susanne's p22 runs filled /data to 100 %) |
| Consolidated analysis | `/mnt/nvme_stage2/jcm_analysis/{ranking,correction,tracking}/` |
| Existing summary page | `/mnt/nvme_stage2/jcm_dagger_p6/jcm_emulator_findings.html` (built by `build_page.py`) |

**Metrics** (`ranking/REPORT.md`):
- surv = survives 360 d
- **bad** = died OR Tprof > 4 K OR bias > 15 m/s
- bias = time-mean zonal-mean u error, days 30–end
- Tprof = global-mean T-profile rms error
- zmV / gmV = max zonal-mean / global-mean v (balance)
- vx corr = polar-vortex corr
- u trop / u extra = time-mean zonal-mean u error, |lat| < 20 / ≥ 20, cos-weighted

**Data splits:** 1996 and 1995 are test (4 ICs); 1998 ×4 + 2009 IC0 is selection; 2009 is the val year; 1998 is a training year.

## Phase 0 — one-step DAgger round 1 + Jacobian (Oct 7–8, `/data/jcm_dagger/`)

- **Bar** (`AUTONOMOUS_PLAN.md`): survival 4/4, vortex ≥ 0.48, std 0.7–1.3, anomR90 ≤ 1.8.
- **Base** (`round1.log`): 2/4 survive (d260, d191), vortex +0.482, std 0.83, anomR90 1.619.
- **ck_r1:** 4/4 survive, but vortex +0.056, std 0.43, anomR90 2.66. **Survival bought by damping.**
- **p11a-target arms:** ck_A median 31 d; ck_B with a Jacobian-gain loss 24 d. `gate_C.log` is empty.
- **Jacobian gain** (z6b/z7b): reported "CONFIRMED amplification", 1.4× planetary and 53× grid. **Largely retracted 10-15** (R6): ~28× of
  the 70× is T63 round-trip truncation, and every JVP carries a 9–17 % non-linearity warning. Every arm with the gain loss died or anti-correlated.

## q20261015–16 — fixed/analog targets, seeds (**superseded; unbalanced**)

- **Ana_s1** looked best: 4/4, vortex .784 (`q20261015/summary.txt`).
- Analog seeds s3–s8 spread from vortex .04 to −.41 (s8 0/4 survive).
- Round-2 warm-start destroyed vortex phase (`q20261016/summary.txt`).
- **All rankings from these two dirs are superseded:** Ana_s1 is unbalanced (R5).

## q20261017 — balance diagnostics and the harness fix

- **Ana_s1:** Tprof 16.1 K, zmV 35.1, bias 20.2 (base 1.3 / 7.3 / 8.9). Held-out 1998 one-step loss 0.0271 vs base 0.0123.
- **Cause: the harness, not DAgger.** Archive-only fine-tuning on the old batch-1 / no-warmup harness also breaks balance: day 45 gm v max
  20.05 vs truth 1.16, dT(level 8) +86 K (`logs/bal_ab_archonly.log`).
- **Fixed harness:** effective batch 36, archive 1997–2008, warmup, β2 0.95. Null test passes: `null_lr1e-5` 2/4, vortex .49, held-out 0.01226.
- **One-step DAgger on the fixed harness** (`dv2_w0.005…1.0`): every weight is worse than base (bias 17.1–38.4). |g_dag|/|g_arch| ≈ 200, cos 0.17–0.26.
- **Weight soups and interpolations:** all unbalanced (bias 22–26).

## q20261018 → p5 — rollout loss + masked dycore term (Oct 17–18)

Sources: `/data/jcm_dagger/q20261018/AUTONOMOUS_PLAN.md`; `/mnt/nvme_stage2/jcm_dagger_p5/{PLAN_LOG,REPORT}.md`.

| arm | what | 1996 result | verdict |
|---|---|---|---|
| R4 / R8 / P2R4slow | archive rollout K=4/8 | vortex and 96-h error improve (0.0250 vs 0.0278); bias not reduced (R4 13.0, slow 10.0). R4's 4/4 failed its seed replicate (0/2) | helps survival, not mean state |
| R4D / R8D | + **unmasked** dycore term | bias 15–27, Tprof 3–13 | worse |
| flux180 | flux adjustment τ = 180 | 1996 bias 8.7 vs 10.0; 1995 12.4 vs 9.7 | does not generalise |
| **SD3** | K=4 rollout, lr 5e-6, 100 steps + dycore term on zonal-mean u, v over 48 h, **free band only (levels 9–58)**, w 0.042 | s0: 4/4, vortex .78, bias 7.8, Tprof 2.1; s1: vortex .79/.87 (1996/1995). 2-yr runs stable, bias 7.4–7.6 | **adopted** |
| SD3v2 | variant | bias 15.7/18.4, Tprof 5.5/6.3 | fails |

- **Method flaw:** SD3 step 100 was first chosen on test ICs. Re-selection on 2009 IC0 alone picked the same ckpts.
- **Phase 9:** bias is bimodal (modes at 8–10 and 18–22 m/s). The recipe yields 3 of 4 seeds, and ~30 % of checkpoint-runs land in the bad
  regime. A strict "every selection run good" rule is needed, because median selection let seed 2 through.

## p6 — anchors, controls, replication (Oct 20–21)

Source: `/mnt/nvme_stage2/jcm_dagger_p6/tasks.txt`. **There is no REPORT/PLAN_LOG**, although `build_page.py` cites them.
The numbers below are from `ranking/REPORT.md`.

| comparison (selection set, 20 runs/arm) | bad A vs B | bias A / B |
|---|---|---|
| **global-mean T anchor w300 vs plain, pooled seeds 2–5** | **4/80 vs 24/80** (p < 1e-4) | 8.8 / 11.8 |
| fresh seeds 4, 5 only | 4/40 vs 6/40 (p = 0.74) | — |
| **anchor vs low-lr movement control** (seeds 2+4) | **1/40 vs 12/40** (p = 0.0015) | 9.0 / 11.3 |
| low-lr vs normal-lr plain | 12/40 vs 19/40 (p = 0.17) | — |
| anchor w1000 vs w300 | 3/20 vs 0/20 | = |
| free-band T in dycore term vs plain | 5/20 vs 15/20 (p = 0.004); no gain on top of the anchor | — |
| EMA 0.98 vs raw | 9/20 vs 15/20; 5/20 vs 3/20 | within noise |
| within-seed soups | worse (land in the bad regime) | — |
| cross-seed soup of 3 accepted ckpts | 0/5 | 9.0 |

The 20 runs of an arm share a seed and overlapping ckpts, so these p-values are optimistic.

**Test results (1996 / 1995, 4 ICs each):**

| ckpt | surv | bad | vortex | bias | Tprof | u trop | u extra |
|---|---|---|---|---|---|---|---|
| base | 2/4, 0/4 | 2/4, 4/4 | .48/.37 | 8.9/9.1 | 1.3/3.7 | 10.5/13.3 | 7.2/10.8 |
| SD3s0_step100 | 4/4, 4/4 | 0, 0 | .78/.79 | 7.8/7.8 | 2.1/2.1 | 9.4/14.4 | 5.4/6.0 |
| SD3s1_step100 | 4/4, 4/4 | 0, 0 | .79/.87 | 8.2/8.1 | 2.3/2.2 | 9.9/13.9 | 5.9/6.3 |
| **G300_step200** | 4/4, 4/4 | 0, 0 | .62/.79 | 6.7/7.0 | 1.6/1.7 | 9.4/10.7 | 4.8/4.6 |
| DTG300_step200 | 4/4, 4/4 | 0, 0 | .73/.65 | 6.2/7.3 | 1.3/1.9 | 11.4/8.3 | 5.4/5.7 |

Vortex numbers are against the 0.727 climatology floor. **No ckpt clears it on both years.**

## p7 — prescribed QBO (Oct 21–22)

- **p7/diag (Q300, first G300+QBO): RETRACTED** (R1). Trained/scored with the scrambled W_QBO.
- **p7/fix, G300_step200 + corrected QBO at inference only:** 8/8, 0 bad, bias **5.3/5.9**, Tprof 1.1/1.4, tropical u error **1.7/1.7**
  (vs 9.4/10.7), extratropical 3.3/3.7, vortex .61/.75. This is the #1 entry in the ranking.
  It needs the observed QBO for the year (AMIP-style), and it is one ckpt from one seed.
- **QF2/QF3 (anchor + corrected QBO in training, seeds 2, 3):** training finished 10-21 19:10–19:13; ckpts in `p7/fix/ck/`.

| run | surv | bias | vortex | Tprof | zmV |
|---|---|---|---|---|---|
| QF2_s50 selection 1998×4 | 4/4 | 5.9 | .58 | 1.6 | — |
| QF3_s50 selection 1998×4 | 4/4 | 5.4 | .56 | 1.15 | — |
| QF2_s50 test 1996 | 4/4 | 5.7 | .76 | 1.5 | 7.4 |
| QF2_s50 test 1995 (**3 ICs**) | 3/3 | 6.2 | .78 | 2.8 | 7.8 |

These are medians computed from `p7/fix/diag/*.json` by the survey agent, not taken from a report. One 1995 IC has Tprof 3.0, and zmV is higher
than G300+QBO (5.6–6.1).
**Unscored:** QF3 tests, and all QF steps 100–200. Status: 87 eval tasks are `.fail` in `p6/status`. A RAM guard (≥ 80 % host RAM, driven by
susanne's jobs) killed them on 10-21 at 19:50. The 10-22 06:27 requeue was killed again at 07:23–07:30. As of this survey nothing JCM is
running, the GPUs are idle, and host RAM is low, so they can be re-queued.

## Does the learned correction generalise? (`jcm_analysis/correction/REPORT.md`, 10-22)

Metric: fraction of the free JCM's correction from a drifted state that the emulator reproduces.

| states | metric | base | SD3 | G300 |
|---|---|---|---|---|
| base's drift (**in-sample** for SD3/G300) | zonal-mean u, 10 d | −8 % | 72 % | 79 % |
| | l = 0–2, 10 d | −11 % | 77 % | 89 % |
| | max zonal-mean v (8.05 start; JCM 2.05) | 8.05 | 5.03 | 4.97 |
| **G300's own 1998 drift (out-of-sample)** | zonal-mean u: start 11.56 → G300 11.36 vs free JCM 7.58 | | | **~0 %** |

**The dycore term taught a correction to the specific drift it saw, not a general restoring force.** G300 drifts less to begin with,
but once drifted it does not pull back. This undercuts the general-mechanism story in the p5 REPORT.

## Tracking (`jcm_analysis/tracking/`, no written report)

Days for free-band u error to reach climatology level (`data/forecast_summary.json`):

| ckpt | days |
|---|---|
| base | 13.75 |
| SD3 | 12.75 |
| G300 | 12.25 |
| G300+QBO | 15.25 |

Stabilisation did not buy forecast skill.

## Ingredient verdicts (ranking §2)

| ingredient | verdict |
|---|---|
| fixed harness (batch 36, 12 yr, warmup) | prerequisite |
| one-step DAgger, any weight | worse |
| archive rollout K = 4 | survival up, bias not; failed seed replicate |
| masked dycore term (SD3) | better than rollout-only on test (0/8 vs 4/8 bad); ~30 % bad over ckpts |
| unmasked dycore term | worse |
| **global-mean T anchor w300** | **strongest finding: 6× fewer bad**; confound ruled out |
| EMA, within-seed soups, flux adjustment | no / negative / does not generalise |
| prescribed QBO (corrected, inference) | big tropical gain, bias better; 1 ckpt |

