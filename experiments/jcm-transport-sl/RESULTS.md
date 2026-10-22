# RESULTS — jcm-transport-sl

# 05 — Tracer transport emulators (Sep 10 – Sep 30) — dormant

The design rule throughout is **"nonlinear in flow, linear in tracer"**: tracers are never inputs to a CNN. The corpus is p11a tracers
(pulse_1–5, src_1–4, sai, n2o, cfc11, aoa).

## T1. Hybrid SL transport emulator (b-series), `/data/jcm_tracer_emulator/`

- **Docs:** `/data/TRACER_TRANSPORT_EMULATOR_SPEC.md` (83 kB, with a running §9 log) and `/data/TRACER_TRANSPORT_EMULATOR_RESULTS.md`.
- **Model:** analytic departure-point solve + cubic-Lagrange gather + quasi-monotone limiter, plus a learned trajectory correction (1.46 M
  params, levels-as-channels).
- **Test setup:** test year 1996, holdout site 4.

| result | number | source |
|---|---|---|
| single-step R², learned vs analytic | **0.8668 vs 0.8195** (+0.0473 ± 0.0016); better on **468/468** frames | `c1_eval.json`, RESULTS §1 |
| ablation D1 (remap kept, departure learned) / D2 (direct CNN) | 0.4931 / **−65.55** | `c1_eval.json` |
| data scaling | +0.0044 (8 frames) → +0.0473 (7,020), no turnover | RESULTS §3.4 |
| timestep 6 h / 12 h / 24 h | 0.886 / 0.733 / −0.373: long steps closed | `b6_timestep.json` |
| speed | 2.92 ms vs JCM 198 ms per 6-h step = **68×** on H200; transport is 32.3 % of a JCM step | RESULTS §3.5 |

The state head that accompanied it is covered in 01.

## T2. Offline multi-year validation (m-series), `/data/TRANSPORT_MULTIYEAR_VALIDATION.md`

Code: `m0_pipeline_check.py`, `m1_offline_transport.py`, `m2_report.py`. The operators are driven by **archive** winds.

| result | number |
|---|---|
| horizontal (within-level) structure | mean \|ln ratio\| 0.170 at yr 1, flat from day 100: **sound** |
| vertical | pulse_4 above 100 hPa at yr 1: truth 25.9 % vs model **100 %**; centroid 216.5 vs **4.53 hPa** |
| 3-D structure ratio (analytic/truth) | pulse_4 2.30 (yr 1) → 25.71 (yr 5) |
| mass without the fixer | −59 % to +164 % over 5 yr |
| positivity / max principle | 0 violations in 8,763 steps |
| sub-stepping K = 8 | < 2 % effect over 5 yr |
| **learned correction head over rollout** | within-level 0.426 vs analytic 0.170 at yr 1; better on **0/10** tracers from month 3; **2.3× over-diffusion** on the held-out site |

**Verdict:** the missing **vertical remap** is the multi-year blocker. A per-step "few-percent" term is not a per-year one, and the earlier
"vertical is a few-percent term" is retracted. Single-step skill is not a rollout gate.

**Stale doc:** the doc says the 20-yr run is "still running at 8,763 steps". `m1_offline_20yr.log` shows it completed 29,219 steps in 61.8 min.
At 2010-01-01, analytic pulse_1 is 1.616 vs truth 0.017. The §3/§4/§6/§8 tables were never regenerated with `m2_report.py`.

## T3. Learned 3-D transport operator (owner: **ali**), `/data/jcm_transport_operator/`

- **Docs:** spec `/data/JCM_TRACER_TRANSPORT_SPEC.md` (written 09-28), results `/data/jcm_transport_operator/RESULTS.md` (09-30).
  The README venv points at `/home/shreyas/venvs/jto`.
- **Model:** row-stochastic trilinear operator with a 3-D Cartesian departure solve and a zero-initialised 3.13 M-param correction.
  Trained on box g265. Gates G1–G6 run on multi-year rollouts; gate JSONs are in `gates/`.

| stage | result (RESULTS.md) |
|---|---|
| Phase 0, analytic | single-step R² 0.942. The archive is not locally conservative; without the global rescale, mass grows 4.7×/yr (§3–4) |
| analytic, 1 yr | pulse_4 at 295 hPa, G2 1.13, age-of-air 123 vs 147 d: **G4 fail** (§5.1) |
| Stage 1, r1 step 7500 | 1 yr: G1/G2/G4/G5 pass, nRMSE halved. **10 yr: aoa 585 vs 427 d (+37 %, fail)** (§5.2) |
| Stage 2, K = 40 | does not fix the clock (+23 to +32 %) |
| **Stage 3, `train_clock.py`** (1-yr backprop through aoa), **r3 step 7545** | 10-yr aoa 420 d (**−1.6 %, pass**), aoa nRMSE 0.04. Cost: homogenised-pulse variance **2.9×** at yr 10 (§5.3) |
| G6 vs real dycore (perturbed 1997 restart) | response corr 0.988 (blob) / 0.994 (source) at day 80 (`g6_dycore/README.md`) |

- **G3 (local conservation) cannot be met by this operator class.** The archive does not meet it either.
- **Not started:** coupling to *emulated* winds (`/data/jcm_emulated_winds` exists for this); extra diffusion for pulses.

## Unresolved inconsistencies across T2/T3

- Truth pulse_4 position at yr 1 is 216.5 hPa centroid in T2 but "397 hPa" in T3, from the same corpus and start date. The metric definitions
  probably differ (centroid vs another statistic). **Not reconciled.**
- Single-step R² 0.8195 (T1, cubic horizontal) vs 0.942 (T3, trilinear 3-D) use different frames and references. Don't compare them.
- Memory note `project_jax_gcm_transport_head.md` (09-03) still says the transport head is "dropped". The 09-17 and 09-28 specs supersede it.

