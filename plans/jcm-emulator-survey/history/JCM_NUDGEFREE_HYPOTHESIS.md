# The nudge-free arm — "emulate the dycore by itself"

**Registered 2026-10-05, BEFORE the result is known.** Experiment running. Written so the
prediction and the falsifiers are on record rather than reconstructed afterwards.

---

## 1. The hypothesis

**The free-running emulator fails because the learning problem is ILL-POSED, not because of model
capacity, architecture, or loss.**

In the p11a training corpus, model levels 0-8 are relaxed toward ERA5 with tau = 6 h. Over a single
6-h step that removes **63%** of the deviation (`1 - exp(-1)`). Those levels are therefore **not a
function of the model state** — they are an external observational time series played back. No
state-function emulator can reproduce them, because the information required is not in the state.

We then trained a state-function emulator on all 63 levels and asked it to free-run. It cannot
reproduce the term, so it diverges exactly there.

**Claim: train on a corpus where every level is determined by the state, and the failure
disappears.** The nudge-free corpus has `nudging.enabled=false` and the QBO nudge rate set to ~0,
so nothing is truth-dependent. The emulator's own analytic terms (Polvani-Kushner relaxation +
Rayleigh sponge) then match the data generator **exactly**, and every prognostic level is learnable.

## 2. The evidence it rests on — all measured, with sources

| fact | value | source |
|---|---|---|
| nudged band | `min_pressure_hpa=150.0`, `tau_hours=6.0` -> levels 0-8 = 985.34 -> 151.53 hPa; level 9 = 137.92 hPa free | `runs/p11a_19970101/.hydra/config.yaml` |
| relaxation strength | **63% of the deviation removed per 6-h step** | `1 - exp(-6/6)` |
| blow-up is co-located with the nudging | day 250 max\|u\|: **levels 0-8 = 420.0**, free band 323.1, sponge 54.3 (truth ~100) | `emulate_winds.py` on `/data/jcm_emulated_winds/1990` |
| whole field, not outliers | day 250 median \|u\| 88.3 vs truth 11.3 | same |
| the operator itself is GOOD on-manifold | relative error at TRUTH states **0.14-0.20, flat from day 30 to day 300** | `z2_operator_error.py` vs the real dycore |
| it is only bad where it goes | at VISITED states 0.58-0.82 -> **exposure gap 4.8-7x** | same |
| the state leaves the physics manifold | by day 210 the real dycore returns **NaN** from the emulator's state; at day 180 it steps it cleanly and *dissipates* (12.80 -> 12.71) | `z1_dycore_targets.py` |

The combination is the argument: the operator is accurate everywhere the archive can supervise it,
and the archive cannot supervise the one layer that is externally driven.

## 3. Prediction

If the hypothesis is right, the nudge-free arm should show:

1. **No tropospheric runaway.** Levels 0-8 stay comparable to the free band; no 420 m/s.
2. **Reliable 360-day survival** under the FIXED gate (real survival, FAIL threshold 200 m/s).
3. **Small temperature drift** — the nudge-free dycore itself holds 232.79-232.83 K across 16 years.

## 4. Falsifiers — stated in advance

- **If the nudge-free emulator ALSO runs away**, the hypothesis is WRONG. Ill-posedness is then not
  the cause and the problem is architecture/training: the 4.8-7x exposure gap, or the missing
  spectral filter (the largest lever this campaign ever measured, day 25 -> 369 survival, never
  ported to PARADIS).
- **If it is stable but the vortex is still below the (recomputed) climatology floor**, then
  well-posedness is **necessary but not sufficient**, and the next lever is dycore-in-the-loop
  supervision, which this corpus makes clean (section 7).
- **If it is stable AND beats its floor**, the diagnosis is confirmed and the platform is sound;
  what was removed (a realistic troposphere, the QBO) then gets added back deliberately.

## 5. What this experiment explicitly does NOT claim

- **It will not produce a QBO.** The nudge-free system has none at any Hines amplitude (measured,
  earlier campaign). The QBO in the nudged data came from the nudge, which is a trajectory-matching
  term with no autonomous form. See the retraction in `PARADIS_JCM_RESULTS.md` section 5.1.
- **It is not fidelity to the nudged JCM.** The nudge-free attractor has median \|u\| ~3.5 vs the
  nudged truth's 9.43 — roughly **2.3x weaker circulation**. Circulation strength sets the
  Brewer-Dobson transport timescale, so transport timescales learned here will NOT match the system
  we ultimately care about.
- Therefore this is a **dynamics-correctness** experiment: can a state-function emulator reproduce
  an autonomous dycore at all? It deliberately trades climate fidelity for a well-posed question.

## 6. Design — one variable

Diff against the nudged baseline config (`config_jcm_free_big_fast.yaml`) is **exactly `root_dir`
and the date ranges**. Identical: 25.0 M params, 6 GPUs, batch 3 x accum 2 = effective 36,
60k steps = 2.16M presentations, PK + sponge on, `prescribe: false`, `qbo: false`, same loss,
same schedule. **The corpus is the only variable.**

Splits: train 2009-2018 (10 contiguous equilibrated years), val 2019, test 2020, plus 2021
converted because the fixed gate spans into `year+1` so late ICs have truth.
Spin-down 2005-2008 discarded: KE falls 325.5 -> 297.8 over those years and is flat (291-294) from
~2012. Residual drift 2009->2020 is 1.7% in KE.

## 7. Why this corpus also unlocks dycore-in-the-loop supervision

With the nudged corpus, a DAgger target would have required the ERA5 cache and would have taught
the emulator a relaxation term it can never apply when free. **With the nudge-free corpus the
dycore config and the data generator are the same operator**, so "what should happen from THIS
state?" is well-posed at any drift and is exactly what the archive contains.

Validated: outside 60 deg latitude the dycore reproduces the nf archive's own 6-h step to
**0.6-3.3% of persistence**. A whole-globe figure of 8-12% is NOT dynamical error — it is the
`u/cos_lat` polar transform artifact (polar v round-trips at 52%, T at 0.03%). Area weighting by
cos(phi) suppresses it ~10x automatically and must not be dropped.
See `reference_jcm_dycore_roundtrip_polar` and `z4_roundtrip.py`.

## 8. REQUIRED before the result means anything

**The climatology floors must be recomputed on the nudge-free corpus.** The published floors
(sigma_ratio **0.811**, vortex correlation **0.727**, day-30 RMSE 11.4) were measured against the
*nudged* truth. This corpus has a materially different climate, so scoring a nudge-free model
against nudged-truth floors compares a model of one climate to the climatology of another. Floors
must be re-derived from the nf test year before any verdict.

The gate must also be run in its FIXED form: real survival (`died` / `DATA-END` / `ok`), FAIL
threshold 200 m/s, sigma_ratio at d270/d359, and the global-mean-T drift column. Three of four ICs
previously reported fabricated 360-day survivals.

## 8b. MEASURED floors on this corpus — and what they cost the experiment

Computed 2026-10-05 (`nf_floors.py`, climatology fit on train 2009-2018, scored on test 2020,
same free band / vortex index / IC offsets as `gate_paradis.py`):

| metric | **nudge-free floor** | published (nudged) |
|---|---|---|
| sigma_ratio d30 | **0.985** | 0.811 |
| sigma_ratio d90 / d180 / d270 / d359 | **1.013 / 1.013 / 1.007 / 1.015** | 0.811 |
| **vortex correlation** | **0.998** | 0.727 |
| vortex std ratio | 0.988 | 1.00 |

**A vortex floor of 0.998 means climatology predicts this system's vortex almost perfectly.** The
nudge-free attractor is essentially purely seasonally forced: seasonal variance fraction is 0.44
(median) here against 0.09 in the nudged corpus. There is almost no internal vortex variability to
learn.

Consequences, stated plainly:
- "Beat climatology on the vortex" is **not a meaningful target on this corpus** (ceiling 0.998).
- The vortex and sigma_ratio numbers from this arm are **NOT comparable** to the nudged line. Any
  cross-corpus comparison of those metrics is apples to oranges.
- What this arm CAN still settle is the primary prediction: **does the tropospheric runaway
  disappear when every level is state-determined?** That remains a clean test of section 1.
- It also means the nudge-free system is a materially EASIER problem, which is consistent with the
  observed val_loss (0.01169 at step 19.9k, already below both nudged runs' FINAL best of 0.01231
  and 0.01222) and is a reason to discount that comparison, not to celebrate it.

**Implication for the programme.** The nudged corpus is only unlearnable BELOW 150 hPa; levels 9-62
are state-determined (outside the tropics there is no nudging at all). So the realistic circulation
AND the QBO can be kept by excluding the unlearnable layer rather than by changing the corpus --
which is the boundary-forced arm (`/data/JCM_BOUNDARY_FORCED_SPEC.md`). On this evidence that arm
is the better production bet; the nudge-free arm is a DIAGNOSTIC that settles the cause.

## 9. Cost and status

Corpus generation ~11 h (16 years + a leap-year regeneration), conversion ~85 min,
training ESTIMATE ~12 h / ~72 GPU-h from the baseline's measured 0.706-0.720 s/step.

**Defects found and fixed while building this arm** (each would have silently corrupted the run):
1. `run.total_time=365` hardcoded -> leap years 2012/2016/2020 ran a day short, leaving **30 h
   seams**. Samples straddling them would have paired frames a day apart while labelled 6 h apart.
   Caught by `verify_nf.py`, which blocked the launch. Regenerated 2012-2021 with computed lengths.
2. `experiment_name` left as `jcm_paradis_free_big` -> would have written into the completed
   baseline's directory and corrupted its 129-checkpoint ladder.
3. The gate needs `year+1` in the **zarr**, not just as netCDF; conversion stopped at 2020.
4. ~30 h of idle GPUs: the conversion/training chain was killed to fix defect 1 and not rebuilt.
