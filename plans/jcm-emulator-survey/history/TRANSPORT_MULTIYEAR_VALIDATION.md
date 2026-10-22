# Offline multi-year tracer transport — validation

**Status 2026-09-22.** First execution of the SPEC §7 evaluation protocol and the first multi-year
test of anything in this project. Companion to `/data/TRACER_TRANSPORT_EMULATOR_SPEC.md` and
`/data/TRACER_TRANSPORT_EMULATOR_RESULTS.md`, whose §5.5 ("run the §7 protocol — none of it has
been run") and §5.6 ("multi-year — entirely untested") this closes **for the transport half only**.

Code (new, reusable): `/data/jcm_tracer_emulator/m0_pipeline_check.py`, `m1_offline_transport.py`,
`m2_report.py`. Outputs: `m1_offline_20yr.json`, `m1_offline_5yr_K8.json`,
`m1_offline_5yr_lev.json` (+ `.log`, + `.partial` written at every truth read).

---

## 1. Headline

| | result |
|---|---|
| **Does the HORIZONTAL operator over-diffuse over years?** | **Not through year 1.** Measured within levels — the only thing it modifies — its structure at one year is **0.71–1.53×** truth's across ten tracers (mean \|ln ratio\| **0.17**, flat from d100). Past year 2 the metric loses power as truth goes uniform within levels, so this is bounded at one year, not five (§3b, §10.10). |
| **Does the 3-D field track truth?** | **No — it retains 2.3–3.9× truth's structure at yr 1 and 3.7–26× at yr 5.** The entire gap is vertical: the operator is horizontal-only, so it holds **96–100 %** of each stratospheric pulse above 100 hPa while truth drains `pulse_4` to **26 %** by yr 1 and **10 %** by yr 2. |
| **Mass** | The operator alone is **unusable**: −59 % to +164 % per tracer over 5 years. With the archive's own global proportional fixer it is exact by construction. Truth drifts 1.0–1.5 %. |
| **Box vs Gaussian twins** (SPEC §7's discriminating test) | **Passes in the horizontal.** Within-level box/Gaussian ratios at yr 1: truth 0.79–1.37, analytic 0.92–1.38, matching site 1 to two decimals. The 3-D divergence is the vertical again, not the limiter. |
| **Positivity / max principle** | **Clean.** 0 negative cells and 0 new maxima in **8,763 consecutive steps**, both drivers. |
| **Learned correction head** | **Neutral on the 3-D metric, clearly HARMFUL on the horizontal one** — the metric that isolates what it actually changes. Mean \|ln ratio\| at yr 1: analytic **0.170**, learned **0.426**; **0 of 10** tracers better over months 3–12 (indistinguishable after year 2, once the metric loses power). On the one held-out site it **over-diffuses by 2.3×** at year 1. |
| **Verdict** | The transport half is **not multi-year-viable as it stands, and the blocker is the missing vertical remap, not horizontal diffusion.** The analytic horizontal operator is sound and accurate; the learned correction on top of it is not, over years. |

---

## 2. What was actually run

`m1_offline_transport.py` drives the real dinosaur semi-Lagrangian operator with the **archive's own
winds** (no state head, so no dynamics instability), transporting the ten pulse tracers
autoregressively from an initial condition at the first archived frame, chaining contiguous 10-day
chunks. Run A applies no source, no sink and no lid to the pulses, so pure transport is the whole
of their truth evolution and the archive is an exact target.

| run | period | steps | drivers | wall |
|---|---|---|---|---|
| `m1_offline_20yr` | 1990-01-01, 20 yr target | **8,763 done** (6.0 yr) | analytic, analytic+fixer, learned, learned+fixer | ~3.1 min/sim-yr |
| `m1_offline_5yr_K8` | 1990-01-01 → 1995-01-01 | **7,303 (complete)** | analytic, analytic+fixer, **8 sub-steps per 6 h** | 9.9 min |
| `m1_offline_5yr_lev` | 1990-01-01 → 1995-01-01 | **7,303 (complete)** | analytic, learned, + within-level and f100 metrics | 13.1 min |

Only the 20-year run was still going in tmux (`eval_m1_transport`) when this was written; its
`.partial` JSON is rewritten at every truth read, so nothing here depends on it finishing. **Every
number quoted below comes from a completed 5-year span** except where explicitly labelled a
snapshot (§10.9).

Four drivers on identical frames and an identical IC:

- **analytic** — midpoint-wind departure points, cubic Lagrange gather, quasi-monotone limiter.
- **analytic_fix** — the same, plus the archive's own global proportional mass fixer
  (`jcm_strat/main.py:70`). SPEC 2.4 is right that the archive's 0.2 %/19 yr conservation is the
  *fixer's*, not the operator's; this driver makes that explicit rather than inheriting it.
- **learned** / **learned_fix** — analytic departure point plus the trained levels-as-channels
  correction head, `b5_lc_ck180.npz` (1,456,701 params).

**Chain integrity.** Every chunk seam is checked at run time to be exactly 6 h and the chain stops
if it is not. 8,763 steps were chained with zero seam breaks, across chunk *and* year boundaries
(the year-end chunks hold 20 frames, not 40; `p11a_20yr/` day numbering is continuous).

**Pipeline verified before any multi-year claim.** `m0_pipeline_check.py` runs the same `carts` and
`remap` code single-step on 78 held-out 1996 frames: **learned − analytic = +0.0462 on holdout
site 4, 77/78 frames beaten**, against SPEC 9.3's +0.0473 / 468 of 468. The rollout is driven by
the same operator that produced the published numbers.

**The truth curve was reproduced, not assumed.** The std/mean metric here reproduces the SPEC 2.2
table to three decimals on the archive itself:

| std/mean | d100 | yr1 | yr2 | yr5 | yr10 |
|---|---|---|---|---|---|
| `pulse_1` measured here | 1.414 | 0.590 | 0.436 | 0.312 | 0.125 |
| `pulse_1` SPEC 2.2 | 1.41 | 0.59 | 0.44 | 0.31 | 0.125 |
| `pulse_4` measured here | 1.739 | 0.881 | 0.348 | 0.061 | 0.019 |
| `pulse_4` SPEC 2.2 | 1.74 | 0.88 | 0.35 | 0.061 | 0.019 |

Five of the six SPEC 2.2 columns, to the precision the table is quoted at.

---

## 3. (a) Structure decay — the headline

std/mean of the full 3-D field, unweighted, on the same frames for model and truth.

### `pulse_1` (45 hPa)

| | d100 | yr1 | yr2 | yr5 |
|---|---|---|---|---|
| truth | 1.414 | 0.590 | 0.436 | 0.312 |
| analytic | 2.530 | 1.650 | 1.612 | 1.616 |
| learned | 2.468 | 1.632 | 1.537 | 1.503 |
| **ratio analytic / truth** | **1.79** | **2.80** | **3.70** | **5.18** |

### `pulse_4` (3 hPa, the held-out training site)

| | d100 | yr1 | yr2 | yr5 |
|---|---|---|---|---|
| truth | 1.739 | 0.881 | 0.348 | 0.061 |
| analytic | 2.603 | 2.026 | 1.711 | 1.564 |
| learned | 2.303 | 1.827 | 1.784 | 1.760 |
| **ratio analytic / truth** | **1.50** | **2.30** | **4.91** | **25.71** |

All ten tracers, ratio analytic/truth: **yr1 1.69–3.90, yr2 2.60–6.15, yr5 3.72–26.97.**

**The model's structure stops decaying.** `pulse_1` sits at 1.61 ± 0.01 from year 2 to year 5;
`pulse_4` at 1.56–1.71. Truth keeps mixing; the emulator reaches a fixed point and stays there.
The gap is not a slowly accumulating bias — it is the difference between a process that continues
and one that has stopped.

**But this 3-D number is not a diffusion measurement.** std/mean over the full field mixes
horizontal mixing with vertical redistribution, and a horizontal-only operator cannot do the
second at all. §3b separates them, and the separation is what the result actually is.

**What survives regardless:** SPEC §7's framing — "over-diffusion is the failure mode, not mass
drift" — is wrong on both counts at multi-year range. The operator does not over-diffuse, and mass
is the worst-behaved of the three quantities.

---

## 3b. Decomposing it: the horizontal branch is accurate, the vertical is absent

`m1_offline_5yr_lev.json` adds two metrics computed on the same frames:

- **within-level structure** — std/mean inside each level, averaged over levels weighted by that
  level's tracer mass. This isolates horizontal mixing.
- **f100** — the fraction of tracer mass above 100 hPa. This isolates vertical redistribution.

At **year 1**:

| tracer | within-level truth | analytic | ratio | f100 truth | f100 analytic |
|---|---|---|---|---|---|
| `pulse_1` | 0.208 | 0.212 | **1.02** | 13.4 % | 96.1 % |
| `pulse_1_box` | 0.285 | 0.292 | **1.02** | 14.6 % | 100.0 % |
| `pulse_2` | 0.525 | 0.733 | 1.40 | 6.2 % | 58.0 % |
| `pulse_2_box` | 0.528 | 0.807 | 1.53 | 5.9 % | 75.3 % |
| `pulse_3` | 0.921 | 0.986 | **1.07** | 10.5 % | 99.9 % |
| `pulse_3_box` | 0.960 | 0.969 | **1.01** | 9.8 % | 100.0 % |
| `pulse_4` | 0.887 | 0.627 | 0.71 | 25.9 % | 100.0 % |
| `pulse_4_box` | 0.924 | 0.702 | 0.76 | 28.1 % | 100.0 % |
| `pulse_5` | 0.126 | 0.106 | 0.84 | 6.7 % | 0.5 % |
| `pulse_5_box` | 0.100 | 0.097 | **0.97** | 6.6 % | 0.0 % |

Mean |ln(ratio)| over the ten tracers, analytic driver:

| | d40 | d100 | d200 | yr1 | yr2 | yr4 | yr5 |
|---|---|---|---|---|---|---|---|
| **3-D** std/mean | 0.517 | 0.528 | 0.704 | **0.930** | 1.369 | 1.865 | 2.012 |
| **within-level** std/mean | 0.341 | 0.207 | 0.169 | **0.170** | 0.440 | 0.985 | 1.206 |
| *smallest truth within-level value* | 0.850 | 0.436 | 0.210 | 0.100 | 0.058 | 0.037 | 0.021 |

**Through year 1 the horizontal branch does not degrade — it is ~17 % off and flat from d100 to
yr 1**, while the 3-D error doubles from 0.5 to 0.9 over the same period. All of that growth is
vertical. `pulse_4` and `pulse_4_box` are the two tracers where the operator is *more* mixed than
truth horizontally (0.71×, 0.76×), so there is no uniform under-mixing bias either.

**After year 1 the within-level agreement does degrade — but the metric stops being informative at
about the same time, and the two cannot be separated here.** The third row is why: truth's own
within-level structure collapses to 0.02–0.04 by year 5, i.e. truth is very nearly uniform *inside
every level*. At year 5 the per-tracer ratios are 0.18×, 0.42×, 0.51×, 0.54×, 1.86×, 2.86×, 4.23×,
4.79×, 5.84×, 5.88× — spread across both directions, on absolute differences of ~0.02–0.17 in a
quantity whose true value is ~0.03. A 5.9× ratio there is not five times too much mixing in any
meaningful sense; it is a small residue against a denominator that has gone to zero.

**So the defensible claim is bounded: the horizontal branch tracks truth to ~17 % through year 1,
and past year 2 this metric cannot say.** §10.10 records what would be needed to push it further.
What is *not* ambiguous at year 5 is the vertical: truth has 8.5–10.3 % of every pulse above
100 hPa (fully mixed), the operator still has 96–100 % for all six stratospheric pulses.

This is the most load-bearing result in the document, and it is good news for the operator: **the
piece that was built works; the piece that was never built is what fails.**

---

## 4. (b) Mass

Exact mass-weighted integral: hybrid `a, b` boundaries from the run's own `strat63` level table
(`level_table: strat` in the run's `.hydra/config.yaml`) × the archive's
`normalized_surface_pressure`, × Gaussian quadrature weights. Not nominal sigma, not cos(lat).
Air mass itself is constant to 1 × 10⁻⁵ over the period, so these ratios are burden ratios too.

Total tracer mass relative to the IC:

| | yr1 | yr2 | yr5 |
|---|---|---|---|
| truth `pulse_1` | 1.0124 | 1.0133 | 1.0138 |
| analytic `pulse_1` | 1.6286 | 1.7187 | **1.7415** |
| learned `pulse_1` | 1.4333 | 1.5337 | 1.5338 |
| analytic `pulse_1_box` | 2.3326 | 2.5506 | **2.6427** |
| analytic `pulse_3_box` | 0.4325 | 0.4083 | **0.4097** |
| any `*_fix` driver | 1.0000 | 1.0000 | 1.0000 |

Worst |mass − 1| over all ten tracers at year 5: **analytic 164 %, learned 96 %, truth 1.5 %.**
The error is **not a one-sided bias** — site 3 loses 59 % while site 1 gains 164 % — so it cannot
be absorbed into a constant correction; only the global fixer removes it.

Per-step this is ~1.3 × 10⁻³, exactly the magnitude SPEC 2.4 states for the bare SL scheme. **The
finding is not that the number is new; it is that over 7,300 steps it is fatal, and that the
archive's conservation genuinely belongs to the fixer.** Any product claim about offline transport
must ship the fixer, and must say so.

Truth's own +1.0–1.5 % is concentrated in the first weeks after injection, when the pulses are
sharpest; it is flat thereafter, consistent with SPEC 2.2's +0.07 %/19 yr measured from 1991.

---

## 5. The dominant error is the missing vertical remap

§3b establishes it from the model's own rollout; this section gives the truth-side magnitude.

**Fraction of tracer mass above 100 hPa, truth, measured directly from the archive:**

| | d10 | yr1 | yr2 | yr5 | yr10 | yr20 |
|---|---|---|---|---|---|---|
| `pulse_1` | 79.6 % | 13.4 % | 9.5 % | 8.8 % | 9.4 % | 9.8 % |
| `pulse_4` | 100.0 % | 25.9 % | 11.9 % | 10.2 % | 10.0 % | 9.9 % |

≈10 % of atmospheric mass lies above 100 hPa, so both pulses are **vertically well mixed within
about two years**. The horizontal-only operator moves none of this: its remap is per level and, in
the stratosphere, `b ≈ 0` makes layer mass essentially independent of surface pressure, so the
tracer stays where it was put.

**Mass-weighted centroid pressure (hPa):**

| | yr1 | yr2 | yr5 |
|---|---|---|---|
| `pulse_1` truth | 332.8 | 380.5 | 388.0 |
| `pulse_1` analytic | 40.8 | 40.3 | 40.2 |
| `pulse_4` truth | 216.5 | 349.7 | 367.3 |
| `pulse_4` analytic | 4.53 | 4.58 | 4.67 |

and the vertical spread of `pulse_4` is frozen at 0.26–0.27 decades against truth's 0.44–0.64.

**This overturns the SPEC's own priority ordering.** SPEC 6.1 finding 4 and §8.1 price the vertical
at "0.097 layers per 6 h … a few-percent term, not the critical path." That is correct *per step*
and wrong *per year*: the displacement is small but systematic, and 1,460 steps of it is the
difference between a tracer confined to 4 hPa and a tracer that has reached the troposphere.
RESULTS §5.4 already suspected this ("a systematic 2 %-per-step omission compounds over ~40,000
steps"); this measures it.

One useful control: `pulse_5` is injected at ~325 hPa, already near where truth ends up, so its
centroid error is small (367 vs 393 hPa at yr 5) — yet its 3-D structure ratio is still **5.96×**.
Its *vertical spread* is half of truth's (0.221 vs 0.393 decades) and its within-level structure is
0.84× truth's, so even for a tracer whose centroid is right, the missing vertical **mixing** — not
just the missing mean descent — is what the 3-D metric is seeing.

---

## 6. (c) Box vs Gaussian twins

`pulse_N` and `pulse_N_box` are the same nominal size at the same site, differing only in edge
sharpness. Ratio of `_box` std/mean to Gaussian std/mean:

| site | yr1 truth / anal / lrn | yr2 truth / anal / lrn | yr5 truth / anal / lrn |
|---|---|---|---|
| 1 | 1.082 / 1.432 / 1.539 | 0.999 / 1.391 / 1.497 | 0.998 / 1.382 / 1.510 |
| 2 | 1.049 / 1.208 / 1.253 | 1.031 / 1.212 / 1.300 | 1.026 / 1.236 / 1.398 |
| 3 | 1.106 / 1.440 / 1.416 | 1.076 / 1.429 / 1.338 | 1.081 / 1.435 / 1.370 |
| 4 | 1.040 / 1.312 / 1.287 | 1.049 / 1.312 / 1.264 | 1.257 / 1.319 / 1.259 |
| 5 | 1.014 / 1.525 / 1.514 | 1.007 / 1.525 / 1.516 | 1.004 / 1.526 / 1.526 |

On the 3-D metric truth erases the twins' difference — by year 2 its box/Gaussian ratio is
1.00–1.08 at every site — while the operator holds 1.21–1.53 for five years, flat.

**But the same decomposition applies here, and it flips the reading.** Taking the ratio of the
*within-level* structures at year 1:

| site | truth | analytic | learned |
|---|---|---|---|
| 1 | 1.37 | **1.38** | 1.38 |
| 2 | 1.01 | 1.10 | 1.17 |
| 3 | 1.04 | 0.98 | 1.07 |
| 4 | 1.04 | 1.12 | 0.89 |
| 5 | 0.79 | 0.92 | 0.83 |

**In the horizontal the operator transports the twins the way truth does** — site 1 to two decimal
places, the rest within 0.1 — which is exactly what SPEC §7 says this test is for ("the same `W`
must transport both … an operator cannot cheat"). **The operator passes the discriminating test.**

The 3-D divergence is again the vertical: truth's twins converge because *both* are mixed down into
the troposphere towards uniformity, and the operator's twins cannot follow. There is **no** sign
that the quasi-monotone limiter mishandles the sharp edge — the limiter's own invariants are clean
(§7) and the horizontal twin ratios track truth.

Site 4 at yr 5 is the one 3-D row not to read: both twins are near-uniform there (0.061 / 0.076),
so the ratio is noise.

---

## 7. (d) Positivity and the maximum principle

Checked at **every** step, not at the diagnostic stride.

| driver | steps with q < 0 | worst q | steps with a new max | worst relative rise |
|---|---|---|---|---|
| analytic | **0** | 0.0 | **0** | 0.0 |
| learned | **0** | 0.0 | **0** | 0.0 |
| analytic_fix | 0 | 0.0 | 8,527 | 3.6 × 10⁻² |
| learned_fix | 0 | 0.0 | 8,569 | 6.6 × 10⁻² |

*(counts at step 8,763 — six simulated years — of the 20-year run; the completed 5-year K = 8 run
gives the same zeros over its own 7,303 steps.)*

Over 8,763 consecutive steps the gather plus quasi-monotone limiter produced **no negative cell and
no new extremum**, with or without the learned trajectory correction. This is the one §7 test the
transport half passes outright, and it is the argument against learning the remap (SPEC 3.6).

The `_fix` rows are the **fixer**, not the gather: a global proportional rescale by a factor > 1
raises the field maximum by construction. Reported separately so the operator's own record is not
contaminated.

---

## 8. Two levers priced: sub-stepping is closed, the learned head is a liability

**Sub-stepping does nothing over years.** A full 5-year run with **K = 8** sub-steps per 6 h
(`m1_offline_5yr_K8.json`), winds linearly interpolated to each sub-step midpoint:

| std/mean | yr1 | yr2 | yr4 | yr5 |
|---|---|---|---|---|
| `pulse_1` K=1 | 1.650 | 1.612 | 1.615 | 1.616 |
| `pulse_1` K=8 | 1.677 | 1.605 | 1.608 | 1.609 |
| `pulse_4` K=1 | 2.026 | 1.711 | 1.571 | 1.564 |
| `pulse_4` K=8 | 2.019 | 1.718 | 1.564 | 1.552 |

Differences are < 2 % and not signed consistently; truth is 0.312 and 0.061. SPEC 6.1 found
sub-stepping buys +0.017 at a single step and plateaus by K = 4; **it buys nothing at all over five
years.** The lever is closed at both ends.

**The learned correction head, on the 3-D metric, looks neutral.** Aggregated over all ten
tracers, mean |ln(model structure / truth structure)| — lower is better:

| | d100 | yr1 | yr2 | yr5 |
|---|---|---|---|---|
| analytic | 0.528 | 0.930 | 1.369 | 2.012 |
| learned | 0.517 | 0.929 | 1.387 | 2.038 |
| tracers where learned is better | 4/10 | 6/10 | 4/10 | 4/10 |

and on mass, mean |ln(mass ratio)|: analytic 0.485 vs learned 0.493 at yr 5, 5 of 10 tracers
better. Per tracer the head is not small — it moves `pulse_4`'s year-1 ratio from 2.30 to 2.07 (a
real improvement on its own holdout site) and `pulse_1_box`'s from 3.70 to 3.94 (a real
degradation) — but the signs do not agree across tracers and they do not persist: the head helps
`pulse_4` at yr 1 and hurts it at yr 5 (25.71 → 28.93).

**On the metric that isolates what the head actually changes, it is not neutral — it is harmful.**
The head modifies horizontal departure points and nothing else, so the within-level structure of
§3b is the metric it should be judged on. Mean |ln(model/truth)| over ten tracers:

| | d40 | d100 | d200 | yr1 |
|---|---|---|---|---|
| analytic | 0.341 | 0.207 | 0.169 | **0.170** |
| learned | 0.344 | 0.260 | 0.295 | **0.426** |
| tracers where learned is better | 6/10 | 0/10 | 1/10 | **0/10** |

Between roughly month 3 and year 1 the learned head is worse on **every one of the ten tracers**,
with the gap widening monotonically. The direction is not uniform, which is why the 3-D metric
cancelled it out: on sites 1, 2, 3 and 5 the head *retains* too much structure (1.13–1.67× truth at
yr 1), while on site 4 it *over-diffuses* (0.43× and 0.37×, versus the analytic operator's 0.71×
and 0.76×).

**The window matters and must be stated.** Carried to five years the gap narrows and then closes:

| | yr1 | yr2 | yr4 | yr5 |
|---|---|---|---|---|
| analytic | **0.170** | 0.440 | 0.985 | 1.206 |
| learned | **0.426** | 0.689 | 1.080 | 1.156 |
| tracers where learned is better | **0/10** | 3/10 | 4/10 | 6/10 |

By year 5 the learned head is nominally *ahead*. I do not read that as a recovery: by then truth's
within-level structure is 0.02–0.04 and the metric is uninformative (§3b). **The honest statement
is that the head is clearly harmful over the horizon where the metric has power — months 3 to 12,
0/10 tracers — and indistinguishable thereafter.**

**Site 4 is the only site held out of the head's training loss**, and it is where the head does
most damage: a 2.3× over-diffusion after one year, from a correction that beats the analytic
operator on 77 of 78 single steps at that same site. This is the sharpest statement the run
supports — *single-step skill on a held-out site did not survive autoregressive composition.*

**Answer to the question posed: neutral on the 3-D metric, harmful on the horizontal one, and the
horizontal one is the honest test.** It does not destabilise anything — positivity and the maximum
principle are untouched — but there is no evidence it should be carried into a multi-year product,
and there is now a measurement that says it should not be.

*Caveats that must travel with this:* one checkpoint (`b5_lc_ck180`), one holdout split, one
realisation. Nine of the ten tracers were in the head's training loss, so only site 4 is a clean
generalisation test — and it happens to be the worst case. The project's own rule (score every
checkpoint on the multi-year metric, take the peak) has not been applied; `m1_offline_transport.py`
takes `--ckpt`, so that sweep is now cheap.

---

## 9. R² against truth, and why it stops being the metric

Gauss-weighted, climatology-referenced: `1 − MSE / var(truth)`.

| `pulse_1` | d80 | d100 | d160 | yr1 |
|---|---|---|---|---|
| analytic | −17.7 | −25.6 | −68.4 | −579.8 |
| analytic_fix | −8.2 | −11.2 | −27.6 | −202.9 |
| learned_fix | −6.7 | −9.3 | −23.5 | −175.3 |

Pattern agreement with truth is gone within ~2–3 weeks and then collapses. Two things inflate these
numbers and both must travel with them: (i) the un-fixed drivers carry a 30–160 % mass error
straight into the MSE, which is why `_fix` is 2–3× better; (ii) this is **not** the project's usual
R², which is persistence-referenced (`1 − MSE/MSE_persistence`, b2/b5). At a single step the two
differ enormously — `m0_pipeline_check.py` measures 0.855 persistence-referenced against 0.9999
variance-referenced on the same frames. Do not compare these numbers with SPEC 9.3's 0.8195/0.8668.

Point-wise agreement after a year was never a realistic target for a chaotic advection problem;
the structure, mass and vertical-distribution statistics are the ones that carry information at
these horizons, which is what SPEC §7 says.

---

## 10. What was NOT tested

Stated plainly, because an honest boundary is worth more than an extrapolation.

1. **The vertical remap itself.** It does not exist; this run measures its *absence*. Nothing here
   says the SPEC 8.1 route through archived ω will work.
2. **`sai`, `src_*`, `n2o`, `cfc11`, `aoa*`.** Only the ten pulses were transported, because they
   are the only tracers with no analytic source/sink that this driver would have to reproduce.
   The continuously sourced tracers carry signal for all 20 years (SPEC 2.2) and are untested.
3. **Age of air, polar vortex, SSW frequency.** Three of the six §7 tests. Age of air in particular
   is the scientific point of the stratosphere programme and is exactly the diagnostic the missing
   vertical would corrupt.
4. **Synthetic ICs** (delta, checkerboard, constant; "constants stay constant, row sums = 1").
   Untested here; cheap, and should be done.
5. **All five holdout sites for the learned head.** Every learned number still inherits the
   holdout-site-4 training split (RESULTS §5.3). The multi-year comparison above transports all ten
   pulses, but nine of them were in the head's training loss, so only `pulse_4` / `pulse_4_box` are
   a clean generalisation test over years.
6. **Other checkpoints.** One checkpoint (`b5_lc_ck180`). The project's own non-negotiable rule is
   to score every checkpoint on the multi-year metric and take the peak; that sweep was not run.
   The verdict in §8 is a verdict on this checkpoint, not on learned trajectory corrections
   in general.
7. **ERA5-driven transport.** RESULTS §4c offers offline transport under ERA5 winds as a product;
   only archive winds were used.
8. **One realisation, one IC, one injection date.** SPEC 8.3. Every number above is a single
   trajectory.
9. **Years beyond the snapshot below, and the within-level decomposition beyond year 2.** Both
   runs were still going when this was written (the 5-year per-level run has since completed and
   is folded into §3b and §8). Snapshots from the 20-year run for the record: at **year 6** the
   3-D ratios are `pulse_1` 6.1× and `pulse_4` 32.5×; at **year 10** (the last mark seen, and a
   SPEC 2.2 column) truth is 0.125 and 0.019 against the operator's 1.616 and 1.560 — **12.9×**
   and **82×** — with positivity/max-principle violations still exactly zero at step 14,607.
   The model curve is pinned at its floor (`pulse_1` 1.61–1.62 from year 2 to year 10) while truth
   decays by a further factor of 3.5. The trend of §3 continues and the model
   curve stays pinned at its floor while truth keeps decaying. **For the completed record, re-run
   `python m2_report.py --json m1_offline_20yr.json`** (and `m1_offline_5yr_lev.json`) once the
   tmux sessions `eval_m1_transport` / `eval_m1_lev` have finished — the tables in §3, §4, §6
   and §8 regenerate from it verbatim.
10. **Whether the horizontal branch is still accurate past year 1 remains unresolved — now with
   the data in hand rather than for want of it.** The 5-year per-level run completed (§3b): the
   within-level error grows 0.170 → 1.206 from yr 1 to yr 5, but truth's within-level structure
   falls to 0.02–0.04 over the same span, so the log-ratio loses its meaning at roughly the same
   rate. A better-conditioned horizontal diagnostic — equivalent length, a spectral slope at fixed
   levels, or tracer–tracer correlation — would be needed to decide it, and was not built. **Do not
   read §3b's year-5 ratios as a measurement of horizontal diffusion.**

---

## 11. Bugs and notes found in existing code

- **No bug found in the operator or the head.** `m0_pipeline_check.py` reproduces SPEC 9.3's
  single-step gain (+0.0462 vs +0.0473, 77/78 frames) through the same code path the rollout uses.
- **`b0_reference_operator.gauss_weights` and `scripts/tracer_budget.layer_dp` are correct** and
  were reused unchanged. `tracer_budget.py` needs `jcm_strat.levels.install("strat")` first — the
  run records `level_table: strat` in `.hydra/config.yaml` and the resulting mid-layer pressures
  match `b2_frames_big.npz`'s `p_lev` exactly. Skipping that step silently gives the wrong `dp`.
- **A documented priority is wrong, not a code bug:** SPEC 6.1/8.1's "the missing vertical is a
  few-percent term, not the critical path" holds per step and fails per year (§5). The retraction
  belongs in SPEC §7's corrections table.
- **Method note for whoever runs this next:** host-side float64 reductions, not the GPU, are the
  bottleneck (~3.5 min per simulated year, 1.2 cores, GPU at ~1 %). I/O is negligible with one
  prefetch thread — 0.1 min of waiting across 8,763 steps. Do not enable `jax_enable_x64` to get
  float64 mass: it promotes the departure-point solve out of the float32 the head was trained in.

---

## 12. Verdict and what to do next

**The transport half is not multi-year-viable as it stands — but the reason is a missing term, not
a broken operator.** The analytic horizontal branch holds its structure to ~17 % of truth over a
full year with exact positivity and an exact maximum principle. What kills it is the vertical,
which does not exist, and mass, which needs the archive's fixer. The learned correction, judged on
the branch it actually modifies, makes the horizontal worse.

Ordered by what unblocks the most:

1. **Build the vertical remap.** It is no longer a §5.4 nice-to-have; it is the blocker. SPEC 8.1's
   gate stands — reproduce the archived ω before trusting any σ̇ — but the cost of not having it is
   now measured: a factor 5–26 on structure and a tracer that never leaves its injection layer.
2. **Ship the mass fixer with the operator, always.** −59 % to +164 % over five years without it.
   It is one line and it is what the archive does.
3. **Do not spend anything more on sub-stepping.** Zero multi-year effect, measured over 5 years
   at K = 8.
4. **Re-price the learned transport head before shipping it.** Over months 3–12 it is worse than
   the bare analytic operator on the within-level metric on 10 of 10 tracers, and worst on the only
   site held out of its training. Cheapest next step: run `m1_offline_transport.py --ckpt` over the
   saved `b5_lc_ck*.npz` checkpoints for one year each (~3.5 min apiece) and see whether any
   checkpoint is multi-year-neutral, or whether the single-step objective is simply the wrong one.
   Rollout-horizon training for the transport head is the obvious alternative and has never been
   tried — only the *state* head was trained that way.
5. **Then re-run this protocol**, plus the four §7 tests listed in §10 that were not run —
   especially age of air, which is the one the vertical directly determines.

The cheap good news is real: 8,763 autoregressive steps with zero positivity or
maximum-principle violations, a chain that ran without a seam break, ~3.5 minutes of wall clock
per simulated year, and a horizontal operator that is measurably *not* the problem. The instrument
works; the physics it carries is one term short.
