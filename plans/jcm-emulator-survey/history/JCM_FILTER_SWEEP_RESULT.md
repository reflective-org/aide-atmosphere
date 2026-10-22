# Spectral filter sweep — the instability is at PLANETARY scales, not the grid scale

**2026-10-07.** Run on the nudge-free arm's final checkpoint (`jcm_paradis_nudgefree` epoch 147),
which free-runs to death at day 56. Every number from `gate_paradis.py --filter-alpha/--filter-p`;
logs `fsweep_a*.log`, `fsweep_p*.log`. Floors are the NUDGE-FREE climatology floors
(`nf_floors.py`), not the nudged ones: sigma_ratio ~1.01, vortex **0.998**, vx std 0.988, anomR 0.

Filter applied to the state after every step: `coeffs *= exp(-alpha * (l/lmax)^p)`, lmax 96.
`alpha=0` reproduces the unfiltered control exactly (day 56), so any movement is the filter.

---

## 1. Amplitude sweep (p=8) — the filter fixes DEATH and nothing else

| alpha | IC 0 survival | sr180 | anomR90 | vortex |
|---|---|---|---|---|
| **0** | **56 d died** | - | - | +0.758 |
| 1 | 56 d died | - | - | +0.774 |
| **3** | **360 d ok** | 2.857 | 14.59 | -0.507 |
| 10 | 360 d ok | 2.877 | 15.38 | -0.460 |
| 30 | 360 d ok | 3.001 | 15.54 | -0.434 |
| 100 | 360 d ok | 2.935 | 14.93 | -0.462 |

- **Survival is fixed**: alpha >= 3 turns day-56 death into a full 360 days. This reproduces the
  campaign's day-25 -> day-369 filter result at PARADIS scale and confirms grid-scale numerical
  instability is what KILLS these runs. PARADIS has no dissipation on the prognostic state (its
  "diffusion" is a learned latent-space module inside the ADR layer), while the dycore it emulates
  applies per-level hyperdiffusion every 12-min step.
- **Nothing else improves.** sigma_ratio stays ~2.9-3.5 vs a floor of 1.01; anomR stays ~15 vs 0;
  the vortex stays ANTI-correlated vs a floor of +0.998.
- **It saturates immediately.** alpha 3 -> 100 is a 33x stronger filter and is indistinguishable.
  With p=8 the filter only bites near l=lmax, so once the grid scale is damped the death stops --
  meaning the excess variance is NOT at the grid scale.

## 2. Order sweep (alpha=10) — widening the damped band does not help either

| p | cutoff l_c | sr30 | sr180 | **anomR90** | vortex |
|---|---|---|---|---|---|
| 1 | ~10 | **0.213** | **1.088** | **10.0** | -0.512 |
| 2 | ~30 | 0.753 | 3.217 | 19.4 | -0.707 |
| 4 | ~54 | 0.948 | 3.138 | 16.0 | -0.536 |
| 8 | ~72 | 0.987 | 2.877 | 15.4 | -0.460 |

`l_c = lmax * alpha^(-1/p)` is the wavenumber where the filter reaches e^-1.

- The ONLY setting that brings sigma_ratio near its floor (p=1) does so by crushing everything
  above wavenumber 10, which destroys short-lead skill (sr30 **0.213**) -- and it STILL leaves
  anomR at 10 and the vortex anti-correlated.
- **anomR never falls below ~10 at any setting**, across damping from l_c=72 down to l_c=10.

## 3. Conclusion

**The excess variance is at PLANETARY wavenumbers (l < 10)**, where dissipation cannot be applied
without destroying the jets and planetary waves that constitute the circulation. The learned
operator is unstable there. A spectral filter can convert a catastrophic divergence into a bounded
but WRONG model; it cannot make the model right.

This independently reproduces the v4 diagnosis from the CESM line -- *planetary runaway at k=1-4
alongside eddy-scale collapse* -- on a different corpus and a different model line.

**Severity.** The nudge-free system's vortex is 99.8% predictable from climatology alone. The
emulator scores -0.5 on it. It fails at a nearly trivial task, because ~10-20x spurious internal
variance swamps the seasonal signal that is almost the entire system.

## 4. What this does and does not settle

- **Settles:** missing scale-selective dissipation explains why free runs DIE. That is a real,
  cheap, structural fix and should be in any future run (alpha ~3-10, p=8 costs nothing measurable).
- **Settles:** it does NOT explain why the models are WRONG. Three corpora (nudged 12 yr, anomaly
  formulation, nudge-free 10 yr), a scale-up, and now a two-dimensional filter sweep have all left
  the same residue.
- **Does NOT settle:** whether the planetary instability is reachable by training. The single-step
  operator is excellent and FLAT on-manifold (relative error 0.14-0.20, day 30 to day 300) and
  4.8-7x worse at the states the model visits. The model has never been supervised where it goes.

## 5. The one evidenced lever left

**Dycore-in-the-loop (DAgger) supervision.** It is the only learned intervention that has ever
worked in this campaign (0.67 vs 6.33 divergence events per 1000 steps against a matched control,
identical across every seed). It attacks the exposure gap directly, and the nudge-free corpus makes
it clean: the dycore config and the data generator are the SAME operator, so "what should happen
from THIS state?" is well-posed at any drift. The contraction signal is already measured -- from
the emulator's own over-energetic day-180 state the dycore's answer is **dissipate**
(12.80 -> 12.71 median |u|).

Target fidelity is validated: outside 60 deg the dycore reproduces the nf archive's 6-h step to
0.6-3.3% of persistence. The whole-globe 8-12% figure is the `u/cos_lat` polar artifact; keep the
cos(phi) area weighting and it is suppressed ~10x. See `reference_jcm_dycore_roundtrip_polar`.

**If DAgger also fails to pull anomR toward 1, the honest conclusion is that deterministic
autoregressive emulation of this system does not work**, and the question becomes whether a
different formulation (stochastic, or hard-constrained to a balanced manifold) is required.
