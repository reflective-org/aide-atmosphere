# RESULTS — jcm-nudgefree

# 03 — Nudge-free arm and the filter sweep (Oct 5 – Oct 7)

## Hypothesis (registered in advance)

`/data/JCM_NUDGEFREE_HYPOTHESIS.md` (10-05). The claim: the free-running failure is an **ill-posed learning problem**. Levels 0–8 are 63 %-per-step
ERA5 and not a function of state, so no emulator can learn them. The test is to emulate a nudge-free JCM, where every level is
state-determined. The **primary falsifier**: "if the nudge-free emulator ALSO runs away, the hypothesis is WRONG."

## Corpus route: can a free JCM produce a QBO at all?

Hines GWD sweep, T63 L95, 10 yr each, `nudging.enabled=false`, `held_suarez.qbo=null`, ~2 h and 478 GB per run (`PARADIS_JCM_RESULTS.md` §6).

| Hines rms launch wind | eq. ū 30 hPa mean | deseasonalised std |
|---|---|---|
| 1.0 | +0.7 | 0.3 |
| 2.0 | −0.2 | 1.2 |
| 3.0 | −0.0 | 5.0 |
| ERA5 | — | 17 |

- **No QBO at any amplitude.** Amplitude gives stronger noise, not an oscillation.
- Without nudging the troposphere runs at about half strength: the 500 hPa jet is +9.6 vs +17.2 m/s. Waves are weak, so there is no QBO driving.
- A dry model lacks convective tropical waves. **A JCM run can track reality or run free, not both.**

## Emulator: `config_jcm_nf.yaml`, run `jcm_paradis_nudgefree`

- Corpus `/data/jcm_nf_c1`. Train 2009–18, val 2019, test 2020; 60k steps (`run_nf2.log`).
- Floors recomputed on this corpus (`nf_floors.py`): σ-ratio ~1.01, **vortex 0.998**. With no nudging, climatology is near-perfect on vortex
  timing, so the nudged floors are not comparable.
- Gate: `run_gate_nf.sh`, fixed gate. **Incomplete**: no SUMMARY line, and ep88/ep118 were never scored.

| ckpt | result (`gate_nf_g*.log`) |
|---|---|
| ep0 | 360/360 d; vortex +0.415/−0.460; anomR90 7.3/8.6 |
| ep29 | IC0: vortex −0.384, anomR90 12.3 |
| ep59 | **died 51 d and 69 d** on 2/4 ICs; survivors anomR90 18.5/22.3 |
| ep147 | IC0 **died d56** |

**Verdict: the hypothesis is falsified as registered.** The emulator runs away on a nudge-free corpus too, and anomR *grows* with
training (7 → 12 → 22). The nudged troposphere did stay bounded: ep147 maxU over levels 0–8 is 13.2 m/s vs 187.5 in the free band
(`fsweep_a0.log`). So ill-posedness explained the 420 m/s tropospheric runaway in 02/B3, but not the failure itself.
This verdict is written down for the first time here; the source docs only imply it.

## Filter sweep: `/data/JCM_FILTER_SWEEP_RESULT.md`

nf ep147, test 2020, **2 ICs**. Logs: `fsweep_a{0,1,3,10,30,100}.log` and `fsweep_p{1,2,4,8}.log`.

- **Doc verdict: the filter fixes death, not wrongness.** α ≥ 3 at p = 8 survives 360 d, but anomR never drops below ~10. Only p = 1 brings
  σ-ratio near 1 (sr180 1.088), at the cost of sr30 0.213.
- The excess variance is **planetary (l < 10)**. Damping there would destroy the circulation.
- Caveats the doc omits:
  - Its tables are IC0 only. IC364 survives 360 d *unfiltered*, with anomR90 13.3.
  - The 2-IC median vortex is +0.609 at α=0 and −0.076 at α=3, so the filter *lowers* vortex skill.
  - p=4 IC364 died at d64 despite the filter.
  - The gate printout shows the nudged floors; the doc uses the nf floors.
- **Conclusion recorded in memory:** always include a filter, stop tuning corpora and formulations, and treat DAgger as the remaining lever.
  That led to 04.

