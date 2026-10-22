# RESULTS — jcm-small-head-stability

# 01 — Small-head state-emulator campaign (Sep 17 – Oct 1)

**Code and logs:** `/data/jcm_tracer_emulator/`
- x-series: `x1`–`x6`, `x11_rung1.py`, `x14_gate_multiic.py`, `x15_verify_balance.py`, `x21`/`x23`/`x24`
- y-series: `y1_*`, `y3_*`, `y5_*`
- protocol: `OVERNIGHT_PROTOCOL.md`

The state head (957 k params) plus analytic PK/sponge/nudge/QBO terms began as the "dynamics" half of the hybrid transport emulator
(05). The rolled-up summary is `/data/PARADIS_JCM_RESULTS.md` §7. It overstates distillation; see A7.

**Gate:** test year 1996, 360-d free runs. Measured floors: climatology vortex corr **0.727**, σ-ratio **0.811**.
Error saturation is ~16 m/s when compared on the same day-of-year across years. RMSE discriminates out to ~day 60.

| # | experiment | result | verdict |
|---|---|---|---|
| A0 | Longer training horizon (K=20) | Day-7 RMSE 8.24, but both heads explode in a 90-d run. The stable window is ~4–5× the training horizon (`b11_long_rollout.json`; SPEC §9.9) | the §4b claim is retracted |
| A1 | Diagnose the blow-up | Zonal power at k = 49–96 goes 1.4× → 1.1e5× truth by day 20, before the planetary scales move. JCM has spectral truncation + hyperdiffusion every 12-min step; the emulator had neither | **grid-scale numerical instability** |
| A2 | Rung 1: exponential filter *in the training loop*, plus the PK season fix | Median survival: none 25 d, o8 62, o4 ~85, o2 ~200 (6 seeds span 130–369). Filter **form** does not matter (ECHAM hyperdiffusion ≈ tuned exp filter ≈ JCM per-level profile, 3 seeds each). 18 yr of data + 3× steps → 44–223 d (worse). Energy-budget loss: no effect | filter width is the lever. "25 → 369" is the best seed, not the median |
| A3 | Spectral ζ/δ prognostics | State: 231 vs 195 d control (1.18×, null); ζ/δ head 90 d (0.46×), d60 RMSE 33.96 vs 16.63 | negative |
| A4 | PV advect-and-invert backbone | 15× worse; it is an unconstrained energy source. Scalar SL advection of u, v, T is 43–114 % worse than persistence | negative |
| A5 | Thermal-wind balance operator (y3) | T anchored at level 8 gives **96 %** of extratropical 3-D u, verified over 6 years. Heating-response operator ~90 % of the zonal-mean response at day 10. Survival 17/24 vs 3/24, **but** balance without the bound gives 0/24 (167 d) | the tanh bound did the work, not the balance |
| A6 | Anchor bound \|u − u(L8)\| ≤ 80 m/s | 24/24 full year, σ-ratio 0.92–1.09. Vortex corr **0.711 < 0.727 floor**. With B=150 the bound saturates on 99.8 % of cells at days 250–360: the model has diverged behind the clamp | **survival is invalid under any bound** |
| A7 | Dycore distillation (`x23_distill.py`) | Exposure gap measured: operator error ~2–3.6 m/s at truth states vs **27× worse** at visited states. Resets per 1000 steps 0.67 vs 6.33 against an archive-only control. **But** the kill criterion failed: day-50 operator error went 15.8 (warm start) → 22.0, reaching 534 at d300 (`x24_probe.log`). Low-lr round 2 is NaN by d200 (`x24_probe2.log`). The "typical ckpt bounded at d200" ckpt is 4.5e8 at d300 (`x24_probe3.log`) | beats a matched control; **loses to doing nothing** |
| A8 | Real dycore as a reference (y5, z8) | The 6-h step reproduces the archive to **1.3 %** of the persistence increment. A free year gives σ-ratio 1.000, vortex 0.997. Cost 3.1 h per 20 yr | dycore + learned correction saturates at a 3–4× speedup |
| A9 | Radiative heating applied as a tendency vs as a step | Does not shorten the ~2-d geostrophic adjustment, but cuts spurious wind 24–86× (0.99 → 0.023 m/s) | design rule |
| A10 | Anomaly formulation of the state head (n1–n3) | Better to day 10, explodes sooner (SPEC §9.10) | negative |
| A11 | Single-step (K=1) training stage | Survival 35 → 12 d during that stage, recovered by rollout. Coarse ckpt scanning understated one run 3× | feedback |

## Bugs found in this line

- PK `tyear` was hard-coded to 0.0 in five scripts, giving a **24 K rms** seasonal error in T_eq, and the net had no seasonal input.
  All absolute state-head numbers from SPEC §9.6 onward are confounded (SPEC §9.12).
- The gate RMSE was missing the nlon factor, inflating it by √192 = 13.9×.
- `pgrep -f <pattern>` in wait loops matched the loop itself and cost two days of wall time. Use PID files.

