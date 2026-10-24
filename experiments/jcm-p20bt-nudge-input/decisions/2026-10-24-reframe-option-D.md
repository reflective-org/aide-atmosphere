# Reframe to "1 % of the 6-h change" and option D (2026-10-23/24, decided by Shreyas)

## Goal

An emulator of the p20bt jcm-strat step with one-step rms error ≤ 1 % of the rms truth 6-h change, in every variable (u, v, T,
ω, ps), per level band, area-weighted. Its winds must also feed downstream tracer transport. The top priority is correct
**stratospheric** dynamics in free runs.

## Evidence behind the choice

Numbers are from /data/paradis_jcm/dycgen/VALIDATION.md and SPLIT.md, and from logs_p20bt/floor_oy.

- **Baseline.** The current free model (the last checkpoint of jcm_p20bt_free) is at 15–40 % of the 6-h change, so it is
  20–40× off the goal.
- **Generator.** The JAX generator (a frozen copy of jcm-strat phase 19) reproduces the archive's next frame to 0.06–0.07 % of
  the 6-h change for u, v and T. It is bitwise deterministic.
- **The step depends on ERA5 the state doesn't carry.** Feeding the generator ERA5 from 6 h earlier (same year) instead of the
  real target changes the step by 64 % at h0–15, 18 % at h16–73 and 5–10 % above, as % of the 6-h change. A free-running
  emulator with no ERA5 input therefore cannot reach 1 % for the nudged step.
- **Splitting doesn't rescue it.** An un-nudged emulator step followed by analytic nudging is off by 46 % at h0–15 for 6-h
  coupling. With Strang ordering it is 24 % at 6 h, 6.5 % at 2 h, and 3.6 / 1.5 / 0.8 / 2.4 % (u bands) at 1 h.
- **The right bar for free runs is other-year truth, not climatology.** Another year of truth scores vortex corr 0.755–0.887
  against climatology's 0.93.

## Options considered

- **A.** Un-nudged truth.
- **B.** Free-running emulator of the nudged step with no ERA5 input. Its floor is set by the unpredictable ERA5.
- **C.** Short (1–2 h) un-nudged emulator steps with analytic nudging between them.
- **D (chosen).** Keep the full nudged step as truth. The emulator gets ERA5(t) at h0–h15 for u, v, T, plus the calendar year,
  as forcing inputs. Then the step is deterministic given state + forcing, so 1 % is well-posed. "Free-running" means the
  emulator steps its own state with ERA5 supplied only as forcing, which is how p20bt itself is built.

Order: D first, then B and C ("try a; and then after that go for the others", amended to D: "we want to understand the dynamics
with the nudge in place").

## Implementation (option D)

- **Analytic physics.** The truth's ERA5 nudge −(X − E)·NUDGE_INV_TAU on h0–15 was added to the closed-form 6-h relaxation
  (model/jcm_physics_p20bt.py, `era5=`), alongside the dated QBO nudge (`qbo=True`, now with the real year).
  - It matches a direct 30-substep forward-Euler integration of the summed tendencies to 1e-12.
  - With the nudge off it is bitwise unchanged.
  - A falsifier with the target's levels flipped fails, as it should.
  - Test: /data/paradis_jcm/test_era5_nudge_p20bt.py.
- **Inputs.** ERA5 target channels and a year channel are appended after the forcings: data/era5_dataset.py
  `era5_target_dir`, trainer `jcm_physics.era5_nudge`.
- **Warm start.** From the free base, with zero-initialised new input columns (`init.expand_input_channels`).
