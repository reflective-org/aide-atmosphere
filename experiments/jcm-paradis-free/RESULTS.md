# RESULTS — jcm-paradis-free

## B2. Non-prescribed emulator: `config_jcm_free.yaml`, run `jcm_paradis_free` (1990–94)


- Fully free gate: σ-ratio 1.044/1.009/1.016, vortex **+0.580** (`gate_freemodel.log`), below the 0.727 floor.
- **10-yr free run:** 3652/3652 days, max|u| 104 → 70 → 67 → 69 → 72 → 81 m/s (truth 93–123).
- **QBO-dead** (the original claim is retracted, R2). At 30 hPa equatorial: 0 sign reversals; easterly 3 % of the time (truth 57 %);
  spectral peak 2 months (truth 30); mean +13.5 vs truth −7.3 m/s after year 2. Figure `/data/paradis_jcm/qbo_comparison.png`.


## B3. 12-yr scale-up: `config_jcm_free_big_fast.yaml`, run `jcm_paradis_free_big` → **base ckpt `epoch=123` for 04**


- Train 1997–2008, val 2009, 60k steps × eff. batch 36, ckpt ladder `gate_ladder_big`.
- **Fixed gate** (`rescore_base_g*.log`, ep123): survival 260/191/360/360 (median 310), vortex **+0.482**, std 0.83, anomR90 1.585.
- **Old gate** (`gate_big.log`): vortex +0.238, std 2.63, anomR90 1.619, RMSE d30 15.09 (climatology 11.4, persistence 14.80).
- ep99 dies at day 36. Scale did not move the vortex above the floor.
- `PARADIS_JCM_RESULTS.md` §1 says "anomR 1.17, failure is PHASE". No log supports this (R3).
- Free rollout of ep123 in the nudged band: max|u| 420 m/s at levels 0–8 by day 250 (`/data/jcm_emulated_winds`; boundary spec §2.3).
  This motivated 03.


## B5. Filter transfer to the nudged corpus (logs only: `filter_transfer.sh`, `ftr_{base,anom}_a{0,3,10}.log`; ep123, 1996, 2 ICs)


| α | base survival | base vortex | base anomR90 | anom survival | anom vortex |
|---|---|---|---|---|---|
| 0 | 225 | +0.435 | 1.41 | 360 | +0.439 |
| 3 | 226 | +0.431 | 1.43 | 360 | +0.405 |
| 10 | 322 | +0.508 | 1.46 | 198 (IC0 died d36) | +0.203 |

**The filter does not transfer.** Nudged-corpus deaths are not grid-scale, and anomR is unchanged.
The anomaly vortex differs again from both gates, so vortex medians over ≤4 ICs are noise-dominated.


