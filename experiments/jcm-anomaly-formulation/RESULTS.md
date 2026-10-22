# RESULTS — jcm-anomaly-formulation

## B4. Anomaly formulation (X = seasonal climatology + learned anomaly): `config_jcm_anom.yaml`, run `jcm_paradis_anom`


- 72k steps × eff. batch 30. Write-up `/data/JCM_ANOMALY_FORMULATION_RESULTS.md`, verdict "refuted".
- Val loss is *better* (0.01222 vs 0.01231) while climate skill is worse: val loss is decoupled from free-run skill.

| gate | base ep24 | anom ep24 | base ep123 | anom ep123 |
|---|---|---|---|---|
| old gate (in the doc) | +0.390 | +0.320 | +0.238 (std 2.63) | +0.047 (std 1.18) |
| **fixed gate** (`rescore_*_g*.log`, not in any doc) | +0.409 | **+0.579** | +0.482 (std 0.83) | **−0.117 (std 0.46)** |

- Under the fixed gate the ep24 ranking flips. The doc's "std 2.63 → 1.18 improvement" turns out to be anomaly **damping** (std 0.46).
- `/data/jcm_emulated_winds_anom`: the anomaly ep123 free rollout **diverged at step 31**, max|u| 503 (`emul_anom.log`). Not in any doc.
- A day-of-year climatology cannot carry the QBO: it is ~28-month and not calendar-locked, so it averages out.


