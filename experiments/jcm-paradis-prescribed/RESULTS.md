# RESULTS — jcm-paradis-prescribed

# 02 — PARADIS 25 M on JCM (Sep 23 – Oct 7)

- **Code:** `/data/paradis_jcm`. Git: v2 snapshot vendored verbatim at `a8a588c`; last commit `4cc8bba` (2026-09-25).
  **164 uncommitted changes since**, including the W_QBO fix in `model/jcm_physics.py`, `trainer.py`, `gate_paradis.py` and all anomaly/nf configs.
- **Runs:** `/data/paradis_logs/lightning_logs/jcm_paradis_{smoke,v1,free,free_big,anom,nudgefree}`.
- **Write-up:** `/data/PARADIS_JCM_RESULTS.md`. Stale; see RETRACTIONS.md R1–R4.

**Design.** The v2 (cesm_run_2) recipe was used, not v5/v6, because the v5/v6 recipe was the measured regression.
- Pruned: moisture, precip, surface, conservation, CRPS, sigmoid clamp.
- Native u, v instead of Cartesian wind.
- JCM forcing carried structurally: PK relaxation, Rayleigh sponge, QBO nudge, prescription of levels 0–8, increment factor
  `g = (1 − e^{−a dt})/(a dt)`.
- Verified before training (results §2): adapter bit-exact vs netCDF; Gauss–Legendre weights to 5.4e-8; T_eq port to 6.1e-5 K;
  overfit one batch 1.307 → 0.00086.

**Gate:** `gate_paradis.py`, test year 1996, 4 ICs (0/364/728/1092), 360 d. The gate was itself fixed around 10-02; old-gate and fixed-gate
numbers differ (see B4). `--free` turns off prescription and the QBO nudge. Without `--free`, **qbo=True** (the `trainer.py` default `jp.get("qbo", True)`).

## B1. Prescribed emulator: `config_jcm.yaml`, run `jcm_paradis_v1` — **VOID (W_QBO)**


- Train 1990–94, val 1995, 15k steps, 5.5 h on 4 H200s. Val 0.0302 → 0.0128.
- `gate_a.log` "best": 4/4 × 360 d, σ-ratio ≈ 1.00, vortex corr +0.979/+0.930/+0.925/+0.881 (median **+0.927**).
- **Void.** The config sets no `qbo` key, so the QBO nudge was on, with the scrambled W_QBO that put 67 % of the weight at |lat| ≥ 30. That nudges
  the polar vortex toward truth (RETRACTIONS.md R1). The decomposition table (results §4) is void for the same reason. Row (c), "prescription off,
  QBO on", scoring *higher* (+0.933) than the full model is the tell.
- **Still valid:** the no-network control (`gate_nonet.log`, vortex std 0.00) and the observation that a frozen state scores σ-ratio 0.94–1.24.


## Silent bugs found in this line (results §8)


| bug | effect |
|---|---|
| `jcm_to_zarr.py` ran `main()` on import | a test's *import statement* launched a second full conversion, corrupting a year mid-verification |
| dataset `__len__` = full frame count | last samples returned empty targets |
| zarr chunking `(5,24,48,64)` | 5.1× read amplification; fixed by one-timestep chunks, 21× dataloader speedup (`70edda8`) |
| PyTorch silently on CPU | needs `LD_LIBRARY_PATH=/run/nvidia/driver/...` |
| W_QBO `permute(1,0)` (`882b1fb`, Sep 23 → fixed Oct 21) | see RETRACTIONS.md R1 |

