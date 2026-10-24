# 00 — Truth model and corpora

## The truth model: jcm-strat (owner: susanne)

- **Code:** `/data/JCM_stripped/jcm-strat*`. Phases 1–23; the latest is **phase 23** (2026-10-22): a latitude-dependent nudging top,
  restarting from P20b. Its 100 hPa w* went from 0.13 to 0.17, against WACCM6's 0.40 (`jcm-strat-phase23/PROGRESS.md`).
- **Not an emulator.** It is the generator of every corpus below. Design origin: `/data/JCM_stripped/Duncan_paradis_tracer_transport_handover.md`
  (Approach A = stripped JCM on Dinosaur-SL; Approach B = PARADIS transport head).
- **p11a (phase 11 run A)** is the source of all nudged corpora. It covers 1990–2009, 6-hourly (29,220 frames), T63 192×96, L63, 24 tracers, 2.4 TB
  (`/data/TRACER_TRANSPORT_EMULATOR_SPEC.md` §2):
  - Levels **0–8** (985 → 151.5 hPa) are relaxed to ERA5 u, v, T with τ = 6 h, which removes **63 % of the deviation per step**
    (`/data/JCM_BOUNDARY_FORCED_SPEC.md` §2.1). The troposphere is therefore not a function of the model state.
  - Levels 9–58 are free; 59–62 are sponge. The QBO is a zonal-mean nudge toward a prescribed target.
  - 1990 is spin-up (T 218 → 234 K, boundary spec §4.1). Mass conservation comes from a global proportional fixer.
- **Phase 13** runs (`p13gwd`, `p13gwdl95`, …) are dry runs with Hines + Lott–Miller GWD. Their only emulator use is as the IC for the Hines sweep below.
- **Related, not emulators:** `/data/cesm-jax-dycore` (susanne) is a JAX port of the CAM FV dycore running the P20b protocol 1990–1997.
  `/data/jax-gcm`, `/data/jax-gcm-fork` and `/home/ubuntu/noah/jax-gcm` are upstream jax-gcm.

## Corpora

| path | source | years | size | nudging / QBO | used by |
|---|---|---|---|---|---|
| `/data/jcm_paradis_c1` | p11a | 1990–2009 | 415 G | ERA5 levels 0–8; QBO nudge | 02, 04 |
| `/data/jcm_paradis_1deg` | p11a | 1990–95 | 104 G | as above | smoke config only |
| `/data/jcm_nf63` (raw) → `/data/jcm_nf_c1` (zarr) | nudge-free chain | raw 2005–2021; zarr 2009–2021 (2005–08 is spin-down) | 3.2 T / 262 G | `nudging.enabled=false`, QBO rate ~0 | 03 |
| `/data/jcm_nudgefree/gwd_free_rms{2.0,3.0}` | IC from `p13gwdl95_19910101` | 10 yr each | 479 G | free, Hines rms 1/2/3 | QBO-in-corpus test (03) |
| `/data/jcm_tracers_c1` (ali) | p11a tracers | 1990, 1996–2009 | 850 G | — | 05 |
| `/data/jcm_emulated_winds{,_anom}/1990` | free rollouts of ep123 | 1990 | 20 G / 0.5 G | — | wind source for transport coupling |
| `/data/jcm_era5_cache` | WB2 ERA5 192×96 L63 u, v, T | 1997 | 20 G | — | nudging targets |

### Corpus defects found

- The nudge-free corpus failed verification at a 30-h leap-year seam (`/data/paradis_jcm/run_nf.log`: "CORPUS VERIFY FAILED").
  It was regenerated with `/data/jcm_nf63/regen_leapfix.sh`.
- The first PARADIS corpus copied v2's year list, which skips 1992. Samples straddling the gap paired frames a year apart.
  The val year (1993) was also inside the training range. Both were fixed in commit `52534ed`.
- `open_mfdataset` coordinate union: `terrain.nc` latitudes differed by 2.8e-14, which silently turned 96 latitudes into 144 (commit `b1b314d`).
