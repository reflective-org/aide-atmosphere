# jcm-p2-hl: a stochastic emulator of a stratosphere-resolving JAX dynamical core

**P2@200** is the checkpoint that produced the 1997-1998 emulated winds and temperatures for the AIDE stratospheric
aerosol injection (SAI) campaign. This folder holds what you need to run it: the inference code, a smoke test and a
rollout script. The weights, statistics and initial condition are attached as release assets (see [Assets](#assets)).
It runs exactly as in the campaign, including the zonal-mean momentum fixer, which evaluates two drag terms of the
JAX dycore on every step.

## What the model is

- **Emulated system.** The model emulates JCM (`jcm-strat`, a dry spectral GCM in JAX built on
  [jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)). JCM is run as a specified-dynamics dycore: the 16 levels
  below 400 hPa are nudged to ERA5, temperature is relaxed toward a Polvani-Kushner/Jucker equilibrium, there is a sponge
  and boundary-layer friction, and a Held-Larichev-type parameterised wave drag forces the QBO.
- **Grid.** T63 Gaussian, 96 × 192 (lat south-to-north). 95 hybrid sigma-pressure levels from about 996 hPa to
  0.01 hPa, surface first.
- **State.** u, v, T and ω on all 95 levels, plus p_s / 10⁵ Pa: 381 channels.
- **Step.** 6 hours. The emulated dycore takes 30 × 12-minute steps per emulator step.
- **Architecture.** PARADIS-style neural advection-diffusion-reaction:
  8 layers, latent width 1024, 256 advecting velocity fields, and a vertical-mixing MLP. A zero-initialised linear 1×1
  skip is scaled by the per-channel 6-hour tendency scale. The model has 25.7 M parameters. Inputs: two consecutive
  states (t−6 h, t), the sin/cos of year progress, 48 channels of Gaussian random fields (three spatial scales) and 6
  static fields.
- **Analytic physics on every step.** These terms are applied in closed form rather than learned: the Jucker/PK
  temperature relaxation, the sponge, boundary-layer friction and the QBO wave drag (`hlr_p.npz`). The global-mean
  surface pressure is held fixed (`massfix`). The model gets **no ERA5 input at run time**: it steps its own state.
- **Zonal-mean momentum fixer (campaign configuration).** After each step, the zonal-mean u increment over
  15°S-15°N and 100-1 hPa (tapered) is replaced by the one implied by the zonal-mean momentum budget of the two states:
  advection, Coriolis, curvature and eddy-flux divergence, plus the dycore's own HL and Hines gravity-wave drag. The
  correction is smoothed 1-2-1 four times in latitude and twice in the vertical. The two drag terms are evaluated by
  jcm-strat itself (JAX) in a subprocess, `fixer/fixer_server.py`. That server builds only the physics terms; the
  dycore is not stepped.
- **Stochastic.** The noise channels make each seed a different trajectory. The model was trained as a CRPS
  generator, and the campaign used one member per year: seed 0 for 1997, seed 1 for 1998.

## What it was trained on

1. **Base model.** A deterministic PARADIS + linear-skip model trained on a JCM corpus nudged to ERA5 (`p20bt`), with
   train years 1997-2008. That is the first checkpoint in the lineage below.
2. **Generator and DAgger fine-tunes.** It was then turned into a noise-driven generator and fine-tuned through the
   stages listed below (afCRPS loss on archive rollouts, plus dycore-correction ("DAgger") targets started from the
   emulator's own drifted states).
3. **The P-stages.** P1 and P2 trained against `hlcorpus`: a JCM corpus from 1995 to 2005, nudged to ERA5 below
   400 hPa, with the QBO forced by the Held-Larichev drag instead of being nudged. Train years were 1997-2005 and 1996
   was held out.
4. **Statistics.** The normalisation statistics are train-years-only from the original corpus.

Full lineage:
`A_paradislin@30000 → D0@4000 → D0R@400 → S2@1000 → S3@1000 → S4_noqbo@1000 → J2_s4q0@200 → P1_hl@600 → P2_hl@200`.

The training data are not distributed here.

## How to run

The model needs two Python environments on a CUDA GPU:

1. **Emulator (PyTorch):** Python ≥ 3.10 with `torch`, `torch_harmonics` and `numpy` (tested: Python 3.10.12, torch
   2.10.0+cu128, torch_harmonics 0.8.0, numpy 2.2.6).
2. **Fixer (JAX):** [jcm-strat](https://github.com/reflective-org/jcm-strat) at commit `6c0fd661e8bcc79635d8fbe0de2b7ab57e7aed33`
   with its submodule [jax-gcm](https://github.com/climate-analytics-lab/jax-gcm) at
   `35dc19974670044eb7ab8a3a8944a6143eae54eb`. These are the exact sources the campaign ran. Tested with Python 3.11.16,
   jax/jaxlib 0.10.2 (CUDA 12), flax 0.12.8 and dinosaur 1.5.0; the full package list of the tested environment is in
   `fixer/requirements-jax.txt`.

```bash
git clone https://github.com/reflective-org/jcm-strat && cd jcm-strat
git checkout 6c0fd661e8bcc79635d8fbe0de2b7ab57e7aed33 && git submodule update --init external/jax-gcm   # -> 35dc199
python3.11 -m venv .venv && .venv/bin/pip install -e external/jax-gcm -e .                            # + jax[cuda12]==0.10.2
```

These install commands have not been tested from scratch. What has been verified is that the pinned sources, run in
the tested environment, give bit-identical fixer output.

Download the release assets into `./assets` and unpack the ERA5 reference files (`tar xf era5_zm_monthly_1989-2022.tar
-C assets`). Then:

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_test.py --assets assets --jax-python /path/to/jcm-strat/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 python rollout.py --assets assets --ic assets/ic_19970101T06.npz --steps 1460 --seed 0 \
    --jax-python /path/to/jcm-strat/.venv/bin/python --out run_1997    # one year; ~41 GB host RAM for every frame
```

The fixer server runs on the same GPU with a 20 % JAX memory cap. `--no-fixer` runs the emulator alone, which does
not match the campaign configuration (see Limitations).

`rollout.py` writes `u, v, T, omega` as (frames, 95, 96, 192) arrays and `ps` (p_s / 10⁵ Pa) as (frames, 96, 192),
all float32 in physical units. It also writes `plev_hPa.npy` (level pressures at p_s = 1000 hPa), `lat.npy`, `run.json`
(arguments and the time of every frame) and the fixer server's log. Frame k is t0 + 6 h × k. Use `--save-every 4` for
daily output.

From Python:

```python
from p2emu.emulator import Emulator, NoiseGen
E = Emulator("assets", "assets/p2_hl_step200.pt")       # states are z-scored (B,381,96,192); E.physical(z) converts
E.enable_fixer(1997, "/path/to/jcm-strat/.venv/bin/python")
new = E.step(prev_z, cur_z, t_cur, NoiseGen("cuda", seed=1000).sample(1))
E.close()
```

## Verification of this package

All measurements were taken on one H200 against the internal campaign code, from the shipped IC with seed 0.

- **With the fixer: bit-identical over 30 days.** `rollout.py` reproduces the campaign rollout script bit for bit (max
  |difference| = 0.0 in u, v, T, ω and p_s) over 120 six-hour steps. The reference ran the campaign code unchanged,
  except that its fixer server was started deterministically (see the next point).
- **The campaign's own fixer runs were not bit-reproducible.** Its JAX server's HL tendency varied by 1.7-3.4e-13
  m/s/s (on values up to 5e-7) between identical runs. Most likely the cause is that it imported JAX before setting
  `--xla_gpu_deterministic_ops`: with the flag set from process start, the same code reproduces itself exactly. It also loaded
  executables from a persistent compilation cache, which shifted HL by 1.4e-13. This package sets the flag before
  JAX starts and does not use the cache. Over 30 days, two identical campaign-code runs differ by up to 0.12 m/s
  (latitude-RMS zonal-mean u, worst level), and this package differs from a campaign-code run by up to 0.14 m/s.
  Equatorial u at 30 hPa agrees to 0.01 m/s in all three runs.
- **The fixer server reproduces the dycore code exactly.** HL and Hines tendencies from `fixer/fixer_server.py`, run
  on the pinned public jcm-strat/jax-gcm sources, equal the campaign server's (run deterministically) to 0.0 on 24 real
  states.
- **Without the fixer (`--no-fixer`):** also bit-identical to the campaign code run with `--fixer 0`, over 120 steps.
- **Forcing channels.** The year-progress channels are bit-identical to the training dataset's at every 6-hourly time
  from 1990 to 2030.
- **Speed.** With the fixer, 0.165 s per 6-hour step (median), or about 4 min per simulated year without I/O; the
  fixer server takes about 15 s to start and the first step about 6 s while JAX compiles. Without the fixer, 0.068 s/step.

## Evaluation (from the AIDE campaign, not re-run here)

For the 1997-1998 free run, compared over DJF/JJA zonal means from 1 to 300 hPa:

| Comparison | T RMS (K) | u RMS (m/s) |
|---|---|---|
| vs its own dycore corpora | 1.5-2.7 | 6.1-8.7 |
| vs ERA5 | 6.2-6.5 | 10.3-10.5 |
| vs CESM | 6.4-7.1 | 9.6-10.7 |

The mean T bias vs ERA5 is +1.5 K (DJF) and +2.4 K (JJA).

## Known limitations

- **Inherited climate biases.** The model reproduces its dycore, not the real atmosphere. Against CESM 1997 annual
  means it is +7 K warm at the tropical tropopause (100 hPa equator: 200.2 vs 193.4 K) and up to +15-20 K warm near
  1 hPa. The SH jet is weak (36 vs 48 m/s) and the tropical easterlies too strong (−27 vs −15 m/s). Age of air on its
  winds is 0.5-1.5 yr older than on CESM winds, which suggests weak tropical upwelling.
- **No QBO claim.** The tropical stratospheric winds are not a validated QBO. Do not use this model for QBO
  period/amplitude studies.
- **The fixer matters in the tropical stratosphere.** Without it the tropical stratosphere departs quickly. Measured 10
  days into the same run, equatorial zonal-mean u at 30 hPa is −16.3 m/s without the fixer and −7.5 m/s with it, while
  the troposphere still agrees to 0.2 m/s RMS at 290 hPa. Whether the fixer brings the emulator closer to the dycore
  truth has not been measured here.
- **The fixer needs a second environment.** The two drag terms come from jcm-strat (JAX), so the full model needs
  both a PyTorch and a JAX environment. The fixer server only builds the dycore's physics terms. It reads the ERA5
  zonal-mean reference files only to construct the QBO term, and its patched drag does not use them. It requires the
  file for the start year, and warns if the following year's file is missing.
- **One initial condition is shipped.** It is 1997-01-01 06 UTC from the ERA5-nudged JCM corpus. Any (t−6 h, t) state
  pair on the same grid and in the same channel order will work.
- **A short training lineage at the end.** P2 is 200 optimiser steps past P1. Its one-step QBO-region gate failed on
  the mean error at step 200 (see the campaign record).

## Assets

The assets are attached to the GitHub release `jcm-p2-hl-v1`. They are not stored in git. Check them against
`SHA256SUMS`.

| File | Bytes | Contents |
|---|---|---|
| `p2_hl_step200.pt` | 104,296,379 | weights only (`{"net": state_dict, "meta": ...}`), bit-identical to the training checkpoint's tensors |
| `era5_zm_monthly_1989-2022.tar` | 59,402,240 | ERA5 monthly zonal-mean u, 1989-2022 (`era5_ref/`): read when the fixer server builds the QBO term |
| `ic_19970101T06.npz` | 56,223,218 | IC: z-scored states at 1997-01-01 00 and 06 UTC |
| `jcm_analytic_p20bt.npz` | 11,303,834 | grid, hybrid coefficients, analytic-physics coefficients (incl. an ERA5-derived QBO target, unused in this mode) |
| `terrain_t63.nc` | 1,197,205 | T63 terrain from [jax-gcm-data](https://huggingface.co/datasets/climate-analytics-lab/jax-gcm-data) `bundles/t63/terrain.nc` at revision `2ec867b` (CC-BY-4.0) |
| `meta.npz` | 640,838 | channel names, normalisation and tendency statistics, static fields, grid |
| `hlr_p.npz` | 5,840 | QBO wave-drag parameters |

## Licences and attribution

- **Code:** licensed under the repository's LICENSE. `p2emu/model/{paradis,advection,blocks,padding}.py` are modified
  from PARADIS (Stéphane Gaudreault et al., Apache-2.0); see `NOTICE`. `p2emu/model/jcm_physics_p20bt.py` is a torch
  port of analytic terms from JCM/jax-gcm (Apache-2.0). `fixer/` calls jcm-strat (Apache-2.0) and jax-gcm (Apache-2.0),
  which are installed separately, not vendored.
- **Terrain:** `terrain_t63.nc` is from the jax-gcm-data dataset (climate-analytics-lab), CC-BY-4.0, unmodified.
- **ERA5:** the data assets contain modified Copernicus Climate Change Service information (ERA5; the IC is 1997, the QBO
  target in `jcm_analytic_p20bt.npz` spans 1989-2020, and the reference files span 1989-2022). Neither the
  European Commission nor ECMWF is responsible for any use of the Copernicus information or data it contains.

## Citation

If you use this model, cite this repository, PARADIS and JCM/jax-gcm. A paper is not yet
available.
