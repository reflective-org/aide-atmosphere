# Results: jcm-p20bt-free-base

All paths are under /data/paradis_jcm unless absolute. One training seed (42).

## Training

125 epochs, launched 2026-10-22 23:37:35 UTC, finished 2026-10-23 13:36:46 UTC on GPUs 2-7 (84 GPU-h).
Val loss fell monotonically from 0.0371 to 0.01425 (logs_p20bt/train_free.log), so it cannot pick a checkpoint.

## 1. Unfiltered ladder: no checkpoint is good

Source: logs_p20bt/gate/g96_*.log. 1996, 4 ICs, 360 d, free. Floors: climatology vortex corr 0.932, Tprof 0.15 K, bias 0.19 m/s;
persistence bias 27.1 m/s.

| ckpt | survived 360 d | good | notes |
| --- | --- | --- | --- |
| ep09 | 4/4 | 0/4 | Tprof 8.9-9.5 K, bias 19-21, gmV 80-82 (truth 10-12), vortex corr 0.35-0.58 |
| ep25 | 4/4 | 0/4 | Tprof 13-17 K, bias 16-19, vortex corr 0.17-0.30 |
| ep41 | 0/4 | 0/4 | died d30-246 |
| ep57 | 0/4 | 0/4 | died d21-356 |
| ep73 | 0/4 | 0/4 | died d15-244 |
| ep89 | 1/4 | 0/4 | died d18-286 |
| ep105 | 0/4 | 0/4 | died d20-108 |
| ep121 | 2/4 | 0/4 | died d27/29; survivors Tprof 33-35 K |
| last | 1/4 | 0/4 | died d17-303 |

## 2. Mechanism: an unstable zonal-mean meridional-wind mode at the top level

- **Every checkpoint**, including the ep09/25 "survivors", sends the h94 zonal-mean u to 76-273 m/s by day 9-10 (truth 28-33).
  Source: logs_p20bt/diag/d_*.log.
- **Local to h94.** With h0-h93 and ps reset to truth every step, so only h94 runs free, ep121's h94 zonal-mean u still reaches
  529 m/s by day 30. Source: logs_p20bt/diag/hy_121_94.log.
- **Invisible to teacher forcing.** The one-step zonal-mean u bias at h94 is -0.05 m/s per step, with a latitude-rms of 0.14.
  The free-run drift is about 3 m/s per step. Source: logs_p20bt/diag/ob_121.log.
- **A uniform zonal-mean u perturbation is damped,** with gain 0.67 per step at ep121 and 0.82-0.89 at ep09/25. Source:
  logs_p20bt/diag/gp_*.log.
- **Power iteration finds the unstable mode.** On the one-step Jacobian of the zonal-mean (u, v, T) at h94 it gives |lambda| of
  1.16-1.37 per 6 h for all four checkpoints probed, and 94-100 % of the mode is zonal-mean **v**. Source:
  logs_p20bt/diag/ep_*.log.
- **It starts at the pole rows.** At 88.6N, zonal-mean v94 goes 3 → 21 → 134 → 265 m/s at days 0.25 / 2 / 5 / 8; truth there is
  ~0.0-0.3. The Coriolis term f·v then drives u.
- **It sits at high total wavenumber.** In the vector-SH spectrum, the m=0 error in the v component grows at l = 32-48:
  6e-4 → 3e-2 → 0.34 → 4.4 at days 1 / 3 / 5 / 8, against truth power of ~1e-5. Source: diag_spec_p20bt.py.
- **Why truth has no such mode:** the spectral dycore's hyperdiffusion keeps that band empty. Nothing in the emulator damps it.

## 3. Inference-time constraint: damp only the m=0 vector harmonics

Each step, the m=0 vector-SH coefficients of (u, v) at every level are multiplied by exp(-36 (l/lmax)^4), via gate_p20bt.py
--vfilt-m0 36,4 (analysis/gate_p20bt_vfilt_m0.diff). This is not part of the trained model.

**What didn't work, and how much the filter costs:**
- A plain vector-SHT projection changes nothing; the run is identical to unfiltered.
- A filter on all m stops the runaway but removes the small-scale eddies (high-k ratio 0.00), and it changes truth by 15 % rms
  per application in the troposphere.
- The m=0-only filter keeps strat high-k power at 0.4-1.6 of truth.

Source: logs_p20bt/gate_vf/*.log. Floors: 1996 clim vortex corr 0.932; 1995 0.880.

| ckpt | year | ICs | survived | good | vortex corr | Tprof K | bias m/s | rm30 m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | 1996 | 0/364/728/1092 | 4/4 | 3/4 | 0.835 / 0.850 / 0.877 / 0.885 | 1.13 / 1.25 / 4.52 / 1.13 | 6.5 / 7.0 / 8.4 / 6.8 | 24.9 / 18.8 / 22.5 / 18.9 |
| last | 1995 | 0/364/728 | 3/3 | 3/3 | 0.814 / 0.808 / 0.785 | 1.25 / 1.61 / 1.20 | 6.9 / 8.6 / 8.3 | 22.9 / 18.2 / 21.1 |
| ep121 | 1996 | 4 | 4/4 | 1/4 | 0.71-0.84 | 2.7-12.1 | 11.8-18.5 | 18.2-27.2 |
| ep105 | 1996 | 0 | 0/1 | 0/1 | died d113 | | | |

**Reading:**
- The last checkpoint, which was among the worst unfiltered, becomes the best once the mode is removed.
- It is stable and balanced: zmV 17.6-22.8 vs truth 19.3-23.0, and gmV 8.0-10.5 vs 8.8-12.2.
- It **does not beat climatology:**
  - vortex corr medians are 0.864 (1996) and 0.808 (1995), against floors of 0.932 and 0.880;
  - rm30 is 19-25 m/s, against 14 for climatology;
  - its bias of 6.5-8.6 m/s is above truth's own 4.2.
- The mesosphere still drifts, at rms ~40 m/s in the 60-d diagnostics.
