# Log: jcm-p20bt-nudge-input

## 2026-10-24

- Reframe decided with Shreyas (see decisions/). The generator, the ERA5-target zarr, the physics nudge, the dataset/trainer plumbing, the overfit and the smoke run are all done (see EXPERIMENT.md, Ladder record).
- The first launch on GPUs 2-7 was stopped during setup: susanne holds GPU 7. Relaunched on GPUs 2-6. A gate on GPU 1 was refused by the permission classifier, since GPU 1 is susanne's, so evals run on GPU 6 alongside training.

## 2026-10-24 08:47 UTC: step-0 D with the m=0 filter; filter damage to truth

- **Step-0 D model** (base network, structural ERA5 + QBO nudges, no D training) with `--vfilt-m0 36,4`, 1996, 4 ICs × 360 d:
  - all 4 survive and are "good";
  - vortex corr 0.878 / 0.940 / 0.945 / 0.947;
  - rm30 21.3 / 14.3 / 16.1 / 14.2;
  - Tprof 1.37 / 1.29 / 1.13 / 0.90 K;
  - bias_vs_clim 7.9 / 7.8 / 8.0 / 6.3 (truth 4.1–4.5).
  - Without the filter the same model dies at days 15–37.
  - With the filter, vortex corr is at or above the other-year floor (0.755–0.887). The bias is still about 2× truth's.
- **Filter damage to truth** (|filter(truth) − truth| as % of the 6-h change, u):

  | filter | h0-15 | h16-73 | h74-93 | h94 | damping at l=16/32/48 |
  |---|---|---|---|---|---|
  | 36,4 | 9.3 | 5.6 | 2.6 | 3.6 | (earlier run) |
  | 36,8 | 3.5 | 1.1 | 0.8 | 0.9 | 1.000/0.994/0.858 |
  | 200,8 | 5.2 | 2.0 | 1.3 | 1.5 | 1.000/0.967/0.428 |

  A sharper filter (36,8) costs about 1 % above h16. That is near the target but not below it. Testing whether 36,8 still stabilises.
- Launched on GPU 6: the step-0 gate with 36,8, and an unfiltered gate of D at step 5500 (does D training remove the h94 mode?).
