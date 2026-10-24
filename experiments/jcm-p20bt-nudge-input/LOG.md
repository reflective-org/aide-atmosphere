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

## 2026-10-24 ~10:30 UTC: one-step table at step 5500, loss mis-weighting, precision floor

- **One-step error**, % of the 6-h change (onestep_rel_err_p20bt.py, 200 states of 1996), step-0 D → D step 5500:
  - u h0-15 49.8 → 19.9; v h0-15 42.4 → 13.8; T h0-15 57.5 → 28.7; ω h0-15 40.2 → 29.5;
  - h16-h94 unchanged at 15–26 % (u 23/20/25, v 15/15/18, T 20/19/25);
  - **ps 32 → 68 %**, and 100 % at epoch 9 (check_DT_loss.py, 40 states). D training is degrading ps.
  - val_loss is flat (0.0148 → 0.0143).
- **Cause of the ps regression: loss weighting.**
  - The 6-h change, in the state-std units the loss uses, measured on train years 1997–2008 (2504 pairs,
    meta/tend_std_p20bt_1997_2008.json):

    | channel | 6-h change |
    |---|---|
    | ps | 0.019 |
    | T h0-15 | 0.061 |
    | u h16-73 | 0.118 |
    | v h94 | 0.98 |
    | ω | 0.65–1.24 |

  - Errors are ≪ the Huber delta of 1, so berHu is effectively L1 on state-std errors. Per unit of relative error, ps
    carries about 1/50 of v_h94's weight (1/2400 in squared terms).
  - Fix: `training.loss_function.tend_scale_file`, which scales each channel by 1/σ(6-h change) before an MSE, so the loss
    equals the 1 %-metric squared. Implemented in utils/loss.py and trainer.py; the default path is unchanged; backups are
    *.bak_pre_tendscale.
  - Falsifier: loss_fn equals the hand computation to 1.6e-8, and the band values reproduce the one-step table.
    Config: config_jcm_p20bt_DT.yaml.
- **Numerical precision floor** (precision_floor_p20bt.py, epoch 9, 8 states). Deviation of the network output from an fp64
  forward, % of the 6-h change:

  | precision | u/v/T/ω | ps |
  |---|---|---|
  | strict fp32 | 0.00 | 0.00 |
  | TF32 | 0.1–0.3 | 1.0 |
  | **bf16 (training's bf16-mixed)** | **2.3–5.6** | **16** |

  The current ~20 % error is not a precision effect: the error against truth is identical across precisions. **But
  bf16-mixed puts a hard floor of 3–6 % (ps 16 %) under any model trained or run in it, so the 1 % goal needs fp32/TF32**
  (use_amp false, about 2× the step cost; to be measured).
- **Overfit with the DT loss:** 4 samples, lr 3e-4, from epoch 9. rms err/σ went 26 → 17 % in 600 steps, not to ~0, and ps
  oscillates between 13 and 165 %. Running 2000 steps at lr 1e-4 with a residual SH spectrum, to see whether the floor is
  representability (high-l) or optimisation.
