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

## 2026-10-24 ~10:10 UTC: gates, scorecard validation, first 2-yr stratospheric free run

- **Unfiltered gate of D at step 5500** (1996, 4 ICs): all BAD. ICs 728, 1092 and 364 die at days 32, 92 and 212. IC 0
  survives but with Tprof 49 K. **D training does not remove the instability.**
- **Step-0 D with the gentle filter 36,8** (1.1 % truth damage at h16-73): 3 of 4 die (days 19/42/24). Only 36,4 (5.6 %
  damage) stabilises. A filter cannot give both stability and the 1 % goal, so the mode has to be trained out.
- **Scorecard validated** (strat_metrics_p20bt.py):
  - truth vs truth is exact;
  - 1995 truth scored as a model of 1996: QBO10 corr −0.84, QBO30 +0.02 (wrong phase), VX60N10 0.83,
    h16-73 u/T bias 8.7 m/s / 1.35 K.
- **Step-0 D + filter 36,4, 720-d free run from 1995-01-01** (logs_dyc/frD/step0vf364_1995_ic0.npz): survives, no growing
  drift.
  - QBO10/30 corr 1.00, std 0.98, reversals 2/2 and 1/1. This is structural, from the τ = 1 d QBO nudge, which truth has too.
  - VX60N10 corr 0.88 (yr 2 0.76), std 0.94, SSW 3/3.
  - VX60S10 0.98.
  - Polar-cap T bias +1.2 / +1.2 / −1.3 K (10N / 50N / 10S).
  - h16-73 u bias 3.4 → 2.0 m/s and T 1.5 → 1.7 K (first vs last 120 d).

## 2026-10-24 ~10:45 UTC: the 20 % plateau is underfitting; learning curve; T63 test; DT smoke

- **No generalisation gap.** Epoch 9 of D scores the same on train year 2001 and test year 1996: u 19.5/25.1/20.9/24.0 vs
  19.8/25.4/21.6/25.0, and so on. **The model underfits; more data would not help.**
- **Overfit, 4 samples, lr 1e-4, 2000 steps:** 26 → 9.3 % and still falling, so it is not floored. 60–90 % of the residual
  energy is at l ≥ 22, where the 6-h change has 3–25 % of its energy.
- **T63 truncation test** (t63_truncation_p20bt.py):
  - T is exactly T63.
  - u and v carry 1–6 % above T63 (4.5 % even above T65 at h0-15), ω h0-15 48 % and ps 59 %.
  - Truncating the epoch-9 output only helps T h0-15 (28 → 23 %). The trained model's error lives below T63. **Not a free
    constraint; dropped.**
- **Learning curve of the free base** (onestep, 1996, 60 states; % of the 6-h change):

  | epoch (≈ step) | u h16-73 | v h16-73 | T h16-73 | u h0-15 | T h94 | ps |
  |---|---|---|---|---|---|---|
  | 3 (1.5k) | 85.0 | 56.5 | 77.8 | 54.8 | 101 | 671 |
  | 12 (6k) | 51.3 | 34.4 | 45.0 | 38.1 | 52 | 93 |
  | 30 (15k) | 38.0 | 25.3 | 32.6 | 31.1 | 39 | 93 |
  | 60 (29k) | 30.8 | 19.8 | 26.9 | 28.3 | 37 | 80 |
  | 90 (44k) | 28.3 | 17.6 | 23.6 | 27.1 | 35 | 33 |
  | 123 (60k, after LR decay) | 23.3 | 15.2 | 19.8 | 25.5 | 24 | 136 |

  From 6k to 44k steps the error falls roughly as steps^−0.3. **Extrapolation, not measured:** reaching 1 % at that rate
  needs ~10⁴× more training, and ~500× even at steps^−0.5. The 1 % goal is **out of reach for this architecture and
  training regime by orders of magnitude**; it needs a structural change.
- **DT smoke** (1 GPU, 200 steps, from D epoch 9; tendency MSE + fp32 + m=0 filter 36,4 in the loop): passed. Loss
  0.034 → 0.008–0.02, 1.33 s/step on a shared GPU 6. Projected full DT: 6000 steps × 5 GPUs ≈ 11 GPU-h. Launches when D
  finishes (~13:00 UTC).

## 2026-10-24 10:45 UTC: trained D breaks free runs even with the filter; D stopped; DT launched from step-0 D

- **Epoch 9 of D (~step 5.9k) + filter 36,4, 1996:** all 4 ICs die (days 137 / 48 / 172 / 80); Tprof 5–7 K. The step-0 D
  model with the same filter survives 4/4 (good) and 2 yr. **D training (bf16, berHu ≈ L1 on state-std errors, 1-step)
  degrades free-run stability**, alongside the ps regression (32 → 100 %). This matches the earlier finding that more
  training of this kind degrades multi-year behaviour.
- **D stopped at step ~10.5k** (epoch 16) to free GPUs. All epoch checkpoints are kept, and last.ckpt is copied to
  ckpt_D/D_final.ckpt. The post-D eval (one-step 200 states, filtered gate, 2-yr free run) runs on it, queued as
  logs_dyc/evalD_final.sh.
- **DT launched 10:47:48 UTC** (config_jcm_p20bt_DT.yaml md5 b06c75c8, snapshot .SNAPSHOT), GPUs 2–6, 6000 steps.
  - **Warm start from step-0 D**, not trained D.
  - Tendency-scaled MSE, fp32, m=0 filter 36,4 in the loop.

## dt probe: the p20bt step's own sensitivity to its time step (subagent, dycgen/probe_dt.py)

- **Setup.** Same model, same round-tripped start state (bitwise), 6 states of 1996, only `run.time_step` changed. The
  dt = 12 sanity row is 0.000 everywhere (bitwise equal to generate()). rms(y_dt − y_12) / rms(6-h change), %:

  | dt | u h0-15/16-73/74-93/94 | v | T | ω (inst) | ps |
  |---|---|---|---|---|---|
  | 24 | 12.8 / 11.9 / 5.9 / 8.8 | 9.6 / 9.8 / 4.7 / 9.0 | 12.8 / 9.4 / 6.1 / 10.6 | 61 / 78 / 20 / 22 | 7.7 |
  | 36 | 26.1 / 20.6 / 10.3 / 15.5 | 19.6 / 17.4 / 8.2 / 15.8 | 25.5 / 16.6 / 10.5 / 18.5 | 108 / 132 / 31 / 34 | 17.2 |
  | 60 | 53.6 / 33.0 / 18.0 / 27.6 | 40.2 / 28.8 / 13.9 / 26.8 | 49.7 / 28.7 / 17.5 / 31.0 | 173 / 202 / 44 / 50 | 37.5 |

- **Roughly first order in dt.** Warm s/step on a shared GPU: 3.6 (12 min), 2.0 (24), 1.6 (36), 1.0 (60).
- **Reading.**
  - The 1 % target is about 10× tighter than the difference between the p20bt step at dt 12 vs 24 min. An emulator meeting
    it must reproduce the 12-min integration's own time-discretisation error.
  - A "cheaper numerical step + learned correction" hybrid buys only 2–3.5× speed and leaves a 10–50 % correction. At the
    current ~20 % relative network error that is ~2 % at best (estimate). Not pursued.
- **Bug noted** (frozen gen_pairs.py, not edited): `return_extra=True` and `omega_mode='instant'` use `rs.dycore_state`; the
  attribute is `rs.dynamics`.

## 2026-10-24 ~11:15 UTC: first DT launch crashed at its first validation; fixed and relaunched

- **Crash.** The first DT launch (10:47) died at step ~584, its first validation: `CUDA error: initialization error`.
  - The C++ stack ends in `c10::TensorImpl::~TensorImpl`, i.e. a forked val DataLoader worker freeing a CUDA tensor, most
    likely uncollected cyclic garbage inherited at fork.
  - Workers are persistent: the train workers fork before step 1, but the val workers forked only at the first validation,
    after the in-loop filter (torch_harmonics) and the fp32 graphs existed. D (bf16, no filter) never hit it.
  - A 1-GPU short-epoch repro ran 8 epochs cleanly, so the crash needs DDP or is intermittent; the trigger is unconfirmed.
- **Fix.** `training.num_sanity_val_steps` (train.py; default 0 unchanged) is set to 2 for DT, so the val workers fork at
  startup before any of that exists. Output of the crashed run moved to jcm_p20bt_DT_crash1 (not deleted).
- **Launcher bug, also fixed.** `echo "... $(date) exit=$?"` always printed exit=0, because the `$(date)` substitution
  resets `$?`. Launchers now capture `rc=$?` right after train.py. The eval runner refuses to evaluate unless exit=0.
- **Relaunched** 11:12:51 UTC, config md5 beb75cdc.
- **Update ~11:25 UTC: the real cause of the DT crash, and the fix.**
  - `num_sanity_val_steps` alone did not help: the second launch crashed at the sanity pass, and a GC hook in the
    LightningModule did not help either.
  - A 5-GPU short-epoch repro with `PYTHONFAULTHANDLER=1` (config_jcm_p20bt_DT_forkrepro5.yaml) crashes every time (44
    errors, exit 137). Workers die at startup in DataLoader's `_worker_loop` → `torch.manual_seed` →
    `torch.xpu.manual_seed_all` → `_lazy_call` → `traceback.format_stack`. The allocations there trigger the child's
    cyclic GC, which frees a CUDA tensor inherited from the parent → abort. Training and val forks are both affected.
  - **Fix:** `GCFreezeDataLoader` (data/datamodule.py) runs `gc.collect(); gc.freeze()` in `__iter__`, immediately before
    the fork. The same repro then runs 6 epochs with validation, 0 errors, exit 0. Backups: *.bak_pre_gcfreeze.
  - The LightningModule GC hooks stay; they are harmless. Crashed outputs were moved to jcm_p20bt_DT_crash{1,2,3}.
  - Crashed DDP launches also left orphaned workers (23 processes). Clear them with `kill -KILL -- -PGID` before relaunching.
  - **DT relaunched ~11:24 UTC.**

## 2026-10-24 11:50 UTC: D final (step 10.5k) confirms; DT healthy

- **D_final with filter 36,4, 1996:** all 4 die (days 108 / 284 / 186 / 92).
- **One-step, 200 states:** h16-94 unchanged at 15–26 %; v h0-15 12.8, T h0-15 26.5; **ps 90.7 %**.
- **D verdict:** the bf16 + berHu + state-std-weighted single-step fine-tune improves h0-15 but degrades ps and free-run
  stability. Not a candidate.
- **DT:** past 2 epochs and 2 validations with 0 CUDA errors. 1.04 s/step, loss 0.012–0.022. ETA ~13:10 UTC (~8.7 GPU-h).
