# Log: jcm-p20bt-free-base

## 2026-10-22: corpus, physics, launch

- Converted p20bt 1990-2009 to /mnt/nvme_stage2/jcm_p20bt_c1, with stats from train years only.
- Ported the analytic physics to torch and verified it against JAX.
- Ran the overfit, smoke and RAM checks.
- Launched jcm_p20bt_free at 23:37 UTC under a RAM watchdog (minimum available RAM 312 GB).

## 2026-10-23: evaluation and diagnosis

- Training finished at 13:36 UTC.
- Unfiltered ladder: no checkpoint is good.
- Diagnosed the top-level runaway:
  - it is an unstable m=0, high-l zonal-mean v mode at h94 that starts at the poles;
  - the one-step check cannot see it;
  - it runs away even with only h94 free.
- An m=0-only vector-SH filter at inference makes the last checkpoint survive 7/7 ICs, 6/7 good, but still below the
  climatology floors.
- Next: carry the m=0 damping into training. This needs sign-off (it changes the model's output path), plus a tiny run.
- Manifest gpu_hours for the gate jobs (9.0, 6.6) and the smoke (0.2) are estimates from wall time x GPUs, not measured; the training 84.0 is measured.
