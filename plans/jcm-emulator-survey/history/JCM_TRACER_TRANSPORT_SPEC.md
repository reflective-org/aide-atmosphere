# JCM tracer transport — emulate the OPERATOR, not the tracers

Spec for a second machine. Self-contained: needs only `/data/JCM_stripped/jcm-strat-phase11/runs/`
and a GPU. **Independent of the dynamics run** training on the main box (see §8), because the
operator is trained against TRUTH winds.

Written 2026-09-28. Every number tagged MEASURED was produced by a command in the session that
wrote this file; everything else is tagged ESTIMATE or TO VERIFY.

---

## 1. Objective

Advance an arbitrary tracer field one 6-h step under the JCM flow:

    q_{t+1} = A(flow_t) · q_t          A linear in q

so that the same operator works on tracers it never saw, **including tracers modified by an
external process between steps** (aerosol microphysics, chemistry, injection). Must stay stable
over multi-year rollouts.

**Success = the multi-year gates in §6.** Single-step skill is explicitly NOT success; see §7.1.

## 2. Why an operator and not extra output channels

Transport is linear in the tracer: `dq/dt + div(u q) = S`. For a fixed flow the map
`q_t -> q_{t+1}` is a linear operator. Design follows from that:

| | tracers as predicted channels | learned operator A(flow) |
|---|---|---|
| externally modified tracer | off-manifold, undefined behaviour | works — A never sees q's climatology |
| unseen tracer | needs retraining | works |
| training data | 24 channels to fit | **24 tracers = 24x data for ONE operator** |
| mass conservation | not enforced | structural (§4.2) |
| stability | hope | structural: spectral radius = 1 (§4.2) |

The channel route fails the stated use case (transport -> modify externally -> transport again),
because a network that learned the joint distribution of (dynamics, that tracer) is off-manifold
the moment the tracer is perturbed. This is the one requirement that decides the architecture.

Prior related decision in this project went the other way ("transport head DROPPED, tracers become
predicted channels") on the grounds that *no extensibility was needed*. That premise does not hold
here: extensibility to externally modified tracers is the whole point.

## 3. Data — MEASURED

Source: `/data/JCM_stripped/jcm-strat-phase11/runs/p11a_{YYYY}0101/longrun_day*.nc`,
20 years 1990–2009, 37 chunk files each, all `exit=0`.

- grid **96 lat x 192 lon x 63 levels**, 985.3 hPa -> 0.0 hPa model top, **6 h** sampling
- netCDF dim order is `(time, level, lon, lat)` — note **lon before lat**
- 1460 frames/yr (1464 on leap years 2000/2004/2008)

Tracer inventory (ranges from 1997 day-1 chunk, MEASURED):

| tracer | min | max | note |
|---|---|---|---|
| `aoa`, `aoa150`, `aoa_sfc` | ~ -7.6e-06 | 2.0e+03 | age of air, days. Tiny NEGATIVE undershoots exist in TRUTH |
| `cfc11`, `n2o` | 4.2e-15 | 1.0 | normalised |
| `sai` | 0.132 | 8.18 | |
| `pulse_1..5` (+ `_box`) | 1.1e-04 | 1.1e-02 | localized releases — the transport Green's function |
| `src_1..4` (+ `_box`) | 1.2e-03 | 0.20 | |
| `specific_humidity`, `geopotential` | | | also present |

24 tracer fields total. **The current dynamics adapter (`jcm_to_zarr.py`) drops all of them** —
it converts only `u_wind, v_wind, temperature, omega, normalized_surface_pressure` (253 features).

Two facts that matter and are already known:
- `aoa` global mean sits at ~1503 days (≈4.1 yr) and is steady => it carries an **interior clock
  source plus surface reset**. It is NOT a passive tracer.
- Truth `aoa` contains small negative values (-7.6e-6), i.e. the real dycore's advection itself
  slightly overshoots. Do not "fix" this in the data (see §7.4).

## 4. Architecture

### 4.1 Reuse and extend PARADIS's semi-Lagrangian advection

`/data/paradis_jcm/model/advection.py` already computes departure points and gathers with
`grid_sample`. MEASURED limitations that must change:

1. **It is 2-D.** `grid_sample` over (H, W) only. No vertical displacement. This is exactly the
   known blocker: with no vertical remap, `pulse_4` froze at 4.5 hPa against truth's 217 hPa.
   **Adding the vertical is the main engineering task.**
2. **It interpolates with `bicubic`.** Bicubic weights are not non-negative — they overshoot,
   can produce negative tracer, and break the stability guarantee in §4.2.
3. It acts on latent velocity channels, not on a physical tracer field.

### 4.2 The operator and its structural guarantees

Per cell, a network reads the local flow `(u, v, omega, p_s, dp)` and emits a 3-D departure point
`(dlon, dlat, dlev)`. The update is a **trilinear gather** at that departure point:

    q_{t+1}(i) = sum_j w_ij q_t(j),   w_ij >= 0,   sum_j w_ij = 1

Trilinear interpolation weights are non-negative and sum to one **by construction**. Therefore A is
row-stochastic and:

- **spectral radius exactly 1** => q cannot blow up, for any number of steps. Indefinite tracer
  stability is structural, not trained-for.
- **no new extrema** => positivity preserved; a positive tracer stays positive.
- **mass conservation**: enforce in flux form on the Dp-weighted field. Diagnose
  `sum(Dp*q)` every step; it must close to roundoff (NOT be rescaled — §7.4).

The remaining failure mode is **over-diffusion** (variance decay), which is measurable and was
already observed at 2.3x on a held-out site at year 1. That is the quantity to optimise against.

Optional second term: a learned anti-diffusive correction to offset trilinear's numerical
diffusion. It must also be linear in q. Adding it forfeits the monotonicity guarantee unless it
is flux-limited — implement the limiter, or leave the term out and accept diffusion. Decide this
with the §6 gates, not a priori.

Trade-off to settle empirically: trilinear (guaranteed, more diffusive) vs tricubic (sharper,
overshoots — note truth itself overshoots slightly). Run both through §6.

### 4.3 Conditioning, not memorising

The network must see **only the flow**, never the tracer's identity or climatology. One operator,
shared across all tracers. This is what makes it apply to an externally modified field. Do not add
per-tracer embeddings; that quietly reintroduces the channel approach.

## 5. Training protocol

**Decoupled from the dynamics emulator: use TRUTH winds.** The operator depends only on the flow,
so it trains and validates without the dynamics model existing. Couple them only at the end.

- **Phase 0 (do this first — cheap, and it gates everything).**
  - For each tracer, measure whether it is passive over one step: compare `q_{t+1}` against a
    reference advection of `q_t`, and check `d/dt sum(Dp*q)`. Tracers with interior sources
    (`aoa` at minimum) must either be excluded or have their known source subtracted before use
    as pure-transport targets. Do not skip: training pure transport on a tracer with a source
    teaches the operator to fake the source.
  - Confirm the vertical coordinate handling (hybrid a/b -> pressure; `p = a + b*p_s`) and that
    `omega` maps to a level displacement consistently.
  - Sort chunk files **numerically** — `sorted(glob(...))` is lexicographic and puts `day10`
    before `day5`. This already produced one wrong measurement while writing this spec.
- **Loss**: relative error per tracer (ranges span 1e-15 to 2e3 — a plain MSE is dominated by
  `aoa`). Normalise per tracer, or train on log/relative residual. Area-weight by cos(phi) and
  Dp-weight column integrals.
- **Rollout curriculum**: single-step first, then extend. Single-step is necessary but NOT
  sufficient (§7.1), so the curriculum must reach at least month-scale before any verdict.
- **Splits**: train 1997–2008, val 2009, test 1996 — matches the dynamics run so results compose.
  Normalisation stats fit on TRAIN ONLY (this bit the dynamics run: the shipped accumulator had
  pooled the test year into an undecomposable sum).

## 6. Gates — the falsifiers, with thresholds

Cheap because tracer-only rollouts with prescribed truth winds are just repeated gathers; stencil
weights can be precomputed per timestep. **This is what makes multi-year validation affordable and
is the main reason to prefer this design.**

| gate | metric | pass |
|---|---|---|
| G1 vertical penetration | `pulse_4` height after 1 yr | reaches ~217 hPa, not frozen at 4.5 |
| G2 over-diffusion | variance retained vs truth at yr 1 | ratio -> 1; prior work measured 2.3x too diffusive |
| G3 mass | `sum(Dp*q)` drift over 10 yr | closes to roundoff without rescaling |
| G4 age of air | `aoa` profile after multi-yr | matches truth's ~1503 d steady state |
| G5 positivity | min(q) over 10 yr | >= 0 (trilinear) or no worse than truth's -7.6e-6 |
| G6 **extensibility** | operator applied to a PERTURBED tracer | behaves like the dycore on the same perturbation |

**G6 is the one that justifies the whole design.** Perturb a tracer (scale a region, inject a
blob), advance with both the operator and the real dycore, compare. A channel-based model has no
defensible answer here; this operator should. Run it early — it is cheap and it is the product
requirement.

## 7. Traps — all MEASURED in this project, do not rediscover

1. **Single-step transport skill does NOT predict rollout behaviour.** A variant scored +0.0473 on
   468/468 single steps yet was worse on 0/10 tracers from month 3, with 2.3x over-diffusion at
   yr 1. Never accept a single-step number as a verdict.
2. **The vertical, not the horizontal, is the multi-year blocker.** The horizontal operator was
   ~17%/yr accurate while the missing vertical froze `pulse_4` at 4.5 vs 217 hPa. A "few-percent
   per step" term is not a few-percent-per-year term.
3. **Metrics here are noisy.** Free-run survival spread 130–369 d across seeds; same config+seed
   gave 132 vs 40 d. Use >= 4 ICs and report medians. n=1 is not decision-grade.
4. **Never rescale, clamp, or epsilon a tracer to make a run work.** Truth itself carries small
   negatives. Clamping masks collapse — a prior run produced bit-identical val MSE across epochs
   because a clamp was saturating, and NaN/Inf guards did not catch it. Surface the failure.
5. **Don't take the last checkpoint.** More rollout training has repeatedly degraded multi-year
   behaviour in this project. Keep every checkpoint (`save_top_k=-1`) and score the §6 gates
   across them, then pick the peak.
6. **Area-weight with cos(phi); Dp-weight column integrals.** Gaussian grid — use Gauss-Legendre
   weights (`/data/paradis_jcm/utils/loss.py` has the exact-node version).

## 8. Target machine and MEASURED throughput

Second box: **8x NVIDIA H200, 143771 MiB (140 GB) each**, compute cap 9.0, driver 580.126.20 —
identical to the box these numbers came from, so they transfer directly. Assumes `/data` is
mounted (this spec lives there).

Note `/data/CLAUDE.md` records "8x H100 80 GB" — that is **stale**; the cards are H200 140 GB.
Budget memory against 140 GB.

If the driver lives in the GPU Operator container as it does here, `nvidia-smi` is
`sudo chroot /run/nvidia/driver nvidia-smi`, and every reboot wipes `/dev/nvidia*`, after which
**torch silently runs on CPU**. Check `ls /dev/nvidia0` and
`torch.cuda.is_available()` before any launch — that cost 19 h once. Training needs
`LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu`.

GPUs 0 and 1 are off-limits **on the original box only**; confirm the policy on the new one
before assuming all 8 are usable.

### MEASURED on this hardware (25M-param dynamics model, 96x192x63, 253 features)

Compute ceiling, cached batch, one GPU:

| grad_ckpt | batch | s/pass | samples/s/GPU | peak GB |
|---|---|---|---|---|
| on | 1 | 0.1293 | 7.75 | 3.1 |
| off | 1 | 0.1010 | 9.90 | 12.9 |
| off | 4 | 0.3441 | 11.62 | 50.0 |
| off | 8 | 0.6641 | 12.05 | 99.4 |

99.4 GB fits in 140 GB but would NOT fit an 80 GB card — one reason to get the hardware right.
Gradient checkpointing is a **1.28x loss** whenever memory is not the constraint; here it never is.

Dataloader, per rank, 104 cores:

| workers | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| samples/s/rank | 4.84 | 8.34 | 15.29 | 21.27 |

**The default 4 workers (4.84) sits BELOW the 7.75 compute ceiling** — the job is dataloader-bound
and adding GPUs scales at ~65%. Real 6-rank run: 0.706–0.720 s/step, 8.33 samples/s/GPU, 23 GB/GPU.
Note 12 workers gave 15.29/rank measured on an *idle* box; with 6 ranks x 12 workers contending for
104 cores the achieved figure was 8.33, i.e. **74% of the single-GPU prediction**. Benchmark at the
rank count you will actually run, and re-measure on the new box — core count may differ.

Storage is NOT the lever: raw NFS reads measure **2713 MB/s** (5.6 ms per 15 MB chunk) and the
corpus is page-cached (1007 GB RAM here). NVMe staging buys nothing; do not repeat that experiment.

Reusable harnesses: `/data/paradis_jcm/bench_config.py` (compute) and `bench_io2.py` (dataloader,
shuffled distinct samples — an earlier version re-read the same first 20 batches and was
cache-flattered).

## 9. Storage and conversion — ESTIMATE

Derived from MEASURED dynamics conversion (253 features = 22.3 GB/yr, ~6 min/yr):
~88 MB per feature-level-year, so **5.5 GB/yr per 63-level tracer**.

| scope | GB/yr | 13 yr (1997–2009) | convert time |
|---|---|---|---|
| 10 tracers (aoa, n2o, cfc11, sai, pulse_1..5, +1) | 55 | **~715 GB** | ~3.3 h |
| all 24 tracers | 132 | ~1.7 TB | ~8 h |

`/data` had 17 TB free at writing (94% used of 245 TB). **Start with the 10-tracer subset.**

Write tracers to a **separate zarr** (`/data/jcm_tracers_c1/`), same time axis, so the dynamics
corpus stays lean. Note the loader globs `root_dir/*` — a stray directory inside a data root
collides; keep them separate roots.

Operator network is far smaller than the 25M-param dynamics model (it emits 3 displacements per
cell, not 253 fields), so training cost is dominated by data movement, not FLOPs. ESTIMATE: well
under the dynamics run's 72 GPU-h.

**Before any long run**: report s/step and projected GPU-h from a short measurement. On the main
box, skipping this launched a 146 GPU-h job that was dataloader-bound and should have been 72 —
the default 4 workers delivered 4.84 samples/s/rank against a 7.75 compute ceiling. See
`/data/paradis_jcm/bench_config.py` and `bench_io2.py`, which are reusable.

## 10. Deliverables

1. `tracer_to_zarr.py` — adapter for the 10-tracer subset (fork `jcm_to_zarr.py`; keep the
   `if __name__ == "__main__"` guard — a missing one once launched a full conversion from an
   import and corrupted a year mid-verification).
2. `phase0_verify.py` — §5 Phase 0 checks. Must run and pass before training.
3. `transport_operator.py` — 3-D semi-Lagrangian operator, trilinear, flow-conditioned.
4. `train_transport.py` — decoupled training against truth winds.
5. `gates.py` — G1–G6 with the §6 thresholds.
6. A results doc reporting each gate, with the commands that produced each number.

## 11. Relationship to the dynamics run

The dynamics emulator (`/data/paradis_jcm`, config `config_jcm_free_big_fast.yaml`, 12 yr
1997–2008, 60k steps) supplies the flow at deployment. Until it is gated, develop transport
against truth winds. Final integration test: run the operator on EMULATED winds and re-score §6 —
the gap between truth-wind and emulated-wind scores isolates dynamics error from transport error,
which is the decomposition this project has never had.

## 12. Launch pattern — operational, learned the hard way

Long jobs detached so they survive logout, and **waits keyed on output markers, never `pgrep -f`**:

```bash
# launch
cd <workdir>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # confirm the 0/1 policy on this box first
export LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu
setsid nohup ./run_transport.sh > run_transport.log 2>&1 < /dev/null &

# capture the REAL pid: $! is the setsid wrapper, which exits immediately.
# Use a bracketed pattern so the grep cannot match its own command line.
ps -eo pid,cmd --no-headers | grep "run_[t]ransport.sh" | awk '{print $1}' | head -1 > run.pid

# wait on the script's own marker, not on a process table scan
until grep -q "TRAINING DONE" run_transport.log; do sleep 300; done
```

Four failures this pattern exists to prevent, all of which happened:

1. **`pgrep -f <pattern>` matches the wait loop's own command line** -> a loop that can never exit,
   and a `pkill` that killed the issuing command. Cost two days of wall time.
2. **`$!` after `setsid` is the wrapper, not the job.** A PID file written that way reported a
   perfectly healthy run as dead.
3. **A failure-detector regex of `Error|Traceback` matches Lightning's benign startup line
   `Error log: ...`** and aborts the watch during initialisation. Match
   `Traceback|CUDA out of memory|exit=[1-9]` instead.
4. **Check the entry point actually runs.** `trainer.py` here defines the LightningModule and has
   no `__main__`; invoking it exits 0 in 4 seconds and a naive wrapper prints "DONE". The real
   entry point is `train.py`. Verify a log line advances before trusting an exit code.

Also: keep every checkpoint (`save_top_k=-1`, step-based) — ~300 MB each — and score the §6 gates
across them rather than taking the last. Snapshot the config per run; never edit one in place.
