#!/usr/bin/env python3
"""Zonal-mean self-gain of the emulator's top levels. Add a zonally uniform delta (m/s) to u at levels L0..L1 in both input
frames of a TRUTH state; gain = cos-weighted mean over the band of (pred_zm(perturbed) - pred_zm(clean)) / delta after one 6-h
step. A stable top needs gain < 1 by about the truth's damping; we compare against the lag-1 (6 h) autocorrelation of the
truth h94 zonal-mean anomaly (from the train-years harmonic climatology). Usage: --ckpt X --gpu 2"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/data/paradis_jcm")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--gpu", default="2"); ap.add_argument("--n", type=int, default=40)
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import gate_p20bt as G
from omegaconf import OmegaConf
from data.datamodule import Era5DataModule
from trainer import LitParadis
cfg = OmegaConf.load("/data/paradis_jcm/config/config_jcm_p20bt_free.yaml"); OmegaConf.resolve(cfg); cfg.compute.num_devices = 1
cfg.training.dataset.start_date = "1996-01-01T06:00:00"; cfg.training.dataset.end_date = "1996-12-31"
dm = Era5DataModule(cfg); dm.setup(stage="fit"); ds = dm.dataset
dev = "cuda"; lit = LitParadis(dm, cfg).to(dev).eval()
lit.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["state_dict"], strict=False)
mean, std = lit._jcm_mean, lit._jcm_std; iu = lit.jcm_physics.i_u
times = np.asarray(ds.ds_input.time.values); fio = G.FastIO(ds, cfg.dataset.root_dir, times, cfg.dataset.n_time_inputs)
stride = dm.num_dyn_inputs_single
w = torch.cos(torch.deg2rad(ds.lat.to(dev).float())); w = w / w.mean()
idx = np.linspace(0, len(times) - 10, a.n).astype(int)
sd = std.view(-1)[iu].view(-1); print("u std at h84,h90,h94:", [round(float(sd[l]), 2) for l in (84, 90, 94)], " mean:", [round(float(mean.view(-1)[iu][l]), 2) for l in (84, 90, 94)])
def run(x):
    out = lit._apply_jcm_physics(lit(x), x, x[:, stride:2 * stride] if False else Y, prescribe=False, qbo=False)
    return (out * std + mean)[0][iu].mean(-1)                                 # (95, lat)
for (L0, L1) in [(94, 95), (90, 95), (84, 95), (74, 84)]:
    for d in [-20., -5., 5., 20.]:
        g = []
        with torch.no_grad():
            for i in idx:
                x, Y = fio.get(int(i)); x = x.to(dev); Y = Y.to(dev)
                x = x[None] if x.dim() == 3 else x; Y = Y[None] if Y.dim() == 3 else Y
                if x.dim() == 5: x = x[:, 0]
                if Y.dim() == 5: Y = Y[:, 0]
                globals()["Y"] = Y
                z0 = run(x); xp = x.clone()
                for fr in (0, 1):
                    for l in range(L0, L1): xp[0, fr * stride + iu[l]] += d / std.view(-1)[iu[l]]
                z1 = run(xp)
                g.append(float((((z1 - z0)[L0:L1] * w).mean()) / d))
        print(f"band h{L0}-h{L1-1}  delta {d:+5.0f}  gain {np.mean(g):.3f} +- {np.std(g):.3f}", flush=True)
# truth persistence of the h94 zonal-mean anomaly (6 h lag), over 1996
CLIM = G.load_harmonics("/mnt/nvme_stage2/jcm_p20bt_meta/clim_harmonics_p20bt_1997_2008.npz", [str(f) for f in ds.dyn_output_features], dev)
from model.jcm_physics_p20bt import gregorian_tyear
TY, _ = gregorian_tyear(times); an = []
with torch.no_grad():
    for i in range(0, len(times) - 10, 2):
        _, Y = fio.get(i); Y = Y.to(dev); Y = Y[None] if Y.dim() == 3 else Y
        if Y.dim() == 5: Y = Y[:, 0]
        z = (Y * std + mean)[0][iu[94]].mean(-1) - CLIM(float(TY[i + cfg.dataset.n_time_inputs + int(ds.prediction_shift)]))[iu][94].mean(-1)
        an.append(z.cpu().numpy())
an = np.array(an)   # every 12 h
for lag in (1, 2, 8):
    r = [np.corrcoef(an[:-lag, j], an[lag:, j])[0, 1] for j in range(an.shape[1])]
    print(f"truth h94 zm-anomaly autocorr at lag {lag*12} h: lat-mean {np.nanmean(r):.3f}  anomaly std {an.std():.2f} m/s")
