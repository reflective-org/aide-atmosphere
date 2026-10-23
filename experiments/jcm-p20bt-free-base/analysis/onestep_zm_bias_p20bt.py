#!/usr/bin/env python3
"""Teacher-forced one-step zonal-mean u bias per level for a p20bt checkpoint. For N samples of a held-out year:
mean over samples of (pred_zm - truth_zm) [m/s per 6 h], area-weighted over lat, and its rms; plus the truth 6-h zm tendency rms.
A drift of D m/s/day in free runs needs a one-step bias of ~D/4 if it is a plain mean bias. Usage: --ckpt X --gpu 2 --n 240"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/data/paradis_jcm")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--gpu", default="2")
ap.add_argument("--n", type=int, default=240); ap.add_argument("--year", type=int, default=1996)
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import gate_p20bt as G
from omegaconf import OmegaConf
from data.datamodule import Era5DataModule
from trainer import LitParadis
cfg = OmegaConf.load("/data/paradis_jcm/config/config_jcm_p20bt_free.yaml"); OmegaConf.resolve(cfg); cfg.compute.num_devices = 1
cfg.training.dataset.start_date = f"{a.year}-01-01T06:00:00"; cfg.training.dataset.end_date = f"{a.year}-12-31"
dm = Era5DataModule(cfg); dm.setup(stage="fit"); ds = dm.dataset
dev = "cuda"; lit = LitParadis(dm, cfg).to(dev).eval()
lit.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["state_dict"], strict=False)
mean, std = lit._jcm_mean, lit._jcm_std; iu = lit.jcm_physics.i_u
times = np.asarray(ds.ds_input.time.values); fio = G.FastIO(ds, cfg.dataset.root_dir, times, cfg.dataset.n_time_inputs)
stride = dm.num_dyn_inputs_single
w = torch.cos(torch.deg2rad(ds.lat.to(dev).float())); w = w / w.mean()
idx = np.linspace(0, len(times) - 10, a.n).astype(int)
bias = torch.zeros(95, len(w), device=dev); sq = torch.zeros(95, device=dev); tend = torch.zeros(95, device=dev)
phys_only = torch.zeros(95, len(w), device=dev)
with torch.no_grad():
    for i in idx:
        x, y = fio.get(int(i)); x = x.to(dev); y = y.to(dev)
        x = x[None] if x.dim() == 3 else x; y = y[None] if y.dim() == 3 else y
        if x.dim() == 5: x = x[:, 0]
        if y.dim() == 5: y = y[:, 0]
        out = lit._apply_jcm_physics(lit(x), x, y, prescribe=False, qbo=False)
        up = (out * std + mean)[0][iu].mean(-1); ut = (y * std + mean)[0][iu].mean(-1)   # (95, lat) zonal means
        u0 = (x[:, stride:2 * stride] * std + mean)[0][iu].mean(-1)                        # current-state zonal mean
        d = up - ut; bias += d; sq += (d ** 2 * w).mean(-1); tend += ((ut - u0) ** 2 * w).mean(-1)
n = len(idx); bias /= n
gb = (bias * w).mean(-1); rb = (bias ** 2 * w).mean(-1).sqrt(); mxb = bias.abs().max(-1)
print(f"ckpt {os.path.basename(a.ckpt)}  n={n}  year {a.year}")
print("lev  gm_bias  rms_latbias  max|latbias|(lat)  rms_err  truth_tend_rms   [m/s per 6h step]")
for l in list(range(0, 95, 6)) + [84, 87, 90, 92, 93, 94]:
    print(f"h{l:<3d} {float(gb[l]):+8.3f} {float(rb[l]):10.3f} {float(mxb.values[l]):10.3f} ({float(ds.lat[int(mxb.indices[l])]):+5.1f}) "
          f"{float((sq[l]/n).sqrt()):8.3f} {float((tend[l]/n).sqrt()):10.3f}", flush=True)
