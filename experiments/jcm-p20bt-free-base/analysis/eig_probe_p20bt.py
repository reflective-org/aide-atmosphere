#!/usr/bin/env python3
"""Leading eigenmode of the emulator's one-step map restricted to the zonal-mean (u, v, T) at levels L0..L1, by power
iteration with finite differences around a TRUTH state (perturbation in std units, applied to both input frames, zonally
uniform). |lambda| > 1 means an unstable zonal-mean mode exists even with everything else held at truth. Usage: --ckpt X"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/data/paradis_jcm")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--gpu", default="2")
ap.add_argument("--l0", type=int, default=94); ap.add_argument("--l1", type=int, default=95); ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--samples", default="0,400,800,1200"); ap.add_argument("--eps", type=float, default=0.05)
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
mean, std = lit._jcm_mean, lit._jcm_std
FN = [str(f) for f in ds.dyn_output_features]; stride = dm.num_dyn_inputs_single
CH = [FN.index(f"{v}_h{l}") for v in ("u_wind", "v_wind", "temperature") for l in range(a.l0, a.l1)]
times = np.asarray(ds.ds_input.time.values); fio = G.FastIO(ds, cfg.dataset.root_dir, times, cfg.dataset.n_time_inputs)
lat = ds.lat.numpy(); w = torch.cos(torch.deg2rad(ds.lat.to(dev).float())).sqrt()   # sqrt(cos) so the 2-norm is area-weighted
def step(x, y):
    out = lit._apply_jcm_physics(lit(x), x, y, prescribe=False, qbo=False)
    return out[0, CH].mean(-1)                                       # (nch, lat), normalised units
torch.manual_seed(0)
for s in [int(v) for v in a.samples.split(",")]:
    x, y = fio.get(s); x = x.to(dev); y = y.to(dev)
    x = x[None] if x.dim() == 3 else x; y = y[None] if y.dim() == 3 else y
    if x.dim() == 5: x = x[:, 0]
    if y.dim() == 5: y = y[:, 0]
    with torch.no_grad():
        z0 = step(x, y); p = torch.randn(len(CH), len(lat), device=dev) / w; p /= (p * w).norm()
        for it in range(a.iters):
            xp = x.clone()
            for fr in (0, 1): xp[0, [fr * stride + c for c in CH]] += a.eps * p[:, :, None]
            jp = (step(xp, y) - z0) / a.eps
            lam = float((jp * w).norm()); sgn = float(torch.sign((jp * p * w * w).sum()))
            p = jp / (jp * w).norm()
    m = p.cpu().numpy(); nl = a.l1 - a.l0
    shares = [float((m[k * nl:(k + 1) * nl] ** 2 * (w.cpu().numpy() ** 2)).sum()) for k in range(3)]
    ju = int(np.abs(m[0:nl]).max(0).argmax()); jt = int(np.abs(m[2 * nl:]).max(0).argmax())
    print(f"sample {s} ({str(times[s])[:10]}): |lambda| {lam:.3f} sign {sgn:+.0f}  energy share u/v/T {shares[0]:.2f}/{shares[1]:.2f}/{shares[2]:.2f}"
          f"  u peak lat {lat[ju]:+.1f}  T peak lat {lat[jt]:+.1f}", flush=True)
    print("   u-mode (every 8th lat):", " ".join(f"{v:+.2f}" for v in m[0, ::8]), flush=True)
    print("   T-mode (every 8th lat):", " ".join(f"{v:+.2f}" for v in m[2 * nl, ::8]), flush=True)
