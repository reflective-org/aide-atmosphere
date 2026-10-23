#!/usr/bin/env python3
"""Where does a p20bt free rollout blow up first? Per day: max|u| and cos-weighted rms(u - truth) per band
(nudged h0-15, strat h16-73, meso h74-93, sponge h94), the level holding the global max|u|, and the fraction of
zonal-wavenumber power at k>=48 relative to truth (grid-scale noise test). Usage: --ckpt X --ic 0 --days 40 --gpu 2"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/data/paradis_jcm")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--ic", type=int, default=0)
ap.add_argument("--days", type=float, default=40); ap.add_argument("--gpu", default="2"); ap.add_argument("--year", type=int, default=1996)
ap.add_argument("--clim-h94-zm", action="store_true", help="DIAGNOSTIC ONLY: replace zonal-mean u at h94 by its train-years harmonic climatology")
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import gate_p20bt as G                          # mallopt fix applied at import
from omegaconf import OmegaConf
from data.datamodule import Era5DataModule
from trainer import LitParadis
cfg = OmegaConf.load("/data/paradis_jcm/config/config_jcm_p20bt_free.yaml"); OmegaConf.resolve(cfg); cfg.compute.num_devices = 1
cfg.training.dataset.start_date = f"{a.year}-01-01T06:00:00"; cfg.training.dataset.end_date = f"{a.year}-12-31"
dm = Era5DataModule(cfg); dm.setup(stage="fit"); ds = dm.dataset
dev = "cuda"; lit = LitParadis(dm, cfg).to(dev).eval()
lit.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["state_dict"], strict=False)
mean, std = lit._jcm_mean, lit._jcm_std; iu = lit.jcm_physics.i_u
stride = dm.num_dyn_inputs_single; times = np.asarray(ds.ds_input.time.values)
fio = G.FastIO(ds, cfg.dataset.root_dir, times, cfg.dataset.n_time_inputs)
from model.jcm_physics_p20bt import gregorian_tyear
TY, _ = gregorian_tyear(times); OFF = cfg.dataset.n_time_inputs + int(ds.prediction_shift)
CLIM = G.load_harmonics("/mnt/nvme_stage2/jcm_p20bt_meta/clim_harmonics_p20bt_1997_2008.npz", [str(f) for f in ds.dyn_output_features], dev)
w = torch.cos(torch.deg2rad(ds.lat.to(dev).float())).view(1, -1, 1); w = w / w.mean()
bands = {"nud0-15": (0, 16), "str16-73": (16, 74), "meso74-93": (74, 94), "spg94": (94, 95)}
def hf(u):   # fraction of zonal power at k>=48, per band-pooled field
    P = torch.fft.rfft(u.float(), dim=-1).abs() ** 2
    return float(P[..., 48:].sum() / P.sum())
prev = cur = None
print("day  " + "  ".join(f"{b:>22s}" for b in bands) + "   argmax_lev  hf_strat(pred/truth) hf_meso(pred/truth)")
with torch.no_grad():
    for t in range(int(a.days * 4)):
        x, y = fio.get(a.ic + t); x = x.to(dev)[None] if x.dim() == 3 else x.to(dev); y = y.to(dev)[None] if y.dim() == 3 else y.to(dev)
        if x.dim() == 5: x = x[:, 0]
        if y.dim() == 5: y = y[:, 0]
        if cur is not None: x = x.clone(); x[:, :stride] = prev; x[:, stride:2 * stride] = cur
        out = lit._apply_jcm_physics(lit(x), x, y, prescribe=False, qbo=False)
        if a.clim_h94_zm:
            tyt = float(TY[a.ic + t + OFF]); cu = CLIM(tyt)[iu][94]              # physical u at h94 from climatology
            ph = out * std + mean; u94 = ph[0, iu[94]]
            ph[0, iu[94]] = u94 - u94.mean(-1, keepdim=True) + cu.mean(-1, keepdim=True)
            out = (ph - mean) / std
        up = (out * std + mean)[0][iu]; ut = (y * std + mean)[0][iu]
        prev, cur = (cur if cur is not None else x[:, :stride]), out
        if t % 4 == 3:
            cells = []
            for b, (l0, l1) in bands.items():
                mx = float(up[l0:l1].abs().max()); rm = float((((up[l0:l1] - ut[l0:l1]) ** 2) * w).mean().sqrt())
                cells.append(f"max {mx:6.1f} rms {rm:6.2f}")
            lev = int(up.abs().amax(dim=(1, 2)).argmax())
            zp, zt = up[94].mean(-1), ut[94].mean(-1)
            print(f"      h94 zonal-mean |u| max pred {float(zp.abs().max()):6.1f} truth {float(zt.abs().max()):6.1f}   "
                  f"anomaly |u| max pred {float((up[94]-zp[:,None]).abs().max()):6.1f} truth {float((ut[94]-zt[:,None]).abs().max()):6.1f}   "
                  f"h90 zm pred {float(up[90].mean(-1).abs().max()):6.1f} truth {float(ut[90].mean(-1).abs().max()):6.1f}", flush=True)
            print(f"{(t+1)/4:4.0f}  " + "  ".join(f"{c:>22s}" for c in cells) +
                  f"   h{lev:<3d}        {hf(up[16:74])/hf(ut[16:74]):6.2f}             {hf(up[74:94])/hf(ut[74:94]):6.2f}", flush=True)
            if not np.isfinite(float(up.abs().max())) or float(up[16:94].abs().max()) > 300: print("DIED"); break
