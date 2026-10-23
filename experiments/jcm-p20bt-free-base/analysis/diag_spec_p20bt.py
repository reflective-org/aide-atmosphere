#!/usr/bin/env python3
"""Where does a p20bt free rollout blow up first? Per day: max|u| and cos-weighted rms(u - truth) per band
(nudged h0-15, strat h16-73, meso h74-93, sponge h94), the level holding the global max|u|, and the fraction of
zonal-wavenumber power at k>=48 relative to truth (grid-scale noise test). Usage: --ckpt X --ic 0 --days 40 --gpu 2"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/data/paradis_jcm")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--ic", type=int, default=0)
ap.add_argument("--days", type=float, default=40); ap.add_argument("--gpu", default="2"); ap.add_argument("--year", type=int, default=1996)
ap.add_argument("--free-from", type=int, default=0, help="DIAGNOSTIC: levels below this (and ps) are reset to truth every step; only levels >= free-from run free")
ap.add_argument("--vsht", action="store_true", help="DIAGNOSTIC: project every level of output (u, v) through a vector SHT round trip each step")
ap.add_argument("--vfilt", default="", help="DIAGNOSTIC: alpha,p -> multiply vector-SH coeffs of (u,v) by exp(-alpha (l/(lmax-1))^p) each step (implies --vsht)")
ap.add_argument("--m0only", action="store_true", help="filter only the zonal-mean (m=0) coefficients")
ap.add_argument("--spec-every", action="store_true")
ap.add_argument("--clim-h94-zm", action="store_true", help="DIAGNOSTIC ONLY: replace zonal-mean u at h94 by its train-years harmonic climatology")
a = ap.parse_args(); a.vsht = a.vsht or bool(a.vfilt); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
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
import re; FN = [str(f) for f in ds.dyn_output_features]; print("features", FN[:3], FN[-3:])
FORCED = torch.tensor([not (re.search(r"_h(\d+)$", f) and int(re.search(r"_h(\d+)$", f).group(1)) >= a.free_from) for f in FN])
print("forced channels", int(FORCED.sum()), "of", len(FN))
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
        if a.vsht:
            if "VS" not in globals():
                import torch_harmonics as th
                VS = (th.RealVectorSHT(96, 192, grid="legendre-gauss").to(dev), th.InverseRealVectorSHT(96, 192, grid="legendre-gauss").to(dev))
                IUo = [FN.index(f"u_wind_h{l}") for l in range(95)]; IVo = [FN.index(f"v_wind_h{l}") for l in range(95)]
                FLIP = bool(ds.lat[0] < ds.lat[-1])
            ph = out * std + mean; uv = torch.stack([ph[0, IUo], ph[0, IVo]], 1)
            uv = uv.flip(-2) if FLIP else uv; C = VS[0](uv.double())
            if a.vfilt:
                al, pp = [float(v) for v in a.vfilt.split(",")]; ll = torch.arange(C.shape[-2], device=dev, dtype=torch.float64)
                fw = torch.exp(-al * (ll / (C.shape[-2] - 1)) ** pp)
                if a.m0only: C = C.clone(); C[..., 0] = C[..., 0] * fw
                else: C = C * fw.view(-1, 1)
            r = VS[1](C).float(); r = r.flip(-2) if FLIP else r
            ph = ph.clone(); ph[0, IUo] = r[:, 0]; ph[0, IVo] = r[:, 1]; out = (ph - mean) / std
        if a.free_from > 0: out = out.clone(); out[:, FORCED.to(dev)] = y[:, FORCED.to(dev)]
        up = (out * std + mean)[0][iu]; ut = (y * std + mean)[0][iu]
        prev, cur = (cur if cur is not None else x[:, :stride]), out
        if t in (3, 11, 19, 31) or (t % 4 == 3 and a.spec_every):
            import torch_harmonics as th
            if "VS2" not in globals():
                VS2 = th.RealVectorSHT(96, 192, grid="legendre-gauss").to(dev); FL2 = bool(ds.lat[0] < ds.lat[-1])
            def coef(z, l):
                uv = torch.stack([(z * std + mean)[0, FN.index(f"u_wind_h{l}")], (z * std + mean)[0, FN.index(f"v_wind_h{l}")]])
                uv = uv.flip(-2) if FL2 else uv; return VS2(uv.double())        # (2: div/curl, l, m)
            for lv in (94, 90):
                e = coef(out, lv) - coef(y, lv); P = e.abs() ** 2; T = coef(y, lv).abs() ** 2
                edges = [0, 4, 8, 16, 32, 48, 64, 96]
                row = lambda X, k: " ".join(f"{float(X[k, l0:l1, 0].sum()):9.2e}" for l0, l1 in zip(edges[:-1], edges[1:]))
                print(f"  SPEC t={(t+1)/4:5.2f}d h{lv} m=0 err power by l {edges}: DIV {row(P,0)} | CURL {row(P,1)}")
                print(f"  SPEC t={(t+1)/4:5.2f}d h{lv} m=0 TRUTH power by l          : DIV {row(T,0)} | CURL {row(T,1)}")
                print(f"  SPEC t={(t+1)/4:5.2f}d h{lv} err share m=0 {float(P[:, :, 0].sum()/P.sum()):.2f}  m=1 {float(P[:, :, 1].sum()/P.sum()):.2f}  div share {float(P[0].sum()/P.sum()):.2f}", flush=True)
        if t in (0, 3, 7, 11, 19, 31, 47):
            vp = (out * std + mean)[0][FN.index("v_wind_h94")].mean(-1); vt = (y * std + mean)[0][FN.index("v_wind_h94")].mean(-1)
            zp, zt = up[94].mean(-1), ut[94].mean(-1); L = ds.lat.numpy()
            print(f"  t={(t+1)/4:5.2f}d lat:  " + " ".join(f"{L[j]:+6.0f}" for j in range(0, 96, 8)) + f" {L[95]:+6.0f}")
            print(f"     zm v94 pred " + " ".join(f"{float(vp[j]):+6.2f}" for j in list(range(0, 96, 8)) + [95]))
            print(f"     zm v94 true " + " ".join(f"{float(vt[j]):+6.2f}" for j in list(range(0, 96, 8)) + [95]))
            print(f"     zm u94 pred " + " ".join(f"{float(zp[j]):+6.1f}" for j in list(range(0, 96, 8)) + [95]))
            print(f"     zm u94 true " + " ".join(f"{float(zt[j]):+6.1f}" for j in list(range(0, 96, 8)) + [95]), flush=True)
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
