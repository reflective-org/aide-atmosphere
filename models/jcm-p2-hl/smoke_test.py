#!/usr/bin/env python3
"""Smoke test: load the checkpoint (strict), run N 6-h steps on one GPU from the shipped IC, check the state stays
finite and bounded, report s/step. With --ref DIR (frames u,v,T,omega,ps_norm .npy from the original stack, same IC,
seed and flags) also report the max abs difference per variable and step.

  CUDA_VISIBLE_DEVICES=0 python smoke_test.py --assets ASSETS_DIR [--steps 8] [--ref DIR]
"""
import argparse, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2emu.emulator import Emulator, NoiseGen

ap = argparse.ArgumentParser(); ap.add_argument("--assets", required=True); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--jax-python", dest="jax_python", default="", help="also run the momentum fixer (jcm-strat env python)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ref", default=""); ap.add_argument("--timing-steps", type=int, default=40)
a = ap.parse_args()
assert torch.cuda.is_available(), "no GPU visible to torch (check LD_LIBRARY_PATH / driver)"
dev = "cuda"; print("device:", torch.cuda.get_device_name(0))
t0 = time.time(); E = Emulator(a.assets, os.path.join(a.assets, "p2_hl_step200.pt"), dev=dev)
print(f"loaded (strict state_dict) in {time.time()-t0:.1f} s; params {sum(p.numel() for p in E.net.parameters()):,}")
ic = np.load(os.path.join(a.assets, "ic_19970101T06.npz")); t = np.datetime64(str(ic["t_cur"]))
if a.jax_python:
    t1 = time.time(); E.enable_fixer(int(str(t)[:4]), a.jax_python); print(f"fixer server up in {time.time()-t1:.1f} s")
prev = torch.tensor(ic["prev_z"], device=dev)[None]; cur = torch.tensor(ic["cur_z"], device=dev)[None]   # z-scored
NG = NoiseGen(dev, seed=1000 + a.seed)
R = {k: np.load(f"{a.ref}/{k}.npy", mmap_mode="r") for k in ("u", "v", "T", "omega", "ps_norm")} if a.ref else None
ok = True; dts = []
nst = max(a.steps, a.timing_steps)
for k in range(nst):
    t1 = time.time(); new = E.step(prev, cur, t, NG.sample(1)); torch.cuda.synchronize(); dts.append(time.time() - t1)
    prev, cur, t = cur, new, t + np.timedelta64(6, "h")
    if k < a.steps:
        pc = E.physical(cur); f = {n: pc[0, E.idx[n]].cpu().numpy() for n in ("u", "v", "T", "omega")}; f["ps_norm"] = pc[0, E.ips].cpu().numpy()
        fin = all(np.isfinite(v).all() for v in f.values()); umax = float(np.abs(f["u"]).max())
        Tmin, Tmax = float(f["T"].min()), float(f["T"].max())
        line = f"step {k+1} {str(t)}  finite {fin}  max|u| {umax:6.1f} m/s  T [{Tmin:.1f}, {Tmax:.1f}] K  gm ps {float(E.gmean(pc[:, E.ips])[0])*1e3:.3f} hPa"
        if R is not None:
            line += "  |diff vs ref| " + " ".join(f"{n} {float(np.abs(f[n] - R[n][k + 1]).max()):.2e}" for n in f)
        print(line, flush=True)
        ok &= fin and umax < 300 and 150 < Tmin and Tmax < 350
E.close()
print("fixer:", "ON" if a.jax_python else "off", end="; ")
print(f"s/step: median {np.median(dts[1:]):.3f} over {len(dts)-1} steps (first step {dts[0]:.2f} s incl. warm-up)")
print("SMOKE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
