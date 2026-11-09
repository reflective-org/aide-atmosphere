#!/usr/bin/env python3
"""Free-running rollout of the P2@200 JCM emulator from a two-frame initial condition.

  python rollout.py --assets ASSETS_DIR --ic ASSETS_DIR/ic_19970101T06.npz --steps 1460 --seed 0 --out OUT_DIR

Writes OUT_DIR/{u,v,T,omega,ps}.npy, each (steps+1, [95,] 96, 192) float32, frame 0 = the IC's current state, frame k at
t0 + 6h*k. Levels are surface-first (see plev_hPa.npy); ps is p_s/1e5 Pa; lat south-to-north (lat.npy, lon.npy).
Memory: ~28 MB per saved frame (a 1-year run, 1461 frames, needs ~41 GB host RAM); use --save-every to thin.
"""
import argparse, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2emu.emulator import Emulator, NoiseGen, NSTATE

ap = argparse.ArgumentParser()
ap.add_argument("--assets", required=True); ap.add_argument("--ckpt", default=None, help="default: ASSETS/p2_hl_step200.pt")
ap.add_argument("--ic", required=True, help="npz with prev_z, cur_z (381,96,192) z-scored with meta.npz mean/std, and t_cur")
ap.add_argument("--steps", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--det", action="store_true", help="zero noise (deterministic mean model; NOT how it was trained to run)")
ap.add_argument("--no-massfix", dest="massfix", action="store_false")
ap.add_argument("--jax-python", dest="jax_python", default=None, help="python of the jcm-strat JAX env: runs the momentum fixer (campaign configuration)")
ap.add_argument("--no-fixer", dest="fixer", action="store_false", help="run without the momentum fixer (not the campaign configuration)")
ap.add_argument("--save-every", type=int, default=1); ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
assert torch.cuda.is_available(), "no GPU visible to torch"
os.makedirs(a.out, exist_ok=True)
E = Emulator(a.assets, a.ckpt or os.path.join(a.assets, "p2_hl_step200.pt"), dev=dev, massfix=a.massfix)
ic = np.load(a.ic); t = np.datetime64(str(ic["t_cur"]))
if a.fixer:
    assert a.jax_python, "give --jax-python (the jcm-strat env) for the momentum fixer, or --no-fixer"
    E.enable_fixer(int(str(t)[:4]), a.jax_python, log_dir=a.out)
assert [str(f) for f in ic["FN"]] == E.FN, "IC channel order differs from the model's"
prev = torch.tensor(ic["prev_z"], device=dev)[None]; cur = torch.tensor(ic["cur_z"], device=dev)[None]   # z-scored
NG = None if a.det else NoiseGen(dev, seed=1000 + a.seed)       # seed convention of the campaign runs (seed0=0 -> 1000)
nsave = a.steps // a.save_every + 1
F = {k: np.zeros((nsave, 95, 96, 192), np.float32) for k in ("u", "v", "T", "omega")}; PS = np.zeros((nsave, 96, 192), np.float32)
def put(j, z):
    p = E.physical(z)
    for k in F: F[k][j] = p[0, E.idx[k]].cpu().numpy()
    PS[j] = p[0, E.ips].cpu().numpy()
put(0, cur); times = [str(t)]
tt = time.time(); dts = []
for k in range(a.steps):
    t1 = time.time()
    new = E.step(prev, cur, t, None if NG is None else NG.sample(1))
    torch.cuda.synchronize(); dts.append(time.time() - t1)
    umax = float(E.physical(new)[0, E.idx["u"]].abs().max())
    if not np.isfinite(umax) or umax > 300: print(f"step {k+1}: max|u| {umax} -- run blew up, stopping"); break
    prev, cur, t = cur, new, t + np.timedelta64(6, "h")
    if (k + 1) % a.save_every == 0: put((k + 1) // a.save_every, cur); times.append(str(t))
    if (k + 1) % 400 == 0: print(f"step {k+1} ({str(t)})  max|u| {umax:.1f} m/s  {time.time()-tt:.0f} s", flush=True)
E.close()
n = len(times)
for k in F: np.save(f"{a.out}/{k}.npy", F[k][:n])
np.save(f"{a.out}/ps.npy", PS[:n]); np.save(f"{a.out}/plev_hPa.npy", E.plev); np.save(f"{a.out}/lat.npy", E.lat)
json.dump(dict(vars(a), times=times), open(f"{a.out}/run.json", "w"), indent=1)
print(f"{n-1} steps, {np.median(dts[1:] or dts):.3f} s/step (median, excl. first), wrote {a.out}")
