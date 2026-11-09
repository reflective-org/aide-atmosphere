#!/usr/bin/env python3
# PROVENANCE ONLY: how meta.npz and the IC were extracted from the internal training stack (paths are that machine's). Not runnable outside it.
"""Dump everything the standalone package needs from the ORIGINAL stack (g2lib_hl + LitParadis + dataset), read-only.
Writes into argv[1]: meta.npz (stats, grid, constants, channel names, model dims), ic_19970101T06.npz (z-scored prev_z/cur_z),
forcing_ref.npz (dataset x[762:821] at several times, to check the standalone forcing generator)."""
import sys, os, json, numpy as np, torch
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/data/paradis_jcm/logs_dyc/gen/g2/code"); sys.path.insert(0, "/data/paradis_jcm/logs_dyc/qnf/code")
import g2lib_hl as G, gate_mine as GM
L = G.build("1996-01-01T00:00:00", "2006-01-01"); ds = L["ds"]; lit = L["lit"]
times = L["times"]; fio = GM.FastIO(ds, L["cfg"].dataset.root_dir, times, 2)
FN = L["FN"]
print("FN[:3]", FN[:3], "n", len(FN)); print("num_in_dyn", ds.num_in_dyn_features, "num_in_static", ds.num_in_static_features, "num_out", L["dm"].num_out_features)
print("forcing_inputs", list(ds.forcing_inputs), "isin/icos", lit._jcm_isin, lit._jcm_icos, "iyear", lit._jcm_iyear)
print("lat_rad_grid", tuple(ds.lat_rad_grid.shape), "lon_rad_grid", tuple(ds.lon_rad_grid.shape))
x0, _ = fio.get(0)
np.savez(f"{OUT}/meta.npz", mean=L["mean"].cpu().numpy().ravel(), std=L["std"].cpu().numpy().ravel(), ts=L["ts"].cpu().numpy().ravel(),
         FN=np.array(FN), lat=np.asarray(ds.lat), lon=np.asarray(ds.lon), lat_rad_grid=np.asarray(ds.lat_rad_grid), lon_rad_grid=np.asarray(ds.lon_rad_grid),
         constants=x0[815:821].numpy(), num_in_dyn=int(ds.num_in_dyn_features), num_in_static=int(ds.num_in_static_features), num_out=int(L["dm"].num_out_features))
# constants must not vary in time
xa, _ = fio.get(1000); print("constants time-invariant:", bool(torch.equal(xa[815:821], x0[815:821])))
sel = {}
for ts_ in ("1997-01-01T00", "1997-01-01T06", "1997-07-02T12", "1997-12-31T18", "1998-01-01T00", "2000-02-29T18", "2000-12-31T18", "2001-01-01T00"):
    i = int(np.where(times == np.datetime64(ts_))[0][0]); x, _ = fio.get(i)
    sel[ts_] = x[762:766].numpy()[:, 0, 0].tolist() + [float(x[814, 0, 0])]; print(ts_, "x[762:766],x[814] =", sel[ts_], "(t-1, t) =", times[i], times[i + 1])
json.dump(sel, open(f"{OUT}/forcing_ref.json", "w"), indent=1)
i = int(np.where(times == np.datetime64("1997-01-01T00"))[0][0]); x, _ = fio.get(i)   # ic for --t0 1997-01-01T06: t-1 = 00Z, t = 06Z
assert times[i + 1] == np.datetime64("1997-01-01T06")
m, s = L["mean"][0].cpu(), L["std"][0].cpu()
np.savez(f"{OUT}/ic_19970101T06.npz", prev_z=x[:381].numpy().astype(np.float32), cur_z=x[381:762].numpy().astype(np.float32),
         t_prev="1997-01-01T00", t_cur="1997-01-01T06", FN=np.array(FN))
print("saved")
