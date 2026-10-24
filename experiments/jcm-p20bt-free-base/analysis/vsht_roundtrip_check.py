"""Is a vector-SHT round trip of (u, v) near-identity on p20bt truth? Rel. rms error by level band and latitude band."""
import os, sys, numpy as np, torch; os.environ["CUDA_VISIBLE_DEVICES"] = "3"; sys.path.insert(0, "/data/paradis_jcm")
import gate_p20bt as G, torch_harmonics as th
from omegaconf import OmegaConf
from data.datamodule import Era5DataModule
cfg = OmegaConf.load("config/config_jcm_p20bt_free.yaml"); OmegaConf.resolve(cfg); cfg.compute.num_devices = 1
cfg.training.dataset.start_date = "1996-01-01T06:00:00"; cfg.training.dataset.end_date = "1996-12-31"
dm = Era5DataModule(cfg); dm.setup(stage="fit"); ds = dm.dataset
FN = [str(f) for f in ds.dyn_output_features]; IU = [FN.index(f"u_wind_h{l}") for l in range(95)]; IV = [FN.index(f"v_wind_h{l}") for l in range(95)]
times = np.asarray(ds.ds_input.time.values); fio = G.FastIO(ds, cfg.dataset.root_dir, times, cfg.dataset.n_time_inputs)
mean = torch.tensor(np.asarray(ds.output_mean if hasattr(ds, "output_mean") else 0.0)) if False else None
lat = ds.lat.numpy(); print("lat order", lat[:2], lat[-2:])
sht = th.RealVectorSHT(96, 192, grid="legendre-gauss").cuda(); isht = th.InverseRealVectorSHT(96, 192, grid="legendre-gauss").cuda()
import trainer
from trainer import LitParadis
lit = LitParadis(dm, cfg); m, s = lit._jcm_mean.cuda(), lit._jcm_std.cuda()
for i in (0, 600, 1200):
    _, y = fio.get(i); y = y.cuda(); y = y[None] if y.dim() == 3 else y
    if y.dim() == 5: y = y[:, 0]
    ph = (y * s + m)[0]; uv = torch.stack([ph[IU], ph[IV]], 1)          # (95, 2, lat, lon)
    flip = lat[0] < lat[-1]
    x = uv.flip(-2) if flip else uv                                       # torch_harmonics wants north->south
    r = isht(sht(x.double())).float(); r = r.flip(-2) if flip else r
    e = (r - uv)
    for name, (a, b) in {"h0-15": (0, 16), "h16-73": (16, 74), "h74-93": (74, 94), "h94": (94, 95)}.items():
        pol = np.abs(lat) > 80
        print(f"sample {i} {name}: rel rms err all {float(e[a:b].pow(2).mean().sqrt()/uv[a:b].pow(2).mean().sqrt()):.2e}  "
              f"polar(|lat|>80) {float(e[a:b][..., pol, :].pow(2).mean().sqrt()/uv[a:b][..., pol, :].pow(2).mean().sqrt()):.2e}  "
              f"zm v at pole rows truth {float(uv[a:b,1][:, [0,-1]].mean(-1).abs().max()):.3f}", flush=True)
