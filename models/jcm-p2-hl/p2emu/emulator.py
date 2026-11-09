"""P2@200 JCM emulator: standalone inference.

Everything the original training/inference stack supplied (dataset statistics, grid, static fields, forcing channels,
analytic physics, Held-Larichev-type QBO wave drag, generator noise) is rebuilt here from the release assets, without
the training dataset. The step is

    out = phys( core(x) + lin(x) * ts )      x = [state(t-6h), state(t), year-progress forcings, noise, 0, constants]

with 381 state channels per frame: u, v, T, omega on 95 hybrid levels (surface first) + normalized surface pressure
(p_s / 1e5 Pa). All network I/O is z-scored with the training statistics in meta.npz; the analytic physics works in
physical units.
"""
import math, os, sys
import numpy as np, torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # model/ uses `from model.x import ...`
from model.paradis import Paradis, VerticalMixing                    # noqa: E402
from model.jcm_physics_p20bt import JCMPhysicsP20bt, gregorian_tyear  # noqa: E402

NLEV, NLAT, NLON, NSTATE = 95, 96, 192, 381
NOISE_SCALES = (4.0, 10.0, 25.0)   # l-cutoffs of the three noise families, 16 channels each = 48 noise slots
I_FORC, I_NOISE, I_YEAR, I_CONST = slice(762, 766), slice(766, 814), 814, slice(815, 821)

# PARADIS hyper-parameters of the trained network (from the training config)
MODEL_CFG = dict(latent_multiplier=-1, latent_size=1024, forecast_steps=1, base_dt=21600, num_layers=8, bias_channels=4,
                 velocity_vectors=256, diffusion_size=512, reaction_size=256, adv_interpolation="bicubic",
                 projected_advection=True, num_encoder_layers=1)


class _Cfg(SimpleNamespace):
    def get(self, k, default=None): return getattr(self, k, default)


class Net(torch.nn.Module):
    """PARADIS core + linear 1x1 skip scaled per channel by the 6-h tendency scale ts."""
    def __init__(s, core, ts, cin=821, cout=381):
        super().__init__()
        s.lin = torch.nn.Conv2d(cin, cout, 1); s.core = core; s.register_buffer("ts", ts.clone(), persistent=False)
    def forward(s, x):
        return s.core(x) + s.lin(x) * s.ts


class NoiseGen:
    """Isotropic Gaussian random fields on the 96x192 Gauss grid (16 channels per l-cutoff), spectrum exp(-(l/lc)^2),
    unit gridpoint variance. Identical to the generator noise the model was trained with."""
    def __init__(s, dev, nlat=NLAT, nlon=NLON, scales=NOISE_SCALES, nper=16, seed=0):
        import torch_harmonics as th
        s.isht = th.InverseRealSHT(nlat, nlon, grid="legendre-gauss").to(dev)
        s.lmax, s.mmax = s.isht.lmax, s.isht.mmax; s.dev = dev; s.nper = nper; s.scales = scales
        l = torch.arange(s.lmax, device=dev, dtype=torch.float32).view(-1, 1)
        m = torch.arange(s.mmax, device=dev).view(1, -1)
        tri = (m <= l).float()
        s.amp = torch.stack([torch.exp(-(l / lc) ** 2) * tri for lc in scales])
        s.amp[:, 0, 0] = 0.0
        s.g = torch.Generator(device=dev); s.g.manual_seed(seed)
        z = s._raw(64); s.norm = z.std(dim=(0, 2, 3, 4), keepdim=True)[0]
    def _raw(s, B):
        S = len(s.scales)
        re = torch.randn(B, S, s.nper, s.lmax, s.mmax, device=s.dev, generator=s.g)
        im = torch.randn(B, S, s.nper, s.lmax, s.mmax, device=s.dev, generator=s.g); im[..., 0] = 0.0
        return s.isht(torch.complex(re, im) * s.amp[None, :, None])
    def sample(s, B):
        z = s._raw(B) / s.norm[None]
        return z.reshape(B, -1, *z.shape[-2:])


def year_progress(t):
    """numpy datetime64 -> (sin, cos) of the Gregorian fraction of the calendar year (365 or 366 d), computed exactly as
    the training dataset did (hour resolution, float64), so the forcing channels are bit-identical to training."""
    t = np.asarray([t]).astype("datetime64[h]")
    y0 = t.astype("datetime64[Y]")
    doy = (t - y0) / np.timedelta64(1, "D")
    diy = ((y0 + np.timedelta64(1, "Y")).astype("datetime64[D]") - y0.astype("datetime64[D]")) / np.timedelta64(1, "D")
    ph = 2 * np.pi * (doy / diy)
    return float(np.sin(ph)[0]), float(np.cos(ph)[0])


class HLDrag:
    """Parameterised QBO wave drag (Held-Larichev-style spectrum of eastward/westward waves with critical-level
    absorption), driven by the model's own equatorial zonal-mean u. Applied once per 6-h step after the physics."""
    def __init__(s, coeff_npz, force_npz, dev):
        z = np.load(coeff_npz); FP = np.load(force_npz); th = FP["th"]; assert len(th) == 8
        CS = FP["CS"].astype(np.float64); L0 = int(FP["L0"])
        BE, cE, wE, BW, cW, wW, adamp = (np.exp(th[0]), th[1], np.exp(th[2]), np.exp(th[3]), th[4], np.exp(th[5]), np.exp(th[6])); dl = float(np.exp(th[7]))
        if int(FP["west"]): BE = 0.0
        B = np.where(CS > 0, BE * np.exp(-((CS - cE) / wE) ** 2), 0.0) * 1e-3 + np.where(CS < 0, BW * np.exp(-((CS - cW) / wW) ** 2), 0.0) * 1e-3
        ps = z["sigma_eff"].astype(np.float64) * 101325.0 / 100.0; assert np.all(np.diff(ps) < 0)
        up = np.arange(L0, NLEV); dz = np.r_[0.0, 7.0 * np.log(ps[up[:-1]] / ps[up[1:]])]; dp = np.r_[1.0, (ps[up[:-1]] - ps[up[1:]]) * 100.0]
        lat = np.asarray(z["lat"], np.float64); eqw = np.where(np.abs(lat) <= 5, np.cos(np.deg2rad(lat)), 0.0); eqw /= eqw.sum()
        t = lambda a_: torch.tensor(np.asarray(a_), dtype=torch.float32, device=dev)
        s.P = dict(CS=t(CS), L0=L0, B=t(B), adamp=float(adamp), dl=dl, dz=t(dz), dp=t(dp), eqw=t(eqw), Gc=t(FP["Gc"]), fscale=float(FP["fscale"]),
                   A=t(FP["A"] if "A" in FP.files else np.zeros((NLEV, NLEV))), W=t(z["W_QBO"]))

    @torch.no_grad()
    def G(s, ueq, tyear):
        P = s.P; U = ueq[:, P["L0"]:]; du = U[:, :, None] - P["CS"][None, None, :]
        s0 = torch.sign(du[:, :1, :]); s0 = torch.where(s0 == 0, torch.ones_like(s0), s0)
        trans = torch.sigmoid(torch.clamp(s0 * du / P["dl"], -30, 30)) * torch.exp(-P["adamp"] * P["dz"][None, :, None] / (du ** 2 + P["dl"] ** 2))
        trans = torch.cat([torch.ones_like(trans[:, :1]), trans[:, 1:]], 1); alive = torch.cumprod(trans, 1)
        flux = (P["B"][None, None, :] * (-s0) * alive).sum(-1)
        X = torch.zeros_like(ueq); X[:, P["L0"] + 1:] = 9.80616 * (flux[:, :-1] - flux[:, 1:]) / P["dp"][None, 1:] * 86400.0
        for _ in range(2): X[:, 1:-1] = 0.25 * X[:, :-2] + 0.5 * X[:, 1:-1] + 0.25 * X[:, 2:]
        ph = 2 * math.pi * tyear; h = torch.stack([torch.ones_like(ph), torch.cos(ph), torch.sin(ph), torch.cos(2 * ph), torch.sin(2 * ph)], 1)
        return P["fscale"] * X + h @ P["Gc"] + ueq @ P["A"].T          # m/s/day

    @torch.no_grad()
    def delta(s, u, tyear):
        """u physical (B,95,96,192) -> 6-h u increment."""
        P = s.P; ueq = torch.einsum("blyx,y->bl", u.float(), P["eqw"]) / u.shape[-1]
        Gz = s.G(ueq, tyear.float())
        return (21600.0 / 86400.0) * (P["W"][None, :, :, None] * Gz[:, :, None, None]).expand_as(u).to(u.dtype)


class Emulator:
    """Load once, then step(). States are Z-SCORED (B,381,96,192) tensors, channel order meta['FN']; physical() converts."""
    def __init__(s, assets, ckpt=None, dev="cuda", massfix=True):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        A = lambda f: os.path.join(assets, f)
        M = np.load(A("meta.npz")); s.dev = dev
        s.FN = [str(f) for f in M["FN"]]; assert len(s.FN) == NSTATE
        s.mean = torch.tensor(M["mean"], device=dev).view(1, -1, 1, 1); s.std = torch.tensor(M["std"], device=dev).view(1, -1, 1, 1)
        s.const = torch.tensor(M["constants"], device=dev)                     # (6,96,192), already z-scored
        s.lat = np.asarray(M["lat"]); assert s.lat[0] < s.lat[-1], "grid is south-to-north"
        idx = {k: torch.tensor([s.FN.index(f"{n}_h{l}") for l in range(NLEV)]) for k, n in
               (("u", "u_wind"), ("v", "v_wind"), ("T", "temperature"), ("omega", "omega"))}
        idx["ps"] = torch.tensor([s.FN.index("normalized_surface_pressure")])
        s.idx = idx; s.ips = int(idx["ps"][0])
        dm = SimpleNamespace(dataset=SimpleNamespace(num_in_dyn_features=int(M["num_in_dyn"]), num_in_static_features=int(M["num_in_static"])),
                             num_out_features=int(M["num_out"]))
        cfg = _Cfg(model=_Cfg(**MODEL_CFG))
        core = Paradis(dm, cfg, torch.tensor(M["lat_rad_grid"]), torch.tensor(M["lon_rad_grid"]))
        core.vertical_mixing = VerticalMixing(4, NLEV, 1, 2)
        s.net = Net(core, torch.tensor(M["ts"]).float().view(1, -1, 1, 1)).to(dev).eval()
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            s.net.load_state_dict(sd["net"] if "net" in sd else sd)              # strict: every key must match
        s.phys = JCMPhysicsP20bt(A("jcm_analytic_p20bt.npz"), idx, nlev=NLEV).to(dev)
        s.hl = HLDrag(A("jcm_analytic_p20bt.npz"), A("hlr_p.npz"), dev)
        z = np.load(A("jcm_analytic_p20bt.npz")); s.plev = z["hyam_pa"] / 100 + z["hybm"] * 1000   # hPa at p_s = 1000 hPa
        xg, wg = np.polynomial.legendre.leggauss(NLAT); wg = wg[np.argsort(xg)]
        s.WG = torch.tensor(wg / wg.sum(), device=dev, dtype=torch.float32)
        s.massfix = massfix; s.gm0 = None; s.fixer = None; s.assets = assets; s.coeff = z

    def enable_fixer(s, year, jax_python, gpu=None, latsm=4, levsm=2, log_dir=None, tag="p2"):
        """Start the zonal-mean momentum fixer (the campaign configuration). `jax_python`: the python of an environment with
        jcm-strat + jax-gcm installed (see README). `year`: the year of the rollout's first step (the dycore is built for it)."""
        from p2emu.fixer import Fixer
        gpu = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        s.fixer = Fixer(s.dev, s.FN, s.mean, s.std, s.coeff, int(year), gpu, tag, s.assets, jax_python, latsm=latsm, levsm=levsm, log_dir=log_dir)

    def close(s):
        if s.fixer is not None: s.fixer.close(); s.fixer = None

    # ---- the state is carried Z-SCORED (as in training and in the campaign runs); physical() converts for output ----
    def physical(s, z): return z * s.std + s.mean
    def zscore(s, p): return (p - s.mean) / s.std

    def _split(s, z):
        p = z.float() * s.std + s.mean
        return {k: p[:, s.idx[k].to(p.device)] for k in ("u", "v", "T", "omega", "ps")}

    def _join(s, d, like):
        p = torch.empty(like.shape, dtype=torch.float32, device=like.device)
        for k in ("u", "v", "T", "omega", "ps"): p[:, s.idx[k].to(p.device)] = d[k]
        return ((p - s.mean) / s.std).to(like.dtype)

    def build_input(s, prev_z, cur_z, t_cur, noise):
        """prev_z/cur_z z-scored (B,381,H,W) at t_cur-6h / t_cur (numpy datetime64). noise (B,48,H,W) or None."""
        B = cur_z.shape[0]
        x = torch.zeros((B, 821, NLAT, NLON), device=s.dev, dtype=torch.float32)
        x[:, :381] = prev_z; x[:, 381:762] = cur_z
        sp, cp = year_progress(t_cur - np.timedelta64(6, "h")); sc, cc = year_progress(t_cur)
        for j, v in enumerate((sp, sc, cp, cc)): x[:, 762 + j] = v                    # [sin(t-6h), sin(t), cos(t-6h), cos(t)]
        if noise is not None: x[:, I_NOISE] = noise
        x[:, I_YEAR] = 0.0                                                            # calendar-year channel: zero for the net
        x[:, I_CONST] = s.const
        return x

    @torch.no_grad()
    def step(s, prev_z, cur_z, t_cur, noise=None):
        """One 6-h step on z-scored states. Returns the z-scored state at t_cur + 6 h."""
        x = s.build_input(prev_z, cur_z, t_cur, noise)
        out = s.net(x)
        # tyear is recovered from the float32 forcing channels (as in training), NOT recomputed in float64 from t_cur:
        # the two round differently at some steps and the trajectories then separate (measured: first at step 20)
        tyear = (torch.atan2(x[:, 763, 0, 0], x[:, 765, 0, 0]) / (2 * math.pi)) % 1.0
        year = torch.full((cur_z.shape[0],), int(str(t_cur)[:4]), device=s.dev, dtype=torch.int64)
        po = s._split(out)
        d = s.phys(po, s._split(cur_z), po, tyear, prescribe=False, qbo=False, year=year)   # PK/Jucker T relaxation, sponge, BL friction
        d = dict(d); d["u"] = d["u"] + s.hl.delta(d["u"], tyear)                       # parameterised QBO wave drag
        out = s._join(d, out)
        if s.massfix:                                                                  # global-mean p_s held at its first-step value
            i, m, sd = s.ips, s.mean[0, s.ips], s.std[0, s.ips]
            ph = out[:, i] * sd + m
            if s.gm0 is None: s.gm0 = s.gmean(cur_z[:, i] * sd + m)
            out = out.clone(); out[:, i] = (ph * (s.gm0 / s.gmean(ph)).view(-1, 1, 1) - m) / sd
        if s.fixer is not None:                                                        # tropical zonal-mean u budget fix
            out = s.fixer.apply(cur_z, out, t_cur, t_cur + np.timedelta64(6, "h"))
        return out

    def gmean(s, f):   # (B,H,W) -> (B,) area-weighted (Gaussian weights) global mean
        return (s.WG.view(1, -1, 1) * f).sum(-2).mean(-1)
