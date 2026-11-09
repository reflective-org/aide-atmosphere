"""Zonal-mean momentum fixer (torch side), as used in the AIDE campaign runs (--fixer 1, FIX_LATSM=4, FIX_LEVSM=2).
Replaces the m=0 u increment of one 6-h emulator step in 15S-15N x 100-1 hPa (tapered) by
dt * 0.5 * (RHS(s0) + RHS(s1)), RHS = -vbar ubar_y - omegabar ubar_p + f vbar + ubar vbar tan/a - eddy y/p flux div + HL + Hines.
omegabar from mean continuity of vbar (the emulator's own omega-bar is inconsistent with its vbar); eddy omega' = omega minus
its zonal mean. HL and Hines are the DYCORE's terms, evaluated by fixer/fixer_server.py in the jcm-strat JAX environment
(a subprocess, exchanging states through /dev/shm). T and eddies are untouched.
"""
import os, subprocess, tempfile, numpy as np, torch
A_E = 6.371e6; OM = 7.292e-5; DT = 21600.0
class Fixer:
    def __init__(s, dev, FN, mean, std, z, year, gpu, tag, assets, jax_python, latsm=4, levsm=2, log_dir=None,
                 taper_lat=(10.0, 15.0), p_hi=100.0, p_lo=1.0, nlv_taper=2, mode="fix"):
        s.dev = dev; s.mean, s.std = mean, std; s.mode = mode
        s.iu = [FN.index(f"u_wind_h{l}") for l in range(95)]; s.iv = [FN.index(f"v_wind_h{l}") for l in range(95)]; s.iw = [FN.index(f"omega_h{l}") for l in range(95)]
        p = z["hyam_pa"] / 100 + z["hybm"] * 1000; lat = np.asarray(z["lat"], np.float64); phi = np.deg2rad(lat); s.p = p
        eye = np.eye(96); s.Dphi = torch.tensor(np.gradient(eye, phi, axis=0, edge_order=2).T, dtype=torch.float64, device=dev)    # (F @ Dphi.T) -> use einsum below
        s.Dphi = torch.tensor(np.stack([np.gradient(eye[j], phi, edge_order=2) for j in range(96)], 1), dtype=torch.float64, device=dev)  # Dphi[i,j] = d(delta_j)/dphi at i -> dF_i = sum_j Dphi[i,j] F_j
        pl = p * 100.0; I = np.eye(95); s.Dp = torch.tensor(np.stack([np.gradient(I[j], pl, edge_order=2) for j in range(95)], 1), dtype=torch.float64, device=dev)  # same convention
        s.pl = pl; s.cs = torch.tensor(np.cos(phi), device=dev); s.tn = torch.tensor(np.tan(phi), device=dev); s.f = torch.tensor(2 * OM * np.sin(phi), device=dev)
        s.dpk = torch.tensor(pl[:-1] - pl[1:], device=dev)       # layer thickness (Pa), surface-first, layer k between k and k+1
        # weights: lat taper (1 inside taper_lat[0], cos^2 to 0 at taper_lat[1]); level: p in [p_lo,p_hi], 2 outermost levels at each end ramp 1/3, 2/3
        a = np.abs(lat); wl = np.where(a <= taper_lat[0], 1.0, np.where(a >= taper_lat[1], 0.0, np.cos(0.5 * np.pi * (a - taper_lat[0]) / (taper_lat[1] - taper_lat[0])) ** 2))
        idx = np.where((p <= p_hi) & (p >= p_lo))[0]; wv = np.zeros(95); wv[idx] = 1.0
        for j in range(nlv_taper):
            wv[idx[0] + j] = (j + 1) / (nlv_taper + 1); wv[idx[-1] - j] = (j + 1) / (nlv_taper + 1)
        s.W = torch.tensor(wv[:, None] * wl[None, :], dtype=torch.float64, device=dev)
        s.lev_idx = idx
        # JAX server
        s.shm = f"/dev/shm/fixer_{os.getpid()}_{tag}"
        s.Ui = np.memmap(f"{s.shm}_in", np.float32, "w+", shape=(2, 3, 95, 96, 192)); s.Pi = np.memmap(f"{s.shm}_ps", np.float32, "w+", shape=(2, 96, 192)); s.Uo = np.memmap(f"{s.shm}_out", np.float32, "w+", shape=(2, 2, 95, 96))
        server = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixer", "fixer_server.py")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.2", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()   # before JAX starts
        s.log = os.path.join(log_dir or tempfile.gettempdir(), f"fixer_server_{tag}_{os.getpid()}.log")
        s.srv = subprocess.Popen([jax_python, "-u", server, "serve", s.shm, str(int(year)), os.path.abspath(assets)], env=env,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(s.log, "w"), text=True, bufsize=1)
        while True:
            l = s.srv.stdout.readline()
            if l.startswith("serve ready"): break
            assert l != "", f"fixer server died, see {s.log}"
        s.stats = []; s.latsm = int(latsm); s.levsm = int(levsm)
    def close(s):
        s.srv.stdin.close(); s.srv.wait(timeout=60)
        for ext in ("_in", "_ps", "_out"): os.remove(s.shm + ext)
    def _phys(s, p):   # p (1,381,H,W) physical
        return p[:, s.iu].double(), p[:, s.iv].double(), p[:, s.iw].double()
    def _dyn(s, U, V, W):
        ub, vb, wb = U.mean(-1), V.mean(-1), W.mean(-1)                                    # (1,95,96)
        Dv = torch.einsum("blj,ij->bli", vb * s.cs, s.Dphi) / (A_E * s.cs)
        c = 0.5 * (Dv[:, :-1] + Dv[:, 1:]) * s.dpk[None, :, None]
        om = torch.zeros_like(Dv); om[:, :-1] = -torch.flip(torch.cumsum(torch.flip(c, [1]), 1), [1])
        uv = ((U - ub[..., None]) * (V - vb[..., None])).mean(-1); uw = ((U - ub[..., None]) * (W - wb[..., None])).mean(-1)
        ddp = lambda F: torch.einsum("bjl,ij->bil", F, s.Dp)
        ddy = lambda F: torch.einsum("blj,ij->bli", F, s.Dphi)
        R = dict(adv_y=-vb * ddy(ub) / A_E, adv_p=-om * ddp(ub), cor=s.f * vb, curv=ub * vb * s.tn / A_E,
                 eddy_y=-ddy(s.cs ** 2 * uv) / (A_E * s.cs ** 2), eddy_p=-ddp(uw))
        return R, ub
    def apply(s, s0n, s1n, t0, t1, info=False):
        """s0n, s1n (1,381,H,W) NORMALISED previous / new state; t0, t1 numpy datetime64. Returns the corrected normalised new state."""
        p0 = s0n.float() * s.std + s.mean; p1 = s1n.float() * s.std + s.mean
        for j, p in enumerate((p0, p1)):
            s.Ui[j, 0] = p[0, s.iu].cpu().numpy(); s.Ui[j, 1] = p[0, s.iv].cpu().numpy(); s.Ui[j, 2] = p[0, 190:285].cpu().numpy(); s.Pi[j] = p[0, 380].cpu().numpy()
        s.srv.stdin.write(f"{np.datetime_as_string(np.datetime64(t0, 's'))} {np.datetime_as_string(np.datetime64(t1, 's'))}\n"); assert s.srv.stdout.readline().strip() == "ok"
        ph = torch.tensor(np.array(s.Uo), dtype=torch.float64, device=s.dev)                  # (2,2,95,96): [state][HL,Hines]
        R0, ub0 = s._dyn(*s._phys(p0)); R1, ub1 = s._dyn(*s._phys(p1))
        r0 = sum(R0.values()) + ph[0, 0] + ph[0, 1]; r1 = sum(R1.values()) + ph[1, 0] + ph[1, 1]
        fix = DT * 0.5 * (r0 + r1); net = ub1 - ub0
        dlt = fix - net                                                                        # (1,95,96)
        for _ in range(s.latsm):   # 1-2-1 meridional low-pass of the correction (edge rows copied); the grid-scale part of the increment stays the network's
            dlt = torch.cat([dlt[..., :1], 0.25 * dlt[..., :-2] + 0.5 * dlt[..., 1:-1] + 0.25 * dlt[..., 2:], dlt[..., -1:]], -1)
        for _ in range(s.levsm):
            dlt = torch.cat([dlt[:, :1], 0.25 * dlt[:, :-2] + 0.5 * dlt[:, 1:-1] + 0.25 * dlt[:, 2:], dlt[:, -1:]], 1)
        corr = s.W * dlt                                                               # (1,95,96) m/s, applied uniformly in longitude
        out = s1n.clone(); out[:, s.iu] += (corr[..., None] / s.std[:, s.iu].double()).float()
        if info:
            k = lambda pp: int(np.argmin(np.abs(s.p - pp))); eq = slice(46, 50)
            return out, dict(net=net[0, :, eq].mean(-1).cpu().numpy(), fix=fix[0, :, eq].mean(-1).cpu().numpy(), HL=ph[:, 0][:, :, eq].mean(-1).cpu().numpy(), Hines=ph[:, 1][:, :, eq].mean(-1).cpu().numpy(),
                             terms={n: ((R0[n] + R1[n]) * 0.5 * DT)[0, :, eq].mean(-1).cpu().numpy() for n in R0})
        return out
