"""Analytic forcing of the jcm-strat phase-19 truth run "p20bt" (T63 L95 hybrid), carried structurally.

Drop-in sibling of model/jcm_physics.py (same constructor/forward/dict interface; extra `year=` kwarg for
the QBO target). Coefficients: jcm_analytic_p20bt.npz, written by export_analytic_p20bt.py from the JAX
objects the run itself builds. All per-level arrays are SURFACE-FIRST (index 0 = lowest level, 94 = model
top at ~0.01 hPa), matching the p20bt corpus. Tensors are PARADIS layout (B, nlev, nlat, nlon), latitude
ascending (S->N).

What the truth run does every 12-min step (op-split Lie: physics tendency added FORWARD-EULER over dt,
then the dynamics step; jcm/model.py L1138-1214), restricted to the closed-form terms:

  T:  dT/dt = -k_T(t,p,lat) (T - T_eq(t, p_s, lat))              jcm_strat/jucker.py L242
            - s (T - [T])                     top level only      upper_sponge.py L161-166
  u:  du/dt = -k_v u - s (u - [u]) - k_q W ([u] - u_ERA5(t))      jucker.py L237-240, qbo_nudging.py L196-200
  v:  dv/dt = -k_v v - s (v - [v])                                jucker.py L241
  [p_ref >= 400 hPa: + (x_ERA5 - x)/6h for u, v, T]                jcm/nudging.py L214-243 (NOT carried:
                                                                  the 3-D ERA5 target is not in the corpus;
                                                                  `prescribe` replaces those levels by truth)
  [Hines GWD, Lott-Miller SSO: state-dependent, NOT carried -- the network learns them]

[.] = zonal mean, s = zlf/(1+zlf dt) (ECHAM implicit factor written as a rate), k_v = Held-Suarez boundary
friction (levels 0-8, inside the nudged region). Over a 6-h emulator step = n = 30 substeps, with the
network increment Delta = out - x0 spread evenly over the substeps, every one of these linear relaxations
x' = -a (x - x_e(t)) integrates EXACTLY (for constant a and x_e linear in time) to

    x1 = D x0 + (1 - D) x_e(start) + C (x_e(end) - x_e(start)) + G Delta
    r = 1 - a dt,  D = r^n,  G = (1 - D) / (n a dt),  C = sum_k r^(n-1-k) (1 - r) k / (n-1)

(the discrete twin of the old module's e^{-a dt}, g = (1 - e^{-a dt})/(a dt)). x_e(start/end) are evaluated
at substeps 0 and n-1 (T_eq with p_s of the start and of the step end; the ERA5 QBO target at both times),
k_T at mid-step. The zonal-mean and zonal-anomaly parts relax at different rates, so each field is split:

    T: [T] at a = k_T toward [T_eq];  T - [T] at a = k_T + s toward k_T (T_eq - [T_eq]) / (k_T + s)
    u: [u] at a = k_v + k_q W toward k_q W u_ERA5 / (k_v + k_q W);   u - [u] at a = k_v + s toward 0
    v: [v] at a = k_v;  v - [v] at a = k_v + s

D, G, C are computed in float64 (float32 cancels in 1 - r^n at the QBO-window edge).

T_eq is PORTED (not tabulated) because below 250 hPa it depends on the column's surface pressure:
    T_eq = BETA * JFV_TE(t) + (1 - BETA) * T_PK(p, lat, t),  p = sigma_eff * nsp * 101325 Pa
with sigma_eff = a/101325 + b (get_sigma_centers(101325)) and nsp = p_s / 1e5 (jcm.constants.p0). That
1.3 % pressure inconsistency (p = 1.01325 x the hybrid pressure at the level's sigma_eff) is IN THE TRUTH
RUN and is reproduced deliberately. JFV_TE/JFV_K are the run's own interpolated JFV2013 tables (12 months)
and the month interpolation is jucker.py L210-219 verbatim.

tyear is the Gregorian fraction of year, (zero-based day of year + fraction of day) / (365 or 366)
(jcm/date.py L130-170). PARADIS' sin/cos year_progress uses day/365.25 and is up to ~0.5 day off; use
gregorian_tyear() on the timestamps instead (see test_physics_p20bt.py for the size of the error).
"""
import numpy as np
import torch
import torch.nn as nn

P0_PA = 101325.0
R_DRY, GRAVITY = 287.05, 9.80665
# US Standard Atmosphere 1976 (jcm_strat/polvani_kushner.py L62-70)
_USSA = ((101325.0, 288.15, -6.5e-3), (22632.1, 216.65, 0.0), (5474.89, 216.65, 1.0e-3),
         (868.019, 228.65, 2.8e-3), (110.906, 270.65, 0.0), (66.9389, 270.65, -2.8e-3),
         (3.95642, 214.65, -2.0e-3))


def _ussa_temperature(p_pa: torch.Tensor) -> torch.Tensor:
    """polvani_kushner.py L75-85."""
    t = torch.full_like(p_pa, _USSA[0][1])
    for p_b, t_b, lapse in _USSA:
        in_layer = p_pa <= p_b
        val = torch.full_like(p_pa, t_b) if lapse == 0.0 else t_b * (p_pa / p_b) ** (-R_DRY * lapse / GRAVITY)
        t = torch.where(in_layer, val, t)
    return t


def gregorian_tyear(times):
    """numpy datetime64 array -> (tyear float64, year int) exactly as jcm.date.fraction_of_year_elapsed."""
    t = np.asarray(times).astype("datetime64[s]")
    y = t.astype("datetime64[Y]")
    year = y.astype(int) + 1970
    days_in_year = ((y + np.timedelta64(1, "Y")).astype("datetime64[D]") - y.astype("datetime64[D]")).astype(int)
    elapsed_s = (t - y.astype("datetime64[s]")).astype(np.int64)
    doy, sec = elapsed_s // 86400, elapsed_s % 86400
    return (doy + sec / 86400.0) / days_in_year, year


class JCMPhysicsP20bt(nn.Module):
    def __init__(self, npz_path, idx, nlev=95, nlat=96, nlon=192):
        """idx: dict of channel-index tensors for u, v, T, omega, ps (as JCMPhysics)."""
        super().__init__()
        z = np.load(npz_path)
        assert str(z["LEVEL_ORDER"]) == "surface_first"
        f = lambda k: torch.from_numpy(np.asarray(z[k], np.float32))
        self.nlev, self.nlat, self.nlon = nlev, nlat, nlon
        self.n_sub = int(z["N_SUB"]); self.dt = float(z["MODEL_DT_S"])
        assert self.n_sub * self.dt == float(z["EMUL_DT_S"]) == 21600.0

        for k, shp in (("JFV_TE", (12, nlev, nlat)), ("JFV_K", (12, nlev, nlat)), ("K_HS", (nlev, nlat)),
                       ("W_QBO", (nlev, nlat))):
            assert tuple(z[k].shape) == shp, (k, z[k].shape, shp)
        for k in ("BETA", "KV", "SPONGE_RATE", "sigma_eff", "NUDGE_MASK"):
            assert tuple(z[k].shape) == (nlev,), (k, z[k].shape)
        assert tuple(z["lat"].shape) == (nlat,) and np.all(np.diff(z["lat"]) > 0), "lat must be ascending"
        assert z["QBO_TARGET"].shape[1:] == (nlev, nlat)
        # sponge must sit where T_eq is zonally uniform (the [y]/y' split of T relies on it)
        assert np.all(z["BETA"][z["SPONGE_RATE"] > 0] == 1.0)
        # the QBO window must be tropical: |lat| >= 25 carries no weight (guards the 2026-10-21 permute bug)
        assert np.abs(z["W_QBO"][:, np.abs(z["lat"]) >= 25.0]).max() == 0.0

        L = lambda a: a.view(1, nlev, 1, 1)
        self.register_buffer("JFV_TE", f("JFV_TE").view(12, 1, nlev, nlat, 1))
        self.register_buffer("JFV_K", f("JFV_K").view(12, 1, nlev, nlat, 1))
        self.register_buffer("NODES", torch.from_numpy(np.asarray(z["MONTH_NODES"], np.float64)))
        self.register_buffer("BETA", L(f("BETA")))
        self.register_buffer("K_HS", f("K_HS").view(1, nlev, nlat, 1))
        self.register_buffer("KV", L(f("KV")))
        self.register_buffer("S", L(f("SPONGE_RATE")))
        self.register_buffer("sigma", L(f("sigma_eff")))
        self.register_buffer("lat_rad", f("lat_rad").view(1, 1, nlat, 1))
        # W_QBO: (nlev surface-first, nlat ascending) -> (1, nlev, nlat, 1). NO permute.
        self.register_buffer("W_QBO", f("W_QBO").view(1, nlev, nlat, 1))
        self.register_buffer("QBO_TARGET", f("QBO_TARGET"))                     # (nmonths, nlev, nlat)
        self.register_buffer("KN", L(f("NUDGE_INV_TAU")))                       # 1/s per level, 0 above 400 hPa
        self.qbo_year0 = int(z["QBO_YEAR0"]); self.qbo_nmonths = int(z["QBO_NMONTHS"])
        self.k_qbo = float(z["K_QBO"])
        # p = sigma_eff * nsp * 101325 with nsp = p_s / 1e5: both constants as in the truth run
        assert float(z["P0_SIGMA_PA"]) == P0_PA and float(z["P0_NSP_PA"]) == 1.0e5
        m = np.asarray(z["NUDGE_MASK"], bool)
        self.nudge_top = int(m.sum())
        assert m[:self.nudge_top].all() and not m[self.nudge_top:].any(), "nudged levels must be 0..n-1"
        self.pk = {k: float(z[f"pk_{k}"]) for k in ("gamma", "p_t_pa", "phi0", "delta_phi", "epsilon", "t_peak",
                                                     "season_offset", "p_vortex_top_pa", "kappa")}
        self.hs = {k: float(z[f"hs_{k}"]) for k in ("maxT", "dTy", "dThz")}
        for k, v in idx.items():
            self.register_buffer(f"i_{k}", v)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # never let a checkpoint override the analytic weights/tables built from the file
        for k in ("W_QBO", "QBO_TARGET", "JFV_TE", "JFV_K", "K_HS", "BETA", "KV", "S", "sigma", "lat_rad", "NODES"):
            state_dict.pop(prefix + k, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ---- time interpolation (jucker.py L210-219) ------------------------------------------
    def _month_weights(self, tyear):
        nodes = self.NODES
        t = torch.remainder(tyear.to(nodes.dtype), 1.0)
        idx = torch.clamp(torch.searchsorted(nodes, t.contiguous(), right=True) - 1, 0, nodes.numel() - 2)
        w = (t - nodes[idx]) / (nodes[idx + 1] - nodes[idx])
        return (idx - 1) % 12, idx % 12, w

    def _jfv(self, table, tyear):
        m0, m1, w = self._month_weights(tyear)
        w = w.to(table.dtype).view(-1, 1, 1, 1)
        return (1.0 - w) * table[m0, 0] + w * table[m1, 0]                      # (B, nlev, nlat, 1)

    def k_t(self, tyear):
        """jucker.py L227-230: (B, nlev, nlat, 1) in 1/s."""
        return self.BETA * self._jfv(self.JFV_K, tyear) + (1.0 - self.BETA) * self.K_HS

    def _t_pk(self, nsp, tyear):
        """PolvaniKushnerColumns._equilibrium_temperature, polvani_kushner.py L128-154."""
        p = self.pk
        p_over_p0 = self.sigma * nsp                                             # (B, nlev, nlat, nlon)
        p_pa = p_over_p0 * P0_PA
        s = torch.cos(2.0 * np.pi * (tyear.view(-1, 1, 1, 1) - p["t_peak"]))
        c = p["season_offset"]
        a_n = torch.clamp((s + c) / (1.0 + c), 0.0, 1.0)
        a_s = torch.clamp((-s + c) / (1.0 + c), 0.0, 1.0)
        lat = self.lat_rad
        t_us = _ussa_temperature(p_pa)
        t_trop = p_over_p0 ** p["kappa"] * (
            self.hs["maxT"] - self.hs["dTy"] * torch.sin(lat) ** 2 - p["epsilon"] * s * torch.sin(lat)
            - self.hs["dThz"] * torch.log(p_over_p0) * torch.cos(lat) ** 2)
        t_trop = torch.maximum(t_us, t_trop)
        t_us_pt = _ussa_temperature(torch.tensor(p["p_t_pa"], dtype=p_pa.dtype, device=p_pa.device))
        t_pv = t_us_pt * (p_pa / p["p_t_pa"]) ** (R_DRY * p["gamma"] / GRAVITY)
        w_n = 0.5 * (1.0 + torch.tanh((lat - p["phi0"]) / p["delta_phi"]))
        w_s = 0.5 * (1.0 - torch.tanh((lat + p["phi0"]) / p["delta_phi"]))
        w = (a_n * w_n + a_s * w_s) * 0.5 * (1.0 + torch.tanh(torch.log(p_pa / p["p_vortex_top_pa"])))
        t_strat = (1.0 - w) * t_us + w * t_pv
        return torch.where(p_pa < p["p_t_pa"], t_strat, t_trop)

    def t_eq(self, nsp, tyear):
        """JuckerColumns._equilibrium_temperature (jucker.py L222-225). nsp: (B,1,nlat,nlon) = p_s/1e5."""
        tyear = tyear.to(nsp.dtype)
        return self.BETA * self._jfv(self.JFV_TE, tyear) + (1.0 - self.BETA) * self._t_pk(nsp, tyear)

    def qbo_target(self, tyear, year):
        """QboNudging._target_now (qbo_nudging.py L182-188): (B, nlev, nlat, 1) m/s."""
        t = (year.to(torch.float64) - self.qbo_year0) * 12.0 + tyear.to(torch.float64) * 12.0 - 0.5
        t = torch.clamp(t, 0.0, self.qbo_nmonths - 1.0)
        i0 = torch.clamp(torch.floor(t).long(), 0, self.qbo_nmonths - 2)
        f = (t - i0).to(self.QBO_TARGET.dtype).view(-1, 1, 1)
        return ((1.0 - f) * self.QBO_TARGET[i0] + f * self.QBO_TARGET[i0 + 1]).unsqueeze(-1)

    def _coeffs(self, a):
        """For per-substep factor r = 1 - a dt over n substeps (forward-Euler Lie split, jcm/model.py):
        D = r^n; G = (1-D)/(n a dt) (network increment spread evenly); C = sum_k r^(n-1-k)(1-r) k/(n-1),
        the weight of (x_e(end) - x_e(start)) when the target x_e drifts linearly over the substeps."""
        # float64 ALWAYS: at the QBO window edge a dt ~ 3e-5, and in float32 1 - (1 - a dt)^30 cancels
        # catastrophically (measured 1.2e-3 m/s error on u at level 73). These arrays are (B, nlev, nlat, 1).
        dtype = a.dtype
        ad = a.to(torch.float64) * self.dt
        r = 1.0 - ad
        n = self.n_sub
        D = r ** n
        G = torch.where(ad > 0, (1.0 - D) / torch.clamp(n * ad, min=1e-300), torch.ones_like(ad))
        C = torch.zeros_like(ad)
        for k in range(1, n):
            C = C + r ** (n - 1 - k) * ad * (k / (n - 1))
        return D.to(dtype), G.to(dtype), C.to(dtype), (1.0 - D).to(dtype)

    def _relax(self, x0, delta, a_m, a_a, xe0_m=None, xe1_m=None, xe0_a=None, xe1_a=None):
        """Zonal mean relaxes at a_m toward xe_m(t), zonal anomaly at a_a toward xe_a(t); targets linear in time
        from xe0 (substep 0) to xe1 (substep n-1); None = 0."""
        m0, d_m = x0.mean(dim=3, keepdim=True), delta.mean(dim=3, keepdim=True)

        def part(x, d, a, e0, e1):
            D, G, C, omD = self._coeffs(a)
            out = D * x + G * d
            if e0 is not None:
                out = out + omD * e0 + C * (e1 - e0)
            return out
        return part(m0, d_m, a_m, xe0_m, xe1_m) + part(x0 - m0, delta - d_m, a_a, xe0_a, xe1_a)

    def _pad_nudged(self, e):
        """(B, nudge_top, nlat, nlon) -> (B, nlev, nlat, nlon), zero above the nudged levels (KN is 0 there)."""
        B, n, H, W = e.shape
        assert n == self.nudge_top, (n, self.nudge_top)
        return torch.cat([e, e.new_zeros(B, self.nlev - n, H, W)], dim=1)

    def _substep_tyear(self, tyear, year, k):
        """tyear of substep k: within a 6-h step starting at 00/06/12/18 UTC the year never wraps."""
        y = year.to(torch.int64)
        diy = 365.0 + ((y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))).to(torch.float64)
        return tyear.to(torch.float64) + k * self.dt / (diy * 86400.0)

    def forward(self, out_p, cur_p, true_p, tyear, prescribe=True, qbo=True, year=None, era5=None):
        """PHYSICAL-unit dicts of (B,nlev,nlat,nlon) / (B,1,nlat,nlon) tensors (ps = p_s/1e5, the corpus'
        normalized_surface_pressure); tyear (B,) Gregorian fraction of year at the START of the step
        (gregorian_tyear); year (B,) calendar year. year is required: it fixes the leap-year rate of tyear
        within the step and the dated ERA5 QBO target.
        era5: optional dict u, v, T of (B, nudge_top, nlat, nlon) PHYSICAL ERA5 nudging target, held constant over the
        step as in the truth run. When given, the truth's ERA5 nudging -(X - E) * NUDGE_INV_TAU on levels 0..nudge_top-1
        is added to the relaxation (rates add exactly under the forward-Euler Lie split). Not combinable with prescribe."""
        if year is None:
            raise ValueError("JCMPhysicsP20bt needs `year` (Gregorian tyear rate + dated ERA5 QBO target)")
        u0, v0, T0 = cur_p["u"], cur_p["v"], cur_p["T"]
        assert u0.shape[1:] == (self.nlev, self.nlat, self.nlon), u0.shape
        n = self.n_sub
        ty0 = self._substep_tyear(tyear, year, 0)
        ty1 = self._substep_tyear(tyear, year, n - 1)
        tym = self._substep_tyear(tyear, year, 0.5 * (n - 1))
        ps1 = true_p["ps"] if prescribe else out_p["ps"]
        # temperature: dT/dt = -k_T (T - T_eq) - s (T - [T]); split into [T] (rate k_T, target [T_eq]) and
        # T - [T] (rate k_T + s, target k_T T_eq' / (k_T + s)); T_eq(t, p_s) taken linear over the step
        Te0 = self.t_eq(cur_p["ps"], ty0)
        Te1 = self.t_eq(ps1, ty1)
        KT = self.k_t(tym)
        a_a = KT + self.S
        m0, m1 = Te0.mean(dim=3, keepdim=True), Te1.mean(dim=3, keepdim=True)
        if era5 is None:
            f = KT / a_a
            T1 = self._relax(T0, out_p["T"] - T0, KT, a_a, m0, m1, f * (Te0 - m0), f * (Te1 - m1))
        else:
            assert not prescribe, "era5 nudging replaces prescribe"
            E = {k: self._pad_nudged(era5[k]) for k in ("u", "v", "T")}
            KN = self.KN.to(T0.dtype)
            Em = {k: E[k].mean(dim=3, keepdim=True) for k in E}
            # zonal mean: rate KT + KN toward (KT [Te] + KN [E]) / (KT + KN); anomaly: rate KT + S + KN toward
            # (KT Te' + KN E') / (KT + S + KN); the ERA5 part is constant over the step
            am, aa = KT + KN, a_a + KN
            T1 = self._relax(T0, out_p["T"] - T0, am, aa,
                             (KT * m0 + KN * Em["T"]) / am, (KT * m1 + KN * Em["T"]) / am,
                             (KT * (Te0 - m0) + KN * (E["T"] - Em["T"])) / aa, (KT * (Te1 - m1) + KN * (E["T"] - Em["T"])) / aa)
        # winds: [u] at k_v + k_q W toward k_q W u_ERA5 / (k_v + k_q W); anomalies at k_v + s toward 0
        if era5 is not None:
            # [u]: rate KV + kq + KN toward (kq u_qbo + KN [E_u]) / rate; u': rate KV + S + KN toward KN E_u' / rate
            kq = self.k_qbo * self.W_QBO if qbo else torch.zeros_like(self.W_QBO)
            am, aa = self.KV + kq + KN, self.KV + self.S + KN
            sm = torch.where(am > 0, 1.0 / torch.clamp(am, min=1e-30), torch.zeros_like(am))
            sa = torch.where(aa > 0, 1.0 / torch.clamp(aa, min=1e-30), torch.zeros_like(aa))
            q0 = self.qbo_target(ty0, year) if qbo else 0.0
            q1 = self.qbo_target(ty1, year) if qbo else 0.0
            ea = KN * (E["u"] - Em["u"]) * sa
            u1 = self._relax(u0, out_p["u"] - u0, am, aa, (kq * q0 + KN * Em["u"]) * sm, (kq * q1 + KN * Em["u"]) * sm, ea, ea)
            amv = self.KV + KN
            smv = torch.where(amv > 0, 1.0 / torch.clamp(amv, min=1e-30), torch.zeros_like(amv))
            eva = KN * (E["v"] - Em["v"]) * sa                                  # v' rate is KV + S + KN, as u'
            v1 = self._relax(v0, out_p["v"] - v0, amv, aa, KN * Em["v"] * smv, KN * Em["v"] * smv, eva, eva)
        elif qbo:
            kq = self.k_qbo * self.W_QBO                                         # (1, nlev, nlat, 1)
            a_m = self.KV + kq
            g = torch.where(a_m > 0, kq / torch.clamp(a_m, min=1e-30), torch.zeros_like(a_m))
            u1 = self._relax(u0, out_p["u"] - u0, a_m, self.KV + self.S,
                             g * self.qbo_target(ty0, year), g * self.qbo_target(ty1, year))
        else:
            u1 = self._relax(u0, out_p["u"] - u0, self.KV, self.KV + self.S)
        if era5 is None:
            v1 = self._relax(v0, out_p["v"] - v0, self.KV, self.KV + self.S)
        if prescribe:
            # levels 0..nudge_top-1 (p_ref >= 400 hPa) are ERA5-nudged (tau 6 h) in the truth run; the
            # 3-D ERA5 target is not available, so they are replaced by the truth state.
            n = self.nudge_top
            u1 = torch.cat([true_p["u"][:, :n], u1[:, n:]], dim=1)
            v1 = torch.cat([true_p["v"][:, :n], v1[:, n:]], dim=1)
            T1 = torch.cat([true_p["T"][:, :n], T1[:, n:]], dim=1)
        return {"u": u1, "v": v1, "T": T1, "omega": out_p["omega"], "ps": ps1}
