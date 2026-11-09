"""J3: dycore-side HL forcing = exact copy of qbo_arms.py install_force_patch (stoch sigma 0, as hlr_p runs); call install_force_patch(npz) BEFORE building the model."""
import numpy as np, jax, jax.numpy as jnp
def install_force_patch(force_npz):
    """Replace QboNudging.tendency by the fitted wave-drag forcing. Everything in surface-first order internally."""
    import jcm_strat.qbo_nudging as QN
    from flax import nnx
    from jcm.physics.coords_util import column_lat_lon
    FP = np.load(force_npz)
    CS = jnp.asarray(FP["CS"], jnp.float32); L0 = int(FP["L0"]); th = FP["th"]
    BE, cE, wE, BW, cW, wW, adamp = (np.exp(th[0]), th[1], np.exp(th[2]), np.exp(th[3]), th[4], np.exp(th[5]), np.exp(th[6]))
    # v2 (fit_wave.py): soft critical level of width delta, + 1-2-1 x2 vertical smoothing; v1 (hard cut) if len(th) == 7
    V2 = len(th) == 8; dl = float(np.exp(th[7])) if V2 else 1.0
    if V2 and int(FP["west"]): BE = 0.0
    Bmag_E = jnp.asarray(np.where(FP["CS"] > 0, BE * np.exp(-((FP["CS"] - cE) / wE) ** 2), 0.0) * 1e-3, jnp.float32)
    Bmag_W = jnp.asarray(np.where(FP["CS"] < 0, BW * np.exp(-((FP["CS"] - cW) / wW) ** 2), 0.0) * 1e-3, jnp.float32)
    Gc = FP["Gc"]                         # (5, 95) surface-first: untapered [offset, cos1, sin1, cos2, sin2] m/s/day
    fscale = float(FP["fscale"]) if "fscale" in FP.files else 1.0
    Alin = jnp.asarray(FP["A"] if "A" in FP.files else np.zeros((95, 95)), jnp.float32)   # (95 out, 95 in) surface-first, untapered, 1/day
    # stochastic multipliers: daily AR(1) in log space, index = days since 1990-01-01
    nd = 45 * 366; rng = np.random.default_rng(0 + 12345)
    phi = np.exp(-1.0 / 30.0); eps = np.zeros((nd, 2))
    z = rng.standard_normal((nd, 2))
    for i in range(1, nd): eps[i] = phi * eps[i - 1] + np.sqrt(1 - phi ** 2) * z[i]
    mult = np.exp(0.0 * eps - 0.5 * 0.0 ** 2) if 0.0 > 0 else np.ones((nd, 2))
    MULT = jnp.asarray(mult, jnp.float32)
    G0 = 9.80616; HS = 7.0
    orig_cache = QN.QboNudging.cache_coords

    def cache(self, coords):
        orig_cache(self, coords)
        vertical = coords.vertical
        sigma = vertical.centers if hasattr(vertical, "centers") else vertical.get_sigma_centers(QN.P0_PA)
        p_ref = np.asarray(sigma, dtype=np.float64) * QN.P0_PA / 100.0
        order = np.argsort(-p_ref)                       # surface-first permutation
        assert len(order) == 95 and np.all(np.diff(p_ref[order]) < 0)
        inv = np.argsort(order)
        lat_cols, _ = column_lat_lon(coords.horizontal)
        lats = np.rad2deg(np.unique(np.round(np.asarray(lat_cols, np.float64), 10)))
        eqw = np.where(np.abs(lats) <= 5, np.cos(np.deg2rad(lats)), 0.0); eqw /= eqw.sum()
        ps = p_ref[order]; up = np.arange(L0, 95)
        dz = np.r_[0.0, HS * np.log(ps[up[:-1]] / ps[up[1:]])]
        dp = np.r_[1.0, (ps[up[:-1]] - ps[up[1:]]) * 100.0]
        for k_, v_ in dict(order=jnp.asarray(order), inv=jnp.asarray(inv), eqw=jnp.asarray(eqw, jnp.float32),
                           dz=jnp.asarray(dz, jnp.float32), dp=jnp.asarray(dp, jnp.float32), Gc=jnp.asarray(Gc, jnp.float32)).items():
            setattr(self, "_g3_" + k_, nnx.Variable(v_))
        print(f"G3 force patch: p_ref sfc-first {ps[0]:.1f}..{ps[-1]:.3f} hPa, eq lats {lats[eqw > 0].round(2)}, "
              f"window max w {float(np.asarray(self._w.get_value()).max()):.2f}, fscale {fscale}, stoch sigma {0.0}", flush=True)

    def tendency(self, u_cols, tyear):
        g = {k_: getattr(self, "_g3_" + k_).get_value() for k_ in ("order", "inv", "eqw", "dz", "dp", "Gc")}
        ubar = self.zonal_mean(u_cols)                                   # (nlev JAX order, nlat)
        ueq = (ubar @ g["eqw"])[g["order"]]                              # (95,) surface-first
        U = ueq[L0:]
        du = U[:, None] - CS[None, :]                                    # (m, nc)
        s0 = jnp.sign(du[:1, :])
        if V2:
            s0 = jnp.where(s0 == 0, 1.0, s0)
            trans = jax.nn.sigmoid(jnp.clip(s0 * du / dl, -30, 30)) * jnp.exp(-adamp * g["dz"][:, None] / (du ** 2 + dl ** 2))
            trans = trans.at[0].set(1.0)
            alive = jnp.cumprod(trans, axis=0); tau = 0.0
        else:
            alive = jnp.cumprod((jnp.sign(du) == s0).astype(jnp.float32), axis=0)
            tau = jnp.cumsum(adamp * g["dz"][:, None] / jnp.maximum(du ** 2, 1.0), axis=0)
        yrs = self.year - 1990; diy = 366.0 if self.year % 4 == 0 else 365.0
        di = jnp.clip(jnp.floor(yrs * 365.2425 + tyear * diy).astype(jnp.int32), 0, MULT.shape[0] - 1)
        B = Bmag_E * MULT[di, 0] + Bmag_W * MULT[di, 1]
        flux = (B[None, :] * (-s0) * alive * jnp.exp(-tau)).sum(-1)      # (m,) Pa
        X = jnp.zeros(95, jnp.float32).at[L0 + 1:].set(G0 * (flux[:-1] - flux[1:]) / g["dp"][1:] * 86400.0)
        if V2:
            for _ in range(2):
                X = X.at[1:-1].set(0.25 * X[:-2] + 0.5 * X[1:-1] + 0.25 * X[2:])
        ph = 2 * jnp.pi * tyear
        h = jnp.stack([jnp.ones_like(ph), jnp.cos(ph), jnp.sin(ph), jnp.cos(2 * ph), jnp.sin(2 * ph)])
        Gz = fscale * X + h @ g["Gc"] + Alin @ ueq                                   # (95,) m/s/day surface-first
        Gj = Gz[g["inv"]]                                               # JAX order
        per_day = self.k * self.tau_days                                 # nondim rate of 1/day
        tend_lat = per_day * self._w.get_value() * Gj[:, None]           # (nlev, nlat)
        return tend_lat[:, self._lat_idx.get_value()]

    QN.QboNudging.cache_coords = cache
    QN.QboNudging.tendency = tendency


