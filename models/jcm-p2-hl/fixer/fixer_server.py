#!/usr/bin/env python3
"""JAX side of the zonal-mean momentum fixer: evaluates the DYCORE's own HL (QBO wave drag, hlr_p patch) and Hines
gravity-wave-drag u tendencies on arbitrary gridpoint states, by building the jcm-strat p20bt physics and calling its
terms. Returns zonal-mean (lon-mean) u tendencies in m/s/s, surface-first, (95, 96 lat ascending).

Runs in the jcm-strat environment (JAX), as a subprocess of the torch emulator (fixer.py starts it):
    python fixer_server.py serve SHM_PREFIX YEAR ASSETS_DIR
Protocol: states in SHM_PREFIX_in (2,3,95,96,192: [state][u,v,T]) and SHM_PREFIX_ps (2,96,192); each stdin line
"ISO_t0 ISO_t1" -> tendencies of both states into SHM_PREFIX_out (2,2,95,96: [state][HL,Hines]); replies "ok".

Same construction as the campaign's server, minus everything the two terms do not read: ERA5 nudging is off (it needs
the ERA5 windows), there is no archive state (humidity = 0 and passive tracers = 0, as the campaign used them).
"""
import os, sys, time
import numpy as np

if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):   # bitwise-reproducible GPU results
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp                                   # noqa: E402
from omegaconf import OmegaConf                                 # noqa: E402
from flax import nnx                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_cfg(year, assets):
    s = open(os.path.join(HERE, "dycore_config_template.yaml")).read()
    s = s.replace("ASSETS", os.path.abspath(assets)).replace("YEAR", str(int(year)))
    return OmegaConf.create(s)


def build_model(year, assets):
    """The p20bt dycore of `year` (start_time and the QBO-term year fixed by the config)."""
    import hl_dyc_patch as H
    H.install_force_patch(os.path.join(assets, "hlr_p.npz"))   # must precede the model build
    import jcm.runners as R
    import jcm_strat.main as M
    from jcm_strat import levels
    cfg = load_cfg(year, assets)
    raw = cfg.get("sl_mass_fixer_exclude", None)
    exclude = list(OmegaConf.to_container(raw)) if raw is not None and OmegaConf.is_config(raw) else list(raw or [])
    M.install_mass_fixer_policy(bool(cfg.get("sl_mass_fixer", True)), exclude)
    di = cfg.get("sl_departure_iterations", None)
    M.install_sl_options({"departure_iterations": int(di) if di is not None else None})
    M.install_calendar(cfg.get("calendar", None))
    M.check_jcm3_run_keys(cfg)
    M.log_dot_precision()                                       # also re-applies the HIGHEST float32 dot precision
    levels.install(cfg.get("level_table", None))
    R.apply_constants_overrides(cfg)
    model = R.build_model(cfg)
    M.log_dot_precision()
    forcing = R.build_forcing(cfg, model.coords, dycore=model.dycore)
    from jcm.forcing_assembly import _ensure_parent_forcing
    forcing = _ensure_parent_forcing(forcing, model.coords)
    return model, cfg, forcing


def clock(model, t):
    """(initial_time Datetime, step int32, sim_time s) for time t inside the model's year."""
    import jax_datetime as jdt
    start = np.datetime64(str(model.start_time.to_datetime64())[:19], "s")
    el = int((np.datetime64(t, "s") - start) / np.timedelta64(1, "s"))
    dt = int(model.dt_si.m)
    assert el >= 0 and el % dt == 0, (t, start)
    days, secs = divmod(el, 86400)
    it = model.start_time + jdt.Timedelta(days=jnp.int32(days), seconds=jnp.int32(secs))
    return it, jnp.int32(el // dt), float(el)


class PhysEval:
    def __init__(self, year, assets, keep=("MoistAirColumnState", "JuckerColumns", "HinesGwdLaunch")):
        import jcm_strat.qbo_nudging as QN
        self.model, self.cfg, self.forcing0 = build_model(year, assets); m = self.model
        self.year = year; self.dt = float(m.dt_si.m)
        m.physics.checkpoint_terms = False
        self.all_terms = list(m.physics.terms)
        m.physics.terms = nnx.List([t for t in self.all_terms if type(t).__name__ in keep])
        self.G = {}
        for term in m.physics.terms:
            cls = type(term)
            if getattr(cls, "_fx_wrapped", False): continue
            orig = cls.__call__
            def mk(orig, nm):
                def w(s, st, diag, frc, ter):
                    tend, d = orig(s, st, diag, frc, ter)
                    self.G[nm] = tend.u_wind.reshape(95, 192, 96).mean(1)      # (nlev top-first, lat); columns are lon-major
                    return tend, d
                return w
            cls.__call__ = mk(orig, cls.__name__); cls._fx_wrapped = True
        if not getattr(QN.QboNudging, "_fx_wrapped", False):
            orig_t = QN.QboNudging.tendency
            def hl_w(s, u_cols, tyear):
                o = orig_t(s, u_cols, tyear); self.G["HL"] = o.reshape(95, 192, 96).mean(1); return o
            QN.QboNudging.tendency = hl_w; QN.QboNudging._fx_wrapped = True
        self.carry = m.initial_physics_carry()
        self.tr_names = tuple(sorted(s.name for s in m.physics.required_tracers()))
        self.q = jnp.zeros((95, 192, 96), jnp.float32)                         # dry: the campaign passed q = 0
        self.f = jax.jit(self._f)

    def _f(self, u, v, T, ps, q, clock_, sc, forcing):
        from jcm.date import DateData
        from jcm.physics_interface import PhysicsState
        self.G.clear(); m = self.model
        st = PhysicsState(u_wind=u, v_wind=v, temperature=T, specific_humidity=q, geopotential=jnp.zeros_like(T),
                          normalized_surface_pressure=ps, tracers={n: jnp.zeros_like(T) for n in self.tr_names})
        fnow = forcing.select(DateData(clock_, sc, self.dt))
        m.physics.compute_tendencies(st, fnow, m.terrain, prev_physics_data=self.carry)
        return {k: v_[::-1] for k, v_ in self.G.items()}                         # surface-first

    def eval(self, u, v, T, ps, t64):
        """u,v,T (95, 96 lat, 192 lon) surface-first float32 physical; ps (96,192) = p_s/1e5; t64 time of the state.
        Returns dict HL, JuckerColumns, HinesGwdLaunch: (95, 96) zonal-mean u tendency [m/s/s], surface-first."""
        to_top = lambda a: jnp.asarray(np.ascontiguousarray(np.asarray(a, np.float32).transpose(0, 2, 1)[::-1]))
        it, istep, _ = clock(self.model, np.datetime64(t64, "s"))
        out = self.f(to_top(u), to_top(v), to_top(T), jnp.asarray(np.ascontiguousarray(np.asarray(ps, np.float32).T)), self.q, it, istep, self.forcing0)
        return {k: np.asarray(v_) for k, v_ in out.items()}


def serve(shm, year, assets):
    pe = PhysEval(year, assets)
    Ui = np.memmap(f"{shm}_in", np.float32, "r", shape=(2, 3, 95, 96, 192)); Pi = np.memmap(f"{shm}_ps", np.float32, "r", shape=(2, 96, 192))
    Uo = np.memmap(f"{shm}_out", np.float32, "r+", shape=(2, 2, 95, 96))
    print("serve ready", flush=True)
    for line in sys.stdin:
        ts = line.split()
        for j in range(2):
            o = pe.eval(Ui[j, 0], Ui[j, 1], Ui[j, 2], Pi[j], np.datetime64(ts[j], "s")); Uo[j, 0] = o["HL"]; Uo[j, 1] = o["HinesGwdLaunch"]
        Uo.flush(); print("ok", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[2], int(sys.argv[3]), sys.argv[4])
