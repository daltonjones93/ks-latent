"""alpha = 0 pilot (2026-10-03): does the energy of drag-free Kolmogorov flow saturate,
and how long does it take? Without drag, large-scale energy is removed only by
viscosity at rate nu*k_min^2 (~900 time units at 3x), so the standard 100-unit spinup
may sample a transient. Integrates from the usual laminar+noise initial condition at
1x and 2x for T_TOTAL time units, recording total energy and the domain-scale
streamwise mode's energy every time unit. Gate: alpha=0 datasets/truths are built
only with a spinup >= the time at which the energy's 200-unit running mean has
changed by < 5% over the previous 200 units."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math
from pathlib import Path
import numpy as np
from ks_latent.config import KolmogorovConfig
from ks_latent.solver.kolmogorov import default_initial_condition_kolmogorov, integrate_kolmogorov
T_TOTAL, CHUNK = 1500.0, 1.0
out = Path("artifacts/experiments/alpha0"); out.mkdir(parents=True, exist_ok=True)
res = {}
for scale in (1, 2):
    cfg = KolmogorovConfig(Lx=2 * math.pi * scale, Ly=2 * math.pi, NX=128 * scale, NY=128, nu=0.01, alpha=0.0,
                           forcing_n=4, dt=0.01, seed=0)
    w = default_initial_condition_kolmogorov(cfg, np.random.default_rng(0))
    n = int(round(CHUNK / cfg.dt))
    E, Edom = [], []
    kx = np.fft.fftfreq(cfg.NX, d=cfg.Lx / cfg.NX) * 2 * np.pi
    ky = np.fft.fftfreq(cfg.NY, d=cfg.Ly / cfg.NY) * 2 * np.pi
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    K2[0, 0] = 1.0
    for i in range(int(T_TOTAL / CHUNK)):
        w = integrate_kolmogorov(w, cfg, n_steps=n, snapshot_every=n)[-1]
        P = np.abs(np.fft.fft2(w)) ** 2 / K2 / w.size ** 2
        E.append(float(P.sum()))
        Edom.append(float(P[1].sum() + P[-1].sum()))
    E = np.array(E)
    sat = None
    for t in range(400, len(E)):
        a, b = E[t - 400:t - 200].mean(), E[t - 200:t].mean()
        if abs(b - a) / b < 0.05:
            sat = t
            break
    res[f"x{scale}"] = {"energy": E.tolist(), "domain_mode_energy": Edom, "saturation_time": sat,
                        "energy_mean_last_300": float(E[-300:].mean())}
    print(f"[alpha0] x{scale}: saturation time {sat}, final mean energy {E[-300:].mean():.4g}", flush=True)
(out / "alpha0_pilot.json").write_text(json.dumps(res))
