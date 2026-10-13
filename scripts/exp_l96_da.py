"""Lorenz-96 physics-vs-learned ensemble filter under model error (Z17; FUTURE-WORK material, not for
Papers I/II unless the user decides otherwise).

Question: does the Paper II Sec. 6 ordering (physics filter wins when its post-remedy model error is below
~1/4 of the observation error, latent/learned filter wins above ~1/3) reproduce on Lorenz-96?

Setup (fixed before running): truth L96 N = 40, F = 8, RK4 dt 0.01; snapshots/analyses every 0.1; every
site observed with sigma_obs = 1.0 (0.27 of the attractor std). Local ETKF (R-localization, Gaspari-Cohn
half-width c sites, multiplicative inflation), 20 members. Filters:
  physics  : L96 with a wrong parameter (F_model or advection coefficient a), without and with online
             estimation of that parameter (global ETKF regression, RTPS 0.9, floor), as in Paper II.
  learned  : `l96_crop_c1` (the window-trained per-site local model of Z16; its latent is x itself),
             trained on true-system data, no parameter.
Settings (inflation in {1.0, 1.05, 1.1, 1.2, 1.4}, c in {2, 4}) are tuned per filter and case on seed 0
(150 analysis cycles) and scored on seeds 1-3 (250 cycles after 50 burn-in). Decision rule as in Paper II:
a filter wins if its 3-seed mean analysis RMSE is >= 10% lower and lower on every seed; otherwise a tie.
Post-remedy axis: one-cycle forecast RMSE of the model (final parameter mean) on true states, in
units of sigma_obs.
Usage: python scripts/exp_l96_da.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.config import Lorenz96Config
from ks_latent.da.localization import gaspari_cohn_taper
from ks_latent.da.param_est import global_param_update
from ks_latent.models import load_propagator_checkpoint_resized
from ks_latent.solver.lorenz96 import integrate, spinup

N, F_TRUE, A_TRUE, SIG = 40, 8.0, 1.0, 1.0
DT, EVERY, NENS = 0.01, 10, 20
OUT = Path("artifacts/experiments/l96")
CKPT = "artifacts/stage2_prop_patched_full_l96_crop_c1.pt"
MEAN, STD = 2.339044517746416, 3.6387645259074954


def rhs(x, F, a, g=0.0):
    """g: strength of an extra term x_i x_{i+2} that no (F, a) setting can represent."""
    out = a * (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + F
    return out + g * x * np.roll(x, -2, -1) if g else out


def rk4(x, F, a, g=0.0, dt=DT):
    k1 = rhs(x, F, a, g); k2 = rhs(x + .5 * dt * k1, F, a, g)
    k3 = rhs(x + .5 * dt * k2, F, a, g); k4 = rhs(x + dt * k3, F, a, g)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def phys_cycle(X, F, a, g=0.0):
    """X (n, N); F, a scalars or (n, 1)."""
    for _ in range(EVERY):
        X = rk4(X, F, a, g)
    return X


class Learned:
    def __init__(self):
        self.prop, _, _ = load_propagator_checkpoint_resized(CKPT, new_d_latent=N, new_n_tokens=N)
        self.prop.eval()

    @torch.no_grad()
    def cycle(self, X):
        z = torch.tensor((X - MEAN) / STD, dtype=torch.float32)
        return self.prop.step_one(z).numpy().astype(np.float64) * STD + MEAN


def gc_weights(c):
    d = np.minimum(np.arange(N), N - np.arange(N)).astype(float)
    return gaspari_cohn_taper(torch.as_tensor(d, dtype=torch.float64), c).numpy()


def local_etkf(Xf, y, c, infl):
    """Per-site local ETKF with every site observed; R-localization with GC half-width c (sites)."""
    n = Xf.shape[0]
    xm = Xf.mean(0); Xp = Xf - xm; d = y - xm
    w = gc_weights(c)
    Xa = np.empty_like(Xf)
    eye = np.eye(n)
    for s in range(N):
        rho = w[(np.arange(N) - s) % N]
        used = np.nonzero(rho > 1e-10)[0]
        Yu = Xp[:, used]
        C = Yu * (rho[used] / SIG ** 2)[None, :]
        A = (n - 1) * eye / infl + C @ Yu.T
        ev, V = np.linalg.eigh(A)
        Pa = (V / ev) @ V.T
        wbar = Pa @ (C @ d[used])
        W = (V * np.sqrt((n - 1) / ev)) @ V.T + wbar[:, None]
        Xa[:, s] = xm[s] + W.T @ Xp[:, s]
    return Xa


def truth_run(seed, n_cycles):
    cfg = Lorenz96Config(N=N, F=F_TRUE, dt=DT, snapshot_every=EVERY, spinup_time=50.0)
    x0 = spinup(cfg, np.random.default_rng(seed))
    return integrate(x0, cfg, n_cycles * EVERY)  # (n_cycles+1, N)


def run_filter(kind, case, c, infl, seed, n_cycles, burn, est=False, learned=None):
    """kind: 'phys' or 'learned'; case: (name, F_model, a_model, which) -- which in {'F','a'} is
    the estimated parameter."""
    truth = truth_run(seed, n_cycles + 200)
    clim = truth[:200]
    rng = np.random.default_rng(10_000 + seed)
    ens = clim[rng.choice(200, NENS, replace=False)]
    truth = truth[200:]
    _, Fm, am, which, g = case
    th = np.tile([Fm, am], (NENS, 1))  # columns: F, a
    idx = [{"F": 0, "a": 1}[ch] for ch in ("F", "a") if ch in which]
    if est:
        for j in idx:
            th[:, j] += rng.normal(0, 0.5 if j == 0 else 0.05, NENS)
    floors = np.array([0.05, 0.02])
    errs, spreads, prior_errs, thetas = [], [], [], []
    for k in range(1, n_cycles + 1):
        if kind == "learned":
            ens = learned.cycle(ens)
        else:
            ens = phys_cycle(ens, th[:, :1], th[:, 1:], g)
        if not np.isfinite(ens).all() or np.abs(ens).max() > 1e3:
            return {"rmse": float("inf"), "diverged": True}
        y = truth[k] + rng.normal(0, SIG, N)
        pe = np.sqrt(np.mean((ens.mean(0) - truth[k]) ** 2))
        xm = ens.mean(0)
        Xa = local_etkf(ens, y, c, infl)
        if est and kind == "phys":
            Y = ens - xm
            for j in idx:
                th[:, j] = global_param_update(th[:, j], Y, y - xm, SIG, rtps=0.9, min_std=floors[j])
        ens = Xa
        if k > burn:
            errs.append(np.sqrt(np.mean((ens.mean(0) - truth[k]) ** 2)))
            spreads.append(np.sqrt(np.mean(ens.var(0, ddof=1))))
            prior_errs.append(pe)
            thetas.append(th.mean(0).copy())
    return {"rmse": float(np.mean(errs)), "spread": float(np.mean(spreads)), "prior": float(np.mean(prior_errs)),
            "theta": np.mean(thetas[-50:], axis=0).tolist() if est else None, "diverged": False}


def one_cycle_error(kind, case, theta, learned=None, seed=99):
    """theta = (F, a) used by the model."""
    truth = truth_run(seed, 400)[50:]
    if kind == "learned":
        pred = learned.cycle(truth[:-1])
    else:
        pred = phys_cycle(truth[:-1], theta[0], theta[1], case[4])
    return float(np.sqrt(np.mean((pred - truth[1:]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_tune, n_eval = (60, 100) if a.quick else (150, 300)
    burn_tune, burn_eval = (20, 30) if a.quick else (50, 50)
    cases = [("F8.5", 8.5, 1.0, "F", 0.0), ("F9", 9.0, 1.0, "F", 0.0), ("F10", 10.0, 1.0, "F", 0.0),
             ("a1.1", 8.0, 1.1, "a", 0.0), ("a1.25", 8.0, 1.25, "a", 0.0)]
    # structural error: an extra g x_i x_{i+2} term, with F and a estimated (it cannot be absorbed)
    cases += [(f"g{g}", 8.0, 1.0, "Fa", g) for g in (0.02, 0.05, 0.1, 0.2, 0.4)]
    learned = Learned()
    grid = [(c, i) for c in (2, 4) for i in (1.0, 1.05, 1.1, 1.2, 1.4)]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    def tune_and_score(kind, case, est):
        best = min(grid, key=lambda g: run_filter(kind, case, g[0], g[1], 0, n_tune, burn_tune, est, learned)["rmse"])
        runs = [run_filter(kind, case, best[0], best[1], s, n_eval, burn_eval, est, learned) for s in (1, 2, 3)]
        return {"c": best[0], "infl": best[1], "rmse": [r["rmse"] for r in runs], "theta": [r["theta"] for r in runs],
                "mean": float(np.mean([r["rmse"] for r in runs])), "spread": [r.get("spread") for r in runs]}

    t0 = time.time()
    res["perfect"] = tune_and_score("phys", ("true", F_TRUE, 1.0, "F", 0.0), False)
    print("perfect", res["perfect"]["mean"], flush=True)
    res["learned"] = tune_and_score("learned", ("true", F_TRUE, 1.0, "F", 0.0), False)
    res["learned"]["one_cycle_err_sig"] = one_cycle_error("learned", cases[0], None, learned) / SIG
    print("learned", res["learned"]["mean"], res["learned"]["one_cycle_err_sig"], flush=True)
    for case in cases:
        for est in (False, True):
            key = f"phys_{case[0]}_{'est' if est else 'fixed'}"
            r = tune_and_score("phys", case, est)
            th = (np.mean([t for t in r["theta"] if t is not None], axis=0).tolist() if est else [case[1], case[2]])
            r["one_cycle_err_sig"] = one_cycle_error("phys", case, th) / SIG
            r["theta_final"] = th
            res[key] = r
            print(key, r["mean"], r["rmse"], "theta", th, "1cyc/sig", round(r["one_cycle_err_sig"], 3),
                  f"[{time.time() - t0:.0f}s]", flush=True)
    L = res["learned"]
    for case in cases:
        P = res[f"phys_{case[0]}_est"]
        win = ("physics" if all(p <= 0.9 * l for p, l in zip(P["rmse"], L["rmse"])) and P["mean"] <= 0.9 * L["mean"]
               else "learned" if all(l <= 0.9 * p for p, l in zip(P["rmse"], L["rmse"])) and L["mean"] <= 0.9 * P["mean"]
               else "tie")
        P["outcome"] = win
        P["outcome"] = win
        print(f"{case[0]}: 1cyc/sig {P['one_cycle_err_sig']:.3f} physics(est) {P['mean']:.3f} vs learned {L['mean']:.3f} -> {win}")
    (OUT / ("da_quick.json" if a.quick else "da_main.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
