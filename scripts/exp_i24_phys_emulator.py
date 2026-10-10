"""Task I24 (2026-10-09): the equation-free physical-space baseline.

Same learned model as the latent filter, used as a PHYSICAL-space emulator, with the analysis done
by the standard physical-space local ETKF (ks_latent/da/letkf_phys.py, the I11 filter):

    forecast:  u -> blend_k roll_k^-1 D( M^n( E( roll_k u ) ) )      (K shifted tilings, Hann blend;
               latent model noise as in the latent runs)
    analysis:  LocalETKFPhys2D on the grid (GC half-width c in physical units, RTPS)

This isolates WHERE the analysis is done (latent sites vs grid blocks) while holding the model fixed.
Every member is re-encoded each cycle (the physical analysis is not in any latent space), which the
latent filter avoids (I8: re-encoding a blend costs accuracy).

Same truth, observation network and noise draws as exp_h9_tiled_da.py (default_rng(seed + 1)).
Physical RMSE in field units; steady = cycles 51-100. Writes artifacts/experiments/da2d/i24_<out>.json.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from ks_latent.config import KSConfig2D
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.letkf_phys import LocalETKFPhys2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d
from ks_latent.solver.twin import twin_truth
from ks_latent.utils.seeding import set_seed


def interp_periodic(obs, s, NX):
    m = obs.shape[0]
    xo = np.arange(-3, m + 3) * s
    sp = RectBivariateSpline(xo, xo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)
    x = np.arange(NX)
    return sp(x, x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p7c12_ks2d")
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--obs-stride", type=int, default=8)
    ap.add_argument("--obs-noise-std", type=float, default=0.1)
    ap.add_argument("--model-noise-std", type=float, default=0.02)
    ap.add_argument("--n-tilings", type=int, default=4, choices=[1, 4])
    ap.add_argument("--loc-c", type=float, default=16.0, help="GC half-width, physical units")
    ap.add_argument("--rtps", type=float, default=0.0)
    ap.add_argument("--ic-perturb-std", type=float, default=0.1, help="latent units, as the latent runs")
    ap.add_argument("--truth-lap-coef", type=float, default=1.0)
    ap.add_argument("--truth-advect-x", type=float, default=0.0, help="I28: truth has mean-flow advection c (x); models do not")
    ap.add_argument("--truth-forcing-amp", type=float, default=0.0, help="I28: truth has terrain-like forcing (std); models do not")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()
    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    PR = f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX, ns = g.NX, g.n_sites
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
    mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])

    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=a.seed)
    u0, truth_kw = twin_truth(kcfg, a.seed, a.truth_lap_coef, a.truth_advect_x, a.truth_forcing_amp)
    truth = integrate2d(u0, kcfg, n_steps=a.n_prop_steps * a.n_cycles * kcfg.snapshot_every, **truth_kw)
    truth = (truth - mean) / std  # standardized, as the latent runs

    s, N, hp = a.obs_stride, a.n_ensemble, patch // 2
    offs = [(0, 0)] if a.n_tilings == 1 else [(0, 0), (hp, 0), (0, hp), (hp, hp)]

    def hann(off):
        t = (np.arange(NX) - off) % patch
        return np.sin(np.pi * (t + 0.5) / patch) ** 2
    W = [hann(ox)[:, None] * hann(oy)[None, :] for ox, oy in offs]
    Wsum = sum(W); W = [w / Wsum for w in W]

    def enc(F, o):
        X = torch.tensor(np.roll(F, (-o[0], -o[1]), axis=(1, 2)).reshape(len(F), -1), dtype=torch.float32)
        with torch.no_grad():
            return torch.cat([ae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])

    def dec(Z, o):
        with torch.no_grad():
            U = torch.cat([ae.decode(Z[i:i + 32]) for i in range(0, len(Z), 32)]).double().numpy()
        return np.roll(U.reshape(len(Z), NX, NX), (o[0], o[1]), axis=(1, 2))

    def emulate(X):  # physical members (N, NX, NX), standardized -> one cycle later
        out = 0
        for w, o in zip(W, offs):
            Z = enc(X, o)
            fc = forecast_ensemble(prop, Z.clone(), Z, a.n_prop_steps, a.model_noise_std)
            out = out + w[None] * dec(fc.z_curr_final, o)
        return out

    xg = np.arange(0, NX, s)
    obs_xy = np.array([(i, j) for i in xg for j in xg])
    obs_idx = obs_xy[:, 0] * NX + obs_xy[:, 1]
    filt = LocalETKFPhys2D(NX, g.L, obs_xy, r_std=a.obs_noise_std * std, c=a.loc_c, block=4, rtps=a.rtps)

    # initial ensemble: the latent runs' initial ensemble (encoded truth + latent noise), decoded
    gen = torch.Generator().manual_seed(a.seed)
    z0 = enc(truth[:1], (0, 0))[0]
    X = dec(z0 + a.ic_perturb_std * torch.randn(N, z0.shape[0], generator=gen), (0, 0))
    rng = np.random.default_rng(a.seed + 1)
    rec = {k: [] for k in ("analysis", "prior", "interp", "spread")}
    f = lambda v, u: float(np.sqrt(np.mean((v - u) ** 2)) * std)  # noqa: E731
    for cyc in range(a.n_cycles):
        Xf = emulate(X)
        u = truth[(cyc + 1) * a.n_prop_steps]
        yg = u[::s, ::s] + a.obs_noise_std * rng.standard_normal((NX // s, NX // s))
        Xa = filt.analyze((Xf * std).reshape(N, -1), (yg * std).reshape(-1), obs_idx).reshape(N, NX, NX) / std
        if not np.isfinite(Xa).all():
            raise FloatingPointError(f"non-finite analysis at cycle {cyc}")
        rec["analysis"].append(f(Xa.mean(0), u)); rec["prior"].append(f(Xf.mean(0), u))
        rec["interp"].append(f(interp_periodic(yg, s, NX), u))
        rec["spread"].append(float(np.sqrt(np.mean(Xa.var(0, ddof=1))) * std))
        X = Xa
        if rec["analysis"][-1] > 5 * float(u.std() * std):
            print(f"DIVERGED at cycle {cyc + 1}", flush=True)
            break
        if cyc % 10 == 9:
            print(f"cycle {cyc + 1}: " + " ".join(f"{k} {v[-1]:.3f}" for k, v in rec.items()), flush=True)
    late = slice(a.n_cycles // 2, None)
    steady = {k: float(np.mean(v[late])) for k, v in rec.items() if len(v) > a.n_cycles // 2}
    if "analysis" in steady:
        steady["calib"] = steady["spread"] / steady["analysis"]
    steady["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    out = {"args": vars(a), "steady": steady, "per_cycle": rec, "wall_s": time.time() - t0}
    Path(f"artifacts/experiments/da2d/i24_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, {k: round(v, 3) for k, v in steady.items()}, f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
