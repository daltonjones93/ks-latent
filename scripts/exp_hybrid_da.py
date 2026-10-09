"""Accuracy study (2026-10-08): hybrid latent-forecast / physical-analysis ensemble filter, 2D KS.

Same trained models, same truth, observation network and noise as the Part II latent filter
(run_da_pff_2d_ltransfer.py); no retraining. Each cycle:
  1. forecast the latent ensemble n_prop_steps with the learned propagator (+ model noise),
  2. decode every member to a standardized field: the prior ensemble X_f (N, NX*NY),
  3. analysis in physical space, one deterministic ensemble transform (Hunt et al. 2007) per patch:
     observations within the Gaspari-Cohn support of the patch (same site taper, c = 3 sites),
     R-localization R_i / rho_i, observation operator = grid-point selection (linear, exact);
     the patch's grid points are updated with the patch's transform weights,
  4. re-encode the analysis members as the next cycle's latent ensemble.
Reported per analysis: physical RMSE of the analysis mean (mean of the analysed fields), of the
prior mean, of the decoder floor, and of a periodic cubic-spline interpolation of the SAME
observations (no model). Writes artifacts/experiments/da2d/hybrid_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from ks_latent.config import KSConfig2D
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.localization import build_latent_taper_matrix_2d
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed

AE = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"
PR = "artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt"


def interp_periodic(obs: np.ndarray, s: int, NX: int) -> np.ndarray:
    m = obs.shape[0]
    xo = np.arange(-3, m + 3) * s
    sp = RectBivariateSpline(xo, xo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)
    x = np.arange(NX)
    return sp(x, x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--obs-stride", type=int, default=8)
    ap.add_argument("--obs-noise-std", type=float, default=0.1)
    ap.add_argument("--filter-obs-std", type=float, default=None)
    ap.add_argument("--model-noise-std", type=float, default=0.08)
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--gc-c", type=float, default=3.0)
    ap.add_argument("--ic-perturb-std", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-prior", action="store_true",
                    help="Save prior mean, truth, observations and decoded perturbations per cycle (npz).")
    ap.add_argument("--diag", action="store_true",
                    help="Record, per cycle, how much of the prior error lies in the span of the decoded "
                         "ensemble perturbations (global least squares), and band spectra of the prior error "
                         "and of the perturbations.")
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()

    ck = torch.load(AE, map_location="cpu", weights_only=False)
    cfg0 = ck["ae_config"]
    patch = cfg0.NX // cfg0.n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX, ns = g.NX, g.n_sites
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
    mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])

    dt = 0.02
    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                      seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    u0 = np.load(cache) if cache.exists() else spinup2d(kcfg, np.random.default_rng(a.seed + 500))
    truth = integrate2d(u0, kcfg, n_steps=a.n_prop_steps * a.n_cycles * kcfg.snapshot_every)
    truth = (truth.reshape(truth.shape[0], -1) - mean) / std  # standardized, (T+1, NX*NY)
    truth_t = torch.tensor(truth, dtype=torch.float32)

    s = a.obs_stride
    obs_pix = (np.arange(0, NX, s)[:, None] * NX + np.arange(0, NX, s)[None, :]).reshape(-1)
    site_of_obs = obs_site_map_2d(NX, NX, ns, ns, s, s).numpy()
    taper = build_latent_taper_matrix_2d(ns, ns, channels=1, c=a.gc_c).numpy()  # (P, P) site taper
    r_std = a.obs_noise_std if a.filter_obs_std is None else a.filter_obs_std
    # grid points of each patch (site-major ordering, x outer / y inner, as the autoencoder)
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NX), indexing="ij")
    site_of_pix = ((ix // patch) * ns + (iy // patch)).reshape(-1)
    pix_of_site = [np.nonzero(site_of_pix == p)[0] for p in range(ns * ns)]

    rng = np.random.default_rng(a.seed + 1)
    saved = []
    with torch.no_grad():
        z0 = ae.encode(truth_t[0:1])[0]
    Zc = z0 + a.ic_perturb_std * torch.randn(a.n_ensemble, z0.shape[0], generator=torch.Generator().manual_seed(a.seed))
    Zp = Zc.clone()
    N = a.n_ensemble
    rec = {k: [] for k in ("analysis", "prior", "floor", "interp", "spread_phys", "analysis_plus_resid",
                           "prior_plus_resid", "span_fraction", "err_bands", "pert_bands")}
    kk = 2 * np.pi * np.fft.fftfreq(NX, d=g.L / NX); K = np.sqrt(kk[:, None] ** 2 + kk[None, :] ** 2)
    BANDS = [(0, 0.6), (0.6, 1.14), (1.14, 1.6), (1.6, 99)]
    bandpow = lambda v: [float((np.abs(np.fft.fft2(v.reshape(NX, NX))) ** 2 / NX ** 4)[(K >= lo) & (K < hi)].sum()) for lo, hi in BANDS]  # noqa: E731
    t = 0
    for cyc in range(a.n_cycles):
        fc = forecast_ensemble(prop, Zp, Zc, a.n_prop_steps, a.model_noise_std)
        Zp_next, Zf = fc.z_prev_final, fc.z_curr_final
        t += a.n_prop_steps
        with torch.no_grad():
            Xf = torch.cat([ae.decode(Zf[i:i + 16]) for i in range(0, N, 16)]).double().numpy()  # (N, NX*NX)
        u = truth[t]
        y = u[obs_pix] + a.obs_noise_std * rng.standard_normal(obs_pix.shape[0])
        xm = Xf.mean(0)
        Xp = (Xf - xm) * a.inflation
        if a.save_prior and cyc >= a.n_cycles // 2:
            saved.append((xm.astype(np.float32), u.astype(np.float32), y.astype(np.float32),
                          Xp[::4].astype(np.float32)))
        if a.diag:
            e = xm - u
            coef, *_ = np.linalg.lstsq(Xp.T, e, rcond=None)
            rec["span_fraction"].append(float(1 - np.sum((e - Xp.T @ coef) ** 2) / np.sum(e ** 2)))
            rec["err_bands"].append(bandpow(e * std))
            rec["pert_bands"].append(list(np.mean([bandpow(Xp[i] * std) for i in range(0, N, 8)], axis=0)))
        Yp = Xp[:, obs_pix]  # (N, m): linear observation operator
        dinn = y - xm[obs_pix]
        Xa = np.empty_like(Xf)
        eye = np.eye(N)
        for p in range(ns * ns):
            rho = taper[p, site_of_obs]
            used = rho > 1e-10
            C = Yp[:, used] * (rho[used] / r_std**2)[None, :]
            A = (N - 1) * eye + C @ Yp[:, used].T
            ev, V = np.linalg.eigh(A)
            Pa = (V / ev) @ V.T
            wbar = Pa @ (C @ dinn[used])
            W = (V * np.sqrt((N - 1) / ev)) @ V.T + wbar[:, None]
            pts = pix_of_site[p]
            Xa[:, pts] = xm[pts][None, :] + W.T @ Xp[:, pts]
        xa = Xa.mean(0)
        with torch.no_grad():
            Za = torch.cat([ae.encode(torch.tensor(Xa[i:i + 16], dtype=torch.float32)) for i in range(0, N, 16)])
            floor = ae.decode(ae.encode(torch.tensor(u[None], dtype=torch.float32)))[0].double().numpy()
        ui = interp_periodic(y.reshape(NX // s, NX // s), s, NX).reshape(-1)
        f = lambda v: float(np.sqrt(np.mean((v - u) ** 2)) * std)  # noqa: E731  field units
        rec["analysis"].append(f(xa)); rec["prior"].append(f(xm)); rec["floor"].append(f(floor))
        rec["interp"].append(f(ui)); rec["spread_phys"].append(float(Xa.std(0).mean() * std))
        # background + interpolated observation residual (optimal interpolation with a model
        # background and an isotropic, spline-shaped correction); not fed back into the cycle
        for name, bg in (("analysis_plus_resid", xa), ("prior_plus_resid", xm)):
            res = (y - bg[obs_pix]).reshape(NX // s, NX // s)
            rec[name].append(f(bg + interp_periodic(res, s, NX).reshape(-1)))
        Zp, Zc = Zp_next, Za
        if not np.isfinite(xa).all():
            raise FloatingPointError(f"non-finite analysis at cycle {cyc}")
    late = slice(a.n_cycles // 2, None)
    summ = {k: st.mean(v[late]) for k, v in rec.items() if v and not isinstance(v[0], list)}
    for k in ("err_bands", "pert_bands"):
        if rec[k]:
            summ[k] = list(np.mean(np.array(rec[k])[late], axis=0))
    out = {"tag": a.tag, "args": vars(a), "steady": summ, "per_cycle": rec, "wall_s": time.time() - t0,
           "field_std": float(truth.std() * std)}
    Path("artifacts/experiments/da2d").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/experiments/da2d/hybrid_{a.tag}.json").write_text(json.dumps(out, indent=2))
    if saved:
        np.savez_compressed(f"artifacts/experiments/da2d/hybrid_{a.tag}_prior.npz",
                            xm=np.stack([t[0] for t in saved]), u=np.stack([t[1] for t in saved]),
                            y=np.stack([t[2] for t in saved]), Xp=np.stack([t[3] for t in saved]),
                            obs_pix=obs_pix, NX=NX, std=std)
    print(a.tag, {k: (round(v, 3) if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in summ.items()}, f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
