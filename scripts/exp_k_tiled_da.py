"""Paper II, Kolmogorov-flow data assimilation (SK2-SK3, 2026-10-11): the latent filter of the 2D KS
experiments (per-site perturbed-observation EnKF with R-localization and RTPS, shifted tilings blended
with Hann weights; `scripts/exp_h9_tiled_da.py`) on Kolmogorov flow in a channel of fixed height 2*pi and
width `--scale` * 2*pi.

Truth: `ks_latent.solver.kolmogorov` (ETDRK4, the training data's nu, alpha, forcing_n, dt and snapshot
interval, read from the model's training-set metadata) from a spun-up state (cached per scale, seed and
drag). Grid: the training spacing 2*pi/128 in x and y, so NX = 128 * scale, NY = 128, and the model is
resized to (8 * scale) x 8 sites with the same weights. Fields (vorticity) are standardized with the
checkpoint's statistics; errors are reported in vorticity units (x std). Observations: every
`--obs-stride` grid point in x and y, Gaussian errors `--obs-noise-std` in standardized units.

Tilings: `--n-tilings` with `--tiling-axes xy` (half-patch offsets in x and y) or `x` (x only).
Kolmogorov flow is invariant to x-shifts but to y-shifts only by whole forcing periods (32 grid points
for n = 4), so y-shifted tilings show the encoder patches at forcing phases outside the training
distribution; `--floor-per-tiling` reports each tiling's own decoder floor to measure this.

Reported (steady state = second half of the cycles): analysis error of tiling 0 and of the blend,
decoder floors (tiling 0 and blended, and per tiling), periodic cubic-spline interpolation of the
observations, spread over error, free-run error. Writes artifacts/experiments/kda/k_<out>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from ks_latent.config import KolmogorovConfig
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D
from ks_latent.da.tilings import blend_weights, frame_obs_sites, tiling_offsets
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov
from ks_latent.utils.seeding import set_seed

OUT = Path("artifacts/experiments/kda")


def interp_periodic(obs: np.ndarray, s: int, NX: int, NY: int) -> np.ndarray:
    """Periodic bicubic spline through observations on the grid [::s, ::s] of an (NX, NY) field."""
    mx, my = obs.shape
    xo, yo = np.arange(-3, mx + 3) * s, np.arange(-3, my + 3) * s
    sp = RectBivariateSpline(xo, yo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)
    return sp(np.arange(NX), np.arange(NY))


def kolmogorov_truth(md: dict, scale: int, seed: int, n_snap: int) -> tuple[np.ndarray, KolmogorovConfig]:
    """Truth trajectory (n_snap + 1, NX, NY) in vorticity units at the training spacing and snapshot
    interval; the spun-up initial state is cached per (alpha, scale, seed)."""
    cfg = KolmogorovConfig(Lx=2 * math.pi * scale, Ly=2 * math.pi, NX=128 * scale, NY=128, nu=float(md["nu"]),
                           alpha=float(md["alpha"]), forcing_n=int(md["forcing_n"]), dt=float(md["dt"]),
                           snapshot_every=int(md["snapshot_every"]), spinup_time=100.0, seed=seed)
    cache = Path(f"artifacts/cache/kolm_spinup_a{cfg.alpha:g}_x{scale}_seed{seed}.npy")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, spinup_kolmogorov(cfg, np.random.default_rng(seed + 500)))
    traj = integrate_kolmogorov(np.load(cache), cfg, n_steps=n_snap * cfg.snapshot_every)
    return traj, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="k_xs005_c12d10", help="checkpoint tag (stage{1,2}_*_kolmogorov_crop_<tag>.pt)")
    ap.add_argument("--scale", type=int, default=1, help="channel width in units of 2*pi (height fixed at 2*pi)")
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5, help="model steps per cycle (dt_snap 0.2 each)")
    ap.add_argument("--obs-stride", type=int, default=8)
    ap.add_argument("--obs-noise-std", type=float, default=0.1)
    ap.add_argument("--model-noise-std", type=float, default=0.02)
    ap.add_argument("--gc-c", type=float, default=1.5)
    ap.add_argument("--rtps", type=float, default=0.7)
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--ic-perturb-std", type=float, default=0.1)
    ap.add_argument("--n-tilings", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--tiling-axes", default="x", choices=["xy", "x"],
                    help="x (default): Kolmogorov flow is invariant to x-shifts only; y-shifted tilings fail (SK4 smoke)")
    ap.add_argument("--weights", default="hann", choices=["hann", "uniform"])
    ap.add_argument("--floor-per-tiling", action="store_true", help="also report each tiling's decoder floor")
    ap.add_argument("--no-free", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()

    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    PR = f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    acfg = ck["ae_config"]
    patch, patch_y = acfg.NX // acfg.n_sites_x, acfg.NY // acfg.n_sites_y
    with h5py.File(ck["train_dataset"], "r") as f:
        md = dict(f["metadata"].attrs)
    dx, dy = float(md["Lx"]) / int(md["NX"]), float(md["Ly"]) / int(md["NY"])
    assert patch == patch_y and abs(dx - 2 * math.pi / 128) < 1e-12 and abs(dy - dx) < 1e-12, "grid must match training"
    NX, NY = 128 * a.scale, 128
    nsx, nsy = NX // patch, NY // patch
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=nsx, new_n_sites_y=nsy, new_NX=NX, new_NY=NY)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=nsx, new_n_sites_y=nsy)[0].eval()
    mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])

    traj, kcfg = kolmogorov_truth(md, a.scale, a.seed, a.n_prop_steps * a.n_cycles)
    truth = (traj - mean) / std  # (T+1, NX, NY), standardized

    s, N = a.obs_stride, a.n_ensemble
    offs = tiling_offsets(patch, a.n_tilings, a.tiling_axes)
    W = blend_weights(NX, NY, patch, offs, uniform=a.weights == "uniform")

    def enc(F, o):  # (B, NX, NY) original frame -> latents of tiling o
        X = torch.tensor(np.roll(F, (-o[0], -o[1]), axis=(1, 2)).reshape(len(F), -1), dtype=torch.float32)
        with torch.no_grad():
            return torch.cat([ae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])

    def dec(Z, o):  # latents of tiling o -> (B, NX, NY) original frame
        with torch.no_grad():
            U = torch.cat([ae.decode(Z[i:i + 32]) for i in range(0, len(Z), 32)]).double().numpy()
        return np.roll(U.reshape(len(Z), NX, NY), (o[0], o[1]), axis=(1, 2))

    FO = {o: frame_obs_sites(NX, NY, patch, s, o) for o in set(offs)}
    assert torch.equal(FO[(0, 0)][2], obs_site_map_2d(NX, NY, nsx, nsy, s, s).to(torch.int64)), "site convention"

    def make_h(o):
        ix, iy = FO[o][0], FO[o][1]
        return lambda z: ae.decode(z.to(torch.float32).unsqueeze(0)).squeeze(0).reshape(NX, NY)[ix][:, iy].reshape(-1).double()

    d = enc(truth[:1], (0, 0)).shape[1]
    c = d // (nsx * nsy)
    R_diag = torch.full((FO[(0, 0)][2].shape[0],), a.obs_noise_std ** 2, dtype=torch.float64)
    gen = torch.Generator().manual_seed(a.seed)
    Zc = [enc(truth[:1], o)[0] + a.ic_perturb_std * torch.randn(N, d, generator=gen) for o in offs]
    Zp = [z.clone() for z in Zc]
    Zfree_p, Zfree_c = Zp[0].clone(), Zc[0].clone()
    rng = np.random.default_rng(a.seed + 1)
    keys = ("tiling0", "blend", "floor1", "floorK", "interp", "spread0", "spread_blend", "free")
    rec = {k: [] for k in keys}
    rec["analysis_per_tiling"] = []
    if a.floor_per_tiling:
        rec["floor_per_tiling"] = []
    diverged = None
    t = 0
    for cyc in range(a.n_cycles):
        Zf = []
        for k in range(len(offs)):
            fc = forecast_ensemble(prop, Zp[k], Zc[k], a.n_prop_steps, a.model_noise_std)
            Zp[k] = fc.z_prev_final
            Zf.append(fc.z_curr_final)
        if not a.no_free:
            fcf = forecast_ensemble(prop, Zfree_p, Zfree_c, a.n_prop_steps, 0.0)
            Zfree_p, Zfree_c = fcf.z_prev_final, fcf.z_curr_final
        t += a.n_prop_steps
        u = truth[t]
        yg = u[::s, ::s] + a.obs_noise_std * rng.standard_normal(u[::s, ::s].shape)
        yv = torch.tensor(yg.reshape(-1))
        Ua = []
        for k, o in enumerate(offs):
            filt = LETKFLocal2D(make_h(o), R_diag, FO[o][2], nsx, nsy, c, a.gc_c, inflation=a.inflation,
                                rtps=a.rtps, seed=a.seed * 1000 + cyc * 20 + k)
            Zc[k] = filt.analyze(Zf[k].double(), yv).Z_analysis.float()
            Ua.append(dec(Zc[k], o))
        Ub = sum(w[None] * U for w, U in zip(W, Ua))
        f = lambda v: float(np.sqrt(np.mean((v - u) ** 2)) * std)  # noqa: E731  vorticity units
        rec["tiling0"].append(f(Ua[0].mean(0)))
        rec["analysis_per_tiling"].append([f(U.mean(0)) for U in Ua])
        rec["blend"].append(f(Ub.mean(0)))
        rec["spread0"].append(float(np.sqrt(Ua[0].var(0, ddof=1).mean()) * std))
        rec["spread_blend"].append(float(np.sqrt(Ub.var(0, ddof=1).mean()) * std))
        floors = [dec(enc(u[None], o), o)[0] for o in offs]
        rec["floor1"].append(f(floors[0]))
        rec["floorK"].append(f(sum(w * F for w, F in zip(W, floors))))
        if a.floor_per_tiling:
            rec["floor_per_tiling"].append([f(F) for F in floors])
        rec["interp"].append(f(interp_periodic(yg, s, NX, NY)))
        rec["free"].append(f(dec(Zfree_c, (0, 0)).mean(0)) if not a.no_free else float("nan"))
        if not np.isfinite(rec["blend"][-1]) or rec["blend"][-1] > 20 * std:
            diverged = cyc + 1
            break
        if (cyc + 1) % 10 == 0 or cyc == 0:
            print(f"cycle {cyc + 1}: blend {rec['blend'][-1]:.3f} tiling0 {rec['tiling0'][-1]:.3f} "
                  f"spread {rec['spread_blend'][-1]:.3f} floorK {rec['floorK'][-1]:.3f} interp {rec['interp'][-1]:.3f} "
                  f"free {rec['free'][-1]:.3f}", flush=True)

    late = slice(len(rec["blend"]) // 2, None)
    steady = {k: float(np.mean(rec[k][late])) for k in keys}
    steady["calib"] = steady["spread_blend"] / steady["blend"]
    steady["analysis_per_tiling"] = np.mean(np.array(rec["analysis_per_tiling"])[late], axis=0).round(4).tolist()
    if a.floor_per_tiling:
        steady["floor_per_tiling"] = np.mean(np.array(rec["floor_per_tiling"])[late], axis=0).round(4).tolist()
    steady["diverged_at_cycle"] = diverged
    out = {"args": vars(a), "grid": {"NX": NX, "NY": NY, "n_sites": [nsx, nsy], "patch": patch, "d_latent": d,
                                     "offsets": offs, "dt_snap": float(md["dt_snap"]), "alpha": float(md["alpha"]),
                                     "field_std": std},
           "steady": steady, "per_cycle": rec, "wall_s": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"k_{a.out}.json").write_text(json.dumps(out, indent=1))
    print(a.out, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in steady.items()}, flush=True)


if __name__ == "__main__":
    main()
