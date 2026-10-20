"""Z20 E2 (Paper II App. D): multivariate normality of one-cycle latent priors, in latent and in physical
space, with enough members to have power, against climatology.

Input: the forecast ensembles of tiling 0 saved by `exp_h9_tiled_da.py --save-priors` (latent `Z` and the
decoded fields `U`, standardized units). Per cycle and site: Mardia skewness / kurtosis of the site's
c latent channels, and of the leading c principal components of its decoded patch (the same number of
dimensions), each with a Monte Carlo p-value against Gaussian samples of the same size passed through the
same procedure (`ks_latent.analysis.gaussianity`; for the patch, Gaussian samples with the ensemble's own
patch covariance, then the same PCA). Climatology: 256 states of an independent run (seed 3, 2 time units
apart), encoded, tested site by site in the same way (latent channels; PCs of the true patch).

Writes artifacts/experiments/da2d/<out>.json: rejection fractions at 5% and mean |marginal skewness|.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import skew

from ks_latent.analysis.gaussianity import mardia, mc_pvalues


def pcs(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def site_tests(Zs, Up, c, n_mc, rng):
    """Zs: (N, c) latent at one site; Up: (N, q) decoded patch pixels. Returns dict of p-values."""
    N = Zs.shape[0]
    pz = mc_pvalues(mardia(Zs.astype(np.float64)), N, c, n_mc, rng)
    Upc = pcs(Up.astype(np.float64), c)
    Xc = Up - Up.mean(0)
    w, V = np.linalg.eigh(Xc.T @ Xc / (N - 1))
    L = V * np.sqrt(np.clip(w, 0, None))  # Gaussian with the same patch covariance

    def proj(Zg):
        return pcs(Zg @ L.T, c)
    pu = mc_pvalues(mardia(Upc), N, c, n_mc, rng, project=proj, dim_in=Up.shape[1])
    return {"z_skew_p": pz[0], "z_kurt_p": pz[1], "u_skew_p": pu[0], "u_kurt_p": pu[1],
            "z_abs_marg_skew": float(np.mean(np.abs(skew(Zs, axis=0)))),
            "u_abs_marg_skew": float(np.mean(np.abs(skew(Upc, axis=0))))}


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k.replace("_p", "_reject_frac") if k.endswith("_p") else k] = \
            float(np.mean(v < 0.05)) if k.endswith("_p") else float(v.mean())
    out["n_tests"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--tag", default="p7c12d10_ks2d")
    ap.add_argument("--n-mc", type=int, default=100)
    ap.add_argument("--n-clim", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    P = np.load(a.priors)
    Z, U = P["Z"], P["U"]  # (C, N, d), (C, N, NX, NX)
    ns, patch, c = int(P["n_sites"]), int(P["patch"]), int(P["channels"])
    C, N = Z.shape[:2]

    def patches(F):  # (B, NX, NX) -> (B, ns*ns, patch*patch), site-major (x outer)
        B = F.shape[0]
        return F.reshape(B, ns, patch, ns, patch).transpose(0, 1, 3, 2, 4).reshape(B, ns * ns, patch * patch)
    rows = []
    for t in range(C):
        Zt, Ut = Z[t].reshape(N, ns * ns, c), patches(U[t])
        for s in range(ns * ns):
            rows.append(site_tests(Zt[:, s], Ut[:, s], c, a.n_mc, rng))
        print(f"cycle {int(P['cycle'][t])} done", flush=True)
    res = {"prior": summarize(rows), "N": N, "c": c, "cycles": P["cycle"].tolist(), "n_mc": a.n_mc}

    # climatology: independent run (seed 3), encoded
    from ks_latent.config import KSConfig2D
    from ks_latent.models import load_autoencoder_checkpoint_resized_2d
    from ks_latent.solver.ks2d import integrate2d
    NX = int(P["NX"]); L = float(P["L"]) if "L" in P else 44.0
    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    mean, std = float(P["mean"]), float(P["std"])
    cache = Path(f"artifacts/cache/ks2d_spinup_L{L:g}_NX{NX}_dt0.02_T200_seed3.npy")
    kcfg = KSConfig2D(Lx=L, Ly=L, NX=NX, NY=NX, dt=0.02, snapshot_every=100, spinup_time=200.0, seed=3)
    clim = integrate2d(np.load(cache), kcfg, n_steps=a.n_clim * 100, snapshot_every=100)[1:]
    clim = (clim.reshape(a.n_clim, -1) - mean) / std
    with torch.no_grad():
        Zc = torch.cat([ae.encode(torch.tensor(clim[i:i + 32], dtype=torch.float32)) for i in range(0, a.n_clim, 32)]).numpy()
    Zc, Uc = Zc.reshape(a.n_clim, ns * ns, c), patches(clim.reshape(a.n_clim, NX, NX))
    rows_c = [site_tests(Zc[:, s], Uc[:, s], c, a.n_mc, rng) for s in range(ns * ns)]
    res["climatology"] = summarize(rows_c)
    Path(f"artifacts/experiments/da2d/{a.out}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("prior", "climatology")}, indent=1))


if __name__ == "__main__":
    main()
