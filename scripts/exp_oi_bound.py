"""Optimal-interpolation bound (accuracy study, 2026-10-09): the best linear analysis for THIS
model's prior, with the prior error covariance estimated from the prior errors themselves
(isotropic, from the mean error spectrum over the saved cycles), observations as in the DA runs.
    x_a = x_f + B H^T (H B H^T + R)^-1 (y - H x_f),   B isotropic from the empirical error spectrum.
Also reports the same analysis with B taken from the ENSEMBLE perturbation spectrum (isotropized),
and with the analysis restricted to the ensemble span (what a 64-member subspace filter can do).
Input: artifacts/experiments/da2d/hybrid_<tag>_prior.npz (exp_hybrid_da.py --save-prior).
"""
import sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "prior_mn0.02"
D = np.load(f"artifacts/experiments/da2d/hybrid_{tag}_prior.npz")
xm, u, y, Xp, obs, NX, std = D["xm"].astype(float), D["u"].astype(float), D["y"].astype(float), D["Xp"].astype(float), D["obs_pix"], int(D["NX"]), float(D["std"])
T = xm.shape[0]; r2 = 0.1 ** 2
E = xm - u  # standardized prior errors, (T, NX*NX)
# isotropic spectral covariance from the errors: power per wavenumber shell
k = 2 * np.pi * np.fft.fftfreq(NX); K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
P = np.mean(np.abs(np.fft.fft2(E.reshape(T, NX, NX))) ** 2, axis=0) / NX ** 2  # E|e_hat|^2 / NX^2 -> covariance spectrum
Pp = np.mean(np.abs(np.fft.fft2(Xp.reshape(-1, NX, NX))) ** 2, axis=0) / NX ** 2
def cov_from_spectrum(S):
    """Circulant covariance C[x, x'] = c(x - x'); return the (m x m) obs block and (NX^2 x m) cross block."""
    c = np.real(np.fft.ifft2(S))  # correlation function on the grid, c(0) = variance
    idx = np.arange(NX * NX); ix, iy = idx // NX, idx % NX
    ox, oy = obs // NX, obs % NX
    dX = (ix[:, None] - ox[None, :]) % NX; dY = (iy[:, None] - oy[None, :]) % NX
    BHt = c[dX, dY]                     # (NX^2, m)
    dXo = (ox[:, None] - ox[None, :]) % NX; dYo = (oy[:, None] - oy[None, :]) % NX
    HBHt = c[dXo, dYo]                  # (m, m)
    return BHt, HBHt
def oi(S, scale=1.0):
    BHt, HBHt = cov_from_spectrum(S * scale)
    K = BHt @ np.linalg.inv(HBHt + r2 * np.eye(len(obs)))
    errs = []
    for t in range(T):
        xa = xm[t] + K @ (y[t] - xm[t][obs])
        errs.append(np.sqrt(np.mean((xa - u[t]) ** 2)) * std)
    return float(np.mean(errs))
prior = float(np.mean(np.sqrt(np.mean(E ** 2, axis=1))) * std)
print(f"prior RMSE {prior:.3f} (field units); obs noise {0.1 * std:.3f}")
print(f"OI with the TRUE (empirical, isotropic) prior error spectrum:     {oi(P):.3f}")
for sc in (0.25, 0.5, 2.0):
    print(f"   same, B scaled x{sc}:                                        {oi(P, sc):.3f}")
print(f"OI with the ENSEMBLE perturbation spectrum (isotropized):         {oi(Pp):.3f}")
# subspace bound: analysis restricted to the span of the saved perturbations (16 per cycle) -- illustrative

# ---- decomposition: large-scale (|k| < kc) vs small-scale prior error, and OI on the large-scale part
# with the small-scale error treated as extra observation noise (it is white-like at the obs points).
for kc in (1.14, 1.6):
    lo = K < kc
    var_L = P[lo].sum() / NX ** 2; var_S = P[~lo].sum() / NX ** 2
    PL = np.where(lo, P, 0.0)
    BHt, HBHt = cov_from_spectrum(PL)
    Kg = BHt @ np.linalg.inv(HBHt + (r2 + var_S) * np.eye(len(obs)))
    errs_L, errs_tot = [], []
    EL = np.real(np.fft.ifft2(np.fft.fft2(E.reshape(T, NX, NX)) * lo[None])).reshape(T, -1)
    for t in range(T):
        inc = Kg @ (y[t] - xm[t][obs])
        errs_L.append(np.sqrt(np.mean((EL[t] + inc) ** 2)) * std)
        errs_tot.append(np.sqrt(np.mean((E[t] + inc) ** 2)) * std)
    print(f"kc={kc}: prior error split L/S (RMSE) {np.sqrt(var_L) * std:.3f}/{np.sqrt(var_S) * std:.3f}; "
          f"OI on L with R_eff = r^2 + var_S: large-scale error {np.mean(errs_L):.3f}, total {np.mean(errs_tot):.3f}; "
          f"interpolation-style bound if var_S were 0: see below")
    Kg0 = BHt @ np.linalg.inv(HBHt + r2 * np.eye(len(obs)))
    errs0 = [np.sqrt(np.mean((EL[t] + Kg0 @ (y[t] - xm[t][obs] - (E[t] - EL[t])[obs])) ** 2)) * std for t in range(T)]
    print(f"        same with the small-scale error removed from the innovations (what a seam-free decoder would give): "
          f"large-scale error {np.mean(errs0):.3f}")

# ---- strongest no-model baseline: statistical (climatological) interpolation, x_f = 0, B = truth spectrum
Pc = np.mean(np.abs(np.fft.fft2(u.reshape(T, NX, NX))) ** 2, axis=0) / NX ** 2
BHt, HBHt = cov_from_spectrum(Pc)
Kc = BHt @ np.linalg.inv(HBHt + r2 * np.eye(len(obs)))
errs = [np.sqrt(np.mean((Kc @ y[t] - u[t]) ** 2)) * std for t in range(T)]
print(f"no-model statistical interpolation (climatological B, x_f = 0):   {np.mean(errs):.3f}")
