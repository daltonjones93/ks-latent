"""Per-site latent LETKF for the CM1 column local field (E2), with open lateral boundaries.

The local ensemble transform Kalman filter of Hunt, Kostelich & Szunyogh (2007, Physica D 230:112),
applied site by site to the latent field `(P, c)`:

  for each site s with centre x_s:
    weights      w_i  = GC(|x_s - x_obs_i| / c_loc)          Gaspari-Cohn, zero beyond 2 c_loc
    used obs     i with w_i > 0;   R-localization: R_i -> R_i / w_i
    Yp  = Yb - mean(Yb)            (N, m_s)  observation-space perturbations of the forecast ensemble
    C   = Yp R_loc^-1              (N, m_s)
    Pt  = [ (N-1)/rho I + C Yp^T ]^-1                        (N, N), rho: multiplicative inflation
    wbar = Pt C (y - mean(Yb))                               (N,)
    W   = [ (N-1) Pt ]^{1/2}                                 symmetric square root
    Za[:, s] = mean(Zb[:, s]) + (wbar + W)^T Zp[:, s]        all c channels of site s at once

Yb = H(decode(Zb)) for each member, so the observation operator may be nonlinear and the decoder may
have reach > 0: nothing needs a Jacobian. Distances are plain |x_s - x_obs| in km: the lateral
boundaries are open, so nothing wraps around (unlike the periodic KS / Kolmogorov filters in
`ks_latent/da/`). Optional RTPS relaxes the analysis spread toward the prior spread (Whitaker &
Hamill 2012), per site and channel, after the update.

With no localization (c_loc large), rho = 1 and a linear H, the analysis mean and covariance equal the
Kalman filter's applied to the ensemble's sample covariance; `tests/unit/test_cm1_letkf.py` checks this
to machine precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """Gaspari & Cohn (1999) fifth-order compactly supported correlation; r = distance / c, zero at r >= 2."""
    r = np.abs(np.asarray(r, np.float64))
    out = np.zeros_like(r)
    a = r <= 1
    b = (r > 1) & (r < 2)
    ra, rb = r[a], r[b]
    out[a] = -0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3 - 5 / 3 * ra**2 + 1
    out[b] = rb**5 / 12 - 0.5 * rb**4 + 0.625 * rb**3 + 5 / 3 * rb**2 - 5 * rb + 4 - 2 / (3 * rb)
    return out


@dataclass
class LETKFConfig:
    loc_km: float = 10.0       # Gaspari-Cohn half-width c: weights reach zero at 2c
    inflation: float = 1.0     # multiplicative, in ensemble space (rho)
    rtps: float = 0.0          # relaxation to prior spread, 0 = off


def letkf_analysis(Zb: np.ndarray, Yb: np.ndarray, y: np.ndarray, r_var: np.ndarray, x_site_km: np.ndarray,
                   x_obs_km: np.ndarray, cfg: LETKFConfig) -> np.ndarray:
    """`Zb` (N, P, c) forecast latent ensemble; `Yb` (N, m) its observation-space values; `y` (m,)
    observations with error variances `r_var` (m,), at `x_obs_km`; site centres `x_site_km` (P,).
    Returns the analysis ensemble (N, P, c)."""
    Zb, Yb, y = np.asarray(Zb, np.float64), np.asarray(Yb, np.float64), np.asarray(y, np.float64)
    N, P, c = Zb.shape
    if Yb.shape != (N, y.size) or r_var.shape != y.shape or x_obs_km.shape != y.shape or x_site_km.shape != (P,):
        raise ValueError("inconsistent shapes")
    if N < 2:
        raise ValueError("need at least 2 members")
    zbar = Zb.mean(0)
    Zp = Zb - zbar
    ybar = Yb.mean(0)
    Yp = Yb - ybar
    d = y - ybar
    Za = Zb.copy()
    for s in range(P):
        w = gaspari_cohn((x_site_km[s] - x_obs_km) / cfg.loc_km)
        u = w > 0
        if not u.any():
            continue
        Ys = Yp[:, u]
        C = Ys * (w[u] / r_var[u])[None, :]                       # Yp R_loc^-1
        A = (N - 1) / cfg.inflation * np.eye(N) + C @ Ys.T
        ev, V = np.linalg.eigh(A)
        if ev.min() <= 0:
            raise np.linalg.LinAlgError(f"site {s}: ensemble-space matrix not positive definite (min eig {ev.min():.3g})")
        Pt = (V / ev) @ V.T
        wbar = Pt @ (C @ d[u])
        W = (V * np.sqrt((N - 1) / ev)) @ V.T
        Za[:, s] = zbar[s] + (wbar[None, :] + W) @ Zp[:, s]       # rows: members
    if cfg.rtps > 0:
        sb, za_bar = Zp.std(0, ddof=1), Za.mean(0)
        Zap = Za - za_bar
        sa = Zap.std(0, ddof=1)
        fac = np.where(sa > 0, 1 + cfg.rtps * (sb - sa) / np.where(sa > 0, sa, 1), 1.0)
        Za = za_bar + Zap * fac
    return Za
