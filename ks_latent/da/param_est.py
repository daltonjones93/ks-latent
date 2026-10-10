"""Online estimation of a global model parameter by state augmentation (task J16, 2026-10-09).

Each ensemble member carries its own value theta_i of a scalar model parameter (e.g. the coefficient
of the nabla^2 term, or a mean-flow speed) and is forecast with it. At each analysis the parameter
is updated with a GLOBAL ensemble transform (Hunt et al. 2007 ETKF, no localization: the parameter
is a single global quantity), using all observations:

    Y  = H X_f' (N x m),  C = Y R^{-1},  P_a = [(N-1) I + C Y^T]^{-1},
    w_bar = P_a C d,  W = [(N-1) P_a]^{1/2} + w_bar 1^T,  theta_a = theta_bar + W^T theta'.

The analysis mean then equals the ensemble Kalman regression
theta_bar + Cov(theta, HX) [Cov(HX) + R]^{-1} d with Cov from the ensemble (N - 1 normalization),
tested in tests/unit/test_param_est.py. Two safeguards keep the parameter ensemble alive (standard in
EnKF parameter estimation, e.g. Anderson 2001; Aksoy, Zhang & Nielsen-Gammon 2006 -- verify):
relaxation of the parameter spread to its prior (RTPS, alpha) and a minimum spread (floor).
"""
from __future__ import annotations

import numpy as np


def global_params_update(Theta: np.ndarray, Y: np.ndarray, d: np.ndarray, r_std: float,
                         rtps: float = 0.0, min_std=0.0) -> np.ndarray:
    """Joint version for p parameters (J17H, 2026-10-10): `Theta` (N, p). The same global ETKF weights W
    update every column, so the analysis mean is the joint ensemble Kalman regression
    theta_bar + Cov(theta, HX)[Cov(HX) + R]^{-1} d (one column at a time this is the scalar case: the
    transform depends only on Y, not on theta). RTPS and the floor (`min_std`, scalar or (p,)) act per
    column. Returns the (N, p) analysis."""
    Theta = np.asarray(Theta, dtype=np.float64)
    if Theta.ndim != 2:
        raise ValueError(f"Theta must be (N, p), got {Theta.shape}")
    floors = np.broadcast_to(np.asarray(min_std, dtype=np.float64), (Theta.shape[1],))
    return np.stack([global_param_update(Theta[:, j], Y, d, r_std, rtps, float(floors[j]))
                     for j in range(Theta.shape[1])], axis=1)


def global_param_update(theta: np.ndarray, Y: np.ndarray, d: np.ndarray, r_std: float,
                        rtps: float = 0.0, min_std: float = 0.0) -> np.ndarray:
    """`theta`: (N,) prior parameter values; `Y`: (N, m) prior observation-space perturbations
    (H x_i - mean_i H x_i); `d`: (m,) innovation y - mean H x; `r_std`: observation error std.
    Returns the (N,) analysis parameter values."""
    theta = np.asarray(theta, dtype=np.float64)
    N = theta.shape[0]
    if Y.shape[0] != N:
        raise ValueError(f"Y has {Y.shape[0]} rows, expected N = {N}")
    tb = theta.mean()
    tp = theta - tb
    C = Y / r_std ** 2
    A = (N - 1) * np.eye(N) + C @ Y.T
    ev, V = np.linalg.eigh(A)
    if ev.min() <= 0:
        raise np.linalg.LinAlgError(f"parameter transform matrix not positive definite: min eig {ev.min():.3e}")
    Pa = (V / ev) @ V.T
    wbar = Pa @ (C @ d)
    W = (V * np.sqrt((N - 1) / ev)) @ V.T + wbar[:, None]
    theta_a = tb + W.T @ tp
    # spread safeguards
    sb = tp.std(ddof=1)
    ta_bar = theta_a.mean()
    tap = theta_a - ta_bar
    sa = tap.std(ddof=1)
    if sa > 0:
        target = sa + rtps * (sb - sa)
        target = max(target, min_std)
        tap = tap * (target / sa)
    elif min_std > 0:
        raise ValueError("analysis parameter spread is exactly zero; cannot rescale to the floor")
    return ta_bar + tap
