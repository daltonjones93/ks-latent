"""FPe: libraries, fits and closed rollouts for a sparse law of the leading latent channels.

State z (T, P, k): the first k channels of an ordered (nested-dropout) latent field on P sites, 1-min frames.
Library terms (all built from the first k channels only):
  ode   z, and quadratic products z_i z_j                              (a local ODE per site)
  pde   + d1 z, d2 z (centred differences over sites) and products z_i d z_j (no derivative x derivative)
With time_order = 2 the first time derivatives v = dz/dt are added as library variables (v, and products z_i v_j,
v_i v_j, plus d2 v for "pde"), and the target becomes d2z/dt2: the law is second order in time,
  d2z/dt2 = F(z, dz/dt, d1z, d2z, ...),
which can represent waves (d2z/dt2 = c^2 d2z/dx2, e.g. gravity waves) and carries memory of the dropped channels
(dz/dt of the kept channels is extra state, like a delay embedding; Bakarji et al. 2023; Gilani et al. 2021).
Rollouts integrate (z, v) with Heun's method; v at the start is a second-order BACKWARD difference
(`backward_velocity`; no future information).
Time derivatives for fitting are centred differences, optionally after Gaussian smoothing in time (`t_smooth`
frames), since second differences of 1-min latents are noisy.
"""
from __future__ import annotations

import itertools

import numpy as np

from ks_latent.cm1.fieldpde import bootstrap_stlsq, r2


def site_derivs(z: np.ndarray):
    """z (..., P, k) -> centred first and second differences over sites (edges replicated)."""
    zp = np.concatenate([z[..., :1, :], z, z[..., -1:, :]], axis=-2)
    return 0.5 * (zp[..., 2:, :] - zp[..., :-2, :]), zp[..., 2:, :] - 2 * z + zp[..., :-2, :]


def library(z: np.ndarray, kind: str, v: np.ndarray | None = None):
    """Features (..., P, F) and names. `v` (same shape as z): first time derivatives (time_order 2)."""
    if kind not in ("ode", "pde"):
        raise ValueError(f"kind must be 'ode' or 'pde', got {kind!r}")
    k = z.shape[-1]
    lin, names = [z], [f"z{j}" for j in range(k)]
    if v is not None:
        lin.append(v); names += [f"v{j}" for j in range(k)]
    n_state = len(names)                         # z (and v): products among these are allowed
    if kind == "pde":
        d1, d2 = site_derivs(z)
        lin += [d1, d2]; names += [f"d1z{j}" for j in range(k)] + [f"d2z{j}" for j in range(k)]
        if v is not None:
            lin.append(site_derivs(v)[1]); names += [f"d2v{j}" for j in range(k)]
    L = np.concatenate(lin, -1)
    quad, qn = [], []
    for a, b in itertools.combinations_with_replacement(range(L.shape[-1]), 2):
        if a >= n_state and b >= n_state:        # no products of two spatial derivatives
            continue
        quad.append(L[..., a] * L[..., b]); qn.append(f"{names[a]}*{names[b]}")
    F = np.concatenate([np.ones(z.shape[:-1] + (1,)), L] + ([np.stack(quad, -1)] if quad else []), -1)
    return F, ["1"] + names + qn


def smooth_t(z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (frames), edges reflected; sigma 0 returns z."""
    if sigma <= 0:
        return z
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); w /= w.sum()
    zp = np.concatenate([z[r:0:-1], z, z[-2:-r - 2:-1]], 0)
    return sum(w[i] * zp[i: i + len(z)] for i in range(2 * r + 1))


def training_rows(z: np.ndarray, kind: str, time_order: int, halo: int, t_smooth: float = 0.0):
    """z (T, P, k) -> library rows X (n, F), targets Y (n, k) and names, interior sites, frames 1..T-2."""
    zs = smooth_t(z, t_smooth)
    v = 0.5 * (zs[2:] - zs[:-2])                                  # dz/dt at frames 1..T-2
    if time_order == 1:
        F, names = library(zs[1:-1], kind)
        Y = v
    elif time_order == 2:
        F, names = library(zs[1:-1], kind, v)
        Y = zs[2:] - 2 * zs[1:-1] + zs[:-2]                        # d2z/dt2
    else:
        raise ValueError("time_order must be 1 or 2")
    sl = slice(halo, -halo) if halo else slice(None)
    return F[:, sl].reshape(-1, F.shape[-1]), Y[:, sl].reshape(-1, z.shape[-1]), names


def fit(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, threshold: float, n_boot: int = 10):
    """Bootstrap STLSQ per target column on centred data; returns coefficients (F, k) incl. the intercept (row 0,
    which must be the constant column) and selection frequencies."""
    if not np.all(X[:, 0] == 1):
        raise ValueError("column 0 of the library must be the constant")
    xi, freq = np.zeros((X.shape[1], Y.shape[1])), np.zeros((X.shape[1], Y.shape[1]))
    mx = X[:, 1:].mean(0)
    for j in range(Y.shape[1]):
        b = bootstrap_stlsq(X[:, 1:] - mx, Y[:, j] - Y[:, j].mean(), groups, threshold, n_boot=n_boot)
        xi[1:, j], freq[1:, j] = b["full"], b["freq"]
        xi[0, j], freq[0, j] = Y[:, j].mean() - mx @ b["full"], 1.0
    return xi, freq


def backward_velocity(z: np.ndarray, t0: int) -> np.ndarray:
    """dz/dt at frame t0 from frames t0-2..t0 only (second-order backward difference; no future information)."""
    if t0 < 2:
        raise ValueError("need two frames before t0")
    return 0.5 * (3 * z[t0] - 4 * z[t0 - 1] + z[t0 - 2])


def rollout(xi: np.ndarray, z0: np.ndarray, steps: int, kind: str, cap: float, halo: int,
            time_order: int = 1, v0: np.ndarray | None = None) -> np.ndarray:
    """Closed rollout of the fitted law from z0 (P, k) (and v0 for time_order 2), edge sites held; Heun; the
    (second) time derivative is capped as cap * tanh(f / cap). Returns (steps, P, k); nan after a blow-up."""
    capf = (lambda f: cap * np.tanh(f / cap)) if cap else (lambda f: f)
    if time_order == 1:
        rhs = lambda z, v: (capf(library(z, kind)[0] @ xi), None)
    else:
        if v0 is None:
            raise ValueError("time_order 2 needs v0")
        rhs = lambda z, v: (v, capf(library(z, kind, v)[0] @ xi))
    z, v = z0.copy(), (None if v0 is None else v0.copy())
    out = []
    for _ in range(steps):
        a1, b1 = rhs(z, v)
        z1 = z + a1
        v1 = None if v is None else v + b1
        a2, b2 = rhs(z1, v1)
        z = z + 0.5 * (a1 + a2)
        if v is not None:
            v = v + 0.5 * (b1 + b2)
            if halo:
                v[:halo], v[-halo:] = 0.0, 0.0
        if halo:
            z[:halo], z[-halo:] = z0[:halo], z0[-halo:]
        if not np.isfinite(z).all():
            break
        out.append(z.copy())
    out += [np.full_like(z0, np.nan)] * (steps - len(out))
    return np.stack(out)


__all__ = ["backward_velocity", "site_derivs", "library", "smooth_t", "training_rows", "fit", "rollout", "r2"]
