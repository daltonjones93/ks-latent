"""FPe: libraries, fits and closed rollouts for a sparse law of the leading latent channels.

State z (T, P, k): the first k channels of an ordered (nested-dropout) latent field on P sites, 1-min frames.
Library terms (all built from the first k channels only):
  ode   z, and quadratic products z_i z_j                              (a local ODE per site)
  pde   + d1 z, d2 z (centred differences over sites) and products z_i d z_j (no derivative x derivative)
  pde3, pde4  the same with spatial derivatives up to third / fourth order (five-point stencils, reach 2 sites):
        d3 (dispersion) and d4 (hyperdiffusion, as in Kuramoto-Sivashinsky), which can damp the shortest waves
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


def site_derivs(z: np.ndarray, order: int = 2):
    """z (..., P, k) -> centred differences over sites up to `order` (2: d1, d2; 3: + d3; 4: + d3, d4), edges
    replicated. d3 = (z[i+2] - 2 z[i+1] + 2 z[i-1] - z[i-2]) / 2, d4 = z[i+2] - 4 z[i+1] + 6 z[i] - 4 z[i-1] + z[i-2]."""
    zp = np.concatenate([z[..., :1, :], z[..., :1, :], z, z[..., -1:, :], z[..., -1:, :]], axis=-2)
    P = z.shape[-2]
    m2, m1, p1, p2 = (zp[..., i: i + P, :] for i in (0, 1, 3, 4))
    out = [0.5 * (p1 - m1), p1 - 2 * z + m1]
    if order >= 3:
        out.append(0.5 * (p2 - 2 * p1 + 2 * m1 - m2))
    if order >= 4:
        out.append(p2 - 4 * p1 + 6 * z - 4 * m1 + m2)
    return out


def _dx_order(kind: str) -> int:
    orders = {"pde": 2, "pde3": 3, "pde4": 4}
    if kind != "ode" and kind not in orders:
        raise ValueError(f"kind must be 'ode', 'pde', 'pde3' or 'pde4', got {kind!r}")
    return orders.get(kind, 0)


def library(z: np.ndarray, kind: str, v: np.ndarray | None = None):
    """Features (..., P, F) and names. `v` (same shape as z): first time derivatives (time_order 2)."""
    nd = _dx_order(kind)
    k = z.shape[-1]
    lin, names = [z], [f"z{j}" for j in range(k)]
    if v is not None:
        lin.append(v); names += [f"v{j}" for j in range(k)]
    n_state = len(names)                         # z (and v): products among these are allowed
    if nd:
        ds = site_derivs(z, nd)
        for n, d in enumerate(ds, 1):
            lin.append(d); names += [f"d{n}z{j}" for j in range(k)]
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


def torch_library(z, kind: str, v=None):
    """Torch twin of `library` (same columns, same order), differentiable; z (..., P, k)."""
    import torch
    def sd(x, order):
        xp = torch.cat([x[..., :1, :], x[..., :1, :], x, x[..., -1:, :], x[..., -1:, :]], dim=-2)
        P = x.shape[-2]
        m2, m1, p1, p2 = (xp[..., i: i + P, :] for i in (0, 1, 3, 4))
        out = [0.5 * (p1 - m1), p1 - 2 * x + m1]
        if order >= 3:
            out.append(0.5 * (p2 - 2 * p1 + 2 * m1 - m2))
        if order >= 4:
            out.append(p2 - 4 * p1 + 6 * x - 4 * m1 + m2)
        return out
    nd = _dx_order(kind)
    lin = [z] + ([v] if v is not None else [])
    n_state = sum(t.shape[-1] for t in lin)
    if nd:
        lin += sd(z, nd)
        if v is not None:
            lin.append(sd(v, 2)[1])
    L = torch.cat(lin, -1)
    cols = [torch.ones_like(z[..., :1]), L]
    quad = [L[..., a] * L[..., b] for a, b in itertools.combinations_with_replacement(range(L.shape[-1]), 2)
            if not (a >= n_state and b >= n_state)]
    if quad:
        cols.append(torch.stack(quad, -1))
    return torch.cat(cols, -1)


def refit_rollout(xi: np.ndarray, trajs: list, kind: str, halo: int, time_order: int = 1, k_max: int = 30,
                  iters: int = 1500, batch: int = 8, lr: float = 3e-3, cap: float = 5.0, seed: int = 0,
                  log=None) -> np.ndarray:
    """Keep the support of `xi` (F, k) and refit its nonzero coefficients so that closed rollouts of up to `k_max`
    steps (curriculum 2 -> k_max over the first half) match the trajectories `trajs` [(T, P, k)] on interior sites,
    edges held at the start state, as in `rollout`. Adam on the mean squared rollout error; returns the new xi."""
    import torch
    g = torch.Generator().manual_seed(seed)
    mask = torch.tensor(xi != 0)
    w = torch.nn.Parameter(torch.tensor(xi, dtype=torch.float64))        # a copy: never write into the caller's xi
    opt = torch.optim.Adam([w], lr=lr)
    Z = [torch.as_tensor(t, dtype=torch.float64) for t in trajs]
    capf = (lambda f: cap * torch.tanh(f / cap)) if cap else (lambda f: f)
    sl = slice(halo, -halo) if halo else slice(None)
    last = (w * mask).detach().clone()
    for it in range(iters):
        K = int(round(2 + (k_max - 2) * min(1.0, it / (0.5 * iters))))
        idx = torch.randint(len(Z), (batch,), generator=g)
        z0s, tgt, v0s = [], [], []
        for i in idx.tolist():
            t0 = int(torch.randint(2, Z[i].shape[0] - K, (1,), generator=g))
            z0s.append(Z[i][t0]); tgt.append(Z[i][t0 + 1: t0 + 1 + K])
            v0s.append(0.5 * (3 * Z[i][t0] - 4 * Z[i][t0 - 1] + Z[i][t0 - 2]))
        z0, tgt, v0 = torch.stack(z0s), torch.stack(tgt, 1), torch.stack(v0s)     # (B, P, k), (K, B, P, k)
        W = w * mask
        z, v, loss = z0, (v0 if time_order == 2 else None), 0.0
        for s in range(K):
            if time_order == 1:
                f = lambda z_, v_: (capf(torch_library(z_, kind) @ W), None)
            else:
                f = lambda z_, v_: (v_, capf(torch_library(z_, kind, v_) @ W))
            a1, b1 = f(z, v)
            a2, b2 = f(z + a1, None if v is None else v + b1)
            zn = z + 0.5 * (a1 + a2)
            if v is not None:
                v = v + 0.5 * (b1 + b2)
            if halo:
                zn = torch.cat([z0[:, :halo], zn[:, halo:-halo], z0[:, -halo:]], 1)
            z = zn
            loss = loss + ((z[:, sl] - tgt[s][:, sl]) ** 2).mean()
        loss = loss / K
        if not torch.isfinite(loss):
            if log:
                log(f"      refit: non-finite loss at iteration {it} (K {K}); keeping the last finite coefficients")
            break
        last = (w * mask).detach().clone()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
        if log and (it + 1) % max(1, iters // 5) == 0:
            log(f"      refit iteration {it + 1}/{iters}: K {K}, rollout MSE {float(loss):.4f}")
    out = (w * mask).detach()
    return (out if torch.isfinite(out).all() else last).numpy()


__all__ = ["torch_library", "refit_rollout", "backward_velocity", "site_derivs", "library", "smooth_t", "training_rows", "fit", "rollout", "r2"]
