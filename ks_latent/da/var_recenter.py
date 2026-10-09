"""Variational recentring of a latent ensemble analysis (accuracy study, 2026-10-08).

After the ensemble analysis, the ensemble mean is replaced by the minimizer of

    J(z) = 1/2 sum_i (y_i - H(D(z))_i)^2 / r^2                 observation fit
         + gamma/2 * sum_k (z_k - zf_k)^2 / s_k^2              forecast background (diagonal B)
         + beta/2  * sum_x (Lap D(z))(x)^2                     curvature of the decoded field

with zf the forecast ensemble mean, s_k the forecast ensemble spread of coordinate k (floored),
Lap the periodic 5-point Laplacian on the grid (grid units), r the observation-error std the
filter assumes. The curvature term is the prior behind spline interpolation; it also penalizes
the decoder's patch-edge seams. The ensemble perturbations of the ensemble analysis are kept and
recentred on the minimizer (as in hybrid ensemble-variational systems). Every term is local in
space, so a gradient evaluation costs one decode and one backward pass at any domain size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class VarRecenterConfig:
    beta: float = 1.0          # curvature weight
    gamma: float = 1.0         # background weight (1 = Bayesian with the forecast spread as B)
    n_iter: int = 30           # L-BFGS iterations
    spread_floor: float = 1e-3
    grid_shape: tuple[int, int] | None = None  # (NX, NY) of the decoded field
    # Background covariance: "diag" (forecast spread per coordinate) or "ens_loc" (forecast
    # ensemble covariance Schur-multiplied by `taper` (d, d), plus jitter * diag). "ens_loc" is
    # dense (d x d): a proof of concept at small d, not the scalable form.
    cov: str = "diag"
    taper: torch.Tensor | None = None
    jitter: float = 1e-2


def laplacian_periodic(u: torch.Tensor) -> torch.Tensor:
    """u: (NX, NY). Periodic 5-point Laplacian in grid units."""
    return (torch.roll(u, 1, 0) + torch.roll(u, -1, 0) + torch.roll(u, 1, 1) + torch.roll(u, -1, 1) - 4 * u)


def var_recenter_mean(
    decoder: Callable[[torch.Tensor], torch.Tensor],
    obs_operator: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    r_std: float,
    z_start: torch.Tensor,
    z_forecast_ens: torch.Tensor,
    cfg: VarRecenterConfig,
) -> torch.Tensor:
    """Return the minimizer of J (see module docstring), starting from z_start (d,)."""
    if cfg.grid_shape is None:
        raise ValueError("VarRecenterConfig.grid_shape is required")
    dtype = z_start.dtype
    zf = z_forecast_ens.mean(0).detach()
    s2 = z_forecast_ens.std(0).detach().clamp_min(cfg.spread_floor) ** 2
    Lb = None
    if cfg.cov == "ens_loc":
        Xp = (z_forecast_ens - zf).detach().to(torch.float64)
        B = (Xp.T @ Xp) / (Xp.shape[0] - 1) * cfg.taper.to(torch.float64)
        B = B + cfg.jitter * torch.diag(s2.to(torch.float64))
        Lb = torch.linalg.cholesky(B).to(dtype)
    y = y.to(dtype)
    z = z_start.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=cfg.n_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = decoder(z.unsqueeze(0))[0]
        j_obs = 0.5 * ((y - obs_operator(u)) ** 2).sum() / r_std**2
        if Lb is None:
            j_b = 0.5 * cfg.gamma * ((z - zf) ** 2 / s2).sum()
        else:
            v = torch.linalg.solve_triangular(Lb, (z - zf).unsqueeze(1), upper=False)
            j_b = 0.5 * cfg.gamma * (v ** 2).sum()
        j_c = 0.5 * cfg.beta * (laplacian_periodic(u.reshape(cfg.grid_shape)) ** 2).sum()
        J = j_obs + j_b + j_c
        J.backward()
        return J

    opt.step(closure)
    return z.detach()
