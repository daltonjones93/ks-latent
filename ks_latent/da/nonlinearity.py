"""Diagnostics of the nonlinearity of the latent observation operator (Z20 E1, Paper II).

The latent filter observes through `H = H_obs o D`, nonlinear in `z`, and linearizes its gain at
the ensemble mean (`ks_latent.da.letkf_local`). These functions measure how far that linearization
is from the nonlinear map on a given ensemble, and whether the ensemble is consistent with the
observations:

* `jdelta_compact` -- the linearized observation perturbations `J (z^(e) - z_bar)`, computed from
  the compact block-diagonal Jacobian (`jacobian_coloring.coloured_obs_jacobian_2d_compact`).
* `linearization_stats` -- with `r^(e) = H(z^(e)) - H(z_bar) - J delta^(e)`:
  `eta   = rms(r) / rms(J delta)` (relative size of the neglected terms; second order, so it grows
  linearly with the spread) and `eta_o = rms(r) / sigma_o` (the same against the observation error;
  the linearization is adequate when `eta_o << 1`), plus per-site distributions of `eta_o`.
* `innovation_chi2_diag` -- `sum_i d_i^2 / sum_i (var_e H_i (N+1)/N + sigma_o^2)` with
  `d = y - mean_e H(z^(e))`; 1 for a consistent ensemble (ratio of sums, which avoids the bias of
  dividing by a sample variance per observation; correlations between observations do not enter the
  expectation).
* `rank_histogram` -- rank of each observation among the ensemble's predicted observations, each
  perturbed by its own observation-error draw (Hamill 2001); flat for a consistent ensemble.
"""
from __future__ import annotations

import numpy as np
import torch


def jdelta_compact(Z_pert: torch.Tensor, J_compact: torch.Tensor, site_of_obs: torch.Tensor,
                   channels: int) -> torch.Tensor:
    """`(N, m)` linearized observation perturbations: observation `i` reads only the `channels`
    coordinates of site `site_of_obs[i]` (block-diagonal Jacobian), so `(J delta)_i = delta_block(i) . J_i`."""
    cols = site_of_obs.unsqueeze(1) * channels + torch.arange(channels).unsqueeze(0)  # (m, c)
    return (Z_pert[:, cols] * J_compact.unsqueeze(0)).sum(-1)


def linearization_stats(Hz: torch.Tensor, h_at_mean: torch.Tensor, Jdelta: torch.Tensor, sigma_o: float,
                        site_of_obs: torch.Tensor, n_sites: int) -> dict:
    """See module docstring. `Hz`: `(N, m)` nonlinear predicted observations of the members;
    `h_at_mean`: `(m,)` = H(z_bar); `Jdelta`: `(N, m)`."""
    r = Hz - h_at_mean.unsqueeze(0) - Jdelta  # (N, m)
    if not torch.isfinite(r).all():
        raise FloatingPointError("non-finite linearization residual")
    rms_r = float(r.pow(2).mean().sqrt())
    rms_j = float(Jdelta.pow(2).mean().sqrt())
    if rms_j == 0.0:
        raise ValueError("zero ensemble perturbation in observation space: eta undefined")
    # per site: rms over members and the site's observations
    ms = torch.zeros(n_sites, dtype=r.dtype).index_add_(0, site_of_obs, r.pow(2).mean(0))
    cnt = torch.zeros(n_sites, dtype=r.dtype).index_add_(0, site_of_obs, torch.ones_like(site_of_obs, dtype=r.dtype))
    occ = cnt > 0
    eta_o_site = (ms[occ] / cnt[occ]).sqrt() / sigma_o
    # mean residual = the part of the ensemble-mean prediction the linearization misses (Jensen gap
    # in observation space): mean_e H(z^(e)) - H(z_bar), against sigma_o
    jensen = float((Hz.mean(0) - h_at_mean).pow(2).mean().sqrt())
    return {
        "eta": rms_r / rms_j,
        "eta_o": rms_r / sigma_o,
        "eta_o_site_median": float(eta_o_site.median()),
        "eta_o_site_p90": float(torch.quantile(eta_o_site, 0.9)),
        "eta_o_site_max": float(eta_o_site.max()),
        "jensen_obs_over_sigma": jensen / sigma_o,
        "spread_obs_over_sigma": float(Hz.std(0).pow(2).mean().sqrt()) / sigma_o,
    }


def innovation_chi2_diag(y: torch.Tensor, Hz: torch.Tensor, sigma_o: float) -> float:
    """`sum_i (y_i - mean_e Hz_i)^2 / sum_i (var_e Hz_i (N+1)/N + sigma_o^2)`; expectation close to 1
    when the truth is a draw from the ensemble's predictive distribution and the observation errors
    are as assumed. Above 1: the ensemble is underdispersed (or biased); below 1: overdispersed."""
    N = Hz.shape[0]
    d = y - Hz.mean(0)
    v = Hz.var(0, unbiased=True) * (N + 1) / N + sigma_o ** 2
    return float(d.pow(2).sum() / v.sum())


def rank_histogram(y: np.ndarray, Hz: np.ndarray, sigma_o: float, rng: np.random.Generator) -> np.ndarray:
    """Counts `(N+1,)` of the rank of each observation `y_i` among `Hz[:, i] + eps` (`eps ~ N(0, sigma_o^2)`
    drawn per member and observation, so that a consistent ensemble gives a flat histogram)."""
    N = Hz.shape[0]
    pred = Hz + sigma_o * rng.standard_normal(Hz.shape)
    ranks = (pred < y[None, :]).sum(0)
    return np.bincount(ranks, minlength=N + 1)
