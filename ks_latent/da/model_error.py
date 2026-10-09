"""Latent model-error statistics estimated from data (task I22, 2026-10-09).

For a local latent field the model error of one DA cycle,

    r = E(u(t + T)) - M^n(E(u(t))),      T = n * dt_snap,

is a field of c-vectors, one per site. If the dynamics are statistically homogeneous (2D KS on
a periodic box; the crop-trained models are translation-equivariant by whole sites), every site
has the same distribution, so its mean b (bias) and c x c covariance Q can be estimated by pooling
residuals over sites and times. Cross-site covariance is measured too, to show how far a
block-diagonal (per-site independent) model error is justified.

Q replaces the isotropic latent noise (std 0.02 per propagator step) as the ensemble's model-error
term: after each forecast, every member gets xi_s ~ N(0, scale * Q) at every site s,
independently across sites. This is standard additive model error with Q estimated offline (e.g.
the "additive inflation from model-error samples" family); locality makes Q a per-site, domain-
size-independent statistic. The perfect-model filter (task I11) needs no such term; the latent one
is under-dispersed (spread/RMSE ~0.3) because its model error is not represented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SiteModelError:
    """Per-site model-error statistics. `bias`: (c,); `Q`: (c, c), SPD; `neighbour_corr`: mean
    absolute correlation between a site's residual and its 4 nearest neighbours' (same channel),
    a check on the block-diagonal assumption; `n_samples`: pooled samples."""
    bias: np.ndarray
    Q: np.ndarray
    neighbour_corr: float
    n_samples: int


def estimate_site_model_error(R: np.ndarray, n_sites_x: int, n_sites_y: int, channels: int) -> SiteModelError:
    """`R`: (T, n_sites_x * n_sites_y * channels) residuals, site-major (site index = ix * n_sites_y
    + iy, channels contiguous within a site, as the per-patch autoencoder lays out its latent).
    Pools over time and sites."""
    T = R.shape[0]
    Rs = R.reshape(T, n_sites_x, n_sites_y, channels)
    pooled = Rs.reshape(-1, channels)
    bias = pooled.mean(0)
    X = pooled - bias
    Q = X.T @ X / (X.shape[0] - 1)
    ev = np.linalg.eigvalsh(Q)
    if ev.min() <= 0:
        raise np.linalg.LinAlgError(f"estimated site Q not positive definite: eigenvalues {ev}")
    C = Rs - bias
    sd = C.reshape(-1, channels).std(0)
    corrs = []
    for ax in (1, 2):
        nb = np.roll(C, 1, axis=ax)
        corrs.append(np.abs((C * nb).reshape(-1, channels).mean(0) / sd ** 2))
    return SiteModelError(bias=bias, Q=Q, neighbour_corr=float(np.mean(corrs)), n_samples=X.shape[0])


def sample_site_noise(n_members: int, n_sites: int, Q: np.ndarray, scale: float,
                      generator: torch.Generator) -> torch.Tensor:
    """(n_members, n_sites * c) draws, independent across members and sites, each site ~ N(0,
    scale * Q). Float32, ready to add to a latent ensemble."""
    c = Q.shape[0]
    L = torch.as_tensor(np.linalg.cholesky(scale * Q), dtype=torch.float32)
    xi = torch.randn(n_members, n_sites, c, generator=generator)
    return (xi @ L.T).reshape(n_members, n_sites * c)
