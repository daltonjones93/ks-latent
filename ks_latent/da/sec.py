"""Distance-free localization on the existing latent (brief §9, addendum
§13.3).

Two methods, both estimated offline and applied without any distance
function on the latent (which has none):

1. **Sampling error correction (SEC)** (Anderson 2012, "Localization and
   sampling error correction in ensemble Kalman filter applications",
   *Mon. Wea. Rev.* 140:2359): the sample correlation `r` between two
   variables, measured from an `N`-member ensemble, is a *noisy* estimate
   of the true correlation `rho` -- observing a large `|r|` at small `N` is
   more often the result of an unlucky finite sample from a *smaller*
   `rho` than of a lucky sample from a `rho` as large as `r` itself (this
   is a real "regression to the mean" effect under a flat prior on `rho`,
   not a bias-correction of `E[r]`). SEC estimates, once per ensemble size
   `N` via Monte Carlo, the mapping `r -> E[rho | r observed]`, and applies
   *that* corrected correlation in place of the raw one when building the
   prior covariance for the filter -- shrinking exactly the correlations
   that are least trustworthy at that `N`, with no notion of physical
   distance anywhere.
2. **Fixed empirical taper**: a simpler alternative -- estimate the
   correlation matrix once from a *large* reference archive (so it's an
   accurate, low-variance estimate) and use its magnitude directly as a
   fixed per-pair trust weight, independent of the live ensemble's size.
   Simpler, but (unlike SEC) fitted once at one implicit "ensemble size"
   and one training distribution -- flagged as the reason it can't give
   L-transfer (brief §9, Phase 10's motivation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class SECTable:
    n_ens: int
    sample_corr_bins: np.ndarray  # bin centers, sorted ascending
    expected_true_corr: np.ndarray  # E[rho | r in this bin], same shape


def build_sec_table(
    n_ens: int,
    n_true_grid: int = 41,
    n_trials: int = 2000,
    n_bins: int = 100,
    seed: int = 0,
) -> SECTable:
    """Monte Carlo SEC table for ensemble size `n_ens` (Anderson 2012).

    For each of `n_true_grid` true correlations `rho_true` in `[-0.98,
    0.98]`, draws `n_trials` independent `n_ens`-sized bivariate-Gaussian
    samples with that population correlation and records the *sample*
    correlation of each. Pooling all `(rho_true, sample_corr)` pairs across
    every `rho_true` and binning by the observed `sample_corr` gives the
    empirical `E[rho_true | sample_corr in bin]` -- the correction to apply
    to a newly observed sample correlation at this ensemble size.
    """
    rng = np.random.default_rng(seed)
    true_grid = np.linspace(-0.98, 0.98, n_true_grid)

    all_true = np.empty(n_true_grid * n_trials)
    all_sample = np.empty(n_true_grid * n_trials)
    for i, rho in enumerate(true_grid):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        samples = rng.multivariate_normal(np.zeros(2), cov, size=(n_trials, n_ens))
        x, y = samples[..., 0], samples[..., 1]
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        denom = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
        r = np.divide((xc * yc).sum(axis=1), denom, out=np.zeros(n_trials), where=denom > 0)
        all_true[i * n_trials : (i + 1) * n_trials] = rho
        all_sample[i * n_trials : (i + 1) * n_trials] = r

    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(all_sample, bin_edges) - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    expected_true = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        expected_true[b] = all_true[mask].mean() if mask.any() else bin_centers[b]

    return SECTable(n_ens=n_ens, sample_corr_bins=bin_centers, expected_true_corr=expected_true)


def apply_sec_correction(sample_corr: np.ndarray, table: SECTable) -> np.ndarray:
    """Look up (via linear interpolation) the SEC-corrected correlation for
    each entry of `sample_corr`."""
    return np.interp(sample_corr, table.sample_corr_bins, table.expected_true_corr)


def _covariance_to_correlation(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    std = torch.sqrt(torch.diagonal(B).clamp_min(1e-300))
    corr = B / (std.unsqueeze(0) * std.unsqueeze(1))
    return corr, std


def sec_localize_covariance(B: torch.Tensor, table: SECTable) -> torch.Tensor:
    """Replace `B`'s implied sample correlations with their SEC-corrected
    values, keeping the sample variances (diagonal) unchanged."""
    corr, std = _covariance_to_correlation(B)
    corr_np = corr.detach().cpu().numpy()
    corrected = apply_sec_correction(corr_np, table)
    corrected_t = torch.as_tensor(corrected, dtype=B.dtype, device=B.device)
    corrected_t.fill_diagonal_(1.0)
    return corrected_t * (std.unsqueeze(0) * std.unsqueeze(1))


def fixed_empirical_taper(archive: torch.Tensor) -> torch.Tensor:
    """A fixed `(d, d)` taper from a *large* reference archive of latent
    points: the archive's own correlation-matrix magnitude, used directly
    as a per-pair trust weight (`1.0` on the diagonal). Unlike SEC, this
    does not depend on the live ensemble size and is not recomputed per
    analysis step -- estimate once, reuse always."""
    archive = archive.to(torch.float64)
    B_archive = torch.cov(archive.T)
    corr, _ = _covariance_to_correlation(B_archive)
    taper = corr.abs()
    taper.fill_diagonal_(1.0)
    return taper


def apply_fixed_taper(B: torch.Tensor, taper: torch.Tensor) -> torch.Tensor:
    """Schur (Hadamard) product localization: `B_localized = B * taper`."""
    return B * taper.to(dtype=B.dtype, device=B.device)


def make_sec_localizer(table: SECTable) -> Callable[[torch.Tensor], torch.Tensor]:
    """`ParticleFlowFilter(..., localize_fn=make_sec_localizer(table))` --
    curries the (ensemble-size-specific) SEC table into the
    `Callable[[B], B]` interface `ParticleFlowFilter`/`CycleConfig` expect."""
    return lambda B: sec_localize_covariance(B, table)


def make_fixed_taper_localizer(taper: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Same currying for the fixed empirical taper."""
    return lambda B: apply_fixed_taper(B, taper)
