"""End-to-end validation: does SEC-localizing a small-ensemble-estimated
prior covariance actually improve the resulting DA analysis (brief §9),
not just shrink correlations in isolation (that's `test_sec.py`)?
"""

from __future__ import annotations

import torch
import pytest

from ks_latent.da.pff import ParticleFlowFilter, PFFConfig
from ks_latent.da.sec import build_sec_table, sec_localize_covariance


def _kalman_posterior_cov(H, R, B):
    Binv = torch.linalg.inv(B)
    Rinv = torch.linalg.inv(R)
    F = H.T @ Rinv @ H + Binv
    return torch.linalg.inv(F)


@pytest.mark.slow
def test_sec_localized_prior_gives_analysis_closer_to_true_kalman_answer():
    torch.manual_seed(0)
    d, m = 6, 3
    n_ens = 12  # small, realistic DA ensemble size

    # True prior: banded (only adjacent-index correlation), unit variance.
    # rho=0.3 keeps this tridiagonal Toeplitz matrix safely positive
    # definite for any d (eigenvalues 1 + 2*rho*cos(k*pi/(d+1)) > 0).
    B_true = torch.eye(d, dtype=torch.float64)
    for i in range(d - 1):
        B_true[i, i + 1] = B_true[i + 1, i] = 0.3

    # A single small-ensemble draw from the true prior gives a noisy sample
    # covariance with spurious off-diagonal structure -- the realistic
    # small-N estimation problem SEC is meant to fix.
    L_true = torch.linalg.cholesky(B_true)
    small_ensemble = torch.randn(n_ens, d, dtype=torch.float64) @ L_true.T
    B_raw = torch.cov(small_ensemble.T)

    table = build_sec_table(n_ens=n_ens, n_trials=4000, seed=1)
    B_sec = sec_localize_covariance(B_raw, table)

    H = torch.randn(m, d, dtype=torch.float64) * 0.5
    R = torch.eye(m, dtype=torch.float64) * 0.1
    y = torch.tensor([0.5, -0.3, 0.2], dtype=torch.float64)
    kalman_cov_true_prior = _kalman_posterior_cov(H, R, B_true)

    def h(z):
        return H @ z

    def analysis_cov_from_prior(B_prior, n_particles=8000):
        # A large proxy ensemble representing "belief = B_prior exactly",
        # isolating the prior's quality from PFF's own finite-N noise.
        L = torch.linalg.cholesky(B_prior)
        Z0 = torch.randn(n_particles, d, dtype=torch.float64) @ L.T
        pff = ParticleFlowFilter(h, R, config=PFFConfig(method="NAT", n_steps=100))
        result = pff.analyze(Z0, y)
        return torch.cov(result.Z_analysis.T)

    cov_from_raw = analysis_cov_from_prior(B_raw)
    cov_from_sec = analysis_cov_from_prior(B_sec)

    err_raw = torch.linalg.norm(cov_from_raw - kalman_cov_true_prior)
    err_sec = torch.linalg.norm(cov_from_sec - kalman_cov_true_prior)
    assert err_sec < err_raw, (
        f"SEC-localized prior should give an analysis closer to the true-prior "
        f"Kalman answer: err_sec={err_sec:.4f}, err_raw={err_raw:.4f}"
    )
