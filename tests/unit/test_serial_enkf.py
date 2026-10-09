"""Task D3: serial SEC EnKF and the PSD-projected SEC covariance."""

from __future__ import annotations

import numpy as np
import torch

from ks_latent.da.sec import SECTable, build_sec_table, sec_localize_covariance, sec_localize_covariance_psd
from ks_latent.da.serial_enkf import SerialSECEnKF


def _identity_table(n_ens):
    g = np.linspace(-1.0, 1.0, 401)
    return SECTable(n_ens=n_ens, sample_corr_bins=g, expected_true_corr=g.copy())


def _linear_problem(d=4, m=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(m, d, generator=g, dtype=torch.float64)
    R_diag = torch.full((m,), 0.2, dtype=torch.float64)
    y = torch.randn(m, generator=g, dtype=torch.float64)
    return H, R_diag, y


def test_serial_enkf_identity_table_equals_plain_filter():
    H, R_diag, y = _linear_problem()
    Z0 = torch.randn(30, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    a = SerialSECEnKF(lambda z: H @ z, R_diag, table=None, seed=3).analyze(Z0, y).Z_analysis
    b = SerialSECEnKF(lambda z: H @ z, R_diag, table=_identity_table(30), seed=3).analyze(Z0, y).Z_analysis
    assert torch.allclose(a, b, atol=1e-10)


def test_serial_enkf_matches_kalman_posterior_at_large_n():
    H, R_diag, y = _linear_problem()
    d = H.shape[1]
    N = 20000
    Z0 = 0.3 + torch.randn(N, d, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    out = SerialSECEnKF(lambda z: H @ z, R_diag, seed=0).analyze(Z0, y).Z_analysis
    B, R = torch.eye(d, dtype=torch.float64), torch.diag(R_diag)
    K = B @ H.T @ torch.linalg.inv(H @ B @ H.T + R)
    mean_k = 0.3 * torch.ones(d, dtype=torch.float64)
    mean_k = mean_k + K @ (y - H @ mean_k)
    cov_k = B - K @ H @ B
    assert float((out.mean(0) - mean_k).norm() / mean_k.norm()) < 0.03
    assert float((torch.cov(out.T) - cov_k).norm() / cov_k.norm()) < 0.05


def test_serial_sec_shrinks_update_of_uncorrelated_variable():
    """A latent coordinate unrelated to the observation gets a smaller spurious increment with
    SEC than without, at small N (the purpose of the correction)."""
    N, trials = 10, 200
    table = build_sec_table(n_ens=N, n_trials=2000, seed=0)
    R_diag = torch.tensor([0.1], dtype=torch.float64)
    plain, sec = [], []
    for t in range(trials):
        Z0 = torch.randn(N, 2, generator=torch.Generator().manual_seed(100 + t), dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)
        h = lambda z: z[:1]  # noqa: E731 -- observes coordinate 0 only; coordinate 1 is independent
        za = SerialSECEnKF(h, R_diag, table=None, seed=t).analyze(Z0, y).Z_analysis
        zs = SerialSECEnKF(h, R_diag, table=table, seed=t).analyze(Z0, y).Z_analysis
        plain.append(float((za[:, 1].mean() - Z0[:, 1].mean()).abs()))
        sec.append(float((zs[:, 1].mean() - Z0[:, 1].mean()).abs()))
    assert np.mean(sec) < 0.8 * np.mean(plain)


def test_sec_psd_projection_is_psd_and_identity_on_psd_input():
    torch.manual_seed(0)
    X = torch.randn(8, 30, dtype=torch.float64)  # N = 8 < d = 30: rank-deficient sample covariance
    B = torch.cov(X.T) + 1e-3 * torch.eye(30, dtype=torch.float64)
    table = build_sec_table(n_ens=8, n_trials=2000, seed=0)
    psd = sec_localize_covariance_psd(B, table)
    assert float(torch.linalg.eigvalsh(psd).min()) > 0
    ident = sec_localize_covariance_psd(B, _identity_table(8))
    assert torch.allclose(ident, B, atol=1e-10)
    # An elementwise map of a valid correlation matrix can be indefinite; the projection must
    # repair it. A map that inflates |r| makes this 3x3 matrix indefinite.
    C = torch.tensor([[1.0, 0.6, 0.6], [0.6, 1.0, 0.0], [0.6, 0.0, 1.0]], dtype=torch.float64)
    assert float(torch.linalg.eigvalsh(C).min()) > 0
    g = np.linspace(-1.0, 1.0, 401)
    inflate = SECTable(n_ens=8, sample_corr_bins=g, expected_true_corr=np.clip(1.4 * g, -1, 1))
    raw = sec_localize_covariance(C, inflate)
    assert float(torch.linalg.eigvalsh(raw).min()) < 0
    assert float(torch.linalg.eigvalsh(sec_localize_covariance_psd(C, inflate)).min()) > 0
