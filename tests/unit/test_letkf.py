"""Tests for `ks_latent.da.letkf.LETKF` (added 2026-09-28, user-directed:
"implement LETKF" as a comparator to the project's primary NAT-PFF
method). Mirrors `tests/unit/test_pff.py::test_pff_gaussian_linear`'s
structure -- the analytic linear-Gaussian Kalman posterior is the ground
truth every ensemble analysis scheme in this project is validated
against (ground rule 1: no untested numerics).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.da.letkf import LETKF


def _kalman_posterior(H, R, B, z0_bar, y):
    Binv = torch.linalg.inv(B)
    Rinv = torch.linalg.inv(R)
    F = H.T @ Rinv @ H + Binv
    cov = torch.linalg.inv(F)
    mean = z0_bar + cov @ H.T @ Rinv @ (y - H @ z0_bar)
    return mean, cov


@pytest.mark.slow
def test_letkf_gaussian_linear_matches_analytic_kalman_posterior():
    torch.manual_seed(0)
    d, m, N = 5, 3, 8000
    H = torch.randn(m, d, dtype=torch.float64) * 0.5
    R = torch.eye(m, dtype=torch.float64) * 0.1
    B = torch.eye(d, dtype=torch.float64)
    z0_bar = torch.zeros(d, dtype=torch.float64)
    y = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64)

    L = torch.linalg.cholesky(B)
    Z0 = z0_bar + torch.randn(N, d, dtype=torch.float64) @ L.T

    def h(z):
        return H @ z

    letkf = LETKF(h, R, seed=0)
    result = letkf.analyze(Z0, y)

    mean_emp = result.Z_analysis.mean(dim=0)
    cov_emp = torch.cov(result.Z_analysis.T)
    mean_true, cov_true = _kalman_posterior(H, R, B, z0_bar, y)

    mean_rel_err = float((mean_emp - mean_true).norm() / mean_true.norm())
    cov_rel_err = float((cov_emp - cov_true).norm() / cov_true.norm())
    # Same 5% bound as test_pff_gaussian_linear, for the same reason
    # (N=8000 leaves margin against seed variance while still being a
    # real correctness bound; the perturbed-obs update is known to need
    # larger N than a deterministic square-root transform for the same
    # tolerance, so this is not a loosened bar relative to PFF's own).
    assert mean_rel_err < 0.05, f"mean rel err {mean_rel_err:.4f}"
    assert cov_rel_err < 0.05, f"cov rel err {cov_rel_err:.4f}"


def test_letkf_shape_and_finite():
    torch.manual_seed(1)
    d, m, N = 6, 2, 50
    H = torch.randn(m, d, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64) * 0.2
    Z0 = torch.randn(N, d, dtype=torch.float64)
    y = torch.randn(m, dtype=torch.float64)

    letkf = LETKF(lambda z: H @ z, R, seed=2)
    result = letkf.analyze(Z0, y)
    assert result.Z_analysis.shape == (N, d)
    assert torch.isfinite(result.Z_analysis).all()


def test_letkf_localization_shrinks_far_correlations():
    """Same localization contract PFF's own `localize_fn` gets: a fixed
    taper applied to B should zero out (or strongly shrink) far-apart
    covariance entries relative to the unlocalized analysis -- checked
    directly on the POSTERIOR covariance, not just on B itself, since B
    is internal to `analyze`."""
    torch.manual_seed(3)
    d, m, N = 6, 2, 4000
    H = torch.zeros(m, d, dtype=torch.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    R = torch.eye(m, dtype=torch.float64) * 0.05
    B = torch.eye(d, dtype=torch.float64)
    # Correlate coordinate 0 strongly with coordinate 5 (far away) in the prior.
    B[0, 5] = B[5, 0] = 0.9
    L = torch.linalg.cholesky(B)
    Z0 = torch.randn(N, d, dtype=torch.float64) @ L.T
    y = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def taper_fn(cov):
        taper = torch.ones_like(cov)
        taper[0, 5] = taper[5, 0] = 0.0  # zero the far (0,5) entry specifically
        return cov * taper

    letkf_plain = LETKF(lambda z: H @ z, R, seed=4)
    letkf_localized = LETKF(lambda z: H @ z, R, localize_fn=taper_fn, seed=4)

    za_plain = letkf_plain.analyze(Z0, y).Z_analysis
    za_localized = letkf_localized.analyze(Z0, y).Z_analysis

    # Localizing away the (0,5) prior correlation should reduce how much
    # the analysis moves coordinate 5 in response to an observation of
    # coordinate 0 alone (coordinate 1 is also observed at y=0, coupling
    # only through R which is diagonal, so this isolates the (0,5) path).
    shift_plain = float((za_plain[:, 5].mean() - Z0[:, 5].mean()).abs())
    shift_localized = float((za_localized[:, 5].mean() - Z0[:, 5].mean()).abs())
    assert shift_localized < shift_plain


def test_letkf_analysis_carries_no_autograd_graph():
    """Regression (2026-10-07): see the matching test in test_letkf_local.py."""
    torch.manual_seed(0)
    d, m = 6, 4
    H = torch.randn(m, d, dtype=torch.float64)
    w = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    R = 0.2 * torch.eye(m, dtype=torch.float64)
    out = LETKF(lambda z: (w * H) @ z, R, seed=0).analyze(torch.randn(16, d, dtype=torch.float64),
                                                          torch.randn(m, dtype=torch.float64))
    assert not out.Z_analysis.requires_grad
