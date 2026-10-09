"""Tests for `ks_latent.da.letkf_local.LETKFLocal2D` (T2,
docs/ks_local_field_ltransfer.tex sec:c19_da_hypotheses). Mirrors
`tests/unit/test_letkf.py`'s structure: the analytic linear-Gaussian
Kalman posterior is the ground truth (ground rule 1), used directly
rather than only comparing against the dense `LETKF` class (which uses
a different localization formulation -- Schur-product covariance
localization vs. this module's R-localization -- so the two are
expected to agree closely, not bitwise, per H1's own "within a few
percent" framing)."""

from __future__ import annotations

import pytest
import torch

from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D


def _kalman_posterior(H, R, B, z0_bar, y):
    Binv = torch.linalg.inv(B)
    Rinv = torch.linalg.inv(R)
    F = H.T @ Rinv @ H + Binv
    cov = torch.linalg.inv(F)
    mean = z0_bar + cov @ H.T @ Rinv @ (y - H @ z0_bar)
    return mean, cov


def _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=0):
    """A linear `h(z) = H @ z` with EXACTLY the block-diagonal structure
    a zero-decoder-reach `KSAutoencoderLocalField2D` has: pixel `p`
    (here, one pixel per site -- `px=py=1`) depends only on its own
    site's `channels` latent coordinates. `site_of_obs` is the identity
    map (`obs_site_map_2d` with `px=py=1`, stride 1)."""
    torch.manual_seed(seed)
    n_sites = n_sites_x * n_sites_y
    d = n_sites * channels
    H = torch.zeros(n_sites, d, dtype=torch.float64)
    for s in range(n_sites):
        H[s, s * channels:(s + 1) * channels] = torch.randn(channels, dtype=torch.float64)
    site_of_obs = obs_site_map_2d(n_sites_x, n_sites_y, n_sites_x, n_sites_y, 1, 1)
    assert torch.equal(site_of_obs, torch.arange(n_sites, dtype=torch.int64))

    def h(z):
        return H @ z

    return h, H, site_of_obs, d


@pytest.mark.slow
def test_letkf_local_matches_analytic_kalman_posterior_with_wide_taper():
    """With a Gaspari-Cohn half-width far larger than the whole (small,
    periodic) lattice, the taper is ~1 everywhere (not exactly 1 for any
    r > 0, but negligibly different -- see module docstring), so the
    per-site R-localized analysis should recover the same posterior as
    the unlocalized, exact global Kalman update, to the same 5% bound
    `test_letkf_gaussian_linear_matches_analytic_kalman_posterior` uses."""
    n_sites_x, n_sites_y, channels = 3, 3, 2
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels)
    N = 8000
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.1, dtype=torch.float64)
    R = torch.diag(R_diag)
    B = torch.eye(d, dtype=torch.float64)
    z0_bar = torch.zeros(d, dtype=torch.float64)
    y = torch.randn(n_sites_x * n_sites_y, generator=torch.Generator().manual_seed(7), dtype=torch.float64)

    L = torch.linalg.cholesky(B)
    Z0 = z0_bar + torch.randn(N, d, dtype=torch.float64) @ L.T

    letkf_local = LETKFLocal2D(
        h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=1.0e6, seed=0,
    )
    result = letkf_local.analyze(Z0, y)

    mean_emp = result.Z_analysis.mean(dim=0)
    cov_emp = torch.cov(result.Z_analysis.T)
    mean_true, cov_true = _kalman_posterior(H, R, B, z0_bar, y)

    mean_rel_err = float((mean_emp - mean_true).norm() / mean_true.norm())
    cov_rel_err = float((cov_emp - cov_true).norm() / cov_true.norm())
    assert mean_rel_err < 0.05, f"mean rel err {mean_rel_err:.4f}"
    assert cov_rel_err < 0.05, f"cov rel err {cov_rel_err:.4f}"


def test_letkf_local_shape_and_finite():
    n_sites_x, n_sites_y, channels = 2, 2, 3
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=1)
    N = 40
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.2, dtype=torch.float64)
    Z0 = torch.randn(N, d, dtype=torch.float64)
    y = torch.randn(n_sites_x * n_sites_y, dtype=torch.float64)

    letkf_local = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=2)
    result = letkf_local.analyze(Z0, y)
    assert result.Z_analysis.shape == (N, d)
    assert torch.isfinite(result.Z_analysis).all()


def test_letkf_local_raises_on_wrong_latent_dimension():
    n_sites_x, n_sites_y, channels = 2, 2, 3
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=1)
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.2, dtype=torch.float64)
    letkf_local = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=2)
    Z0_wrong = torch.randn(10, d + 1, dtype=torch.float64)
    y = torch.randn(n_sites_x * n_sites_y, dtype=torch.float64)
    with pytest.raises(ValueError, match="expected"):
        letkf_local.analyze(Z0_wrong, y)


def test_letkf_local_tight_taper_gives_exactly_zero_shift_beyond_support_radius():
    """Direct analogue of `test_letkf.py
    ::test_letkf_localization_shrinks_far_correlations`, but stronger:
    because the Gaspari-Cohn taper has COMPACT support (exactly zero
    beyond `2*gc_c`, not merely small), a site far enough from every
    observation should see EXACTLY zero shift with a tight taper, and a
    real shift with a wide one -- not just "smaller."""
    n_sites_x, n_sites_y, channels = 6, 6, 2
    n_sites = n_sites_x * n_sites_y
    d = n_sites * channels
    site_A, site_B = 0, 3 * n_sites_y + 3  # (0,0) and (3,3): max torus distance in a 6x6 grid

    # Observe only site A, both channels.
    H = torch.zeros(2, d, dtype=torch.float64)
    H[0, site_A * channels + 0] = 1.0
    H[1, site_A * channels + 1] = 1.0
    site_of_obs = torch.tensor([site_A, site_A], dtype=torch.int64)

    def h(z):
        return H @ z

    R_diag = torch.full((2,), 0.05, dtype=torch.float64)
    torch.manual_seed(4)
    B = torch.eye(d, dtype=torch.float64)
    # Strongly correlate site A's channel 0 with the FAR site B's channel 0 in the prior.
    iA, iB = site_A * channels + 0, site_B * channels + 0
    B[iA, iB] = B[iB, iA] = 0.9
    L = torch.linalg.cholesky(B)
    N = 2000
    Z0 = torch.randn(N, d, generator=torch.Generator().manual_seed(5), dtype=torch.float64) @ L.T
    y = torch.tensor([1.0, 0.0], dtype=torch.float64)

    letkf_tight = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=1.0, seed=6)
    letkf_wide = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=10.0, seed=6)

    za_tight = letkf_tight.analyze(Z0, y).Z_analysis
    za_wide = letkf_wide.analyze(Z0, y).Z_analysis

    shift_tight = float((za_tight[:, iB].mean() - Z0[:, iB].mean()).abs())
    shift_wide = float((za_wide[:, iB].mean() - Z0[:, iB].mean()).abs())

    assert shift_tight == 0.0, f"expected exactly zero shift beyond the taper's compact support, got {shift_tight}"
    assert shift_wide > 0.0, "a wide taper should let the far correlation actually move site B"


def test_letkf_local_analysis_carries_no_autograd_graph():
    """Regression (2026-10-07): with a decoder whose parameters require
    gradients, the analysis came back attached to the Jacobian's autograd
    graph; a cycling loop that stores analyses then retained every cycle's
    graph (~0.24 GB per cycle at d=384, tens of GB per run)."""
    n_sites_x, n_sites_y, channels = 2, 2, 3
    _, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=3)
    w = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

    def h(z):
        return (w * H) @ z

    R_diag = torch.full((n_sites_x * n_sites_y,), 0.2, dtype=torch.float64)
    Z0 = torch.randn(16, d, dtype=torch.float64)
    y = torch.randn(n_sites_x * n_sites_y, dtype=torch.float64)
    out = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=0).analyze(Z0, y)
    assert not out.Z_analysis.requires_grad
    assert out.Z_analysis.grad_fn is None


def _kalman_with_sample_cov(H, R, Z0, y):
    """Kalman update with the ensemble's own sample covariance (no inverse of B, which is
    singular for N < d)."""
    zb = Z0.mean(0)
    B = torch.cov(Z0.T)
    K = B @ H.T @ torch.linalg.inv(H @ B @ H.T + R)
    return zb + K @ (y - H @ zb), B - K @ H @ B


def test_letkf_local_etkf_equals_kalman_with_sample_covariance():
    """D4: deterministic transform, linear h, taper ~1 everywhere: analysis mean and covariance
    equal the Kalman update computed from the forecast ensemble's sample covariance exactly
    (no sampling noise; small N suffices)."""
    from ks_latent.da.letkf_local import LETKFLocal2DETKF

    n_sites_x, n_sites_y, channels = 3, 2, 2
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=3)
    N = 24
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.3, dtype=torch.float64)
    Z0 = 0.5 + torch.randn(N, d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    y = torch.randn(n_sites_x * n_sites_y, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    out = LETKFLocal2DETKF(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=1.0e6).analyze(Z0, y)
    mean_k, cov_k = _kalman_with_sample_cov(H, torch.diag(R_diag), Z0, y)
    assert torch.allclose(out.Z_analysis.mean(0), mean_k, atol=1e-8)
    assert torch.allclose(torch.cov(out.Z_analysis.T), cov_k, atol=1e-8)


def test_letkf_local_etkf_is_deterministic_and_respects_support():
    """Same inputs, any seed -> identical analysis; sites with no observation in range keep
    their forecast exactly (same support rule as the perturbed-observation filter)."""
    from ks_latent.da.letkf_local import LETKFLocal2DETKF

    n_sites_x, n_sites_y, channels = 6, 1, 2
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=6)
    keep = torch.tensor([0])  # observe only site 0
    R_diag = torch.full((1,), 0.2, dtype=torch.float64)

    def h0(z):
        return h(z)[keep]

    Z0 = torch.randn(16, d, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    y = torch.zeros(1, dtype=torch.float64)
    a = LETKFLocal2DETKF(h0, R_diag, site_of_obs[keep], n_sites_x, n_sites_y, channels, gc_c=1.0, seed=0).analyze(Z0, y)
    b = LETKFLocal2DETKF(h0, R_diag, site_of_obs[keep], n_sites_x, n_sites_y, channels, gc_c=1.0, seed=9).analyze(Z0, y)
    assert torch.equal(a.Z_analysis, b.Z_analysis)
    far = torch.arange(3 * channels, 4 * channels)  # site 3: periodic distance 3 > support 2*gc_c = 2
    assert torch.equal(a.Z_analysis[:, far], Z0[:, far])
    assert not torch.allclose(a.Z_analysis[:, :channels], Z0[:, :channels])


def test_letkf_local_rtps_keeps_mean_and_sets_spread():
    """RTPS (task I21): the analysis mean is unchanged; alpha = 1 restores the prior spread per
    coordinate exactly; alpha = 0 is the plain filter."""
    n_sites_x, n_sites_y, channels = 2, 2, 3
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=3)
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.05, dtype=torch.float64)
    Z0 = torch.randn(30, d, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    y = torch.randn(n_sites_x * n_sites_y, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    plain = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=2).analyze(Z0, y)
    full = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=2,
                        rtps=1.0).analyze(Z0, y)
    half = LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, seed=2,
                        rtps=0.5).analyze(Z0, y)
    torch.testing.assert_close(full.Z_analysis.mean(0), plain.Z_analysis.mean(0), rtol=0, atol=1e-12)
    torch.testing.assert_close(full.Z_analysis.std(0), Z0.std(0), rtol=1e-12, atol=1e-12)
    sa, sb = plain.Z_analysis.std(0), Z0.std(0)
    torch.testing.assert_close(half.Z_analysis.std(0), sa + 0.5 * (sb - sa), rtol=1e-12, atol=1e-12)


def test_letkf_local_rtps_rejects_out_of_range():
    n_sites_x, n_sites_y, channels = 2, 2, 3
    h, H, site_of_obs, d = _block_diagonal_decoder(n_sites_x, n_sites_y, channels, seed=1)
    R_diag = torch.full((n_sites_x * n_sites_y,), 0.2, dtype=torch.float64)
    with pytest.raises(ValueError, match="rtps"):
        LETKFLocal2D(h, R_diag, site_of_obs, n_sites_x, n_sites_y, channels, gc_c=2.0, rtps=1.5)
