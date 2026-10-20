"""Tests for Z20 E1 diagnostics (`ks_latent.da.nonlinearity`) and the iterative per-site update
(`ks_latent.da.letkf_local.LETKFLocal2DIterative`, Z20 E3)."""
from __future__ import annotations

import numpy as np
import torch

from ks_latent.da.jacobian_coloring import coloured_obs_jacobian_2d_compact, obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D, LETKFLocal2DIterative
from ks_latent.da.nonlinearity import (innovation_chi2_diag, jdelta_compact, linearization_stats,
                                       rank_histogram)


def _setup(nsx=3, nsy=3, c=2, quad=0.0, seed=0):
    """Block-diagonal observation operator, one observation per site: h_s(z) = a_s . z_s + quad*(b_s . z_s)^2."""
    g = torch.Generator().manual_seed(seed)
    P = nsx * nsy
    A = torch.randn(P, c, generator=g, dtype=torch.float64)
    Bq = torch.randn(P, c, generator=g, dtype=torch.float64)
    site = obs_site_map_2d(nsx, nsy, nsx, nsy, 1, 1)

    def h(z):
        zz = z.reshape(P, c)
        return (A * zz).sum(-1) + quad * ((Bq * zz).sum(-1)) ** 2
    return h, site, P, c


def _stats(h, site, P, c, spread, N=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    zbar = torch.randn(P * c, generator=g, dtype=torch.float64)
    Zp = spread * torch.randn(N, P * c, generator=g, dtype=torch.float64)
    Zp = Zp - Zp.mean(0)
    Z = zbar + Zp
    J = coloured_obs_jacobian_2d_compact(h, zbar, site, c)
    Hz = torch.stack([h(z) for z in Z])
    return linearization_stats(Hz, h(zbar), jdelta_compact(Zp, J, site, c), 0.1, site, P)


def test_eta_zero_for_linear_operator():
    h, site, P, c = _setup(quad=0.0)
    s = _stats(h, site, P, c, 0.3)
    assert s["eta"] < 1e-12 and s["eta_o"] < 1e-12 and s["jensen_obs_over_sigma"] < 1e-12


def test_eta_grows_linearly_with_spread_for_quadratic_operator():
    h, site, P, c = _setup(quad=0.5)
    e1, e2 = _stats(h, site, P, c, 0.05)["eta"], _stats(h, site, P, c, 0.1)["eta"]
    assert abs(e2 / e1 - 2.0) < 1e-6  # residual is exactly quadratic, J delta linear


def test_jdelta_matches_dense_jacobian():
    h, site, P, c = _setup(quad=0.3)
    z = torch.randn(P * c, dtype=torch.float64)
    Jd = torch.autograd.functional.jacobian(h, z)
    Zp = torch.randn(5, P * c, dtype=torch.float64)
    Jc = coloured_obs_jacobian_2d_compact(h, z, site, c)
    assert torch.allclose(jdelta_compact(Zp, Jc, site, c), Zp @ Jd.T, atol=1e-12)


def test_chi2_and_rank_histogram_consistent_ensemble():
    rng = np.random.default_rng(0)
    m, N, so = 40000, 20, 0.1
    mu, sd = rng.standard_normal(m), 0.3
    Hz = mu + sd * rng.standard_normal((N, m))
    y = mu + sd * rng.standard_normal(m) + so * rng.standard_normal(m)  # truth drawn like a member, observed
    chi2 = innovation_chi2_diag(torch.tensor(y), torch.tensor(Hz), so)
    assert abs(chi2 - 1.0) < 0.03  # sd of the ratio ~ sqrt(2/m) ~ 0.007
    h = rank_histogram(y, Hz, so, rng)
    assert h.sum() == m and h.shape == (N + 1,)
    expected = m / (N + 1)
    assert np.all(np.abs(h - expected) < 5 * np.sqrt(expected))
    # an underdispersed ensemble gives chi2 > 1 and a U-shaped histogram
    Hz_u = mu + 0.3 * sd * rng.standard_normal((N, m))
    assert innovation_chi2_diag(torch.tensor(y), torch.tensor(Hz_u), so) > 2.0
    hu = rank_histogram(y, Hz_u, so, rng)
    assert hu[0] + hu[-1] > 3 * 2 * expected


def _ens(P, c, N, spread, seed):
    g = torch.Generator().manual_seed(seed)
    return 0.5 + spread * torch.randn(N, P * c, generator=g, dtype=torch.float64)


def test_iterative_one_iteration_equals_standard_update():
    h, site, P, c = _setup(quad=0.5)
    Z0, y = _ens(P, c, 32, 0.3, 2), torch.randn(P, dtype=torch.float64)
    R = torch.full((P,), 0.01, dtype=torch.float64)
    a = LETKFLocal2D(h, R, site, 3, 3, c, gc_c=1.0, seed=5, rtps=0.5).analyze(Z0, y).Z_analysis
    b = LETKFLocal2DIterative(h, R, site, 3, 3, c, gc_c=1.0, seed=5, rtps=0.5, n_iter=1).analyze(Z0, y).Z_analysis
    assert torch.allclose(a, b, atol=1e-12)


def test_iterative_is_a_no_op_for_linear_operator():
    h, site, P, c = _setup(quad=0.0)
    Z0, y = _ens(P, c, 32, 0.3, 2), torch.randn(P, dtype=torch.float64)
    R = torch.full((P,), 0.01, dtype=torch.float64)
    a = LETKFLocal2DIterative(h, R, site, 3, 3, c, gc_c=1.0, seed=5, n_iter=1).analyze(Z0, y).Z_analysis
    b = LETKFLocal2DIterative(h, R, site, 3, 3, c, gc_c=1.0, seed=5, n_iter=3, tol=0.0).analyze(Z0, y).Z_analysis
    assert torch.allclose(a, b, atol=1e-10)


def test_iterative_moves_mean_towards_exact_posterior_mean():
    """One site, c = 1, many observations of a strongly nonlinear h(z) = z + q z^2: the exact posterior
    mean (quadrature) is matched better after Gauss-Newton iterations than after the one-step update."""
    q, N, m, so = 1.5, 4000, 6, 0.1
    site = torch.zeros(m, dtype=torch.int64)

    def h(z):
        return (z + q * z ** 2).repeat(m)
    g = torch.Generator().manual_seed(0)
    zb, sb = 0.3, 0.3
    Z0 = zb + sb * torch.randn(N, 1, generator=g, dtype=torch.float64)
    zt = 0.55
    y = torch.full((m,), zt + q * zt ** 2, dtype=torch.float64) + so * torch.randn(m, generator=g, dtype=torch.float64)
    R = torch.full((m,), so ** 2, dtype=torch.float64)
    grid = np.linspace(-1.5, 2.0, 20001)
    logp = -0.5 * ((grid - float(Z0.mean())) / float(Z0.std())) ** 2 \
        - 0.5 * (((y.numpy()[:, None] - (grid + q * grid ** 2)[None]) / so) ** 2).sum(0)
    w = np.exp(logp - logp.max())
    post_mean = float((w * grid).sum() / w.sum())
    errs = []
    for n_iter in (1, 4):
        f = LETKFLocal2DIterative(h, R, site, 1, 1, 1, gc_c=1.0, seed=3, n_iter=n_iter, tol=0.0)
        errs.append(abs(float(f.analyze(Z0, y).Z_analysis.mean()) - post_mean))
    assert errs[1] < 0.5 * errs[0], errs
