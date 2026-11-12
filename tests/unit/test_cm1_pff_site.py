"""Per-site PFF (ks_latent/cm1/pff_site.py): with the whole domain in one window, no localization and a linear observation operator the
analysis ensemble is the Kalman posterior; a nonlinear operator is solved by the re-linearization; localization leaves far sites alone."""
import numpy as np
import torch

from ks_latent.cm1.pff_site import PFFSiteConfig, pff_site_analysis


def _setup(P=7, c=2, N=8000, n_obs=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = P * c
    M = torch.randn(d, d, generator=g, dtype=torch.float64) * 0.5
    Sig = M @ M.T + 0.3 * torch.eye(d, dtype=torch.float64)
    L = torch.linalg.cholesky(Sig)
    Zb = (torch.randn(N, d, generator=g, dtype=torch.float64) @ L.T + 0.4).reshape(N, P, c)
    H = torch.randn(n_obs, d, generator=g, dtype=torch.float64)
    return Zb, H


def test_whole_domain_linear_matches_kalman():
    P, c = 7, 2
    Zb, H = _setup(P, c)
    N, d = Zb.shape[0], P * c
    z0 = Zb.reshape(N, d).mean(0); B = torch.cov(Zb.reshape(N, d).T)
    R = 0.2 * torch.eye(H.shape[0], dtype=torch.float64)
    y = H @ (z0 + 0.5) + 0.1
    obs_fn = lambda z: H @ z.reshape(-1).double()
    cfg = PFFSiteConfig(loc_km=1e6, taper_km=1e6, half_width=3, n_steps=40, ridge=1e-9, ridge_floor=1e-12, early_stop_after=39)
    Za, info = pff_site_analysis(Zb, obs_fn, y.numpy(), np.diag(R.numpy()), (np.arange(P) + 0.5) * 2.0, np.linspace(1, 13, H.shape[0]), cfg,
                                 gen=torch.Generator().manual_seed(1))
    S = H @ B @ H.T + R
    K = B @ H.T @ torch.linalg.inv(S)
    mean_k = z0 + K @ (y - H @ z0)
    cov_k = B - K @ H @ B
    za = Za.reshape(N, d).double()
    assert torch.allclose(za.mean(0), mean_k, atol=2e-3), (za.mean(0) - mean_k).abs().max()
    cov_a = torch.cov(za.T)
    assert float((cov_a - cov_k).norm() / cov_k.norm()) < 0.06


def test_nonlinear_operator_is_relinearized_to_the_root():
    P, c, N = 7, 2, 4000
    g = torch.Generator().manual_seed(2)
    Zb = torch.randn(N, P, c, generator=g, dtype=torch.float64)
    h = lambda z: (z[3, 0] + 0.2 * z[3, 0] ** 3)[None].double()
    root = float(np.roots([0.2, 0, 1, -3.0]).real[np.abs(np.roots([0.2, 0, 1, -3.0]).imag) < 1e-9][0])
    cfg = PFFSiteConfig(loc_km=1e6, taper_km=1e6, half_width=3, n_steps=40, ridge=1e-9, early_stop_after=39)
    Za, _ = pff_site_analysis(Zb, h, np.array([3.0]), np.array([0.01]), (np.arange(P) + 0.5) * 2.0, np.array([7.0]), cfg,
                              gen=torch.Generator().manual_seed(3))
    assert abs(float(Za[:, 3, 0].mean()) - root) < 0.03, (float(Za[:, 3, 0].mean()), root)


def test_localization_leaves_distant_sites_unchanged_and_tail_untouched():
    P, c, N = 21, 3, 64
    g = torch.Generator().manual_seed(4)
    Zb = torch.randn(N, P, c, generator=g, dtype=torch.float64)
    h = lambda z: z[2, :2].double()                       # observes channels 0-1 of site 2 only
    cfg = PFFSiteConfig(loc_km=2.0, taper_km=2.0, half_width=3, n_steps=6)
    Za, _ = pff_site_analysis(Zb, h, np.array([1.0, 1.0]), np.array([0.1, 0.1]), (np.arange(P) + 0.5) * 2.0, np.array([5.0, 5.0]), cfg, m=2,
                              gen=torch.Generator().manual_seed(5))
    assert torch.allclose(Za[:, 5:], Zb[:, 5:])         # beyond 2 c_loc of the observed site 2 (x = 5 km): untouched
    assert torch.allclose(Za[..., 2], Zb[..., 2])        # the channel above m is never updated
    assert not torch.allclose(Za[:, 2, :2], Zb[:, 2, :2])
