"""Per-site latent LETKF with open boundaries (ks_latent/cm1/letkf.py) and the surface-theta operator."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.letkf import LETKFConfig, gaspari_cohn, letkf_analysis

pytestmark = pytest.mark.unit
rng = np.random.default_rng(0)


def test_gaspari_cohn_shape():
    r = np.linspace(0, 2.5, 251)
    g = gaspari_cohn(r)
    assert g[0] == 1 and np.all(g[r >= 2] == 0) and np.all(np.diff(g) <= 1e-12)
    assert abs(gaspari_cohn(np.array([1.0]))[0] - 0.2083333) < 1e-6   # known value at r = 1


def test_equals_kalman_update_without_localization():
    """Linear H, no localization, rho = 1: ETKF mean and covariance = Kalman filter on the sample covariance."""
    N, P, c, m = 40, 6, 3, 5
    Zb = rng.normal(size=(N, P, c)) @ rng.normal(size=(c, c))
    Hm = rng.normal(size=(m, P * c))
    Yb = Zb.reshape(N, -1) @ Hm.T
    y = rng.normal(size=m)
    R = rng.uniform(0.5, 2.0, size=m)
    cfg = LETKFConfig(loc_km=1e9)
    Za = letkf_analysis(Zb, Yb, y, R, np.arange(P) * 2.0, rng.uniform(0, 10, m), cfg).reshape(N, -1)
    X = Zb.reshape(N, -1)
    B = np.cov(X, rowvar=False)
    K = B @ Hm.T @ np.linalg.inv(Hm @ B @ Hm.T + np.diag(R))
    np.testing.assert_allclose(Za.mean(0), X.mean(0) + K @ (y - Hm @ X.mean(0)), atol=1e-10)
    np.testing.assert_allclose(np.cov(Za, rowvar=False), (np.eye(P * c) - K @ Hm) @ B, atol=1e-10)


def test_locality_and_open_boundaries():
    N, P, c = 20, 50, 2
    x_site = (np.arange(P) + 0.5) * 2.0                   # 2 km sites, domain 0-100 km
    Zb = rng.normal(size=(N, P, c))
    x_obs = np.array([3.0])                               # one observation near the WEST edge
    Yb = Zb[:, 1, :1].copy()                              # it observes site 1
    cfg = LETKFConfig(loc_km=5.0)                         # zero weight beyond 10 km
    Za = letkf_analysis(Zb, Yb, np.array([5.0]), np.array([0.1]), x_site, x_obs, cfg)
    far = np.abs(x_site - x_obs[0]) >= 10.0
    np.testing.assert_array_equal(Za[:, far], Zb[:, far])  # untouched beyond 2c...
    assert far[-1] and far[-5:].all()                     # ...including the EAST edge: no wrap-around
    assert np.abs(Za[:, 1].mean(0) - Zb[:, 1].mean(0)).max() > 0.1


def test_rtps_restores_prior_spread():
    N, P, c = 30, 4, 2
    Zb = rng.normal(size=(N, P, c))
    Yb = Zb[:, :, 0]
    x = np.arange(P) * 1.0
    kw = dict(y=np.zeros(P), r_var=np.full(P, 0.01), x_site_km=x, x_obs_km=x)
    a0 = letkf_analysis(Zb, Yb, cfg=LETKFConfig(loc_km=1e9), **kw)
    a1 = letkf_analysis(Zb, Yb, cfg=LETKFConfig(loc_km=1e9, rtps=1.0), **kw)
    np.testing.assert_allclose(a1.std(0, ddof=1), Zb.std(0, ddof=1), rtol=1e-10)
    np.testing.assert_allclose(a1.mean(0), a0.mean(0), atol=1e-12)
    assert (a0.std(0, ddof=1)[:, 0] < 0.5 * Zb.std(0, ddof=1)[:, 0]).all()


def test_surface_theta_operator_matches_decoded_state():
    from ks_latent.cm1.data import VARS, CM1Member, Normalizer, positive_channel_mask
    from ks_latent.cm1.models import ColumnAEConfig, ColumnFieldAE
    from ks_latent.cm1.obs import SurfaceTheta
    Z, NX = 4, 64
    X = rng.normal(size=(3, NX, len(VARS), Z)).astype(np.float16)
    nrm = Normalizer.fit([CM1Member("a", X, np.zeros((3, NX), np.float32), np.arange(3) * 60.0)])
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=len(VARS) * Z, n_levels=Z, patch=8, hidden=8, latent_channels=2),
                       positive_channel_mask(Z)).double()
    op = SurfaceTheta(np.array([3, 20, 41]), nrm, Z)
    Zl = torch.randn(5, NX // 8, 2, dtype=torch.float64)
    got = op(ae, Zl)
    dec = ae.decode(Zl).detach()
    expect = dec[:, VARS.index("th") * Z, [3, 20, 41]].numpy() * nrm.scale[VARS.index("th")]
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    assert np.allclose(op.x_km, [0.875, 5.125, 10.375]) and op.r_var(0.5).tolist() == [0.25] * 3


def test_radial_velocity_operator_known_field():
    from ks_latent.cm1.data import VARS, CM1Member, Normalizer
    from ks_latent.cm1.obs import RadialVelocity
    from ks_latent.cm1.physics import BaseState
    base = BaseState.from_sounding()
    Z, NX = base.z.size, 1600
    X = np.ones((1, 4, len(VARS), Z), np.float16)
    nrm = Normalizer.fit([CM1Member("a", X, np.zeros((1, 4), np.float32), np.zeros(1))])
    op = RadialVelocity(nrm, base.z, NX, radar_x_km=330.0, tilts_deg=(0.5, 5.0))
    # beam height: 4/3-Earth model
    r = np.abs(op.x_km - 330.0) * 1e3
    np.testing.assert_allclose(op.height_m, r * np.sin(np.deg2rad(np.where(op.cos > np.cos(np.deg2rad(1)), 0.5, 5.0)))
                               + r**2 / (2 * 4 / 3 * 6.371e6), rtol=1e-12)
    assert (np.abs(op.x_km - 330.0) <= 150.0).all() and (op.sign[op.x_km < 330] == -1).all()
    # u' = 10 m/s and w = 2 m/s everywhere: Vr = sign * 10 cos(a) + 2 sin(a), exactly (linear interpolation of constants)
    u = torch.zeros(len(VARS) * Z, NX, dtype=torch.float64)
    u[VARS.index("u") * Z:(VARS.index("u") + 1) * Z] = 10.0 / nrm.scale[VARS.index("u")]
    u[VARS.index("w") * Z:(VARS.index("w") + 1) * Z] = 2.0 / nrm.scale[VARS.index("w")]
    np.testing.assert_allclose(op.truth_values(u), op.sign * 10.0 * op.cos + 2.0 * op.sin, rtol=1e-10)


@pytest.mark.parametrize("loc,rtps", [(5.0, 0.0), (1e9, 0.5), (3.0, 0.9)])
def test_batched_letkf_equals_per_site_loop(loc, rtps):
    from ks_latent.cm1.letkf import letkf_analysis_torch
    N, P, c, m = 15, 37, 4, 23
    Zb = rng.normal(size=(N, P, c))
    Yb = rng.normal(size=(N, m))
    y, R = rng.normal(size=m), rng.uniform(0.3, 2.0, m)
    xs, xo = np.arange(P) * 2.0, rng.uniform(-5, 80, m)
    cfg = LETKFConfig(loc_km=loc, inflation=1.1, rtps=rtps)
    a = letkf_analysis(Zb, Yb, y, R, xs, xo, cfg)
    b = letkf_analysis_torch(Zb, Yb, y, R, xs, xo, cfg, chunk=8)
    np.testing.assert_allclose(b, a, atol=1e-10, rtol=0)
