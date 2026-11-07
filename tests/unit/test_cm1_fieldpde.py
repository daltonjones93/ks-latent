"""Track FP0 numerics: conserved moist variables, vorticity, semi-Lagrangian transport, STLSQ."""
import numpy as np
import pytest

from ks_latent.cm1.fieldpde import (CP, DX_M, LV, bootstrap_stlsq, departure_points, exner, moist_fields,
                                    semi_lagrangian, stlsq)


def _raw(T=2, Z=12, NX=40, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(50.0, 6000.0, Z) ** 1.0
    p = 1e5 * np.exp(-z / 8000.0)
    f = dict(th=300 + 0.004 * z[None, :, None] + rng.normal(0, 0.5, (T, Z, NX)),
             prs=np.broadcast_to(p[None, :, None], (T, Z, NX)).copy(),
             qv=np.clip(0.014 * np.exp(-z / 2500)[None, :, None] + rng.normal(0, 1e-4, (T, Z, NX)), 0, None),
             uinterp=rng.normal(0, 3, (T, Z, NX)), winterp=rng.normal(0, 1, (T, Z, NX)))
    for k in ("qc", "qr", "qi", "qs", "qg"):
        f[k] = np.abs(rng.normal(0, 1e-4, (T, Z, NX)))
    return f, z


def test_theta_il_and_qt_are_invariant_under_condensation_at_fixed_pressure():
    f, z = _raw()
    rho0 = np.exp(-z / 9000.0)
    a = moist_fields(f, z, base_i=0, rho0=rho0)
    dq = 3e-4 * np.random.default_rng(1).random(f["qv"].shape)
    g = dict(f)
    g["qv"], g["qc"] = f["qv"] - dq, f["qc"] + dq
    g["th"] = f["th"] + LV / (CP * exner(f["prs"])) * dq
    b = moist_fields(g, z, base_i=0, rho0=rho0)
    np.testing.assert_allclose(b["theta_il"], a["theta_il"], atol=1e-10)
    np.testing.assert_allclose(b["qt"], a["qt"], atol=1e-15)
    assert np.abs(b["theta"] - a["theta"]).max() > 0.1          # theta itself is not conserved


def test_vorticity_of_shear_and_of_solid_body_rotation():
    Z, NX = 30, 50
    z = np.linspace(0, 7250.0, Z)
    x = np.arange(NX) * DX_M
    zz, xx = np.meshgrid(z, x, indexing="ij")
    f, _ = _raw(T=1, Z=Z, NX=NX)
    om = 0.01
    f["uinterp"] = (om * (zz - z.mean()))[None]
    f["winterp"] = (-om * (xx - x.mean()))[None]
    eta = moist_fields(f, z, base_i=0, rho0=np.ones(Z))["eta"]
    np.testing.assert_allclose(eta, 2 * om, rtol=1e-10)


def test_semi_lagrangian_translates_a_smooth_field_under_uniform_flow():
    Z, NX = 20, 200
    z = np.cumsum(np.linspace(50, 400, Z))                      # stretched levels
    x = np.arange(NX) * DX_M
    k, c, wz, dt = 2 * np.pi / (40 * DX_M), 21.0, 1.5, 60.0      # 5.04 cells per step
    phi = np.sin(k * x)[None, :] * np.ones((Z, 1)) + 1e-4 * z[:, None]
    U, W = np.full((Z, NX), c), np.full((Z, NX), wz)
    dep = departure_points(U, W, U, W, dt, z)
    out = semi_lagrangian(phi, dep)
    exact = np.sin(k * (x - c * dt))[None, :] + 1e-4 * (z[:, None] - wz * dt)
    inner = (slice(2, Z - 2), slice(20, NX - 20))
    np.testing.assert_allclose(out[inner], exact[inner], atol=2e-3)


def test_stlsq_recovers_a_sparse_law_and_bootstrap_reports_frequencies():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8)) * np.array([1, 10, 0.1, 2, 1, 1, 5, 1])
    y = 2.0 * X[:, 0] - 0.5 * X[:, 3] + 0.01 * rng.normal(size=4000)
    xi = stlsq(X, y, threshold=0.05)
    np.testing.assert_allclose(xi, [2, 0, 0, -0.5, 0, 0, 0, 0], atol=2e-3)
    res = bootstrap_stlsq(X, y, groups=np.arange(4000) // 100, threshold=0.05, n_boot=10)
    np.testing.assert_array_equal(res["freq"], [1, 0, 0, 1, 0, 0, 0, 0])
    assert res["r2_full"] > 0.999
    with pytest.raises(ValueError):
        stlsq(np.ones((10, 2)), np.arange(10.0), 0.1)


def test_pipeline_recovers_a_known_source_along_trajectories():
    Z, NX, dt = 16, 160, 60.0
    z = np.linspace(100.0, 8000.0, Z)
    x = np.arange(NX) * DX_M
    rng = np.random.default_rng(3)
    U = 15 + 3 * np.sin(2 * np.pi * x / (NX * DX_M))[None] * np.ones((Z, 1))
    W = np.zeros((Z, NX))
    phi0 = np.sin(2 * np.pi * x / (30 * DX_M))[None] * np.cos(z / 3000.0)[:, None]
    h = np.cos(2 * np.pi * x / (25 * DX_M))[None] * np.ones((Z, 1))           # static source field
    d = rng.normal(size=(Z, NX)) * 0.1                                        # distractor term
    dep = departure_points(U, W, U, W, dt, z)
    h_traj = 0.5 * (semi_lagrangian(h, dep) + h)
    phi1 = semi_lagrangian(phi0, dep) + dt * 0.003 * h_traj
    r = (phi1 - semi_lagrangian(phi0, dep)) / dt
    X = np.stack([h_traj.ravel(), d.ravel()], axis=1)
    xi = stlsq(X, r.ravel(), threshold=0.05)
    np.testing.assert_allclose(xi, [0.003, 0.0], atol=1e-8)
