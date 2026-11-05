"""Physical diagnostics (ks_latent/cm1/physics.py) against analytic fields."""
import numpy as np
import pytest

from ks_latent.cm1.data import VARS
from ks_latent.cm1.physics import (DX_M, G, BaseState, cold_pool_intensity, condensate_mass, divergence_ratio,
                                   low_level_liquid)

pytestmark = pytest.mark.unit
base = BaseState.from_sounding()
Z, NX = len(base.z), 400
x = np.arange(NX) * DX_M


def fields():
    return np.zeros((len(VARS), Z, NX))


def test_base_state_is_physical():
    assert Z == 62 and np.all(np.diff(base.z) > 0)
    assert 1.10 < base.rho0[0] < 1.20 and base.rho0[-1] < 0.15 and np.all(np.diff(base.rho0) < 0)
    assert abs(base.theta0[0] - 300.0) < 0.2


def test_divergence_zero_for_streamfunction_flow_and_one_for_divergent():
    kx, kz = 2 * np.pi / 20e3, np.pi / 12e3
    psi = np.sin(kx * x)[None, :] * np.sin(kz * base.z)[:, None]           # rho0 u = -dpsi/dz, rho0 w = dpsi/dx
    X = fields()
    X[VARS.index("u")] = -(kz * np.sin(kx * x)[None, :] * np.cos(kz * base.z)[:, None]) / base.rho0[:, None]
    X[VARS.index("w")] = (kx * np.cos(kx * x)[None, :] * np.sin(kz * base.z)[:, None]) / base.rho0[:, None]
    assert divergence_ratio(X, base) < 0.05
    Y = fields(); Y[VARS.index("u")] = np.sin(kx * x)[None, :]
    assert abs(divergence_ratio(Y, base) - 1.0) < 1e-9


def test_condensate_mass_of_uniform_field():
    X = fields()
    X[VARS.index("qliq")] = np.sqrt(1.0)   # 1 g/kg everywhere
    expect = 1e-3 * np.sum(base.rho0 * np.gradient(base.z)) * NX * DX_M
    assert abs(condensate_mass(X, base) / expect - 1) < 1e-12
    two = condensate_mass(np.stack([X, 2 * X]), base)          # 2x sqrt(q) is 4x the mass
    assert two.shape == (2,) and abs(two[1] / two[0] - 4) < 1e-12


def test_cold_pool_intensity_of_uniform_layer():
    X = fields()
    k = base.z < 1500.0
    X[VARS.index("th"), k] = -4.0
    X[VARS.index("th"), ~k] = 1.0
    C = cold_pool_intensity(X, base)
    h_eff = np.sum(np.where(k, 4.0 / base.theta0 * np.gradient(base.z), 0.0))
    assert np.allclose(C, np.sqrt(2 * G * h_eff)) and 15 < C[0] < 22   # about sqrt(2 g dtheta/theta h)
    X[VARS.index("th"), 0, :NX // 2] = 0.5                              # warm surface: no surface-based pool
    assert np.all(cold_pool_intensity(X, base)[:NX // 2] == 0)


def test_low_level_liquid():
    X = fields()
    X[VARS.index("qliq"), base.z < 800, 10] = 2.0
    X[VARS.index("qliq"), base.z > 3000, 20] = 5.0                       # aloft: not counted
    ll = low_level_liquid(X, base)
    assert ll[10] == 2.0 and ll[20] == 0.0


@pytest.mark.integration
def test_reports_include_physics_on_62_level_columns():
    """The reports compute physics only for real 62-level CM1 columns; check they run and are sane."""
    import torch
    from ks_latent.cm1.data import CM1Member, ColumnData, Normalizer, positive_channel_mask
    from ks_latent.cm1.evaluate import forecast_report, reconstruction_report
    from ks_latent.cm1.models import ColumnAEConfig, ColumnFieldAE, StencilPropagator, StencilPropConfig
    from ks_latent.cm1.train import encode_members
    rng = np.random.default_rng(0)
    T, NXs = 8, 128
    ms = []
    for i in range(2):
        X = rng.normal(size=(T, NXs, len(VARS), Z)).astype(np.float32)
        X[:, :, [VARS.index("qliq"), VARS.index("qice")]] = np.abs(X[:, :, [VARS.index("qliq"), VARS.index("qice")]])
        ms.append(CM1Member(f"m{i}", X.astype(np.float16), np.full((T, NXs), 30.0, np.float32), np.arange(T) * 60.0))
    nrm = Normalizer.fit(ms)
    d = ColumnData(ms, nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=len(VARS) * Z, n_levels=Z, patch=8, hidden=16, latent_channels=4),
                       positive_channel_mask(Z))
    prop = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1))
    rr = reconstruction_report(ae, d, nrm)
    assert set(rr["physics"]) == {"div_ratio_truth", "div_ratio_recon", "condensate_mass_rel_err"}
    fc = forecast_report(ae, prop, d, encode_members(ae, d), nrm, t0s=[1], n_lead=3, halo=1)
    assert len(fc["div_ratio_fc"]) == 3 and np.isfinite(fc["cold_pool_cmax_truth"]).all()
    assert set(fc["fss_lowliq"]) == {"0.3", "1"}
