"""Twin-experiment cycling (ks_latent/cm1/twin.py) on synthetic data: structure, reproducibility,
and that assimilation pulls the ensemble toward the observations."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.data import ColumnData, Normalizer, positive_channel_mask, synthetic_members
from ks_latent.cm1.letkf import LETKFConfig
from ks_latent.cm1.models import ColumnAEConfig, ColumnFieldAE, StencilPropagator, StencilPropConfig
from ks_latent.cm1.train import encode_members
from ks_latent.cm1.twin import TwinConfig, _forecast, run_twin, summarize_twin

pytestmark = pytest.mark.integration
Zl, NX = 8, 256


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    ms = synthetic_members(6, T=30, NX=NX, Z=Zl)
    nrm = Normalizer.fit(ms)
    dev = torch.device("cpu")
    truth = ColumnData(ms[:1], nrm, dev)
    others = ColumnData(ms[1:], nrm, dev)
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=7 * Zl, n_levels=Zl, patch=8, hidden=16, latent_channels=4),
                       positive_channel_mask(Zl)).eval()
    prop = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1)).eval()
    z_truth = encode_members(ae, truth)[0]
    Z0 = torch.stack([z[3] for z in encode_members(ae, others)])          # other members at the start frame
    return ae, prop, truth, z_truth, Z0, nrm


def test_free_run_and_reproducibility(setup):
    ae, prop, truth, z_truth, Z0, nrm = setup
    cfg = TwinConfig(t0=3, n_cycles=4, steps_per_cycle=2, halo=1, letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    free = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg, assimilate=False)
    assert len(free["cycles"]) == 4 and all("analysis" not in c for c in free["cycles"])
    a = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg)
    b = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg)
    assert a["cycles"][-1]["analysis"]["nmse_mean_dec"] == b["cycles"][-1]["analysis"]["nmse_mean_dec"]
    s = summarize_twin(a)
    assert set(s) == {"prior", "analysis"} and sum(s["analysis"]["rank_hist"]) > 0
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(t0=3, n_cycles=40, steps_per_cycle=2))


def test_assimilation_reduces_error_at_the_observations(setup):
    """Mechanics: accurate dense observations must pull the ensemble toward them, every cycle.
    Test-only ensemble: 30 members around the encoded truth. The fixture's autoencoder is untrained, so its
    decoder barely responds to the latent (spread at the observations ~0.03 K) and cannot represent the
    truth (decode(encode(truth)) misses by ~0.7 K): the filter can only creep toward the observations.
    With the 5 other synthetic members it barely moves at all - the small-ensemble limit, not a code
    error. Hence: analysis better than prior at every cycle, and >= 20% lower than the first prior."""
    ae, prop, truth, z_truth, Z0, nrm = setup
    g = torch.Generator().manual_seed(1)
    Zs = z_truth[3][None] + 1.0 * torch.randn(30, *z_truth.shape[1:], generator=g)
    cfg = TwinConfig(t0=3, n_cycles=5, steps_per_cycle=2, halo=1, sigma_obs=0.05, obs_every_cells=4,
                     letkf=LETKFConfig(loc_km=6.0, rtps=0.5))
    r = run_twin(ae, prop, truth, z_truth, Zs, nrm, cfg)
    pri = [c["prior"]["th_obs_rmse"] for c in r["cycles"]]
    ana = [c["analysis"]["th_obs_rmse"] for c in r["cycles"]]
    assert all(a < p for a, p in zip(ana, pri)), (pri, ana)
    assert ana[-1] < 0.8 * pri[0], (pri, ana)


def test_halo_carries_the_truth(setup):
    ae, prop, truth, z_truth, Z0, nrm = setup
    Z = _forecast(prop, Z0, z_truth, 3, 2, halo=2)
    torch.testing.assert_close(Z[:, :2], z_truth[5][None, :2].expand(Z.shape[0], -1, -1))
    torch.testing.assert_close(Z[:, -2:], z_truth[5][None, -2:].expand(Z.shape[0], -1, -1))


def test_model_noise_keeps_spread(setup):
    ae, prop, truth, z_truth, Z0, nrm = setup
    no = _forecast(prop, Z0, z_truth, 3, 3, halo=1)
    g = torch.Generator().manual_seed(0)
    yes = _forecast(prop, Z0, z_truth, 3, 3, halo=1, model_noise=0.1, gen=g)
    inner = slice(1, -1)
    assert yes[:, inner].std(0).mean() > no[:, inner].std(0).mean()
    torch.testing.assert_close(yes[:, :1], no[:, :1])          # the halo still carries the truth exactly


def test_reflectivity_and_multimodel(setup):
    from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
    ae, prop, truth, z_truth, Z0, nrm = setup
    head = ReflectivityHead(ReflHeadConfig(n_channels=7 * Zl, hidden=8, layers=2)).eval()
    g = torch.Generator().manual_seed(2)
    Zs = z_truth[3][None] + 1.0 * torch.randn(30, *z_truth.shape[1:], generator=g)
    cfg = TwinConfig(t0=3, n_cycles=3, steps_per_cycle=2, halo=1, assim_theta=False, refl_every_cells=8,
                     letkf=LETKFConfig(loc_km=6.0, rtps=0.5))
    r = run_twin(ae, prop, truth, z_truth, Zs, nrm, cfg, refl_head=head)
    c = r["cycles"][-1]
    assert "refl_obs_rmse" in c["prior"] and len(c["prior"]["ranks_refl"]) == NX // 8
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Zs, nrm, cfg)                 # reflectivity needs the head
    p2 = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1)).eval()
    torch.nn.init.normal_(p2.head.weight, std=0.5)
    Zm = _forecast([prop, p2], Zs, z_truth, 3, 1, halo=1)
    torch.testing.assert_close(Zm[0::2], _forecast(prop, Zs[0::2], z_truth, 3, 1, halo=1))
    torch.testing.assert_close(Zm[1::2], _forecast(p2, Zs[1::2], z_truth, 3, 1, halo=1))
    rm = run_twin(ae, [prop, p2], truth, z_truth, Zs, nrm, TwinConfig(t0=3, n_cycles=2, steps_per_cycle=2, halo=1))
    assert rm["config"]["n_models"] == 2


def test_baseline_modes_and_forecasts_from_analyses(setup):
    ae, prop, truth, z_truth, Z0, nrm = setup
    g = torch.Generator().manual_seed(3)
    Zs = z_truth[3][None] + 0.5 * torch.randn(20, *z_truth.shape[1:], generator=g)
    base = dict(t0=3, n_cycles=3, steps_per_cycle=2, halo=1, letkf=LETKFConfig(loc_km=6.0, rtps=0.5))
    # persistence, no DA: the members never change
    per = run_twin(ae, prop, truth, z_truth, Zs, nrm, TwinConfig(**base, forecast_model="persistence"), assimilate=False)
    assert per["config"]["forecast_model"] == "persistence"
    Zp = _forecast(None, Zs, z_truth, 3, 2, halo=0)          # without the boundary halo, nothing changes
    torch.testing.assert_close(Zp, Zs)
    # EnOI: prior = forecast analysis mean + static anomalies about their own mean
    S = torch.randn(20, *z_truth.shape[1:], generator=g)
    en = run_twin(ae, prop, truth, z_truth, Zs, nrm, TwinConfig(**base, ensemble_mode="enoi"),
                  static_ens=lambda t: S)
    assert len(en["cycles"]) == 3 and "analysis" in en["cycles"][0]
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Zs, nrm, TwinConfig(**base, ensemble_mode="enoi"))
    # physical-space analysis, re-encoded; forecasts from the last analysis
    ph = run_twin(ae, prop, truth, z_truth, Zs, nrm,
                  TwinConfig(**base, analysis_space="physical", forecast_from=(2,), forecast_frames=4))
    c = ph["cycles"][-1]
    assert "analysis_reencoded" in c and len(c["forecast"]) == 4
    assert c["forecast"][0]["lead_frames"] == 1 and np.isfinite(c["forecast"][-1]["nmse_mean_dec"])
    # full surface mesonet
    ms = run_twin(ae, prop, truth, z_truth, Zs, nrm, TwinConfig(**base, surface_vars=("th", "u", "prs", "qv")))
    assert ms["config"]["surface_vars"] == ["th", "u", "prs", "qv"]
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Zs, nrm, TwinConfig(**base, surface_vars=("w",)))


def test_enoi_prior_is_mean_plus_static_anomalies(setup):
    """Check the EnOI prior construction directly: one cycle, no assimilation."""
    from ks_latent.cm1.twin import _forecast as fc
    ae, prop, truth, z_truth, Z0, nrm = setup
    g = torch.Generator().manual_seed(4)
    Zs = torch.randn(10, *z_truth.shape[1:], generator=g)
    S = torch.randn(10, *z_truth.shape[1:], generator=g)
    mean = fc(prop, Zs.mean(0, keepdim=True), z_truth, 3, 2, 1)
    expect = mean + (S - S.mean(0, keepdim=True))
    torch.testing.assert_close(expect.mean(0), mean[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(expect - expect.mean(0, keepdim=True), S - S.mean(0, keepdim=True), atol=1e-6, rtol=0)


def test_enoi_phys_uses_no_latent_and_assimilates(setup):
    ae, prop, truth, z_truth, Z0, nrm = setup
    g = torch.Generator().manual_seed(5)
    Zs = z_truth[3][None] + 0.5 * torch.randn(20, *z_truth.shape[1:], generator=g)
    A = torch.randn(20, truth.C, truth.NX, generator=g)
    cfg = TwinConfig(t0=3, n_cycles=3, steps_per_cycle=2, halo=1, ensemble_mode="enoi_phys", forecast_model="persistence",
                     sigma_obs=0.05, obs_every_cells=4, letkf=LETKFConfig(loc_km=6.0))
    r = run_twin(ae, prop, truth, z_truth, Zs, nrm, cfg, static_phys=lambda t: A)
    pri = [c["prior"]["th_obs_rmse"] for c in r["cycles"]]
    ana = [c["analysis"]["th_obs_rmse"] for c in r["cycles"]]
    assert all(a < p for a, p in zip(ana, pri))
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Zs, nrm, cfg)


def test_stochastic_propagator_is_reproducible_in_the_twin(setup):
    """The learned model error of a stochastic propagator is drawn from the twin's seeded generator:
    the same seed gives the same run, another seed another run, and the ensemble keeps spread with no
    added model noise."""
    ae, _, truth, z_truth, Z0, nrm = setup
    torch.manual_seed(1)
    sp = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1, noise_channels=2)).eval()
    with torch.no_grad():
        sp.head.weight.normal_(0, 0.1)        # zero-init head would ignore the noise
    cfg = lambda s: TwinConfig(t0=3, n_cycles=3, steps_per_cycle=2, halo=1, model_noise=0.0, seed=s,
                               letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    a, b, c = (run_twin(ae, sp, truth, z_truth, Z0, nrm, cfg(s)) for s in (0, 0, 1))
    na, nb, nc = (r["cycles"][-1]["prior"]["nmse_mean_dec"] for r in (a, b, c))
    assert na == nb and na != nc
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    Z1 = _forecast(sp, Z0[:1].repeat(4, 1, 1), z_truth, 3, 3, 1, 0.0, g1)
    Z2 = _forecast(sp, Z0[:1].repeat(4, 1, 1), z_truth, 3, 3, 1, 0.0, g2)
    assert torch.equal(Z1, Z2) and Z1.std(0).mean() > 0


def test_held_boundary_uses_no_future_truth(setup):
    """H0 for the twin: with boundary='persistence' the halo is the truth at t0 throughout, so changing
    the truth's later latent states changes nothing."""
    ae, prop, truth, z_truth, Z0, nrm = setup
    z2 = z_truth.clone(); z2[4:] += 1.0
    a = _forecast(prop, Z0, z_truth, 3, 4, halo=2, hold=3)
    b = _forecast(prop, Z0, z2, 3, 4, halo=2, hold=3)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[:, :2], z_truth[3, :2].expand(a.shape[0], -1, -1))
    assert not torch.allclose(_forecast(prop, Z0, z2, 3, 4, halo=2), a)
    cfg = TwinConfig(t0=3, n_cycles=2, steps_per_cycle=2, halo=1, boundary="persistence",
                     letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    r = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg)
    assert r["config"]["boundary"] == "persistence"
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(t0=3, n_cycles=2, steps_per_cycle=2, boundary="zero"))


def test_exclude_km_scores_only_the_kept_columns(setup):
    """T1: nMSE over the kept columns; excluding nothing equals the default; excluding everything raises."""
    from ks_latent.cm1.twin import _keep
    ae, prop, truth, z_truth, Z0, nrm = setup
    base = dict(t0=3, n_cycles=2, steps_per_cycle=2, halo=1, letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    a = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base))
    b = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, exclude_km=((10.0, 30.0),)))
    c = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, exclude_km=((20.0, 20.0),)))
    na, nb, nc = (r["cycles"][-1]["prior"]["nmse_mean_dec"] for r in (a, b, c))
    assert na == nc and na != nb
    k = _keep(TwinConfig(**base, exclude_km=((0.0, 10.0), (50.0, 60.0))), NX, torch.device("cpu"))
    assert int((~k).sum()) == 80
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, exclude_km=((0.0, 1e6),)))


def test_padded_tiling_observes_the_same_physical_columns(setup):
    """E3c: a tiling padded by o cells observes the same physical columns (same values, same noise) and
    keeps fields on them only; o = 0 with no padding is unchanged."""
    from ks_latent.cm1.data import CM1Member
    ae, prop, truth, z_truth, Z0, nrm = setup
    base = dict(t0=3, n_cycles=2, steps_per_cycle=2, halo=1, letkf=LETKFConfig(loc_km=10.0, rtps=0.3), keep_fields=True)
    a = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base))
    b = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, col_offset=0, n_cols_phys=NX))
    assert a["cycles"][-1]["analysis"]["nmse_mean_dec"] == b["cycles"][-1]["analysis"]["nmse_mean_dec"]
    assert a["cycles"][0]["prior_field"].shape == (7 * Zl, NX)
    o = 3
    m = synthetic_members(6, T=30, NX=NX, Z=Zl)[0]                 # the fixture's truth member
    pad = lambda x, ax: np.concatenate([np.repeat(np.take(x, [0], ax), o, ax), x,
                                        np.repeat(np.take(x, [-1], ax), 8 - o, ax)], ax)
    pm = CM1Member(name="pad", X=pad(m.X, 1), cref=pad(m.cref, 1), t=m.t)
    pt = ColumnData([pm], nrm, torch.device("cpu"))
    u, up = truth.full(0), pt.full(0)
    cols = np.arange(4, NX, 8)
    torch.testing.assert_close(up[:, 2 * Zl, cols + o], u[:, 2 * Zl, cols])   # same theta' at the obs
    zp = encode_members(ae, pt)[0]
    Zp = zp[3][None].expand(Z0.shape[0], -1, -1) + 0.1 * torch.randn(Z0.shape[0], *zp.shape[1:])
    r = run_twin(ae, prop, pt, zp, Zp, nrm, TwinConfig(**base, col_offset=o, n_cols_phys=NX))
    assert r["cycles"][0]["prior_field"].shape == (7 * Zl, NX)
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, col_offset=4, n_cols_phys=NX))


def test_radial_velocity_col_offset_matches_unpadded():
    from ks_latent.cm1.obs import RadialVelocity
    from ks_latent.cm1.physics import BaseState
    zl = BaseState.from_sounding().z
    nrm = Normalizer(scale=np.ones(7), vars=("u", "w", "th", "qv", "prs", "qliq", "qice"))
    a = RadialVelocity(nrm, zl, 1600, radar_x_km=330.0)
    b = RadialVelocity(nrm, zl, 1600, radar_x_km=330.0, col_offset=6)
    np.testing.assert_array_equal(b.columns, a.columns + 6)
    np.testing.assert_allclose(b.x_km, a.x_km + 1.5)
    np.testing.assert_allclose(b.height_m, a.height_m)
    u = torch.randn(1, 7 * zl.size, 1600)
    up = torch.cat([u[..., :6], u, u[..., :2]], dim=-1)
    np.testing.assert_allclose(b._from_columns(up), a._from_columns(u))


def test_observation_only_mode_has_no_memory(setup):
    """G2c: the background is the archive at each frame (no model, no memory), so the prior does not depend
    on the initial ensemble and the analysis improves on it at the observations."""
    ae, prop, truth, z_truth, Z0, nrm = setup
    others = ColumnData(synthetic_members(6, T=30, NX=NX, Z=Zl)[1:], nrm, torch.device("cpu"))
    static_phys = lambda t: torch.stack([others.full(m, t)[0] for m in range(len(others.u))])
    cfg = TwinConfig(t0=3, n_cycles=3, steps_per_cycle=2, halo=1, ensemble_mode="oi_clim",
                     forecast_model="persistence", model_noise=0.0, letkf=LETKFConfig(loc_km=10.0, rtps=0.0))
    a = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg, static_phys=static_phys)
    b = run_twin(ae, prop, truth, z_truth, Z0 + 1.0, nrm, cfg, static_phys=static_phys)
    assert a["cycles"][-1]["prior"]["nmse_mean_dec"] == b["cycles"][-1]["prior"]["nmse_mean_dec"]
    c = a["cycles"][-1]
    assert c["analysis"]["th_obs_rmse"] < c["prior"]["th_obs_rmse"]
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg)


def test_perturbed_ranks_g3_scores_and_analog_forecast_model(setup):
    """Claim 2 / G3 / G2b plumbing: perturbed-member ranks are recorded beside the plain ones (same counts);
    forecasts carry persistence-of-analysis scores and reflectivity FSS (persistence scored against the moving
    truth, so its error differs by lead); the analog forecast model runs in the twin."""
    from ks_latent.cm1.analog import AnalogConfig, AnalogPropagator
    from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
    ae, prop, truth, z_truth, Z0, nrm = setup
    head = ReflectivityHead(ReflHeadConfig(n_channels=7 * Zl, hidden=8, layers=2)).eval()
    cfg = TwinConfig(t0=3, n_cycles=4, steps_per_cycle=2, halo=1, refl_every_cells=8, forecast_from=(1,),
                     forecast_frames=3, letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    r = run_twin(ae, prop, truth, z_truth, Z0, nrm, cfg, refl_head=head)
    s = summarize_twin(r)
    N = Z0.shape[0]
    assert len(s["prior"]["rank_hist_pert"]) == N + 1 and sum(s["prior"]["rank_hist_pert"]) == sum(s["prior"]["rank_hist"])
    assert len(s["prior"]["ranks_pert_cycles"]) == 2                      # second half of 4 cycles
    fc = r["cycles"][1]["forecast"]
    assert {"nmse_persist", "front_disp_km_persist", "fss_refl35", "fss_refl35_persist"} <= set(fc[0])
    assert len({round(f["nmse_persist"], 12) for f in fc}) == 3           # truth moves, the held analysis does not
    # the analog archive: the other members' encoded trajectories
    arch = [z_truth[:20]]
    an = AnalogPropagator(arch, AnalogConfig(radius=1, k=3, frame_stride=1, pca_dim=0))
    ra = run_twin(ae, an, truth, z_truth, Z0, nrm, TwinConfig(t0=3, n_cycles=2, steps_per_cycle=2, halo=1,
                                                              forecast_model="analog",
                                                              letkf=LETKFConfig(loc_km=10.0, rtps=0.3)))
    assert ra["config"]["forecast_model"] == "analog" and np.isfinite(ra["cycles"][-1]["prior"]["nmse_mean_dec"])


def test_adaptive_noise_follows_the_innovations():
    """N2: the amplitude grows where innovations exceed the prior spread plus R, shrinks where the ensemble is
    over-dispersed, stays put where a site sees no observations, and respects the bounds."""
    from ks_latent.cm1.twin import adapt_noise
    rng = np.random.default_rng(0)
    N, m = 40, 50
    x_obs = np.linspace(0.0, 20.0, m)
    x_site = np.array([5.0, 15.0, 100.0])                   # the last site is far from every observation
    r = np.ones(m)
    y = np.zeros(m)
    Yb = rng.normal(0.0, 0.3, (N, m))                        # spread 0.3 << innovation scale
    Yb[:, x_obs > 10] = rng.normal(0.0, 3.0, (N, (x_obs > 10).sum()))   # right half over-dispersed
    Yb -= Yb.mean(0) - np.where(x_obs > 10, 0.0, 2.0)        # left: mean 2 away from y (d^2 = 4 > s^2 + R)
    amp = torch.full((3,), 0.1)
    cfg = TwinConfig(model_noise=0.1, adaptive_noise="site", letkf=LETKFConfig(loc_km=2.0))
    a1 = adapt_noise(amp, Yb, y, r, x_obs, x_site, cfg)
    assert a1[0] > 0.1 and a1[1] < 0.1 and a1[2] == 0.1
    a = amp
    for _ in range(50):
        a = adapt_noise(a, Yb, y, r, x_obs, x_site, cfg)
    assert torch.allclose(a, torch.tensor([0.5, 0.02, 0.1]))   # pinned at the bounds (5x, 0.2x); far site untouched
    g = adapt_noise(amp, Yb, y, r, x_obs, x_site, TwinConfig(model_noise=0.1, adaptive_noise="global"))
    assert torch.allclose(g, g[0].expand(3))                  # one factor everywhere


def test_adaptive_noise_and_anchored_ensemble_in_the_twin(setup):
    """N2 and M3 run end to end. Anchored: one forecast step puts the mean exactly on the deterministic model's
    step of the mean and keeps the stochastic model's anomalies; the stochastic member must share the latent."""
    ae, prop, truth, z_truth, Z0, nrm = setup
    base = dict(t0=3, n_cycles=4, steps_per_cycle=2, halo=1, letkf=LETKFConfig(loc_km=10.0, rtps=0.3))
    r = run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, model_noise=0.1, adaptive_noise="site"))
    amps = [c["noise_amp"] for c in r["cycles"]]
    assert all(0.02 <= x <= 0.5 for x in amps) and len(set(amps)) > 1
    with pytest.raises(ValueError):
        run_twin(ae, prop, truth, z_truth, Z0, nrm, TwinConfig(**base, model_noise=0.0, adaptive_noise="site"))
    torch.manual_seed(2)
    sp = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1, noise_channels=2)).eval()
    det = StencilPropagator(StencilPropConfig(latent_channels=4, hidden=8, layers=1, radius=1)).eval()
    with torch.no_grad():
        sp.head.weight.normal_(0, 0.1)
        det.head.weight.normal_(0, 0.1)
    g = torch.Generator().manual_seed(3)
    Z1 = _forecast({"det": det, "stoch": sp}, Z0, z_truth, 3, 1, 0, 0.0, g)
    g = torch.Generator().manual_seed(3)
    S = sp(Z0, gen=g)
    with torch.no_grad():
        assert torch.allclose(Z1.mean(0), det(Z0.mean(0, keepdim=True))[0], atol=1e-5)
        assert torch.allclose(Z1 - Z1.mean(0), S - S.mean(0), atol=1e-5)
    ra = run_twin(ae, (det, sp), truth, z_truth, Z0, nrm, TwinConfig(**base, forecast_model="anchored"))
    assert np.isfinite(ra["cycles"][-1]["prior"]["nmse_mean_dec"])
    with pytest.raises(ValueError):
        run_twin(ae, sp, truth, z_truth, Z0, nrm, TwinConfig(**base, forecast_model="anchored"))
