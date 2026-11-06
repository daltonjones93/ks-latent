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
