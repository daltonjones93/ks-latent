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
