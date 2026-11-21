"""10a: Stage C weight EMA."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.data import ColumnData, Normalizer, synthetic_members
from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder
from ks_latent.cm1.train import StageCConfig, train_stage_c

Z = 4


def _setup(seed=0):
    ms = synthetic_members(3, T=60, NX=96 * 4, Z=Z)
    d = ColumnData(ms, Normalizer.fit(ms), torch.device("cpu"))
    torch.manual_seed(seed)
    ae = build_autoencoder(ColumnAEConfig(n_channels=d.C, n_levels=Z, patch=4, hidden=16, latent_channels=6, n_mix=1), None)
    prop = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, layers=2, radius=1))
    return d, ae, prop


KW = dict(window_sites=40, halo_sites=2, batch=2, steps=6, k_max=4, val_every=3, val_batches=1, warmup=1, select_last=True)


def _params(ae, prop):
    return torch.cat([q.detach().flatten() for q in list(ae.parameters()) + list(prop.parameters())])


def test_ema_off_is_unchanged_and_ema_differs():
    d, ae0, p0 = _setup(); train_stage_c(ae0, p0, d, d, StageCConfig(**KW), log=lambda *_: None)
    d, ae1, p1 = _setup(); train_stage_c(ae1, p1, d, d, StageCConfig(**KW, ema=0.0), log=lambda *_: None)
    assert torch.equal(_params(ae0, p0), _params(ae1, p1))
    d, ae2, p2 = _setup(); train_stage_c(ae2, p2, d, d, StageCConfig(**KW, ema=0.9), log=lambda *_: None)
    d, aei, pi_ = _setup()
    a, b, c = _params(ae0, p0), _params(ae2, p2), _params(aei, pi_)
    assert not torch.equal(a, b)
    # the EMA lies between the initial and the final weights, closer to the final (decay 0.9 over 6 steps with warm-up)
    assert float((b - a).norm()) < float((c - a).norm())


def test_ema_validation_and_bad_decay():
    d, ae, prop = _setup()
    out = train_stage_c(ae, prop, d, d, StageCConfig(**KW, ema=0.5), log=lambda *_: None)
    assert all(np.isfinite(h["val_decoded_rollout_mse"]) for h in out["history"])
    with pytest.raises(ValueError):
        train_stage_c(ae, prop, d, d, StageCConfig(**KW, ema=1.0), log=lambda *_: None)


def test_noise_temperature():
    """10b: the runtime noise temperature scales the drawn noise channels; 1 is unchanged, 0 is deterministic."""
    torch.manual_seed(0)
    p = StencilPropagator(StencilPropConfig(latent_channels=5, hidden=8, layers=2, radius=1, noise_channels=2))
    with torch.no_grad():
        p.head.weight.normal_()
    z = torch.randn(2, 20, 5)
    a = p(z, gen=torch.Generator().manual_seed(1))
    p.noise_temp = 1.0
    assert torch.equal(a, p(z, gen=torch.Generator().manual_seed(1)))
    p.noise_temp = 0.0
    assert torch.equal(p(z, gen=torch.Generator().manual_seed(1)), p(z, gen=torch.Generator().manual_seed(2)))
    p.noise_temp = 2.0
    e = torch.randn((2, 2, 20), generator=torch.Generator().manual_seed(1))
    assert torch.allclose(p(z, gen=torch.Generator().manual_seed(1)), p(z, eps=2.0 * e), atol=1e-6)
