"""SM4: the flow-matching propagator."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.flow import FlowPropagator, FlowPropConfig
from ks_latent.cm1.models import rollout_from


def _cfg(**kw):
    return FlowPropConfig(**{"latent_channels": 3, "hidden": 16, "layers": 1, "flow_hidden": 16, "flow_layers": 1,
                             "n_flow": 4, "time_features": 4, "delta_scale": [0.5, 1.0, 2.0], **kw})


def test_shapes_reproducibility_and_interface():
    torch.manual_seed(0)
    m = FlowPropagator(_cfg()).eval()
    z = torch.randn(2, 20, 3)
    a = m(z, gen=torch.Generator().manual_seed(1))
    b = m(z, gen=torch.Generator().manual_seed(1))
    c = m(z, gen=torch.Generator().manual_seed(2))
    assert a.shape == z.shape and torch.equal(a, b) and not torch.equal(a, c)
    assert m.cfg.noise_channels == 3 and m.cfg.n_hist == 1 and m.cfg.reach_sites == 1 + 4
    out = rollout_from(m, torch.randn(10, 20, 3), 2, 4, 1, n_samples=3, gen=torch.Generator().manual_seed(0))
    assert out.shape == (4, 3, 20, 3)
    with pytest.raises(ValueError):
        FlowPropagator(_cfg(delta_scale=[1.0]))


def test_locality_of_one_sample():
    """With the noise fixed, changing z at one site changes the sample only within reach_sites."""
    torch.manual_seed(0)
    m = FlowPropagator(_cfg()).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, 0.3)
    z = torch.randn(1, 40, 3)
    eps = torch.randn(1, 3, 40)
    z2 = z.clone(); z2[0, 20] += 1.0
    d = (m(z, eps=eps) - m(z2, eps=eps)).abs().sum(-1)[0]
    far = [j for j in range(40) if abs(j - 20) > m.cfg.reach_sites]
    assert d[20] > 0 and float(d[far].max()) == 0.0


def test_learns_a_bimodal_step():
    """Each element of the next state is z + 1 or z - 1 with equal probability (independently): a regression
    would predict z (the mean); the flow model's samples must split into the two modes, element by element."""
    torch.manual_seed(0)
    m = FlowPropagator(_cfg(delta_scale=[1.0, 1.0, 1.0], n_flow=16, hidden=32, flow_hidden=64, flow_layers=2))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(1500):
        z0 = torch.randn(32, 12, 3, generator=g)
        sign = torch.where(torch.rand(32, 12, 3, generator=g) < 0.5, -1.0, 1.0)
        loss = m.loss(z0, z0 + sign, gen=g)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        z0 = torch.zeros(100, 12, 3)
        d = (m(z0, gen=torch.Generator().manual_seed(5)) - z0).numpy().ravel()
    assert np.mean(np.abs(d) < 0.25) < 0.15                       # rarely near the mean (0), where regression sits
    assert np.mean(np.abs(d)) > 0.7                                # mass at the modes
    assert 0.35 < np.mean(d > 0) < 0.65                            # both modes, about equally
