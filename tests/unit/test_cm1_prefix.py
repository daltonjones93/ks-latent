"""5s: pruned latent (prefix_m): channels >= m are the constant fill in encode and in the propagator's input and output."""
import torch
from ks_latent.cm1.models import ColumnAEConfig, StencilPropConfig, StencilPropagator, build_autoencoder


def _models(m=0):
    ae = build_autoencoder(ColumnAEConfig(n_channels=14, n_levels=2, hidden=16, latent_channels=6, prefix_m=m))
    prop = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, body="law", law_residual_hidden=0, prefix_m=m))
    return ae, prop


def test_prefix_tail_is_constant_and_head_unchanged():
    torch.manual_seed(0)
    ae0, p0 = _models(0)
    ae, p = _models(3)
    ae.load_state_dict(ae0.state_dict(), strict=False); p.load_state_dict(p0.state_dict(), strict=False)
    fill = torch.arange(6.0)
    ae.set_prefix(3, fill); p.set_prefix(3, fill)
    u = torch.randn(2, 14, 32)
    z0, z = ae0.encode(u), ae.encode(u)
    assert torch.allclose(z[..., :3], z0[..., :3])
    assert torch.allclose(z[..., 3:], fill[3:].expand_as(z[..., 3:]))
    with torch.no_grad():
        for q in p.parameters():
            q.normal_(0, 0.1)
        out = p(torch.randn(2, 32, 6))
    assert torch.allclose(out[..., 3:], fill[3:].expand_as(out[..., 3:]))


def test_prefix_state_dict_roundtrip():
    ae, p = _models(3)
    ae2, p2 = _models(3)
    ae2.load_state_dict(ae.state_dict()); p2.load_state_dict(p.state_dict())


def test_twin_forecast_noise_does_not_touch_pruned_channels():
    from ks_latent.cm1.twin import _forecast
    _, p = _models(3)
    fill = torch.arange(6.0); p.set_prefix(3, fill)
    Z = torch.randn(5, 20, 6); Z[..., 3:] = fill[3:]
    zt = torch.randn(30, 20, 6)
    out = _forecast(p, Z, zt, 0, 3, 2, model_noise=0.3, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(out[:, 2:-2, 3:], fill[3:].expand_as(out[:, 2:-2, 3:]))
    assert out[:, 2:-2, :3].std(0).min() > 0          # the kept channels do get the noise
