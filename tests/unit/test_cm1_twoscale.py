"""6a: two-scale wrappers. Decoding is D_L + D_S (condensate clamped); a zero-initialized small model leaves z_S unchanged; the coupled
step equals the large model on z_L; the small model's samples differ only through its noise; the twin's noise projection keeps the
large model's pruned channels at their fill."""
import torch

from ks_latent.cm1.models import ColumnAEConfig, StencilPropConfig, StencilPropagator, build_autoencoder
from ks_latent.cm1.data import positive_channel_mask
from ks_latent.cm1.twoscale import SmallProp, SmallPropConfig, TwoScaleAE, TwoScaleProp


def _models():
    torch.manual_seed(0)
    Z = 2
    ae_l = build_autoencoder(ColumnAEConfig(n_channels=7 * Z, n_levels=Z, hidden=16, latent_channels=6, prefix_m=3), positive_channel_mask(Z))
    ae_l.set_prefix(3, torch.arange(6.0))
    ae_s = build_autoencoder(ColumnAEConfig(n_channels=7 * Z, n_levels=Z, hidden=16, latent_channels=4), None)
    pl = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, body="law", law_residual_hidden=0, prefix_m=3))
    pl.set_prefix(3, torch.arange(6.0))
    with torch.no_grad():
        for q in pl.parameters():
            q.normal_(0, 0.05)
    sm = SmallProp(SmallPropConfig(small_channels=4, cond_channels=3, noise_channels=2, hidden=8))
    return ae_l, ae_s, pl, sm


def test_twoscale_ae_and_step():
    ae_l, ae_s, pl, sm = _models()
    ae, prop = TwoScaleAE(ae_l, ae_s), TwoScaleProp(pl, sm)
    u = torch.rand(2, 14, 32)
    z = ae.encode(u)
    assert z.shape == (2, 4, 10)
    a, b = ae.decode_parts(z)
    dec = ae.decode(z)
    pm = ae.positive_mask[None, :, None]
    assert torch.allclose(dec, torch.where(pm, (a + b).clamp(min=0), a + b))
    z1 = prop(z, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(z1[..., :6], pl(z[..., :6]))
    assert torch.allclose(z1[..., 6:], z[..., 6:])                     # zero-initialized head: identity
    with torch.no_grad():
        for q in sm.parameters():
            q.normal_(0, 0.2)
    s1, s2 = prop(z, gen=torch.Generator().manual_seed(1)), prop(z, gen=torch.Generator().manual_seed(2))
    assert torch.allclose(s1[..., :6], s2[..., :6]) and not torch.allclose(s1[..., 6:], s2[..., 6:])
    Zn = z1 + torch.randn_like(z1)
    Zp = prop.project_latent(Zn)
    assert torch.allclose(Zp[..., 3:6], torch.arange(3.0, 6.0).expand_as(Zp[..., 3:6]))
    assert torch.allclose(Zp[..., :3], Zn[..., :3]) and torch.allclose(Zp[..., 6:], Zn[..., 6:])


def test_filter_split_and_feedback():
    from ks_latent.cm1.twoscale import FeedbackNet, FilterTwoScaleAE, gaussian_lowpass
    ae_l, ae_s, pl, sm = _models()
    ae = FilterTwoScaleAE(ae_l, ae_s, sigma_km=1.0)
    u = torch.rand(2, 14, 64)
    ul = gaussian_lowpass(u, 1.0)
    assert torch.allclose(ae.encode(u)[..., :6], ae_l.encode(ul)) and torch.allclose(ae.encode(u)[..., 6:], ae_s.encode(u - ul))
    assert torch.allclose(gaussian_lowpass(torch.ones(1, 3, 64), 1.0), torch.ones(1, 3, 64), atol=1e-6)      # unit-sum kernel
    fb = FeedbackNet(4, 3)
    prop0, prop1 = TwoScaleProp(pl, sm), TwoScaleProp(pl, sm, fb)
    z = ae.encode(u)
    g = lambda: torch.Generator().manual_seed(0)
    assert torch.allclose(prop0(z, gen=g()), prop1(z, gen=g()))                  # zero-initialized feedback: identical step
    with torch.no_grad():
        for q in fb.parameters():
            q.normal_(0, 0.3)
    o0, o1 = prop0(z, gen=g()), prop1(z, gen=g())
    assert not torch.allclose(o0[..., :3], o1[..., :3]) and torch.allclose(o0[..., 3:6], o1[..., 3:6])   # only the kept channels
