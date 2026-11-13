"""7a: global coupling of the leading latent channels in the stencil propagator."""
import torch

from ks_latent.cm1.models import StencilPropagator, StencilPropConfig, _GlobalAttention, _rel_bin


def _prop(kind, **kw):
    torch.manual_seed(0)
    return StencilPropagator(StencilPropConfig(latent_channels=6, hidden=16, layers=3, radius=1, global_kind=kind,
                                               global_k=2, global_hidden=8, global_layers=1, global_heads=2,
                                               global_feat=4, **kw))


def test_local_at_init_and_ablation():
    """At init the stencil ignores the global features; with them switched off the model is exactly local."""
    p = _prop("attn")
    with torch.no_grad():
        p.head.weight.normal_()                               # the zero-init head would hide everything
    z = torch.randn(2, 40, 6)
    z2 = z.clone(); z2[:, 35:, :2] += 3.0                     # far from site 0, beyond the stencil's reach (3)
    assert torch.allclose(p(z)[:, 0], p(z2)[:, 0])
    with torch.no_grad():
        p.convs[0].weight.normal_()
    assert not torch.allclose(p(z)[:, 0], p(z2)[:, 0])        # trained: site 0 now feels site 35
    p.global_off = True
    assert torch.allclose(p(z)[:, 0], p(z2)[:, 0])


def test_global_sees_only_leading_channels():
    for kind in ("attn", "mean"):
        p = _prop(kind)
        with torch.no_grad():
            p.convs[0].weight.normal_(); p.head.weight.normal_()
        z = torch.randn(1, 40, 6)
        z2 = z.clone(); z2[:, 35:, 2:] += 3.0                 # trailing channels far away: invisible to site 0
        assert torch.allclose(p(z)[:, 0], p(z2)[:, 0], atol=1e-6)


def test_any_width_and_shared_bias():
    p = _prop("attn")
    for P in (16, 48, 300):
        assert p(torch.randn(1, P, 6)).shape == (1, P, 6)
    blk = _GlobalAttention(8, 2)
    assert blk.rel.shape == (2, 19)                           # per distance bin, no per-site parameters
    off = torch.tensor([-200, -65, -3, -1, 0, 1, 2, 4, 5, 128, 129])
    b = _rel_bin(off)
    assert b.tolist() == [0, 1, 6, 8, 9, 10, 11, 12, 13, 17, 18]



def test_mean_kind_is_domain_mean():
    p = _prop("mean")
    z = torch.randn(1, 40, 6)
    g = p.glob(z)
    assert torch.allclose(g, p.glob(z[:, torch.randperm(40)]), atol=1e-6)   # permutation-invariant summary
    assert torch.allclose(g[:, 0], g[:, -1])                                  # same at every site
