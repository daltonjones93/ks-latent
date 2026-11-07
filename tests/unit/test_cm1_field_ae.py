"""Track FP1: 2D latent field autoencoder."""
import pytest
import torch

from ks_latent.cm1.data import positive_channel_mask
from ks_latent.cm1.field_ae import FieldAE, FieldAEConfig, jet

V, Z = 7, 12


def _ae(**kw):
    torch.manual_seed(0)
    cfg = FieldAEConfig(n_vars=V, n_levels=Z, width=16, dec_hidden=32,
                        dilations=((1, 1), (2, 2), (1, 4)), **kw)
    return FieldAE(cfg, positive_channel_mask(Z)).eval()


@pytest.mark.parametrize("m,sx,sz,dec", [(1, 1, 1, "jet"), (3, 1, 1, "point"), (4, 2, 1, "jet"), (2, 4, 2, "jet"),
                                         (2, 3, 3, "jet")])
def test_shapes_positivity_and_latent_grid(m, sx, sz, dec):
    ae = _ae(m=m, stride_x=sx, stride_z=sz, decoder=dec)
    u = torch.randn(3, V * Z, 64)
    y, phi = ae(u)
    assert y.shape == u.shape
    assert phi.shape == (3, m, -(-Z // sz), -(-64 // sx))
    pos = positive_channel_mask(Z)
    assert (y[:, pos] >= 0).all()
    assert ae.cfg.numbers_per_point == m / (sx * sz)


def test_jet_is_exact_for_quadratics_in_the_interior():
    zz, xx = torch.meshgrid(torch.arange(9.0), torch.arange(11.0), indexing="ij")
    phi = (0.3 * xx ** 2 - 0.7 * xx * zz + 0.2 * zz ** 2 + 1.5 * xx - 2 * zz + 4)[None, None]
    j = jet(phi)[0, :, 1:-1, 1:-1]
    xi, zi = xx[1:-1, 1:-1], zz[1:-1, 1:-1]
    expect = [phi[0, 0, 1:-1, 1:-1], 0.6 * xi - 0.7 * zi + 1.5, -0.7 * xi + 0.4 * zi - 2,
              torch.full_like(xi, 0.6), torch.full_like(xi, -0.7), torch.full_like(xi, 0.4)]
    for k, e in enumerate(expect):
        torch.testing.assert_close(j[k], e, atol=1e-4, rtol=0)


@pytest.mark.parametrize("sx", [1, 2])
def test_translation_equivariance_away_from_the_boundaries(sx):
    ae = _ae(m=2, stride_x=sx)
    u = torch.randn(1, V * Z, 96)
    s = 4 * sx
    y0, _ = ae(u)
    y1, _ = ae(torch.roll(u, s, dims=-1))
    torch.testing.assert_close(y1[..., 40:60], y0[..., 40 - s:60 - s], atol=1e-5, rtol=1e-5)


def test_decoders_are_local_in_the_latent_point_and_jet():
    for dec, reach in (("point", 0), ("jet", 1)):
        ae = _ae(m=2, decoder=dec)
        phi = torch.randn(1, 2, Z, 20, requires_grad=True)
        y = ae.decode(phi, 20).reshape(1, V, Z, 20)
        y[0, :, 6, 10].sum().backward()
        nz = phi.grad[0].abs().sum(0).nonzero()
        assert (nz[:, 0] - 6).abs().max() <= reach and (nz[:, 1] - 10).abs().max() <= reach


def test_overfits_a_small_batch():
    ae = _ae(m=3).train()
    u = torch.randn(2, V * Z, 32).abs()
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = ((ae(u)[0] - u) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3 * first


def test_bad_config_raises():
    with pytest.raises(ValueError):
        FieldAEConfig(decoder="conv")
    with pytest.raises(ValueError):
        FieldAEConfig(m=0)


def test_field_prop_is_identity_at_init_and_has_its_reach():
    from ks_latent.cm1.field_ae import FieldProp, FieldPropConfig
    torch.manual_seed(0)
    p = FieldProp(FieldPropConfig(m=3, n_levels_latent=10, width=8, layers=2))
    phi = torch.randn(2, 3, 10, 24)
    torch.testing.assert_close(p(phi), phi)
    torch.nn.init.normal_(p.head.weight)
    phi.requires_grad_(True)
    p(phi)[0, :, 5, 12].sum().backward()
    nz = phi.grad[0].abs().sum(0).nonzero()
    assert (nz[:, 0] - 5).abs().max() == 2 and (nz[:, 1] - 12).abs().max() == 2
    assert p.cfg.reach == 2
