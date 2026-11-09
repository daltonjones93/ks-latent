"""FPf: the constrained channel-0 law (LatentLaw lead='z0')."""
import torch
import torch.nn.functional as F

from ks_latent.cm1.models import LatentLaw, StencilPropagator, StencilPropConfig


def _randomize(law):
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in law.named_parameters():
            if n != "nu0_raw":
                p.normal_(0, 0.1)


def test_channel0_depends_on_z0_only():
    c = 5
    law = LatentLaw(c, residual_hidden=16, lead="z0", lead_nu_pos=True, global_mean=True).double()
    _randomize(law)
    z = torch.randn(2, 12, c, dtype=torch.float64, requires_grad=True)
    f0 = law.tendency(z)[..., 0].sum()
    g, = torch.autograd.grad(f0, z)
    assert g[..., 1:].abs().max() == 0          # no other channel enters channel 0's law
    assert g[..., 0].abs().max() > 0
    # and the other channels still see everything
    g1, = torch.autograd.grad(law.tendency(z)[..., 1].sum(), z)
    assert g1[..., 2:].abs().max() > 0


def test_channel0_law_matches_closed_form():
    c = 4
    law = LatentLaw(c, residual_hidden=8, lead="z0", lead_nu_pos=True).double()
    _randomize(law)
    z = torch.randn(3, 10, c, dtype=torch.float64)
    co = law.effective_coefficients()
    A, Q1, Q2 = co["A"], co["Q1"], co["Q2"]
    z0 = z[..., :1]
    zp = torch.cat([z0[:, :1], z0, z0[:, -1:]], 1)
    d1, d2 = 0.5 * (zp[:, 2:] - zp[:, :-2]), zp[:, 2:] - 2 * z0 + zp[:, :-2]
    nu = F.softplus(law.nu0_raw)
    assert torch.isclose(A[2, 0, 0], nu)
    want = law.b[0] + A[0, 0, 0] * z0 + A[1, 0, 0] * d1 + nu * d2 + Q1[0, 0] * z0 ** 2 + Q2[0, 0] * z0 * d1
    torch.testing.assert_close(law.tendency(z)[..., :1], want)
    assert int((A[:, 0] != 0).sum()) == 3 and int((Q1[0] != 0).sum()) == 1 and int((Q2[0] != 0).sum()) == 1


def test_nu_nonnegative_and_near_identity_at_init():
    law = LatentLaw(3, lead="z0", lead_nu_pos=True)
    assert float(law.effective_coefficients()["A"][2, 0, 0]) > 0
    z = torch.randn(1, 8, 3)
    assert (law(z) - z).abs().max() < 0.05


def test_default_is_unchanged_and_propagator_wiring():
    cfg = StencilPropConfig(latent_channels=3, body="law", law_lead="z0", law_lead_nu_pos=True)
    p = StencilPropagator(cfg)
    assert p.law.lead == "z0" and p.law.lead_nu_pos
    off = LatentLaw(3)
    assert not hasattr(off, "lead_A")
