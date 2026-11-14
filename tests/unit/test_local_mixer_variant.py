"""The CM1 mixer's details (RMSNorm + SwiGLU) in the Paper I/II `local_mlp` and `local_mlp_2d` propagators."""
import pytest
import torch

from ks_latent.config import AuxPropagatorConfig
from ks_latent.models.propagator import AuxPropagator


def _cfg1d(variant):
    return AuxPropagatorConfig(d_latent=48, mode="markovian", backbone="local_mlp", n_tokens=16, token_d_model=32,
                               token_n_layers=2, token_mlp_ratio=4, attn_window=3, zero_init=True, mixer_variant=variant)


def _cfg2d(variant, nx=12, ny=8, c=6, layers=2):
    return AuxPropagatorConfig(d_latent=nx * ny * c, mode="markovian", backbone="local_mlp_2d", attn_window=3,
                               local_mlp2d_n_sites_x=nx, local_mlp2d_n_sites_y=ny, local_mlp2d_local_channels=c,
                               local_mlp2d_token_d_model=32, local_mlp2d_n_layers=layers, token_mlp_ratio=4,
                               zero_init=True, mixer_variant=variant)


def _n(m):
    return sum(p.numel() for p in m.body.parameters())


@pytest.mark.parametrize("mk", [_cfg1d, _cfg2d])
def test_variants_same_size_and_identity_at_init(mk):
    base, new = AuxPropagator(mk("gelu_ln")), AuxPropagator(mk("swiglu_rms"))
    assert abs(_n(new) / _n(base) - 1) < 0.03, (_n(new), _n(base))
    d = new.cfg.d_latent
    z = torch.randn(3, d)
    # zero-initialized output projection: the body's delta is exactly zero
    assert torch.equal(new.body(z), torch.zeros_like(z))
    assert torch.equal(base.body(z), torch.zeros_like(z))


def test_default_is_the_published_block():
    p = AuxPropagator(_cfg1d("gelu_ln"))
    keys = {k for k in p.state_dict() if "blocks.0" in k}
    assert any(k.endswith("mlp.0.weight") for k in keys) and not any("w12" in k for k in keys)
    assert isinstance(p.body.norm, torch.nn.LayerNorm)
    q = AuxPropagator(_cfg1d("swiglu_rms"))
    assert any("w12" in k for k in q.state_dict()) and isinstance(q.body.norm, torch.nn.RMSNorm)
    assert AuxPropagatorConfig.__dataclass_fields__["mixer_variant"].default == "gelu_ln"
    with pytest.raises(ValueError):
        AuxPropagator(_cfg1d("nope"))


def test_reach_is_unchanged_1d():
    """The reach of the 1D mixer is n_layers x window tokens whichever variant (locality, not the block, sets it)."""
    for variant in ("gelu_ln", "swiglu_rms"):
        p = AuxPropagator(_cfg1d(variant))
        with torch.no_grad():
            p.body.token_unembed.weight.normal_(); p.body.token_unembed.bias.zero_()
        z = torch.randn(1, 48); z2 = z.clone()
        z2[:, 3 * 8:3 * 8 + 3] += 5.0                       # token 8 (3 channels per token), perturbed
        with torch.no_grad():
            d1, d2 = p.body(z), p.body(z2)
        diff = (d1 - d2).abs().reshape(16, 3).amax(1)
        far = [i for i in range(16) if min((i - 8) % 16, (8 - i) % 16) > 2 * 3]     # beyond 2 layers x window 3 on the ring
        assert float(diff[far].max()) < 1e-6
        assert float(diff.max()) > 1e-6


def test_2d_swiglu_forward_shape_and_zero_init_2layers():
    p = AuxPropagator(_cfg2d("swiglu_rms", layers=2))
    z = torch.randn(2, p.cfg.d_latent)
    assert p.body(z).shape == z.shape
