"""Tests for `KSAutoencoderLocalFieldAniso2D` (`encoder_kind=
"local_field_2d_aniso"`, added 2026-09-29, user-directed after G2's own
reconciliation check confirmed Kolmogorov flow's transfer gap is
genuine -- see `docs/ks_local_field_ltransfer.tex`'s "orientation
hypothesis" note). Mirrors `test_autoencoder_local_field_2d.py`'s own
discipline, with the KEY difference being the anisotropic equivariance
tests: exact in x, exact at the y-forcing-period shift, and explicitly
NOT exact for a generic y-shift (ground rule 19, "test the failure
modes" -- this is a genuine physical asymmetry, not a bug).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalFieldAutoencoderAniso2DConfig
from ks_latent.models.autoencoder_local_field_2d_aniso import KSAutoencoderLocalFieldAniso2D


def _make(NX=64, NY=64, n_sites_x=4, n_sites_y=8, local_channels=3, forcing_n=4, **kw):
    cfg = LocalFieldAutoencoderAniso2DConfig(
        NX=NX, NY=NY, n_sites_x=n_sites_x, n_sites_y=n_sites_y,
        local_channels=local_channels, forcing_n=forcing_n, **kw,
    )
    return KSAutoencoderLocalFieldAniso2D(cfg), cfg


def test_shapes():
    ae, cfg = _make()
    u = torch.randn(3, cfg.NX * cfg.NY)
    z = ae.encode(u)
    assert z.shape == (3, cfg.d_latent)
    u_hat = ae.decode(z)
    assert u_hat.shape == (3, cfg.NX * cfg.NY)


def test_gauge_anchor_is_fixed_physical_average():
    ae, cfg = _make(local_channels=3)
    u = torch.randn(2, cfg.NX * cfg.NY)
    with torch.no_grad():
        z = ae.encode(u)
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    u_patches = u.reshape(2, cfg.n_sites_x, cfg.patch_x, cfg.n_sites_y, cfg.patch_y)
    expected_anchor = u_patches.mean(dim=(2, 4))
    assert torch.allclose(z_field[..., 0], expected_anchor, atol=1e-5)


def test_x_equivariance_exact():
    ae, cfg = _make()
    ae.eval()
    torch.manual_seed(0)
    u = torch.randn(1, cfg.NX, cfg.NY)
    shift_sites = 1
    u_shifted = torch.roll(u, shift_sites * cfg.patch_x, dims=1)
    with torch.no_grad():
        z1 = ae.encode(u.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
        z2 = ae.encode(u_shifted.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    assert torch.allclose(torch.roll(z1, shift_sites, dims=1), z2, atol=1e-5)


def test_y_forcing_period_equivariance_exact():
    ae, cfg = _make(n_sites_y=8, forcing_n=4)
    ae.eval()
    period_sites = cfg.n_sites_y // cfg.forcing_n
    torch.manual_seed(1)
    u = torch.randn(1, cfg.NX, cfg.NY)
    u_shifted = torch.roll(u, period_sites * cfg.patch_y, dims=2)
    with torch.no_grad():
        z1 = ae.encode(u.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
        z2 = ae.encode(u_shifted.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    assert torch.allclose(torch.roll(z1, period_sites, dims=2), z2, atol=1e-5)


def test_generic_y_shift_NOT_equivariant():
    """Ground rule 19: the anisotropic design deliberately does NOT
    generalize across an arbitrary y-shift -- if this ever started
    passing, that would mean the phase channels had silently stopped
    depending on y (a regression), not an improvement."""
    ae, cfg = _make(n_sites_y=8, forcing_n=4)  # period_sites=2
    ae.eval()
    period_sites = cfg.n_sites_y // cfg.forcing_n
    torch.manual_seed(2)
    u = torch.randn(1, cfg.NX, cfg.NY)
    shift_sites = 1
    assert shift_sites % period_sites != 0
    u_shifted = torch.roll(u, shift_sites * cfg.patch_y, dims=2)
    with torch.no_grad():
        z1 = ae.encode(u.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
        z2 = ae.encode(u_shifted.reshape(1, -1)).reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    rolled = torch.roll(z1, shift_sites, dims=2)
    assert not torch.allclose(rolled, z2, atol=1e-3)


def test_gradient_flow():
    ae, cfg = _make()
    u = torch.randn(2, cfg.NX * cfg.NY, requires_grad=True)
    u_hat, z = ae(u)
    loss = u_hat.pow(2).sum() + z.pow(2).sum()
    loss.backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    for name, p in ae.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_no_bias_output_projection():
    ae, cfg = _make(local_channels=3)
    assert ae.enc_out.bias is None


def test_rejects_n_sites_y_not_multiple_of_forcing_n():
    with pytest.raises(ValueError, match="multiple of forcing_n"):
        LocalFieldAutoencoderAniso2DConfig(NX=64, NY=64, n_sites_x=4, n_sites_y=8, forcing_n=3)


def test_rejects_invalid_forcing_n():
    with pytest.raises(ValueError, match="forcing_n"):
        LocalFieldAutoencoderAniso2DConfig(NX=64, NY=64, n_sites_x=4, n_sites_y=8, forcing_n=0)
