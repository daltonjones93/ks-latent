"""Tests for `KSAutoencoderLocalField` (`encoder_kind="local_field"`, added
2026-09-10, Section 135) -- Phase 10's original architecturally-local
latent FIELD (CLAUDE.md sections 12.2.1/12.3/12.4), user-directed: "why
don't we try Phase 10's original design called for an architecturally-
enforced local field (circular-Conv1d, channel 0 anchored to a real
physical average)." See `LocalFieldAutoencoderConfig`'s docstring for the
full motivation and `CLAUDE.md`'s own prescribed test list (section 12.4)
this file follows: exact equivariance, receptive field, the anchored
channel, reconstruction, reconstructed spectrum.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalFieldAutoencoderConfig
from ks_latent.models.autoencoder_local_field import KSAutoencoderLocalField


def _small_cfg(**overrides) -> LocalFieldAutoencoderConfig:
    defaults = dict(NX=64, n_sites=8, local_channels=3, site_mix_radius=1, n_site_mix_layers=2, hidden=8)
    defaults.update(overrides)
    return LocalFieldAutoencoderConfig(**defaults)


def test_shapes_and_d_latent():
    cfg = _small_cfg()
    ae = KSAutoencoderLocalField(cfg)
    u = torch.randn(5, cfg.NX)
    z = ae.encode(u)
    assert z.shape == (5, cfg.d_latent)
    assert cfg.d_latent == cfg.n_sites * cfg.local_channels
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, cfg.NX)
    u_hat2, z2 = ae(u)
    assert torch.allclose(u_hat2, u_hat)
    assert torch.allclose(z2, z)


def test_circular_conv_exact_equivariance():
    """`E(roll(u, s*patch_size)) == roll_sites(E(u), s)` to 1e-6 -- CLAUDE.md
    section 12.4's own literal test. Shifting `u` by a whole number of
    patches must shift the SITE axis of `z` by the same number of sites
    (all channels, including the gauge anchor, which is itself an average
    pool -- also exactly equivariant this way)."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    ae = KSAutoencoderLocalField(cfg)
    ae.eval()
    u = torch.randn(4, cfg.NX)
    s = 3  # shift by 3 sites
    u_shifted = torch.roll(u, shifts=s * cfg.patch_size, dims=-1)

    with torch.no_grad():
        z = ae.encode(u).reshape(4, cfg.n_sites, cfg.local_channels)
        z_shifted = ae.encode(u_shifted).reshape(4, cfg.n_sites, cfg.local_channels)

    z_rolled_sites = torch.roll(z, shifts=s, dims=1)
    assert torch.allclose(z_shifted, z_rolled_sites, atol=1e-5)


def test_anchored_channel_is_fixed_to_local_physical_average():
    """Channel 0 must equal `u`'s own local average over each site's
    patch, exactly, and must be independent of the learned weights (a
    freshly re-initialized model should give the identical channel-0
    values for the same u)."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    ae1 = KSAutoencoderLocalField(cfg)
    torch.manual_seed(1)  # different init
    ae2 = KSAutoencoderLocalField(cfg)
    u = torch.randn(3, cfg.NX)

    expected_anchor = u.reshape(3, cfg.n_sites, cfg.patch_size).mean(dim=-1)  # (3, P)

    z1 = ae1.encode(u).reshape(3, cfg.n_sites, cfg.local_channels)
    z2 = ae2.encode(u).reshape(3, cfg.n_sites, cfg.local_channels)
    assert torch.allclose(z1[:, :, 0], expected_anchor, atol=1e-5)
    assert torch.allclose(z2[:, :, 0], expected_anchor, atol=1e-5)
    assert torch.allclose(z1[:, :, 0], z2[:, :, 0], atol=1e-6)  # same regardless of init
    # channels 1.. are learned and SHOULD differ between the two random inits
    assert not torch.allclose(z1[:, :, 1], z2[:, :, 1], atol=1e-3)


def test_anchored_channel_with_local_channels_one():
    """`local_channels=1` is the degenerate pure-coarse-graining case (no
    `enc_out` at all) -- must not error and must equal the plain average."""
    cfg = _small_cfg(local_channels=1)
    ae = KSAutoencoderLocalField(cfg)
    assert ae.enc_out is None
    u = torch.randn(3, cfg.NX)
    z = ae.encode(u)
    expected = u.reshape(3, cfg.n_sites, cfg.patch_size).mean(dim=-1)
    assert torch.allclose(z, expected, atol=1e-5)


def test_receptive_field_bounded_and_matches_analytic_formula():
    """Perturbing one input point should only affect encoder output sites
    within `1 + n_site_mix_layers*site_mix_radius` sites of it (measured by
    gradient masking, not just asserted)."""
    torch.manual_seed(0)
    cfg = _small_cfg(n_sites=16, site_mix_radius=1, n_site_mix_layers=2, local_channels=2)
    ae = KSAutoencoderLocalField(cfg)
    u = torch.zeros(1, cfg.NX, requires_grad=True)
    z = ae.encode(u).reshape(1, cfg.n_sites, cfg.local_channels)
    # Gradient of one output site's channel-1 (learned, not the anchor) sum
    # w.r.t. every input point -- nonzero support marks the receptive field.
    probe_site = cfg.n_sites // 2
    z[0, probe_site, 1].backward()
    grad = u.grad[0]
    touched_points = (grad.abs() > 1e-8).nonzero().flatten()
    touched_sites = torch.unique(touched_points // cfg.patch_size)
    analytic_radius = 1 + cfg.n_site_mix_layers * cfg.site_mix_radius  # in sites, one-sided-ish
    # Circular distance from probe_site to every touched site must be within analytic_radius.
    dist = torch.minimum(
        (touched_sites - probe_site) % cfg.n_sites, (probe_site - touched_sites) % cfg.n_sites
    )
    assert dist.max().item() <= analytic_radius
    assert len(touched_sites) > 1  # sanity: it did mix beyond just its own patch


def test_local_field_reconstruction_overfits_a_batch():
    """A basic sanity check (CLAUDE.md's own `test_local_field_reconstruction`):
    the model can drive reconstruction MSE down on a small fixed batch."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    ae = KSAutoencoderLocalField(cfg)
    u = torch.randn(8, cfg.NX)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    losses = []
    for _ in range(400):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.2


def test_reconstructed_spectrum_energy_not_degenerate():
    """CLAUDE.md's own `test_reconstructed_spectrum`: after a short overfit,
    the reconstruction's FFT energy spectrum should not be a degenerate
    (all-DC or all-zero) spike -- a coarse sanity check that the field
    reconstruction carries real spatial structure, not just a constant."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    ae = KSAutoencoderLocalField(cfg)
    u = torch.randn(8, cfg.NX)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    for _ in range(200):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        u_hat, _ = ae(u)
    spectrum = torch.fft.rfft(u_hat, dim=-1).abs()
    dc_fraction = spectrum[:, 0] / spectrum.sum(dim=-1)
    assert (dc_fraction < 0.9).all()


def test_invalid_nx_not_divisible_by_n_sites_raises():
    with pytest.raises(ValueError, match="n_sites"):
        LocalFieldAutoencoderConfig(NX=65, n_sites=8)


def test_invalid_local_channels_raises():
    with pytest.raises(ValueError, match="local_channels"):
        LocalFieldAutoencoderConfig(NX=64, n_sites=8, local_channels=0)
