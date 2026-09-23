"""Tests for `backbone="fno_mlp"` (added 2026-09-03, user-directed: "a
FNO based MLP might work better for the propagator" -- direct follow-up
to docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 66's finding that the
latent index already exhibits genuine periodic structure under
`w_spatial_signed` training). See `PropagatorConfig`'s `backbone=
"fno_mlp"` docstring section for the full architecture: FNO spectral-conv
layers + per-token `TokenMLPBlock`s (no attention), no positional
encoding -- genuinely translation-equivariant end to end.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.autoencoder_vit import SpectralConv1d
from ks_latent.models.propagator import LatentPropagator


def test_fno_mlp_backbone_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=44, mode="markovian", backbone="fno_mlp",
        n_tokens=4, token_d_model=32, token_n_layers=2, token_mlp_ratio=4,
        fno_modes=None, fno_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(5, 44)
    out = prop.step_one(z)
    assert out.shape == (5, 44)


def test_fno_mlp_is_identity_at_init():
    cfg = PropagatorConfig(d_latent=20, mode="markovian", backbone="fno_mlp", n_tokens=4)
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 20)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-6)


def test_fno_mlp_is_translation_equivariant():
    torch.manual_seed(0)
    cfg = PropagatorConfig(
        d_latent=44, mode="markovian", backbone="fno_mlp",
        n_tokens=4, token_d_model=32, token_n_layers=2, fno_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    # Perturb the zero-initialized output head so the delta isn't trivially
    # zero (an all-zero map is trivially equivariant and would not exercise
    # the claim).
    with torch.no_grad():
        prop.body.token_unembed.weight.add_(0.05 * torch.randn_like(prop.body.token_unembed.weight))
        prop.body.token_unembed.bias.add_(0.05 * torch.randn_like(prop.body.token_unembed.bias))
    prop.eval()

    z = torch.randn(4, 44)
    chunk = 44 // 4
    out = prop.step_one(z)
    out_shifted_input = prop.step_one(torch.roll(z, shifts=chunk, dims=1))
    out_rolled = torch.roll(out, shifts=chunk, dims=1)
    assert torch.allclose(out_shifted_input, out_rolled, atol=1e-5)
    assert out.abs().max().item() > 1e-3  # sanity: the perturbation actually did something


def test_fno_mlp_backbone_requires_divisible_d_latent():
    with pytest.raises(ValueError, match="must be divisible"):
        PropagatorConfig(mode="markovian", backbone="fno_mlp", d_latent=10, n_tokens=3)


def test_fno_mlp_requires_positive_fno_n_layers():
    with pytest.raises(ValueError, match="fno_n_layers"):
        PropagatorConfig(mode="markovian", backbone="fno_mlp", fno_n_layers=0)


def test_two_step_with_fno_mlp_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="fno_mlp")


def test_spectral_conv1d_works_under_bfloat16_autocast():
    """Regression test (added 2026-09-03, found via a real crash launching
    the first fno_mlp Stage 2 run under --amp): torch.fft.rfft/irfft raise
    outright on a bfloat16 input ('Unsupported dtype BFloat16') rather than
    autocast silently upcasting them the way softmax/layer_norm are --
    same bug class as logdet_barrier_loss's slogdet (test_losses_
    anticollapse.py). SpectralConv1d (used by both backbone='fno_vit' and
    'fno_mlp') must explicitly cast to float32 itself."""
    torch.manual_seed(3)
    conv = SpectralConv1d(in_channels=6, out_channels=6, modes=3)
    x = torch.randn(4, 6, 8, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = conv(x.to(torch.bfloat16))
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_fno_mlp_stage2_step_works_under_bfloat16_autocast():
    """End-to-end version of the regression above: a full fno_mlp
    propagator step under --amp-style autocast, the actual path that
    crashed."""
    cfg = PropagatorConfig(
        d_latent=44, mode="markovian", backbone="fno_mlp",
        n_tokens=4, token_d_model=32, token_n_layers=2, fno_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 44)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_one(z)
    assert torch.isfinite(out).all()
