"""Tests for the plain-MLP autoencoder (`KSAutoencoderMLP`, "Track A"),
added 2026-08-29 as a ported alternative to the patched-transformer -- see
CLAUDE_CODE_BRIEF.md §5.1/5.2 addendum.
"""

from __future__ import annotations

import torch

from ks_latent.config import MLPAutoencoderConfig
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP


def test_encode_decode_shapes():
    cfg = MLPAutoencoderConfig(NX=32, hidden=(64, 32), d_latent=6)
    ae = KSAutoencoderMLP(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_forward_matches_encode_then_decode():
    cfg = MLPAutoencoderConfig(NX=16, hidden=(32, 16), d_latent=4)
    ae = KSAutoencoderMLP(cfg)
    ae.eval()
    u = torch.randn(3, 16)
    u_hat, z = ae(u)
    z2 = ae.encode(u)
    u_hat2 = ae.decode(z2)
    assert torch.allclose(z, z2)
    assert torch.allclose(u_hat, u_hat2)


def test_gradients_reach_both_encoder_and_decoder():
    cfg = MLPAutoencoderConfig(NX=16, hidden=(32, 16), d_latent=4)
    ae = KSAutoencoderMLP(cfg)
    u = torch.randn(8, 16)
    u_hat, z = ae(u)
    loss = ((u_hat - u) ** 2).mean()
    loss.backward()
    assert ae.enc_out.weight.grad is not None
    assert torch.any(ae.enc_out.weight.grad != 0)
    assert ae.dec_out.weight.grad is not None
    assert torch.any(ae.dec_out.weight.grad != 0)


def test_overfits_a_single_batch():
    torch.manual_seed(0)
    cfg = MLPAutoencoderConfig(NX=16, hidden=(32, 16), d_latent=8)
    ae = KSAutoencoderMLP(cfg)
    u = torch.randn(4, 16)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    for _ in range(500):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-4


def test_bottleneck_has_no_nonlinearity_applied_directly():
    """The encoder's final layer (`enc_out`) is a bare Linear -- z should be
    able to take values outside GELU's effective range without saturating."""
    cfg = MLPAutoencoderConfig(NX=8, hidden=(16,), d_latent=4)
    ae = KSAutoencoderMLP(cfg)
    assert isinstance(ae.enc_out, torch.nn.Linear)
    assert isinstance(ae.dec_out, torch.nn.Linear)
