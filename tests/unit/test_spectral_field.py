"""Tests for `ks_latent.models.spectral_field` (added 2026-09-06, see
docs/sine_transform_pde_plan.md): the fixed, non-learned rFFT
transform/derivative-synthesis helpers shared by
`KSAutoencoderSpectralField` and `_SpectralPDEDeltaBody`
(`backbone="spectral_pde"`).
"""

from __future__ import annotations

import math

import torch

from ks_latent.models.spectral_field import (
    decode_from_spectrum,
    encode_to_spectrum,
    rfft_wavenumbers,
    synthesize_derivatives,
)


def test_rfft_wavenumbers_shape_and_values():
    k = rfft_wavenumbers(5, L=100.0)
    assert k.shape == (5,)
    assert k[0].item() == 0.0
    expected = 2.0 * math.pi * torch.arange(5).float() / 100.0
    assert torch.allclose(k, expected)


def test_encode_decode_round_trip_full_spectrum():
    """No truncation (K = N_w//2+1): decode(encode(w)) should recover w
    exactly (up to float32 precision)."""
    torch.manual_seed(0)
    N_w = 32
    K = N_w // 2 + 1
    w = torch.randn(4, N_w)
    z = encode_to_spectrum(w, K)
    w_hat = decode_from_spectrum(z, K, N_w)
    assert torch.allclose(w_hat, w, atol=1e-5)


def test_decode_encode_round_trip_truncated_and_valid_z():
    """Truncated (K < N_w//2+1): decode(z) then re-encode should recover
    the ORIGINAL z exactly, provided z is a physically-valid spectrum (came
    from encode_to_spectrum of a real field -- see spectral_field.py's
    module docstring for the one caveat about a synthetic all-random z's
    DC-imaginary slot)."""
    torch.manual_seed(1)
    N_w, K = 32, 8
    w = torch.randn(3, N_w)
    z = encode_to_spectrum(w, K)  # physically valid: Im(mode 0) == 0 exactly
    w_hat = decode_from_spectrum(z, K, N_w)
    z2 = encode_to_spectrum(w_hat, K)
    assert torch.allclose(z2, z, atol=1e-5)


def test_dc_imaginary_slot_is_exactly_zero_for_any_real_field():
    """Index K of z (Im(mode 0)) must be exactly 0 for a real signal's rFFT
    -- the structural property the round-trip caveat rests on."""
    torch.manual_seed(2)
    N_w, K = 32, 8
    w = torch.randn(5, N_w)
    z = encode_to_spectrum(w, K)
    assert torch.allclose(z[:, K], torch.zeros(5), atol=1e-6)


def test_synthesize_derivatives_order_zero_matches_decode():
    torch.manual_seed(3)
    N_w, K = 32, 8
    w = torch.randn(2, N_w)
    z = encode_to_spectrum(w, K)
    derivs = synthesize_derivatives(z, K, N_w, L=100.0, max_order=3)
    assert derivs.shape == (2, N_w, 4)
    w_hat = decode_from_spectrum(z, K, N_w)
    assert torch.allclose(derivs[:, :, 0], w_hat, atol=1e-5)


def test_synthesize_derivatives_matches_analytic_sine_wave():
    """Strong correctness check: a single-mode sine/cosine combination has
    hand-computable derivatives of every order -- verify the spectral
    synthesis matches them exactly (up to float32 precision), independent
    of any trained model."""
    N_w = 64
    L = 100.0
    K = 8
    x = torch.linspace(0, L, N_w + 1)[:-1]
    m3, m5 = 3, 5
    k3, k5 = 2 * math.pi * m3 / L, 2 * math.pi * m5 / L
    w = torch.sin(k3 * x) + 0.5 * torch.cos(k5 * x)

    z = encode_to_spectrum(w.unsqueeze(0), K)
    derivs = synthesize_derivatives(z, K, N_w, L, max_order=4)

    def analytic(order: int) -> torch.Tensor:
        # d^n/dx^n [sin(k3 x)] and d^n/dx^n [0.5*cos(k5 x)], cycling
        # sin<->cos with sign flips every derivative.
        cyc = order % 4
        sin_term = {
            0: torch.sin(k3 * x), 1: k3 * torch.cos(k3 * x),
            2: -(k3**2) * torch.sin(k3 * x), 3: -(k3**3) * torch.cos(k3 * x),
        }[cyc] * (k3 ** (order - cyc) if order >= 4 else 1.0)
        cos_term = {
            0: 0.5 * torch.cos(k5 * x), 1: -0.5 * k5 * torch.sin(k5 * x),
            2: -0.5 * (k5**2) * torch.cos(k5 * x), 3: 0.5 * (k5**3) * torch.sin(k5 * x),
        }[cyc] * (k5 ** (order - cyc) if order >= 4 else 1.0)
        return sin_term + cos_term

    for order in range(5):
        expected = analytic(order)
        got = derivs[0, :, order]
        assert torch.allclose(got, expected, atol=1e-3), f"order={order} mismatch"


def test_encode_to_spectrum_drops_high_modes():
    """Truncation is a HARD cutoff -- energy beyond K contributes nothing
    to z (by construction, since it's simply sliced off)."""
    torch.manual_seed(4)
    N_w = 32
    w = torch.randn(2, N_w)
    z_k4 = encode_to_spectrum(w, K=4)
    z_k8 = encode_to_spectrum(w, K=8)
    assert z_k4.shape == (2, 8)
    assert z_k8.shape == (2, 16)
    # the first 4 real + first 4 imaginary entries of z_k8 must match z_k4
    assert torch.allclose(z_k8[:, :4], z_k4[:, :4])
    assert torch.allclose(z_k8[:, 8:12], z_k4[:, 4:8])
