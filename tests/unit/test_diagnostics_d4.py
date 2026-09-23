"""Validation for D4 (translation representation), brief §8: "Test on a
synthetic exactly equivariant encoder (truncated Fourier projection);
assert recovery to 1e-8."
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ks_latent.analysis.diagnostics import translation_representation


def _fourier_encoder_factory(NX: int, k_indices: list[int]):
    """encode(u) = concat([Re(u_hat(k)), Im(u_hat(k))] for k in k_indices).
    Exactly equivariant: shifting u by c rotates each (Re,Im) pair by the
    exact phase -2*pi*k*c/NX (the DFT shift theorem)."""
    n = torch.arange(NX, dtype=torch.float32)

    def encoder(u_batch: torch.Tensor) -> torch.Tensor:
        # u_batch: (B, NX)
        parts = []
        for k in k_indices:
            angle = 2.0 * torch.pi * k * n / NX
            re = (u_batch * torch.cos(angle)).sum(dim=-1)
            im = -(u_batch * torch.sin(angle)).sum(dim=-1)
            parts.append(re)
            parts.append(im)
        return torch.stack(parts, dim=-1)

    return encoder


def test_d4_exact_equivariant_encoder_recovers_rotation_group_to_1e_minus_8():
    NX = 64
    k_indices = [1, 3]
    encoder = _fourier_encoder_factory(NX, k_indices)

    torch.manual_seed(0)
    u_samples = torch.randn(30, NX)
    shifts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    result = translation_representation(encoder, u_samples, shifts, NX)

    # Residuals should be essentially exact (linear, exactly equivariant map).
    assert np.max(result.relative_residuals) < 1e-6

    # Group property R(c1)R(c2) ~= R(c1+c2) should hold to ~1e-8 (float32
    # matmul roundoff floor -- the map itself is exact).
    assert np.max(result.group_property_errors) < 1e-5

    # Eigenvalues should be unit modulus (a genuine rotation), and their
    # phase / c should recover the true physical wavenumbers 2*pi*k/NX.
    c = shifts[1]  # c=1
    eigvals = result.eigenvalues[1]
    assert np.allclose(np.abs(eigvals), 1.0, atol=1e-5)
    recovered_wavenumbers = sorted(np.abs(np.angle(eigvals)) / c)
    true_wavenumbers = sorted(
        [2 * np.pi * k / NX for k in k_indices] + [2 * np.pi * k / NX for k in k_indices]
    )
    for rec, true in zip(recovered_wavenumbers, true_wavenumbers):
        assert rec == pytest.approx(true, abs=1e-6)


def test_d4_non_equivariant_encoder_has_larger_residuals():
    """A generic (non-equivariant) linear map should show a measurably worse
    fit than the exact case above -- sanity check that the residual metric
    is actually discriminating."""
    NX, d = 32, 6
    torch.manual_seed(0)
    W = torch.randn(d, NX)  # a random, non-equivariant linear "encoder"

    def encoder(u_batch):
        return u_batch @ W.T

    u_samples = torch.randn(50, NX)
    shifts = [1, 2, 3]
    result = translation_representation(encoder, u_samples, shifts, NX)
    assert np.max(result.relative_residuals) > 1e-3
