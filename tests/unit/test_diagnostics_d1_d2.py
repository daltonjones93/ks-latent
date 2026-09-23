"""Validation for D1 (sensitivity maps) and D2 (wavenumber content), brief §8."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ks_latent.analysis.diagnostics import decoder_sensitivity_map, wavenumber_content
from ks_latent.utils.circstats import circular_distance


def _circular_dist_phys(x: torch.Tensor, c: float, L: float) -> torch.Tensor:
    d = torch.abs(x - c) % L
    return torch.minimum(d, L - d)


def _bump_decoder(d: int, NX: int, L: float, centers: list[float], width: float):
    x = torch.arange(NX, dtype=torch.float32) * L / NX
    bumps = torch.stack([torch.exp(-0.5 * (_circular_dist_phys(x, c, L) / width) ** 2) for c in centers])  # (d, NX)

    def decoder_single(z: torch.Tensor) -> torch.Tensor:
        return z @ bumps

    return decoder_single, bumps


def test_d1_sensitivity_map_recovers_known_bump_centers():
    L, NX, d = 100.0, 256, 4
    centers = [10.0, 40.0, 70.0, 95.0]  # last one straddles the wrap point
    decoder_single, _ = _bump_decoder(d, NX, L, centers, width=3.0)

    torch.manual_seed(0)
    z_samples = torch.randn(20, d)
    result = decoder_sensitivity_map(decoder_single, z_samples, L)

    h = L / NX
    for k, true_center in enumerate(centers):
        recovered = result.per_latent[k].centroid
        dist = min(abs(recovered - true_center), L - abs(recovered - true_center))
        assert dist < h, f"latent {k}: recovered {recovered:.3f}, true {true_center}, off by {dist:.3f} (grid h={h:.3f})"
        assert result.per_latent[k].resultant_length > 0.9  # tightly localized


def test_d1_order_by_centroid_sorts_correctly():
    L, NX, d = 100.0, 256, 3
    centers = [80.0, 10.0, 45.0]
    decoder_single, _ = _bump_decoder(d, NX, L, centers, width=3.0)
    torch.manual_seed(0)
    z_samples = torch.randn(10, d)
    result = decoder_sensitivity_map(decoder_single, z_samples, L)
    ordered_centroids = [result.per_latent[i].centroid for i in result.order_by_centroid]
    assert ordered_centroids == sorted(ordered_centroids)


def test_d2_spectral_centroid_recovers_known_wavenumber():
    L, NX = 100.0, 512
    x = np.arange(NX) * L / NX
    k0 = 2 * np.pi * 5 / L  # a pure mode at wavenumber index 5
    S = np.stack([np.cos(k0 * x), np.cos(k0 * x)])  # (d=2, NX), identical rows
    result = wavenumber_content(S, L)
    assert result.spectral_centroid[0] == pytest.approx(k0, rel=1e-6)
    assert result.spectral_bandwidth[0] < 1e-6  # a pure tone has zero bandwidth


def test_d2_narrow_spatial_bump_has_broad_spectrum():
    L, NX = 100.0, 512
    x = np.arange(NX) * L / NX
    narrow = np.exp(-0.5 * (np.minimum(np.abs(x - 50), L - np.abs(x - 50)) / 1.0) ** 2)
    broad = np.exp(-0.5 * (np.minimum(np.abs(x - 50), L - np.abs(x - 50)) / 15.0) ** 2)
    S = np.stack([narrow, broad])
    result = wavenumber_content(S, L)
    assert result.spectral_bandwidth[0] > result.spectral_bandwidth[1]


def test_d2_wavelet_entropy_lower_for_localized_signal():
    L, NX = 100.0, 512
    rng = np.random.default_rng(0)
    x = np.arange(NX) * L / NX
    localized = np.exp(-0.5 * (np.minimum(np.abs(x - 50), L - np.abs(x - 50)) / 1.0) ** 2)
    noise = rng.normal(size=NX)
    S = np.stack([localized, noise])
    result = wavenumber_content(S, L)
    assert result.wavelet_energy_entropy[0] < result.wavelet_energy_entropy[1]
