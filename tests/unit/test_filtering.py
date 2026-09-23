"""Tests for the Gaussian low-pass filter (brief §3.3). Ground rule 1: this
had zero coverage until Phase 3's coverage pass caught it -- validated here
against the exact known per-mode attenuation before it is ever used on real
data in Phase 8/10."""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.solver.filtering import gaussian_lowpass


def test_gaussian_lowpass_attenuates_single_mode_by_known_factor():
    L, NX, ell = 100.0, 256, 3.0
    x = np.arange(NX) * L / NX
    k1 = 2 * np.pi / L  # smallest nonzero wavenumber
    u = np.cos(k1 * x)
    filtered = gaussian_lowpass(u, L, ell)
    expected_factor = np.exp(-0.5 * (k1 * ell) ** 2)
    assert np.allclose(filtered, expected_factor * u, atol=1e-10)


def test_gaussian_lowpass_preserves_mean():
    L, NX, ell = 100.0, 256, 5.0
    rng = np.random.default_rng(0)
    u = rng.normal(size=NX)
    filtered = gaussian_lowpass(u, L, ell)
    assert filtered.mean() == pytest.approx(u.mean(), abs=1e-10)


def test_gaussian_lowpass_zero_width_is_identity():
    L, NX = 100.0, 128
    rng = np.random.default_rng(1)
    u = rng.normal(size=NX)
    filtered = gaussian_lowpass(u, L, ell=0.0)
    assert np.allclose(filtered, u, atol=1e-10)


def test_gaussian_lowpass_large_width_suppresses_high_modes():
    L, NX = 100.0, 256
    x = np.arange(NX) * L / NX
    k_high = 2 * np.pi * (NX // 4) / L
    u = np.cos(k_high * x)
    filtered = gaussian_lowpass(u, L, ell=10.0)
    assert np.abs(filtered).max() < 1e-6 * np.abs(u).max()


def test_gaussian_lowpass_batched_shape():
    L, NX, ell = 100.0, 64, 2.0
    rng = np.random.default_rng(2)
    u = rng.normal(size=(5, NX))
    filtered = gaussian_lowpass(u, L, ell)
    assert filtered.shape == (5, NX)
