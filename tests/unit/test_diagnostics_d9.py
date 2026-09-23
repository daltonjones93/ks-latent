"""Tests for D9 (real-trajectory smoothness -- user-directed 2026-09-05,
not part of the original brief; see `ks_latent.analysis.diagnostics`'s
`SmoothnessResult`/`smoothness_diagnostic` docstrings). Moved from a
one-off comparison script (`scripts/analyze_latent_smoothness.py`, built
2026-09-04 to quantify a GIF-visible "huge jumps" observation) into Gate 4
proper: "please run visualizations, smoothness diagnostic + Gate 3/4 (in
the future add smoothness diagnostic to gate 4)"."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ks_latent.analysis.diagnostics import (
    encoded_trajectory_smoothness,
    propagator_step_jacobian_full_spectrum,
    propagator_step_jacobian_spectral_norms,
    smoothness_diagnostic,
)
from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def test_encoded_trajectory_smoothness_zero_for_constant_trajectory():
    z0 = np.random.default_rng(0).standard_normal((5, 8))
    z_seq = np.repeat(z0[:, None, :], 6, axis=1)  # (5, 6, 8), constant over time
    result = encoded_trajectory_smoothness(z_seq)
    assert result["step_med"] == 0.0
    assert result["curv_med"] == 0.0
    assert result["step_norm_med"] == 0.0
    assert result["curv_norm_med"] == 0.0


def test_encoded_trajectory_smoothness_positive_for_random_trajectory():
    rng = np.random.default_rng(1)
    z_seq = rng.standard_normal((10, 20, 6))
    result = encoded_trajectory_smoothness(z_seq)
    assert result["step_med"] > 0.0
    assert result["curv_med"] > 0.0
    assert np.isfinite(result["step_norm_p95"])
    assert np.isfinite(result["curv_norm_p95"])


def test_propagator_jacobian_norms_markovian_mode():
    torch.manual_seed(0)
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="markovian", backbone="mlp", zero_init=False))
    z_seq = torch.randn(4, 15, 8)
    norms = propagator_step_jacobian_spectral_norms(prop, z_seq, n_samples=20, seed=0)
    assert norms.shape == (20,)
    assert (norms >= 0).all()
    assert np.isfinite(norms).all()


def test_propagator_jacobian_norms_history_mode():
    torch.manual_seed(1)
    prop = LatentPropagator(
        PropagatorConfig(d_latent=8, mode="history", backbone="mlp", n_history=2, zero_init=False)
    )
    z_seq = torch.randn(4, 15, 8)
    norms = propagator_step_jacobian_spectral_norms(prop, z_seq, n_samples=20, seed=0)
    assert norms.shape == (20,)
    assert np.isfinite(norms).all()


def test_propagator_jacobian_norms_rejects_two_step_mode():
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="two_step"))
    z_seq = torch.randn(4, 15, 8)
    with pytest.raises(ValueError, match="unsupported mode"):
        propagator_step_jacobian_spectral_norms(prop, z_seq, n_samples=5, seed=0)


def test_propagator_jacobian_full_spectrum_markovian_shape_and_descending():
    """Added 2026-09-10, Section 134 (propagator_graded_spectrum_shape_loss's
    calibration helper). Each sample's spectrum must be full (d entries,
    not just the top one) and sorted descending -- svdvals' own guarantee,
    checked directly here since the whole per-rank calibration in
    scripts/compute_reference_spectrum.py depends on it."""
    torch.manual_seed(0)
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="markovian", backbone="mlp", zero_init=False))
    z_seq = torch.randn(4, 15, 8)
    spectra = propagator_step_jacobian_full_spectrum(prop, z_seq, n_samples=20, seed=0)
    assert spectra.shape == (20, 8)
    assert np.isfinite(spectra).all()
    assert (spectra >= 0).all()
    for row in spectra:
        assert (np.diff(row) <= 1e-6).all()  # descending


def test_propagator_jacobian_full_spectrum_history_mode():
    torch.manual_seed(1)
    prop = LatentPropagator(
        PropagatorConfig(d_latent=8, mode="history", backbone="mlp", n_history=2, zero_init=False)
    )
    z_seq = torch.randn(4, 15, 8)
    spectra = propagator_step_jacobian_full_spectrum(prop, z_seq, n_samples=20, seed=0)
    assert spectra.shape == (20, 8)  # min(d, n_hist*d) = d
    assert np.isfinite(spectra).all()


def test_propagator_jacobian_full_spectrum_top_entry_matches_spectral_norms():
    """The existing (top-1-only) diagnostic's own output must equal
    column 0 of this function's full spectrum, at the same seed/samples --
    same underlying computation, just retaining more of it."""
    torch.manual_seed(2)
    prop = LatentPropagator(PropagatorConfig(d_latent=6, mode="markovian", backbone="mlp", zero_init=False))
    z_seq = torch.randn(4, 15, 6)
    top1 = propagator_step_jacobian_spectral_norms(prop, z_seq, n_samples=10, seed=0)
    full = propagator_step_jacobian_full_spectrum(prop, z_seq, n_samples=10, seed=0)
    assert np.allclose(top1, full[:, 0], atol=1e-5)


def test_propagator_jacobian_full_spectrum_rejects_two_step_mode():
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="two_step"))
    z_seq = torch.randn(4, 15, 8)
    with pytest.raises(ValueError, match="unsupported mode"):
        propagator_step_jacobian_full_spectrum(prop, z_seq, n_samples=5, seed=0)


def test_smoothness_diagnostic_full_pipeline_markovian():
    torch.manual_seed(2)
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="markovian", backbone="mlp", zero_init=False))
    z_seq_t = torch.randn(4, 15, 8)
    z_seq_np = z_seq_t.numpy()
    result = smoothness_diagnostic(z_seq_np, z_seq_t, prop, n_samples=20, seed=0)
    assert result.propagator_jacobian_supported is True
    assert result.propagator_jacobian_skip_reason == ""
    assert np.isfinite(result.propagator_jacobian_med)
    assert np.isfinite(result.step_med)
    assert np.isfinite(result.curv_med)


def test_smoothness_diagnostic_skips_gracefully_for_two_step_mode():
    """The graceful-skip path this whole wrapper exists for: an unsupported
    propagator mode must not abort the rest of Gate 4, just report
    'not computed' for the propagator-Jacobian half while still returning
    the encoder-side (propagator-independent) measures."""
    prop = LatentPropagator(PropagatorConfig(d_latent=8, mode="two_step"))
    z_seq_t = torch.randn(4, 15, 8)
    z_seq_np = z_seq_t.numpy()
    result = smoothness_diagnostic(z_seq_np, z_seq_t, prop, n_samples=20, seed=0)
    assert result.propagator_jacobian_supported is False
    assert "two_step" in result.propagator_jacobian_skip_reason or "unsupported" in result.propagator_jacobian_skip_reason
    assert np.isnan(result.propagator_jacobian_med)
    assert np.isfinite(result.step_med)  # encoder-side measures still computed
    assert np.isfinite(result.curv_med)
