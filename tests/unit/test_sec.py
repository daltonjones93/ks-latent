"""Tests for distance-free latent localization (brief §9): SEC (Anderson
2012) and the fixed empirical taper.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ks_latent.da.sec import (
    apply_fixed_taper,
    apply_sec_correction,
    build_sec_table,
    fixed_empirical_taper,
    sec_localize_covariance,
)


def test_sec_shrinks_more_at_small_ensemble_size():
    """The core SEC claim: observing the same sample correlation should be
    corrected *more* toward zero at small N than at large N, since a large
    |r| at small N is more often noise than at large N."""
    table_small = build_sec_table(n_ens=10, n_trials=3000, seed=0)
    table_large = build_sec_table(n_ens=1000, n_trials=3000, seed=0)

    r = np.array([0.5])
    corrected_small = apply_sec_correction(r, table_small)[0]
    corrected_large = apply_sec_correction(r, table_large)[0]

    assert abs(corrected_small) < abs(corrected_large)
    assert corrected_large == pytest.approx(0.5, abs=0.05)  # large N: negligible correction
    assert corrected_small < 0.4  # small N: real shrinkage


def test_sec_correction_is_antisymmetric_in_sign():
    table = build_sec_table(n_ens=15, n_trials=2000, seed=1)
    pos = apply_sec_correction(np.array([0.6]), table)[0]
    neg = apply_sec_correction(np.array([-0.6]), table)[0]
    assert pos == pytest.approx(-neg, abs=0.05)


def test_sec_correction_near_zero_at_zero_correlation():
    table = build_sec_table(n_ens=20, n_trials=2000, seed=2)
    corrected = apply_sec_correction(np.array([0.0]), table)[0]
    assert corrected == pytest.approx(0.0, abs=0.05)


def test_sec_localize_covariance_shrinks_spurious_small_ensemble_correlation():
    """A TRUE diagonal covariance (all off-diagonal population correlations
    are exactly 0), sampled with a small ensemble, gives a noisy sample
    covariance with spurious off-diagonal entries; SEC localization should
    pull those back toward zero relative to the raw sample estimate."""
    torch.manual_seed(0)
    d, n_ens = 6, 12
    Z = torch.randn(n_ens, d, dtype=torch.float64)  # true corr = I exactly
    B_raw = torch.cov(Z.T)

    table = build_sec_table(n_ens=n_ens, n_trials=3000, seed=3)
    B_sec = sec_localize_covariance(B_raw, table)

    off_diag_raw = (B_raw - torch.diag(torch.diagonal(B_raw))).abs().mean()
    off_diag_sec = (B_sec - torch.diag(torch.diagonal(B_sec))).abs().mean()
    assert off_diag_sec < off_diag_raw
    # Diagonal (variances) should be essentially untouched.
    assert torch.allclose(torch.diagonal(B_sec), torch.diagonal(B_raw), rtol=1e-6)


def test_fixed_empirical_taper_recovers_known_correlation_structure():
    """A large archive drawn from a KNOWN block structure (two independent
    correlated pairs) should give a taper that's large within each true
    pair and small (near the archive's own sampling noise floor) across
    pairs that are truly independent."""
    torch.manual_seed(0)
    n_archive = 20000
    rho = 0.8
    block = torch.tensor([[1.0, rho], [rho, 1.0]])
    cov = torch.block_diag(block, block)  # (4,4): {0,1} correlated, {2,3} correlated, cross-block independent
    L = torch.linalg.cholesky(cov)
    archive = torch.randn(n_archive, 4) @ L.T

    taper = fixed_empirical_taper(archive)
    assert taper[0, 1] == pytest.approx(rho, abs=0.03)
    assert taper[2, 3] == pytest.approx(rho, abs=0.03)
    assert taper[0, 2] < 0.05  # truly independent pair: taper near zero
    assert taper[0, 3] < 0.05


def test_apply_fixed_taper_is_elementwise_product():
    B = torch.tensor([[4.0, 2.0], [2.0, 9.0]])
    taper = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    result = apply_fixed_taper(B, taper)
    assert torch.allclose(result, torch.tensor([[4.0, 1.0], [1.0, 9.0]]))
