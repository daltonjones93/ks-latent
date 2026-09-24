"""Tests for Gaspari-Cohn latent-field localization (brief §15, Phase 13;
Part 4.3 of `docs/LITERATURE_REVIEW_AND_FINDINGS.md`)."""

from __future__ import annotations

import torch
import pytest

from ks_latent.da.localization import build_latent_taper_matrix, gaspari_cohn_taper
from ks_latent.da.sec import apply_fixed_taper


def test_gaspari_cohn_at_zero_is_one():
    assert gaspari_cohn_taper(torch.tensor(0.0), c=1.0).item() == pytest.approx(1.0)


def test_gaspari_cohn_zero_beyond_support_radius():
    c = 2.0
    assert gaspari_cohn_taper(torch.tensor(2 * c, dtype=torch.float64), c).item() == pytest.approx(0.0, abs=1e-9)
    assert gaspari_cohn_taper(torch.tensor(3 * c, dtype=torch.float64), c).item() == pytest.approx(0.0, abs=1e-9)


def test_gaspari_cohn_continuous_at_branch_boundary():
    """The two polynomial branches must agree exactly at r=c (r=1 in the
    function's own dimensionless units) -- a real discontinuity there
    would make the taper an invalid (non-smooth) correlation function."""
    c = 1.7
    left = gaspari_cohn_taper(torch.tensor(c - 1e-6), c)
    right = gaspari_cohn_taper(torch.tensor(c + 1e-6), c)
    assert left.item() == pytest.approx(right.item(), abs=1e-4)


def test_gaspari_cohn_monotonically_decreasing():
    c = 1.0
    r = torch.linspace(0.0, 2.0, 200, dtype=torch.float64)
    vals = gaspari_cohn_taper(r, c)
    diffs = vals[1:] - vals[:-1]
    assert (diffs <= 1e-9).all()


def test_gaspari_cohn_vectorized_matches_scalar():
    c = 1.3
    r = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    vec = gaspari_cohn_taper(r, c)
    for i in range(r.shape[0]):
        scalar = gaspari_cohn_taper(r[i], c)
        assert vec[i].item() == pytest.approx(scalar.item())


def test_gaspari_cohn_raises_on_nonpositive_c():
    with pytest.raises(ValueError, match="c must be"):
        gaspari_cohn_taper(torch.tensor(1.0), c=0.0)


def test_taper_matrix_shape_and_diagonal():
    n_sites, channels = 8, 3
    taper = build_latent_taper_matrix(n_sites, channels, c=2.0)
    d = n_sites * channels
    assert taper.shape == (d, d)
    assert torch.allclose(torch.diagonal(taper), torch.ones(d, dtype=torch.float64))


def test_taper_matrix_symmetric():
    taper = build_latent_taper_matrix(6, 2, c=1.5)
    assert torch.allclose(taper, taper.T)


def test_taper_matrix_respects_site_major_flattening():
    """Every entry within the SAME site (channel varies, site fixed) must
    be exactly 1.0 (zero self-distance); every pair of sites must use the
    SAME taper weight regardless of which channel pair is compared --
    directly verifies the site(i) = i // channels convention this module's
    docstring claims to respect."""
    n_sites, channels, c = 5, 3, 1.0
    taper = build_latent_taper_matrix(n_sites, channels, c)
    # Site 0 (indices 0,1,2) vs site 1 (indices 3,4,5): every cross pair
    # must share the same value.
    site0 = [0, 1, 2]
    site1 = [3, 4, 5]
    vals = {taper[i, j].item() for i in site0 for j in site1}
    assert len(vals) == 1
    # Within site 0, every entry (including off-diagonal channel pairs)
    # must be exactly 1.0 -- same site, zero site-distance.
    for i in site0:
        for j in site0:
            assert taper[i, j].item() == pytest.approx(1.0)


def test_taper_matrix_is_circular_wraps_around():
    """Site 0 and the LAST site must be treated as neighbours (circular
    distance 1, not n_sites-1) -- both KS and L96 are periodic domains."""
    n_sites, channels, c = 10, 1, 2.0
    taper = build_latent_taper_matrix(n_sites, channels, c)
    # linear distance between site 0 and site 9 is 9; circular is 1.
    expected = gaspari_cohn_taper(torch.tensor(1.0), c).item()
    assert taper[0, n_sites - 1].item() == pytest.approx(expected)


def test_taper_matrix_positive_semidefinite():
    """The core correctness requirement (brief's own
    test_gaspari_cohn_positive_definite spec): a valid correlation taper
    must be PSD, so a Schur product with any PSD covariance stays PSD
    (Schur product theorem) -- the entire reason this construction is
    safe to multiply into an ensemble covariance."""
    taper = build_latent_taper_matrix(16, 3, c=3.0)
    eigvals = torch.linalg.eigvalsh(taper)
    assert eigvals.min().item() > -1e-8


def test_taper_matrix_local_window_actually_localizes():
    """A small c should leave far-apart sites at essentially zero taper
    weight -- the whole point of localization."""
    n_sites, channels = 32, 2
    taper = build_latent_taper_matrix(n_sites, channels, c=1.0)  # support radius 2 sites
    # Site 0 vs site 16 (opposite side of a 32-site ring): circular distance 16, way outside support.
    assert taper[0, 16 * channels].item() == pytest.approx(0.0, abs=1e-10)


def test_apply_fixed_taper_reuses_sec_mechanism_correctly():
    """build_latent_taper_matrix is meant to plug directly into
    ks_latent.da.sec.apply_fixed_taper (same Schur-product mechanism SEC's
    fixed empirical taper already uses) -- verify the composition end to
    end against a hand-checked case: a covariance of all-1.0 entries,
    tapered, should equal the taper matrix itself."""
    taper = build_latent_taper_matrix(4, 1, c=1.0)
    B = torch.ones(4, 4, dtype=torch.float64)
    localized = apply_fixed_taper(B, taper)
    assert torch.allclose(localized, taper)
