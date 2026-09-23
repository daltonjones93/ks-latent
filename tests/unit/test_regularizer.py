"""Tests for the optional banded latent-index-smoothness + off-band
decorrelation regularizer (ks_latent/training/regularizer.py), ported
2026-08-29 from a reference implementation. See CLAUDE_CODE_BRIEF.md §5.1
"Ported improvements" addendum.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ks_latent.config import RegConfig
from ks_latent.training.regularizer import (
    BandedSmoothness,
    LatentIndexPenalty,
    OffBandDecorrelation,
    laplacian_B,
    offband_mask,
    psd_report,
)


def test_reg_config_defaults_off():
    cfg = RegConfig()
    assert cfg.lambda_z == 0.0
    assert cfg.lambda_decorr == 0.0


def test_laplacian_B_is_symmetric_psd_and_banded():
    B = laplacian_B(10, bandwidth=3, decay=0.5)
    assert np.allclose(B, B.T)
    report = psd_report(B)
    assert report["min_eig"] > -1e-8
    # banded: zero beyond bandwidth 3
    idx = np.arange(10)
    dist = np.abs(idx[:, None] - idx[None, :])
    assert np.allclose(B[dist > 3], 0.0)


def test_laplacian_B_null_space_is_constant_vector():
    """The Laplacian's global minimizer of z^T B z is the constant vector --
    documented in the module as the reason the off-band decorrelation
    counterpart is required."""
    d = 8
    B = laplacian_B(d, bandwidth=3, decay=0.5)
    ones = np.ones(d)
    assert ones @ B @ ones == pytest.approx(0.0, abs=1e-8)


def test_banded_smoothness_penalizes_index_roughness():
    torch.manual_seed(0)
    cfg = RegConfig(lambda_z=1.0, bandwidth=2, decay=0.5)
    reg = BandedSmoothness(cfg, d_latent=6)
    smooth = torch.linspace(0, 1, 6).unsqueeze(0)  # smooth in index
    rough = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])  # alternating
    assert reg(smooth).item() < reg(rough).item()


def test_banded_smoothness_zero_for_constant_vector():
    cfg = RegConfig(lambda_z=1.0, bandwidth=2, decay=0.5)
    reg = BandedSmoothness(cfg, d_latent=6)
    z = torch.full((4, 6), 3.0)
    assert reg(z).item() == pytest.approx(0.0, abs=1e-6)


def test_banded_smoothness_accepts_windowed_input():
    cfg = RegConfig(lambda_z=1.0, bandwidth=2, decay=0.5)
    reg = BandedSmoothness(cfg, d_latent=5)
    z2d = torch.randn(4, 5)
    z3d = z2d.unsqueeze(1)  # (batch, window=1, d_latent)
    assert reg(z2d).item() == pytest.approx(reg(z3d).item(), abs=1e-6)


def test_offband_mask_excludes_within_bandwidth():
    mask = offband_mask(6, bandwidth=2)
    idx = np.arange(6)
    dist = np.abs(idx[:, None] - idx[None, :])
    assert not mask[dist <= 2].any()
    assert mask[dist > 2].all()


def test_offband_decorrelation_worst_case_is_full_collapse():
    """Collapse (every coordinate identical across the batch) drives every
    off-band pair to correlation 1, the term's documented maximum."""
    torch.manual_seed(0)
    d = 8
    dec = OffBandDecorrelation(d, bandwidth=2)
    base = torch.randn(64, 1)
    collapsed = base.expand(64, d).clone()
    diverse = torch.randn(64, d)
    assert dec(collapsed).item() == pytest.approx(1.0, abs=1e-4)
    assert dec(diverse).item() < dec(collapsed).item()


def test_offband_decorrelation_rejects_bandwidth_too_large():
    with pytest.raises(ValueError):
        OffBandDecorrelation(4, bandwidth=4)  # no pairs with |i-k| > 4 at d=4


def test_latent_index_penalty_inactive_when_weights_zero():
    cfg = RegConfig(lambda_z=0.0, lambda_decorr=0.0)
    pen = LatentIndexPenalty(cfg, d_latent=6)
    assert not pen.active
    z = torch.randn(4, 6)
    assert pen(z).item() == 0.0


def test_latent_index_penalty_active_and_combines_both_terms():
    cfg = RegConfig(lambda_z=0.5, lambda_decorr=0.25, bandwidth=2)
    pen = LatentIndexPenalty(cfg, d_latent=6)
    assert pen.active
    z = torch.randn(32, 6, requires_grad=True)
    loss = pen(z)
    assert loss.item() >= 0.0
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
