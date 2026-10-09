"""Tests for `backbone="local_mlp_2d_biotsavart"` (`_LocalMLPDeltaBody2DBiotSavart`,
`_biot_savart_velocity`, added 2026-10-01, D2 of docs/oct_1_exp.md --
user-directed: "I'm trying to improve transfer by creating dynamics
between localized data"). Mirrors `test_propagator_local_mlp_2d_aniso.py`'s
discipline, plus the ground-rule-1 numerics check specific to this
backbone: the elliptic solve itself must match an analytic answer before
anything about training is trusted (`test_biot_savart_matches_analytic_taylor_green`).
"""

from __future__ import annotations

import math

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models import load_propagator_checkpoint_resized_2d_biotsavart
from ks_latent.models.propagator import AuxPropagator, LatentPropagator, _biot_savart_velocity


def _cfg(cls=PropagatorConfig, **overrides):
    defaults = dict(
        d_latent=192, mode="markovian", backbone="local_mlp_2d_biotsavart", attn_window=2,
        local_mlp2d_n_sites_x=8, local_mlp2d_n_sites_y=8,
        local_mlp2d_local_channels=3, local_mlp2d_token_d_model=32,
        local_mlp2d_n_layers=2, biot_savart_Lx=2 * math.pi, biot_savart_Ly=2 * math.pi,
    )
    defaults.update(overrides)
    return cls(**defaults)


def test_shape_and_finite():
    cfg = _cfg()
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    cfg = _cfg(zero_init=True)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-5)


def test_exact_equivariance_under_site_shift():
    """Translating the input field by whole patches must translate the
    output identically -- the Biot-Savart feedback is itself a
    shift-equivariant linear operator (a circulant Fourier multiplier),
    so adding it must not break the mixer's existing equivariance
    guarantee."""
    torch.manual_seed(0)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted_x = torch.roll(z_field, shifts=1, dims=1).reshape(2, Px * Py * c)
    z_shifted_y = torch.roll(z_field, shifts=2, dims=2).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted_x = prop.step_one(z_shifted_x)
        out_shifted_y = prop.step_one(z_shifted_y)
    out_field = out.reshape(2, Px, Py, c)
    assert torch.allclose(
        out_shifted_x.reshape(2, Px, Py, c), torch.roll(out_field, shifts=1, dims=1), atol=1e-4
    )
    assert torch.allclose(
        out_shifted_y.reshape(2, Px, Py, c), torch.roll(out_field, shifts=2, dims=2), atol=1e-4
    )


def test_gradient_flows():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    out.sum().backward()
    body = prop.body
    assert body.token_embed.weight.grad is not None and body.token_embed.weight.grad.abs().sum() > 0
    assert body.velocity_embed.weight.grad is not None and body.velocity_embed.weight.grad.abs().sum() > 0
    assert body.blocks[0].mix.weight.grad is not None and body.blocks[0].mix.weight.grad.abs().sum() > 0


def test_aux_propagator_config_works_too():
    cfg = _cfg(cls=AuxPropagatorConfig, zero_init=False)
    prop = AuxPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_missing_attn_window_rejected():
    with pytest.raises(ValueError, match="local_mlp_2d_biotsavart"):
        _cfg(attn_window=None)


def test_wrong_d_latent_rejected():
    with pytest.raises(ValueError, match="local_mlp_2d_biotsavart"):
        _cfg(d_latent=99)


def test_nonpositive_Lx_rejected():
    with pytest.raises(ValueError, match="biot_savart_Lx"):
        _cfg(biot_savart_Lx=0.0)
    with pytest.raises(ValueError, match="biot_savart_Lx"):
        _cfg(biot_savart_Ly=-1.0)


def test_rollout_multistep_finite():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z0 = torch.randn(4, 192)
    roll = prop.rollout(z0, z0, k=5)
    assert roll.shape == (4, 5, 192)
    assert torch.isfinite(roll).all()


# --- _biot_savart_velocity: the numerics, in isolation (ground rule 1) ---

def test_biot_savart_k0_mode_zeroed():
    """A spatially-constant vorticity field (pure k=0) must produce
    EXACTLY zero velocity -- the k=0 divide is singular and must be
    masked out, not silently produce a huge/nan value (ground rule 2:
    fail loudly, never swallow a nan)."""
    omega = torch.full((1, 8, 8), 3.7, dtype=torch.float64)
    uv = _biot_savart_velocity(omega, Lx=2 * math.pi, Ly=2 * math.pi)
    assert torch.isfinite(uv).all()
    assert torch.allclose(uv, torch.zeros_like(uv), atol=1e-10)


def test_biot_savart_matches_analytic_taylor_green():
    """Taylor-Green-like single-mode vorticity `omega = cos(x)*cos(y)`
    (on an `Lx=Ly=2*pi` domain, wavenumber 1 in both directions) has a
    known closed-form streamfunction `psi = omega / 2` (since
    `nabla^2 psi = -2*psi = -omega` requires `psi = omega/2` for this
    mode) and therefore `u = psi_y = -cos(x)*sin(y)/2`,
    `v = -psi_x = sin(x)*cos(y)/2`. The computed field must match this
    to near machine precision -- the one test that proves the elliptic
    solve itself is correct, independent of anything about training."""
    N = 32
    Lx = Ly = 2 * math.pi
    x = torch.linspace(0, Lx, N + 1)[:-1]
    y = torch.linspace(0, Ly, N + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    omega = (torch.cos(X) * torch.cos(Y)).unsqueeze(0).double()

    uv = _biot_savart_velocity(omega, Lx=Lx, Ly=Ly)
    u_expected = -0.5 * torch.cos(X) * torch.sin(Y)
    v_expected = 0.5 * torch.sin(X) * torch.cos(Y)

    assert torch.allclose(uv[0, :, :, 0], u_expected.double(), atol=2e-7)
    assert torch.allclose(uv[0, :, :, 1], v_expected.double(), atol=2e-7)


def test_biot_savart_scales_with_domain_size():
    """Doubling Lx (holding the SAME site-grid shape, i.e. the same
    integer wavenumber n=1 now corresponds to half the physical k) must
    change the recovered velocity amplitude -- `k` depends on physical
    L, not just grid shape, so this is a direct check that `Lx`/`Ly`
    are actually being used, not silently ignored."""
    N = 16
    x = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    y = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    omega = (torch.cos(X) * torch.cos(Y)).unsqueeze(0).double()

    uv_L1 = _biot_savart_velocity(omega, Lx=2 * math.pi, Ly=2 * math.pi)
    uv_L2 = _biot_savart_velocity(omega, Lx=4 * math.pi, Ly=2 * math.pi)
    assert not torch.allclose(uv_L1, uv_L2, atol=1e-6)


# --- resize loader ---

def _make_checkpoint(tmp_path, Lx=2 * math.pi, Ly=2 * math.pi, n_sites_x=8, n_sites_y=8):
    cfg = _cfg(local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=n_sites_y,
               biot_savart_Lx=Lx, biot_savart_Ly=Ly, zero_init=False)
    prop = LatentPropagator(cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": cfg}, path)
    return path


def test_resize_rescales_Lx_and_Ly(tmp_path):
    path = _make_checkpoint(tmp_path, Lx=2 * math.pi, Ly=2 * math.pi, n_sites_x=8, n_sites_y=8)
    prop, new_cfg, _ = load_propagator_checkpoint_resized_2d_biotsavart(
        str(path), new_n_sites_x=16, new_n_sites_y=8,
    )
    assert new_cfg.biot_savart_Lx == pytest.approx(4 * math.pi)
    assert new_cfg.biot_savart_Ly == pytest.approx(2 * math.pi)
    assert new_cfg.d_latent == 16 * 8 * 3
    z = torch.randn(2, new_cfg.d_latent)
    out = prop.step_one(z)
    assert out.shape == (2, new_cfg.d_latent)
    assert torch.isfinite(out).all()


def test_resize_rejects_wrong_backbone(tmp_path):
    cfg = PropagatorConfig(
        d_latent=192, mode="markovian", backbone="local_mlp_2d", attn_window=2,
        local_mlp2d_n_sites_x=8, local_mlp2d_n_sites_y=8, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    path = tmp_path / "ckpt_wrong.pt"
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": cfg}, path)
    with pytest.raises(ValueError, match="local_mlp_2d_biotsavart"):
        load_propagator_checkpoint_resized_2d_biotsavart(str(path), new_n_sites_x=16, new_n_sites_y=8)
