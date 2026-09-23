"""Tests for `EnsemblePropagator` (user-directed 2026-08-30, "Solution 1"
for the Stage-2 fixed-point collapse -- see
docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4 and
`EnsemblePropagatorConfig`'s docstring).
"""

from __future__ import annotations

import torch
import pytest

from ks_latent.config import EnsemblePropagatorConfig, PropagatorConfig
from ks_latent.models.ensemble_propagator import EnsemblePropagator


def _member_cfg(mode: str, **overrides) -> PropagatorConfig:
    kwargs = dict(d_latent=8, hidden=16, n_blocks=1, dropout=0.0, mode=mode, backbone="mlp")
    kwargs.update(overrides)
    return PropagatorConfig(**kwargs)


def test_two_step_member_raises():
    with pytest.raises(ValueError, match="markovian.*history"):
        EnsemblePropagatorConfig(member=_member_cfg("two_step"))


def test_n_members_must_be_positive():
    with pytest.raises(ValueError, match="n_members"):
        EnsemblePropagatorConfig(member=_member_cfg("markovian"), n_members=0)


def test_negative_noise_std_frac_raises():
    with pytest.raises(ValueError, match="noise_std_frac"):
        EnsemblePropagatorConfig(member=_member_cfg("markovian"), noise_std_frac=-0.1)


def test_cfg_proxies_d_latent_and_n_history():
    cfg = EnsemblePropagatorConfig(
        member=PropagatorConfig(
            d_latent=8, mode="history", backbone="vit", n_history=3, n_tokens=4, token_d_model=8,
            token_nhead=2, token_n_layers=1,
        )
    )
    assert cfg.d_latent == 8
    assert cfg.n_history == 3


def test_markovian_identity_at_init_regardless_of_noise():
    cfg = EnsemblePropagatorConfig(
        member=_member_cfg("markovian"), n_members=4, noise_std_frac=0.5, learned_weights=True
    )
    ens = EnsemblePropagator(cfg)
    assert ens.mode == "markovian"
    z = torch.randn(5, 8)
    z_next = ens.step_one(z)
    torch.testing.assert_close(z_next, z)


def test_history_identity_at_init_regardless_of_noise():
    # mode="history" is only implemented with backbone="vit" in LatentPropagator;
    # use a vit-backbone member sized small enough for a fast test.
    cfg = EnsemblePropagatorConfig(
        member=PropagatorConfig(
            d_latent=8, mode="history", backbone="vit", n_history=3, n_tokens=4, token_d_model=8,
            token_nhead=2, token_n_layers=1,
        ),
        n_members=3,
        noise_std_frac=0.3,
    )
    ens = EnsemblePropagator(cfg)
    assert ens.mode == "history"
    z_hist = torch.randn(5, 3, 8)
    z_next = ens.step_history(z_hist)
    torch.testing.assert_close(z_next, z_hist[:, -1])


def test_markovian_shapes_and_wrong_mode_raises():
    cfg = EnsemblePropagatorConfig(member=_member_cfg("markovian"), n_members=3)
    ens = EnsemblePropagator(cfg)
    z_prev = torch.randn(4, 8)
    z_curr = torch.randn(4, 8)
    assert ens.step(z_prev, z_curr).shape == (4, 8)
    assert ens.rollout(z_prev, z_curr, k=5).shape == (4, 5, 8)
    with pytest.raises(ValueError, match="mode='history'"):
        ens.step_history(torch.randn(4, 3, 8))
    with pytest.raises(ValueError, match="mode='history'"):
        ens.rollout_history(torch.randn(4, 3, 8), k=2)


def test_history_shapes_and_wrong_mode_raises():
    cfg = EnsemblePropagatorConfig(
        member=PropagatorConfig(
            d_latent=8, mode="history", backbone="vit", n_history=3, n_tokens=4, token_d_model=8,
            token_nhead=2, token_n_layers=1,
        ),
        n_members=2,
    )
    ens = EnsemblePropagator(cfg)
    z_hist = torch.randn(4, 3, 8)
    assert ens.step_history(z_hist).shape == (4, 8)
    assert ens.rollout_history(z_hist, k=5).shape == (4, 5, 8)
    with pytest.raises(ValueError, match="mode='markovian'"):
        ens.step(torch.randn(4, 8), torch.randn(4, 8))
    with pytest.raises(ValueError, match="mode='markovian'"):
        ens.rollout(torch.randn(4, 8), torch.randn(4, 8), k=2)


def test_uniform_weights_equal_mean_of_member_outputs_when_noise_off():
    cfg = EnsemblePropagatorConfig(
        member=_member_cfg("markovian", zero_init=False), n_members=4,
        noise_std_frac=0.0, learned_weights=False,
    )
    ens = EnsemblePropagator(cfg)
    z = torch.randn(6, 8)
    expected = torch.stack([m.step_one(z) for m in ens.members], dim=0).mean(dim=0)
    actual = ens.step_one(z)
    torch.testing.assert_close(actual, expected)


def test_learned_gate_starts_uniform_at_init():
    cfg = EnsemblePropagatorConfig(member=_member_cfg("markovian"), n_members=5, learned_weights=True)
    ens = EnsemblePropagator(cfg)
    z = torch.randn(3, 8)
    w = ens._weights(z)
    torch.testing.assert_close(w, torch.full((3, 5), 0.2))


def test_set_latent_var_validates_shape_and_scales_noise():
    cfg = EnsemblePropagatorConfig(member=_member_cfg("markovian"), n_members=3, noise_std_frac=0.1)
    ens = EnsemblePropagator(cfg)
    with pytest.raises(ValueError, match="shape"):
        ens.set_latent_var(torch.ones(4))
    var = torch.full((8,), 100.0)
    ens.set_latent_var(var)
    torch.testing.assert_close(ens.latent_var, var)
    eps = ens._noise(torch.randn(2, 8))
    assert eps.shape == (3, 2, 8)
