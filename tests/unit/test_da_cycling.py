"""Tests for ensemble forecasting and the DA cycling driver (brief §7)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from ks_latent.da.cycling import CycleConfig, run_da_experiment
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.pff import PFFConfig
from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def test_forecast_ensemble_noise_free_is_deterministic():
    torch.manual_seed(0)
    cfg = PropagatorConfig(d_latent=4, hidden=16, n_blocks=1, dropout=0.0, zero_init=False)
    prop = LatentPropagator(cfg)
    z_prev = torch.randn(5, 4)
    z_curr = torch.randn(5, 4)
    r1 = forecast_ensemble(prop, z_prev, z_curr, n_steps=10, model_noise_std=0.0)
    r2 = forecast_ensemble(prop, z_prev, z_curr, n_steps=10, model_noise_std=0.0)
    assert torch.equal(r1.history, r2.history)


def test_forecast_ensemble_with_noise_differs_across_calls():
    torch.manual_seed(0)
    cfg = PropagatorConfig(d_latent=4, hidden=16, n_blocks=1, dropout=0.0, zero_init=False)
    prop = LatentPropagator(cfg)
    z_prev = torch.randn(5, 4)
    z_curr = torch.randn(5, 4)
    r1 = forecast_ensemble(prop, z_prev, z_curr, n_steps=10, model_noise_std=0.08)
    r2 = forecast_ensemble(prop, z_prev, z_curr, n_steps=10, model_noise_std=0.08)
    assert not torch.equal(r1.history, r2.history)
    assert r1.history.shape == (10, 5, 4)


class _LinearExpandingPropagator(nn.Module):
    """z_{n+1} = A @ z_n, A slightly expanding (ignores history) -- initial
    ensemble/IC error grows under free-running rollout, so DA's correction
    has something real to demonstrate against."""

    def __init__(self, d: int, growth: float = 1.03):
        super().__init__()
        self.cfg = PropagatorConfig(d_latent=d)
        self.A = nn.Parameter(torch.eye(d) * growth, requires_grad=False)

    def step(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return z_curr @ self.A.T


@pytest.mark.integration
def test_da_beats_free_run_on_mildly_unstable_linear_system():
    torch.manual_seed(0)
    d, NX, N = 3, 12, 40
    torch.manual_seed(1)
    W = torch.randn(NX, d) * 0.5  # decoder: R^d -> R^NX
    W_pinv = torch.linalg.pinv(W)  # encoder: R^NX -> R^d, encode(decode(z))=z exactly

    def decoder(Z):
        return Z @ W.T

    def encoder(U):
        return U @ W_pinv.T

    def obs_operator_phys(u):
        return u[::2]  # observe every other physical grid point

    prop = _LinearExpandingPropagator(d, growth=1.03)

    # True latent trajectory (no model error at all -- the propagator here
    # *is* the true dynamics): a fixed IC rolled forward.
    n_prop_steps, n_cycles = 4, 6
    T = n_prop_steps * n_cycles
    z_true0 = torch.tensor([0.3, -0.2, 0.1])
    z_true = [z_true0]
    for _ in range(T):
        z_true.append((z_true[-1].unsqueeze(0) @ prop.A.T).squeeze(0))
    z_true = torch.stack(z_true, dim=0)  # (T+1, d)
    truth_traj = decoder(z_true)  # (T+1, NX)

    # Both ensembles start with the *same* biased/uncertain IC (deliberately
    # offset from truth), history pair identical for prev/curr.
    rng = torch.Generator().manual_seed(2)
    ic_bias = torch.tensor([0.15, -0.1, 0.08])
    z0_ensemble = z_true0 + ic_bias + 0.05 * torch.randn(N, d, generator=rng)
    z_minus1_ensemble = z0_ensemble.clone()

    cfg = CycleConfig(
        n_prop_steps=n_prop_steps, n_cycles=n_cycles, n_ensemble=N,
        model_noise_std=0.0, obs_noise_std=0.05,
        pff_config=PFFConfig(method="NAT", n_steps=60),
        seed=3,
    )
    result = run_da_experiment(
        prop, decoder, encoder, obs_operator_phys, truth_traj,
        z0_ensemble, z_minus1_ensemble, cfg,
    )

    assert result.rmse_da.mean() < result.rmse_free.mean()
    # DA should track truth increasingly well; free run (biased IC, growth
    # >1) should not.
    assert result.rmse_free[-1] > result.rmse_free[0]
    assert result.u_da.shape == result.u_truth.shape == (T, NX)


def test_cycle_config_localize_fn_flows_through_to_pff():
    """Plumbing check for brief §9: CycleConfig.localize_fn reaches
    ParticleFlowFilter and changes the analysis (the taper's actual DA
    benefit is validated end-to-end in test_sec_pff_integration.py)."""
    torch.manual_seed(0)
    d, NX, N = 3, 12, 40
    W = torch.randn(NX, d) * 0.5
    W_pinv = torch.linalg.pinv(W)

    def decoder(Z):
        return Z @ W.T

    def encoder(U):
        return U @ W_pinv.T

    def obs_operator_phys(u):
        return u[::2]

    prop = _LinearExpandingPropagator(d, growth=1.0)
    z_true0 = torch.tensor([0.3, -0.2, 0.1])
    n_prop_steps, n_cycles = 2, 3
    T = n_prop_steps * n_cycles
    z_true = [z_true0]
    for _ in range(T):
        z_true.append((z_true[-1].unsqueeze(0) @ prop.A.T).squeeze(0))
    truth_traj = decoder(torch.stack(z_true, dim=0))

    rng = torch.Generator().manual_seed(2)
    z0_ensemble = z_true0 + 0.1 + 0.05 * torch.randn(N, d, generator=rng)
    z_minus1_ensemble = z0_ensemble.clone()

    taper = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # zero all cross-terms
    localize_fn = lambda B: B * taper  # noqa: E731

    def run(localize_arg):
        cfg = CycleConfig(
            n_prop_steps=n_prop_steps, n_cycles=n_cycles, n_ensemble=N,
            model_noise_std=0.0, obs_noise_std=0.05,
            pff_config=PFFConfig(method="NAT", n_steps=40), seed=3, localize_fn=localize_arg,
        )
        return run_da_experiment(
            prop, decoder, encoder, obs_operator_phys, truth_traj,
            z0_ensemble.clone(), z_minus1_ensemble.clone(), cfg,
        )

    result_no_taper = run(None)
    result_tapered = run(localize_fn)
    assert not torch.allclose(result_no_taper.z_da_mean, result_tapered.z_da_mean)
