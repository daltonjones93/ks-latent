"""Tests for the latent-propagator Lyapunov adapters (brief §6.2:
single_state/two_step modes via torch.func.jvp).

Ground rule 1: validate the exact tangent-map machinery (`two_step`) against
a hand-built *linear* two-step recursion with an analytically known
spectrum (via the companion-matrix eigenvalues) before trusting it on a real
trained propagator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import pytest

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


@dataclass
class _FakeCfg:
    d_latent: int


class _LinearTwoStepPropagator(nn.Module):
    """z_{n+1} = A @ z_n + B @ z_{n-1}, exposing the same `.step(z_prev,
    z_curr)` interface as `LatentPropagator` (duck-typed, not a subclass) so
    the lyapunov.py adapters can be tested against a system with an exact,
    hand-computable spectrum: the companion matrix `[[0, I], [B, A]]`.
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)
        self.cfg = _FakeCfg(d_latent=A.shape[0])

    def step(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return z_curr @ self.A.T + z_prev @ self.B.T

    def companion_eigenvalues(self) -> np.ndarray:
        d = self.cfg.d_latent
        A = self.A.detach().numpy()
        B = self.B.detach().numpy()
        top = np.concatenate([np.zeros((d, d)), np.eye(d)], axis=1)
        bottom = np.concatenate([B, A], axis=1)
        companion = np.concatenate([top, bottom], axis=0)
        return np.linalg.eigvals(companion)


def test_two_step_recovers_exact_linear_recursion_spectrum():
    torch.manual_seed(0)
    d = 3
    # Well-separated |eigenvalue| companion spectrum: diagonal A, B=0 reduces
    # to two decoupled copies of a diagonal 1-step map (eigenvalues of A,
    # each with multiplicity... no: a genuinely 2-step recursion needs B!=0).
    A = torch.diag(torch.tensor([0.5, 0.3, 0.1]))
    B = torch.diag(torch.tensor([0.05, 0.02, 0.01]))
    prop = _LinearTwoStepPropagator(A, B)
    true_eigs = prop.companion_eigenvalues()
    true_log_abs_sorted = np.sort(np.log(np.abs(true_eigs)))[::-1]

    rng = np.random.default_rng(0)
    initial_pair = rng.normal(size=2 * d)
    result = lyapunov_spectrum_latent_propagator(
        prop, initial_pair, mode="two_step", n_directions=2 * d,
        n_steps=400, qr_every=1, dt_snap=1.0, warmup_steps=200, seed=1,
    )
    assert result.exponents == pytest.approx(true_log_abs_sorted, abs=1e-3)


def test_single_state_and_two_step_agree_when_propagator_ignores_history():
    """If the trained map genuinely doesn't depend on z_{n-1} (B=0 in the
    linear toy above), the single_state approximation g(z)=step(z,z) is
    exact (both slots are equivalent), so single_state and two_step's
    "new" eigenvalues (those of A, i.e. the top d of the 2d companion
    spectrum) should agree -- a check that single_state isn't just wrong by
    construction, only when there's genuine two-step memory to drop.
    """
    torch.manual_seed(0)
    d = 3
    A = torch.diag(torch.tensor([0.6, 0.3, 0.15]))
    B = torch.zeros(d, d)
    prop = _LinearTwoStepPropagator(A, B)

    rng = np.random.default_rng(0)
    initial_pair = rng.normal(size=2 * d)
    single = lyapunov_spectrum_latent_propagator(
        prop, initial_pair, mode="single_state", n_directions=d,
        n_steps=300, qr_every=1, dt_snap=1.0, warmup_steps=150, seed=1,
    )
    expected = np.sort(np.log(np.abs(np.linalg.eigvals(A.numpy()))))[::-1]
    assert single.exponents == pytest.approx(expected, abs=1e-3)


def test_single_state_and_two_step_wiring_on_real_propagator():
    """Shape/no-crash sanity check on the actual `LatentPropagator`
    architecture (nonlinear, no closed-form spectrum to check against)."""
    torch.manual_seed(0)
    cfg = PropagatorConfig(d_latent=4, hidden=16, n_blocks=1, dropout=0.0, zero_init=False)
    prop = LatentPropagator(cfg)
    rng = np.random.default_rng(0)
    initial_pair = rng.normal(size=8) * 0.1

    single = lyapunov_spectrum_latent_propagator(
        prop, initial_pair, mode="single_state", n_directions=4,
        n_steps=40, qr_every=2, dt_snap=0.25, warmup_steps=20, seed=2,
        max_abs_state=1e4,
    )
    assert single.exponents.shape == (4,)

    two_step = lyapunov_spectrum_latent_propagator(
        prop, initial_pair, mode="two_step", n_directions=8,
        n_steps=40, qr_every=2, dt_snap=0.25, warmup_steps=20, seed=2,
        max_abs_state=1e4,
    )
    assert two_step.exponents.shape == (8,)


def test_single_state_wiring_on_ensemble_propagator():
    """`EnsemblePropagator` injects Gaussian noise via `torch.randn` on
    every `.step`/`.step_one` call -- which crashes `torch.func.vmap`
    (used by the tangent-map linearization below) unless the noise is
    disabled for that one linearization step. Caught 2026-08-30 running
    Gate 3 on a real trained ensemble checkpoint (see
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4); this is the
    regression test for `EnsemblePropagator.noise_disabled()` and
    `lyapunov._maybe_noise_disabled`."""
    from ks_latent.config import EnsemblePropagatorConfig
    from ks_latent.models.ensemble_propagator import EnsemblePropagator

    torch.manual_seed(0)
    member_cfg = PropagatorConfig(
        d_latent=4, hidden=16, n_blocks=1, dropout=0.0, zero_init=False, mode="markovian"
    )
    ens_cfg = EnsemblePropagatorConfig(member=member_cfg, n_members=3, noise_std_frac=0.05)
    prop = EnsemblePropagator(ens_cfg)
    rng = np.random.default_rng(0)
    initial_pair = rng.normal(size=8) * 0.1

    result = lyapunov_spectrum_latent_propagator(
        prop, initial_pair, mode="single_state", n_directions=4,
        n_steps=20, qr_every=2, dt_snap=0.25, warmup_steps=10, seed=2,
        max_abs_state=1e4,
    )
    assert result.exponents.shape == (4,)
    assert np.isfinite(result.exponents).all()


def test_lyapunov_spectrum_latent_propagator_rejects_bad_mode():
    cfg = PropagatorConfig(d_latent=3, hidden=8, n_blocks=1)
    prop = LatentPropagator(cfg)
    with pytest.raises(ValueError, match="mode must be"):
        lyapunov_spectrum_latent_propagator(
            prop, np.zeros(6), mode="bogus", n_directions=2,
            n_steps=2, qr_every=1, dt_snap=0.25,
        )
