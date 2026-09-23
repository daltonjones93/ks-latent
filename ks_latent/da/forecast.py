"""Ensemble forecasting through the trained latent propagator (brief §7).

`MODEL_NOISE_STD` is injected at *each* propagator step to represent model
error; the free-running (no-DA) comparison trajectory gets none, since it
is meant to show what the bare propagator does without any error-recovery
mechanism at all -- injecting model noise into it would understate the
free run's degradation and make the DA "skill" comparison unfair.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ks_latent.models.propagator import LatentPropagator

MODEL_NOISE_STD = 0.08  # brief §7 / PROJECT_HANDOFF.md: tuned so forecast spread ~= analysis RMSE
OBS_NOISE_STD = 0.1  # brief §7: OBS_NOISE = 0.1, physical units


@dataclass(frozen=True)
class ForecastResult:
    history: torch.Tensor  # (n_steps, N, d) -- z at each forecast step
    z_prev_final: torch.Tensor  # (N, d)
    z_curr_final: torch.Tensor  # (N, d)


def forecast_ensemble(
    propagator: LatentPropagator,
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    n_steps: int,
    model_noise_std: float = 0.0,
) -> ForecastResult:
    """Roll a `(N, d)` ensemble forward `n_steps` propagator steps, injecting
    `model_noise_std * randn` at each step (0.0 for a free run)."""
    history = []
    with torch.no_grad():
        for _ in range(n_steps):
            z_next = propagator.step(z_prev, z_curr)
            if model_noise_std > 0.0:
                z_next = z_next + model_noise_std * torch.randn_like(z_next)
            history.append(z_next)
            z_prev, z_curr = z_curr, z_next
    return ForecastResult(
        history=torch.stack(history, dim=0), z_prev_final=z_prev, z_curr_final=z_curr
    )


@dataclass(frozen=True)
class ForecastResultHistory:
    history: torch.Tensor  # (n_steps, N, d)
    z_hist_final: torch.Tensor  # (N, n_history, d) -- sliding window after the rollout


def forecast_ensemble_history(
    propagator: LatentPropagator,
    z_hist: torch.Tensor,  # (N, n_history, d), oldest to newest
    n_steps: int,
    model_noise_std: float = 0.0,
) -> ForecastResultHistory:
    """`mode="history"` counterpart of `forecast_ensemble` (added 2026-08-29,
    user-directed), added alongside it rather than replacing it -- keeps
    every existing `markovian`/`two_step` caller and test (some of which
    use a fake propagator with no `.mode` attribute at all) working
    unchanged. Rolls a `(N, n_history, d)` ensemble history window forward
    `n_steps` propagator steps via `.step_history`, sliding the window
    (drop oldest, append the new prediction) each step."""
    history = []
    hist = z_hist
    with torch.no_grad():
        for _ in range(n_steps):
            z_next = propagator.step_history(hist)
            if model_noise_std > 0.0:
                z_next = z_next + model_noise_std * torch.randn_like(z_next)
            history.append(z_next)
            hist = torch.cat([hist[:, 1:], z_next.unsqueeze(1)], dim=1)
    return ForecastResultHistory(history=torch.stack(history, dim=0), z_hist_final=hist)
