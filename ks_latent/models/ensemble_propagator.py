"""Ensemble propagator (user-directed 2026-08-30, "Solution 1" for the
Stage-2 fixed-point collapse -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
Section 4 and `EnsemblePropagatorConfig`'s docstring for the full
motivation, spec, and the identity-at-init design choice).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from ks_latent.config import EnsemblePropagatorConfig
from ks_latent.models.propagator import LatentPropagator


class EnsemblePropagator(nn.Module):
    """`cfg.n_members` independently-initialized `LatentPropagator`s sharing
    one architecture (`cfg.member`), combined via a (optionally learned,
    `z`-dependent) weighted mean.

    Drop-in replacement for a plain `LatentPropagator` everywhere one is
    used (DA cycling, Lyapunov adapters, D3 coupling diagnostic,
    `train_stage2`'s loop): exposes the same `.mode`, `.cfg.d_latent`,
    `.cfg.n_history` surface, plus `.step_one`/`.step`/`.rollout` for
    `mode="markovian"` or `.step_history`/`.rollout_history` for
    `mode="history"` (mixing both on one instance is never possible, since
    `cfg.member.mode` fixes which applies -- see `EnsemblePropagatorConfig`).

    `latent_var` must be set via `set_latent_var(...)` from real data before
    training/inference with `noise_std_frac > 0` (defaults to all-ones,
    isotropic unit noise, until then).
    """

    def __init__(self, cfg: EnsemblePropagatorConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.member.mode
        self.d_latent = cfg.member.d_latent
        self.members = nn.ModuleList([LatentPropagator(cfg.member) for _ in range(cfg.n_members)])
        if cfg.learned_weights:
            self.gate = nn.Sequential(
                nn.Linear(self.d_latent, cfg.gate_hidden),
                nn.GELU(),
                nn.Linear(cfg.gate_hidden, cfg.n_members),
            )
            # Zero-init the gate's output layer: every member starts with
            # exactly equal (1/n_members) weight, matching the members' own
            # zero-init identity-at-init discipline -- the gate has to learn
            # away from uniform weighting from data, not start arbitrary.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            self.gate = None
        self.register_buffer("latent_var", torch.ones(self.d_latent))
        # See `noise_disabled` below: injected noise is part of this model's
        # actual generative process (kept on during training AND normal
        # eval/rollout -- NOT gated on self.training, unlike dropout), but
        # `torch.randn` cannot run inside `torch.func.vmap`/`jvp` (used by
        # the Lyapunov spectrum's tangent-map linearization -- see
        # docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4 for the crash
        # this caused and the fix's rationale) -- Lyapunov exponents are a
        # property of a deterministic map anyway, so that one diagnostic
        # measures the noise-free "skeleton" map's local stability instead.
        self._noise_enabled = True

    @contextlib.contextmanager
    def noise_disabled(self):
        """Temporarily zero out injected noise (see the attribute comment
        above) -- used by `ks_latent.analysis.lyapunov`'s tangent-map
        linearization, which needs a deterministic function and cannot run
        `torch.randn` inside `vmap`/`jvp`."""
        prev = self._noise_enabled
        self._noise_enabled = False
        try:
            yield
        finally:
            self._noise_enabled = prev

    def set_latent_var(self, latent_var: torch.Tensor) -> None:
        """Set the per-dimension `Var(z_i)` used to scale injected noise
        (`noise_std_i = cfg.noise_std_frac * sqrt(latent_var_i)`), computed
        from the encoded training set. Shape `(d_latent,)`."""
        if tuple(latent_var.shape) != (self.d_latent,):
            raise ValueError(
                f"latent_var must have shape ({self.d_latent},), got {tuple(latent_var.shape)}"
            )
        self.latent_var.copy_(latent_var.to(device=self.latent_var.device, dtype=self.latent_var.dtype))

    def _weights(self, z_ref: torch.Tensor) -> torch.Tensor:
        """`(B, n_members)` combination weights from the reference
        (un-noised) state, summing to 1 across members."""
        if self.gate is None:
            B = z_ref.shape[0]
            return z_ref.new_full((B, self.cfg.n_members), 1.0 / self.cfg.n_members)
        return torch.softmax(self.gate(z_ref), dim=-1)

    def _noise(self, z_ref: torch.Tensor) -> torch.Tensor:
        """`(n_members, *z_ref.shape)` diagonal Gaussian noise, std
        `noise_std_frac * sqrt(latent_var)` per latent dimension,
        independently sampled per member (no cross-dimension covariance,
        no cross-member correlation, per spec). Returns exact zeros while
        `noise_disabled()` is active (see its docstring)."""
        shape = (self.cfg.n_members, *z_ref.shape)
        if not self._noise_enabled:
            return z_ref.new_zeros(shape)
        std = self.cfg.noise_std_frac * torch.sqrt(self.latent_var.clamp_min(0.0))
        return std * torch.randn(shape, device=z_ref.device, dtype=z_ref.dtype)

    def _combine(self, z_ref: torch.Tensor, member_outputs: torch.Tensor) -> torch.Tensor:
        """`member_outputs`: `(n_members, B, d_latent)`. Returns `(B,
        d_latent)`, the weighted mean using weights gated on `z_ref`."""
        w = self._weights(z_ref).transpose(0, 1).unsqueeze(-1)  # (M, B, 1)
        return (w * member_outputs).sum(dim=0)

    def _require_mode(self, called: str, expected: str) -> None:
        if self.mode != expected:
            raise ValueError(f"{called}() requires mode={expected!r}, got mode={self.mode!r}")

    # ---- markovian interface (mirrors LatentPropagator) ----

    def step_one(self, z: torch.Tensor) -> torch.Tensor:
        self._require_mode("step_one", "markovian")
        eps = self._noise(z)  # (M, B, d)
        # z + capped_delta(body(z + eps)) -- NOT member.step_one(z + eps), which
        # would add the noised input as the residual base too (z_next = z + eps,
        # even at init). The base stays the clean z; only the delta computation
        # sees the noised input. See EnsemblePropagatorConfig's docstring.
        outputs = torch.stack(
            [
                z + member.capped_delta(member.body(z + eps[m]), z_ref=z)
                for m, member in enumerate(self.members)
            ],
            dim=0,
        )
        return self._combine(z, outputs)

    def step(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return self.step_one(z_curr)  # markovian only; z_prev accepted, ignored

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return self.step(z_prev, z_curr)

    def rollout(
        self, z_prev: torch.Tensor, z_curr: torch.Tensor, k: int, step_noise: float = 0.0
    ) -> torch.Tensor:
        """Same semantics as `LatentPropagator.rollout`: `step_noise`
        perturbs the COMBINED output at every step (training-only, distinct
        from the per-member input noise this class always injects)."""
        self._require_mode("rollout", "markovian")
        outputs = []
        for _ in range(k):
            z_next = self.step_one(z_curr)
            if step_noise > 0:
                z_next = z_next + step_noise * torch.randn_like(z_next)
            outputs.append(z_next)
            z_prev, z_curr = z_curr, z_next
        return torch.stack(outputs, dim=1)

    # ---- history interface (mirrors LatentPropagator) ----

    def step_history(self, z_hist: torch.Tensor) -> torch.Tensor:
        self._require_mode("step_history", "history")
        z_ref = z_hist[:, -1]
        eps = self._noise(z_ref)  # (M, B, d)
        outputs = []
        for m, member in enumerate(self.members):
            # Same clean-base / noised-input-to-delta split as step_one, applied
            # to the most-recent history slice only (the "current input").
            noised = z_hist.clone()
            noised[:, -1] = z_ref + eps[m]
            outputs.append(z_ref + member.capped_delta(member.body(noised), z_ref=z_ref))
        return self._combine(z_ref, torch.stack(outputs, dim=0))

    def rollout_history(self, z_hist: torch.Tensor, k: int, step_noise: float = 0.0) -> torch.Tensor:
        self._require_mode("rollout_history", "history")
        outputs = []
        hist = z_hist
        for _ in range(k):
            z_next = self.step_history(hist)
            if step_noise > 0:
                z_next = z_next + step_noise * torch.randn_like(z_next)
            outputs.append(z_next)
            hist = torch.cat([hist[:, 1:], z_next.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)
