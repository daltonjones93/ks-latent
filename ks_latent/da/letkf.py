"""Localized Ensemble Kalman Filter (LETKF family) -- a second DA
analysis scheme, added as a comparator to this project's primary NAT-PFF
method (`ks_latent/da/pff.py`). User-directed (2026-09-28): "would
another DA technique be more suited to this problem? ... some kind of
kalman filtering?" then "implement LETKF."

**Naming note, to avoid overclaiming precision.** This is NOT literally
Hunt et al. (2007)'s own per-gridpoint local-domain LETKF algorithm
(designed for very high-dimensional systems, where solving many small
local problems in parallel is the whole point). At this project's scale
(`d_latent <= 192`), a full `(d,d)` forecast covariance is cheap to hold
explicitly, so this instead implements the mathematically equivalent,
earlier "covariance-localized EnKF" (Houtekamer & Mitchell 1998/2001):
a standard perturbed-observation ensemble Kalman analysis using the SAME
`(d,d)` Gaspari-Cohn/SEC taper (`localize_fn`) this project's PFF
already accepts (`ks_latent.da.localization`/`ks_latent.da.sec`),
Schur-producted onto the ensemble forecast covariance `B`, rather than
PFF's own Gauss-Newton flow.

**Deliberately isolates one variable.** Same forecast step (unchanged,
`ks_latent.da.forecast`), same ensemble-mean-Jacobian linearization of
the (nonlinear, decoder-based) observation operator PFF's own NAT
variant already uses, same localization taper -- the ONE thing that
differs from PFF is the analysis update rule itself (a linear Kalman
gain vs. PFF's nonlinear flow), by design, so a PFF-vs-LETKF comparison
cleanly answers "does the nonlinear flow matter here," not "did we also
change the forecast or the localization."

Analysis update (perturbed-observation / stochastic EnKF, Evensen 1994 /
Burgers, van Leeuwen & Evensen 1998):

    B           = Cov(Z0)                         (d, d), then localized
    J_bar       = d h/dz |_{z_bar}                 (m, d), ensemble-mean Jacobian
    K           = B J_bar^T (J_bar B J_bar^T + R)^-1   (d, m)  -- Kalman gain
    y_i         = y + eps_i,  eps_i ~ N(0, R)      per-member perturbed obs
    Z_analysis  = Z0 + (y_ens - h(Z0)) K^T

`h(Z0)` is evaluated once per member directly (never approximated by
`J_bar @ Z0`), so any genuine nonlinearity in the true observation
operator still contributes exactly through the innovation term, exactly
as it would in a textbook nonlinear EnKF -- only the GAIN itself is built
from the linearization, which is the standard, well-understood
approximation this whole filter family makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import jacrev

ObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (d,) -> (m,), float64


def _jacobian_at(h: ObsOperator, z: torch.Tensor) -> torch.Tensor:
    """`d h/dz` at `z`, `(m, d)`. CPU/float64 (brief's device policy:
    `jacrev` has known MPS gaps -- see `ks_latent.da.pff._jacobian_at`,
    which does the same thing; duplicated here rather than imported to
    keep this module self-contained and independently testable)."""
    return jacrev(h)(z)


@dataclass(frozen=True)
class LETKFResult:
    Z_analysis: torch.Tensor  # (N, d)


class LETKF:
    """Covariance-localized Ensemble Kalman Filter -- see module
    docstring for the precise algorithm and the naming caveat. Same
    `analyze(Z0, y) -> result.Z_analysis` interface as
    `ks_latent.da.pff.ParticleFlowFilter`, so it is a drop-in replacement
    wherever that interface is used (`ks_latent.da.cycling.run_da_experiment`
    via `CycleConfig.method="letkf"`)."""

    def __init__(
        self,
        h: ObsOperator,
        R: torch.Tensor,  # (m, m), float64, observation error covariance
        localize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        inflation: float = 1.0,
        seed: int = 0,
    ):
        self.h = h
        self.R = R
        self.localize_fn = localize_fn
        self.inflation = inflation
        self._rng = torch.Generator().manual_seed(seed)

    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> LETKFResult:
        """`Z0`: `(N, d)` forecast ensemble, float64. `y`: `(m,)`
        observation, float64."""
        N, d = Z0.shape
        z_bar = Z0.mean(dim=0)
        Z_pert = Z0 - z_bar
        if self.inflation != 1.0:
            Z_pert = Z_pert * self.inflation
        B = (Z_pert.T @ Z_pert) / (N - 1)  # (d, d)
        if self.localize_fn is not None:
            B = self.localize_fn(B)

        J_bar = _jacobian_at(self.h, z_bar).detach()  # (m, d); detached so no (m, d) product keeps a graph alive
        return self._analyze_no_grad(Z0, y, B, J_bar)

    @torch.no_grad()
    def _analyze_no_grad(self, Z0, y, B, J_bar) -> LETKFResult:
        N = Z0.shape[0]
        HB = J_bar @ B  # (m, d)
        HBHt = HB @ J_bar.T  # (m, m)
        gain_denom = HBHt + self.R  # (m, m), SPD (R is SPD, HBHt is PSD)
        # K = B J^T (J B J^T + R)^-1, solved without an explicit inverse:
        # K^T = gain_denom^-1 (HB), so K = (gain_denom^-1 HB)^T = HB^T gain_denom^-T.
        # gain_denom is symmetric, so gain_denom^-T = gain_denom^-1.
        K = torch.linalg.solve(gain_denom, HB).T  # (d, m)

        m = y.shape[0]
        L_R = torch.linalg.cholesky(self.R)
        obs_perturbations = (L_R @ torch.randn(m, N, generator=self._rng, dtype=y.dtype)).T  # (N, m)
        y_ens = y.unsqueeze(0) + obs_perturbations  # (N, m)

        Hz = torch.stack([self.h(Z0[i]) for i in range(N)], dim=0)  # (N, m)
        innovation = y_ens - Hz  # (N, m)
        Za = Z0 + innovation @ K.T  # (N, d)
        return LETKFResult(Z_analysis=Za.detach())  # see letkf_local.py: never return an autograd graph
