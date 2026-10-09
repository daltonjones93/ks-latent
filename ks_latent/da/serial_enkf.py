"""Serial perturbed-observation EnKF with sampling error correction (task D3(ii),
docs/full_paper_tasks.md; Anderson 2012, Mon. Wea. Rev. 140:2359, which applies SEC in exactly
this serial form).

Observations are assimilated one at a time (Houtekamer & Mitchell 2001; Whitaker & Hamill
2002 for the serial form). The joint state is the latent ensemble augmented with the
ensemble's observation-space values, X = [Z, h(Z)] of shape (N, d + m), so that later
observations see the update from earlier ones without re-evaluating the nonlinear `h`. For
observation j, with ensemble prior hat-y_j = X[:, d + j]:

    beta_k   = Cov(X_k, hat-y_j) / (Var(hat-y_j) + R_j)                    regression coefficient
    with SEC: Cov(X_k, hat-y_j) -> SEC(corr(X_k, hat-y_j)) * std(X_k) * std(hat-y_j)
    X_k     += beta_k * (y_j + eps_j - hat-y_j),     eps_j ~ N(0, R_j) per member

The correction acts on each scalar regression coefficient, never on a matrix, so the update is
well defined at every ensemble size whether or not the elementwise-corrected covariance
would be positive definite (cf. `sec.sec_localize_covariance_psd`). With `table=None`, or a
table that is the identity map, this is the plain serial perturbed-observation EnKF, whose
posterior for a linear-Gaussian problem converges to the Kalman posterior as N grows
(tests/unit/test_serial_enkf.py). Same `analyze(Z0, y) -> result.Z_analysis` interface as the
other filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ks_latent.da.sec import SECTable, apply_sec_correction


@dataclass(frozen=True)
class SerialEnKFResult:
    Z_analysis: torch.Tensor  # (N, d)


class SerialSECEnKF:
    def __init__(self, h: Callable[[torch.Tensor], torch.Tensor], R_diag: torch.Tensor,
                 table: SECTable | None = None, seed: int = 0):
        self.h = h
        self.R_diag = R_diag.to(torch.float64)
        self.table = table
        self._rng = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> SerialEnKFResult:
        Z0 = Z0.to(torch.float64)
        N, d = Z0.shape
        Hz = torch.stack([self.h(Z0[i]).to(torch.float64) for i in range(N)])  # (N, m)
        m = Hz.shape[1]
        if y.shape != (m,):
            raise ValueError(f"y has shape {tuple(y.shape)}, expected ({m},)")
        X = torch.cat([Z0, Hz], dim=1)  # (N, d + m)
        eps = torch.randn(N, m, generator=self._rng, dtype=torch.float64) * self.R_diag.sqrt()
        for j in range(m):
            hy = X[:, d + j]
            hp = hy - hy.mean()
            var_h = float(hp.pow(2).sum()) / (N - 1)
            Xp = X - X.mean(dim=0)
            cov = (Xp.T @ hp) / (N - 1)  # (d + m,)
            if self.table is not None and var_h > 0:
                std_x = Xp.pow(2).sum(dim=0).div(N - 1).sqrt()
                std_h = var_h ** 0.5
                ok = std_x > 0
                corr = torch.zeros_like(cov)
                corr[ok] = cov[ok] / (std_x[ok] * std_h)
                corr_c = torch.as_tensor(apply_sec_correction(corr.clamp(-1, 1).numpy(), self.table), dtype=cov.dtype)
                corr_c[d + j] = 1.0  # an observation prior is perfectly correlated with itself
                cov = torch.where(ok, corr_c * std_x * std_h, torch.zeros_like(cov))
            beta = cov / (var_h + float(self.R_diag[j]))
            innov = y[j] + eps[:, j] - hy  # (N,)
            X = X + innov.unsqueeze(1) * beta.unsqueeze(0)
        return SerialEnKFResult(Z_analysis=X[:, :d].clone())
