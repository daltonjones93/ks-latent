"""Observation operators for latent DA on CM1 (E1).

E1a `SurfaceTheta`: surface potential-temperature perturbation theta' (K) at chosen grid columns,
read from the decoded state: H(z) = theta'_level0(decode(z))[columns]. Linear in the decoded field;
nonlinear in the latent through the decoder, which the LETKF handles through the ensemble.
`truth_values` reads the same quantity from the true CM1 columns, for synthetic observations.
"""
from __future__ import annotations

import numpy as np
import torch

from ks_latent.cm1.data import VARS, Normalizer


class SurfaceTheta:
    def __init__(self, columns: np.ndarray, norm: Normalizer, n_levels: int, dx_km: float = 0.25):
        self.columns = np.asarray(columns, dtype=int)
        self.channel = VARS.index("th") * n_levels          # level 0 of theta'
        self.scale = float(norm.scale[VARS.index("th")])
        self.x_km = (self.columns + 0.5) * dx_km

    @torch.no_grad()
    def __call__(self, ae, Z: torch.Tensor) -> np.ndarray:
        """`Z` (N, P, c) latent ensemble -> (N, m) observation-space values (K)."""
        u = ae.decode(Z)                                     # (N, C, NX) normalized
        return (u[:, self.channel, self.columns] * self.scale).double().cpu().numpy()

    def truth_values(self, u_true: torch.Tensor) -> np.ndarray:
        """True normalized columns `(C, NX)` -> (m,) theta' (K) at the observed columns."""
        return (u_true[self.channel, self.columns] * self.scale).double().cpu().numpy()

    def r_var(self, sigma_k: float) -> np.ndarray:
        return np.full(self.columns.size, sigma_k**2)
