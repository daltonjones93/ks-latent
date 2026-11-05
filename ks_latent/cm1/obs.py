"""Observation operators for latent DA on CM1 (E1).

E1a `SurfaceTheta`: surface potential-temperature perturbation theta' (K) at chosen grid columns,
read from the decoded state: H(z) = theta'_level0(decode(z))[columns]. Linear in the decoded field;
nonlinear in the latent through the decoder, which the LETKF handles through the ensemble.
`truth_values` reads the same quantity from the true CM1 columns, for synthetic observations.

E4b `RadialVelocity`: Doppler radial velocity from a radar at `radar_x_km` scanning `tilts_deg`,
observed at every `every_cells`-th column within `max_range_km`. The beam height at range r is the
standard 4/3-Earth-radius model, z = r sin(a) + r^2 / (2 k R), k = 4/3, R = 6371 km. The along-beam
wind is s u' cos(a) + w sin(a), s = sign(x - x_radar) (positive away from the radar), with u' and w
linearly interpolated between model levels to the beam height (beam broadening and hydrometeor fall
speed are neglected). Two simplifications, both stated where used: the radar sits at a fixed x of the
grid, which moves with the storm; and the known environmental wind is removed, so the operator sees the
perturbation u'. It is linear in the decoded state.
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


class RadialVelocity:
    def __init__(self, norm: Normalizer, z_levels_m: np.ndarray, n_columns: int, radar_x_km: float = 330.0,
                 tilts_deg=(0.5, 2.0, 5.0), every_cells: int = 8, max_range_km: float = 150.0,
                 min_range_km: float = 5.0, dx_km: float = 0.25):
        z = np.asarray(z_levels_m, np.float64)
        Zl = z.size
        cols_all = np.arange(every_cells // 2, n_columns, every_cells)
        xc = (cols_all + 0.5) * dx_km
        rows = []
        for a in np.deg2rad(np.asarray(tilts_deg, np.float64)):
            for c, x in zip(cols_all, xc):
                r = abs(x - radar_x_km)
                if r < min_range_km or r > max_range_km:
                    continue
                h = r * 1e3 * np.sin(a) + (r * 1e3) ** 2 / (2 * (4 / 3) * 6.371e6)
                if h < z[0] or h > z[-1]:
                    continue
                k = int(np.searchsorted(z, h) - 1)
                k = min(max(k, 0), Zl - 2)
                f = (h - z[k]) / (z[k + 1] - z[k])
                rows.append((c, k, f, np.sign(x - radar_x_km), np.cos(a), np.sin(a), x, h))
        if not rows:
            raise ValueError("no radial-velocity observations within range")
        R = np.array(rows)
        self.columns, self.k0, self.frac = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2]
        self.sign, self.cos, self.sin = R[:, 3], R[:, 4], R[:, 5]
        self.x_km, self.height_m = R[:, 6], R[:, 7]
        self.iu, self.iw = VARS.index("u") * Zl, VARS.index("w") * Zl
        self.su, self.sw = float(norm.scale[VARS.index("u")]), float(norm.scale[VARS.index("w")])

    def _from_columns(self, u: torch.Tensor) -> torch.Tensor:
        """`u` (N, C, NX) normalized -> (N, m) radial velocity (m/s)."""
        dev = u.device
        c = torch.as_tensor(self.columns, device=dev)
        k = torch.as_tensor(self.k0, device=dev)
        f = torch.as_tensor(self.frac, dtype=u.dtype, device=dev)
        interp = lambda ch: (1 - f) * u[:, ch + k, c] + f * u[:, ch + k + 1, c]
        up, w = interp(self.iu) * self.su, interp(self.iw) * self.sw
        sgn, cs, sn = (torch.as_tensor(v, dtype=u.dtype, device=dev) for v in (self.sign, self.cos, self.sin))
        return sgn * up * cs + w * sn

    @torch.no_grad()
    def __call__(self, ae, Z: torch.Tensor) -> np.ndarray:
        return self._from_columns(ae.decode(Z)).double().cpu().numpy()

    def truth_values(self, u_true: torch.Tensor) -> np.ndarray:
        return self._from_columns(u_true[None])[0].double().cpu().numpy()

