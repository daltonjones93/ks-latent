"""Physical-space local ETKF on a periodic 2D grid (task I11, 2026-10-09).

The perfect-model reference for Part II: the same observation network and noise as the latent
filters, but the ensemble is a set of true-PDE states and the analysis is done on the grid. Its
analysis error is the ceiling against which the latent filter's accuracy is quoted.

Algorithm (Hunt, Kostelich & Szunyogh 2007, Physica D 230:112, local ETKF): the grid is cut into
`block x block` analysis blocks; every grid point of a block gets the block's ensemble transform.
For block b, each observation j gets the R-localization weight rho_bj = GC(d_bj / c), with d_bj
the periodic distance from the block centre to the observation and GC the Gaspari-Cohn taper
(half-width c, zero at 2c); observations with rho_bj = 0 are dropped. With prior perturbations
Y = H X' (N x m_b), C = Y diag(rho / r^2):

    P_a = [(N-1) I / infl + C Y^T]^{-1},   w_bar = P_a C d,   W = [(N-1) P_a]^{1/2},
    X_a = x_bar + (w_bar 1^T + W)^T X'      (rows: members)

Without localization (`c=None`) and with infl = 1 this is the exact square-root ensemble Kalman
filter: the analysis mean equals x_bar + P_f H^T (H P_f H^T + R)^{-1} d with P_f = X'^T X' / (N-1),
and the analysis covariance equals (I - K H) P_f (`tests/unit/test_letkf_phys.py`).

`rtps`: relaxation to prior spread (Whitaker & Hamill 2012, Mon. Wea. Rev. 140:3078), pointwise
sigma_a <- sigma_a + alpha (sigma_b - sigma_a), applied to the analysis perturbations after the
transform. Both are standard remedies for the under-dispersion of a finite ensemble even with a
perfect model; the ceiling is quoted at the best setting of a declared sweep.
"""
from __future__ import annotations

import numpy as np

from ks_latent.da.localization import gaspari_cohn_taper


def _gc(r: np.ndarray, c: float) -> np.ndarray:
    import torch
    return gaspari_cohn_taper(torch.as_tensor(r, dtype=torch.float64), c).numpy()


class LocalETKFPhys2D:
    """`X` members are flattened `(NX * NY,)` fields, x outer / y inner. `obs_xy`: `(m, 2)` grid
    indices of the observed points. `L`: domain length (square, periodic). `c`: GC half-width in
    physical units, or None for no localization (one global transform)."""

    def __init__(self, NX: int, L: float, obs_xy: np.ndarray, r_std: float, c: float | None,
                 block: int = 4, inflation: float = 1.0, rtps: float = 0.0):
        if NX % block:
            raise ValueError(f"NX={NX} not divisible by block={block}")
        self.NX, self.L, self.r_std, self.c = NX, L, r_std, c
        self.inflation, self.rtps = inflation, rtps
        dx = L / NX
        nb = NX // block
        ix, iy = np.meshgrid(np.arange(NX), np.arange(NX), indexing="ij")
        blk = ((ix // block) * nb + (iy // block)).reshape(-1)
        self.pts = [np.nonzero(blk == b)[0] for b in range(nb * nb)]
        if c is None:
            self.used = [np.arange(obs_xy.shape[0])] * (nb * nb)
            self.rho = [np.ones(obs_xy.shape[0])] * (nb * nb)
            return
        cen = (np.arange(nb) * block + (block - 1) / 2) * dx  # block centres, physical units
        ox, oy = obs_xy[:, 0] * dx, obs_xy[:, 1] * dx
        per = lambda a: np.minimum(np.abs(a), L - np.abs(a))  # noqa: E731
        self.used, self.rho = [], []
        for bx in range(nb):
            for by in range(nb):
                d = np.hypot(per(ox - cen[bx]), per(oy - cen[by]))
                rho = _gc(d, c)
                keep = rho > 1e-10
                self.used.append(np.nonzero(keep)[0]); self.rho.append(rho[keep])

    def analyze(self, Xf: np.ndarray, y: np.ndarray, obs_idx: np.ndarray) -> np.ndarray:
        """`Xf`: `(N, NX*NY)` prior members; `y`: `(m,)`; `obs_idx`: `(m,)` flat indices of the
        observed points. Returns the `(N, NX*NY)` analysis members.

        Multi-field states (J17B, 2026-10-10): `Xf` may be `(N, k*NX*NY)`, k co-located fields stacked
        (e.g. the observed u and an unobserved hidden field v); `obs_idx` then indexes the stacked
        vector. Every field at a grid point gets that point's block transform, which is exactly the
        local ETKF for the augmented state (unobserved fields are updated through their ensemble
        covariances with the observed ones)."""
        N = Xf.shape[0]
        n = self.NX * self.NX
        if Xf.shape[1] % n:
            raise ValueError(f"state length {Xf.shape[1]} is not a multiple of NX^2 = {n}")
        k = Xf.shape[1] // n
        xm = Xf.mean(0)
        Xp = Xf - xm
        Y = Xp[:, obs_idx]
        d = y - xm[obs_idx]
        Xa = np.empty_like(Xf)
        eye = np.eye(N)
        for pts1, used, rho in zip(self.pts, self.used, self.rho):
            pts = pts1 if k == 1 else np.concatenate([pts1 + j * n for j in range(k)])
            if used.size == 0:
                Xa[:, pts] = Xf[:, pts]
                continue
            Yu = Y[:, used]
            C = Yu * (rho / self.r_std ** 2)[None, :]
            A = (N - 1) * eye / self.inflation + C @ Yu.T
            ev, V = np.linalg.eigh(A)
            if ev.min() <= 0:
                raise np.linalg.LinAlgError(f"local ETKF matrix not positive definite: min eig {ev.min():.3e}")
            Pa = (V / ev) @ V.T
            wbar = Pa @ (C @ d[used])
            W = (V * np.sqrt((N - 1) / ev)) @ V.T + wbar[:, None]
            Xa[:, pts] = xm[pts][None, :] + W.T @ Xp[:, pts]
        if self.rtps > 0:
            am = Xa.mean(0)
            Ap = Xa - am
            sb, sa = Xp.std(0, ddof=1), Ap.std(0, ddof=1)
            scale = np.where(sa > 0, 1 + self.rtps * (sb - sa) / np.where(sa > 0, sa, 1), 1.0)
            Xa = am + Ap * scale
        return Xa
