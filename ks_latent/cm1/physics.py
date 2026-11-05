"""Physical consistency diagnostics for decoded CM1 columns (D6).

Inputs are physical-unit column fields `(..., V, Z, NX)` in the variables of `data.VARS` or `data.VARS_SPLIT`
(u', w, theta', qv', p', sqrt(q_liq), sqrt(q_ice); g/kg for moisture), on the model levels.

- `BaseState`: base-state height, density rho0(z) and potential temperature theta0(z) from an input
  sounding (hydrostatic integration, as CM1 does). `ks_latent/cm1/base_sounding.txt` is the
  unperturbed control sounding; perturbed members differ by ~0.1 K, negligible for rho0.
- `divergence_ratio`: anelastic mass continuity, d(rho0 u)/dx + d(rho0 w)/dz, relative to the size of
  its first term. The base wind u0(z) does not depend on x, so u' gives d(rho0 u)/dx exactly. CM1 is
  compressible and its u, w here are interpolated to scalar points, so the truth's own ratio is small
  but not zero; decoded or forecast states should come close to the truth's value.
- `condensate_mass`: liquid + ice condensate per unit length of line (kg/m): sum rho0 q dz dx.
- `cold_pool_intensity`: C(x) = sqrt(2 g int_0^h (-theta'/theta0) dz), h the depth of the surface-based
  layer of negative theta' (the standard squall-line cold-pool strength, m/s).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import ICE, LIQUID, vars_for

G, CP, RD = 9.81, 1004.0, 287.04
DX_M = 250.0
BASE_SOUNDING = Path(__file__).with_name("base_sounding.txt")


@dataclass
class BaseState:
    z: np.ndarray       # (Z,) m, model level heights
    rho0: np.ndarray    # (Z,) kg/m^3
    theta0: np.ndarray  # (Z,) K

    @classmethod
    def from_sounding(cls, path=BASE_SOUNDING) -> "BaseState":
        lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
        p0, th_s, q_s = (float(v) for v in lines[0])
        rows = np.array([[float(v) for v in l[:3]] for l in lines[1:]])
        z, th, qv = rows[:, 0], rows[:, 1], rows[:, 2] / 1000.0
        thv = th * (1 + 0.61 * qv)
        pi = np.empty_like(z)
        pi_prev, z_prev, thv_prev = (p0 / 1000.0) ** (RD / CP), 0.0, th_s * (1 + 0.61 * q_s / 1000.0)
        for k in range(len(z)):
            pi_prev -= G / CP * (z[k] - z_prev) * 0.5 * (1 / thv[k] + 1 / thv_prev)
            pi[k], z_prev, thv_prev = pi_prev, z[k], thv[k]
        p = 1e5 * pi ** (CP / RD)
        rho = p / (RD * th * pi * (1 + 0.61 * qv))
        return cls(z=z, rho0=rho, theta0=th)


def _v(name: str, V: int = 7) -> int:
    return vars_for(V).index(name)


def _sq_sum(X: np.ndarray, names: tuple) -> np.ndarray:
    """Sum over the condensate species in `names` present in X of q (g/kg), from the stored sqrt(g/kg)."""
    vs = vars_for(X.shape[-3])
    return sum(X[..., vs.index(n), :, :] ** 2 for n in names if n in vs)


def divergence_ratio(X: np.ndarray, base: BaseState) -> float:
    """RMS of d(rho0 u)/dx + d(rho0 w)/dz over RMS of d(rho0 u)/dx, for fields `(..., V, Z, NX)`."""
    X = np.asarray(X, np.float64)
    r = base.rho0[:, None]
    dudx = r * np.gradient(X[..., _v("u", X.shape[-3]), :, :], DX_M, axis=-1)
    dwdz = np.gradient(r * X[..., _v("w", X.shape[-3]), :, :], base.z, axis=-2)
    return float(np.sqrt(np.mean((dudx + dwdz) ** 2) / np.mean(dudx ** 2)))


def condensate_mass(X: np.ndarray, base: BaseState) -> np.ndarray:
    """Liquid + ice condensate per unit length of line (kg/m), one value per leading index."""
    X = np.asarray(X, np.float64)
    q = (_sq_sum(X, LIQUID) + _sq_sum(X, ICE)) / 1000.0                       # kg/kg, every species
    dz = np.gradient(base.z)
    return (q * (base.rho0 * dz)[:, None]).sum(axis=(-2, -1)) * DX_M


def cold_pool_intensity(X: np.ndarray, base: BaseState) -> np.ndarray:
    """C(x) in m/s, `(..., NX)`: buoyancy integral over the surface-based layer with theta' < 0."""
    X = np.asarray(X, np.float64)
    th = X[..., _v("th", X.shape[-3]), :, :]                              # (..., Z, NX)
    neg = th < 0
    layer = np.cumprod(neg, axis=-2).astype(bool)            # contiguous from the surface
    dz = np.gradient(base.z)[:, None]
    integ = np.where(layer, -th / base.theta0[:, None] * dz, 0.0).sum(axis=-2)
    return np.sqrt(2 * G * integ)


def low_level_liquid(X: np.ndarray, base: BaseState, z_max_m: float = 1000.0) -> np.ndarray:
    """Max over levels below `z_max_m` of sqrt(q_liq) (sqrt(g/kg)): a rain proxy, `(..., NX)`."""
    k = base.z < z_max_m
    X = np.asarray(X, np.float64)
    return np.sqrt(_sq_sum(X, LIQUID))[..., k, :].max(axis=-2)        # cloud + rain in either set
