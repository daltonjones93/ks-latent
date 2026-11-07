"""Track FP, FP0: known-answer checks for a transport-structured PDE on hand-built CM1 fields.

Before learning latent fields whose law is "transport + a few sources", check that the pipeline finds
that law where physics says it exists. From raw CM1 output (total fields, SI units, grid-relative
winds; `io.read_grads`) this module builds

- the moist variables that are conserved under phase change: liquid-ice potential temperature
  theta_il = theta - (Lv q_liq + Ls q_ice) / (cp pi) (linearized; exactly invariant under condensation
  at fixed pressure, see `moist_fields`) and total water q_t = q_v + q_liq + q_ice;
- the y-vorticity eta = du/dz - dw/dx, whose Boussinesq law in 2D is D eta / Dt = -dB/dx exactly
  (pressure drops out); the anelastic correction is eta w dln(rho0)/dz, and mixing adds a Laplacian;
- the buoyancy B = g (theta'/theta_b + 0.608 q_v' - q_cond) relative to the far-field t=0 column.

Transport is tested by a semi-Lagrangian step (`semi_lagrangian`): departure points from the winds at
t and t+dt (midpoint rule, fixed-point iterations), cubic interpolation of the advected field. For a
conserved variable the step should explain most of the change that persistence misses; for theta and
q_v much less in cloud. The Lagrangian residual r = (phi(t+dt) - SL phi(t)) / dt is the material
derivative along the trajectory; its sources are found by sequentially thresholded least squares
(`stlsq`, the SINDy optimizer) on a small physical library evaluated as the trajectory average of its
values at the departure point (time t) and the arrival point (time t+dt), with bootstrap selection
frequencies (`bootstrap_stlsq`). Known answers: the coefficient of dB/dx in the eta law is -1; rain
fall-out appears in q_t's law as +V dq_p/dz with V a fall speed of a few m/s; theta's law contains the
latent heating -(Lv / (cp pi)) w dq_sat/dz in cloud with coefficient ~1, and q_v's the same
condensation with the opposite sign.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

G, CP, RD, LV, LS, P00 = 9.81, 1004.0, 287.04, 2.501e6, 2.834e6, 1.0e5
DX_M = 250.0


def exner(p: np.ndarray) -> np.ndarray:
    return (p / P00) ** (RD / CP)


def qsat_liquid(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Saturation mixing ratio over liquid (kg/kg), Bolton (1980) vapour pressure."""
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def ddx(f: np.ndarray, dx: float = DX_M) -> np.ndarray:
    return np.gradient(f, dx, axis=-1)


def ddz(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.gradient(f, z, axis=-2)


def laplacian(f: np.ndarray, z: np.ndarray, dx: float = DX_M) -> np.ndarray:
    return ddx(ddx(f, dx), dx) + ddz(ddz(f, z), z)


def moist_fields(raw: dict, z: np.ndarray, base_i: int, rho0: np.ndarray, base_profiles: tuple | None = None) -> dict:
    """Derived fields from raw CM1 arrays `(T, Z, NX)` (th K, prs Pa, qv/qc/qr/qi/qs/qg kg/kg, uinterp
    grid-relative and winterp m/s). B's reference state is `base_profiles` = (theta_b, qv_b), each (Z,),
    or, if None, column `base_i` of the first frame; `rho0` (Z,) the base density (only its
    log-derivative is used)."""
    th, p, qv = (raw[k].astype(np.float64) for k in ("th", "prs", "qv"))
    qc, qr = raw["qc"].astype(np.float64), raw["qr"].astype(np.float64)
    qi, qs, qg = (raw[k].astype(np.float64) for k in ("qi", "qs", "qg"))
    u, w = raw["uinterp"].astype(np.float64), raw["winterp"].astype(np.float64)
    pi = exner(p)
    ql, qice = qc + qr, qi + qs + qg
    if base_profiles is None:
        base_profiles = (th[0, :, base_i], qv[0, :, base_i])
    th_b = np.asarray(base_profiles[0], np.float64)[None, :, None]
    qv_b = np.asarray(base_profiles[1], np.float64)[None, :, None]
    B = G * ((th - th_b) / th_b + 0.608 * (qv - qv_b) - ql - qice)
    T = th * pi
    cloud = (qc + qi) > 1e-5
    dqsdz = ddz(qsat_liquid(T, p), z)
    dlnrho = np.gradient(np.log(rho0), z)[None, :, None]
    out = dict(u=u, w=w, theta=th, qv=qv, pi=pi, B=B,
               theta_il=th - (LV * ql + LS * qice) / (CP * pi), qt=qv + ql + qice,
               qp=qr + qs + qg, qcl=qc + qi, eta=ddz(u, z) - ddx(w),
               cond=w * cloud * dqsdz, heat_cond=LV / (CP * pi) * w * cloud * dqsdz,
               w_dlnrho=w * dlnrho)
    return out


def _frac_index(z: np.ndarray, zq: np.ndarray) -> np.ndarray:
    return np.interp(zq, z, np.arange(len(z), dtype=np.float64))


def departure_points(u0, w0, u1, w1, dt: float, z: np.ndarray, dx: float = DX_M, n_iter: int = 3):
    """Departure points (fractional z index, fractional x index), each `(Z, NX)`, of the trajectories
    arriving at every grid point at t+dt; midpoint rule with the time-averaged wind at the midpoint."""
    Z, NX = u0.shape
    ka, ia = np.meshgrid(np.arange(Z, dtype=np.float64), np.arange(NX, dtype=np.float64), indexing="ij")
    za = z[:, None] * np.ones((1, NX))
    um, wm = 0.5 * (u0 + u1), 0.5 * (w0 + w1)
    du, dw = um * dt, wm * dt
    for _ in range(n_iter):
        kmid = _frac_index(z, za - 0.5 * dw)
        imid = ia - 0.5 * du / dx
        coords = np.stack([kmid, imid])
        du = map_coordinates(um, coords, order=1, mode="nearest") * dt
        dw = map_coordinates(wm, coords, order=1, mode="nearest") * dt
    kd = _frac_index(z, za - dw)
    idp = ia - du / dx
    return kd, idp


def semi_lagrangian(phi: np.ndarray, dep: tuple, order: int = 3) -> np.ndarray:
    """`phi` (Z, NX) at time t evaluated at the departure points: the advected field at t+dt."""
    return map_coordinates(phi, np.stack(dep), order=order, mode="nearest")


def stlsq(X: np.ndarray, y: np.ndarray, threshold: float, ridge: float = 1e-6, max_iter: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares on standardized columns: coefficients (in the units of
    the raw columns) of a sparse fit y ~ X xi; terms whose standardized |coefficient| < threshold
    (relative to std(y)) are removed and the rest refitted, until the support stops changing."""
    sx = X.std(axis=0)
    if (sx == 0).any():
        raise ValueError(f"constant library columns: {np.flatnonzero(sx == 0).tolist()}")
    sy = y.std()
    if sy == 0:
        raise ValueError("constant target")
    Xs, ys = X / sx, y / sy
    keep = np.ones(X.shape[1], bool)
    xi = np.zeros(X.shape[1])
    for _ in range(max_iter):
        A = Xs[:, keep]
        c = np.linalg.solve(A.T @ A + ridge * len(ys) * np.eye(A.shape[1]), A.T @ ys)
        xi[:] = 0
        xi[keep] = c
        new = np.abs(xi) >= threshold
        if not new.any():
            xi[:] = 0
            break
        if (new == keep).all():
            break
        keep = new
    return xi * sy / sx


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.mean((y - yhat) ** 2) / np.var(y))


def bootstrap_stlsq(X: np.ndarray, y: np.ndarray, groups: np.ndarray, threshold: float, n_boot: int = 20,
                    seed: int = 0) -> dict:
    """STLSQ on `n_boot` resamples of whole groups (e.g. member-time blocks, so correlated points stay
    together): selection frequency and median coefficient of each term, plus the full-data fit."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    coefs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_of[g] for g in pick])
        coefs.append(stlsq(X[ii], y[ii], threshold))
    C = np.array(coefs)
    full = stlsq(X, y, threshold)
    return {"freq": (C != 0).mean(axis=0), "median": np.median(C, axis=0), "full": full,
            "r2_full": r2(y, X @ full)}
