"""Truth for the imperfect-model twin experiments (task I28, 2026-10-09).

The truth obeys `u_t + a nabla^2 u + nabla^4 u + (1/2)|nabla u|^2 = 0` with `a = lap_coef`; every
filter's model (the learned latent model, and the PDE filter of I11) uses the standard `a = 1`, so
no filter has the true dynamics, as for any real system. `a = 1` returns exactly the standard
perfect-model setup (the cached spin-up state, unchanged), so existing results are reproduced.

The initial state starts from the standard cached spin-up and is integrated `adjust_time` more time
units under the modified equation so that it lies on the modified attractor; the result is cached
so that every script assimilates the identical truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import ETDRK4Operator2D, integrate2d, modified_operator2d


def twin_truth_setup(cfg: KSConfig2D, seed: int, lap_coef: float = 1.0, adjust_time: float = 100.0,
                     cache_dir: str = "artifacts/cache") -> tuple[np.ndarray, ETDRK4Operator2D | None]:
    """Returns `(u0, op)`: the truth's initial state (field units) and its operator (`None` for
    the standard equation, so callers keep the default solver path)."""
    base = Path(cache_dir) / f"ks2d_spinup_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_T200_seed{seed}.npy"
    if not base.exists():
        raise FileNotFoundError(f"{base}: standard spin-up cache missing")
    u0 = np.load(base)
    if lap_coef == 1.0:
        return u0, None
    op = modified_operator2d(cfg, lap_coef)
    adj = Path(cache_dir) / f"ks2d_twin_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_a{lap_coef:g}_adj{adjust_time:g}_seed{seed}.npy"
    if adj.exists():
        return np.load(adj), op
    n = int(round(adjust_time / cfg.dt))
    u_adj = integrate2d(u0, cfg, n, snapshot_every=n, op=op)[-1]
    np.save(adj, u_adj)
    return u_adj, op


def terrain_forcing(cfg: KSConfig2D, amplitude: float, length: float = 5.0, seed: int = 123) -> np.ndarray:
    """A fixed, smooth, zero-mean random field f(x, y) with std `amplitude` and Gaussian correlation
    length `length` (physical units): the I28 'unknown terrain' forcing. Same field for every run."""
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((cfg.NX, cfg.NY))
    kx = 2 * np.pi * np.fft.fftfreq(cfg.NX, d=cfg.Lx / cfg.NX)
    ky = 2 * np.pi * np.fft.fftfreq(cfg.NY, d=cfg.Ly / cfg.NY)
    kern = np.exp(-0.5 * (length ** 2) * (kx[:, None] ** 2 + ky[None, :] ** 2))
    f = np.fft.ifft2(np.fft.fft2(xi) * kern).real
    f -= f.mean()
    return amplitude * f / f.std()


def twin_truth(cfg: KSConfig2D, seed: int, lap_coef: float = 1.0, advect_x: float = 0.0,
               forcing_amp: float = 0.0, adjust_time: float = 100.0,
               cache_dir: str = "artifacts/cache", cubic: float = 0.0,
               aniso: float = 0.0) -> tuple[np.ndarray, dict]:
    """General I28 truth: returns `(u0, integrate_kwargs)` for `integrate2d(u0, cfg, n, **kwargs)`.
    Parametric error (`lap_coef`), missing mean-flow advection (`advect_x`, in x), or missing
    terrain-like forcing (`forcing_amp`). All zero/one: the standard perfect-model truth.
    J17A: `cubic = beta` is the missing process -beta P[u'^3] (`ks2d.cubic_term2d`)."""
    if advect_x == 0.0 and forcing_amp == 0.0 and cubic == 0.0 and aniso == 0.0:
        # parametric-only (or standard): the original cache, so every I28 run shares one truth
        u0, op = twin_truth_setup(cfg, seed, lap_coef, adjust_time, cache_dir)
        return u0, ({} if op is None else {"op": op})
    kw: dict = {}
    if lap_coef != 1.0:
        kw["op"] = modified_operator2d(cfg, lap_coef)
    if advect_x != 0.0:
        kw["advect"] = (advect_x, 0.0)
    if forcing_amp != 0.0:
        kw["forcing"] = terrain_forcing(cfg, forcing_amp)
    if cubic != 0.0:
        kw["cubic"] = cubic
    if aniso != 0.0:  # J17 option 2: -a u' u'_x
        kw["aniso"] = aniso
    base = Path(cache_dir) / f"ks2d_spinup_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_T200_seed{seed}.npy"
    if not base.exists():
        raise FileNotFoundError(f"{base}: standard spin-up cache missing")
    u0 = np.load(base)
    if not kw:
        return u0, kw
    tag = f"a{lap_coef:g}_adv{advect_x:g}_f{forcing_amp:g}" + ("" if cubic == 0.0 else f"_cub{cubic:g}") \
        + ("" if aniso == 0.0 else f"_ani{aniso:g}")
    adj = Path(cache_dir) / f"ks2d_twin_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_{tag}_adj{adjust_time:g}_seed{seed}.npy"
    if adj.exists():
        return np.load(adj), kw
    n = int(round(adjust_time / cfg.dt))
    u_adj = integrate2d(u0, cfg, n, snapshot_every=n, **kw)[-1]
    np.save(adj, u_adj)
    return u_adj, kw
