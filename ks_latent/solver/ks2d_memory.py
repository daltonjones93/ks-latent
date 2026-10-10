"""2D KS with a hidden memory field (task J17 variant B, 2026-10-10).

    u_t = -nabla^2 u - nabla^4 u - (1/2)|nabla u|^2 + eps * v
    v_t = -v / tau + kappa * C[u]          C = nabla^2 ('lap', default) or the identity ('identity')

v is never observed and never shown to the learned model: it is a memory integral of past u,
v(t) = kappa * int_0^inf exp(-s/tau) C[u](t - s) ds, so it is not a function of the current u. It mimics
an unobserved slow process (moisture, microphysics) feeding back on the observed field.

Time stepping: ETDRK4 (Kassam & Trefethen 2005) on the pair, with the DIAGONAL stiff linear parts
treated exactly (KS's K^2 - K^4 for u, -1/tau for v) and the coupling terms eps*v and kappa*C[u] put in
the explicit part together with KS's nonlinearity. Both fields are gauge-fixed to zero mean (k = 0 mode
zeroed), as `ks2d.step2d`: with C = nabla^2 the k = 0 mode of C[u] is 0 anyway; with the identity it is
removed so that v carries no free mean either. All functions accept a leading batch axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, ifft2

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import ETDRK4Operator2D, _nonlinear2d, etdrk4_coefficients_2d, get_operator2d


@dataclass(frozen=True)
class MemoryParams:
    eps: float
    tau: float
    kappa: float
    coupling: str = "lap"  # 'lap' (C = nabla^2) or 'identity'

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.coupling not in ("lap", "identity"):
            raise ValueError(f"coupling must be 'lap' or 'identity', got {self.coupling!r}")

    def tag(self) -> str:
        return f"mem_e{self.eps:g}_t{self.tau:g}_k{self.kappa:g}_{self.coupling}"


class MemoryOperator:
    """ETDRK4 coefficients for both fields at `cfg`'s grid and step."""

    def __init__(self, cfg: KSConfig2D, mp: MemoryParams, op_u: ETDRK4Operator2D | None = None, M: int = 32):
        self.mp = mp
        self.op_u = get_operator2d(cfg, M=M) if op_u is None else op_u
        self.KX, self.KY = self.op_u.KX, self.op_u.KY
        self.K2 = self.KX**2 + self.KY**2
        self.cv = etdrk4_coefficients_2d(np.full(self.K2.shape, -1.0 / mp.tau), cfg.dt, M=M)

    def coupling_hat(self, wu: np.ndarray) -> np.ndarray:
        """kappa * FFT(C[u]) from u's Fourier state (k = 0 removed)."""
        if self.mp.coupling == "lap":
            out = -self.mp.kappa * self.K2 * wu
        else:
            out = self.mp.kappa * np.array(wu, copy=True)
        out[..., 0, 0] = 0.0
        return out


def step_memory(vu: np.ndarray, vv: np.ndarray, mop: MemoryOperator) -> tuple[np.ndarray, np.ndarray]:
    """One ETDRK4 step of the coupled pair (Fourier states, any leading batch shape)."""
    cu, cv = mop.op_u.coeffs, mop.cv
    KX, KY, eps = mop.KX, mop.KY, mop.mp.eps

    def nl(wu, wv):
        return _nonlinear2d(wu, KX, KY) + eps * wv, mop.coupling_hat(wu)

    Nu, Nv = nl(vu, vv)
    au, av = cu.E2 * vu + cu.Q * Nu, cv.E2 * vv + cv.Q * Nv
    Nau, Nav = nl(au, av)
    bu, bv = cu.E2 * vu + cu.Q * Nau, cv.E2 * vv + cv.Q * Nav
    Nbu, Nbv = nl(bu, bv)
    ccu, ccv = cu.E2 * au + cu.Q * (2.0 * Nbu - Nu), cv.E2 * av + cv.Q * (2.0 * Nbv - Nv)
    Ncu, Ncv = nl(ccu, ccv)
    ou = cu.E * vu + Nu * cu.f1 + 2.0 * (Nau + Nbu) * cu.f2 + Ncu * cu.f3
    ov = cv.E * vv + Nv * cv.f1 + 2.0 * (Nav + Nbv) * cv.f2 + Ncv * cv.f3
    ou[..., 0, 0] = 0.0
    ov[..., 0, 0] = 0.0
    return ou, ov


def integrate_memory(u0: np.ndarray, v0: np.ndarray | None, cfg: KSConfig2D, n_steps: int, mp: MemoryParams,
                     *, snapshot_every: int | None = None, M: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (u, v) from `u0` (and `v0`, default 0) for `n_steps` steps. Returns `(U, V)`, each
    `(n_snapshots + 1, NX, NY)` including the initial state."""
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    mop = MemoryOperator(cfg, mp, M=M)
    u0 = np.asarray(u0, dtype=np.float64)
    u0 = u0 - u0.mean()
    v0 = np.zeros_like(u0) if v0 is None else np.asarray(v0, dtype=np.float64) - np.mean(v0)
    wu, wv = fft2(u0), fft2(v0)
    Us, Vs = [u0.copy()], [v0.copy()]
    for i in range(1, n_steps + 1):
        wu, wv = step_memory(wu, wv, mop)
        if i % snapshot_every == 0:
            Us.append(ifft2(wu).real.copy()); Vs.append(ifft2(wv).real.copy())
    return np.stack(Us), np.stack(Vs)


def memory_truth(cfg: KSConfig2D, seed: int, mp: MemoryParams, adjust_time: float = 100.0,
                 cache_dir: str = "artifacts/cache") -> tuple[np.ndarray, np.ndarray]:
    """Initial (u, v) of the J17B truth for DA seed `seed`: the standard cached spin-up u with v = 0,
    integrated `adjust_time` under the coupled system so both fields are on its attractor. Cached."""
    from pathlib import Path
    base = Path(cache_dir) / f"ks2d_spinup_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_T200_seed{seed}.npy"
    if not base.exists():
        raise FileNotFoundError(f"{base}: standard spin-up cache missing")
    adj = Path(cache_dir) / f"ks2d_{mp.tag()}_L{cfg.Lx:g}_NX{cfg.NX}_dt{cfg.dt:g}_adj{adjust_time:g}_seed{seed}.npz"
    if adj.exists():
        d = np.load(adj)
        return d["u"], d["v"]
    n = int(round(adjust_time / cfg.dt))
    U, V = integrate_memory(np.load(base), None, cfg, n, mp, snapshot_every=n)
    np.savez(adj, u=U[-1], v=V[-1])
    return U[-1], V[-1]


def spinup_memory(cfg: KSConfig2D, rng: np.random.Generator, mp: MemoryParams, M: int = 32):
    """Random small IC (as `ks2d.spinup2d`), v = 0, integrated `cfg.spinup_time`. Returns (u, v)."""
    from ks_latent.solver.ks2d import default_initial_condition2d
    n = int(round(cfg.spinup_time / cfg.dt))
    U, V = integrate_memory(default_initial_condition2d(cfg, rng), None, cfg, n, mp, snapshot_every=n, M=M)
    return U[-1], V[-1]
