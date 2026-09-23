"""ETDRK4 spectral solver for the 1D Kuramoto-Sivashinsky equation.

    u_t + u u_x + u_xx + u_xxxx = 0,    x in [0, L),  periodic

CPU, float64/complex128 only (brief §1.2: "Never put the solver on MPS" --
a float32 spectral solve at NX=1024 will not give a trustworthy Lyapunov
spectrum). Uses `scipy.fft`, not `torch.fft`.

Method: Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs",
SIAM J. Sci. Comput. 26(4):1214-1233, 2005 (ETDRK4). In Fourier space with
angular wavenumber k = 2*pi*n/L,

    v_t = Lhat * v + N(v),
    Lhat(k) = k**2 - k**4,
    N(v) = -(i*k/2) * FFT( IFFT(v)**2 ),

where Lhat comes from -u_xx - u_xxxx (FFT multiplier of u_xx is -k**2, of
u_xxxx is +k**4) and N(v) comes from -u*u_x = -(1/2)(u**2)_x.

The ETDRK4 coefficients (Q, f1, f2, f3, and E=exp(dt*Lhat), E2=exp(dt*Lhat/2))
are computed by numerically averaging their defining contour integral over
M=32 points equally spaced on a circle around each dt*Lhat(k), rather than by
the analytic closed forms -- the analytic forms are removable-singularity
quotients (0/0 as Lhat -> 0) that lose catastrophic precision near Lhat=0,
which is exactly the neighborhood of the most physically important
wavenumbers for KS. See `etdrk4_coefficients`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import fft, ifft

from ks_latent.config import KSConfig


def wavenumbers(L: float, NX: int) -> np.ndarray:
    """Angular wavenumbers k_n = 2*pi*n/L in FFT frequency order."""
    return 2.0 * np.pi * np.fft.fftfreq(NX, d=L / NX)


def linear_operator(k: np.ndarray | float) -> np.ndarray | float:
    """Lhat(k) = k**2 - k**4, the Fourier symbol of -d2/dx2 - d4/dx4."""
    k = np.asarray(k)
    return k**2 - k**4


def nonlinear_prefactor(k: np.ndarray) -> np.ndarray:
    """g(k) = -i*k/2, so that N(v) = g * FFT(IFFT(v)**2) implements -u*u_x."""
    return -0.5j * k


@dataclass(frozen=True)
class ETDRK4Coeffs:
    E: np.ndarray
    E2: np.ndarray
    Q: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def etdrk4_coefficients(Lhat: np.ndarray, dt: float, M: int = 32) -> ETDRK4Coeffs:
    """ETDRK4 coefficients via the contour-integral trick (Kassam & Trefethen 2005, eq. 21-25).

    `Lhat` is the linear operator's Fourier symbol evaluated at each mode.
    Each coefficient is a contour average over M points equally spaced on a
    unit circle, recentered at each `dt * Lhat[n]`, of a function that is
    analytic there (the apparent Lhat=0 pole in the closed-form expression is
    removable). Averaging numerically sidesteps the 0/0 cancellation
    entirely instead of relying on floating-point cancellation to resolve it.
    """
    Lhat = np.asarray(Lhat, dtype=np.complex128)
    N = Lhat.shape[0]
    E = np.exp(dt * Lhat)
    E2 = np.exp(dt * Lhat / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # (M,)
    LR = dt * Lhat[:, None] + r[None, :]  # (N, M)

    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1).real
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1
    ).real
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1).real
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1
    ).real

    assert E.shape == (N,)
    return ETDRK4Coeffs(E=E, E2=E2, Q=Q, f1=f1, f2=f2, f3=f3)


@dataclass(frozen=True)
class ETDRK4Operator:
    k: np.ndarray
    Lhat: np.ndarray
    g: np.ndarray
    coeffs: ETDRK4Coeffs


@lru_cache(maxsize=32)
def _get_operator_cached(L: float, NX: int, dt: float, M: int) -> ETDRK4Operator:
    k = wavenumbers(L, NX)
    Lhat = linear_operator(k)
    g = nonlinear_prefactor(k)
    coeffs = etdrk4_coefficients(Lhat, dt, M=M)
    return ETDRK4Operator(k=k, Lhat=Lhat, g=g, coeffs=coeffs)


def get_operator(cfg: KSConfig, M: int = 32) -> ETDRK4Operator:
    """ETDRK4 operator for `cfg`, cached by (L, NX, dt, M) (brief §3.1)."""
    return _get_operator_cached(cfg.L, cfg.NX, cfg.dt, M)


def _nonlinear(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    u = ifft(v).real
    return g * fft(u**2)


def step(v: np.ndarray, op: ETDRK4Operator) -> np.ndarray:
    """One ETDRK4 step in Fourier space (Kassam & Trefethen 2005, eq. 20)."""
    c = op.coeffs
    g = op.g
    Nv = _nonlinear(v, g)
    a = c.E2 * v + c.Q * Nv
    Na = _nonlinear(a, g)
    b = c.E2 * v + c.Q * Na
    Nb = _nonlinear(b, g)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    Nc = _nonlinear(cc, g)
    return c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3


def integrate(
    u0: np.ndarray,
    cfg: KSConfig,
    n_steps: int,
    *,
    snapshot_every: int | None = None,
    M: int = 32,
) -> np.ndarray:
    """Integrate `u0` for `n_steps` solver steps of size `cfg.dt`.

    Returns an array of shape `(n_snapshots + 1, NX)` including the initial
    condition at index 0, snapshotting every `snapshot_every` steps
    (default: `cfg.snapshot_every`).
    """
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    op = get_operator(cfg, M=M)
    v = fft(u0.astype(np.float64))
    snapshots = [u0.astype(np.float64).copy()]
    for step_idx in range(1, n_steps + 1):
        v = step(v, op)
        if step_idx % snapshot_every == 0:
            snapshots.append(ifft(v).real.copy())
    return np.stack(snapshots, axis=0)


def default_initial_condition(cfg: KSConfig, rng: np.random.Generator) -> np.ndarray:
    """Small-amplitude random low-wavenumber initial condition.

    Standard choice for KS: a smooth, small perturbation that grows via the
    linear instability and eventually saturates onto the chaotic attractor
    after the configured spin-up time.
    """
    x = np.arange(cfg.NX) * cfg.L / cfg.NX
    n_modes = 8
    u0 = np.zeros(cfg.NX)
    amps = rng.normal(scale=0.1, size=n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
    for j, (a, phi) in enumerate(zip(amps, phases), start=1):
        u0 += a * np.cos(2.0 * np.pi * j * x / cfg.L + phi)
    return u0


def step_with_tangent(
    v: np.ndarray, dV: np.ndarray, op: ETDRK4Operator
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the base state `v` and a batch of tangent vectors `dV` together.

    `dV` has shape `(k, NX)` (complex, Fourier space): `k` tangent directions
    to be propagated by the variational (tangent-linear) flow of the KS
    equation, linearized about the base trajectory. Used by
    `ks_latent.analysis.lyapunov` for Benettin's method.

    The nonlinear term is bilinear, `N(v) = g * FFT(u**2)`, so its Frechet
    derivative at `u = IFFT(v)` in direction `du = IFFT(dv)` is exactly
    `DN(v)[dv] = g * FFT(2*u*du)`. Rather than deriving a separate tangent
    integrator, this reuses the base trajectory's own ETDRK4 stage values
    (`v, a, b, c` and the physical fields they decode to) to linearize the
    nonlinear term at each of the four RK stages -- the standard way to get a
    variational integrator consistent with an exponential-time-differencing
    base scheme, since the linear part `Lhat` is already exact (it is just
    diagonal multiplication, identical for the base and tangent flow).
    """
    c = op.coeffs
    g = op.g

    def base_nl(state):
        u = ifft(state).real
        return u, g * fft(u**2)

    def tangent_nl(u, dstate):
        du = ifft(dstate, axis=-1).real
        return g[None, :] * fft(2.0 * u[None, :] * du, axis=-1)

    u_v, Nv = base_nl(v)
    dNv = tangent_nl(u_v, dV)

    a = c.E2 * v + c.Q * Nv
    da = c.E2[None, :] * dV + c.Q[None, :] * dNv
    u_a, Na = base_nl(a)
    dNa = tangent_nl(u_a, da)

    b = c.E2 * v + c.Q * Na
    db = c.E2[None, :] * dV + c.Q[None, :] * dNa
    u_b, Nb = base_nl(b)
    dNb = tangent_nl(u_b, db)

    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    dcc = c.E2[None, :] * da + c.Q[None, :] * (2.0 * dNb - dNv)
    u_c, Nc = base_nl(cc)
    dNc = tangent_nl(u_c, dcc)

    v_next = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    dV_next = (
        c.E[None, :] * dV
        + dNv * c.f1[None, :]
        + 2.0 * (dNa + dNb) * c.f2[None, :]
        + dNc * c.f3[None, :]
    )
    return v_next, dV_next


def spinup(cfg: KSConfig, rng: np.random.Generator, *, M: int = 32) -> np.ndarray:
    """Discard `cfg.spinup_time` of transient, return the resulting state."""
    u0 = default_initial_condition(cfg, rng)
    n_spinup_steps = int(round(cfg.spinup_time / cfg.dt))
    if n_spinup_steps == 0:
        return u0
    traj = integrate(u0, cfg, n_spinup_steps, snapshot_every=n_spinup_steps, M=M)
    return traj[-1]
