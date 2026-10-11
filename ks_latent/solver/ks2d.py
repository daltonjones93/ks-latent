"""ETDRK4 spectral solver for the 2D (isotropic) Kuramoto-Sivashinsky
equation (added 2026-09-28, `docs/ks_local_field_ltransfer.tex`
"Staged next steps" Step 1).

    u_t + nabla^2 u + nabla^4 u + (1/2)|nabla u|^2 = 0,
    (x,y) in [0,Lx) x [0,Ly),  periodic

CPU, float64/complex128 only, same device policy as `ks_latent.solver.ks`
(brief §1.2: never put the solver on MPS). Uses `scipy.fft`, not
`torch.fft`.

**Why this specific 2D form, and how it differs from a naive
generalization of 1D KS.** 1D KS's nonlinearity, `-u u_x`, is a pure
divergence: `u u_x = (1/2)(u^2)_x`, which is why the 1D solver's
`_nonlinear` can compute it as one Fourier multiplier (`g = -i*k/2`)
applied to `FFT(u**2)` -- cheap and exact. There is no unique isotropic
2D divergence-form generalization of that specific term. The standard
2D KS equation in the literature (flame-front/thin-film instability,
the Michelson-Sivashinsky form) instead uses the gradient-squared
nonlinearity `-(1/2)|nabla u|^2 = -(1/2)(u_x^2 + u_y^2)`, a genuinely
different (non-divergence) nonlinear term -- computed here by spectral
differentiation in both directions, squaring and summing in physical
space, then transforming forward:

    N(v) = -0.5 * FFT( IFFT(i*kx*v)**2 + IFFT(i*ky*v)**2 )

**The linear operator, by contrast, keeps the exact same functional
form as 1D.** `-nabla^2 u - nabla^4 u` has Fourier multiplier
`Lhat(kx,ky) = K^2 - K^4` with `K^2 = kx^2 + ky^2` -- literally
`ks_latent.solver.ks.linear_operator`'s own `k**2 - k**4`, with `k`
replaced by the 2D radial wavenumber `K`. The ETDRK4 contour-integral
coefficient construction (`etdrk4_coefficients_2d` below) is therefore
the SAME algorithm as `ks.etdrk4_coefficients` -- reimplemented here
independently (not refactored out of `ks.py`) so the already-validated
1D solver carries zero risk from this new module; the only real change
is generalizing the contour-integral's own broadcasting from a 1D array
to an arbitrary-shape (here 2D) `Lhat`, via `Lhat[..., None]`/`axis=-1`
instead of `Lhat[:, None]`/`axis=1` -- a strict generalization that
would also work unchanged if applied to 1D's own `Lhat`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import fft2, ifft2

from ks_latent.config import KSConfig2D


def wavenumbers2d(Lx: float, Ly: float, NX: int, NY: int) -> tuple[np.ndarray, np.ndarray]:
    """Angular wavenumber grids `(KX, KY)`, each `(NX, NY)`, FFT frequency
    order, `indexing='ij'` (axis 0 = x, axis 1 = y, matching how physical
    fields `u[i,j] = u(x_i, y_j)` are stored throughout this module)."""
    kx = 2.0 * np.pi * np.fft.fftfreq(NX, d=Lx / NX)
    ky = 2.0 * np.pi * np.fft.fftfreq(NY, d=Ly / NY)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def linear_operator2d(KX: np.ndarray, KY: np.ndarray) -> np.ndarray:
    """`Lhat(kx,ky) = K^2 - K^4`, `K^2 = kx^2+ky^2` -- the Fourier symbol
    of `-nabla^2 u - nabla^4 u`. Same functional form as
    `ks_latent.solver.ks.linear_operator`'s `k**2 - k**4`."""
    K2 = KX**2 + KY**2
    return K2 - K2**2


@dataclass(frozen=True)
class ETDRK4Coeffs2D:
    E: np.ndarray
    E2: np.ndarray
    Q: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def etdrk4_coefficients_2d(Lhat: np.ndarray, dt: float, M: int = 32) -> ETDRK4Coeffs2D:
    """ETDRK4 coefficients via the contour-integral trick (Kassam &
    Trefethen 2005, eq. 21-25) -- see `ks_latent.solver.ks.
    etdrk4_coefficients`'s own docstring for the full derivation and why
    the contour-average (not the closed-form expression) is used. Same
    algorithm, generalized to an arbitrary-shape `Lhat` (here 2D) via
    `[..., None]` broadcasting and `axis=-1` reductions instead of
    assuming a 1D array."""
    Lhat = np.asarray(Lhat, dtype=np.complex128)
    E = np.exp(dt * Lhat)
    E2 = np.exp(dt * Lhat / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # (M,)
    LR = dt * Lhat[..., None] + r  # (..., M), broadcasts r over the trailing axis

    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1).real
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1
    ).real
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1).real
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1
    ).real

    assert E.shape == Lhat.shape
    return ETDRK4Coeffs2D(E=E, E2=E2, Q=Q, f1=f1, f2=f2, f3=f3)


@dataclass(frozen=True)
class ETDRK4Operator2D:
    KX: np.ndarray
    KY: np.ndarray
    Lhat: np.ndarray
    coeffs: ETDRK4Coeffs2D


@lru_cache(maxsize=32)
def _get_operator2d_cached(Lx: float, Ly: float, NX: int, NY: int, dt: float, M: int) -> ETDRK4Operator2D:
    KX, KY = wavenumbers2d(Lx, Ly, NX, NY)
    Lhat = linear_operator2d(KX, KY)
    coeffs = etdrk4_coefficients_2d(Lhat, dt, M=M)
    return ETDRK4Operator2D(KX=KX, KY=KY, Lhat=Lhat, coeffs=coeffs)


def modified_operator2d(cfg: KSConfig2D, lap_coef: float, M: int = 32) -> ETDRK4Operator2D:
    """Operator for `u_t + lap_coef * nabla^2 u + nabla^4 u + (1/2)|nabla u|^2 = 0` (task I28,
    2026-10-09: the imperfect-model twin's truth). `Lhat = lap_coef * K^2 - K^4`; the most
    unstable wavenumber moves from `1/sqrt(2)` to `sqrt(lap_coef / 2)`. `lap_coef = 1` is the
    standard equation (identical to `get_operator2d`)."""
    KX, KY = wavenumbers2d(cfg.Lx, cfg.Ly, cfg.NX, cfg.NY)
    K2 = KX**2 + KY**2
    Lhat = lap_coef * K2 - K2**2
    return ETDRK4Operator2D(KX=KX, KY=KY, Lhat=Lhat, coeffs=etdrk4_coefficients_2d(Lhat, cfg.dt, M=M))


def hyper_operator2d(cfg: KSConfig2D, lap_coef: float, hyper_coef: float, order: float = 4.0,
                     M: int = 32) -> ETDRK4Operator2D:
    """J17H (2026-10-10): linear operator `Lhat = a K^2 - b K^order` of the structurally wrong physics
    model `u_t = -a nabla^2 u - b (-nabla^2)^{order/2} u - (g/2)|nabla u|^2`. With `order = 6` (nabla^6
    hyperdiffusion in place of KS's nabla^4) no (a, b, g) gives the KS growth-rate curve
    `K^2 - K^4`. `order = 4, a = b = 1` is the standard operator."""
    KX, KY = wavenumbers2d(cfg.Lx, cfg.Ly, cfg.NX, cfg.NY)
    K2 = KX**2 + KY**2
    Lhat = lap_coef * K2 - hyper_coef * K2 ** (order / 2.0)
    return ETDRK4Operator2D(KX=KX, KY=KY, Lhat=Lhat, coeffs=etdrk4_coefficients_2d(Lhat, cfg.dt, M=M))


def get_operator2d(cfg: KSConfig2D, M: int = 32) -> ETDRK4Operator2D:
    """ETDRK4 operator for `cfg`, cached by `(Lx, Ly, NX, NY, dt, M)`."""
    return _get_operator2d_cached(cfg.Lx, cfg.Ly, cfg.NX, cfg.NY, cfg.dt, M)


def _nonlinear2d(v: np.ndarray, KX: np.ndarray, KY: np.ndarray) -> np.ndarray:
    """`N(v) = -0.5 * FFT(u_x**2 + u_y**2)`, `u_x`/`u_y` via spectral
    differentiation (`IFFT(i*KX*v)`/`IFFT(i*KY*v)`). See module docstring
    for why this cannot reuse the 1D solver's divergence-form trick."""
    ux = ifft2(1j * KX * v).real
    uy = ifft2(1j * KY * v).real
    return -0.5 * fft2(ux**2 + uy**2)


def cubic_term2d(w: np.ndarray, beta: float) -> np.ndarray:
    """J17A missing process (2026-10-10): the Fourier-space tendency `-beta * P[(u - mean u)^3]` of
    a batch `(..., NX, NY)` of Fourier states, P = removal of the k = 0 mode.

    The cubic is evaluated on the MEAN-REMOVED field so that the dynamics stay invariant under
    `u -> u + c`, as for standard 2D KS (see `step2d`): without that, the gauge-fixed k = 0 mode
    and ETDRK4's intermediate stages (whose mean is not zero) would feed back into the term. The
    system is therefore `u_t = KS(u) - beta * P[u'^3]` on zero-mean fields u' -- a local,
    saturating, odd damping that no member of the KS family contains."""
    w0 = np.array(w, copy=True)
    w0[..., 0, 0] = 0.0
    u = ifft2(w0, axes=(-2, -1)).real
    out = -beta * fft2(u**3, axes=(-2, -1))
    out[..., 0, 0] = 0.0
    return out


def aniso_term2d(w: np.ndarray, a, KX: np.ndarray) -> np.ndarray:
    """J17 option 2 (2026-10-10): the Fourier-space tendency `-a * u' * d_x u'` (u' = mean-removed field)
    of a batch `(..., NX, NY)` of Fourier states; `a` scalar or broadcastable per member.

    An anisotropic, Burgers-type (1D-KS-like) nonlinearity along x only: it steepens and transports
    structures in x, i.e. it DRIVES the dynamics rather than damping them. `u' u'_x = (1/2)(u'^2)_x` is an
    exact x-derivative, so its spatial mean is zero and the k = 0 gauge fix stays exact; using u' keeps
    the `u -> u + c` invariance (with the raw u, a constant c would add a uniform advection c u_x)."""
    w0 = np.array(w, copy=True)
    w0[..., 0, 0] = 0.0
    u = ifft2(w0, axes=(-2, -1)).real
    return -a * 0.5j * KX * fft2(u**2, axes=(-2, -1))  # -a (1/2) d_x (u'^2); k = 0 entry is 0 (KX = 0)


def grad_aniso_term2d(w: np.ndarray, delta, KX: np.ndarray, KY: np.ndarray) -> np.ndarray:
    """J17S candidate 1 (2026-10-10): the Fourier-space tendency `-(delta/2) P[u_x^2 - u_y^2]` of a batch
    `(..., NX, NY)` of Fourier states (P = removal of the k = 0 mode). Added to standard 2D KS it gives the
    anisotropic KS equation of ion-eroded surfaces (Cuerno & Barabasi 1995; Rost & Krug 1995 -- verify),
    `u_t = -lap u - lap^2 u - (1/2)[(1+delta) u_x^2 + (1-delta) u_y^2]`: local, Markov, translation
    invariant, depending only on derivatives of u. No isotropic member of the KS family (any a, b, g)
    contains it; with an estimated g only its isotropic average is absorbed."""
    ux = ifft2(1j * KX * w, axes=(-2, -1)).real
    uy = ifft2(1j * KY * w, axes=(-2, -1)).real
    out = -0.5 * np.asarray(delta) * fft2(ux**2 - uy**2, axes=(-2, -1))
    out[..., 0, 0] = 0.0
    return out


def step2d(v: np.ndarray, op: ETDRK4Operator2D, forcing_hat: np.ndarray | None = None,
           cubic: float = 0.0, aniso: float = 0.0, ganiso: float = 0.0) -> np.ndarray:
    """One ETDRK4 step in Fourier space (Kassam & Trefethen 2005, eq. 20)
    -- identical algebraic structure to `ks_latent.solver.ks.step`, only
    the nonlinear-term evaluation differs (see `_nonlinear2d`).

    **Zeros the k=0 (mean) Fourier mode after every step -- exact, not
    an approximation.** Every term on this equation's RHS
    (`-nabla^2 u`, `-nabla^4 u`, `-(1/2)|nabla u|^2`) is a function of
    `u`'s spatial DERIVATIVES only, never of `u` itself, so the dynamics
    are exactly invariant under `u -> u + c` for any constant `c` --
    `mean(u)` carries no physical content and is a free, undriven
    degree of freedom. Unlike 1D KS (whose `-u u_x = -(1/2)(u^2)_x`
    nonlinearity is an exact x-derivative, so `mean(u)` is automatically
    conserved with no extra step needed), THIS equation's nonlinearity
    is `-(1/2)(u_x^2+u_y^2)`, whose own spatial mean is generically
    negative and nonzero (`<u_x^2+u_y^2> > 0` unless `u` is constant)
    -- confirmed empirically before this fix was added: an unprojected
    run showed a steady, unbounded downward drift in `mean(u)`
    (~-0.23/time-unit at a representative resolution) and `max|u|`
    growing past 100 well within the spin-up window, both consistent
    with mean-drift, not genuine chaos. Zeroing `v`'s `[0,0]` entry
    (the k=0 mode, `= NX*NY*mean(u)` up to FFT normalization) after
    each step discards exactly this free mode and nothing else -- the
    resulting `u(x,y,t)` is the same trajectory "gauge-fixed" to
    zero mean, losing zero information about the actual pattern."""
    c = op.coeffs
    KX, KY = op.KX, op.KY
    if forcing_hat is None:
        nl0 = lambda w: _nonlinear2d(w, KX, KY)  # noqa: E731
    else:  # I28: a time-independent forcing f(x, y) is part of the explicit term N
        nl0 = lambda w: _nonlinear2d(w, KX, KY) + forcing_hat  # noqa: E731
    if cubic == 0.0:
        nl1 = nl0
    else:  # J17A: the missing -beta u'^3 process, also explicit
        nl1 = lambda w: nl0(w) + cubic_term2d(w, cubic)  # noqa: E731
    if aniso == 0.0:
        nl2 = nl1
    else:  # J17 option 2: the anisotropic -a u' u'_x process, also explicit
        nl2 = lambda w: nl1(w) + aniso_term2d(w, aniso, KX)  # noqa: E731
    nl = nl2 if ganiso == 0.0 else (lambda w: nl2(w) + grad_aniso_term2d(w, ganiso, KX, KY))  # J17S cand. 1
    Nv = nl(v)
    a = c.E2 * v + c.Q * Nv
    Na = nl(a)
    b = c.E2 * v + c.Q * Na
    Nb = nl(b)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    Nc = nl(cc)
    v_next = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    v_next[0, 0] = 0.0
    return v_next


def integrate2d(
    u0: np.ndarray,
    cfg: KSConfig2D,
    n_steps: int,
    *,
    snapshot_every: int | None = None,
    M: int = 32,
    op: ETDRK4Operator2D | None = None,
    advect: tuple[float, float] | None = None,
    forcing: np.ndarray | None = None,
    cubic: float = 0.0,
    aniso: float = 0.0,
    ganiso: float = 0.0,
) -> np.ndarray:
    """Integrate `u0` (`(NX, NY)`) for `n_steps` solver steps of size
    `cfg.dt`. Returns `(n_snapshots + 1, NX, NY)` including the initial
    condition at index 0, snapshotting every `snapshot_every` steps
    (default: `cfg.snapshot_every`).

    `u0`'s own mean is zeroed before the first step (see `step2d`'s
    docstring for why: the dynamics only ever depend on `u`'s spatial
    derivatives, so a nonzero starting mean is pure gauge, not physical
    content, and would otherwise persist as an additive offset on top
    of `step2d`'s own zero-mean-preserving evolution)."""
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    if op is None:
        op = get_operator2d(cfg, M=M)
    # I28 structural model error (2026-10-09). `advect=(cx, cy)`: uniform mean-flow advection
    # u_t + c.grad(u) + ..., applied EXACTLY as a Fourier phase shift after each step (uniform
    # advection is a translation, and the KS dynamics commute with translations). `forcing`:
    # time-independent f(x, y) (NX, NY) added to the right-hand side (zero-mean part only, since the
    # k = 0 mode is gauge). Use one or the other: with both, the splitting is no longer exact.
    if advect is not None and (forcing is not None or cubic != 0.0 or aniso != 0.0 or ganiso != 0.0):
        raise ValueError("integrate2d: advect cannot be combined with forcing or cubic (the splitting is exact only alone)")
    # J17A: `cubic = beta` adds the missing process -beta * P[u'^3] (see `cubic_term2d`).
    shift = None
    if advect is not None:
        shift = np.exp(-1j * (advect[0] * op.KX + advect[1] * op.KY) * cfg.dt)
    forcing_hat = None if forcing is None else fft2(np.asarray(forcing, dtype=np.float64))
    u0 = u0.astype(np.float64) - u0.astype(np.float64).mean()
    v = fft2(u0)
    snapshots = [u0.copy()]
    for step_idx in range(1, n_steps + 1):
        v = step2d(v, op, forcing_hat, cubic, aniso, ganiso)
        if shift is not None:
            v = v * shift
        if step_idx % snapshot_every == 0:
            snapshots.append(ifft2(v).real.copy())
    return np.stack(snapshots, axis=0)


def default_initial_condition2d(cfg: KSConfig2D, rng: np.random.Generator) -> np.ndarray:
    """Small-amplitude random low-wavenumber initial condition, 2D
    analogue of `ks.default_initial_condition` -- a smooth perturbation
    that grows via the linear instability and saturates onto the
    chaotic attractor after `cfg.spinup_time`. `n_modes_per_axis=4`
    (16 total Fourier terms) with amplitude scaled by `1/n_modes_per_axis`
    (not 1D's flat `0.1`) so the total injected variance stays
    comparable to the 1D case despite the extra terms from the second
    sum, rather than growing with the number of modes."""
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    X, Y = np.meshgrid(x, y, indexing="ij")
    n_modes_per_axis = 4
    u0 = np.zeros((cfg.NX, cfg.NY))
    amp_scale = 0.1 / n_modes_per_axis
    for j in range(1, n_modes_per_axis + 1):
        for l in range(1, n_modes_per_axis + 1):
            a = rng.normal(scale=amp_scale)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            u0 += a * np.cos(2.0 * np.pi * j * X / cfg.Lx + 2.0 * np.pi * l * Y / cfg.Ly + phi)
    return u0


def step_with_tangent2d(
    v: np.ndarray, dV: np.ndarray, op: ETDRK4Operator2D, cubic: float = 0.0, aniso: float = 0.0,
    ganiso: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the base state `v` (`(NX,NY)`) and a batch of tangent
    vectors `dV` (`(k, NX, NY)`, complex, Fourier space) together, for
    `ks_latent.analysis.lyapunov`'s Benettin's-method machinery -- 2D
    analogue of `ks_latent.solver.ks.step_with_tangent`.

    The nonlinear term here is NOT bilinear the way 1D's `g*FFT(u**2)`
    is -- it's a sum of two squared LINEAR functionals of `u` (`u_x`,
    `u_y`, both linear in `u` via spectral differentiation). Its Frechet
    derivative at `u` in direction `du` is still exact and cheap:
    `F(u) = -(1/2)(u_x^2+u_y^2)`, `u_x(u+eps*du) = u_x(u) + eps*u_x(du)`
    (linearity of differentiation), so
    `DF(u)[du] = -(u_x(u)*u_x(du) + u_y(u)*u_y(du))`
    -- reusing the base trajectory's own ETDRK4 stage values at each of
    the four RK stages, exactly the same variational-integrator pattern
    `step_with_tangent` uses."""
    c = op.coeffs
    KX, KY = op.KX, op.KY

    # J17A: with `cubic = beta`, N gains -beta P[u'^3] (`cubic_term2d`), whose derivative in direction
    # du is -3 beta P[u'^2 du'] (u', du' mean-removed). `base_nl` then also returns u'^2 (else None).
    def base_nl(state):
        ux = ifft2(1j * KX * state).real
        uy = ifft2(1j * KY * state).real
        N = -0.5 * fft2(ux**2 + uy**2)
        if ganiso != 0.0:  # J17S candidate 1: -(delta/2) P[u_x^2 - u_y^2]
            Ng = -0.5 * ganiso * fft2(ux**2 - uy**2)
            Ng[0, 0] = 0.0
            N = N + Ng
        if cubic == 0.0 and aniso == 0.0:
            return (ux, uy, None, None), N
        s0 = state.copy()
        s0[0, 0] = 0.0
        u = ifft2(s0).real
        if cubic != 0.0:
            N = N + cubic_term2d(state, cubic)
        if aniso != 0.0:
            N = N + aniso_term2d(state, aniso, KX)
        return (ux, uy, u**2, u), N

    # option 2: -a (1/2) d_x(u'^2) has derivative -a d_x(u' du') = -a i KX FFT(u' du').
    def tangent_nl(base, dstate):
        ux, uy, u2, u1 = base
        dux = ifft2(1j * KX[None, :, :] * dstate, axes=(-2, -1)).real
        duy = ifft2(1j * KY[None, :, :] * dstate, axes=(-2, -1)).real
        out = -fft2(ux[None, :, :] * dux + uy[None, :, :] * duy, axes=(-2, -1))
        if ganiso != 0.0:  # derivative of -(delta/2)(u_x^2 - u_y^2): -delta (u_x du_x - u_y du_y)
            dg = -ganiso * fft2(ux[None, :, :] * dux - uy[None, :, :] * duy, axes=(-2, -1))
            dg[:, 0, 0] = 0.0
            out = out + dg
        if u2 is not None:
            d0 = dstate.copy()
            d0[:, 0, 0] = 0.0
            du = ifft2(d0, axes=(-2, -1)).real
            if cubic != 0.0:
                dc = -3.0 * cubic * fft2(u2[None, :, :] * du, axes=(-2, -1))
                dc[:, 0, 0] = 0.0
                out = out + dc
            if aniso != 0.0:
                out = out - aniso * 1j * KX[None] * fft2(u1[None, :, :] * du, axes=(-2, -1))
        return out

    bv, Nv = base_nl(v)
    dNv = tangent_nl(bv, dV)

    a = c.E2 * v + c.Q * Nv
    da = c.E2[None, :, :] * dV + c.Q[None, :, :] * dNv
    ba, Na = base_nl(a)
    dNa = tangent_nl(ba, da)

    b = c.E2 * v + c.Q * Na
    db = c.E2[None, :, :] * dV + c.Q[None, :, :] * dNa
    bb, Nb = base_nl(b)
    dNb = tangent_nl(bb, db)

    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    dcc = c.E2[None, :, :] * da + c.Q[None, :, :] * (2.0 * dNb - dNv)
    bc, Nc = base_nl(cc)
    dNc = tangent_nl(bc, dcc)

    v_next = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    dV_next = (
        c.E[None, :, :] * dV
        + dNv * c.f1[None, :, :]
        + 2.0 * (dNa + dNb) * c.f2[None, :, :]
        + dNc * c.f3[None, :, :]
    )
    # Same gauge-fix as `step2d`, applied to BOTH the base state and every
    # tangent direction: the base trajectory's k=0 mode is pinned to zero
    # by construction, so a tangent direction's own k=0 growth would be
    # measuring sensitivity of a coordinate that has no dynamics on the
    # base trajectory at all -- spurious for a Lyapunov computation, not a
    # real direction of the physically meaningful (zero-mean) phase space.
    v_next[0, 0] = 0.0
    dV_next[:, 0, 0] = 0.0
    return v_next, dV_next


def spinup2d(cfg: KSConfig2D, rng: np.random.Generator, *, M: int = 32, cubic: float = 0.0,
             aniso: float = 0.0, ganiso: float = 0.0) -> np.ndarray:
    """Discard `cfg.spinup_time` of transient, return the resulting `(NX,NY)` state. `cubic`: J17A's
    missing-process strength (see `cubic_term2d`)."""
    u0 = default_initial_condition2d(cfg, rng)
    n_spinup_steps = int(round(cfg.spinup_time / cfg.dt))
    if n_spinup_steps == 0:
        return u0
    traj = integrate2d(u0, cfg, n_spinup_steps, snapshot_every=n_spinup_steps, M=M, cubic=cubic, aniso=aniso,
                       ganiso=ganiso)
    return traj[-1]
