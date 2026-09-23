"""Fixed-step RK4 solver for the Lorenz-96 system.

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F,   i = 1..N,   indices mod N

CPU, float64 only (brief §1.2's device policy applies to every solver in this
project, not just KS's own: "Never put the solver on MPS"). Plain numpy, no
FFT needed -- Lorenz-96 has no spatial-derivative structure, just a local
(radius-2) coupling on a periodic index ring, so `np.roll` is exact and
sufficient. Unlike KS there is no stiff high-order term, so standard
(non-exponential) RK4 integrates it accurately and efficiently.

Added 2026-09-22, user-directed: "I would like to change gears a little bit
and try to test our findings in terms of downprojection and latent space
dynamics on another system. in this case the Lorenz 96 system. can you code
up a working Lorenz 96 system in 256 dimensions, and we can plug it into our
current learning paradigm and see what we get."

**Why Lorenz-96, and why this is a genuinely useful second system (not just
"a different equation"):** it shares the one structural property of KS this
whole project's local/spatially-organized-latent program actually depends
on -- a spatially EXTENSIVE system built from a spatially LOCAL nonlinear
coupling, whose chaos and attractor dimension scale roughly linearly with
system size (`D_KY ~ 0.226*L` for KS; Lorenz-96 has an analogous
near-linear-in-N scaling of its own positive-exponent count and D_KY, well
documented in the data-assimilation literature this model was originally
built to test methods against). It differs from KS in exactly the ways
that matter for stress-testing this project's own machinery: no PDE
structure at all (an ODE system on a discrete index, no continuum limit,
no spatial derivatives to synthesize), a MUCH cheaper single quadratic
nonlinearity (no fourth-derivative hyperviscosity, no `w_xx`/`w_xxxx`-style
dissipation operator for a fitted closure to have to rediscover), and a
different (and, per the literature, generally *faster* per-site) chaos
timescale. A latent-encoding/propagator pipeline that only ever worked on
KS's own specific structure (four spatial derivatives, `w*w_x` advection,
a stiff linear dissipation operator) has an obvious confound: none of its
apparent successes or failures are yet known to generalize past KS's own
narrow structural specifics. Lorenz-96 is a minimal, well-understood,
CHEAP-to-integrate second data point.

**Sign/index convention** (verified against Lorenz's own 1996 paper and
every standard reference implementation): `x_{i+1}` is the next site,
`x_{i-1}` the previous, `x_{i-2}` two sites back, all mod `N`. In
`numpy.roll` terms (`np.roll(x, -1)` shifts array *contents* left by one,
so index `i` of the result holds what was at index `i+1` of the input --
exactly `x_{i+1}` evaluated at every `i` simultaneously):

    dx/dt = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
"""

from __future__ import annotations

import numpy as np

from ks_latent.config import Lorenz96Config


def l96_rhs(x: np.ndarray, F: float) -> np.ndarray:
    """`dx/dt` for the Lorenz-96 system, vectorized over the last axis (so
    `x` may be `(N,)` for a single state or `(..., N)` for a batch --
    `np.roll`'s `axis=-1` keeps every leading batch dimension independent).
    See module docstring for the sign/index convention."""
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def l96_jacobian(x: np.ndarray, F: float) -> np.ndarray:
    """Exact `(N, N)` Jacobian `df_i/dx_j` of `l96_rhs` at state `x`
    (`F` does not appear -- it is a constant, contributes nothing to the
    derivative). From `f_i = (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F`:

        df_i/dx_j = delta_{j,i+1}*x_{i-1} + (x_{i+1}-x_{i-2})*delta_{j,i-1}
                    - delta_{j,i-2}*x_{i-1} - delta_{j,i}

    a sparse matrix with exactly 4 nonzero entries per row (offsets
    -2,-1,0,+1 from the diagonal, cyclically) -- built here as a dense
    `(N,N)` array (N=256 is small, ~512KB, cheap) since Benettin's method
    needs `tangent @ J.T` for a whole batch of directions at once and a
    dense matmul is simpler and just as fast at this size. Used by
    `rk4_step_with_tangent` and validated directly in
    `tests/unit/test_lorenz96.py::test_jacobian_matches_finite_difference`
    and, at the fixed point specifically, against the closed-form
    `fixed_point_jacobian_eigenvalues` circulant spectrum."""
    N = x.shape[-1]
    J = np.zeros((N, N), dtype=np.float64)
    idx = np.arange(N)
    x_im1 = np.roll(x, 1)  # x_{i-1} at position i
    x_diff = np.roll(x, -1) - np.roll(x, 2)  # (x_{i+1} - x_{i-2}) at position i
    J[idx, (idx + 1) % N] += x_im1  # d/dx_{i+1}
    J[idx, (idx - 1) % N] += x_diff  # d/dx_{i-1}
    J[idx, (idx - 2) % N] += -x_im1  # d/dx_{i-2}
    J[idx, idx] += -1.0  # d/dx_i
    return J


def rk4_step_with_tangent(
    x: np.ndarray, dX: np.ndarray, dt: float, F: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the base state `x` (`(N,)`) and a batch of tangent vectors
    `dX` (`(k, N)`) together, by differentiating the RK4 recursion itself
    (the standard way to get a tangent-linear integrator that is exactly
    consistent with the base scheme, not a separate approximation to it --
    same principle as `ks_latent.solver.ks.step_with_tangent`, and the
    identical pattern this project's own `tests/unit/test_lyapunov.py::
    _rk4_step_with_tangent` already uses for Lorenz-63). Used by
    `ks_latent.analysis.lyapunov.benettin` for the Lorenz-96 Lyapunov
    spectrum (`tests/unit/test_lorenz96.py::test_l96_n256_is_chaotic`)."""

    def deriv(state, tangent):
        return l96_rhs(state, F), tangent @ l96_jacobian(state, F).T

    k1s, k1t = deriv(x, dX)
    k2s, k2t = deriv(x + 0.5 * dt * k1s, dX + 0.5 * dt * k1t)
    k3s, k3t = deriv(x + 0.5 * dt * k2s, dX + 0.5 * dt * k2t)
    k4s, k4t = deriv(x + dt * k3s, dX + dt * k3t)
    x_next = x + (dt / 6.0) * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
    dX_next = dX + (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
    return x_next, dX_next


def rk4_step(x: np.ndarray, dt: float, F: float) -> np.ndarray:
    """One classic 4th-order Runge-Kutta step. No stiffness in this system
    (unlike KS's `-u_xxxx`), so a plain fixed-step explicit scheme is both
    standard in the Lorenz-96 literature and numerically sufficient --
    `test_convergence_in_dt` verifies the expected 4th-order convergence
    rate directly rather than assuming it."""
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    x0: np.ndarray, cfg: Lorenz96Config, n_steps: int, *, snapshot_every: int | None = None
) -> np.ndarray:
    """Integrate `x0` for `n_steps` solver steps of size `cfg.dt`.

    Returns an array of shape `(n_snapshots + 1, N)` including the initial
    condition at index 0, snapshotting every `snapshot_every` steps
    (default: `cfg.snapshot_every`) -- exact mirror of
    `ks_latent.solver.ks.integrate`'s own contract, so downstream dataset/
    training code needs no Lorenz-96-specific branch."""
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    x = x0.astype(np.float64).copy()
    snapshots = [x.copy()]
    for step_idx in range(1, n_steps + 1):
        x = rk4_step(x, cfg.dt, cfg.F)
        if step_idx % snapshot_every == 0:
            snapshots.append(x.copy())
    return np.stack(snapshots, axis=0)


def default_initial_condition(cfg: Lorenz96Config, rng: np.random.Generator) -> np.ndarray:
    """`F` at every site plus small random noise -- the standard Lorenz-96
    initialization (the unperturbed fixed point `x_i=F` is exactly
    stationary, see `fixed_point_jacobian_eigenvalues`'s docstring for why
    it is also unstable at `F=8`, so any nonzero perturbation grows and the
    system reaches its chaotic attractor after spin-up)."""
    return cfg.F + rng.normal(scale=0.01, size=cfg.N)


def spinup(cfg: Lorenz96Config, rng: np.random.Generator) -> np.ndarray:
    """Discard `cfg.spinup_time` of transient, return the resulting state
    -- exact mirror of `ks_latent.solver.ks.spinup`."""
    x0 = default_initial_condition(cfg, rng)
    n_spinup_steps = int(round(cfg.spinup_time / cfg.dt))
    if n_spinup_steps == 0:
        return x0
    traj = integrate(x0, cfg, n_spinup_steps, snapshot_every=n_spinup_steps)
    return traj[-1]


def fixed_point_jacobian_eigenvalues(N: int, F: float) -> np.ndarray:
    """Closed-form eigenvalues of the Jacobian at the (always exact, for
    any `F`) fixed point `x_i=F`, derived directly from the RHS:

        df_i/dx_j |_{x=F} = F*delta_{j,i+1} - F*delta_{j,i-2} - delta_{j,i}

    a circulant matrix (depends only on `(j-i) mod N`), whose eigenvalues
    are the standard circulant-matrix closed form evaluated at the N-th
    roots of unity `theta_k = 2*pi*k/N`:

        lambda_k = F*(exp(i*theta_k) - exp(-2i*theta_k)) - 1,   k = 0..N-1

    Used by `test_fixed_point_jacobian_spectrum` (the direct analytic-
    answer test this project's ground rule 1 requires: "a synthetic system
    with a known answer") -- computed here from the closed form, compared
    there against a numerical (finite-difference) Jacobian of `l96_rhs`
    itself, independent code paths that must agree."""
    theta = 2.0 * np.pi * np.arange(N) / N
    return F * (np.exp(1j * theta) - np.exp(-2j * theta)) - 1.0
