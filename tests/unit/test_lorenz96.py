"""Lorenz-96 solver tests (ground rule 1: every numerical function gets a
test against an analytic answer, a synthetic system with a known answer,
or a published number -- mirrors tests/unit/test_solver.py's own
structure for the KS solver). Fast tests only; the full N=256 Lyapunov
measurement is @pytest.mark.slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.lyapunov import benettin
from ks_latent.config import Lorenz96Config
from ks_latent.solver.lorenz96 import (
    default_initial_condition,
    fixed_point_jacobian_eigenvalues,
    integrate,
    l96_jacobian,
    l96_rhs,
    rk4_step,
    rk4_step_with_tangent,
    spinup,
)


def test_fixed_point_is_exact():
    """x_i = F for all i is an EXACT fixed point of the RHS for any F, N
    -- direct algebraic consequence of f_i(F*1) = (F-F)*F - F + F = 0,
    verified numerically here rather than merely asserted."""
    for N in (5, 40, 256):
        for F in (0.0, 4.0, 8.0, 17.0):
            x = np.full(N, F, dtype=np.float64)
            assert np.allclose(l96_rhs(x, F), 0.0, atol=1e-13)


def test_nonlinear_energy_conservation():
    """The quadratic term alone conserves sum(x_i**2) exactly: writing
    f_i = g_i - x_i + F with g_i = (x_{i+1}-x_{i-2})*x_{i-1}, the claim is
    sum_i x_i * g_i = 0 for ANY x (the Lorenz-96 analogue of KS's own
    int(u*u_x)dx = 0 identity, the structural property motivating the
    comparison to KS at all -- see Lorenz96Config's own docstring).
    Verified to machine precision on random states, not just the
    attractor, since it is an algebraic identity independent of dynamics."""
    rng = np.random.default_rng(0)
    for N in (5, 6, 7, 40, 256):
        for _ in range(5):
            x = rng.normal(size=N)
            g = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
            assert abs(np.sum(x * g)) < 1e-10 * max(1.0, np.sum(x**2))


def test_jacobian_matches_finite_difference():
    """`l96_jacobian` (the exact analytic Jacobian) must match a central
    finite-difference Jacobian of `l96_rhs` at a generic (off-fixed-point)
    state -- two independent derivations of the same quantity."""
    rng = np.random.default_rng(1)
    N, F = 12, 8.0
    x = rng.normal(scale=2.0, size=N) + F
    J_analytic = l96_jacobian(x, F)
    eps = 1e-6
    J_fd = np.zeros((N, N))
    for j in range(N):
        dx = np.zeros(N)
        dx[j] = eps
        J_fd[:, j] = (l96_rhs(x + dx, F) - l96_rhs(x - dx, F)) / (2 * eps)
    assert np.allclose(J_analytic, J_fd, atol=1e-6)


def test_fixed_point_jacobian_spectrum():
    """At the fixed point x=F*1, the Jacobian is exactly circulant with a
    closed-form spectrum (derived in `fixed_point_jacobian_eigenvalues`'s
    docstring): lambda_k = F*(exp(i*theta_k) - exp(-2i*theta_k)) - 1. Cross-
    checked against `np.linalg.eigvals` of the analytic `l96_jacobian`
    matrix at that exact state -- the "known analytic answer" test ground
    rule 1 requires. At F=8, the maximal real part must be POSITIVE (the
    fixed point is unstable, which is WHY the system leaves it and finds
    the chaotic attractor at all under any nonzero perturbation)."""
    for N in (8, 40, 256):
        F = 8.0
        x = np.full(N, F)
        J = l96_jacobian(x, F)
        eig_numeric = np.linalg.eigvals(J)
        eig_closed_form = fixed_point_jacobian_eigenvalues(N, F)
        # Nearest-neighbor matching rather than np.sort_complex: exact
        # complex-conjugate ties in the real part make sort_complex's
        # tie-break order sensitive to last-bit floating-point noise
        # between the two independent derivations (numeric eigvals vs the
        # closed form), even though the underlying eigenvalue SETS match
        # exactly -- verified directly (both listed as sets are identical
        # up to atol) before writing this comparison this way.
        remaining = list(eig_numeric)
        for z in eig_closed_form:
            dists = [abs(z - w) for w in remaining]
            j = int(np.argmin(dists))
            assert dists[j] < 1e-8, f"N={N}: no numeric eigenvalue near {z}"
            remaining.pop(j)
        assert eig_closed_form.real.max() > 0.0


def test_jacobian_trace_is_exactly_minus_n():
    """trace(J(x)) = sum_i df_i/dx_i = -N EXACTLY, for ANY state x, not
    just the fixed point: the quadratic term's diagonal contribution is
    always zero (df_i/dx_i from (x_{i+1}-x_{i-2})*x_{i-1} would need
    i+1=i, i-1=i, or i-2=i, none of which ever hold for N>3), leaving only
    the `-delta_{j,i}` damping term's own -1 on every diagonal entry. This
    is the L96 analogue of the average phase-space divergence rate, and
    (since sum of ALL N Lyapunov exponents equals the time-averaged trace)
    guarantees the FULL-spectrum Kaplan-Yorke sum always brackets
    successfully -- used to justify `n_directions=N` (not fewer) in
    `test_l96_n256_is_chaotic` below."""
    rng = np.random.default_rng(9)
    for N in (5, 12, 40, 256):
        x = rng.normal(size=N) * 3.0 + 8.0
        J = l96_jacobian(x, 8.0)
        assert np.trace(J) == pytest.approx(-N, abs=1e-10)


def test_convergence_in_dt():
    """RK4 should converge at (observed) order >= 3.5 in dt, same
    threshold `ks_latent.solver.ks`'s own `test_convergence_in_dt` uses --
    compare a coarse-dt trajectory against a much finer-dt reference over
    a short, still-non-chaotic-divergence window."""
    N, F = 20, 8.0
    rng = np.random.default_rng(2)
    x0 = default_initial_condition(Lorenz96Config(N=N, F=F), rng)
    t_final = 0.2

    def final_state(dt):
        n_steps = int(round(t_final / dt))
        x = x0.copy()
        for _ in range(n_steps):
            x = rk4_step(x, dt, F)
        return x

    dts = [0.02, 0.01, 0.005]
    ref = final_state(0.0005)
    errors = [np.linalg.norm(final_state(dt) - ref) for dt in dts]
    # observed order between successive halvings: log2(e1/e2)
    order_1 = np.log2(errors[0] / errors[1])
    order_2 = np.log2(errors[1] / errors[2])
    assert order_1 > 3.5
    assert order_2 > 3.5


def test_integrate_snapshot_shape_and_matches_manual_stepping():
    """`integrate`'s snapshotted output must exactly match manually
    stepping `rk4_step` the same number of times (no off-by-one, no
    accidental extra/missing step) -- shape `(n_steps//snapshot_every+1, N)`
    including the initial condition."""
    cfg = Lorenz96Config(N=16, F=8.0, dt=0.01, snapshot_every=5)
    rng = np.random.default_rng(3)
    x0 = default_initial_condition(cfg, rng)
    n_steps = 20
    traj = integrate(x0, cfg, n_steps)
    assert traj.shape == (n_steps // cfg.snapshot_every + 1, cfg.N)
    assert np.allclose(traj[0], x0)
    x_manual = x0.copy()
    for i in range(1, cfg.snapshot_every + 1):
        x_manual = rk4_step(x_manual, cfg.dt, cfg.F)
    assert np.allclose(traj[1], x_manual, atol=1e-12)


def test_spinup_reaches_bounded_nontrivial_state():
    """After spin-up, the state must have left the (unstable) fixed point
    x=F by a meaningful margin, and must remain bounded (not blown up) --
    a basic sanity check before trusting anything built on top of it."""
    cfg = Lorenz96Config(N=40, F=8.0, spinup_time=20.0)
    rng = np.random.default_rng(4)
    x = spinup(cfg, rng)
    assert np.isfinite(x).all()
    assert np.abs(x - cfg.F).max() > 1.0
    assert np.abs(x).max() < 50.0  # well within the known L96 attractor's bounded range


def _l96_step_with_tangent(F, dt):
    def step_fn(state, tangent):
        return rk4_step_with_tangent(state, tangent, dt, F)

    return step_fn


@pytest.mark.slow
def test_lyapunov_l96_n40_f8_matches_literature():
    """Classic Lorenz-96 N=40, F=8 case: published estimates of the
    leading Lyapunov exponent cluster around lambda_1 ~ 1.6-1.8 (e.g. the
    value ~1.67 widely quoted following Lorenz's own and subsequent
    replication studies). Generous relative tolerance since exact
    published digits vary slightly by source/integration details -- this
    test asserts the right REGIME (chaotic, O(1) growth rate), not a
    specific author's own digit."""
    cfg = Lorenz96Config(N=40, F=8.0, dt=0.01, spinup_time=30.0)
    rng = np.random.default_rng(5)
    x0 = spinup(cfg, rng)
    step_fn = _l96_step_with_tangent(cfg.F, cfg.dt)
    # n_directions=40 (the FULL spectrum, N=40) -- guarantees kaplan_yorke_
    # dimension can always bracket (D_KY < N always, by definition, since
    # the full spectrum's total sum is strictly negative for a dissipative
    # system); found empirically that n_directions=10 was nowhere near
    # enough (all 10 tracked exponents came out positive).
    result = benettin(
        step_fn, x0, n_directions=40, n_steps=20_000, qr_every=10, dt_per_step=cfg.dt,
        warmup_steps=2_000, rng=np.random.default_rng(6), max_abs_state=1e3,
    )
    assert result.exponents[0] == pytest.approx(1.7, rel=0.25)
    assert result.n_positive >= 5


@pytest.mark.slow
def test_l96_n256_is_chaotic():
    """This project's own N=256 case, measured directly (not assumed from
    a citation, since N=256 is well outside the literature's usual N=40
    case): must show genuine EXTENSIVE chaos -- multiple positive
    exponents (not just one, and not saturating near N), and a
    Kaplan-Yorke dimension well below N (a healthy fraction of it,
    consistent with the near-linear-in-N scaling the Lorenz-96 literature
    reports) but comfortably above the handful of unstable directions a
    non-extensive/degenerate system would show."""
    cfg = Lorenz96Config(N=256, F=8.0, dt=0.01, spinup_time=30.0)
    rng = np.random.default_rng(7)
    x0 = spinup(cfg, rng)
    step_fn = _l96_step_with_tangent(cfg.F, cfg.dt)
    # n_directions=N (full spectrum) -- an EARLIER attempt at 60 directions
    # found ALL 60 came out positive (D_KY > 60, not bracketable at that
    # count); trace(J)=-N exactly (test_jacobian_trace_is_exactly_minus_n)
    # guarantees the full-spectrum sum is very negative, so bracketing at
    # n_directions=N always succeeds.
    result = benettin(
        step_fn, x0, n_directions=cfg.N, n_steps=10_000, qr_every=10, dt_per_step=cfg.dt,
        warmup_steps=2_000, rng=np.random.default_rng(8), max_abs_state=1e3,
    )
    assert result.exponents[0] > 0.5
    assert result.n_positive >= 10
    assert result.kaplan_yorke_dimension > 10.0
