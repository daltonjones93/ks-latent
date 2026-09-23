"""Tests for the generic Benettin/QR Lyapunov machinery (brief §6.2).

Validated on systems with known answers before it is ever pointed at KS:
an exactly linear map (machine-precision recovery), Lorenz-63 and the Henon
map (literature exponents to 2%), and a boundedness check.
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.lyapunov import benettin, kaplan_yorke_dimension


def test_kaplan_yorke_simple():
    # lambda = [1, 0.5, -3]: j=2 (1+0.5=1.5>=0), D_KY = 2 + 1.5/3 = 2.5
    d_ky = kaplan_yorke_dimension(np.array([1.0, 0.5, -3.0]))
    assert d_ky == pytest.approx(2.5)


def test_kaplan_yorke_all_negative():
    d_ky = kaplan_yorke_dimension(np.array([-1.0, -2.0]))
    assert d_ky == 0.0


def test_benettin_linear_map_machine_precision():
    """Lyapunov exponents of a linear map x_{n+1} = A x_n are exactly log|eig(A)|."""
    eigs = np.array([2.0, 1.0, 0.3])
    A = np.diag(eigs)

    def step_fn(x, tangent):
        # The Jacobian of a linear, state-independent map is the map itself,
        # so the base "orbit" is irrelevant to the tangent dynamics; hold it
        # fixed rather than iterating A@x, which would overflow for |eig|>1.
        return x, tangent @ A.T

    x0 = np.array([0.0, 0.0, 0.0])
    # Without a warmup, the raw Benettin estimate over N steps has an O(1/N)
    # bias from the initial (unaligned) transient even for an exactly linear,
    # time-independent map -- a known feature of the estimator, not
    # noise. A warmup burn-in lets the QR frame align to the eigenbasis
    # (geometrically fast, rate = eigenvalue ratio) before any accumulation
    # starts, after which growth per step is exactly the eigenvalues.
    result = benettin(
        step_fn, x0, n_directions=3, n_steps=200, qr_every=1, dt_per_step=1.0,
        warmup_steps=200, rng=np.random.default_rng(0),
    )
    expected = np.sort(np.log(eigs))[::-1]
    assert result.exponents == pytest.approx(expected, abs=1e-10)


def _lorenz_rhs_and_jac(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    f = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    J = np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ]
    )
    return f, J


def _rk4_step_with_tangent(rhs_jac_fn, state, tangent, dt):
    def deriv(s, t):
        f, J = rhs_jac_fn(s)
        return f, t @ J.T

    k1s, k1t = deriv(state, tangent)
    k2s, k2t = deriv(state + 0.5 * dt * k1s, tangent + 0.5 * dt * k1t)
    k3s, k3t = deriv(state + 0.5 * dt * k2s, tangent + 0.5 * dt * k2t)
    k4s, k4t = deriv(state + dt * k3s, tangent + dt * k3t)
    state_next = state + (dt / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
    tangent_next = tangent + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    return state_next, tangent_next


@pytest.mark.slow
def test_lyapunov_lorenz63():
    dt = 0.005

    def step_fn(state, tangent):
        return _rk4_step_with_tangent(_lorenz_rhs_and_jac, state, tangent, dt)

    x0 = np.array([1.0, 1.0, 1.0])
    result = benettin(
        step_fn, x0, n_directions=3, n_steps=400_000, qr_every=20, dt_per_step=dt,
        warmup_steps=20_000, rng=np.random.default_rng(1),
    )
    # Literature values (e.g. Viswanath 1998): lambda ~ (0.906, 0, -14.57).
    assert result.exponents[0] == pytest.approx(0.906, rel=0.05)
    assert abs(result.exponents[1]) < 0.05
    assert result.exponents[2] == pytest.approx(-14.57, rel=0.05)


def _henon_step_with_tangent(state, tangent, a=1.4, b=0.3):
    x, y = state
    state_next = np.array([1.0 - a * x**2 + y, b * x])
    J = np.array([[-2 * a * x, 1.0], [b, 0.0]])
    return state_next, tangent @ J.T


@pytest.mark.slow
def test_lyapunov_henon():
    x0 = np.array([0.1, 0.1])
    result = benettin(
        _henon_step_with_tangent, x0, n_directions=2, n_steps=200_000, qr_every=1,
        dt_per_step=1.0, warmup_steps=1_000, rng=np.random.default_rng(2),
    )
    # Literature values (e.g. Eckmann & Ruelle 1985): lambda ~ (0.4192, -1.6229).
    assert result.exponents[0] == pytest.approx(0.4192, rel=0.02)
    assert result.exponents[1] == pytest.approx(-1.6229, rel=0.02)


def test_benettin_raises_on_unbounded_trajectory():
    def step_fn(x, tangent):
        return 2.0 * x, 2.0 * tangent

    with pytest.raises(RuntimeError, match="escaped bound"):
        benettin(
            step_fn, np.array([1.0]), n_directions=1, n_steps=100, qr_every=1,
            dt_per_step=1.0, max_abs_state=10.0,
        )
