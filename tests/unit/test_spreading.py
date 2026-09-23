"""Tests for the D7 information-spreading/light-cone analysis (brief §4).

Ground rule 1: validate against a synthetic system with a known analytic
answer before ever pointing this at KS. The synthetic system is a linear
advection-diffusion-growth PDE

    dv/dt = -c dv/dx + a*v + D d^2v/dx^2

which, for a Gaussian initial condition, has an exact closed-form solution
at every t (no numerical integration needed): a Gaussian of width
w(t)^2 = w0^2 + 2*D*t, drifting at speed c, with peak amplitude
exp(a*t) * w0/w(t). Its comoving exponent has the well-known "pulled front"
asymptotic form Lambda(v) = a - (v-c)^2/(4D), so the light-cone edge is
v_star = c + 2*sqrt(a*D) exactly, in the large-t limit (brief §4's cited
Deissler & Kaneko / Pikovsky & Politi framework is exactly this).
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.spreading import (
    circular_distance,
    comoving_exponents,
    find_v_star,
    fit_front_velocity,
    front_radius_series,
    light_cone_half_width,
    minimum_localization_radius,
    required_stencil_half_width_sites,
)


def _exact_gaussian_advection_diffusion_growth(
    times: np.ndarray, L: float, NX: int, x0: float, a: float, c: float, D: float, w0: float
) -> np.ndarray:
    x = np.arange(NX) * L / NX
    xs = x - x0
    xs = np.where(xs > L / 2, xs - L, xs)
    xs = np.where(xs < -L / 2, xs + L, xs)
    profiles = np.empty((len(times), NX))
    for i, t in enumerate(times):
        wt2 = w0**2 + 2 * D * t
        profiles[i] = np.exp(a * t) * (w0 / np.sqrt(wt2)) * np.exp(-((xs - c * t) ** 2) / (2 * wt2))
    return profiles


_SYNTH_PARAMS = dict(L=1000.0, NX=4096, x0=0.0, a=0.15, c=0.3, D=0.5, w0=1.0)


def test_comoving_exponent_recovers_analytic_v_star():
    theory_v_star = _SYNTH_PARAMS["c"] + 2 * np.sqrt(_SYNTH_PARAMS["a"] * _SYNTH_PARAMS["D"])
    times = np.linspace(0.1, 100.0, 300)
    profiles = _exact_gaussian_advection_diffusion_growth(times, **_SYNTH_PARAMS)
    log_abs = np.log(np.abs(profiles) + 1e-300)

    v_grid = np.linspace(-2.0, 3.0, 51)
    lambdas = comoving_exponents(
        times, log_abs, _SYNTH_PARAMS["x0"], _SYNTH_PARAMS["L"], v_grid, fit_window_frac=(0.7, 0.95)
    )
    v_star = find_v_star(v_grid, lambdas)
    assert v_star == pytest.approx(theory_v_star, rel=0.05)


def test_comoving_exponent_is_stride_independent():
    """Sampling the same underlying continuous solution more coarsely in
    physical time must not change the measured v_star. If it does, the
    estimator is implicitly using step index instead of the `times` array
    somewhere -- exactly the dt_snap-loss failure mode the brief warns about.
    """
    times_fine = np.linspace(0.1, 100.0, 300)
    times_coarse = times_fine[::5]

    v_grid = np.linspace(-2.0, 3.0, 51)
    stars = []
    for times in (times_fine, times_coarse):
        profiles = _exact_gaussian_advection_diffusion_growth(times, **_SYNTH_PARAMS)
        log_abs = np.log(np.abs(profiles) + 1e-300)
        lambdas = comoving_exponents(
            times, log_abs, _SYNTH_PARAMS["x0"], _SYNTH_PARAMS["L"], v_grid,
            fit_window_frac=(0.7, 0.95),
        )
        stars.append(find_v_star(v_grid, lambdas))

    assert stars[0] == pytest.approx(stars[1], rel=0.1)


def test_front_tracking_recovers_known_drift_velocity():
    """Front-tracking validated against a *different* known analytic answer
    than the comoving exponent: on this toy system the growth rate `a` is
    spatially uniform (translation-invariant coefficients), so the whole
    Gaussian grows and spreads self-similarly while its *shape* drifts
    rigidly at exactly `c` -- a threshold defined relative to the current
    peak is shape-only and therefore measures the drift speed `c`, not the
    light-cone edge `v_star` (those only coincide when growth is spatially
    *inhomogeneous*, e.g. real KS tangent dynamics on a chaotically mixing
    background, which is exactly why the brief calls these "two independent
    estimators" rather than expecting them to agree on this toy problem).
    Convergence to `c` is slow (~1/sqrt(t), from the diffusive width
    correction), hence the long T and loose tolerance.
    """
    params = dict(_SYNTH_PARAMS, L=2000.0, NX=4096)
    times = np.linspace(0.1, 1000.0, 500)
    profiles = _exact_gaussian_advection_diffusion_growth(times, **params)
    norm = np.abs(profiles) / np.abs(profiles).max(axis=1, keepdims=True)

    radii = front_radius_series(times, norm, params["x0"], params["L"], threshold_frac=0.3)
    v_front = fit_front_velocity(
        times, radii, params["L"], fit_window_frac=(0.5, 0.9), max_radius_frac=0.45
    )
    assert v_front == pytest.approx(params["c"], rel=0.15)


def test_comoving_exponents_raises_on_domain_wraparound():
    """A v_grid/total_time combination where a ray would wrap around the
    periodic domain within the fit window must raise, not silently return a
    contaminated Lambda(v) -- this is exactly the bug that first showed up
    as a spurious upturn in Lambda(v) at the grid edges when measuring the
    real KS light cone with too generous a v_grid (see docs/RESULTS.md)."""
    L, NX = 20.0, 64
    times = np.linspace(0.1, 10.0, 50)
    profiles = np.ones((len(times), NX))
    with pytest.raises(ValueError, match="wrap around"):
        comoving_exponents(times, np.log(profiles), 0.0, L, np.array([5.0]))


def test_circular_distance_wraps():
    assert circular_distance(np.array([1.0, 99.0]), 0.0, 100.0) == pytest.approx([1.0, 1.0])


def test_find_v_star_raises_when_grid_does_not_bracket():
    v_grid = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="widen"):
        find_v_star(v_grid, np.full(5, 1.0))
    with pytest.raises(ValueError, match="narrow/shift"):
        find_v_star(v_grid, np.full(5, -1.0))


def test_derived_bounds_formulas():
    v_star, dt, stride, h = 1.2, 0.05, 5, 3.125
    lc = light_cone_half_width(v_star, dt, stride)
    assert lc == pytest.approx(1.2 * 0.05 * 5)
    assert required_stencil_half_width_sites(v_star, dt, stride, h) == pytest.approx(lc / h)
    assert minimum_localization_radius(v_star, dt, stride) == pytest.approx(lc)
    assert minimum_localization_radius(v_star, dt, stride, encoder_receptive_field=100.0) == 100.0
    assert minimum_localization_radius(v_star, dt, stride, encoder_receptive_field=0.01) == pytest.approx(lc)
