"""D7: information-spreading velocity and the light cone (brief §4, addendum
§12.5/§13.2, cited inline).

In spatiotemporal chaos, a local perturbation spreads at a finite velocity
rather than instantaneously (velocity-dependent / comoving Lyapunov
exponents; Deissler & Kaneko, Physica D 25:233, 1987; Pikovsky & Politi,
Nonlinearity 11:1049, 1998). This module measures that velocity for KS by
propagating the *tangent* dynamics of a narrow local perturbation alongside
a nonlinear base trajectory (reusing `ks_latent.solver.ks.step_with_tangent`
with a single tangent direction), via two independent estimators that
should agree:

1. **Front tracking**: the outermost circular distance from the
   perturbation site at which `|delta_u|` still exceeds a threshold
   fraction of its own current peak, as a function of time; its slope is a
   front velocity. Swept over several thresholds to report sensitivity.
2. **Comoving exponent**: `Lambda(v) = d/dt ln|delta_u(x0 + v*t, t)|`,
   estimated by linear regression of `ln|delta_u|` along the ray `x0 + v*t`
   against physical time `t`. `v_star` is the largest `v` with
   `Lambda(v) > 0` -- the fastest direction in which the perturbation still
   grows, i.e. the edge of the light cone.

Everything here operates on *physical time* (via an explicit `times` array),
never on step counts -- `test_comoving_exponent_is_stride_independent`
exists specifically to catch `dt_snap`/stride bugs that would otherwise
silently corrupt every downstream bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, ifft

from ks_latent.config import KSConfig
from ks_latent.solver.ks import get_operator, spinup, step_with_tangent

_TINY = 1e-300


def circular_distance(x: np.ndarray | float, x0: float, L: float) -> np.ndarray | float:
    d = np.abs(np.asarray(x) - x0) % L
    return np.minimum(d, L - d)


def narrow_bump(NX: int, L: float, x0: float, width: float) -> np.ndarray:
    """A single Gaussian bump of the given physical `width`, centered at `x0`,
    wrapped periodically."""
    x = np.arange(NX) * L / NX
    d = circular_distance(x, x0, L)
    return np.exp(-0.5 * (d / width) ** 2)


def _periodic_interp(profile: np.ndarray, L: float, x_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of a periodic 1D field sampled on a uniform grid."""
    NX = profile.shape[-1]
    h = L / NX
    xq = np.mod(x_query, L)
    idx_f = xq / h
    i0 = np.floor(idx_f).astype(int) % NX
    i1 = (i0 + 1) % NX
    frac = idx_f - np.floor(idx_f)
    return profile[..., i0] * (1 - frac) + profile[..., i1] * frac


def comoving_exponents(
    times: np.ndarray,
    log_abs_profiles: np.ndarray,
    x0: float,
    L: float,
    v_grid: np.ndarray,
    fit_window_frac: tuple[float, float] = (0.2, 0.9),
) -> np.ndarray:
    """Lambda(v) for each v in `v_grid` from log-amplitude interpolation along
    x0 + v*t, fit over the middle portion of `times` (skips the initial
    transient before the tangent field has settled and the tail where a
    high-|v| ray may have outrun any real signal into the bump's tails).

    `log_abs_profiles` has shape `(len(times), NX)`, already including any
    cumulative renormalization offset (i.e. it is the true `ln|delta_u(x,t)|`,
    not the log of a periodically-rescaled working copy).
    """
    times = np.asarray(times)
    t0, t1 = times[0], times[-1]
    lo = t0 + fit_window_frac[0] * (t1 - t0)
    hi = t0 + fit_window_frac[1] * (t1 - t0)
    mask = (times >= lo) & (times <= hi)
    t_fit = times[mask]

    max_displacement = np.abs(v_grid).max() * max(abs(hi), abs(lo))
    if max_displacement > 0.45 * L:
        raise ValueError(
            f"v_grid reaches |v|*t up to {max_displacement:.3g}, which exceeds "
            f"0.45*L={0.45 * L:.3g}: some ray x0+v*t would wrap around the periodic "
            "domain within the fit window and re-approach the perturbation site, "
            "contaminating Lambda(v) at the grid edges with a spurious upturn. "
            "Narrow v_grid, shorten fit_window_frac, or shorten the trial's total_time."
        )

    lambdas = np.empty(len(v_grid))
    for j, v in enumerate(v_grid):
        x_query = x0 + v * t_fit
        y = np.array(
            [_periodic_interp(log_abs_profiles[i], L, np.array([x_query[k]]))[0]
             for k, i in enumerate(np.nonzero(mask)[0])]
        )
        slope, _ = np.polyfit(t_fit, y, 1)
        lambdas[j] = slope
    return lambdas


def find_v_star(v_grid: np.ndarray, lambdas: np.ndarray) -> float:
    """Largest v with Lambda(v) > 0, by linear interpolation at the last
    sign change from positive to negative as v increases.

    Raises if `lambdas` never crosses zero over `v_grid` -- that means the
    grid doesn't bracket the light cone, not that one doesn't exist.
    """
    order = np.argsort(v_grid)
    v_sorted, lam_sorted = v_grid[order], lambdas[order]
    positive = lam_sorted > 0
    if positive.all():
        raise ValueError(
            "Lambda(v) > 0 for the entire v_grid; widen the grid to bracket v_star."
        )
    if not positive.any():
        raise ValueError(
            "Lambda(v) <= 0 for the entire v_grid; narrow/shift the grid to bracket v_star."
        )
    # Last index where it's still positive, immediately followed by non-positive.
    idx = np.where(positive[:-1] & ~positive[1:])[0]
    if len(idx) == 0:
        raise ValueError("Lambda(v) is not a single decreasing crossing over v_grid.")
    i = idx[-1]
    v_a, v_b = v_sorted[i], v_sorted[i + 1]
    lam_a, lam_b = lam_sorted[i], lam_sorted[i + 1]
    frac = lam_a / (lam_a - lam_b)
    return float(v_a + frac * (v_b - v_a))


def front_radius_series(
    times: np.ndarray, abs_profiles: np.ndarray, x0: float, L: float, threshold_frac: float
) -> np.ndarray:
    """Outermost circular distance from x0 where `abs_profiles` exceeds
    `threshold_frac` times its own peak, at each recorded time."""
    NX = abs_profiles.shape[-1]
    x = np.arange(NX) * L / NX
    d = circular_distance(x, x0, L)
    radii = np.empty(len(times))
    for i in range(len(times)):
        peak = abs_profiles[i].max()
        above = abs_profiles[i] > threshold_frac * peak
        radii[i] = d[above].max() if above.any() else 0.0
    return radii


def fit_front_velocity(
    times: np.ndarray,
    radii: np.ndarray,
    L: float,
    fit_window_frac: tuple[float, float] = (0.2, 0.9),
    max_radius_frac: float = 0.4,
) -> float:
    """Linear-regression slope of front radius vs. time, restricted to the
    window where the front is neither still forming nor near wrapping around
    the periodic domain (radius < max_radius_frac * L)."""
    t0, t1 = times[0], times[-1]
    lo = t0 + fit_window_frac[0] * (t1 - t0)
    hi = t0 + fit_window_frac[1] * (t1 - t0)
    mask = (times >= lo) & (times <= hi) & (radii < max_radius_frac * L)
    if mask.sum() < 2:
        raise ValueError("Not enough valid samples to fit a front velocity.")
    slope, _ = np.polyfit(times[mask], radii[mask], 1)
    return float(slope)


@dataclass(frozen=True)
class SpreadingResult:
    v_grid: np.ndarray
    lambda_mean: np.ndarray
    lambda_std: np.ndarray
    v_star_mean: float
    v_star_std: float
    front_velocity_mean: dict[float, float]
    front_velocity_std: dict[float, float]
    n_trials: int
    total_time: float


def _run_single_ks_trial(
    cfg: KSConfig,
    x0_idx: int,
    total_time: float,
    bump_width: float,
    record_stride: int,
    seed: int,
    M: int,
    renorm_threshold: float = 1e8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One trial: returns (times, log_abs_profiles, abs_profiles_normalized, x0).

    `abs_profiles_normalized` is peak-normalized per time slice (used by
    front tracking, which only needs relative shape); `log_abs_profiles`
    carries the true cumulative log-amplitude (used by the comoving
    exponent, which needs absolute growth).
    """
    op = get_operator(cfg, M=M)
    rng = np.random.default_rng(seed)
    u0 = spinup(cfg, rng, M=M)
    x0 = x0_idx * cfg.L / cfg.NX
    tangent0 = narrow_bump(cfg.NX, cfg.L, x0, bump_width)[None, :]

    v = fft(u0)
    dV = fft(tangent0, axis=-1)
    log_scale = 0.0

    n_steps = int(round(total_time / cfg.dt))
    times, log_abs_list, norm_list = [0.0], [np.log(np.abs(tangent0[0]) + _TINY)], [
        np.abs(tangent0[0]) / np.abs(tangent0[0]).max()
    ]
    for step_idx in range(1, n_steps + 1):
        v, dV = step_with_tangent(v, dV, op)
        if step_idx % record_stride == 0:
            tangent_phys = ifft(dV, axis=-1).real[0]
            peak = np.abs(tangent_phys).max()
            if peak > renorm_threshold:
                log_scale += np.log(peak)
                dV = dV / peak
                tangent_phys = tangent_phys / peak
                peak = np.abs(tangent_phys).max()
            times.append(step_idx * cfg.dt)
            log_abs_list.append(log_scale + np.log(np.abs(tangent_phys) + _TINY))
            norm_list.append(np.abs(tangent_phys) / max(peak, _TINY))

    return (
        np.array(times),
        np.stack(log_abs_list, axis=0),
        np.stack(norm_list, axis=0),
        x0,
    )


def measure_ks_spreading(
    cfg: KSConfig,
    *,
    n_trials: int = 100,
    total_time: float = 20.0,
    v_grid: np.ndarray | None = None,
    threshold_fracs: tuple[float, ...] = (0.1, 0.3, 0.5),
    bump_width: float | None = None,
    record_stride: int = 1,
    seed: int = 0,
    M: int = 32,
) -> SpreadingResult:
    """Measure v_star for the raw KS PDE, averaged over `n_trials` independent
    base trajectories and perturbation sites (brief: ">= 100")."""
    if v_grid is None:
        v_grid = np.linspace(-4.0, 4.0, 41)
    if bump_width is None:
        bump_width = cfg.L / cfg.NX * 2  # a few grid cells wide

    lambdas_all = np.empty((n_trials, len(v_grid)))
    v_star_all = np.empty(n_trials)
    front_v_all = {tf: np.empty(n_trials) for tf in threshold_fracs}

    for trial in range(n_trials):
        rng = np.random.default_rng(10_000 + seed + trial)
        x0_idx = int(rng.integers(0, cfg.NX))
        times, log_abs, norm_abs, x0 = _run_single_ks_trial(
            cfg, x0_idx, total_time, bump_width, record_stride, seed=seed + trial, M=M
        )
        lambdas_all[trial] = comoving_exponents(times, log_abs, x0, cfg.L, v_grid)
        v_star_all[trial] = find_v_star(v_grid, lambdas_all[trial])
        for tf in threshold_fracs:
            radii = front_radius_series(times, norm_abs, x0, cfg.L, tf)
            front_v_all[tf][trial] = fit_front_velocity(times, radii, cfg.L)

    return SpreadingResult(
        v_grid=v_grid,
        lambda_mean=lambdas_all.mean(axis=0),
        lambda_std=lambdas_all.std(axis=0),
        v_star_mean=float(v_star_all.mean()),
        v_star_std=float(v_star_all.std()),
        front_velocity_mean={tf: float(front_v_all[tf].mean()) for tf in threshold_fracs},
        front_velocity_std={tf: float(front_v_all[tf].std()) for tf in threshold_fracs},
        n_trials=n_trials,
        total_time=total_time,
    )


def light_cone_half_width(v_star: float, dt: float, stride: int) -> float:
    """Physical light-cone half-width for a temporal stride of `stride`
    solver steps (brief §4: "light-cone half-width = v_* * Delta_t,
    Delta_t = dt_snap * stride"; here `dt * stride` so the caller can sweep
    `stride` independent of what `cfg.snapshot_every` happens to be)."""
    return v_star * dt * stride


def required_stencil_half_width_sites(v_star: float, dt: float, stride: int, h: float) -> float:
    """Minimum stencil half-width, in lattice sites of spacing h, implied by
    the light cone (brief §4). A learned stencil wider than this has learned
    spurious nonlocality; narrower is under-resolved."""
    return light_cone_half_width(v_star, dt, stride) / h


def minimum_localization_radius(
    v_star: float, dt: float, stride: int, encoder_receptive_field: float | None = None
) -> float:
    """Minimum valid DA localization radius (brief §4, made an assertion in
    Phase 13, not a comment)."""
    lc = light_cone_half_width(v_star, dt, stride)
    if encoder_receptive_field is None:
        return lc
    return max(encoder_receptive_field, lc)
