"""Kolmogorov-flow solver tests (`docs/ks_local_field_ltransfer.tex`
"Staged next steps" Step 2), mirroring `tests/unit/test_solver_2d.py`'s
own Gate-1-style discipline for 2D KS -- no untested numerics (ground
rule 1). Two validation strategies are combined, per ground rule 19
("every estimator validated on synthetic data with a known answer"):

1. **Exact, literature-independent checks** that need no external number
   at all -- the laminar solution is an EXACT analytic fixed point of the
   full nonlinear equation (`test_base_flow_is_exact_fixed_point`), mean
   vorticity conservation, and the anisotropic translation-equivariance
   structure this system's forcing genuinely has (equivariant in x and in
   y-shifts by the forcing period, NOT in an arbitrary y-shift -- both the
   positive and the negative case are tested, ground rule 19's "test the
   failure modes" applied to a real physics asymmetry rather than a bug).
2. **A qualitative benchmark against the well-documented Kolmogorov-flow
   bifurcation** (Meshalkin & Sinai 1961; Chandler & Kerswell 2013 and
   others study exactly this forcing_n=4, 2*pi-periodic-box configuration):
   the laminar solution is linearly STABLE at low Reynolds number and
   UNSTABLE at a Reynolds number well into the system's known chaotic
   range -- `test_growth_rate_sign_change_with_reynolds_number`. Only the
   qualitative sign is asserted (not a specific critical Re), since this
   project's own domain (a compact 2*pi x 2*pi box, not the elongated/
   near-infinite-x domain the classical Re_c=sqrt(2) result is stated
   for) would need its own finite-domain correction to compare a precise
   number against -- stated honestly rather than asserting a borrowed
   number this test cannot actually justify.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import fft2, ifft2

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.kolmogorov import (
    base_flow_vorticity,
    default_initial_condition_kolmogorov,
    get_operator_kolmogorov,
    integrate_kolmogorov,
    linear_operator_kolmogorov,
    spinup_kolmogorov,
    step_with_tangent_kolmogorov,
)
from ks_latent.solver.ks2d import wavenumbers2d


def test_linear_operator_is_pure_diffusion():
    KX, KY = wavenumbers2d(2 * np.pi, 2 * np.pi, 8, 8)
    nu = 0.03
    Lhat = linear_operator_kolmogorov(KX, KY, nu)  # alpha=0 default
    assert Lhat[0, 0] == 0.0  # k=0 mode
    assert np.all(Lhat[1:, :] < 0.0) or np.all(Lhat[:, 1:] < 0.0)  # pure damping elsewhere
    # exact functional form
    assert np.allclose(Lhat, -nu * (KX**2 + KY**2))


def test_linear_operator_drag_is_uniform_negative_shift():
    """`alpha` shifts `Lhat` down by the SAME constant at every mode,
    including k=0 (large-scale drag applies uniformly in wavenumber, see
    `KolmogorovConfig`'s own docstring) -- the k=0 vorticity/tangent mode
    is separately pinned to zero downstream, in `step_kolmogorov`/
    `step_with_tangent_kolmogorov`, not by any special-casing in
    `linear_operator_kolmogorov` itself."""
    KX, KY = wavenumbers2d(2 * np.pi, 2 * np.pi, 8, 8)
    nu, alpha = 0.03, 0.2
    Lhat_plain = linear_operator_kolmogorov(KX, KY, nu, alpha=0.0)
    Lhat_drag = linear_operator_kolmogorov(KX, KY, nu, alpha=alpha)
    assert np.allclose(Lhat_drag - Lhat_plain, -alpha)


def test_base_flow_is_exact_fixed_point():
    """The laminar solution solves the FULL nonlinear equation exactly
    (module docstring derivation) -- integrating it forward for many
    ETDRK4 steps must reproduce it to float64 roundoff, not merely decay
    slowly toward it. This is the strongest, least ambiguous correctness
    check available for this solver and needs no external literature
    number at all. Uses the DEFAULT `alpha` (checking the re-derived
    forcing balance with drag included is actually wired correctly, not
    just the original alpha=0 case)."""
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, nu=0.05, forcing_n=4, dt=0.02)
    u0 = base_flow_vorticity(cfg)
    traj = integrate_kolmogorov(u0, cfg, n_steps=500, snapshot_every=500)
    final = traj[-1]
    assert np.max(np.abs(final - u0)) < 1e-8


def test_base_flow_is_exact_fixed_point_across_alpha_values():
    """Same check as `test_base_flow_is_exact_fixed_point`, swept over
    several `alpha` values (including `alpha=0`, recovering the original
    undamped equation exactly) -- the forcing rebalancing must hold for
    ANY drag strength, not just the one value the other test happens to use."""
    for alpha in [0.0, 0.05, 0.3, 1.0]:
        cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, nu=0.05, alpha=alpha, forcing_n=4, dt=0.02)
        u0 = base_flow_vorticity(cfg)
        traj = integrate_kolmogorov(u0, cfg, n_steps=500, snapshot_every=500)
        final = traj[-1]
        assert np.max(np.abs(final - u0)) < 1e-8, f"alpha={alpha}"


def test_mean_vorticity_conservation():
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, nu=0.02, forcing_n=4, dt=0.01)
    rng = np.random.default_rng(0)
    u0 = default_initial_condition_kolmogorov(cfg, rng)
    traj = integrate_kolmogorov(u0, cfg, n_steps=2000, snapshot_every=100)
    means = traj.mean(axis=(1, 2))
    assert np.max(np.abs(means)) < 1e-8


def test_translation_equivariance_exact_in_x():
    """No term in the equation depends explicitly on x, so x-translation
    equivariance must be EXACT, same bar as both KS variants' own tests."""
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, nu=0.02, forcing_n=4, dt=0.01)
    rng = np.random.default_rng(1)
    u0 = default_initial_condition_kolmogorov(cfg, rng)
    shift_x = 11
    shifted_u0 = np.roll(u0, shift_x, axis=0)
    traj_a = integrate_kolmogorov(shifted_u0, cfg, n_steps=100)
    traj_b = integrate_kolmogorov(u0, cfg, n_steps=100)
    traj_b_shifted = np.roll(traj_b, shift_x, axis=1)  # axis 1 = x within a (T,NX,NY) stack
    assert np.allclose(traj_a, traj_b_shifted, atol=1e-10)


def test_translation_equivariance_exact_at_forcing_period_in_y():
    """`f(y) = -nu*forcing_n**3*cos(forcing_n*y)` is exactly invariant
    under `y -> y + 2*pi/forcing_n`, so the full equation is equivariant
    under THAT SPECIFIC discrete shift (not an arbitrary one -- see
    `test_translation_NOT_equivariant_for_generic_y_shift` below)."""
    forcing_n = 4
    NY = 32
    assert NY % forcing_n == 0
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=NY, nu=0.02, forcing_n=forcing_n, dt=0.01)
    rng = np.random.default_rng(2)
    u0 = default_initial_condition_kolmogorov(cfg, rng)
    shift_y = NY // forcing_n  # grid points corresponding to exactly 2*pi/forcing_n
    shifted_u0 = np.roll(u0, shift_y, axis=1)
    traj_a = integrate_kolmogorov(shifted_u0, cfg, n_steps=100)
    traj_b = integrate_kolmogorov(u0, cfg, n_steps=100)
    traj_b_shifted = np.roll(traj_b, shift_y, axis=2)
    assert np.allclose(traj_a, traj_b_shifted, atol=1e-10)


def test_translation_NOT_equivariant_for_generic_y_shift():
    """Ground rule 19 ("test the failure modes"): `f(y)` genuinely breaks
    full y-translation invariance, so a shift that is NOT a multiple of
    the forcing period `2*pi/forcing_n` must NOT be equivariant -- if this
    test ever started passing (i.e. equivariance held for a generic
    shift), that would mean the forcing term had silently been lost or
    made shift-invariant somewhere, not that the architecture improved."""
    forcing_n = 4
    NY = 32
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=NY, nu=0.02, forcing_n=forcing_n, dt=0.01)
    rng = np.random.default_rng(2)
    u0 = default_initial_condition_kolmogorov(cfg, rng)
    shift_y = NY // forcing_n + 3  # NOT a multiple of the forcing period
    shifted_u0 = np.roll(u0, shift_y, axis=1)
    traj_a = integrate_kolmogorov(shifted_u0, cfg, n_steps=100)
    traj_b = integrate_kolmogorov(u0, cfg, n_steps=100)
    traj_b_shifted = np.roll(traj_b, shift_y, axis=2)
    rel_diff = np.linalg.norm(traj_a - traj_b_shifted) / max(np.linalg.norm(traj_b_shifted), 1e-12)
    assert rel_diff > 0.05


def test_convergence_in_dt_kolmogorov():
    """Observed order >= 3.5 vs. a dt/8 reference, short smooth (non-chaotic,
    moderate-nu) window -- same discipline as both KS variants' own
    convergence tests."""
    cfg0 = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=24, NY=24, nu=0.5, forcing_n=2, dt=0.01, snapshot_every=1)
    x = np.arange(cfg0.NX) * cfg0.Lx / cfg0.NX
    y = np.arange(cfg0.NY) * cfg0.Ly / cfg0.NY
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = base_flow_vorticity(cfg0, X, Y) + 0.3 * np.cos(3 * X) * np.cos(2 * Y)

    t_final = 0.3
    ref_cfg = cfg0.replace(dt=cfg0.dt / 8)
    n_ref = round(t_final / ref_cfg.dt)
    ref = integrate_kolmogorov(u0, ref_cfg, n_steps=n_ref, snapshot_every=n_ref)[-1]

    errors = []
    for dt in [cfg0.dt, cfg0.dt / 2]:
        cfg = cfg0.replace(dt=dt)
        n_steps = round(t_final / dt)
        out = integrate_kolmogorov(u0, cfg, n_steps=n_steps, snapshot_every=n_steps)[-1]
        errors.append(np.linalg.norm(out - ref))

    order = np.log2(errors[0] / errors[1])
    assert order >= 3.5, f"observed order {order:.2f} < 3.5 (errors={errors})"


def test_spinup_kolmogorov_produces_finite_bounded_state():
    cfg = KolmogorovConfig(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, nu=0.02, forcing_n=4, dt=0.01, spinup_time=20.0)
    rng = np.random.default_rng(4)
    u_final = spinup_kolmogorov(cfg, rng)
    assert np.isfinite(u_final).all()
    assert u_final.mean() == pytest.approx(0.0, abs=1e-8)
    assert np.abs(u_final).max() < 200.0  # sanity bound, not a tight physical target


def _leading_exponent_about_base_flow(
    cfg: KolmogorovConfig, *, n_warmup: int, n_steps: int, renorm_every: int, seed: int
) -> float:
    """Single-direction power iteration on the tangent-linear operator
    ABOUT the laminar fixed point (confirmed exact by
    `test_base_flow_is_exact_fixed_point`, so the base state stays pinned
    near it for the whole run without needing to re-derive it), using
    `step_with_tangent_kolmogorov` directly -- a bespoke, minimal-machinery
    growth-rate estimator, deliberately NOT `lyapunov_spectrum_kolmogorov`/
    `benettin` (which always computes a Kaplan-Yorke dimension internally
    and raises if not enough directions bracket it -- irrelevant noise for
    a single leading-exponent sign check like this one).

    **`n_warmup` matters, not just a nicety** -- discovered the hard way
    during this project's own parameter search (docs/RESULTS.md's "Step 2"
    progress note): a short, un-warmed-up power iteration can be dominated
    by the RANDOM initial tangent direction's transient projection onto
    non-dominant modes, giving a badly wrong (even wrong-signed) growth
    rate. `n_warmup` steps are run and discarded (letting the tangent
    direction align with the true dominant eigenvector) before `n_steps`
    of measurement begin."""
    op = get_operator_kolmogorov(cfg)
    u0 = base_flow_vorticity(cfg)
    v = fft2(u0)
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(1, cfg.NX, cfg.NY))
    d = d / np.linalg.norm(d)
    dV = fft2(d, axes=(-2, -1))

    for _ in range(0, n_warmup, renorm_every):
        for _ in range(renorm_every):
            v, dV = step_with_tangent_kolmogorov(v, dV, op)
        dV = dV / np.linalg.norm(ifft2(dV, axes=(-2, -1)).real)

    log_sum = 0.0
    for _ in range(0, n_steps, renorm_every):
        for _ in range(renorm_every):
            v, dV = step_with_tangent_kolmogorov(v, dV, op)
        d_phys = ifft2(dV, axes=(-2, -1)).real
        norm = np.linalg.norm(d_phys)
        log_sum += np.log(norm)
        dV = dV / norm
    total_time = n_steps * cfg.dt
    return log_sum / total_time


def test_growth_rate_sign_change_with_reynolds_number():
    """Qualitative benchmark against the well-documented Kolmogorov-flow
    bifurcation (module + file docstring): the laminar solution is
    linearly stable at Re=1 and unstable at Re=50 (forcing_n=4, 2*pi x
    2*pi box). **Not a claim that Re=50 is past chaos onset** -- a later,
    more careful investigation (docs/RESULTS.md's "Step 2" progress note)
    found Re=50 in this exact box sits right at a marginal/limit-cycle-like
    secondary state (leading exponent ~0 on the SATURATED nonlinear
    attractor), with genuine chaos only appearing around Re~70+; this test
    only checks the much weaker (and unambiguous) claim that the LAMINAR
    solution's own linear instability threshold lies between Re=1 and
    Re=50, not anything about the long-time nonlinear attractor."""
    base_kwargs = dict(Lx=2 * np.pi, Ly=2 * np.pi, NX=32, NY=32, forcing_n=4, dt=0.01)

    lam1_stable = _leading_exponent_about_base_flow(
        KolmogorovConfig(nu=1.0, **base_kwargs), n_warmup=300, n_steps=300, renorm_every=30, seed=0
    )  # Re=1
    lam1_unstable = _leading_exponent_about_base_flow(
        KolmogorovConfig(nu=0.02, **base_kwargs), n_warmup=300, n_steps=300, renorm_every=30, seed=0
    )  # Re=50

    assert lam1_stable < 0.0, f"expected stable at Re=1, got lambda1={lam1_stable:.4f}"
    assert lam1_unstable > 0.0, f"expected unstable at Re=50, got lambda1={lam1_unstable:.4f}"
