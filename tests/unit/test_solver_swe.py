"""Shallow water equations (SWE) solver tests (added 2026-10-06,
user-directed: "I would like to add one more model to test our L
transfer results"), mirroring `tests/unit/test_solver_kolmogorov.py`'s
own discipline -- no untested numerics (ground rule 1), every estimator/
solver validated against a known answer before being trusted on real
data (ground rule 19).

Two independent, literature-free-where-possible validation routes:

1. **The inertia-gravity wave dispersion relation**
   `omega^2 = f^2 + g*H*k^2` (`alpha=0`) -- a standard, closed-form
   result for the LINEARIZED rotating shallow water equations (Vallis,
   *Atmospheric and Oceanic Fluid Dynamics*, ch. 3; Gill, *Atmosphere-
   Ocean Dynamics*, ch. 6) -- checked directly against the eigenvalues
   of this module's own per-wavenumber linear operator matrix, with NO
   dependence on the nonlinear solver or any training/estimation step.
2. **Exact conservation laws derivable from the equations themselves**
   (mass conservation from the flux-form continuity equation; the
   tangent-linear step matching a finite-difference derivative of the
   full nonlinear step) -- these need no external literature number at
   all, the same kind of self-contained check
   `test_base_flow_is_exact_fixed_point` and
   `test_mean_vorticity_conservation` give the Kolmogorov-flow solver.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import fft2, ifft2

from ks_latent.config import SWEConfig
from ks_latent.solver.swe import (
    _diagonalize_linear_matrix,
    _nonlinear_swe,
    _to_modal,
    _to_phys,
    _wind_stress_forcing_fourier,
    build_linear_matrix_swe,
    default_initial_condition_swe,
    get_operator_swe,
    integrate_swe,
    spinup_swe,
    step_swe,
    step_with_tangent_swe,
)
from ks_latent.solver.ks2d import wavenumbers2d


def test_swe_config_rejects_bad_params():
    with pytest.raises(ValueError):
        SWEConfig(g=0.0)
    with pytest.raises(ValueError):
        SWEConfig(H=-1.0)
    with pytest.raises(ValueError):
        SWEConfig(alpha=-0.1)


def test_linear_matrix_shape_and_entries():
    KX, KY = wavenumbers2d(2 * np.pi, 2 * np.pi, 8, 8)
    g, H, f, alpha = 1.0, 1.0, 4.0, 0.05
    L = build_linear_matrix_swe(KX, KY, g, H, f, alpha)
    assert L.shape == (8, 8, 3, 3)
    assert L.dtype == np.complex128
    # k=0 entries
    assert np.allclose(L[0, 0], [[-alpha, f, 0], [-f, -alpha, 0], [0, 0, 0]])
    # a nonzero wavenumber, spot-check the pressure-gradient/divergence couplings
    ix, iy = 1, 2
    assert np.isclose(L[ix, iy, 0, 2], -1j * g * KX[ix, iy])
    assert np.isclose(L[ix, iy, 1, 2], -1j * g * KY[ix, iy])
    assert np.isclose(L[ix, iy, 2, 0], -1j * H * KX[ix, iy])
    assert np.isclose(L[ix, iy, 2, 1], -1j * H * KY[ix, iy])


def test_eigendecomposition_reconstructs_linear_matrix():
    KX, KY = wavenumbers2d(2 * np.pi, 2 * np.pi, 12, 12)
    L = build_linear_matrix_swe(KX, KY, g=1.0, H=1.0, f=4.0, alpha=0.05)
    eigvals, V, Vinv = _diagonalize_linear_matrix(L)
    recon = np.einsum("xyij,xyj,xyjk->xyik", V, eigvals, Vinv)
    assert np.abs(recon - L).max() < 1e-8


def test_eigendecomposition_raises_on_bad_rtol_for_defective_matrix():
    """An artificially defective (non-diagonalizable) matrix must be
    caught, not silently mishandled -- ground rule 2. A single repeated-
    eigenvalue Jordan block, replicated over the whole (NX,NY) grid so
    the batched shape still matches what `_diagonalize_linear_matrix`
    expects."""
    L_single = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.complex128)
    L = np.broadcast_to(L_single, (4, 4, 3, 3)).copy()
    with pytest.raises(ValueError, match="reconstruction error"):
        _diagonalize_linear_matrix(L, rtol=1e-12)


def test_inertia_gravity_wave_dispersion_relation():
    """At `alpha=0`, the per-wavenumber linear operator's eigenvalues
    must be `{0, +i*omega, -i*omega}` with `omega = sqrt(f^2+g*H*k^2)`
    (the standard closed-form rotating-shallow-water dispersion relation,
    module docstring). Checked to near machine precision -- this is an
    exact analytic property of the LINEAR operator alone, independent of
    the nonlinear solver, the strongest and most literature-independent
    check this module can get for the genuinely new (matrix-valued)
    linear-algebra machinery `_diagonalize_linear_matrix` performs."""
    g, H, f = 1.3, 0.8, 2.5
    KX, KY = wavenumbers2d(2.0 * np.pi, 2.0 * np.pi, 16, 16)
    L = build_linear_matrix_swe(KX, KY, g, H, f, alpha=0.0)
    eigvals, _, _ = _diagonalize_linear_matrix(L)

    K2 = KX**2 + KY**2
    omega_true = np.sqrt(f**2 + g * H * K2)

    real_parts = np.abs(eigvals.real)
    assert real_parts.max() < 1e-9, f"alpha=0 eigenvalues should be purely imaginary, max |Re|={real_parts.max()}"

    imag_sorted = np.sort(np.abs(eigvals.imag), axis=-1)  # ascending: [~0, omega, omega]
    assert imag_sorted[:, :, 0].max() < 1e-9, "the geostrophic (eta-null) mode should have eigenvalue 0"
    rel_err = np.abs(imag_sorted[:, :, 1:] - omega_true[:, :, None]).max() / omega_true[1:, 1:].min()
    assert rel_err < 1e-9, f"dispersion relation mismatch, rel err {rel_err:.3e}"


def test_alpha_is_an_exact_uniform_shift_in_the_decoupled_limit():
    """`alpha` enters `L` only as `-alpha` on the two momentum-equation
    diagonal entries (module docstring), so it is NOT, in general, a
    uniform shift of all three eigenvalues (Coriolis/gravity-wave
    coupling mixes momentum and `eta` in each eigenvector, so each
    mode's effective damping rate is only a FRACTION of the bare
    `alpha` -- the same physical effect as geostrophic/vortical modes
    decaying slower than `alpha` under pure momentum (Rayleigh/Ekman)
    drag, a real and well-known effect, not a bug). The one regime where
    an exact, analytically trivial `-alpha` shift IS guaranteed is the
    fully decoupled limit `f=g=H=0`, where `L` is already diagonal,
    `diag(-alpha,-alpha,0)`, independent of wavenumber -- checked here
    as the honest version of what the (incorrect, and removed) uniform-
    shift assumption would have claimed in general."""
    KX, KY = wavenumbers2d(2.0 * np.pi, 2.0 * np.pi, 8, 8)
    alpha = 0.2
    L = build_linear_matrix_swe(KX, KY, g=0.0, H=0.0, f=0.0, alpha=alpha)
    eigvals, _, _ = _diagonalize_linear_matrix(L)
    sorted_eigs = np.sort(eigvals.real, axis=-1)  # [-alpha, -alpha, 0] at every wavenumber
    assert np.allclose(sorted_eigs[:, :, 0], -alpha)
    assert np.allclose(sorted_eigs[:, :, 1], -alpha)
    assert np.allclose(sorted_eigs[:, :, 2], 0.0)


def test_modal_transform_round_trip():
    KX, KY = wavenumbers2d(2.0 * np.pi, 2.0 * np.pi, 10, 10)
    L = build_linear_matrix_swe(KX, KY, g=1.0, H=1.0, f=4.0, alpha=0.05)
    _, V, Vinv = _diagonalize_linear_matrix(L)
    rng = np.random.default_rng(0)
    q_hat = rng.normal(size=(3, 10, 10)) + 1j * rng.normal(size=(3, 10, 10))
    w = _to_modal(q_hat, Vinv)
    q_hat_recovered = _to_phys(w, V)
    assert np.abs(q_hat_recovered - q_hat).max() < 1e-8


def test_integrate_swe_shapes_and_finite():
    cfg = SWEConfig(NX=16, NY=16, dt=0.005, snapshot_every=4, spinup_time=0.0)
    rng = np.random.default_rng(0)
    q0 = default_initial_condition_swe(cfg, rng)
    assert q0.shape == (3, cfg.NX, cfg.NY)
    assert np.isfinite(q0).all()

    traj = integrate_swe(q0, cfg, n_steps=20)
    assert traj.shape == (6, 3, cfg.NX, cfg.NY)  # 20/4 = 5 snapshots + the IC
    assert np.isfinite(traj).all()


def test_mass_conservation():
    """Total mass `integral(eta) dx dy` is conserved EXACTLY by the
    flux-form continuity equation for any `u`,`v` (module docstring) --
    checked with a deliberately nonzero-mean `eta` (not the ~0-mean
    default IC, which would make this check trivially pass even if
    conservation were broken) over many steps."""
    cfg = SWEConfig(NX=24, NY=24, dt=0.004, snapshot_every=5, spinup_time=0.0)
    rng = np.random.default_rng(1)
    q0 = default_initial_condition_swe(cfg, rng)
    q0 = q0.copy()
    q0[2] += 0.37  # deliberate nonzero mean
    traj = integrate_swe(q0, cfg, n_steps=200)
    mass0 = traj[0, 2].mean()
    massT = traj[-1, 2].mean()
    assert abs(massT - mass0) < 1e-10, f"mass drifted: {mass0} -> {massT}"


def test_default_ic_is_geostrophically_balanced():
    """`default_initial_condition_swe`'s `(u,v)` should satisfy
    `f*v = g*eta_x`, `f*u = -g*eta_y` by construction (module
    docstring) -- checked directly via spectral derivatives, not merely
    trusted from the derivation."""
    cfg = SWEConfig(NX=16, NY=16)
    rng = np.random.default_rng(2)
    q0 = default_initial_condition_swe(cfg, rng)
    u, v, eta = q0
    KX, KY = wavenumbers2d(cfg.Lx, cfg.Ly, cfg.NX, cfg.NY)
    eta_hat = fft2(eta)
    eta_x = ifft2(1j * KX * eta_hat).real
    eta_y = ifft2(1j * KY * eta_hat).real
    assert np.abs(cfg.f * v - cfg.g * eta_x).max() < 1e-8
    assert np.abs(cfg.f * u + cfg.g * eta_y).max() < 1e-8


def test_spinup_runs_and_is_finite():
    cfg = SWEConfig(NX=16, NY=16, dt=0.005, snapshot_every=4, spinup_time=0.2)
    rng = np.random.default_rng(3)
    q_spun = spinup_swe(cfg, rng)
    assert q_spun.shape == (3, cfg.NX, cfg.NY)
    assert np.isfinite(q_spun).all()


def test_wind_stress_forcing_only_in_u_component():
    """`F_u(y)=tau0*sin(forcing_n*y)`, `F_v=F_eta=0` -- checked directly
    in physical space (not just that the Fourier stack has the right
    shape)."""
    cfg = SWEConfig(NX=16, NY=16, tau0=0.3, forcing_n=3)
    forcing_hat = _wind_stress_forcing_fourier(cfg)
    assert forcing_hat.shape == (3, cfg.NX, cfg.NY)
    assert np.abs(forcing_hat[1]).max() == 0.0
    assert np.abs(forcing_hat[2]).max() == 0.0
    Fu_phys = ifft2(forcing_hat[0]).real
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    _, Y = np.meshgrid(np.arange(cfg.NX) * cfg.Lx / cfg.NX, y, indexing="ij")
    Fu_expected = cfg.tau0 * np.sin(cfg.forcing_n * Y)
    assert np.abs(Fu_phys - Fu_expected).max() < 1e-10


def test_tau0_zero_recovers_unforced_behavior_exactly():
    """`tau0=0` must give an exactly all-zero forcing stack (a true
    no-op), not merely a small one -- the original, pre-forcing
    behavior must still be exactly reachable."""
    cfg = SWEConfig(NX=12, NY=12, tau0=0.0)
    forcing_hat = _wind_stress_forcing_fourier(cfg)
    assert np.abs(forcing_hat).max() == 0.0


def test_forcing_is_the_only_nonzero_term_at_rest():
    """At the trivial rest state (`u=v=eta=0`), every advection/
    divergence term in the nonlinear function vanishes identically, so
    `_nonlinear_swe` there must equal EXACTLY `op.forcing_hat` (dealiased)
    -- the cleanest possible test that the forcing term is wired into
    the right place and nothing else leaks in."""
    cfg = SWEConfig(NX=16, NY=16, tau0=0.2, forcing_n=4)
    op = get_operator_swe(cfg)
    q_hat_zero = np.zeros((3, cfg.NX, cfg.NY), dtype=np.complex128)
    N_hat = _nonlinear_swe(q_hat_zero, op)
    expected = op.forcing_hat * op.dealias[None, :, :]
    assert np.abs(N_hat - expected).max() < 1e-12


def test_mass_conservation_with_forcing_on():
    """Wind-stress forcing acts only on the `u`-momentum equation, never
    on `eta`'s continuity equation, so mass conservation must still hold
    EXACTLY with `tau0 != 0` -- re-checked explicitly rather than merely
    assumed, since this is a genuinely new term added to the equations
    (ground rule 1)."""
    cfg = SWEConfig(NX=20, NY=20, dt=0.004, snapshot_every=5, spinup_time=0.0, tau0=0.2, forcing_n=4)
    rng = np.random.default_rng(9)
    q0 = default_initial_condition_swe(cfg, rng)
    q0 = q0.copy()
    q0[2] += 0.21  # deliberate nonzero mean, same discipline as test_mass_conservation
    traj = integrate_swe(q0, cfg, n_steps=200)
    mass0 = traj[0, 2].mean()
    massT = traj[-1, 2].mean()
    assert abs(massT - mass0) < 1e-10, f"mass drifted with forcing on: {mass0} -> {massT}"


def test_tangent_step_matches_finite_difference_with_forcing_on():
    """The real test that forcing is wired correctly into (or correctly
    absent from) the tangent-linear code: `tau0` now defaults to nonzero
    (`SWEConfig`'s own default, 0.1), so
    `test_tangent_step_matches_finite_difference` above already exercises
    the forced case -- this makes that coverage explicit rather than
    incidental, with an even larger `tau0` for a stronger check. A naive
    (wrong) attempt to isolate "forcing shouldn't enter the tangent
    equation" by comparing tangent OUTPUTS between a forced and
    unforced operator at the same literal `q0` does NOT work as a test
    (tried and rejected during development): even one step's
    intermediate RK4 stage values already differ between the two cases,
    because forcing changes the BASE nonlinear term those stages are
    built from -- so the tangent output legitimately differs too, for a
    reason that has nothing to do with whether forcing is (correctly)
    absent from the tangent Jacobian itself. The finite-difference
    comparison bypasses this entirely: it is the literal derivative of
    whatever the forced one-step map actually does, so it validates the
    tangent code's correctness however forcing enters the base
    dynamics."""
    cfg = SWEConfig(NX=16, NY=16, dt=0.004, snapshot_every=5, spinup_time=0.0, tau0=0.5, forcing_n=4)
    rng = np.random.default_rng(10)
    q0 = default_initial_condition_swe(cfg, rng)
    op = get_operator_swe(cfg)
    q_hat = fft2(q0, axes=(-2, -1))

    rng2 = np.random.default_rng(11)
    dQ_phys = rng2.normal(size=(2, 3, cfg.NX, cfg.NY))
    dQ_hat = fft2(dQ_phys, axes=(-2, -1))
    q_hat_next, dQ_hat_next = step_with_tangent_swe(q_hat, dQ_hat, op)

    eps = 1e-6
    q_pert_hat = fft2(q0 + eps * dQ_phys[0], axes=(-2, -1))
    q_pert_next = step_swe(q_pert_hat, op)
    fd_tangent = (q_pert_next - q_hat_next) / eps
    rel_err = np.abs(fd_tangent - dQ_hat_next[0]).max() / (np.abs(dQ_hat_next[0]).max() + 1e-30)
    assert rel_err < 1e-5, f"tangent step with forcing disagrees with finite difference, rel err {rel_err:.3e}"


def test_spinup_zero_time_returns_raw_ic():
    cfg = SWEConfig(NX=12, NY=12, spinup_time=0.0)
    rng = np.random.default_rng(4)
    rng_clone = np.random.default_rng(4)
    q0_direct = default_initial_condition_swe(cfg, rng_clone)
    q_spun = spinup_swe(cfg, rng)
    assert np.allclose(q_spun, q0_direct)


def test_tangent_step_matches_finite_difference():
    """`step_with_tangent_swe`'s linearization must match a direct
    finite-difference derivative of `step_swe` -- the single strongest
    correctness check available for the tangent-linear code (ground rule
    1), exactly analogous to how `ks_latent.analysis.lyapunov`'s own
    tests validate `jvp`-based tangent propagation elsewhere in this
    project."""
    cfg = SWEConfig(NX=16, NY=16, dt=0.004, snapshot_every=5, spinup_time=0.0)
    rng = np.random.default_rng(5)
    q0 = default_initial_condition_swe(cfg, rng)
    op = get_operator_swe(cfg)
    q_hat = fft2(q0, axes=(-2, -1))

    rng2 = np.random.default_rng(6)
    dQ_phys = rng2.normal(size=(2, 3, cfg.NX, cfg.NY))
    dQ_hat = fft2(dQ_phys, axes=(-2, -1))

    q_hat_next, dQ_hat_next = step_with_tangent_swe(q_hat, dQ_hat, op)

    eps = 1e-6
    q_pert_hat = fft2(q0 + eps * dQ_phys[0], axes=(-2, -1))
    q_pert_next = step_swe(q_pert_hat, op)
    fd_tangent = (q_pert_next - q_hat_next) / eps

    rel_err = np.abs(fd_tangent - dQ_hat_next[0]).max() / (np.abs(dQ_hat_next[0]).max() + 1e-30)
    assert rel_err < 1e-5, f"tangent step disagrees with finite difference, rel err {rel_err:.3e}"


def test_tangent_step_is_linear_in_perturbation():
    """Scaling the tangent input by a constant should scale the tangent
    output by the same constant -- a basic sanity property of a genuine
    linearization (not guaranteed to hold for, e.g., a buggy
    implementation that accidentally mixes in a nonlinear term)."""
    cfg = SWEConfig(NX=12, NY=12, dt=0.005, snapshot_every=4, spinup_time=0.0)
    rng = np.random.default_rng(7)
    q0 = default_initial_condition_swe(cfg, rng)
    op = get_operator_swe(cfg)
    q_hat = fft2(q0, axes=(-2, -1))

    rng2 = np.random.default_rng(8)
    dQ_phys = rng2.normal(size=(1, 3, cfg.NX, cfg.NY))
    dQ_hat = fft2(dQ_phys, axes=(-2, -1))

    _, dQ_hat_next_1x = step_with_tangent_swe(q_hat, dQ_hat, op)
    _, dQ_hat_next_3x = step_with_tangent_swe(q_hat, 3.0 * dQ_hat, op)

    assert np.abs(dQ_hat_next_3x - 3.0 * dQ_hat_next_1x).max() < 1e-8 * np.abs(dQ_hat_next_1x).max()
