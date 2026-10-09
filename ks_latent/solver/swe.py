"""ETDRK4 ("integrating factor") spectral solver for the rotating
shallow water equations (SWE) on a doubly periodic domain (added
2026-10-06, user-directed: "I would like to add one more model to test
our L transfer results"), mirroring `ks_latent.solver.kolmogorov`'s
style and public API as closely as the underlying physics allows.

    u_t + u*u_x + v*u_y - f*v + g*eta_x = -alpha*u,
    v_t + u*v_x + v*v_y + f*u + g*eta_y = -alpha*v,
    eta_t + H*(u_x + v_y) + (eta*u)_x + (eta*v)_y = 0,
    (x,y) in [0,Lx) x [0,Ly),  periodic.

CPU, float64/complex128 only, same device policy as every other solver
in this codebase (brief §1.2: never put the solver on MPS). Uses
`scipy.fft`, not `torch.fft`.

**Three prognostic fields, not one -- the genuine new thing this system
adds.** State is kept, in every public function, as a `(3, NX, NY)`
stack `[u, v, eta]` (field-major), the natural generalization of 2D
KS/Kolmogorov's single `(NX, NY)` field.

**Why the linear operator is a 3x3 MATRIX per wavenumber, not a
scalar -- and how this reuses `ks2d.etdrk4_coefficients_2d` anyway.**
2D KS/Kolmogorov's `Lhat(kx,ky)` is a single number per wavenumber (pure
decay). Here, Coriolis rotation and the pressure-gradient/divergence
coupling between `(u,v,eta)` make the per-wavenumber linear operator

    L(kx,ky) = [[-alpha,   f,      -i*g*kx],
                [-f,       -alpha,  -i*g*ky],
                [-i*H*kx, -i*H*ky,  0      ]],

with complex, generically oscillatory eigenvalues (the inertia-gravity
wave dispersion relation `omega^2 = f^2 + g*H*k^2` at `alpha=0` --
`tests/unit/test_solver_swe.py` checks this directly, ground rule 1).
Rather than writing a new matrix-valued contour-integral ETDRK4 scheme
from scratch, this module diagonalizes `L(kx,ky)` once per wavenumber
(`_diagonalize_linear_matrix`, cached alongside everything else in
`get_operator_swe`) and hands the per-wavenumber EIGENVALUES (shape
`(NX,NY,3)` instead of `(NX,NY)`) directly to the existing
`ks_latent.solver.ks2d.etdrk4_coefficients_2d` -- which already
broadcasts over an arbitrary leading shape, so this exercises its
existing generality rather than extending it; no new contour-integral
code is written here. This is the Cox & Matthews (2002, *J. Comput.
Phys.* 176:430) vector-`L` generalization of the scalar Kassam-Trefethen
scheme the rest of this codebase uses -- the "integrating factor" method
by name: the fast linear gravity-wave dynamics are advanced EXACTLY (via
the per-eigenvalue exponential), and only the nonlinear advection terms
go through the (RK4-like) stage recursion, which is what lets `dt` be
set by the slow advective time scale rather than the fast gravity-wave
one.

**Eigenbasis bookkeeping.** `step_swe` carries the state in MODAL
(eigenbasis) coordinates, shape `(NX,NY,3)`, through the ETDRK4 stage
recursion (so the linear part is a plain elementwise multiply by the
cached coefficients, exactly like the scalar solvers) and converts to
the physical-field Fourier stack `(3,NX,NY)` only to evaluate the
nonlinear advection term, which is most naturally computed from
physical-space products. `_to_modal`/`_to_phys` are the two
(batched-`eig`-based) 3x3 matrix-vector transforms that cross that
boundary, applied via `np.einsum` over the full `(NX,NY)` grid at once.

**Dealiasing.** Reused directly from `ks_latent.solver.kolmogorov`
(`_dealias_mask`) -- the same 2/3-rule necessity applies here for the
same reason (quadratic advection nonlinearity, no automatic
hyperdissipative suppression of aliased high-wavenumber content).

**Mass conservation, not a special gauge-fix.** Unlike 2D KS/Kolmogorov
vorticity's own mean-must-be-zero constraint (a genuine physical gauge
freedom enforced by explicit projection every step), SWE's total mass
`integral(eta) dx dy` is conserved automatically by the flux-form
continuity equation -- `eta_t + H*(u_x+v_y) + (eta*u)_x + (eta*v)_y = 0`
integrates to exactly zero over the periodic domain by the divergence
theorem, for ANY `u`,`v`. No projection step is needed or applied here;
`test_mass_conservation` in `tests/unit/test_solver_swe.py` checks this
holds to float64 roundoff over many steps (including with the
wind-stress forcing below switched on, since that forcing never touches
the `eta` equation), exactly the kind of "assert, don't just hope" check
`step_kolmogorov`'s own mean-vorticity projection is validated by.

**Wind-stress forcing (`SWEConfig.tau0`/`forcing_n`, added 2026-10-06,
user-directed: "include the external wind stress forcing term... to
sustain the chaotic state against the linear friction").** A
deterministic, state- and time-INDEPENDENT body force on the
`u`-momentum equation only, `F_u(y) = tau0*sin(forcing_n*y)`, `F_v =
F_eta = 0` -- see `SWEConfig`'s own docstring for the full physical
motivation and the periodicity requirement on `forcing_n`. Precomputed
once as `op.forcing_hat` (`get_operator_swe`, same caching discipline as
every other per-config quantity in this module) and added into
`_nonlinear_from_quantities`'s output, exactly where
`ks_latent.solver.kolmogorov._nonlinear_kolmogorov` adds its own
`op.f_hat` -- both are additive, state-independent forcing terms that
belong in the "everything but the linear operator" slot of the ETDRK4
recursion. Being state-independent, it contributes NOTHING to the
tangent/variational equation `step_with_tangent_swe` integrates (same
reasoning as `_nonlinear_kolmogorov`'s own forcing term): the tangent
nonlinear function only ever differentiates the BILINEAR advection
terms, never this additive constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import fft2, ifft2

from ks_latent.config import SWEConfig
from ks_latent.solver.kolmogorov import _dealias_mask
from ks_latent.solver.ks2d import ETDRK4Coeffs2D, etdrk4_coefficients_2d, wavenumbers2d


def _wind_stress_forcing_fourier(cfg: SWEConfig) -> np.ndarray:
    """`(3,NX,NY)` complex128 Fourier-space forcing stack
    `[FFT(F_u), 0, 0]`, `F_u(y) = tau0*sin(forcing_n*y)` -- see module
    docstring. `tau0=0` (the non-default, original-unforced behavior)
    gives an exact all-zero stack, a no-op in the nonlinear-term sum."""
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    X, Y = np.meshgrid(x, y, indexing="ij")
    Fu_phys = cfg.tau0 * np.sin(cfg.forcing_n * Y)
    Fu_hat = fft2(Fu_phys)
    zeros = np.zeros_like(Fu_hat)
    return np.stack([Fu_hat, zeros, zeros], axis=0)


def build_linear_matrix_swe(KX: np.ndarray, KY: np.ndarray, g: float, H: float, f: float, alpha: float) -> np.ndarray:
    """`(NX,NY,3,3)` complex128 per-wavenumber linear operator matrix
    `L(kx,ky)` -- see module docstring for the exact entries. Field
    order `(u,v,eta)`, matching every other public function's `(3,NX,NY)`
    state-stack convention."""
    NX, NY = KX.shape
    L = np.zeros((NX, NY, 3, 3), dtype=np.complex128)
    L[:, :, 0, 0] = -alpha
    L[:, :, 0, 1] = f
    L[:, :, 0, 2] = -1j * g * KX
    L[:, :, 1, 0] = -f
    L[:, :, 1, 1] = -alpha
    L[:, :, 1, 2] = -1j * g * KY
    L[:, :, 2, 0] = -1j * H * KX
    L[:, :, 2, 1] = -1j * H * KY
    L[:, :, 2, 2] = 0.0
    return L


def _diagonalize_linear_matrix(L: np.ndarray, rtol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched eigendecomposition of `L`, `(NX,NY,3,3) -> (eigvals
    (NX,NY,3), V (NX,NY,3,3), Vinv (NX,NY,3,3))`, one independent 3x3
    problem per grid point (`numpy.linalg.eig`/`inv` batch over all
    leading axes automatically). `L[x,y] = V[x,y] @ diag(eigvals[x,y]) @
    Vinv[x,y]` by construction of `eig`; **verified directly, not just
    assumed** (ground rule 1/2: fail loudly on a numerical claim that is
    not checked) -- a genuinely defective (non-diagonalizable) `L` at
    some wavenumber would silently corrupt every later step without this
    check, and this project's established policy (`robust_cholesky`/
    `robust_eigh` in `ks_latent.utils.linalg`) is to raise with
    diagnostic detail rather than let bad linear algebra propagate
    silently."""
    eigvals, V = np.linalg.eig(L)  # (NX,NY,3), (NX,NY,3,3) -- eigenvectors are columns of V
    Vinv = np.linalg.inv(V)
    recon = np.einsum("xyij,xyj,xyjk->xyik", V, eigvals, Vinv)
    denom = np.abs(L).max()
    rel_err = float(np.abs(recon - L).max() / (denom + 1e-30))
    if rel_err > rtol:
        raise ValueError(
            f"SWE linear-operator eigendecomposition reconstruction error {rel_err:.3e} exceeds "
            f"rtol={rtol:.1e} -- L is likely (near-)defective at some wavenumber; this must be "
            "investigated before trusting any step taken with this operator."
        )
    return eigvals, V, Vinv


def _to_modal(q_hat: np.ndarray, Vinv: np.ndarray) -> np.ndarray:
    """`(3,NX,NY)` physical-field Fourier stack -> `(NX,NY,3)` modal
    (eigenbasis) coordinates: `w = Vinv @ q_hat`, per wavenumber."""
    return np.einsum("xyji,ixy->xyj", Vinv, q_hat)


def _to_phys(w: np.ndarray, V: np.ndarray) -> np.ndarray:
    """`(NX,NY,3)` modal coordinates -> `(3,NX,NY)` physical-field
    Fourier stack: `q_hat = V @ w`, per wavenumber. Inverse of `_to_modal`."""
    return np.einsum("xyij,xyj->ixy", V, w)


def _to_modal_batched(dq_hat: np.ndarray, Vinv: np.ndarray) -> np.ndarray:
    """Batched analogue of `_to_modal` for a stack of `k` tangent
    vectors, `(k,3,NX,NY) -> (k,NX,NY,3)` (`ks_latent.analysis.lyapunov`'s
    Benettin machinery)."""
    return np.einsum("xyji,kixy->kxyj", Vinv, dq_hat)


def _to_phys_batched(dw: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Batched analogue of `_to_phys`, `(k,NX,NY,3) -> (k,3,NX,NY)`."""
    return np.einsum("xyij,kxyj->kixy", V, dw)


@dataclass(frozen=True)
class ETDRK4OperatorSWE:
    KX: np.ndarray
    KY: np.ndarray
    L: np.ndarray  # (NX,NY,3,3), the un-diagonalized linear matrix (kept for diagnostics/tests)
    eigvals: np.ndarray  # (NX,NY,3), complex
    V: np.ndarray  # (NX,NY,3,3)
    Vinv: np.ndarray  # (NX,NY,3,3)
    coeffs: ETDRK4Coeffs2D  # each field (E,E2,Q,f1,f2,f3) shaped (NX,NY,3), MODAL-space ETDRK4 coefficients
    dealias: np.ndarray  # (NX,NY), 2/3-rule mask
    forcing_hat: np.ndarray  # (3,NX,NY), wind-stress forcing (zero stack if tau0=0)


@lru_cache(maxsize=32)
def _get_operator_swe_cached(
    Lx: float, Ly: float, NX: int, NY: int, g: float, H: float, f: float, alpha: float,
    tau0: float, forcing_n: int, dt: float, M: int,
) -> ETDRK4OperatorSWE:
    cfg = SWEConfig(Lx=Lx, Ly=Ly, NX=NX, NY=NY, g=g, H=H, f=f, alpha=alpha, tau0=tau0, forcing_n=forcing_n, dt=dt)
    KX, KY = wavenumbers2d(Lx, Ly, NX, NY)
    L = build_linear_matrix_swe(KX, KY, g, H, f, alpha)
    eigvals, V, Vinv = _diagonalize_linear_matrix(L)
    coeffs = etdrk4_coefficients_2d(eigvals, dt, M=M)  # reuse: Lhat shape (NX,NY,3), not (NX,NY)
    dealias = _dealias_mask(NX, NY)
    forcing_hat = _wind_stress_forcing_fourier(cfg)
    return ETDRK4OperatorSWE(
        KX=KX, KY=KY, L=L, eigvals=eigvals, V=V, Vinv=Vinv, coeffs=coeffs, dealias=dealias, forcing_hat=forcing_hat
    )


def get_operator_swe(cfg: SWEConfig, M: int = 32) -> ETDRK4OperatorSWE:
    """ETDRK4 operator for `cfg`, cached by
    `(Lx,Ly,NX,NY,g,H,f,alpha,tau0,forcing_n,dt,M)`."""
    return _get_operator_swe_cached(
        cfg.Lx, cfg.Ly, cfg.NX, cfg.NY, cfg.g, cfg.H, cfg.f, cfg.alpha, cfg.tau0, cfg.forcing_n, cfg.dt, M
    )


def _base_physical_quantities(q_hat: np.ndarray, op: ETDRK4OperatorSWE):
    """Dealiased `u,v,eta` and the first derivatives the nonlinear term
    needs, all physical-space real arrays `(NX,NY)`. Returned as a tuple
    so `step_with_tangent_swe` can reuse them when linearizing at the
    same base state, without recomputing (same pattern as
    `step_with_tangent_kolmogorov`'s own `base_nl`)."""
    mask = op.dealias
    q_hat = q_hat * mask[None, :, :]
    u_hat, v_hat, eta_hat = q_hat[0], q_hat[1], q_hat[2]
    u = ifft2(u_hat).real
    v = ifft2(v_hat).real
    eta = ifft2(eta_hat).real
    ux = ifft2(1j * op.KX * u_hat).real
    uy = ifft2(1j * op.KY * u_hat).real
    vx = ifft2(1j * op.KX * v_hat).real
    vy = ifft2(1j * op.KY * v_hat).real
    return u, v, eta, ux, uy, vx, vy


def _nonlinear_from_quantities(u, v, eta, ux, uy, vx, vy, op: ETDRK4OperatorSWE) -> np.ndarray:
    """`N_hat = FFT([-(u*ux+v*uy), -(u*vx+v*vy), -((eta*u)_x+(eta*v)_y)])
    + op.forcing_hat`, `(3,NX,NY)` complex, dealiased. The forcing term
    (module docstring) is added here, exactly where
    `ks_latent.solver.kolmogorov._nonlinear_kolmogorov` adds its own
    `op.f_hat` -- the "everything but the linear operator" slot of the
    ETDRK4 recursion, including it in the base-state computation only
    (never in `step_with_tangent_swe`'s tangent term, since it is
    state-independent)."""
    mask = op.dealias
    Nu = -fft2(u * ux + v * uy)
    Nv = -fft2(u * vx + v * vy)
    Neta = -(1j * op.KX * fft2(eta * u) + 1j * op.KY * fft2(eta * v))
    return (np.stack([Nu, Nv, Neta], axis=0) + op.forcing_hat) * mask[None, :, :]


def _nonlinear_swe(q_hat: np.ndarray, op: ETDRK4OperatorSWE) -> np.ndarray:
    return _nonlinear_from_quantities(*_base_physical_quantities(q_hat, op), op)


def _nonlinear_modal(w: np.ndarray, op: ETDRK4OperatorSWE) -> np.ndarray:
    """One ETDRK4 stage's nonlinear evaluation, entirely in modal
    coordinates in and out: `w (NX,NY,3) -> q_hat (3,NX,NY)` (physical
    fields, via `_to_phys`) `-> N_hat (3,NX,NY) -> N_modal (NX,NY,3)`
    (via `_to_modal`). The forward/inverse eigenbasis transforms are
    cheap `(NX,NY,3,3)@(NX,NY,3)` batched matvecs -- negligible next to
    the FFTs either side of them."""
    q_hat = _to_phys(w, op.V)
    N_hat = _nonlinear_swe(q_hat, op)
    return _to_modal(N_hat, op.Vinv)


def step_swe(q_hat: np.ndarray, op: ETDRK4OperatorSWE) -> np.ndarray:
    """One ETDRK4 ("integrating factor") step. `q_hat`: `(3,NX,NY)`
    complex128 physical-field Fourier stack `[u_hat,v_hat,eta_hat]`.
    Same algebraic recursion as `ks2d.step2d`/`kolmogorov.step_kolmogorov`,
    carried out in MODAL coordinates (`op.coeffs` are themselves modal,
    from `etdrk4_coefficients_2d(op.eigvals, ...)`) -- see module
    docstring for why this reuses, rather than reimplements, the scalar
    ETDRK4 machinery."""
    c = op.coeffs
    w = _to_modal(q_hat, op.Vinv)
    Nw = _nonlinear_modal(w, op)
    a = c.E2 * w + c.Q * Nw
    Na = _nonlinear_modal(a, op)
    b = c.E2 * w + c.Q * Na
    Nb = _nonlinear_modal(b, op)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nw)
    Nc = _nonlinear_modal(cc, op)
    w_next = c.E * w + Nw * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    q_hat_next = _to_phys(w_next, op.V)
    return q_hat_next * op.dealias[None, :, :]


def integrate_swe(
    q0: np.ndarray,
    cfg: SWEConfig,
    n_steps: int,
    *,
    snapshot_every: int | None = None,
    M: int = 32,
) -> np.ndarray:
    """Integrate `q0` (`(3,NX,NY)`, field-major `[u,v,eta]`) for
    `n_steps` solver steps of size `cfg.dt`. Returns `(n_snapshots+1,
    3, NX, NY)` including the initial condition at index 0."""
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    op = get_operator_swe(cfg, M=M)
    q_hat = fft2(q0.astype(np.float64), axes=(-2, -1)) * op.dealias[None, :, :]
    snapshots = [ifft2(q_hat, axes=(-2, -1)).real.copy()]
    for step_idx in range(1, n_steps + 1):
        q_hat = step_swe(q_hat, op)
        if step_idx % snapshot_every == 0:
            snapshots.append(ifft2(q_hat, axes=(-2, -1)).real.copy())
    return np.stack(snapshots, axis=0)


def default_initial_condition_swe(cfg: SWEConfig, rng: np.random.Generator) -> np.ndarray:
    """A random, low-wavenumber `eta` perturbation, with `u`,`v`
    initialized in exact GEOSTROPHIC BALANCE with it (`f*v=g*eta_x`,
    `f*u=-g*eta_y`) rather than left at zero. Standard rotating-shallow-
    water initialization practice (Vallis, *Atmospheric and Oceanic
    Fluid Dynamics*, ch. 3/5): starting from an unbalanced state (e.g.
    `u=v=0`) immediately radiates most of the initial energy away as
    fast inertia-gravity waves, leaving little behind to drive the slow,
    vortical, genuinely turbulent dynamics this system is meant to
    exhibit; a geostrophically balanced IC instead seeds that slow
    manifold directly. Unlike both KS variants' / Kolmogorov's own
    default ICs (perturbing a known nontrivial base state or the zero
    state), SWE's rest state (`u=v=eta=0`) has no analogous nontrivial
    laminar solution to perturb -- the chaos here is seeded entirely by
    the initial condition's own energy, sustained against `alpha`-damping
    by ongoing nonlinear energy transfer between scales, not by an
    external steady forcing (there is none in this system, unlike
    Kolmogorov's `f(y)`)."""
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta = np.zeros((cfg.NX, cfg.NY))
    n_modes_per_axis = 4
    amp_scale = (0.1 * cfg.H) / n_modes_per_axis
    for j in range(1, n_modes_per_axis + 1):
        for l in range(1, n_modes_per_axis + 1):
            a = rng.normal(scale=amp_scale)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            eta += a * np.cos(2.0 * np.pi * j * X / cfg.Lx + 2.0 * np.pi * l * Y / cfg.Ly + phi)
    KX, KY = wavenumbers2d(cfg.Lx, cfg.Ly, cfg.NX, cfg.NY)
    eta_hat = fft2(eta)
    u_hat = -(cfg.g / cfg.f) * (1j * KY * eta_hat)
    v_hat = (cfg.g / cfg.f) * (1j * KX * eta_hat)
    u = ifft2(u_hat).real
    v = ifft2(v_hat).real
    return np.stack([u, v, eta], axis=0)


def spinup_swe(cfg: SWEConfig, rng: np.random.Generator, *, M: int = 32) -> np.ndarray:
    """Discard `cfg.spinup_time` of transient (gravity-wave radiation
    plus nonlinear saturation onto the turbulent attractor), return the
    resulting `(3,NX,NY)` state."""
    q0 = default_initial_condition_swe(cfg, rng)
    n_spinup_steps = int(round(cfg.spinup_time / cfg.dt))
    if n_spinup_steps == 0:
        return q0
    traj = integrate_swe(q0, cfg, n_spinup_steps, snapshot_every=n_spinup_steps, M=M)
    return traj[-1]


def step_with_tangent_swe(
    q_hat: np.ndarray, dQ_hat: np.ndarray, op: ETDRK4OperatorSWE
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the base state `q_hat` (`(3,NX,NY)`) and a batch of `k`
    tangent vectors `dQ_hat` (`(k,3,NX,NY)`, complex, physical-field
    Fourier stack) together, for `ks_latent.analysis.lyapunov`'s Benettin
    machinery -- same variational-ETDRK4 pattern as
    `ks2d.step_with_tangent2d`/`kolmogorov.step_with_tangent_kolmogorov`,
    generalized to this system's vector linear operator exactly as
    `step_swe` generalizes the plain (non-tangent) step.

    Every nonlinear term here is BILINEAR in the state (a product of two
    quantities each linear in `(u,v,eta)`: `u*ux`, `eta*u`, etc.), so the
    Frechet derivative follows the product rule exactly, no
    approximation -- same structure, and same justification, as
    `step_with_tangent_kolmogorov`'s own tangent nonlinear term. The
    LINEAR part of the equation is applied identically to the base state
    and every tangent vector (`L` does not depend on the state), so it
    needs no separate derivation at all -- only the nonlinear advection
    term's linearization is genuinely new work here."""
    c = op.coeffs
    KX, KY, mask = op.KX, op.KY, op.dealias

    def tangent_nonlinear(base_quantities, dq_hat_phys):
        u, v, eta, ux, uy, vx, vy = base_quantities
        du_hat, dv_hat, deta_hat = dq_hat_phys[:, 0], dq_hat_phys[:, 1], dq_hat_phys[:, 2]
        du_hat = du_hat * mask[None, :, :]
        dv_hat = dv_hat * mask[None, :, :]
        deta_hat = deta_hat * mask[None, :, :]
        du = ifft2(du_hat, axes=(-2, -1)).real
        dv = ifft2(dv_hat, axes=(-2, -1)).real
        deta = ifft2(deta_hat, axes=(-2, -1)).real
        dux = ifft2(1j * KX[None, :, :] * du_hat, axes=(-2, -1)).real
        duy = ifft2(1j * KY[None, :, :] * du_hat, axes=(-2, -1)).real
        dvx = ifft2(1j * KX[None, :, :] * dv_hat, axes=(-2, -1)).real
        dvy = ifft2(1j * KY[None, :, :] * dv_hat, axes=(-2, -1)).real

        dNu = -fft2(du * ux[None, :, :] + u[None, :, :] * dux + dv * uy[None, :, :] + v[None, :, :] * duy, axes=(-2, -1))
        dNv = -fft2(du * vx[None, :, :] + u[None, :, :] * dvx + dv * vy[None, :, :] + v[None, :, :] * dvy, axes=(-2, -1))
        dNeta = -(
            1j * KX[None, :, :] * fft2(deta * u[None, :, :] + eta[None, :, :] * du, axes=(-2, -1))
            + 1j * KY[None, :, :] * fft2(deta * v[None, :, :] + eta[None, :, :] * dv, axes=(-2, -1))
        )
        dN_hat = np.stack([dNu, dNv, dNeta], axis=1) * mask[None, None, :, :]
        return dN_hat

    def stage(w, dw):
        """One stage's full nonlinear evaluation (base + tangent),
        modal in and out, both base and tangent."""
        q_hat_stage = _to_phys(w, op.V)
        base_quantities = _base_physical_quantities(q_hat_stage, op)
        N_hat = _nonlinear_from_quantities(*base_quantities, op)
        N_modal = _to_modal(N_hat, op.Vinv)
        dq_hat_stage = _to_phys_batched(dw, op.V)
        dN_hat = tangent_nonlinear(base_quantities, dq_hat_stage)
        dN_modal = _to_modal_batched(dN_hat, op.Vinv)
        return N_modal, dN_modal

    w = _to_modal(q_hat, op.Vinv)
    dw = _to_modal_batched(dQ_hat, op.Vinv)

    Nw, dNw = stage(w, dw)
    a = c.E2 * w + c.Q * Nw
    da = c.E2[None, :, :, :] * dw + c.Q[None, :, :, :] * dNw
    Na, dNa = stage(a, da)

    b = c.E2 * w + c.Q * Na
    db = c.E2[None, :, :, :] * dw + c.Q[None, :, :, :] * dNa
    Nb, dNb = stage(b, db)

    cc = c.E2 * a + c.Q * (2.0 * Nb - Nw)
    dcc = c.E2[None, :, :, :] * da + c.Q[None, :, :, :] * (2.0 * dNb - dNw)
    Nc, dNc = stage(cc, dcc)

    w_next = c.E * w + Nw * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    dw_next = (
        c.E[None, :, :, :] * dw
        + dNw * c.f1[None, :, :, :]
        + 2.0 * (dNa + dNb) * c.f2[None, :, :, :]
        + dNc * c.f3[None, :, :, :]
    )

    q_hat_next = _to_phys(w_next, op.V) * mask[None, :, :]
    dQ_hat_next = _to_phys_batched(dw_next, op.V) * mask[None, None, :, :]
    return q_hat_next, dQ_hat_next
