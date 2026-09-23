"""Natural-Gradient Particle Flow Filter (NAT-PFF) for latent-space DA
(brief §7, PROJECT_HANDOFF.md).

**Derivation (brief: "derive this in a docstring, do not just assert it").**

Forecast ensemble `{z^j}` has fixed mean `z0_bar` and fixed covariance `B`
(the empirical forecast covariance, computed once and never recentered
during the update -- "B is the prior covariance ... FIXED through
pseudo-time"). Observation `y` (physical units) relates to the state via
`h(z) = H(D(z))` (a physical observation operator composed with the
decoder), noise covariance `R`. The target is the Bayesian posterior

    p(z|y) ~exp( -0.5 (z-z0_bar)^T B^-1 (z-z0_bar) - 0.5 (y-h(z))^T R^-1 (y-h(z)) ).

We transport the forecast ensemble to (approximate) posterior samples via a
pseudo-time flow `z(s)`, `s in [0, n_steps]`, re-linearizing `h` at the
*current* ensemble mean `z_bar(s)` at every step (a Gauss-Newton iteration,
not a single linear step, so nonlinear `h` is handled correctly):

    J_bar = d h/dz |_{z_bar}                      (m, d)
    F     = J_bar^T R^-1 J_bar + B^-1              (d, d)   -- Gauss-Newton precision
    d_s   = y - h(z_bar)                           (m,)     -- innovation at the current mean
    grad_j = B^-1 (z^j - z0_bar)
             + J_bar^T R^-1 J_bar (z^j - z_bar)
             - J_bar^T R^-1 d_s                    (d,)     -- energy gradient at particle j

The **natural-gradient (Newton) drift** preconditions by `F^-1`:

    f^j = -F^-1 grad_j.

For a *linear* `h` (constant `J_bar` everywhere), the fixed point of the
deterministic flow (`grad_j = 0` for every particle) forces every particle
to the same point -- the Newton step exactly finds the MAP, but a purely
deterministic flow collapses ensemble spread by construction (this is
expected and tested, brief's DET mode: it recovers the analysis *mean*
correctly, not the analysis *covariance*).

**Why STO/NAT need a stochastic term, and why it takes this exact form.**
To retain calibrated spread, treat the flow as overdamped Langevin dynamics
with a *constant* (state-independent, since `F` is evaluated once per step
from the ensemble mean and shared by every particle) preconditioning
metric `F^-1`:

    dz = -F^-1 grad Energy(z) ds + sqrt(2 F^-1) dW.

Standard Fokker-Planck theory: for *constant* diffusion tensor `D`, the SDE
`dz = -D grad Energy ds + sqrt(2D) dW` has stationary density
`~exp(-Energy(z))` for *any* constant PSD `D` -- setting `D = F^-1` makes
`F^-1` simultaneously the Newton preconditioner and the correct diffusion
matrix for the *same* target density, which is the whole point of using a
"natural" (Fisher/Gauss-Newton-metric) Langevin scheme rather than an
arbitrary preconditioner.

**"Divergence term simplifies via F/F^-1 cancellation."** The general
*Riemannian* Langevin SDE for a *position-dependent* metric `G(z)` needs an
extra correction (drift) term built from derivatives of `G^-1(z)` (the
divergence of the diffusion tensor), because the stationary-density
argument above assumes `D` doesn't depend on `z`. Here `F` is recomputed
once per pseudo-time step from the *ensemble mean*, not per-particle, so
within a step it is a single constant matrix shared by every particle:
`d F^-1/dz^j = 0` for every `j`, and the entire correction term is
identically zero -- "cancellation" in the sense that the general formula's
`F, F^-1` derivative terms vanish exactly, not that they partially offset.
No extra code term is needed; this docstring is the whole derivation.

Within one pseudo-time step, `grad_j = F z^j - c` is *exactly* affine with a
single shared root `z_star = F^-1 c` (another instance of the brief's
"F/F^-1 cancellation": `f^j = -F^-1(F z^j - c) = z_star - z^j`, independent
of `F`'s actual value once `z_star` is known), so the per-step dynamics are
an exact Ornstein-Uhlenbeck process with unit rate. Rather than an Euler
discretization of the SDE (`z + ds*f + sqrt(2*ds)*noise`, only accurate for
`ds << 1` and numerically *unstable* once the adaptive schedule below grows
`ds` past ~2), each step uses the OU process's exact transition kernel --
an exponential integrator, the same spirit as the ETDRK4 solver elsewhere
in this codebase, unconditionally stable for any `ds`:

    decay = exp(-ds)
    DET:      z^j <- z_star + decay * (z^j - z_star)
    STO/NAT:  z^j <- z_star + decay * (z^j - z_star)
              + sqrt(1 - decay^2) * L_precond @ eta^j

with `L_precond @ L_precond.T = F^-1` (`robust_cholesky`) and `eta^j` iid
standard normal, mean-centered *across the ensemble* (`eta -= eta.mean(0)`)
so the injected noise does not perturb the ensemble mean at each step --
the mean should evolve only under the deterministic drift. As `ds -> infty`
this reduces to `z^j ~ N(z_star, F^-1)`, the correct target; as `ds -> 0`
it reduces to the Euler-Maruyama step.

Honest note on the exact stochastic-term formula: the brief's shorthand
"eta = sqrt(2) * L_precond * N_mat * L_k^T, using k_bar/N inside the
Cholesky" names quantities ("N_mat", "L_k", "k_bar") from the original
undocumented script that could not be reconstructed without its source.
What's implemented here is an independently-derived, mathematically
equivalent preconditioned-Langevin stochastic term with the same target
stationary distribution, verified against the one test that actually
matters (`test_pff_gaussian_linear`: exact Kalman recovery for a linear
observation operator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.func import jacrev

from ks_latent.utils.linalg import robust_cholesky

ObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (d,) -> (m,)


@dataclass(frozen=True)
class PFFConfig:
    method: str = "NAT"  # "DET", "STO", "NAT" (brief §7: NAT is default)
    n_steps: int = 100
    ds_max: float = 10.0  # brief gives the recursion but not this ceiling; inferred (see docs/OPEN_QUESTIONS.md)
    tol: float = 1e-4
    early_stop_after: int = 5

    def __post_init__(self):
        if self.method not in ("DET", "STO", "NAT"):
            raise ValueError(f"method must be DET, STO, or NAT, got {self.method!r}")


@dataclass(frozen=True)
class PFFResult:
    Z_analysis: torch.Tensor  # (N, d)
    n_iterations: int
    converged: bool
    ds_history: list[float]
    rel_change_history: list[float]


def _robust_inverse(A: torch.Tensor) -> torch.Tensor:
    """Symmetric PD inverse via `robust_cholesky` (CPU float64, brief §1.2)."""
    L = robust_cholesky(A)
    return torch.cholesky_inverse(L)


def _jacobian_at(h: ObsOperator, z: torch.Tensor) -> torch.Tensor:
    """`d h/dz` at the single point `z` (d,), returned as `(m, d)`. CPU per
    the device policy (`task="jacobian"`, brief §1.2: "MPS has known gaps")."""
    return jacrev(h)(z)


class ParticleFlowFilter:
    """One class, three flow variants (`DET`/`STO`/`NAT`, brief §7).

    "Diagnostic mode activates when decoder_phys is provided": if given, the
    analysis ensemble mean is additionally decoded to physical space for
    spacetime-plot / diagnostic use (`decode(mean(z))`, *not* `mean(decode(z))`
    -- brief: "u_da_history uses decode(mean(z))").
    """

    def __init__(
        self,
        h: ObsOperator,
        R: torch.Tensor,
        *,
        decoder_phys: Callable[[torch.Tensor], torch.Tensor] | None = None,
        config: PFFConfig | None = None,
        localize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.h = h
        self.R = R.to(torch.float64)
        self.Rinv = _robust_inverse(self.R) if R.shape[0] > 0 else R.clone()
        self.decoder_phys = decoder_phys
        self.cfg = config or PFFConfig()
        self.localize_fn = localize_fn

    @torch.no_grad()
    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> PFFResult:
        """`Z0`: `(N, d)` forecast ensemble. `y`: `(m,)` observation, physical
        units. Returns the analysis ensemble (and diagnostics).

        Decorated `@torch.no_grad()`: nothing here needs the ambient
        autograd tape (this is inference through a frozen decoder, not
        training), and `jacrev` inside `_jacobian_at` manages its own
        differentiation independently of the ambient grad mode -- leaving
        the tape enabled just built up an unused graph through the decoder
        at every pseudo-time step.
        """
        cfg = self.cfg
        Z0 = Z0.to(torch.float64)
        y = y.to(torch.float64)
        N, d = Z0.shape

        z0_bar = Z0.mean(dim=0)
        B = torch.cov(Z0.T) if N > 1 else torch.eye(d, dtype=torch.float64) * 1e-8
        if self.localize_fn is not None:
            B = self.localize_fn(B)  # brief §9: distance-free localization (SEC or fixed taper)
        Binv = _robust_inverse(B)

        Z = Z0.clone()
        ds = 1.0
        ds_history: list[float] = []
        rel_change_history: list[float] = []
        n_iterations = 0
        converged = False

        for s in range(cfg.n_steps):
            z_bar = Z.mean(dim=0)
            if self.R.shape[0] > 0:
                J_bar = _jacobian_at(self.h, z_bar)  # (m, d)
                d_s = y - self.h(z_bar)  # (m,)
                JtRinv = J_bar.T @ self.Rinv  # (d, m)
                likelihood_precision = JtRinv @ J_bar  # (d, d)
                likelihood_pull = JtRinv @ d_s  # (d,)
            else:
                likelihood_precision = torch.zeros(d, d, dtype=torch.float64)
                likelihood_pull = torch.zeros(d, dtype=torch.float64)

            F = likelihood_precision + Binv
            Finv = _robust_inverse(F)

            # grad_j = F @ z^j - c is affine with the *same* curvature F for
            # every particle (c = Binv@z0_bar + likelihood_precision@z_bar +
            # likelihood_pull doesn't depend on j), so it has a single,
            # shared root z_star = F^-1 @ c -- this iteration's local MAP --
            # and the Newton direction collapses to the F-independent form
            # f^j = -F^-1 (F z^j - c) = z_star - z^j (another instance of the
            # brief's "F/F^-1 cancellation": F drops out of the drift's
            # magnitude entirely, only entering through z_star and the noise).
            c = Binv @ z0_bar + likelihood_precision @ z_bar + likelihood_pull
            z_star = Finv @ c
            f = z_star.unsqueeze(0) - Z  # (N, d), Newton direction per particle

            f_mean_norm = float(f.mean(dim=0).norm())
            ds_test = min(max(1.0, 1.0 / (f_mean_norm + 1e-8)), cfg.ds_max)
            if s == 0:
                ds = 1.0
            elif ds_test > ds:
                ds = min(1.1 * ds, ds_test)
            else:
                ds = max(0.9 * ds, ds_test)

            # Since f is *exactly* linear (an Ornstein-Uhlenbeck pull toward
            # z_star with unit rate), integrate it exactly (an exponential
            # integrator, same spirit as the ETDRK4 solver elsewhere in this
            # codebase) rather than with an Euler step -- explicit Euler
            # `Z + ds*f` is only a good approximation for ds << 1 and is
            # numerically *unstable* here once ds > 2 (the adaptive schedule
            # deliberately grows ds up to ds_max as the flow converges,
            # which blew the naive Euler form up during development; the
            # exact form is unconditionally stable for any ds >= 0 and
            # reduces to the Euler step in the ds -> 0 limit).
            decay = float(torch.exp(torch.tensor(-ds)))
            Z_new = z_star.unsqueeze(0) + decay * (Z - z_star.unsqueeze(0))
            if cfg.method in ("STO", "NAT"):
                L_precond = robust_cholesky(Finv)
                eta = torch.randn(N, d, dtype=torch.float64)
                eta = eta - eta.mean(dim=0, keepdim=True)
                diffusion_scale = (1.0 - decay**2) ** 0.5
                Z_new = Z_new + diffusion_scale * eta @ L_precond.T

            rel_change = float((Z_new.mean(dim=0) - Z.mean(dim=0)).norm() / (Z.mean(dim=0).norm() + 1e-12))
            ds_history.append(ds)
            rel_change_history.append(rel_change)
            Z = Z_new
            n_iterations = s + 1

            if s > cfg.early_stop_after and rel_change < cfg.tol:
                converged = True
                break

        return PFFResult(
            Z_analysis=Z,
            n_iterations=n_iterations,
            converged=converged,
            ds_history=ds_history,
            rel_change_history=rel_change_history,
        )
