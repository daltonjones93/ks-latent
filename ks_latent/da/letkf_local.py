"""Per-site local LETKF for zero-decoder-reach 2D local-field models
(`docs/ks_local_field_ltransfer.tex` sec:c19_da_hypotheses, test T2;
Hunt, Kostelich & Szunyogh 2007's own per-gridpoint local-domain LETKF --
`ks_latent.da.letkf.LETKF`'s module docstring explains why THAT class is
NOT this algorithm, instead implementing the earlier, dense
covariance-localized EnKF of Houtekamer & Mitchell 1998/2001. This
module is the thing the naming caveat there says is missing.

**Why this needs to exist, not just the dense filter.** The dense
filter forms an explicit `(d, d)` forecast covariance and a `(d, d)`
taper -- `float64`, `d = n_sites_x * n_sites_y * channels` -- which is
0.74 GB at 2D KS's native 44^2 (`d=9600`), 3.7 GB at a 66^2 transfer, and
11.8 GB at 88^2, per matrix, before temporaries
(`docs/ks_local_field_ltransfer.tex` sec:c19_da_hypotheses, H2). It also
calls `jacrev` for the observation Jacobian, costing one reverse-mode
pass per observation -- `m`, which grows with domain area. Neither
practice survives the Phase 13/consolidation L-transfer gate at
anything beyond native size. This module never forms a `(d, d)` or
`(m, d)` dense object: each site's analysis touches only a small,
LOCAL slice of the ensemble (its own `channels` latent coordinates, and
the few neighbouring sites' coordinates whose observations fall inside
the Gaspari-Cohn support radius), and the observation Jacobian is
supplied in `ks_latent.da.jacobian_coloring`'s COMPACT `(m, channels)`
form -- `channels` forward passes total, shared across every site,
never scattered into a dense `(m, d)` matrix.

**Precondition (checked by the caller, same as `jacobian_coloring`):**
valid only for a decoder with reach 0
(`ks_latent.da.jacobian_coloring.assert_decoder_zero_reach`). With
nonzero decoder reach, an observation's influence genuinely spans
several sites, and restricting its analysis to a single site's
neighbourhood would silently discard real information -- this is
exactly the "may not be exact for full-reach decoders" caveat in
`docs/ks_local_field_ltransfer.tex` H1.

**R-localization, not covariance (Schur-product) localization.** The
dense `LETKF` Schur-products a taper onto `B` directly. This module
instead inflates each used observation's error variance by
`1/taper(distance)` (Hunt et al. 2007's own R-localization form) --
the standard alternative formulation, equivalent to covariance
localization when the observation operator is itself local (true here,
by the zero-decoder-reach precondition) and asymptotically equivalent
in general; this is WHY H1 predicts "within a few percent," not bitwise
agreement, against the dense filter.

**Analysis update, otherwise identical to `ks_latent.da.letkf.LETKF`**
(perturbed-observation / stochastic EnKF, same family, same
ensemble-mean-Jacobian linearization for the gain, same innovation
computed from the true nonlinear `h`) -- restricted to site `s`'s own
`channels` output rows and to the (small, locality-bounded) set of
observations with nonzero taper weight to site `s`:

    used_i          = {i : taper(dist(s, site(i))) > 0}       (compact support, exact zero beyond 2*gc_c)
    R_loc_i         = R_i / taper(dist(s, site(i)))             for i in used_i
    B_block         = Cov(Z_pert[:, used-sites-cols])           local slice only, never full (d,d)
    HB_s            = Cov(z_s, z_used) @ J_used^T               (channels, m_s)
    HBHt            = J_used @ B_block @ J_used^T               (m_s, m_s)
    K_s             = HB_s @ (HBHt + diag(R_loc))^-1             (channels, m_s)
    Z_analysis[:,s] = Z0[:,s] + (y_ens_used - h(Z0)_used) @ K_s^T

`h(Z0)` (the true nonlinear decoder+observation map) is evaluated once,
globally, for the whole ensemble -- shared across every site's local
analysis -- exactly like the dense filter's own `Hz`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ks_latent.da.jacobian_coloring import coloured_obs_jacobian_2d_compact
from ks_latent.da.localization import build_latent_taper_matrix_2d

ObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (d,) -> (m,), float64


@dataclass(frozen=True)
class LocalLETKFResult:
    Z_analysis: torch.Tensor  # (N, d)


@torch.no_grad()
def _relax_to_prior_spread(Z0: torch.Tensor, Za: torch.Tensor, alpha: float) -> torch.Tensor:
    """RTPS: rescale each coordinate's analysis perturbations so that its spread becomes
    sigma_a + alpha (sigma_b - sigma_a); the analysis mean is kept. Coordinates whose analysis
    spread is zero are left unchanged."""
    sb = Z0.std(dim=0)
    za_bar = Za.mean(dim=0)
    Ap = Za - za_bar
    sa = Ap.std(dim=0)
    safe = sa > 0
    scale = torch.ones_like(sa)
    scale[safe] = 1 + alpha * (sb[safe] - sa[safe]) / sa[safe]
    return za_bar + Ap * scale


class LETKFLocal2D:
    """Per-site local LETKF, same `analyze(Z0, y) -> result.Z_analysis`
    interface as `ks_latent.da.letkf.LETKF`/`ks_latent.da.pff.
    ParticleFlowFilter`, so it is a drop-in replacement wherever that
    interface is used (`ks_latent.da.cycling.run_da_experiment` via
    `CycleConfig.method="letkf_local2d"`). See module docstring for the
    algorithm and why it is needed instead of the dense filter at
    transfer scale.

    `R_diag`: `(m,)`, float64 -- diagonal observation error variance
    (this project's own convention, `obs_noise_std**2` uniformly;
    accepted as a vector rather than assumed scalar so a spatially
    varying observation network is not precluded). `site_of_obs`:
    `(m,)` int64, from `ks_latent.da.jacobian_coloring.obs_site_map_2d`.
    `gc_c`: Gaspari-Cohn half-width in SITE units, same convention as
    `ks_latent.da.localization.build_latent_taper_matrix_2d`."""

    def __init__(
        self,
        h: ObsOperator,
        R_diag: torch.Tensor,
        site_of_obs: torch.Tensor,
        n_sites_x: int,
        n_sites_y: int,
        channels: int,
        gc_c: float,
        inflation: float = 1.0,
        seed: int = 0,
        min_weight: float = 1e-10,
        rtps: float = 0.0,
    ):
        self.h = h
        self.R_diag = R_diag
        self.site_of_obs = site_of_obs
        self.n_sites_x = n_sites_x
        self.n_sites_y = n_sites_y
        self.channels = channels
        self.gc_c = gc_c
        self.inflation = inflation
        self.min_weight = min_weight
        if not 0.0 <= rtps <= 1.0:
            raise ValueError(f"rtps must be in [0, 1], got {rtps}")
        # Relaxation to prior spread (Whitaker & Hamill 2012, Mon. Wea. Rev. 140:3078), per latent
        # coordinate: sigma_a <- sigma_a + rtps * (sigma_b - sigma_a), applied to the analysis
        # perturbations after the update; the analysis mean is unchanged. Task I21 (2026-10-09):
        # the latent ensemble's spread is ~0.3 of its error, so its gain is too small.
        self.rtps = rtps
        self._rng = torch.Generator().manual_seed(seed)
        self.n_sites = n_sites_x * n_sites_y
        self.d = self.n_sites * channels
        # (P, P) site-level taper (channels=1: `build_latent_taper_matrix_2d`
        # with channels=1 degenerates to exactly the site distance taper,
        # repeat_interleave with multiplicity 1 is a no-op).
        self._site_taper = build_latent_taper_matrix_2d(n_sites_x, n_sites_y, channels=1, c=gc_c).to(torch.float64)

    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> LocalLETKFResult:
        """`Z0`: `(N, d)` forecast ensemble, float64. `y`: `(m,)`
        observation, float64."""
        N, d = Z0.shape
        if d != self.d:
            raise ValueError(f"Z0 has d={d}, expected {self.d} ({self.n_sites} sites x {self.channels} channels)")
        c = self.channels
        z_bar = Z0.mean(dim=0)
        Z_pert = (Z0 - z_bar) * self.inflation  # (N, d) -- O(N*d), never O(d^2)

        # Detached: otherwise every per-site product below is recorded in one
        # autograd graph that lives until analyze() returns, so memory grows
        # with the number of sites (~3 GB at 256 sites, ~20 GB at 1600).
        J_compact = coloured_obs_jacobian_2d_compact(self.h, z_bar, self.site_of_obs, c).detach()  # (m, c)
        result = self._analyze_no_grad(Z0, y, Z_pert, J_compact)
        if self.rtps > 0:
            result = LocalLETKFResult(Z_analysis=_relax_to_prior_spread(Z0, result.Z_analysis, self.rtps))
        return result

    @torch.no_grad()
    def _analyze_no_grad(self, Z0, y, Z_pert, J_compact) -> LocalLETKFResult:
        N = Z0.shape[0]
        c = self.channels
        z_bar = Z0.mean(dim=0)
        Hz = torch.stack([self.h(Z0[i]) for i in range(N)], dim=0)  # (N, m) -- shared across all sites

        m = y.shape[0]
        L_R = torch.sqrt(self.R_diag)
        obs_noise = L_R.unsqueeze(0) * torch.randn(N, m, generator=self._rng, dtype=y.dtype)  # (N, m)
        y_ens = y.unsqueeze(0) + obs_noise  # (N, m)
        innovation_full = y_ens - Hz  # (N, m)

        Za = Z0.clone()
        for s in range(self.n_sites):
            rho = self._site_taper[s, self.site_of_obs]  # (m,), exactly 0 beyond support radius 2*gc_c
            used = torch.nonzero(rho > self.min_weight, as_tuple=True)[0]
            if used.numel() == 0:
                continue  # no observation within range of this site: no gain, Z0's own slice is kept

            used_sites = torch.unique(self.site_of_obs[used])  # the (small) neighbourhood of occupied sites
            # Column block for every occupied neighbour site's c channels, concatenated in `used_sites` order.
            block_cols = (used_sites.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)).reshape(-1)  # (n_used_sites*c,)
            Z_pert_block = Z_pert[:, block_cols]  # (N, n_used_sites*c) -- local slice only, never full (N,d)

            idx_s = s * c + torch.arange(c)
            Z_pert_s = Z_pert[:, idx_s]  # (N, c)

            m_s = used.numel()
            # Scatter each used observation's compact (c,) row into its
            # occupied-site column block -- each row has nonzero entries
            # in exactly one c-wide block, by block-diagonality.
            site_pos = torch.searchsorted(used_sites, self.site_of_obs[used])  # position of each obs's site in used_sites
            J_used = torch.zeros(m_s, used_sites.numel() * c, dtype=z_bar.dtype)
            row_idx = torch.arange(m_s)
            col_idx = (site_pos.unsqueeze(1) * c + torch.arange(c).unsqueeze(0))  # (m_s, c)
            J_used[row_idx.unsqueeze(1), col_idx] = J_compact[used]  # (m_s, c) -> scattered

            B_block = (Z_pert_block.T @ Z_pert_block) / (N - 1)  # (n_used_sites*c, n_used_sites*c)
            B_rows_s = (Z_pert_s.T @ Z_pert_block) / (N - 1)  # (c, n_used_sites*c) -- Cov(z_s, z_used-block)

            HB_s = B_rows_s @ J_used.T  # (c, m_s)
            HBHt = J_used @ B_block @ J_used.T  # (m_s, m_s)
            R_loc = self.R_diag[used] / rho[used]  # Hunt et al. 2007 R-localization
            gain_denom = HBHt + torch.diag(R_loc)  # (m_s, m_s), SPD
            K_s = torch.linalg.solve(gain_denom, HB_s.T).T  # (c, m_s): K_s^T = gain_denom^-1 @ HB_s^T

            innovation_s = innovation_full[:, used]  # (N, m_s)
            Za[:, idx_s] = Z0[:, idx_s] + innovation_s @ K_s.T  # (N, c)

        # The Jacobian is computed with autograd; the analysis must not carry
        # that graph, or a cycling loop that stores analyses retains every
        # cycle's graph (observed: ~0.24 GB per cycle at d=384).
        return LocalLETKFResult(Z_analysis=Za.detach())


class LETKFLocal2DETKF(LETKFLocal2D):
    """Deterministic per-site ensemble transform (task D4, docs/full_paper_tasks.md): the
    Hunt, Kostelich & Szunyogh (2007) LETKF analysis in ensemble space, one local analysis per
    site, with exactly the same R-localization, observation selection and compact
    block-diagonal Jacobian as `LETKFLocal2D` -- only the update differs (no perturbed
    observations, so no sampling noise from the observation draws).

    Per site `s`, with `used`, `rho` and `R_loc = R / rho` as in `LETKFLocal2D`:

        Y_b      = Z_pert_block @ J_used^T                       (N, m_s) obs-space perturbations
        P~_a     = [(N-1) I + Y_b R_loc^-1 Y_b^T]^-1              (N, N)
        w_bar    = P~_a Y_b R_loc^-1 (y - mean_j h(Z0_j))_used    (N,)
        W_a      = [(N-1) P~_a]^{1/2}                             symmetric square root
        Za[:, s] = z_bar_s + (w_bar 1^T + W_a)^T Z_pert_s

    `Y_b` uses the linearized (ensemble-mean Jacobian) map, like the perturbed-observation
    filter's gain; the innovation uses the nonlinear ensemble-mean `h`. For a linear `h` and
    no localization the analysis mean and covariance equal the Kalman update computed with
    the forecast ensemble's own sample covariance, exactly (test_letkf_local.py). The
    `seed` argument is accepted for interface compatibility and unused."""

    @torch.no_grad()
    def _analyze_no_grad(self, Z0, y, Z_pert, J_compact) -> LocalLETKFResult:
        N = Z0.shape[0]
        c = self.channels
        z_bar = Z0.mean(dim=0)
        Hz = torch.stack([self.h(Z0[i]) for i in range(N)], dim=0)  # (N, m)
        d_full = y - Hz.mean(dim=0)  # (m,) innovation of the ensemble mean
        # Obs-space perturbations for every observation at once: obs i reads only its own
        # site's c channels (block-diagonal Jacobian), so Y_b[:, i] = Z_pert[:, block(i)] @ J_i.
        cols = self.site_of_obs.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)  # (m, c)
        Yb_full = (Z_pert[:, cols] * J_compact.unsqueeze(0)).sum(-1)  # (N, m)

        Za = Z0.clone()
        eye = torch.eye(N, dtype=Z0.dtype)
        for s in range(self.n_sites):
            rho = self._site_taper[s, self.site_of_obs]
            used = torch.nonzero(rho > self.min_weight, as_tuple=True)[0]
            if used.numel() == 0:
                continue
            Yb = Yb_full[:, used]  # (N, m_s)
            Rinv = rho[used] / self.R_diag[used]  # diag of R_loc^-1
            C = Yb * Rinv.unsqueeze(0)  # (N, m_s) = Y_b R_loc^-1
            A = (N - 1) * eye + C @ Yb.T  # (N, N), SPD
            evals, evecs = torch.linalg.eigh(A)
            if float(evals.min()) <= 0:
                raise FloatingPointError(f"site {s}: ensemble-space matrix not SPD (min eig {float(evals.min()):.3e})")
            Pa = (evecs / evals) @ evecs.T
            w_bar = Pa @ (C @ d_full[used])  # (N,)
            Wa = (evecs * torch.sqrt((N - 1) / evals)) @ evecs.T  # symmetric sqrt of (N-1) P~_a
            W = w_bar.unsqueeze(1) + Wa  # (N, N)
            idx_s = s * c + torch.arange(c)
            Za[:, idx_s] = z_bar[idx_s] + W.T @ Z_pert[:, idx_s]
        return LocalLETKFResult(Z_analysis=Za.detach())


class LETKFLocal2DIterative(LETKFLocal2D):
    """Per-site perturbed-observation update relinearized by Gauss--Newton iterations (task Z20 E3).

    Each member minimizes `|z - z_f^(e)|^2_{B^-1} + |y^(e) - H(z)|^2_{R^-1}` (the same objective the
    one-step update solves exactly when `H` is linear), by the ensemble randomized-maximum-likelihood
    iteration with one Jacobian per iterate, evaluated at the iterate's ensemble mean:

        z_{i+1}^(e) = z_f^(e) + K_i [ y^(e) - H(z_i^(e)) - J_i (z_f^(e) - z_i^(e)) ],
        K_i         = B J_i^T (J_i B J_i^T + R)^-1,     B = forecast sample covariance (fixed),

    localized per site exactly as `LETKFLocal2D` (R-localization, the same observations, the same
    perturbed-observation draws in every iteration). Iteration 0 starts at `z_0 = z_f`, so
    `n_iter = 1` is `LETKFLocal2D` exactly. All sites are updated from the previous iterate (a Jacobi
    sweep); `J_i` is recomputed by the same `channels` forward-mode passes (block-diagonal at zero
    decoder reach), so the exactness of Proposition 1 holds at every iterate. Stops early when the
    relative change of the ensemble mean falls below `tol`. RTPS is applied once, after the last
    iterate, against the forecast spread. For a linear `H` every iterate equals the first."""

    def __init__(self, *args, n_iter: int = 2, tol: float = 1e-4, **kw):
        super().__init__(*args, **kw)
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.n_iter, self.tol = n_iter, tol
        self.iterations_used = 0
        self.mean_changes: list[float] = []

    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> LocalLETKFResult:
        N, d = Z0.shape
        if d != self.d:
            raise ValueError(f"Z0 has d={d}, expected {self.d}")
        c = self.channels
        Z_pert = (Z0 - Z0.mean(dim=0)) * self.inflation
        m = y.shape[0]
        y_ens = y.unsqueeze(0) + torch.sqrt(self.R_diag).unsqueeze(0) * torch.randn(N, m, generator=self._rng, dtype=y.dtype)
        cols = self.site_of_obs.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)  # (m, c)
        Zi = Z0.clone()
        self.mean_changes = []
        for it in range(self.n_iter):
            zi_bar = Zi.mean(dim=0)
            J = coloured_obs_jacobian_2d_compact(self.h, zi_bar, self.site_of_obs, c).detach()
            with torch.no_grad():
                Hz = torch.stack([self.h(Zi[e]) for e in range(N)], dim=0)
                lin = ((Z0 - Zi)[:, cols] * J.unsqueeze(0)).sum(-1)  # J_i (z_f - z_i), (N, m)
                innov = y_ens - Hz - lin
                Znew = self._site_updates(Z0, Z_pert, J, innov)
            change = float((Znew.mean(0) - zi_bar).norm() / zi_bar.norm().clamp_min(1e-12))
            self.mean_changes.append(change)
            Zi = Znew
            self.iterations_used = it + 1
            if it > 0 and change < self.tol:
                break
        if self.rtps > 0:
            Zi = _relax_to_prior_spread(Z0, Zi, self.rtps)
        return LocalLETKFResult(Z_analysis=Zi.detach())

    @torch.no_grad()
    def _site_updates(self, Z0, Z_pert, J_compact, innovation_full):
        """`LETKFLocal2D._analyze_no_grad`'s per-site gain with a given innovation."""
        N = Z0.shape[0]
        c = self.channels
        Za = Z0.clone()
        for s in range(self.n_sites):
            rho = self._site_taper[s, self.site_of_obs]
            used = torch.nonzero(rho > self.min_weight, as_tuple=True)[0]
            if used.numel() == 0:
                continue
            used_sites = torch.unique(self.site_of_obs[used])
            block_cols = (used_sites.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)).reshape(-1)
            Zb = Z_pert[:, block_cols]
            idx_s = s * c + torch.arange(c)
            m_s = used.numel()
            site_pos = torch.searchsorted(used_sites, self.site_of_obs[used])
            J_used = torch.zeros(m_s, used_sites.numel() * c, dtype=Z0.dtype)
            J_used[torch.arange(m_s).unsqueeze(1), site_pos.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)] = J_compact[used]
            B_block = (Zb.T @ Zb) / (N - 1)
            HB_s = ((Z_pert[:, idx_s].T @ Zb) / (N - 1)) @ J_used.T
            denom = J_used @ B_block @ J_used.T + torch.diag(self.R_diag[used] / rho[used])
            K_s = torch.linalg.solve(denom, HB_s.T).T
            Za[:, idx_s] = Z0[:, idx_s] + innovation_full[:, used] @ K_s.T
        return Za
