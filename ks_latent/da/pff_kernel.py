"""Kernel particle flow filter (Hu & van Leeuwen 2021, QJRMS 147:2352) for the latent field (Z20 E6).

Unlike `ks_latent.da.pff.ParticleFlowFilter` -- whose "NAT" flow shares one Gauss--Newton precision,
built at the ensemble mean, across all particles and therefore has the Gaussian N(z*, F^-1) as its
stationary distribution -- this flow uses each particle's own gradient of the log posterior, through
the NONLINEAR observation operator, and a kernel that lets particles interact. It is a Stein variational
flow with a diagonal matrix-valued kernel and the (localized) prior covariance as preconditioner:

    dz_i/ds = B_loc (1/N) sum_j [ K(z_j, z_i) grad log p(z_j | y) + div_{z_j} K(z_j, z_i) ],
    grad log p(z | y) = -B_loc^-1 (z - z_f_bar) + J(z)^T R^-1 (y - H(z)),
    K(z, z')_dd = exp( -(z_d - z'_d)^2 / (2 alpha sigma_d^2) )      (one scalar kernel per dimension d),

with sigma_d the prior standard deviation of coordinate d and alpha the kernel width (Hu & van Leeuwen
use alpha = 1/N_ens). The per-dimension kernel makes the k-th component of the flow use kernel
K_kk only, and its divergence term is d/dz_{j,k} K_kk(z_j, z_i) = -(z_{j,k} - z_{i,k}) / (alpha sigma_k^2) K_kk.
`J(z_j)^T R^-1 (y - H(z_j))` is one vector-Jacobian product per particle (one backward pass for the
whole ensemble). `B_loc = rho o B` (Schur product with a positive-definite taper), held fixed in
pseudo-time; a ridge `ridge * mean(diag B)` keeps it invertible. Explicit Euler in pseudo-time with an
adaptive step: the largest step such that no particle moves more than `max_move` prior standard
deviations in any coordinate, at most `n_steps` steps, stopping when the relative change of the ensemble
mean is below `tol` (after 5 steps). For a linear H and Gaussian prior the flow's fixed point
approximates the Kalman posterior (test_pff_kernel.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ks_latent.utils.linalg import robust_cholesky

BatchObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (N, d) -> (N, m), differentiable


@dataclass(frozen=True)
class KernelPFFResult:
    Z_analysis: torch.Tensor
    n_steps: int
    converged: bool


class KernelPFF:
    def __init__(self, h_batch: BatchObsOperator, R_diag: torch.Tensor, taper: torch.Tensor | None = None,
                 alpha: float | None = None, n_steps: int = 200, max_move: float = 0.05, tol: float = 1e-4,
                 ridge: float = 1e-6, rtps: float = 0.0):
        self.h = h_batch
        self.Rinv = 1.0 / R_diag.to(torch.float64)
        self.taper = taper
        self.alpha, self.n_steps, self.max_move, self.tol, self.ridge, self.rtps = alpha, n_steps, max_move, tol, ridge, rtps

    def _grad_lik(self, Z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Zr = Z.detach().clone().requires_grad_(True)
        r = y.unsqueeze(0) - self.h(Zr)  # (N, m)
        ll = -0.5 * (r.pow(2) * self.Rinv.unsqueeze(0)).sum()
        (g,) = torch.autograd.grad(ll, Zr)
        return g.detach()  # (N, d): J(z_j)^T R^-1 (y - H(z_j))

    def analyze(self, Z0: torch.Tensor, y: torch.Tensor) -> KernelPFFResult:
        Z0 = Z0.to(torch.float64); y = y.to(torch.float64)
        N, d = Z0.shape
        alpha = self.alpha if self.alpha is not None else 1.0 / N
        zf = Z0.mean(0)
        X = Z0 - zf
        B = X.T @ X / (N - 1)
        if self.taper is not None:
            B = B * self.taper
        B = B + self.ridge * float(B.diagonal().mean()) * torch.eye(d, dtype=torch.float64)
        L = robust_cholesky(B)
        sig2 = B.diagonal().clamp_min(1e-30)  # (d,)
        sig = sig2.sqrt()
        Z = Z0.clone()
        converged, s = False, 0
        for s in range(1, self.n_steps + 1):
            g_prior = -torch.cholesky_solve((Z - zf).T, L).T  # (N, d)
            G = g_prior + self._grad_lik(Z, y)  # grad log posterior per particle
            D = Z.unsqueeze(1) - Z.unsqueeze(0)  # D[j, i, :] = z_j - z_i
            K = torch.exp(-0.5 * D.pow(2) / (alpha * sig2))  # (N_j, N_i, d)
            drive = (K * G.unsqueeze(1)).sum(0) / N  # (N_i, d)
            repulse = (-(D / (alpha * sig2)) * K).sum(0) / N  # div_{z_j} K(z_j, z_i)
            f = (drive + repulse) @ B  # B symmetric: (B v_i)^T = v_i^T B
            step = self.max_move / float((f.abs() / sig).max().clamp_min(1e-30))
            Znew = Z + min(step, 1.0) * f
            if not torch.isfinite(Znew).all():
                raise FloatingPointError(f"non-finite particle at pseudo-time step {s}")
            change = float((Znew.mean(0) - Z.mean(0)).norm() / Z.mean(0).norm().clamp_min(1e-12))
            Z = Znew
            if s > 5 and change < self.tol:
                converged = True
                break
        if self.rtps > 0:
            from ks_latent.da.letkf_local import _relax_to_prior_spread
            Z = _relax_to_prior_spread(Z0, Z, self.rtps)
        return KernelPFFResult(Z_analysis=Z.detach(), n_steps=s, converged=converged)
