"""Domain-localized (per-site) natural-gradient particle flow filter for the CM1 column local field (plan E6b).

The global NAT-PFF of `ks_latent/da/pff.py` (derivation there) moves every particle of the whole latent field with one flow whose
prior covariance B and Gauss-Newton metric F = J^T R^-1 J + B^-1 are dense in the full latent dimension (P sites x c channels).
Here, as in the per-site LETKF (`letkf.py`), each site s gets its own small flow on a WINDOW of K = 2w + 1 neighbouring sites
(clipped, not wrapped: the lateral boundaries are open), using only the observations within the localization radius, and only the
window's centre site is kept:

  state      z_w = latent of the K window sites, channels 0..m-1 (m = the pruned latent size, all c if unpruned): d = K m
  prior      z0 = particle mean, B = (sample covariance of the window) o (Gaspari-Cohn taper over the site distance) + ridge
  obs        the observations i within 2 c_loc km of the site, R_i -> R_i / GC(|x_s - x_i| / c_loc) (R-localization, as the LETKF)
  Jacobian   J_i = d h_i / d z_w from the global Jacobian of the observation operator (decoder, reflectivity head, radial velocity)
             at the CURRENT mean field, restricted to the window; sensitivities to sites outside the window are dropped (those
             sites are held, as in any domain localization)
  flow       exactly pff.py: re-linearize at the current mean, c = B^-1 z0 + A zbar + J^T R^-1 d, F = A + B^-1, z* = F^-1 c,
             z <- z* + e^-ds (z - z*) + sqrt(1 - e^-2ds) L eta  (L L^T = F^-1, eta mean-centred), adaptive ds, early stop.
All sites flow in lock-step, so the global Jacobian is computed once per pseudo-step from the mean field assembled from the sites'
centres (the same mean every window would see if it were global). With no localization (window = whole domain, huge radii) and a
linear observation operator this is the global NAT-PFF, and its ensemble mean and covariance are the Kalman posterior
(tests/unit/test_cm1_pff_site.py). Differences from the LETKF: the posterior is a Gaussian resampled around the Gauss-Newton mode, with
the prior covariance fixed (not the ensemble-space transform), and the observation operator is re-linearized at the evolving mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.func import jacrev

from ks_latent.cm1.letkf import gaspari_cohn


@dataclass
class PFFSiteConfig:
    loc_km: float = 3.0         # Gaspari-Cohn half-width of the observation localization (weights vanish at 2 c)
    taper_km: float = 4.0       # Gaspari-Cohn half-width of the taper on B over the site distance
    half_width: int = 6         # window half-width in sites (K = 2 w + 1)
    n_steps: int = 12
    ds_max: float = 10.0
    tol: float = 1e-4
    early_stop_after: int = 5
    ridge: float = 1e-2         # B += ridge * mean variance * I (and an absolute floor) so that B^-1 exists
    ridge_floor: float = 1e-8
    stochastic: bool = True     # False: DET (collapses the spread; the mean is the analysis)
    rtps: float = 0.0
    site_km: float = 2.0        # km per site (8 cells of 250 m)


def _gc(x: torch.Tensor, c: float) -> torch.Tensor:
    return torch.as_tensor(gaspari_cohn(x.detach().cpu().numpy() / c), dtype=x.dtype, device=x.device)


@torch.no_grad()
def pff_site_analysis(Zb: torch.Tensor, obs_fn: Callable[[torch.Tensor], torch.Tensor], y: np.ndarray, r_var: np.ndarray,
                      x_site_km: np.ndarray, x_obs_km: np.ndarray, cfg: PFFSiteConfig, m: int | None = None,
                      gen: torch.Generator | None = None, jac_chunk: int = 256) -> tuple[torch.Tensor, dict]:
    """`Zb` (N, P, c) forecast ensemble; `obs_fn(z)`: (P, c) latent state -> (n_obs,) observation-space values (differentiable in z
    channels < m); `y`, `r_var`, `x_obs_km` (n_obs,); `x_site_km` (P,). Channels >= m (a pruned latent) are not updated.
    Returns the analysis ensemble (N, P, c) and a dict of diagnostics."""
    dev = Zb.device
    dt = torch.float64
    N, P, c = Zb.shape
    mm = c if m is None else m
    K = 2 * cfg.half_width + 1
    if K > P:
        raise ValueError(f"window of {K} sites wider than the domain {P}")
    y_t = torch.as_tensor(y, dtype=dt, device=dev)
    rv = torch.as_tensor(np.array(r_var), dtype=dt, device=dev)
    xs = torch.as_tensor(x_site_km, dtype=dt, device=dev)
    xo = torch.as_tensor(x_obs_km, dtype=dt, device=dev)
    n_obs = y_t.numel()
    Zf = Zb[..., :mm].to(dt)
    tail = Zb[..., mm:]
    # windows (clipped at the edges, never repeated sites)
    ar = torch.arange(P, device=dev)
    start = (ar - cfg.half_width).clamp(0, P - K)
    widx = start[:, None] + torch.arange(K, device=dev)[None]                 # (P, K)
    cen = ar - start                                                          # centre offset inside the window
    d = K * mm
    Zw = Zf[:, widx].permute(1, 0, 2, 3).reshape(P, N, d)                     # (P, N, d), site-major then channel
    z0 = Zw.mean(1)                                                           # prior mean per window (P, d)
    Xp = Zw - z0[:, None]
    cov = Xp.transpose(1, 2) @ Xp / max(N - 1, 1)                             # (P, d, d)
    kk = torch.arange(K, device=dev, dtype=dt)
    rho_k = _gc((kk[:, None] - kk[None, :]).abs() * cfg.site_km, cfg.taper_km)  # (K, K)
    rho = torch.kron(rho_k, torch.ones(mm, mm, dtype=dt, device=dev))         # (d, d)
    B = cov * rho
    tr = torch.diagonal(B, dim1=-2, dim2=-1).mean(-1)                         # (P,)
    B = B + (cfg.ridge * tr + cfg.ridge_floor)[:, None, None] * torch.eye(d, dtype=dt, device=dev)
    Binv = torch.linalg.inv(B)
    # observation localization: per site the nearest observations with positive weight
    wobs = _gc((xs[:, None] - xo[None, :]).abs(), cfg.loc_km)                 # (P, n_obs)
    nmax = int((wobs > 0).sum(1).max())
    if nmax == 0:
        raise ValueError("no observation within the localization radius of any site")
    wsel, idx_o = torch.topk(wobs, nmax, dim=1)                               # (P, nmax)
    Rinv = wsel / rv[idx_o]                                                   # (P, nmax); zero weight = ignored
    Z = Zw.clone()
    ds = torch.ones(P, dtype=dt, device=dev)
    info = {"n_iterations": 0, "rel_change": []}
    ctr = lambda Zc: Zc.reshape(P, N, K, mm)[ar, :, cen]                      # (P, N, mm) centre-site particles
    tail_mean = tail.mean(0).to(dt) if tail.shape[-1] else None
    for s in range(cfg.n_steps):
        zc = ctr(Z).mean(1)                                                   # current mean field (P, mm)
        zfull = zc if tail_mean is None else torch.cat([zc, tail_mean], -1)

        def hfree(zf, zfull=zfull):
            return obs_fn(torch.cat([zf, zfull[:, mm:]], -1).to(Zb.dtype)).to(dt)

        hbar = hfree(zc)
        J = jacrev(hfree, chunk_size=jac_chunk)(zc)                           # (n_obs, P, mm)
        dres = (y_t - hbar)[idx_o]                                            # (P, nmax)
        Jl = J[idx_o[:, :, None], widx[:, None, :]].reshape(P, nmax, d)       # (P, nmax, K*mm)
        JtR = Jl.transpose(1, 2) * Rinv[:, None, :]                           # (P, d, nmax)
        A = JtR @ Jl
        F = A + Binv
        zbar = Z.mean(1)
        cvec = (Binv @ z0[..., None])[..., 0] + (A @ zbar[..., None])[..., 0] + (JtR @ dres[..., None])[..., 0]
        Finv = torch.linalg.inv(F)
        Finv = 0.5 * (Finv + Finv.transpose(1, 2))
        zstar = (Finv @ cvec[..., None])[..., 0]
        f = zstar[:, None] - Z
        fn = f.mean(1).norm(dim=-1)
        ds_test = torch.clamp(1.0 / (fn + 1e-8), 1.0, cfg.ds_max)
        ds = torch.ones_like(ds) if s == 0 else torch.where(ds_test > ds, torch.minimum(1.1 * ds, ds_test), torch.maximum(0.9 * ds, ds_test))
        decay = torch.exp(-ds)[:, None, None]
        Znew = zstar[:, None] + decay * (Z - zstar[:, None])
        if cfg.stochastic:
            L = torch.linalg.cholesky(Finv + 1e-12 * torch.eye(d, dtype=dt, device=dev))
            # one noise field per step, SHARED by the overlapping windows (each window reads its sites from it): with independent
            # draws per window the centre sites' noise would be independent across sites and the analysis ensemble would lose all
            # spatial coherence (the LETKF gets this from smoothly varying weights)
            eg = torch.randn(N, P, mm, generator=gen, device="cpu", dtype=dt).to(dev)
            eg = eg - eg.mean(0, keepdim=True)
            eta = eg[:, widx].permute(1, 0, 2, 3).reshape(P, N, d)
            Znew = Znew + torch.sqrt(1 - decay ** 2) * (eta @ L.transpose(1, 2))
        rel = float(((Znew.mean(1) - Z.mean(1)).norm(dim=-1) / (Z.mean(1).norm(dim=-1) + 1e-12)).max())
        Z = Znew
        info["rel_change"].append(rel); info["n_iterations"] = s + 1
        if s > cfg.early_stop_after and rel < cfg.tol:
            break
    Za_c = ctr(Z).transpose(0, 1)                                             # (N, P, mm)
    # a site with no observation inside its localization radius is left exactly as forecast: the flow would otherwise redraw its
    # particles from the (tapered) prior, discarding the forecast ensemble's own structure where there is nothing to learn
    active = (Rinv.sum(1) > 0)[None, :, None]
    Za_c = torch.where(active, Za_c, Zf)
    if cfg.rtps > 0:
        sb = Zf.std(0, unbiased=True)
        za_bar = Za_c.mean(0)
        sa = (Za_c - za_bar).std(0, unbiased=True)
        fac = torch.where(sa > 0, 1 + cfg.rtps * (sb - sa) / torch.where(sa > 0, sa, torch.ones_like(sa)), torch.ones_like(sa))
        Za_c = za_bar + (Za_c - za_bar) * fac
    Za = torch.cat([Za_c.to(Zb.dtype), tail], -1) if tail.shape[-1] else Za_c.to(Zb.dtype)
    info["n_obs_per_site_max"] = nmax
    return Za, info
