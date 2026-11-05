"""Latent DA twin experiments on CM1 (E3): cycle the per-site LETKF with the learned surrogate.

The truth is a held-out CM1 member (true columns, never decoded); observations are taken from the true
columns with Gaussian noise. The initial ensemble is the encoded state of OTHER members at the start
time, so nothing is derived from the truth. Each cycle:

  forecast   every member runs `steps_per_cycle` propagator steps; the `halo` sites at each domain
             edge are set to the encoded truth before every step (lateral boundary conditions, as a
             limited-area model gets them; the storm stays far from the edges). Optional model-error
             noise: N(0, model_noise^2) added to every latent at every step (the propagator is
             deterministic, so without it the ensemble spread collapses over the cycles: found in the
             first run, spread 0.05 K against an error of 0.8 K at the observations)
  analysis   per-site LETKF (`letkf.py`) with Yb = H(decode(member)) for every member
  scores     against the true columns, for the prior (forecast) and the analysis:
             - error of the mean of the decoded members (`mean_dec`, Paper II's better estimate) and of
               the decoded latent mean (`dec_mean`): normalized MSE over all channels, and RMSE per
               variable in physical units (observed theta' and unobserved w, condensate, ...);
             - surface theta' RMSE at the observed columns, ensemble spread there (consistency);
             - front displacement of the decoded-members mean (`verify.displacement_km`);
             - rank of each observation among the members (rank histogram)
A no-DA run from the same initial ensemble is the baseline (`run_twin(..., assimilate=False)`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ks_latent.cm1.data import VARS, ColumnData, Normalizer
from ks_latent.cm1.letkf import LETKFConfig, letkf_analysis
from ks_latent.cm1.models import ColumnFieldAE, StencilPropagator
from ks_latent.cm1.verify import displacement_km, main_pool_front_km


@dataclass
class TwinConfig:
    t0: int = 60                  # start frame (index into the truth member's frames)
    n_cycles: int = 36
    steps_per_cycle: int = 5      # propagator steps (frames) between analyses
    obs_every_cells: int = 8      # observe surface theta' every this many grid cells (8 = 2 km)
    sigma_obs: float = 0.5        # K
    letkf: LETKFConfig = field(default_factory=LETKFConfig)
    halo: int = 3
    model_noise: float = 0.0      # std of additive latent noise per forecast step (latent units)
    seed: int = 0


def _forecast(prop: StencilPropagator, Z: torch.Tensor, z_truth: torch.Tensor, t: int, steps: int,
              halo: int, model_noise: float = 0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """Advance the latent ensemble `Z` (N, P, c) from frame t by `steps` frames, truth at the edges,
    optionally with additive model-error noise of std `model_noise` per step."""
    P = Z.shape[1]
    for j in range(steps):
        Z = prop(Z)
        if model_noise > 0:
            Z = Z + model_noise * torch.randn(Z.shape, generator=gen, device="cpu").to(Z.device, Z.dtype)
        if halo > 0:
            zt = z_truth[t + j + 1].expand(Z.shape[0], -1, -1)
            Z = torch.cat([zt[:, :halo], Z[:, halo:P - halo], zt[:, P - halo:]], dim=1)
    return Z


@torch.no_grad()
def _scores(ae: ColumnFieldAE, Z: torch.Tensor, u_true: torch.Tensor, obs_cols: np.ndarray, y: np.ndarray | None,
            norm: Normalizer, Z_levels: int) -> dict:
    """Scores of ensemble `Z` against the true normalized columns `u_true` (C, NX)."""
    dec = ae.decode(Z)                                       # (N, C, NX)
    mean_dec = dec.mean(0)
    dec_mean = ae.decode(Z.mean(0, keepdim=True))[0]
    V = len(VARS)
    sc = torch.tensor(norm.scale, dtype=torch.float32, device=dec.device)
    def per_var(est):
        e = ((est - u_true) ** 2).reshape(V, Z_levels, -1).mean(dim=(1, 2)).sqrt() * sc
        return dict(zip(VARS, e.tolist()))
    ith = VARS.index("th") * Z_levels
    s_th = float(norm.scale[VARS.index("th")])
    th_true = (u_true[ith] * s_th).double().cpu().numpy()
    th_mean = (mean_dec[ith] * s_th).double().cpu().numpy()
    th_mem = (dec[:, ith, obs_cols] * s_th).double().cpu().numpy()       # (N, m)
    front = main_pool_front_km(th_true)
    out = {"nmse_mean_dec": float(((mean_dec - u_true) ** 2).mean()),
           "nmse_dec_mean": float(((dec_mean - u_true) ** 2).mean()),
           "rmse_phys_mean_dec": per_var(mean_dec),
           "th_obs_rmse": float(np.sqrt(np.mean((th_mem.mean(0) - th_true[obs_cols]) ** 2))),
           "th_obs_spread": float(np.sqrt(np.mean(th_mem.var(0, ddof=1)))),
           "front_true_km": float(front), "front_disp_km": displacement_km(th_mean, th_true, center_km=front)}
    if y is not None:
        out["ranks"] = (th_mem < y[None, :]).sum(0).tolist()   # rank of each observation among members
    return out


@torch.no_grad()
def run_twin(ae: ColumnFieldAE, prop: StencilPropagator, truth: ColumnData, z_truth: torch.Tensor,
             Z0: torch.Tensor, norm: Normalizer, cfg: TwinConfig, assimilate: bool = True) -> dict:
    """`truth`: ColumnData holding the truth member only (index 0); `z_truth` its encoded latent
    trajectory (T, P, c) (used only for the boundary halo); `Z0` (N, P, c) initial ensemble at frame
    `cfg.t0`. Returns per-cycle prior/analysis scores."""
    ae.eval(); prop.eval()
    dev = Z0.device
    rng = np.random.default_rng(cfg.seed)
    NX, Zl = truth.NX, truth.Z
    obs_cols = np.arange(cfg.obs_every_cells // 2, NX, cfg.obs_every_cells)
    ith = VARS.index("th") * Zl
    s_th = float(norm.scale[VARS.index("th")])
    x_obs = (obs_cols + 0.5) * 0.25
    P = Z0.shape[1]
    x_site = (np.arange(P) + 0.5) * ae.cfg.patch * 0.25
    r_var = np.full(obs_cols.size, cfg.sigma_obs ** 2)
    T_last = cfg.t0 + cfg.n_cycles * cfg.steps_per_cycle
    if T_last >= truth.T[0]:
        raise ValueError(f"truth has {truth.T[0]} frames; the experiment needs {T_last + 1}")
    Z = Z0.clone()
    gen = torch.Generator().manual_seed(cfg.seed)
    hist = []
    for k in range(cfg.n_cycles):
        t = cfg.t0 + k * cfg.steps_per_cycle
        Z = _forecast(prop, Z, z_truth, t, cfg.steps_per_cycle, cfg.halo, cfg.model_noise, gen)
        t1 = t + cfg.steps_per_cycle
        u_true = truth.full(0, t1, device=dev)[0]
        y = (u_true[ith, obs_cols] * s_th).double().cpu().numpy() + rng.normal(0.0, cfg.sigma_obs, obs_cols.size)
        rec = {"cycle": k, "frame": t1, "prior": _scores(ae, Z, u_true, obs_cols, y, norm, Zl)}
        if assimilate:
            Yb = (ae.decode(Z)[:, ith, obs_cols] * s_th).double().cpu().numpy()
            Za = letkf_analysis(Z.double().cpu().numpy(), Yb, y, r_var, x_site, x_obs, cfg.letkf)
            Z = torch.as_tensor(Za, dtype=Z.dtype, device=dev)
            rec["analysis"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl)
        hist.append(rec)
    return {"config": {"t0": cfg.t0, "n_cycles": cfg.n_cycles, "steps_per_cycle": cfg.steps_per_cycle,
                       "obs_every_cells": cfg.obs_every_cells, "sigma_obs": cfg.sigma_obs,
                       "loc_km": cfg.letkf.loc_km, "inflation": cfg.letkf.inflation, "rtps": cfg.letkf.rtps,
                       "model_noise": cfg.model_noise,
                       "n_members": int(Z0.shape[0]), "n_obs": int(obs_cols.size), "assimilate": assimilate},
            "cycles": hist}


def summarize_twin(res: dict) -> dict:
    """Means over the second half of the cycles (after spin-up) of the main scores."""
    cyc = res["cycles"][len(res["cycles"]) // 2:]
    out = {}
    for stage in ("prior", "analysis"):
        if stage not in cyc[0]:
            continue
        g = lambda k: float(np.mean([c[stage][k] for c in cyc]))
        out[stage] = {"nmse_mean_dec": g("nmse_mean_dec"), "nmse_dec_mean": g("nmse_dec_mean"),
                      "th_obs_rmse": g("th_obs_rmse"), "th_obs_spread": g("th_obs_spread"),
                      "front_abs_disp_km_median": float(np.nanmedian([abs(c[stage]["front_disp_km"]) for c in cyc])),
                      "rmse_phys": {v: float(np.mean([c[stage]["rmse_phys_mean_dec"][v] for c in cyc])) for v in VARS}}
        if "ranks" in cyc[0][stage]:
            N = res["config"]["n_members"]
            ranks = np.concatenate([c[stage]["ranks"] for c in cyc])
            out[stage]["rank_hist"] = np.bincount(ranks, minlength=N + 1).tolist()
    return out
