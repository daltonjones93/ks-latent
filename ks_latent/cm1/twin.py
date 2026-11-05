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

Observations: surface theta' (`assim_theta`, every `obs_every_cells`, error `sigma_obs` K) and/or
composite reflectivity (`refl_every_cells` > 0, error `sigma_refl` dBZ) through a learned operator
(`refl.ReflectivityHead`) on the decoded members. Reflectivity observations are CM1's own composite
reflectivity plus Gaussian noise, clipped at 0 dBZ ("no echo" is a hard floor, so the observation error
is not Gaussian near zero), so the operator's error is a genuine representation error.

Doppler radial velocity (E4b, `vr_tilts` non-empty): `obs.RadialVelocity` from a radar at
`radar_x_km`, error `sigma_vr` m/s; observations from the true columns through the same (linear)
operator.

Multi-model ensembles (E3b): `prop` may be a list; member i is forecast by prop[i % len(prop)]. The
propagators must share the autoencoder (trained with `--reuse-ae` on one run), so that their
disagreement is model error in one latent space.
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
    assim_theta: bool = True      # assimilate surface theta'
    refl_every_cells: int = 0     # assimilate composite reflectivity every this many cells (0 = no)
    sigma_refl: float = 5.0       # dBZ
    vr_tilts: tuple = ()          # radar tilts (deg); empty = no radial velocity
    vr_every_cells: int = 8
    sigma_vr: float = 2.0         # m/s
    radar_x_km: float = 330.0
    seed: int = 0


def _forecast(prop: StencilPropagator, Z: torch.Tensor, z_truth: torch.Tensor, t: int, steps: int,
              halo: int, model_noise: float = 0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """Advance the latent ensemble `Z` (N, P, c) from frame t by `steps` frames, truth at the edges,
    optionally with additive model-error noise of std `model_noise` per step."""
    P = Z.shape[1]
    props = prop if isinstance(prop, (list, tuple)) else [prop]
    for j in range(steps):
        if len(props) == 1:
            Z = props[0](Z)
        else:
            Zn = torch.empty_like(Z)
            for k, pk in enumerate(props):
                Zn[k::len(props)] = pk(Z[k::len(props)])
            Z = Zn
        if model_noise > 0:
            Z = Z + model_noise * torch.randn(Z.shape, generator=gen, device="cpu").to(Z.device, Z.dtype)
        if halo > 0:
            zt = z_truth[t + j + 1].expand(Z.shape[0], -1, -1)
            Z = torch.cat([zt[:, :halo], Z[:, halo:P - halo], zt[:, P - halo:]], dim=1)
    return Z


@torch.no_grad()
def _scores(ae: ColumnFieldAE, Z: torch.Tensor, u_true: torch.Tensor, obs_cols: np.ndarray, y: np.ndarray | None,
            norm: Normalizer, Z_levels: int, refl_head=None, refl_cols=None, cref_true=None,
            y_refl=None, vr_op=None, vr_true=None) -> dict:
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
    if refl_head is not None:
        r_mem = refl_head(dec)[:, refl_cols].double().cpu().numpy()          # (N, m_r)
        r_true = cref_true[refl_cols]
        out["refl_obs_rmse"] = float(np.sqrt(np.mean((r_mem.mean(0) - r_true) ** 2)))
        out["refl_obs_spread"] = float(np.sqrt(np.mean(r_mem.var(0, ddof=1))))
        echo = r_true > 10
        out["refl_echo_rmse"] = float(np.sqrt(np.mean((r_mem.mean(0)[echo] - r_true[echo]) ** 2))) if echo.any() else float("nan")
        if y_refl is not None:
            out["ranks_refl"] = (r_mem < y_refl[None, :]).sum(0).tolist()
    if vr_op is not None:
        v_mem = vr_op._from_columns(dec).double().cpu().numpy()
        out["vr_obs_rmse"] = float(np.sqrt(np.mean((v_mem.mean(0) - vr_true) ** 2)))
        out["vr_obs_spread"] = float(np.sqrt(np.mean(v_mem.var(0, ddof=1))))
    return out


@torch.no_grad()
def run_twin(ae: ColumnFieldAE, prop, truth: ColumnData, z_truth: torch.Tensor,
             Z0: torch.Tensor, norm: Normalizer, cfg: TwinConfig, assimilate: bool = True,
             refl_head=None) -> dict:
    """`truth`: ColumnData holding the truth member only (index 0); `z_truth` its encoded latent
    trajectory (T, P, c) (used only for the boundary halo); `Z0` (N, P, c) initial ensemble at frame
    `cfg.t0`. Returns per-cycle prior/analysis scores."""
    ae.eval()
    for pk in (prop if isinstance(prop, (list, tuple)) else [prop]):
        pk.eval()
    if cfg.refl_every_cells > 0 and refl_head is None:
        raise ValueError("refl_every_cells > 0 needs refl_head")
    if not cfg.assim_theta and cfg.refl_every_cells == 0 and not cfg.vr_tilts and assimilate:
        raise ValueError("nothing to assimilate")
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
    refl_cols = np.arange(cfg.refl_every_cells // 2, NX, cfg.refl_every_cells) if cfg.refl_every_cells else obs_cols
    x_refl = (refl_cols + 0.5) * 0.25
    vr_op = None
    if cfg.vr_tilts:
        from ks_latent.cm1.obs import RadialVelocity
        from ks_latent.cm1.physics import BaseState
        zl = BaseState.from_sounding().z
        if zl.size != Zl:
            raise ValueError("radial velocity needs the 62-level CM1 grid")
        vr_op = RadialVelocity(norm, zl, NX, radar_x_km=cfg.radar_x_km, tilts_deg=cfg.vr_tilts,
                               every_cells=cfg.vr_every_cells)
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
        cref_true = truth.cref[0][t1].clamp(min=0).double().numpy()
        y_refl = None
        if cfg.refl_every_cells:
            y_refl = np.clip(cref_true[refl_cols] + rng.normal(0.0, cfg.sigma_refl, refl_cols.size), 0.0, None)
        vr_true = y_vr = None
        if vr_op is not None:
            vr_true = vr_op.truth_values(u_true)
            y_vr = vr_true + rng.normal(0.0, cfg.sigma_vr, vr_true.size)
        sk = dict(refl_head=refl_head, refl_cols=refl_cols, cref_true=cref_true, y_refl=y_refl,
                  vr_op=vr_op, vr_true=vr_true)
        rec = {"cycle": k, "frame": t1, "prior": _scores(ae, Z, u_true, obs_cols, y, norm, Zl, **sk)}
        if assimilate:
            dec = ae.decode(Z)
            Ys, ys, rs, xs = [], [], [], []
            if cfg.assim_theta:
                Ys.append((dec[:, ith, obs_cols] * s_th).double().cpu().numpy()); ys.append(y)
                rs.append(r_var); xs.append(x_obs)
            if cfg.refl_every_cells:
                Ys.append(refl_head(dec)[:, refl_cols].double().cpu().numpy()); ys.append(y_refl)
                rs.append(np.full(refl_cols.size, cfg.sigma_refl ** 2)); xs.append(x_refl)
            if vr_op is not None:
                Ys.append(vr_op._from_columns(dec).double().cpu().numpy()); ys.append(y_vr)
                rs.append(np.full(y_vr.size, cfg.sigma_vr ** 2)); xs.append(vr_op.x_km)
            Za = letkf_analysis(Z.double().cpu().numpy(), np.concatenate(Ys, 1), np.concatenate(ys),
                                np.concatenate(rs), x_site, np.concatenate(xs), cfg.letkf)
            Z = torch.as_tensor(Za, dtype=Z.dtype, device=dev)
            rec["analysis"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl, **sk)
        hist.append(rec)
    return {"config": {"t0": cfg.t0, "n_cycles": cfg.n_cycles, "steps_per_cycle": cfg.steps_per_cycle,
                       "obs_every_cells": cfg.obs_every_cells, "sigma_obs": cfg.sigma_obs,
                       "loc_km": cfg.letkf.loc_km, "inflation": cfg.letkf.inflation, "rtps": cfg.letkf.rtps,
                       "model_noise": cfg.model_noise, "assim_theta": cfg.assim_theta,
                       "refl_every_cells": cfg.refl_every_cells, "sigma_refl": cfg.sigma_refl,
                       "vr_tilts": list(cfg.vr_tilts), "sigma_vr": cfg.sigma_vr, "radar_x_km": cfg.radar_x_km,
                       "n_models": len(prop) if isinstance(prop, (list, tuple)) else 1,
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
        for k in ("refl_obs_rmse", "refl_obs_spread", "refl_echo_rmse", "vr_obs_rmse", "vr_obs_spread"):
            if k in cyc[0][stage]:
                out[stage][k] = float(np.nanmean([c[stage][k] for c in cyc]))
        if "ranks" in cyc[0][stage]:
            N = res["config"]["n_members"]
            ranks = np.concatenate([c[stage]["ranks"] for c in cyc])
            out[stage]["rank_hist"] = np.bincount(ranks, minlength=N + 1).tolist()
    return out
