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

Baselines (G2/G2b), all equation-free and with the same observations:
  - `forecast_model="persistence"`: members do not evolve between analyses.
  - `ensemble_mode="enoi"`: ensemble optimal interpolation. The prior is the previous analysis MEAN
    (forecast with `forecast_model`) plus STATIC anomalies: the encoded training members at the same
    frame about their own mean (`static_ens(t)`). With persistence this uses no forecast model at all.
  - `ensemble_mode="enoi_phys"`: classic EnOI with no machine learning at all. The background mean is
    the previous PHYSICAL analysis mean (persistence); the anomalies are the training archive's TRUE
    columns at the same frame about their mean (`static_phys(t)`, (N, C, NX) normalized); the LETKF runs
    cell by cell. (Latent EnOI with a persistence background is not offered: with a stale background
    the large innovations push latents where the decoder extrapolates badly, a strawman.)
  - `ensemble_mode="oi_clim"`: observation-only analysis (G2c), no model and no memory: every cycle the
    background is the training archive at that frame itself (its mean, the "average storm", with its
    anomalies as the covariance), updated cell by cell with the current observations as in "enoi_phys".
    Its prior is not a forecast, so it is compared on the analysis.
  - `analysis_space="physical"`: the LETKF runs cell by cell on the decoded columns (C channels per
    250 m cell), and the analysis is re-encoded for the next forecast, so the filter leaves the decoder's
    range and is projected back. Condensate is clipped at 0 before re-encoding.
Surface network (`surface_vars`): any of theta', u', p', qv' at the lowest level, same columns, errors
`sigma_surface`. Forecasts from analyses (G3): after each cycle in `forecast_from`, a free forecast of
the analysis ensemble for `forecast_frames` frames is scored against the truth.

Multi-model ensembles (E3b): `prop` may be a list; member i is forecast by prop[i % len(prop)]. The
propagators must share the autoencoder (trained with `--reuse-ae` on one run), so that their
disagreement is model error in one latent space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ks_latent.cm1.data import VARS, ColumnData, Normalizer
from ks_latent.cm1.letkf import LETKFConfig, letkf_analysis_torch
from ks_latent.cm1.models import ColumnFieldAE, StencilPropagator
from ks_latent.cm1.verify import displacement_km, fss, main_pool_front_km

FSS_KM = 10.0   # G3: FSS window for reflectivity >= 35 dBZ


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
    surface_vars: tuple = ("th",)
    sigma_surface: tuple = (("th", 0.5), ("u", 1.0), ("prs", 0.2), ("qv", 0.5))  # K, m/s, hPa, g/kg
    forecast_model: str = "surrogate"      # or "persistence", "analog" (pass an AnalogPropagator as prop), or
                                           # "anchored" (prop = (deterministic, stochastic); M3: the mean follows the
                                           # deterministic model, the perturbations the stochastic one)
    adaptive_noise: str = "none"  # N2: "none", "global" or "site": the additive noise amplitude is rescaled each cycle
    adapt_gamma: float = 0.5      # from the prior innovations (Desroziers-type ratio E[d^2] - R over HPH, obs-error
    adapt_bounds: tuple = (0.2, 5.0)   # normalized); amp *= clip(ratio)^(gamma/2), kept within bounds x model_noise
    ensemble_mode: str = "cycled"          # "cycled", "enoi" (latent static anomalies), "enoi_phys" (no ML), "oi_clim"
    analysis_space: str = "latent"         # or "physical"
    forecast_from: tuple = ()              # cycle indices after which to run a free forecast (G3)
    forecast_frames: int = 60
    boundary: str = "truth"       # halo: the true state every step, or "persistence": held at the truth at t0 (H0)
    exclude_km: tuple = ()        # ((x0, x1), ...) km left out of the nMSE scores (T1: the junction of joined storms)
    col_offset: int = 0           # shifted tilings (E3c): the model domain is the physical one padded by this many
    n_cols_phys: int = 0          # cells on the left; observations sit on the physical domain's columns (0: no padding)
    keep_fields: bool = False     # store each cycle's mean decoded prior/analysis on the physical columns (blending)
    seed: int = 0
    noise_seed: int | None = None   # model-noise generator seed (None: `seed`); observations always use `seed`


def _forecast(prop: StencilPropagator, Z: torch.Tensor, z_truth: torch.Tensor, t: int, steps: int,
              halo: int, model_noise: float = 0.0, gen: torch.Generator | None = None,
              hold: int | None = None) -> torch.Tensor:
    """Advance the latent ensemble `Z` (N, P, c) from frame t by `steps` frames, truth at the edges,
    optionally with additive model-error noise of std `model_noise` per step. A stochastic propagator
    (noise_channels > 0) adds its own learned model error; both draw from `gen`. The `halo` sites at each
    end are set to the truth at every step, or, with `hold` = a frame index, to the truth at that frame
    throughout (no future boundary information)."""
    P = Z.shape[1]
    anchored = isinstance(prop, dict)         # {"det": ..., "stoch": ...}: mean-anchored ensemble (M3)
    props = [None] if anchored else (prop if isinstance(prop, (list, tuple)) else [prop])
    amp = model_noise
    if torch.is_tensor(amp):                  # per-site amplitude (P,) -> (1, P, 1)
        amp = amp.to(Z.device, Z.dtype).reshape(1, -1, 1)
    for j in range(steps):
        if anchored:
            S = prop["stoch"](Z, gen=gen)
            Z = prop["det"](Z.mean(0, keepdim=True)) + (S - S.mean(0, keepdim=True))
        elif props[0] is None:          # persistence
            pass
        elif len(props) == 1:
            Z = props[0](Z, gen=gen)          # a stochastic propagator draws its noise from `gen` (reproducible)
        else:
            Zn = torch.empty_like(Z)
            for k, pk in enumerate(props):
                Zn[k::len(props)] = pk(Z[k::len(props)], gen=gen)
            Z = Zn
        if torch.is_tensor(amp) or model_noise > 0:
            Z = Z + amp * torch.randn(Z.shape, generator=gen, device="cpu").to(Z.device, Z.dtype)
        if halo > 0:
            zt = z_truth[t + j + 1 if hold is None else hold].expand(Z.shape[0], -1, -1)
            Z = torch.cat([zt[:, :halo], Z[:, halo:P - halo], zt[:, P - halo:]], dim=1)
    return Z


def adapt_noise(amp: torch.Tensor, Yb: np.ndarray, y: np.ndarray, r_var: np.ndarray, x_obs_km: np.ndarray,
                x_site_km: np.ndarray, cfg: "TwinConfig") -> torch.Tensor:
    """N2: rescale the additive-noise amplitude per site (or globally) from the prior innovations.

    With d = y - mean(Yb) and s^2 the prior ensemble variance in observation space, a calibrated ensemble has
    E[d^2] = s^2 + R. In units of R: ratio = sum w (d^2/R - 1) / sum w (s^2/R), the variance the prior should
    have over the variance it has (Desroziers et al. 2005; Anderson 2009 in spirit). w are the LETKF's Gaspari-Cohn
    weights for each site ("site") or all ones ("global"). The amplitude moves by clip(ratio, 1/4, 4)^(gamma/2)
    per cycle, kept in [lo, hi] x the base noise. Sites without observations keep their amplitude."""
    from ks_latent.cm1.letkf import gaspari_cohn
    d2 = (y - Yb.mean(0)) ** 2 / r_var - 1.0
    s2 = Yb.var(0, ddof=1) / r_var
    if cfg.adaptive_noise == "global":
        w = np.ones((1, y.size))
    else:
        w = gaspari_cohn((x_site_km[:, None] - x_obs_km[None, :]) / cfg.letkf.loc_km)      # (P, m)
    num, den = w @ d2, w @ s2
    ratio = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
    fac = torch.as_tensor(np.clip(ratio, 0.25, 4.0) ** (cfg.adapt_gamma / 2), dtype=amp.dtype)
    lo, hi = cfg.adapt_bounds
    return (amp * fac).clamp(lo * cfg.model_noise, hi * cfg.model_noise)


def _keep(cfg, NX: int, dev) -> torch.Tensor:
    """(NX,) bool: columns outside every `cfg.exclude_km` interval (cell centres, 250 m cells)."""
    x = (torch.arange(NX, device=dev) + 0.5) * 0.25
    k = torch.ones(NX, dtype=torch.bool, device=dev)
    for x0, x1 in cfg.exclude_km:
        k &= ~((x >= x0) & (x < x1))
    if not k.any():
        raise ValueError(f"exclude_km {cfg.exclude_km} leaves no columns to score")
    return k


def _stash(rec: dict, stage: str, cfg, o: int, NXp: int) -> None:
    """Move the mean decoded field out of the scores; keep it (physical columns, CPU float16) if asked."""
    f = rec[stage].pop("_mean_dec")
    if cfg.keep_fields:
        rec[stage + "_field"] = f[:, o:o + NXp].to("cpu", torch.float16)


def _hold(cfg) -> int | None:
    if cfg.boundary == "truth":
        return None
    if cfg.boundary == "persistence":
        return cfg.t0
    raise ValueError(f"unknown boundary {cfg.boundary!r}")


@torch.no_grad()
def _scores(ae: ColumnFieldAE, Z: torch.Tensor, u_true: torch.Tensor, obs_cols: np.ndarray, y: np.ndarray | None,
            norm: Normalizer, Z_levels: int, refl_head=None, refl_cols=None, cref_true=None,
            y_refl=None, vr_op=None, vr_true=None, dec: torch.Tensor | None = None,
            keep: torch.Tensor | None = None, obs_sigma: float = 0.0, rank_rng=None) -> dict:
    """Scores of ensemble `Z` (or of decoded members `dec`, for physical-space analyses) against the
    true normalized columns `u_true` (C, NX). `keep` (NX,) bool: the columns the nMSE scores use."""
    if dec is None:
        dec = ae.decode(Z)                                   # (N, C, NX)
        dec_mean = ae.decode(Z.mean(0, keepdim=True))[0]
    else:
        dec_mean = dec.mean(0)
    mean_dec = dec.mean(0)
    V = len(norm.vars)
    sc = torch.tensor(norm.scale, dtype=torch.float32, device=dec.device)
    def per_var(est):
        e = ((est - u_true) ** 2).reshape(V, Z_levels, -1).mean(dim=(1, 2)).sqrt() * sc
        return dict(zip(norm.vars, e.tolist()))
    ith = VARS.index("th") * Z_levels
    s_th = float(norm.scale[VARS.index("th")])
    th_true = (u_true[ith] * s_th).double().cpu().numpy()
    th_mean = (mean_dec[ith] * s_th).double().cpu().numpy()
    th_mem = (dec[:, ith, obs_cols] * s_th).double().cpu().numpy()       # (N, m)
    front = main_pool_front_km(th_true)
    kc = slice(None) if keep is None else keep
    out = {"_mean_dec": mean_dec, "nmse_mean_dec": float(((mean_dec - u_true) ** 2)[:, kc].mean()),
           "nmse_dec_mean": float(((dec_mean - u_true) ** 2)[:, kc].mean()),
           "rmse_phys_mean_dec": per_var(mean_dec),
           "th_obs_rmse": float(np.sqrt(np.mean((th_mem.mean(0) - th_true[obs_cols]) ** 2))),
           "th_obs_spread": float(np.sqrt(np.mean(th_mem.var(0, ddof=1)))),
           "front_true_km": float(front), "front_disp_km": displacement_km(th_mean, th_true, center_km=front)}
    if y is not None:
        out["ranks"] = (th_mem < y[None, :]).sum(0).tolist()   # rank of each observation among members
        if rank_rng is not None and obs_sigma > 0:
            # Hamill (2001): with noisy observations the members must carry the observation error too, or the
            # histogram is U-shaped even for a perfect ensemble. Own generator: the observation draws are unchanged.
            th_p = th_mem + rank_rng.normal(0.0, obs_sigma, th_mem.shape)
            out["ranks_pert"] = (th_p < y[None, :]).sum(0).tolist()
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
def _forecast_scores(ae, prop, Z, z_truth, truth, t, cfg, norm, refl_head, gen) -> list:
    """Free forecast of ensemble Z from frame t for cfg.forecast_frames frames; per-lead scores (G3), with the
    same scores for persistence of the starting ensemble (the analysis held fixed: the floor any forecast from
    it must beat). Reflectivity >= 35 dBZ is scored with the FSS at FSS_KM of the members' mean reflectivity."""
    out, Zf = [], Z.clone()
    Zl = truth.Z
    ith = VARS.index("th") * Zl
    s_th = float(norm.scale[VARS.index("th")])
    dev = Z.device
    dec0 = ae.decode(Z)
    md0 = dec0.mean(0)
    th0 = (md0[ith] * s_th).double().cpu().numpy()
    r0 = refl_head(dec0).mean(0).double().cpu().numpy() if refl_head is not None else None
    n_fss = max(1, int(round(FSS_KM / 0.25)))
    for j in range(cfg.forecast_frames):
        if t + j + 1 >= truth.T[0]:
            break
        Zf = _forecast(prop, Zf, z_truth, t + j, 1, cfg.halo, cfg.model_noise, gen, hold=_hold(cfg))
        u_true = truth.full(0, t + j + 1, device=dev)[0]
        dec = ae.decode(Zf)
        md = dec.mean(0)
        th_t = (u_true[ith] * s_th).double().cpu().numpy()
        th_m = (md[ith] * s_th).double().cpu().numpy()
        f = main_pool_front_km(th_t)
        kp = _keep(cfg, truth.NX, dev)
        rec = {"lead_frames": j + 1, "nmse_mean_dec": float(((md - u_true) ** 2)[:, kp].mean()),
               "front_disp_km": displacement_km(th_m, th_t, center_km=f),
               "nmse_persist": float(((md0 - u_true) ** 2)[:, kp].mean()),
               "front_disp_km_persist": displacement_km(th0, th_t, center_km=f)}
        if refl_head is not None:
            cref_t = truth.cref[0][t + j + 1].clamp(min=0).double().numpy()
            r = refl_head(dec).mean(0).double().cpu().numpy()
            echo = cref_t > 10
            rec["refl_echo_rmse"] = float(np.sqrt(np.mean((r[echo] - cref_t[echo]) ** 2))) if echo.any() else float("nan")
            km = kp.cpu().numpy()
            rec["fss_refl35"] = fss(r[km], cref_t[km], 35.0, [n_fss])[n_fss]
            rec["fss_refl35_persist"] = fss(r0[km], cref_t[km], 35.0, [n_fss])[n_fss]
        out.append(rec)
    return out


@torch.no_grad()
def run_twin(ae: ColumnFieldAE, prop, truth: ColumnData, z_truth: torch.Tensor,
             Z0: torch.Tensor, norm: Normalizer, cfg: TwinConfig, assimilate: bool = True,
             refl_head=None, static_ens=None, static_phys=None, env_ens=None) -> dict:
    """`truth`: ColumnData holding the truth member only (index 0); `z_truth` its encoded latent
    trajectory (T, P, c) (used only for the boundary halo); `Z0` (N, P, c) initial ensemble at frame
    `cfg.t0`. Returns per-cycle prior/analysis scores."""
    ae.eval()
    if cfg.forecast_model == "persistence":
        prop = None
    elif cfg.forecast_model == "anchored":
        if not (isinstance(prop, (list, tuple)) and len(prop) == 2):
            raise ValueError("forecast_model='anchored' needs prop = (deterministic, stochastic) sharing one latent")
        prop = {"det": prop[0], "stoch": prop[1]}
    elif cfg.forecast_model not in ("surrogate", "analog"):    # analog: `prop` is an AnalogPropagator (G2b)
        raise ValueError(f"unknown forecast_model {cfg.forecast_model!r}")
    if cfg.adaptive_noise not in ("none", "global", "site"):
        raise ValueError(f"unknown adaptive_noise {cfg.adaptive_noise!r}")
    if cfg.adaptive_noise != "none" and cfg.model_noise <= 0:
        raise ValueError("adaptive_noise rescales model_noise, which must be > 0")
    for pk in (list(prop.values()) if isinstance(prop, dict) else prop if isinstance(prop, (list, tuple)) else [prop]):
        if pk is not None:
            pk.eval()
            if pk.cfg.n_hist > 1:
                raise ValueError("the twin steps frame by frame and cannot feed a history propagator (n_hist > 1) yet")
    if cfg.ensemble_mode in ("enoi_phys", "oi_clim") and static_phys is None:
        raise ValueError(f"ensemble_mode={cfg.ensemble_mode!r} needs static_phys(frame) -> (N, C, NX)")
    if cfg.ensemble_mode not in ("cycled", "enoi", "enoi_phys", "oi_clim"):
        raise ValueError(f"unknown ensemble_mode {cfg.ensemble_mode!r}")
    if cfg.ensemble_mode == "enoi" and static_ens is None:
        raise ValueError("ensemble_mode='enoi' needs static_ens(frame) -> (N, P, c)")
    if cfg.analysis_space not in ("latent", "physical"):
        raise ValueError(f"unknown analysis_space {cfg.analysis_space!r}")
    if cfg.refl_every_cells > 0 and refl_head is None:
        raise ValueError("refl_every_cells > 0 needs refl_head")
    if not cfg.assim_theta and not [v for v in cfg.surface_vars if v != "th"] and cfg.refl_every_cells == 0 \
            and not cfg.vr_tilts and assimilate:
        raise ValueError("nothing to assimilate")
    dev = Z0.device
    rng = np.random.default_rng(cfg.seed)
    NX, Zl = truth.NX, truth.Z
    o, NXp = cfg.col_offset, (cfg.n_cols_phys or NX)
    if o < 0 or o + NXp > NX:
        raise ValueError(f"physical columns [{o}, {o + NXp}) outside the model domain of {NX}")
    obs_cols = np.arange(cfg.obs_every_cells // 2, NXp, cfg.obs_every_cells) + o
    ith = VARS.index("th") * Zl
    s_th = float(norm.scale[VARS.index("th")])
    x_obs = (obs_cols + 0.5) * 0.25
    P = Z0.shape[1]
    x_site = (np.arange(P) + 0.5) * ae.cfg.patch * 0.25
    r_var = np.full(obs_cols.size, cfg.sigma_obs ** 2)
    sig_sfc = dict(cfg.sigma_surface)
    sfc_extra = [v for v in cfg.surface_vars if v != "th"]
    if not set(cfg.surface_vars) <= {"th", "u", "prs", "qv"}:
        raise ValueError(f"surface_vars must be from th, u, prs, qv: {cfg.surface_vars}")
    refl_cols = (np.arange(cfg.refl_every_cells // 2, NXp, cfg.refl_every_cells) + o) if cfg.refl_every_cells else obs_cols
    x_refl = (refl_cols + 0.5) * 0.25
    vr_op = None
    if cfg.vr_tilts:
        from ks_latent.cm1.obs import RadialVelocity
        from ks_latent.cm1.physics import BaseState
        zl = BaseState.from_sounding().z
        if zl.size != Zl:
            raise ValueError("radial velocity needs the 62-level CM1 grid")
        vr_op = RadialVelocity(norm, zl, NXp, radar_x_km=cfg.radar_x_km, tilts_deg=cfg.vr_tilts,
                               every_cells=cfg.vr_every_cells, col_offset=o)
    T_last = cfg.t0 + cfg.n_cycles * cfg.steps_per_cycle
    if T_last >= truth.T[0]:
        raise ValueError(f"truth has {truth.T[0]} frames; the experiment needs {T_last + 1}")
    keep = _keep(cfg, NX, dev)
    if o or NXp != NX:   # padded tilings: score the physical columns only
        keep[:o] = False
        keep[o + NXp:] = False
    Z = Z0.clone()
    U_mean = ae.decode(Z0).mean(0) if cfg.ensemble_mode == "enoi_phys" else None   # physical background mean
    gen = torch.Generator().manual_seed(cfg.seed if cfg.noise_seed is None else cfg.noise_seed)
    rank_rng = np.random.default_rng(10_000 + cfg.seed)     # perturbed-member ranks only (claim 2)
    amp = torch.full((P,), cfg.model_noise, dtype=Z.dtype) if cfg.adaptive_noise != "none" else cfg.model_noise
    hist = []
    for k in range(cfg.n_cycles):
        t = cfg.t0 + k * cfg.steps_per_cycle
        t1 = t + cfg.steps_per_cycle
        dec_prior = None
        if cfg.ensemble_mode == "enoi":
            mean = _forecast(prop, Z.mean(0, keepdim=True), z_truth, t, cfg.steps_per_cycle, cfg.halo, 0.0, gen,
                             hold=_hold(cfg))
            S = static_ens(t1).to(dev, Z.dtype)
            Z = mean + (S - S.mean(0, keepdim=True))
        elif cfg.ensemble_mode == "enoi_phys":
            A = static_phys(t1).to(dev, U_mean.dtype)
            dec_prior = U_mean[None] + (A - A.mean(0, keepdim=True))
        elif cfg.ensemble_mode == "oi_clim":      # the archive itself: average storm + its anomalies
            dec_prior = static_phys(t1).to(dev)
        else:
            Z = _forecast(prop, Z, z_truth, t, cfg.steps_per_cycle, cfg.halo, amp, gen, hold=_hold(cfg))
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
                  vr_op=vr_op, vr_true=vr_true, keep=keep, obs_sigma=cfg.sigma_obs, rank_rng=rank_rng)
        rec = {"cycle": k, "frame": t1, "prior": _scores(ae, Z, u_true, obs_cols, y, norm, Zl, dec=dec_prior, **sk)}
        _stash(rec, "prior", cfg, o, NXp)
        if assimilate:
            dec = ae.decode(Z) if dec_prior is None else dec_prior
            Ys, ys, rs, xs = [], [], [], []
            if cfg.assim_theta and "th" in cfg.surface_vars:
                Ys.append((dec[:, ith, obs_cols] * s_th).double().cpu().numpy()); ys.append(y)
                rs.append(r_var); xs.append(x_obs)
            for v in sfc_extra:
                iv, sv = VARS.index(v) * Zl, float(norm.scale[VARS.index(v)])
                yv = (u_true[iv, obs_cols] * sv).double().cpu().numpy() + rng.normal(0.0, sig_sfc[v], obs_cols.size)
                Ys.append((dec[:, iv, obs_cols] * sv).double().cpu().numpy()); ys.append(yv)
                rs.append(np.full(obs_cols.size, sig_sfc[v] ** 2)); xs.append(x_obs)
            if cfg.refl_every_cells:
                Ys.append(refl_head(dec)[:, refl_cols].double().cpu().numpy()); ys.append(y_refl)
                rs.append(np.full(refl_cols.size, cfg.sigma_refl ** 2)); xs.append(x_refl)
            if vr_op is not None:
                Ys.append(vr_op._from_columns(dec).double().cpu().numpy()); ys.append(y_vr)
                rs.append(np.full(y_vr.size, cfg.sigma_vr ** 2)); xs.append(vr_op.x_km)
            Yall, yall, rall, xall = np.concatenate(Ys, 1), np.concatenate(ys), np.concatenate(rs), np.concatenate(xs)
            if cfg.adaptive_noise != "none" and cfg.ensemble_mode == "cycled":
                amp = adapt_noise(amp, Yall, yall, rall, xall, x_site, cfg)
                rec["noise_amp"] = float(amp.mean())
            if cfg.ensemble_mode in ("enoi_phys", "oi_clim"):   # physical background and analysis, no latent
                x_cell = (np.arange(NX) + 0.5) * 0.25
                Ua = letkf_analysis_torch(dec.permute(0, 2, 1).double().cpu().numpy(), Yall, yall, rall, x_cell, xall, cfg.letkf, device=dev)
                Ua = torch.as_tensor(Ua, dtype=dec.dtype, device=dev).permute(0, 2, 1).contiguous()
                Ua = torch.where(ae.positive_mask[None, :, None], Ua.clamp(min=0), Ua)
                rec["analysis"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl, dec=Ua, **sk)
                _stash(rec, "analysis", cfg, o, NXp)
                U_mean = Ua.mean(0)
            elif cfg.analysis_space == "latent":
                Za = letkf_analysis_torch(Z.double().cpu().numpy(), Yall, yall, rall, x_site, xall, cfg.letkf, device=dev)
                Z = torch.as_tensor(Za, dtype=Z.dtype, device=dev)
                rec["analysis"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl, **sk)
                _stash(rec, "analysis", cfg, o, NXp)
            else:  # per-cell LETKF on the decoded columns, then back into the latent space
                x_cell = (np.arange(NX) + 0.5) * 0.25
                Ua = letkf_analysis_torch(dec.permute(0, 2, 1).double().cpu().numpy(), Yall, yall, rall, x_cell, xall, cfg.letkf, device=dev)
                Ua = torch.as_tensor(Ua, dtype=dec.dtype, device=dev).permute(0, 2, 1).contiguous()
                pm = ae.positive_mask[None, :, None]
                Ua = torch.where(pm, Ua.clamp(min=0), Ua)
                rec["analysis"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl, dec=Ua, **sk)
                _stash(rec, "analysis", cfg, o, NXp)
                Z = ae.encode(Ua, env_ens)
                rec["analysis_reencoded"] = _scores(ae, Z, u_true, obs_cols, y, norm, Zl, **sk)
                rec["analysis_reencoded"].pop("_mean_dec")
        if k in cfg.forecast_from:
            rec["forecast"] = _forecast_scores(ae, prop, Z, z_truth, truth, t1, cfg, norm, refl_head, gen)
        hist.append(rec)
    return {"config": {"t0": cfg.t0, "n_cycles": cfg.n_cycles, "steps_per_cycle": cfg.steps_per_cycle,
                       "obs_every_cells": cfg.obs_every_cells, "sigma_obs": cfg.sigma_obs,
                       "loc_km": cfg.letkf.loc_km, "inflation": cfg.letkf.inflation, "rtps": cfg.letkf.rtps,
                       "model_noise": cfg.model_noise, "assim_theta": cfg.assim_theta,
                       "refl_every_cells": cfg.refl_every_cells, "sigma_refl": cfg.sigma_refl,
                       "vr_tilts": list(cfg.vr_tilts), "sigma_vr": cfg.sigma_vr, "radar_x_km": cfg.radar_x_km,
                       "n_models": len(prop) if isinstance(prop, (list, tuple)) else 1,
                       "surface_vars": list(cfg.surface_vars), "forecast_model": cfg.forecast_model,
                       "ensemble_mode": cfg.ensemble_mode, "analysis_space": cfg.analysis_space,
                       "boundary": cfg.boundary, "exclude_km": [list(e) for e in cfg.exclude_km],
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
                      "rmse_phys": {v: float(np.mean([c[stage]["rmse_phys_mean_dec"][v] for c in cyc]))
                                    for v in cyc[0][stage]["rmse_phys_mean_dec"]}}
        for k in ("refl_obs_rmse", "refl_obs_spread", "refl_echo_rmse", "vr_obs_rmse", "vr_obs_spread"):
            if k in cyc[0][stage]:
                out[stage][k] = float(np.nanmean([c[stage][k] for c in cyc]))
        N = res["config"]["n_members"]
        for key, name in (("ranks", "rank_hist"), ("ranks_pert", "rank_hist_pert")):
            if key in cyc[0][stage]:
                out[stage][name] = np.bincount(np.concatenate([c[stage][key] for c in cyc]), minlength=N + 1).tolist()
                out[stage][key + "_cycles"] = [list(map(int, c[stage][key])) for c in cyc]   # for decorrelated tests
    return out
