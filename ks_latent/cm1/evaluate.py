"""Evaluation of a trained CM1 column local field on held-out members.

Reconstruction: per-variable RMSE in physical units (sqrt(g/kg) for the
condensate channels), over all columns and over storm columns (composite
reflectivity > 10 dBZ).

Forecasts: from the encoded truth at start frames `t0s`, the propagator
runs over the whole domain for `n_lead` frames, with `halo` sites at each
domain edge reset to the encoded truth every step (lateral boundary
conditions, as a limited-area model gets). Decoded forecasts are scored per
lead time against
  - persistence (the truth at t0),
  - the truth-ensemble spread: RMS difference between this member and the
    other evaluated members at the same time, / sqrt(2). This is roughly the
    error of a perfect model started from another member's initial state,
    i.e. the predictability floor for the original perturbation size.
Errors are normalized MSE over all channels (unit-variance variables), plus
surface theta' RMSE (K) and front metrics from `verify.py`:
  - main-pool front error (km): leading edge of the main cold pool, median and 90th percentile
    over cases (heavy-tailed: squall lines advance in discrete jumps);
  - displacement (km): threshold-free signed shift of the forecast's surface theta' against the
    truth around the true front (bias, and median / 90th percentile of its magnitude);
  - Fractions Skill Score of column-max condensate (sqrt(g/kg)) at thresholds `FSS_THRESH` and
    window sizes `FSS_SCALES_KM`.
The older easternmost-cold-cell front error is kept as `front_err_km` for comparison.
Physical consistency (`physics.py`, real 62-level CM1 data only): mass-divergence ratio of decoded or
forecast states against the truth's own (~0.2, from interpolated winds and compressibility),
relative condensate-mass error, maximum cold-pool intensity C, and the FSS of low-level liquid
condensate (z < 1 km, a rain proxy that discriminates better than column-max condensate).
"""
from __future__ import annotations

import numpy as np
import torch

from ks_latent.cm1.data import POSITIVE_VARS, VARS, ColumnData, Normalizer
from ks_latent.cm1.train import crps_elements
from ks_latent.cm1.models import ColumnFieldAE, StencilPropagator, rollout_from
from ks_latent.cm1.physics import (BaseState, cold_pool_intensity, condensate_mass, divergence_ratio,
                                   low_level_liquid)
from ks_latent.cm1.verify import displacement_km, fss, main_pool_front_km

FSS_THRESH = (0.3, 1.0)          # sqrt(g/kg): ~0.1 g/kg (cloud or rain present), 1 g/kg (heavy)
FSS_SCALES_KM = (0.25, 1.0, 4.0, 16.0)


def _th_sfc(u: torch.Tensor, Z: int, norm: Normalizer) -> torch.Tensor:
    """Surface theta' (K) from normalized columns `(..., C, NX)`."""
    return u[..., VARS.index("th") * Z, :] * float(norm.scale[VARS.index("th")])


def front_x_km(th_sfc: torch.Tensor, dx_km: float = 0.25, thresh: float = -1.0) -> torch.Tensor:
    """Easternmost cell with theta' < thresh, per row of `(..., NX)`; NaN if none."""
    cold = th_sfc < thresh
    NX = th_sfc.shape[-1]
    idx = torch.where(cold, torch.arange(NX, device=th_sfc.device), torch.full_like(cold, -1, dtype=torch.long))
    last = idx.max(dim=-1).values.float()
    return torch.where(last >= 0, (last + 0.5) * dx_km, torch.full_like(last, float("nan")))


@torch.no_grad()
def reconstruction_report(ae: ColumnFieldAE, data: ColumnData, norm: Normalizer, stride_t: int = 2) -> dict:
    ae.eval()
    dev = next(ae.parameters()).device
    V, Z = data.V, data.Z
    se = torch.zeros(V); se_act = torch.zeros(V); n = 0; n_act = 0
    for m in range(len(data.u)):
        for t in range(0, data.T[m], stride_t):
            u = data.full(m, t, device=dev)               # (1, C, NX)
            env = data.env_of([m])
            env = None if env is None else env.to(dev)
            e = (ae(u, env)[0] - u)[0].reshape(V, Z, -1).cpu() ** 2  # normalized
            act = data.cref[m][t] > 10
            se += e.mean(dim=1).sum(dim=1); n += e.shape[-1]
            se_act += e.mean(dim=1)[:, act].sum(dim=1); n_act += int(act.sum())
    sc = torch.tensor(norm.scale, dtype=torch.float32)
    if Z == 62:
        base = BaseState.from_sounding()
        chs = torch.tensor(norm.channel_scale(Z), dtype=torch.float32)
        dt_, dr_, mt_, mr_ = [], [], [], []
        for m in range(len(data.u)):
            for t in range(data.T[m] // 3, data.T[m], max(1, data.T[m] // 6)):
                u = data.full(m, t, device=dev)
                env = data.env_of([m])
                tr = (u[0].cpu() * chs[:, None]).reshape(V, Z, -1).numpy()
                rc = (ae(u, None if env is None else env.to(dev))[0][0].cpu() * chs[:, None]).reshape(V, Z, -1).numpy()
                dt_.append(divergence_ratio(tr, base)); dr_.append(divergence_ratio(rc, base))
                mt_.append(float(condensate_mass(tr, base))); mr_.append(float(condensate_mass(rc, base)))
        phys = {"div_ratio_truth": float(np.mean(dt_)), "div_ratio_recon": float(np.mean(dr_)),
                "condensate_mass_rel_err": float(np.median(np.abs(np.array(mr_) / np.array(mt_) - 1)))}
    else:
        phys = {}
    out = {"physics": phys, "rmse_phys_all": dict(zip(norm.vars, (sc * (se / n).sqrt()).tolist())),
           "nmse_all": float((se / n).mean())}
    if n_act:
        out["rmse_phys_storm"] = dict(zip(norm.vars, (sc * (se_act / n_act).sqrt()).tolist()))
        out["nmse_storm"] = float((se_act / n_act).mean())
    return out


@torch.no_grad()
def forecast_report(ae: ColumnFieldAE, prop, data: ColumnData, z: list[torch.Tensor],
                    norm: Normalizer, t0s: list[int], n_lead: int, halo: int, n_samples: int = 1,
                    boundary: str = "truth") -> dict:
    """`prop` may be one propagator or a list sharing the autoencoder (D14a: their decoded forecasts are
    averaged); a stochastic propagator is run `n_samples` times and its decoded samples averaged."""
    props = prop if isinstance(prop, (list, tuple)) else [prop]
    ae.eval()
    for p_ in props:
        p_.eval()
    dev = next(ae.parameters()).device
    Z = data.Z
    M = len(data.u)
    cases = []  # numpy (surface theta', column-max condensate) of forecast, persistence and truth, per case
    base = BaseState.from_sounding() if Z == 62 else None
    chs = torch.tensor(norm.channel_scale(Z), dtype=torch.float32, device=dev)
    phys_cases = []
    def _phys(u):  # (L, C, NX) normalized -> per-lead physics (small)
        X = (u * chs[None, :, None]).reshape(u.shape[0], len(norm.vars), Z, -1).cpu().numpy()
        return {"div": np.array([divergence_ratio(x, base) for x in X]), "mass": condensate_mass(X, base),
                "cmax": cold_pool_intensity(X, base).max(axis=-1), "ll": low_level_liquid(X, base)}
    cond_ix = [(norm.vars.index(v) * Z, float(norm.scale[norm.vars.index(v)])) for v in POSITIVE_VARS if v in norm.vars]
    def _np_fields(u):
        th = _th_sfc(u, Z, norm).cpu().numpy()
        q2 = sum((u[..., i:i + Z, :] * s).clamp(min=0) ** 2 for i, s in cond_ix)   # total condensate, g/kg
        cond = torch.sqrt(q2).amax(dim=-2).cpu().numpy()
        return th, cond
    rows = {k: [] for k in ("nmse", "nmse_pers", "nmse_spread", "th_sfc_rmse", "th_sfc_rmse_pers", "front_err_km",
                            "front_err_pers_km", "crps", "crps_pers", "spread_err")}
    for t0 in t0s:
        if any(t0 + n_lead >= T for T in data.T):
            continue
        truth = torch.stack([data.full(m, slice(t0, t0 + n_lead + 1), device=dev) for m in range(M)])  # (M, L+1, C, NX)
        for m in range(M):
            pred = 0
            tr = truth[m, 1:]
            pers = truth[m, :1].expand_as(tr)
            ens_all = []
            for p_ in props:
                ns = n_samples if p_.cfg.noise_channels else 1
                lat = rollout_from(p_, z[m], t0, n_lead, halo, ns, gen=torch.Generator().manual_seed(m),
                                   boundary=boundary)
                dec = torch.stack([ae.decode(lat[j]) for j in range(n_lead)])                # (L, ns, C, NX)
                pred = pred + dec.mean(1) / len(props)
                ens_all.append(dec)
            # pred: (L, C, NX). CRPS of the forecast ensemble (all samples of all propagators; for one
            # deterministic forecast the CRPS is the MAE), and spread / error of its mean (E >= 2 only).
            ens = torch.cat(ens_all, dim=1)                                                  # (L, E, C, NX)
            if ens.shape[1] > 1:
                rows["crps"].append(torch.stack([crps_elements(ens[j], tr[j]).mean() for j in range(n_lead)]).cpu())
                var = ens.var(dim=1).mean(dim=(1, 2))
                rows["spread_err"].append((var / ((pred - tr) ** 2).mean(dim=(1, 2))).sqrt().cpu())
            else:
                rows["crps"].append((ens[:, 0] - tr).abs().mean(dim=(1, 2)).cpu())
            rows["crps_pers"].append((pers - tr).abs().mean(dim=(1, 2)).cpu())
            rows["nmse"].append(((pred - tr) ** 2).mean(dim=(1, 2)).cpu())
            rows["nmse_pers"].append(((pers - tr) ** 2).mean(dim=(1, 2)).cpu())
            others = [o for o in range(M) if o != m]
            if others:
                rows["nmse_spread"].append(torch.stack([((truth[o, 1:] - tr) ** 2).mean(dim=(1, 2))
                                                        for o in others]).mean(0).cpu() / 2)
            ths_p, ths_t, ths_0 = _th_sfc(pred, Z, norm), _th_sfc(tr, Z, norm), _th_sfc(pers, Z, norm)
            rows["th_sfc_rmse"].append(((ths_p - ths_t) ** 2).mean(-1).sqrt().cpu())
            rows["th_sfc_rmse_pers"].append(((ths_0 - ths_t) ** 2).mean(-1).sqrt().cpu())
            ft = front_x_km(ths_t)
            rows["front_err_km"].append((front_x_km(ths_p) - ft).abs().cpu())
            rows["front_err_pers_km"].append((front_x_km(ths_0) - ft).abs().cpu())
            cases.append({"fc": _np_fields(pred), "pers": _np_fields(pers), "ob": _np_fields(tr)})
            if base is not None:
                phys_cases.append({"fc": _phys(pred), "pers": _phys(pers), "ob": _phys(tr)})
    out = {"t0s": t0s, "n_lead": n_lead, "boundary": boundary, "lead_s": [(j + 1) * data.dt for j in range(n_lead)], "halo_sites": halo}
    out.update(_front_and_fss(cases, n_lead))
    out.update(_physics_stats(phys_cases))
    for k, v in rows.items():
        if v:
            out[k] = torch.stack(v).nanmean(0).tolist()
    return out


def summarize(fc: dict) -> str:
    lines = ["lead_min  nmse   pers   spread  thsfc_K pers_K  front_km pers_km"]
    for j, ls in enumerate(fc["lead_s"]):
        g = lambda k: fc[k][j] if k in fc else float("nan")
        lines.append(f"{ls / 60:7.1f} {g('nmse'):6.3f} {g('nmse_pers'):6.3f} {g('nmse_spread'):7.3f} "
                     f"{g('th_sfc_rmse'):7.2f} {g('th_sfc_rmse_pers'):6.2f} {g('front_err_km'):8.2f} "
                     f"{g('front_err_pers_km'):7.2f}")
    return "\n".join(lines)


def _front_and_fss(cases: list, n_lead: int) -> dict:
    """Front, displacement and FSS statistics per lead over all cases (see module docstring)."""
    if not cases:
        return {}
    dx = 0.25
    scales = [max(1, int(round(s / dx))) for s in FSS_SCALES_KM]
    fe = {"fc": np.full((len(cases), n_lead), np.nan), "pers": np.full((len(cases), n_lead), np.nan)}
    dp = {"fc": np.full((len(cases), n_lead), np.nan), "pers": np.full((len(cases), n_lead), np.nan)}
    fs = {k: {th: {n: np.full((len(cases), n_lead), np.nan) for n in scales} for th in FSS_THRESH}
          for k in ("fc", "pers")}
    for c, case in enumerate(cases):
        th_ob, cond_ob = case["ob"]
        f_ob = main_pool_front_km(th_ob)
        for k in ("fc", "pers"):
            th_k, cond_k = case[k]
            fe[k][c] = np.abs(main_pool_front_km(th_k) - f_ob)
            for j in range(n_lead):
                dp[k][c, j] = displacement_km(th_k[j], th_ob[j], center_km=f_ob[j])
                for thr in FSS_THRESH:
                    for n, v in fss(cond_k[j], cond_ob[j], thr, scales).items():
                        fs[k][thr][n][c, j] = v
    med = lambda a: np.nanmedian(a, axis=0).tolist()
    p90 = lambda a: np.nanpercentile(a, 90, axis=0).tolist()
    out = {"n_cases": len(cases),
           "front_main_err_km_median": med(fe["fc"]), "front_main_err_km_p90": p90(fe["fc"]),
           "front_main_err_pers_km_median": med(fe["pers"]), "front_main_err_pers_km_p90": p90(fe["pers"]),
           "disp_km_bias": np.nanmean(dp["fc"], axis=0).tolist(),
           "disp_abs_km_median": med(np.abs(dp["fc"])), "disp_abs_km_p90": p90(np.abs(dp["fc"])),
           "disp_pers_abs_km_median": med(np.abs(dp["pers"])), "disp_pers_abs_km_p90": p90(np.abs(dp["pers"]))}
    for k, name in (("fc", "fss"), ("pers", "fss_pers")):
        out[name] = {f"{thr:g}": {f"{n * dx:g}km": np.nanmean(fs[k][thr][n], axis=0).tolist() for n in scales}
                     for thr in FSS_THRESH}
    return out


def summarize_fronts(fc: dict, leads_min=(1, 5, 15, 30, 45, 60)) -> str:
    """Front, displacement and FSS lines for the leads in `leads_min` that exist."""
    if "disp_abs_km_median" not in fc:
        return ""
    lead = [round(s / 60) for s in fc["lead_s"]]
    js = [lead.index(m) for m in leads_min if m in lead]
    fk = "4km" if "4km" in fc["fss"]["0.3"] else sorted(fc["fss"]["0.3"])[0]
    lines = [f"lead_min  |disp| med/p90 (pers)    bias   main-front med/p90 (pers)   FSS q>0.3 @{fk} (pers)"]
    for j in js:
        lines.append(f"{lead[j]:7d}   {fc['disp_abs_km_median'][j]:5.2f}/{fc['disp_abs_km_p90'][j]:5.2f} "
                     f"({fc['disp_pers_abs_km_median'][j]:5.2f})  {fc['disp_km_bias'][j]:+6.2f}   "
                     f"{fc['front_main_err_km_median'][j]:5.2f}/{fc['front_main_err_km_p90'][j]:5.2f} "
                     f"({fc['front_main_err_pers_km_median'][j]:5.2f})       "
                     f"{fc['fss']['0.3'][fk][j]:.3f} ({fc['fss_pers']['0.3'][fk][j]:.3f})")
    return "\n".join(lines)


def _physics_stats(phys_cases: list) -> dict:
    """Per-lead physical consistency of forecasts against the truth (see module docstring)."""
    if not phys_cases:
        return {}
    st = lambda k, who: np.stack([c[who][k] for c in phys_cases])
    rel = lambda a, b: np.abs(a / b - 1)
    dx = 0.25
    out = {"div_ratio_fc": st("div", "fc").mean(0).tolist(), "div_ratio_truth": st("div", "ob").mean(0).tolist(),
           "condensate_mass_rel_err_median": np.median(rel(st("mass", "fc"), st("mass", "ob")), 0).tolist(),
           "condensate_mass_rel_err_pers_median": np.median(rel(st("mass", "pers"), st("mass", "ob")), 0).tolist(),
           "cold_pool_cmax_fc": st("cmax", "fc").mean(0).tolist(), "cold_pool_cmax_truth": st("cmax", "ob").mean(0).tolist(),
           "cold_pool_cmax_rel_err_median": np.median(rel(st("cmax", "fc"), st("cmax", "ob")), 0).tolist()}
    n_lead = st("div", "fc").shape[1]
    scales = [max(1, int(round(s / dx))) for s in (1.0, 4.0)]
    for who, name in (("fc", "fss_lowliq"), ("pers", "fss_lowliq_pers")):
        out[name] = {}
        for thr in (0.3, 1.0):
            vals = {n: np.full((len(phys_cases), n_lead), np.nan) for n in scales}
            for c, case in enumerate(phys_cases):
                for j in range(n_lead):
                    for n, v in fss(case[who]["ll"][j], case["ob"]["ll"][j], thr, scales).items():
                        vals[n][c, j] = v
            out[name][f"{thr:g}"] = {f"{n * dx:g}km": np.nanmean(vals[n], 0).tolist() for n in scales}
    return out


def summarize_physics(fc: dict, leads_min=(1, 5, 15, 30, 60)) -> str:
    if "div_ratio_fc" not in fc:
        return ""
    lead = [round(s / 60) for s in fc["lead_s"]]
    js = [lead.index(m) for m in leads_min if m in lead]
    lines = ["lead_min  div ratio fc (truth)   condensate |rel err| fc (pers)   Cmax fc/truth m/s   "
             "FSS low-level liquid q>1 @1km (pers)"]
    for j in js:
        lines.append(f"{lead[j]:7d}   {fc['div_ratio_fc'][j]:.3f} ({fc['div_ratio_truth'][j]:.3f})        "
                     f"{fc['condensate_mass_rel_err_median'][j]:.3f} ({fc['condensate_mass_rel_err_pers_median'][j]:.3f})"
                     f"            {fc['cold_pool_cmax_fc'][j]:5.1f}/{fc['cold_pool_cmax_truth'][j]:5.1f}       "
                     f"{fc['fss_lowliq']['1']['1km'][j]:.3f} ({fc['fss_lowliq_pers']['1']['1km'][j]:.3f})")
    return "\n".join(lines)

