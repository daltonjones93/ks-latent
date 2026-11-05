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
surface theta' RMSE (K) and the gust-front position error (km): the front
is the easternmost cell with surface theta' < -1 K.
"""
from __future__ import annotations

import numpy as np
import torch

from ks_latent.cm1.data import VARS, ColumnData, Normalizer
from ks_latent.cm1.models import ColumnFieldAE, StencilPropagator


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
    V, Z = data.V, data.Z
    se = torch.zeros(V); se_act = torch.zeros(V); n = 0; n_act = 0
    for m in range(len(data.u)):
        for t in range(0, data.T[m], stride_t):
            u = data.full(m, t)                           # (1, C, NX)
            e = (ae(u)[0] - u)[0].reshape(V, Z, -1).cpu() ** 2  # normalized
            act = data.cref[m][t] > 10
            se += e.mean(dim=1).sum(dim=1); n += e.shape[-1]
            se_act += e.mean(dim=1)[:, act].sum(dim=1); n_act += int(act.sum())
    sc = torch.tensor(norm.scale, dtype=torch.float32)
    out = {"rmse_phys_all": dict(zip(VARS, (sc * (se / n).sqrt()).tolist())),
           "nmse_all": float((se / n).mean())}
    if n_act:
        out["rmse_phys_storm"] = dict(zip(VARS, (sc * (se_act / n_act).sqrt()).tolist()))
        out["nmse_storm"] = float((se_act / n_act).mean())
    return out


@torch.no_grad()
def forecast_report(ae: ColumnFieldAE, prop: StencilPropagator, data: ColumnData, z: list[torch.Tensor],
                    norm: Normalizer, t0s: list[int], n_lead: int, halo: int) -> dict:
    ae.eval(); prop.eval()
    Z = data.Z
    M = len(data.u)
    rows = {k: [] for k in ("nmse", "nmse_pers", "nmse_spread", "th_sfc_rmse", "th_sfc_rmse_pers", "front_err_km",
                            "front_err_pers_km")}
    for t0 in t0s:
        if any(t0 + n_lead >= T for T in data.T):
            continue
        truth = torch.stack([data.full(m, slice(t0, t0 + n_lead + 1)) for m in range(M)])  # (M, L+1, C, NX)
        for m in range(M):
            state = z[m][t0:t0 + 1]                                                       # (1, P, c)
            P = state.shape[1]
            preds = []
            for j in range(n_lead):
                state = prop(state)
                if halo > 0:
                    zt = z[m][t0 + j + 1:t0 + j + 2]
                    state = torch.cat([zt[:, :halo], state[:, halo:P - halo], zt[:, P - halo:]], dim=1)
                preds.append(ae.decode(state)[0])
            pred = torch.stack(preds)                                                     # (L, C, NX)
            tr = truth[m, 1:]
            pers = truth[m, :1].expand_as(tr)
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
    out = {"t0s": t0s, "n_lead": n_lead, "lead_s": [(j + 1) * data.dt for j in range(n_lead)], "halo_sites": halo}
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
