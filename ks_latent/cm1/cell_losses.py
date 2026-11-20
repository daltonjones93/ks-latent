"""Terms for "when and where do cells form" (9a, 2026-11-18), on decoded columns `x` of shape (..., C, W), C = V * Z channels (variable-major),
in NORMALIZED units (W-axis = x, 0.25 km per column). Thresholds are given in physical units and converted with the per-variable scales:
w in m/s; condensate in g/kg (the condensate channels hold sqrt(q) / scale, so a threshold q0 is sqrt(q0) / scale).
  tw_mae          threshold-weighted absolute error: |v(x) - v(y)| for the smooth chaining functions v = softrelu(x - tau) (updrafts), softrelu(-x - tau) (downdrafts)
                  and softrelu(x - tau_q) (liquid, ice). For a single forecast this is the threshold-weighted CRPS: zero weight where both are below
                  the threshold (the quiet region), proper in the sense of Allen, Ginsbourger and Ziegel (2022).
  pooled_exceed_mae  the same exceedance amplitudes averaged over horizontal windows of r km (stride r/2): a cell within r km scores,
                  without demanding the exact column (the displacement double penalty; a fractions-skill-score-like neighbourhood term).
  feature_log_score  9d: cross-entropy of updraft / downdraft presence (w above tau_up; w below -tau_down): the decoded w becomes a probability
                  through a sigmoid of width `soft` (calibrated to the reconstruction error of w); an ensemble's probability is the member mean of
                  the members' probabilities (the log score of the ensemble probability is proper; per-member cross-entropy would collapse the spread).
  lowpass_mse     mean squared error after a Gaussian low-pass along x (large scales; indifferent to where small cells sit).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ks_latent.cm1.data import VARS

DX_KM = 0.25


def var_slice(name: str, Z: int) -> slice:
    i = VARS.index(name)
    return slice(i * Z, (i + 1) * Z)


def tau_norm(scale: dict | np.ndarray, w_thresh: float, q_thresh: float) -> tuple[float, float]:
    """Normalized thresholds: w / scale_w; sqrt(q) / scale_q for the (sqrt-stored) condensate."""
    sc = {v: float(scale[VARS.index(v)]) for v in ("w", "qliq", "qice")} if not isinstance(scale, dict) else scale
    return w_thresh / sc["w"], float(np.sqrt(q_thresh)) / sc["qliq"]


SOFT = 0.25      # softness of the chaining function as a fraction of the threshold


def soft_relu(z: torch.Tensor, s: float) -> torch.Tensor:
    """s * softplus(z / s): a smooth, non-decreasing chaining function (a hard relu has zero gradient below the threshold, so a missed
    cell could never be pulled up; this keeps a small gradient there and tends to the hard relu as s -> 0)."""
    return s * F.softplus(z / s)


def exceed_fields(x: torch.Tensor, Z: int, tau_w: float, tau_q: float, soft: float = SOFT) -> tuple:
    """(updraft, downdraft, liquid, ice) smooth exceedance amplitudes, each (..., Z, W)."""
    w, ql, qi = x[..., var_slice("w", Z), :], x[..., var_slice("qliq", Z), :], x[..., var_slice("qice", Z), :]
    sw, sq = max(soft * tau_w, 1e-3), max(soft * tau_q, 1e-3)
    return soft_relu(w - tau_w, sw), soft_relu(-w - tau_w, sw), soft_relu(ql - tau_q, sq), soft_relu(qi - tau_q, sq)


def tw_mae(d: torch.Tensor, t: torch.Tensor, Z: int, tau_w: float, tau_q: float) -> torch.Tensor:
    fd, ft = exceed_fields(d, Z, tau_w, tau_q), exceed_fields(t, Z, tau_w, tau_q)
    return torch.stack([(a - b).abs().mean() for a, b in zip(fd, ft)]).mean()


def _pool_x(f: torch.Tensor, kernel: int) -> torch.Tensor:
    sh = f.shape
    p = F.avg_pool1d(f.reshape(-1, 1, sh[-1]), kernel_size=kernel, stride=max(1, kernel // 2))
    return p.reshape(*sh[:-1], p.shape[-1])


def pooled_exceed_mae(d: torch.Tensor, t: torch.Tensor, Z: int, tau_w: float, tau_q: float,
                      scales_km: tuple = (2.0, 4.0, 8.0, 16.0)) -> torch.Tensor:
    """Mean over the scales of |pooled exceedance(d) - pooled exceedance(t)| for the vertical-wind and the condensate exceedances."""
    ud, dd, ld, id_ = exceed_fields(d, Z, tau_w, tau_q)
    ut, dt, lt, it = exceed_fields(t, Z, tau_w, tau_q)
    ad, at = ud + dd, ut + dt                     # |w| above the threshold
    qd, qt = ld + id_, lt + it                    # condensate above the threshold
    W = d.shape[-1]
    out = []
    for s in scales_km:
        K = int(min(max(1, round(s / DX_KM)), W))
        out.append(((_pool_x(ad, K) - _pool_x(at, K)).abs().mean() + (_pool_x(qd, K) - _pool_x(qt, K)).abs().mean()) / 2)
    return torch.stack(out).mean()


def pooled_exceed_crps(d: torch.Tensor, t: torch.Tensor, Z: int, tau_w: float, tau_q: float,
                       scales_km: tuple = (2.0, 4.0, 8.0, 16.0)) -> torch.Tensor:
    """9c, the stochastic version of `pooled_exceed_mae`: `d` is the ensemble (S, ..., C, W), `t` the truth (..., C, W). The |w| and condensate
    exceedances of every member are pooled over r-km windows and scored against the pooled truth with the fair CRPS (a proper score of the
    ensemble's distribution of neighbourhood exceedance amplitudes: it rewards a calibrated spread of cell positions within r km, not just a
    good mean)."""
    from ks_latent.cm1.train import crps_elements
    ud, dd, ld, id_ = exceed_fields(d, Z, tau_w, tau_q)
    ut, dt, lt, it = exceed_fields(t, Z, tau_w, tau_q)
    ad, at, qd, qt = ud + dd, ut + dt, ld + id_, lt + it
    W = d.shape[-1]
    out = []
    for s in scales_km:
        K = int(min(max(1, round(s / DX_KM)), W))
        out.append((crps_elements(_pool_x(ad, K), _pool_x(at, K)).mean() + crps_elements(_pool_x(qd, K), _pool_x(qt, K)).mean()) / 2)
    return torch.stack(out).mean()


def feature_masks(t: torch.Tensor, Z: int, tau_up: float, tau_down: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Truth (updraft, downdraft) presence masks, each (..., Z, W) float: w > tau_up; w < -tau_down. A minimum-width rule (LeMone and Zipser
    1980, 500 m = 2 columns) was measured to remove under 1.5% of the feature points at 250 m spacing (scripts/cm1/feature_calibration.py), so
    none is applied."""
    w = t[..., var_slice("w", Z), :]
    return (w > tau_up).to(t.dtype), (w < -tau_down).to(t.dtype)


def feature_logits(d: torch.Tensor, Z: int, tau_up: float, tau_down: float, soft: float) -> tuple[torch.Tensor, torch.Tensor]:
    w = d[..., var_slice("w", Z), :]
    return (w - tau_up) / soft, (-w - tau_down) / soft


def feature_log_score(d: torch.Tensor, t: torch.Tensor, Z: int, tau_up: float, tau_down: float, soft: float,
                      eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean binary cross-entropy (nats per point) of updraft and downdraft presence. `d`: forecast samples (S, ..., C, W) (S = 1:
    deterministic, computed from the logits for stability); `t`: truth (..., C, W). No class reweighting: a weighted cross-entropy is
    minimized by a biased probability (improper), so a rare class keeps its natural weight."""
    yu, yd = feature_masks(t, Z, tau_up, tau_down)
    lu, ld = feature_logits(d, Z, tau_up, tau_down, soft)
    if d.shape[0] == 1:
        return (F.binary_cross_entropy_with_logits(lu[0], yu), F.binary_cross_entropy_with_logits(ld[0], yd))
    out = []
    for lg, y in ((lu, yu), (ld, yd)):
        p = torch.sigmoid(lg).mean(0).clamp(eps, 1 - eps)          # ensemble probability
        out.append(F.binary_cross_entropy(p, y))
    return tuple(out)


# 9e: presence features of every variable (scripts/cm1/feature_calibration_all.py on 30 training storms; thresholds in physical units of the
# variable, condensate in g/kg; base rate = fraction of all domain points). Rule: a round physical threshold with a base rate of about
# 1.5-7% and at least 5 reconstruction errors from zero; theta' and p' are perturbations from the initial sounding and drift (median
# theta' +0.5 K, median p' +0.9 hPa), so their warm / high thresholds sit beyond the drift.
FEATURES = {
    "u+": ("u", 1.0, 10.0),      # strong positive u' (outflow / front-to-rear jet), 3.6%
    "u-": ("u", -1.0, 10.0),     # strong negative u' (inflow / rear-inflow jet), 2.8%
    "w+": ("w", 1.0, 2.0),       # updraft, 1.6%
    "w-": ("w", -1.0, 1.5),      # downdraft, 1.7%
    "th-": ("th", -1.0, 2.0),    # cold pool / cold anomaly below -2 K, 6.6%
    "th+": ("th", 1.0, 5.0),     # warm core above +5 K (latent heating), 2.5%
    "qv+": ("qv", 1.0, 1.0),     # moist anomaly above +1 g/kg, 2.8%
    "qv-": ("qv", -1.0, 3.0),    # dry anomaly below -3 g/kg (downdraft / cold-pool air), 3.7%
    "prs+": ("prs", 1.0, 3.0),   # mesohigh above +3 hPa, ~3%
    "prs-": ("prs", -1.0, 1.0),  # mesolow below -1 hPa, 1.7%
    "qliq+": ("qliq", 1.0, 0.5),  # liquid core above 0.5 g/kg, 2.5%
    "qice+": ("qice", 1.0, 1.0),  # ice above 1 g/kg, 3.4%
}
# sigmoid width per variable in STORED units (sqrt(g/kg) for condensate): twice the domain-mean reconstruction error (F7 / F4, lag 0)
FEATURE_SOFT = {"u": 1.25, "w": 0.5, "th": 0.7, "qv": 0.3, "prs": 0.12, "qliq": 0.07, "qice": 0.075}
FEATURE_GROUPS = {"u": "u+,u-", "w": "w+,w-", "th": "th-,th+", "qv": "qv+,qv-", "prs": "prs+,prs-", "qliq": "qliq+", "qice": "qice+"}


def feature_specs(names: str, scale) -> list[tuple]:
    """Comma list of FEATURES keys (or a variable name: its group) -> [(name, var, sign, tau, soft)] in NORMALIZED stored units, with `scale`
    the per-variable normalizer scale (array over VARS or dict)."""
    sc = {v: float(scale[VARS.index(v)]) for v in VARS} if not isinstance(scale, dict) else scale
    out = []
    for nm in [x.strip() for x in names.split(",") if x.strip()]:
        for key in FEATURE_GROUPS.get(nm, nm).split(","):
            if key not in FEATURES:
                raise ValueError(f"unknown feature {key!r}; known: {sorted(FEATURES)} or variables {sorted(FEATURE_GROUPS)}")
            v, sign, thr = FEATURES[key]
            stored = float(np.sqrt(thr)) if v in ("qliq", "qice") else thr
            out.append((key, v, sign, stored / sc[v], FEATURE_SOFT[v] / sc[v]))
    if len({o[0] for o in out}) != len(out):
        raise ValueError(f"duplicate features in {names!r}")
    return out


def feature_scores(d: torch.Tensor, t: torch.Tensor, Z: int, specs: list, eps: float = 1e-4, pos_weight: torch.Tensor | None = None,
                   kind: str = "bce") -> dict:
    """{name: loss} for each spec; `d` forecast samples (S, ..., C, W), S = 1 deterministic (logits), else the ensemble probability (member
    mean); `t` truth (..., C, W). As `feature_log_score`, generalized to any variable.
    9f, the fix for the 9d failure (a cross-entropy-trained deterministic model predicts a feature only where its probability exceeds 0.5,
    and under position uncertainty that under-forecasts the area: bias 0.5 for updrafts at +30 min):
      pos_weight  class weight of the feature (broadcast against (..., Z, W), e.g. (k, 1, 1) for one weight per rollout step): the weighted
                  cross-entropy is minimized by a probability that crosses 0.5 where the true probability exceeds 1 / (1 + pos_weight), so the
                  model draws the feature down to that probability. The loss is divided by the mean weight, keeping its scale.
      kind="csi"  soft critical success index, 1 - sum(p y) / (sum p + sum y - sum(p y)) over the batch (Lagerquist and Ebert-Uphoff 2022):
                  no class weight to calibrate; the optimum covers the area that maximizes CSI."""
    if kind not in ("bce", "csi"):
        raise ValueError(f"unknown feature loss {kind!r}")
    out = {}
    for name, v, sign, tau, soft in specs:
        sl = var_slice(v, Z)
        y = (sign * t[..., sl, :] > tau).to(t.dtype)
        lg = (sign * d[..., sl, :] - tau) / soft
        if kind == "csi":
            p = torch.sigmoid(lg).mean(0)
            inter = (p * y).sum()
            out[name] = 1.0 - inter / (p.sum() + y.sum() - inter + 1.0)
        elif pos_weight is not None:
            w_ = pos_weight.to(lg.dtype)
            if d.shape[0] == 1:
                l_ = F.binary_cross_entropy_with_logits(lg[0], y, pos_weight=w_, reduction="none")
            else:
                p = torch.sigmoid(lg).mean(0).clamp(eps, 1 - eps)
                l_ = -(w_ * y * torch.log(p) + (1 - y) * torch.log(1 - p))
            out[name] = l_.mean() / (w_ * y + (1 - y)).mean()
        elif d.shape[0] == 1:
            out[name] = F.binary_cross_entropy_with_logits(lg[0], y)
        else:
            out[name] = F.binary_cross_entropy(torch.sigmoid(lg).mean(0).clamp(eps, 1 - eps), y)
    return out


def gaussian_kernel(sigma_cols: float, device, dtype) -> torch.Tensor:
    r = int(np.ceil(3 * sigma_cols))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=device, dtype=dtype) / sigma_cols) ** 2)
    return k / k.sum()


def lowpass_x(x: torch.Tensor, sigma_km: float, chunk: int = 4096) -> torch.Tensor:
    """Gaussian low-pass along the last axis (reflect padding; the radius is limited to the window). The signals are processed in chunks of
    `chunk` rows: one convolution over ~80 000 signals exceeds a CUDA grid limit ("invalid configuration argument")."""
    W = x.shape[-1]
    s = sigma_km / DX_KM
    k = gaussian_kernel(s, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    if r >= W:
        raise ValueError(f"low-pass radius {r} columns does not fit a window of {W} columns")
    sh = x.shape
    flat = x.reshape(-1, 1, W)
    out = [F.conv1d(F.pad(flat[i:i + chunk], (r, r), mode="reflect"), k.view(1, 1, -1)) for i in range(0, flat.shape[0], chunk)]
    return torch.cat(out, 0).reshape(sh)


def lowpass_mse(d: torch.Tensor, t: torch.Tensor, sigma_km: float) -> torch.Tensor:
    return ((lowpass_x(d, sigma_km) - lowpass_x(t, sigma_km)) ** 2).mean()
