"""10b: keeping the low RMSE of the ensemble average while keeping realistic structure (user, 2026-11-24: "test 1-4 as postprocessing").

From the pooled forecast (--models, --members each; 4 x 8 = 32 members), per case and lag, single-field products built AFTER the forecast
(no retraining), in model 0's normalized units with condensate as sqrt(q) (the training units):
  mean          the plain average (reference: lowest RMSE, smooth)
  member        one member (reference: realistic, ~sqrt(2) x the mean's RMSE)
  medoid        (4) the member closest to the average
  align         (1) members shifted along x to agree (global shift each, 2 passes against the running aligned mean, search +-40 km),
                    re-centred on the members' average position, then averaged
  align_local   (1b) as align, then a smooth x-displacement field per member (shifts in 40 km windows every 10 km, +-12 km,
                    linearly interpolated) before averaging, re-centred by the mean displacement
  pm            (2) probability matching: the average's ranks at each level, filled with the pooled members' values at that level
  pm_align      (2 + 1) probability matching of the aligned average
  blend10/20    (3) large scales of the average + small scales of one member (Gaussian split, half power at ~10 / ~20 km wavelength)
  blend10_align (3 + 1) large scales of the average + small scales of the aligned average
Scores summed over cases (merge with postproc_report.py): squared error, pattern correlation, power by wavelength band (forecast and truth),
FSS of |w| > 3 m/s and liquid > 0.5 g/kg at 4 / 10 km, and the contingency table of every feature in cell_losses.FEATURES.
Storms are split over jobs with --job / --n-jobs.

  python scripts/cm1/postproc_eval.py --models R0 R1 R2 R3 --members 8 --job 0 --n-jobs 11 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ensemble_design import box_x, roll, to_phys  # noqa: E402
from ks_latent.cm1.cell_losses import FEATURES  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

DX = 0.25
BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
KINDS = ["mean", "member", "medoid", "align", "align_local", "pm", "pm_align", "blend10", "blend20", "blend10_align"]
FSS = {"w3": ("w", 3.0, True), "liq05": ("qliq", 0.5, False)}
FSS_KM = [4, 10]
COND = [VARS.index("qliq"), VARS.index("qice")]


def shift_x(f, s):
    """f (..., W) shifted by s columns (content moves right for s > 0), edges replicated"""
    if s == 0:
        return f
    W = f.shape[-1]
    idx = torch.clamp(torch.arange(W) - s, 0, W - 1)
    return f[..., idx]


def best_shift(m, target, smax, weight=None):
    """integer shift of m (V, Z, W) minimising the squared distance to target over [-smax, smax]"""
    W = m.shape[-1]
    pad = torch.nn.functional.pad(m.reshape(1, -1, W), (smax, smax), mode="replicate")[0].reshape(*m.shape[:-1], W + 2 * smax)
    best, bs = None, 0
    for s in range(-smax, smax + 1):
        c = pad[..., smax - s: smax - s + W]
        d = (c - target) ** 2
        if weight is not None:
            d = d * weight
        e = float(d.sum())
        if best is None or e < best:
            best, bs = e, s
    return bs


def warp_x(f, d):
    """f (V, Z, W) sampled at x - d(x) (d in columns, float, shape (W,)): moves content right by d, linear interpolation"""
    W = f.shape[-1]
    pos = torch.clamp(torch.arange(W, dtype=torch.float32) - d, 0, W - 1)
    i0 = pos.floor().long(); i1 = torch.clamp(i0 + 1, max=W - 1); w = pos - i0
    return f[..., i0] * (1 - w) + f[..., i1] * w


def local_disp(m, target, win, step, smax):
    """displacement field (W,) of m against target: best shift in windows of `win` columns every `step`, interpolated"""
    W = m.shape[-1]
    pad = torch.nn.functional.pad(m.reshape(1, -1, W), (smax, smax), mode="replicate")[0].reshape(*m.shape[:-1], W + 2 * smax)
    cs, ds = [], []
    for c in range(win // 2, W - win // 2 + 1, step):
        lo, hi = c - win // 2, c + win // 2
        tw = target[..., lo:hi]
        if float(tw.abs().sum()) == 0:
            continue
        best, bs = None, 0
        for s in range(-smax, smax + 1):
            e = float(((pad[..., smax - s + lo: smax - s + hi] - tw) ** 2).sum())
            if best is None or e < best:
                best, bs = e, s
        cs.append(c); ds.append(bs)
    if not cs:
        return torch.zeros(W)
    return torch.as_tensor(np.interp(np.arange(W), cs, ds), dtype=torch.float32)


def lowpass(f, sigma_km):
    s = sigma_km / DX; r = int(np.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=f.dtype) / s) ** 2); k = k / k.sum()
    sh = f.shape
    y = torch.nn.functional.pad(f.reshape(-1, 1, sh[-1]), (r, r), mode="replicate")
    return torch.nn.functional.conv1d(y, k.view(1, 1, -1)).reshape(sh)


def prob_match(mean, ens):
    """per variable and level: the mean's ranks along x filled with every S-th value of the pooled members' sorted values"""
    S, V, Z, W = ens.shape
    pool = ens.permute(1, 2, 0, 3).reshape(V, Z, S * W).sort(-1).values
    q = pool[..., (torch.arange(W) * S + S // 2)]                              # (V, Z, W) quantiles
    out = torch.empty_like(mean)
    out.scatter_(-1, mean.argsort(-1), q)
    return out


def band_power(x):
    W = x.shape[-1]
    f = (torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs() ** 2).mean(-2)
    k = np.arange(f.shape[-1]); wl = np.where(k > 0, W * DX / np.maximum(k, 1), np.inf)
    return torch.stack([f[..., torch.as_tensor((wl >= lo) & (wl < hi))].sum(-1) for lo, hi in BANDS], -1)


def to_norm(x, sc0):
    y = x.clone()
    for i in COND:
        y[..., i, :, :] = y[..., i, :, :].clamp(min=0).sqrt()
    return y / sc0[:, None, None]


def to_physical(xn, sc0):
    y = xn * sc0[:, None, None]
    for i in COND:
        y[..., i, :, :] = y[..., i, :, :].clamp(min=0) ** 2
    return y


class Acc:
    def __init__(self):
        V, B = len(VARS), len(BANDS)
        self.d = {"se": np.zeros(V), "cor": np.zeros(V), "pw": np.zeros((V, B)), "pt": np.zeros((V, B)), "n": 0,
                  "fss": {f"{k}@{r}": [0.0, 0.0] for k in FSS for r in FSS_KM}, "cont": {f: [0.0, 0.0, 0.0] for f in FEATURES}}

    def add(self, fn, tn, fp, tp):
        d = self.d
        d["se"] += ((fn - tn) ** 2).double().mean((1, 2)).numpy()
        a = fn - fn.mean((1, 2), keepdim=True); b = tn - tn.mean((1, 2), keepdim=True)
        d["cor"] += ((a * b).sum((1, 2)) / ((a ** 2).sum((1, 2)).sqrt() * (b ** 2).sum((1, 2)).sqrt() + 1e-12)).double().numpy()
        d["pw"] += band_power(fn).double().numpy(); d["pt"] += band_power(tn).double().numpy(); d["n"] += 1
        for k, (v, thr, ab) in FSS.items():
            i = VARS.index(v)
            mf = ((fp[i].abs() if ab else fp[i]) > thr).double(); mo = ((tp[i].abs() if ab else tp[i]) > thr).double()
            for r in FSS_KM:
                rc = int(round(r / 2 / DX))
                pf, po = box_x(mf, rc), box_x(mo, rc)
                d["fss"][f"{k}@{r}"][0] += float(((pf - po) ** 2).sum()); d["fss"][f"{k}@{r}"][1] += float((pf ** 2).sum() + (po ** 2).sum())
        for f, (v, sign, thr) in FEATURES.items():
            i = VARS.index(v)
            pf = sign * fp[i] > thr; po = sign * tp[i] > thr
            c = d["cont"][f]
            c[0] += float((pf & po).sum()); c[1] += float((~pf & po).sum()); c[2] += float((pf & ~po).sum())

    def to_json(self):
        d = self.d
        return {"se": d["se"].tolist(), "cor": d["cor"].tolist(), "pw": d["pw"].tolist(), "pt": d["pt"].tolist(), "n": d["n"],
                "fss": d["fss"], "cont": d["cont"]}


def products(ens, smax, win, step, smax_loc):
    """ens (S, V, Z, W) normalized -> dict of single fields (V, Z, W)"""
    S = ens.shape[0]
    mean = ens.mean(0)
    out = {"mean": mean, "member": ens[0]}
    d2 = ((ens - mean) ** 2).flatten(1).mean(1)
    out["medoid"] = ens[int(d2.argmin())]
    # (1) global alignment: two passes against the running aligned mean, then re-centre
    target = mean; shifts = [0] * S
    for _ in range(2):
        shifts = [best_shift(ens[j], target, smax) for j in range(S)]
        target = torch.stack([shift_x(ens[j], shifts[j]) for j in range(S)]).mean(0)
    sbar = int(round(float(np.mean(shifts))))
    al = torch.stack([shift_x(ens[j], shifts[j] - sbar) for j in range(S)])
    out["align"] = al.mean(0)
    # (1b) local displacement on top of the global alignment
    tgt = out["align"]
    disp = torch.stack([local_disp(al[j], tgt, win, step, smax_loc) for j in range(S)])
    disp = disp - disp.mean(0, keepdim=True)
    out["align_local"] = torch.stack([warp_x(al[j], disp[j]) for j in range(S)]).mean(0)
    # (2) probability matching
    out["pm"] = prob_match(mean, ens)
    out["pm_align"] = prob_match(out["align"], ens)
    # (3) scale blends (Gaussian sigma 1.9 / 3.7 km: half power at ~10 / ~20 km wavelength)
    lp10, lp20 = lowpass(mean, 1.9), lowpass(mean, 3.7)
    out["blend10"] = lp10 + (ens[0] - lowpass(ens[0], 1.9))
    out["blend20"] = lp20 + (ens[0] - lowpass(ens[0], 3.7))
    out["blend10_align"] = lp10 + (out["align"] - lowpass(out["align"], 1.9))
    return out


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--split", choices=["val", "test"], default="test"); ap.add_argument("--t0", default="120,180,240")
    ap.add_argument("--lags", default="5,15,30,60,90"); ap.add_argument("--job", type=int, default=0); ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-storms", type=int, default=99)
    ap.add_argument("--smax-km", type=float, default=40.0); ap.add_argument("--win-km", type=float, default=40.0)
    ap.add_argument("--step-km", type=float, default=10.0); ap.add_argument("--smax-local-km", type=float, default=12.0)
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False); t_start = time.time()
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    models = [load_run(Path(m).expanduser(), torch.device("cpu")) for m in a.models]
    names = models[0][4][a.split][: a.n_storms][a.job::a.n_jobs]
    M = a.members; files = member_files(a.source)
    sc0 = torch.as_tensor(np.array(models[0][2].scale, float), dtype=torch.float32)
    mar = max((m[3]["stage_b"]["halo_sites"] + m[0].cfg.reach_sites) * m[0].cfg.patch for m in models)
    smax, win, step, sl = (int(round(x / DX)) for x in (a.smax_km, a.win_km, a.step_km, a.smax_local_km))
    acc = {(k, L): Acc() for k in KINDS for L in lags}
    for nm in names:
        per = []
        for ae, prop, norm, cfg, _ in models:
            d = ColumnData([load_member(files[nm])], norm, torch.device("cpu"))
            per.append((ae, prop, norm, cfg["stage_b"]["halo_sites"], encode_members(ae, d)[0], d))
        Z = per[0][5].Z
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > per[0][4].shape[0]:
                continue
            ens = {L: [] for L in lags}
            for j, (ae, prop, norm, halo, z, _) in enumerate(per):
                tr = roll(prop, z, t0, n, halo, M, a.seed + 1000 * j + t0, 1.0, 0.0)
                for L in lags:
                    ens[L].append(to_phys(ae.decode(tr[L]), norm, Z)[..., mar:-mar])
            for L in lags:
                tp = to_phys(per[0][5].full(0, t0 + L)[0], per[0][2], Z)[..., mar:-mar]
                en = to_norm(torch.cat(ens[L]), sc0); tn = to_norm(tp, sc0)
                for k, fn in products(en, smax, win, step, sl).items():
                    acc[(k, L)].add(fn, tn, to_physical(fn, sc0), tp)
            print(f"[postproc] {nm} t0 {t0} ({time.time() - t_start:.0f}s)", flush=True)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / f"part_{a.job:02d}.json").write_text(json.dumps({"config": vars(a), "vars": list(VARS), "bands_km": BANDS, "kinds": KINDS,
                                                            "lags": lags, "storms": names, "features": list(FEATURES),
                                                            "acc": {f"{k}|{L}": A.to_json() for (k, L), A in acc.items()},
                                                            "wall_s": time.time() - t_start}))
    print(f"[postproc] wrote part {a.job} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
