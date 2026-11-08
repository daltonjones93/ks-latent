"""FPe-S: does the fitted advection speed of channel 0 track each storm's measured propagation speed?

Per storm (train + test, 40 by default; nothing is trained here, so there is no leakage to guard), fit by OLS on frames
f0:f1, interior sites only,
    dz0/dt = c0 + a1 d1z0 + a2 z0 d1z0            (z0: leading channel of fpc_nested_stagec, sites 2 km, frames 1 min)
and define the effective advection speed c_eff = mean over rows, weighted by (d1z0)^2, of -(a1 + a2 z0) (sites/min; x2 km).
c_eff is GAUGE-FREE: shifting z0 by a constant changes a1 but not a1 + a2 z0 at the actual z0.
Measured speeds from the CM1 truth columns of the same storm (Theil-Sen slope of position against time, km/min):
    cond:  condensate-weighted centroid of the column condensate
    cold:  centroid of the cold pool, weights max(0, -theta'_sfc)
Reported: Pearson and Spearman between c_eff and each measured speed over storms, bootstrap CI over storms, permutation p,
the OLS slope of measured on c_eff (1 if the law's speed is the storm speed), and the pooled-fit c_eff.
  python scripts/cm1/storm_speed_law.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/fpe_speed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, theilslopes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0


def storm_fit(z0: np.ndarray, f0: int, f1: int, halo: int):
    """z0 (T, P): returns (c_eff sites/min, coefficients, rows)."""
    z = z0[f0 - 1: f1 + 1][:, :, None]
    dz = 0.5 * (z[2:] - z[:-2])[..., 0]
    zc = z[1:-1, :, 0]
    d1 = site_derivs(z[1:-1], 2)[0][..., 0]
    sl = slice(halo, -halo)
    y, zc, d1 = dz[:, sl].ravel(), zc[:, sl].ravel(), d1[:, sl].ravel()
    X = np.stack([np.ones_like(y), d1, zc * d1], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = d1 ** 2
    c_eff = float(np.sum(w * -(coef[1] + coef[2] * zc)) / np.sum(w))
    return c_eff, coef, len(y)


def measured_speeds(d: ColumnData, norm, ae, f0: int, f1: int, step: int = 5):
    Zl, p = d.Z, ae.cfg.patch
    sc = np.repeat(np.asarray(norm.scale, float), Zl)[:, None]
    ts, xc, xh = [], [], []
    for t in range(f0, f1, step):
        u = d.full(0, t)[0].cpu().numpy() * sc
        cond = (u[VARS.index("qliq") * Zl: (VARS.index("qliq") + 1) * Zl] ** 2).sum(0) + \
               (u[VARS.index("qice") * Zl: (VARS.index("qice") + 1) * Zl] ** 2).sum(0)
        th = u[VARS.index("th") * Zl]
        cond_s, th_s = cond.reshape(-1, p).mean(1), th.reshape(-1, p).mean(1)
        x = np.arange(len(cond_s)) * SITE_KM
        wc, wh = cond_s, np.maximum(0.0, -th_s)
        if wc.sum() > 0 and wh.sum() > 0:
            ts.append(t); xc.append((wc * x).sum() / wc.sum()); xh.append((wh * x).sum() / wh.sum())
    ts = np.asarray(ts, float)
    return float(theilslopes(xc, ts)[0]), float(theilslopes(xh, ts)[0])


def corr_block(c, m, rng, n_boot=2000):
    c, m = np.asarray(c), np.asarray(m)
    r = pearsonr(c, m)[0]
    rs = spearmanr(c, m)[0]
    n = len(c)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(c[i]) > 0 and np.ptp(m[i]) > 0:
            boots.append(pearsonr(c[i], m[i])[0])
    perm = np.mean([abs(pearsonr(c, rng.permutation(m))[0]) >= abs(r) for _ in range(5000)])
    slope = float(np.polyfit(c, m, 1)[0])
    return {"pearson": float(r), "spearman": float(rs), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "perm_p": float(perm), "slope_measured_on_law": slope, "mean_law": float(c.mean()), "mean_measured": float(m.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330")
    ap.add_argument("--x-smooth", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))
    names = list(split["train"][: a.n_train]) + list(split["test"][: a.n_test])
    rows = []
    zall = []
    for n in names:
        d = ColumnData([load_member(files[n])], norm, dev)
        with torch.no_grad():
            z = encode_members(ae, d)[0].cpu().numpy().astype(np.float64)
        z0 = z[..., 0]
        if a.x_smooth > 0:
            from scipy.ndimage import gaussian_filter1d
            z0 = gaussian_filter1d(z0, a.x_smooth, axis=1, mode="nearest")
        c_eff, coef, nrow = storm_fit(z0, f0, min(f1, z0.shape[0] - 2), halo)
        v_cond, v_cold = measured_speeds(d, norm, ae, f0, min(f1, z0.shape[0] - 1))
        rows.append({"storm": n, "c_eff_km_min": SITE_KM * c_eff, "coef": coef.tolist(), "rows": nrow,
                     "v_cond_km_min": v_cond, "v_cold_km_min": v_cold})
        zall.append(z0)
        print(f"[speed] {n}: law {SITE_KM * c_eff:+.3f} km/min, condensate {v_cond:+.3f}, cold pool {v_cold:+.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    # pooled fit on the same storms (for the headline coefficient)
    c_pool = []
    ce = np.array([r["c_eff_km_min"] for r in rows])
    res = {"run": a.run, "frames": a.frames, "x_smooth": a.x_smooth, "storms": rows,
           "law_speed_mean_sd": [float(ce.mean()), float(ce.std())]}
    rng = np.random.default_rng(0)
    for key in ("v_cond_km_min", "v_cold_km_min"):
        m = np.array([r[key] for r in rows])
        res[key] = corr_block(ce, m, rng)
        b = res[key]
        print(f"[speed] law speed vs {key}: Pearson {b['pearson']:+.2f} (95% CI {b['ci95'][0]:+.2f}..{b['ci95'][1]:+.2f}, "
              f"perm p {b['perm_p']:.3f}), Spearman {b['spearman']:+.2f}, slope {b['slope_measured_on_law']:+.2f}; "
              f"mean law {b['mean_law']:+.3f} vs measured {b['mean_measured']:+.3f} km/min", flush=True)
    json.dump(res, open(out / "speed.json", "w"), indent=1)
    print(f"[speed] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
