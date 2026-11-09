"""FPe-U: the leading-channel advection law in physical units.

Channel 0 of the ordered latent has an arbitrary sign, scale and zero point (the gauge). Here it is tied to a physical
quantity, the site-mean surface theta' (K) from the truth columns, and the one-step law is rewritten in K:
  1. gauge map theta' = s z0 + b from the TRAINING storms (site means, every 5th frame): s by geometric-mean
     regression sign(r) sd(theta')/sd(z0) (OLS and inverse-OLS slopes are the bounds, r is reported);
  2. one-step OLS (no sparsity, no rollout refit) on the same storms, interior sites, frames f0:f1:
         dz0/dt = k0 + a1 dx z0 + a2 z0 dx z0 + nu d2x z0        (sites 2 km, minutes)
  3. in theta' units:   d theta'/dt = s k0 - c(theta') dx theta' + nu d2x theta',   c(theta') = c0 + c1 theta'
         c0 = -(a1 + a2 (-b)/s) [km/min x 2],  c1 = -a2 / s  [km/min per K],  nu [m^2/s]
     c1 > 0 with c0 < 0 means: the colder the pool, the faster the front moves (toward -x).
  4. uncertainty: bootstrap over storms; also the same numbers from the 10 TEST storms with the training map.
Density-current scaling for comparison: |c| ~ sqrt(2 g H |theta'| / theta0), so d|c|/d|theta'| = |c| / (2 |theta'|),
evaluated at the mean |c| and the (dx theta')^2-weighted mean |theta'| of the data.
  python scripts/cm1/law_physical_units.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/fpe_phys
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0
NU_UNIT = (SITE_KM * 1000.0) ** 2 / 60.0       # sites^2/min -> m^2/s


def rows_of(z0: np.ndarray, f0: int, f1: int, halo: int):
    z = z0[f0 - 1: f1 + 1][:, :, None]
    y = (0.5 * (z[2:] - z[:-2]))[..., 0]
    zc = z[1:-1, :, 0]
    d1, d2 = (d[..., 0] for d in site_derivs(z[1:-1], 2))
    sl = slice(halo, -halo)
    return tuple(a[:, sl].ravel() for a in (y, zc, d1, d2))


def solve(y, zc, d1, d2):
    X = np.stack([np.ones_like(y), d1, zc * d1, d2], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def convert(coef, s, b, zc, d1):
    k0, a1, a2, nu = coef
    c0 = -(a1 + a2 * (-b) / s) * SITE_KM
    c1 = -a2 / s * SITE_KM
    w = d1 ** 2
    th = s * zc + b
    c_mean = float(np.sum(w * -(a1 + a2 * zc)) / np.sum(w) * SITE_KM)
    th_w = float(np.sum(w * th) / np.sum(w))
    dc = abs(c_mean) / (2.0 * abs(th_w)) if th_w != 0 else float("nan")
    return {"c0_km_min": float(c0), "c1_km_min_per_K": float(c1), "nu_m2_s": float(nu * NU_UNIT),
            "source_K_per_min": float(s * k0), "c_eff_km_min": c_mean, "theta_weighted_K": th_w,
            "density_current_c1_km_min_per_K": float(np.sign(c1) * dc) if np.isfinite(dc) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))
    Zl, p = None, ae.cfg.patch

    def load(names):
        zs, ths = [], []
        for n in names:
            d = ColumnData([load_member(files[n])], norm, dev)
            with torch.no_grad():
                z = encode_members(ae, d)[0].cpu().numpy().astype(np.float64)
            sc = np.repeat(np.asarray(norm.scale, float), d.Z)[:, None]
            zz, tt = [], []
            for t in range(f0, min(f1, z.shape[0] - 2), 5):
                u = d.full(0, t)[0].cpu().numpy() * sc
                th = u[VARS.index("th") * d.Z]
                tt.append(th.reshape(-1, p).mean(1)); zz.append(z[t, :, 0])
            zs.append(z[..., 0]); ths.append((np.concatenate(zz), np.concatenate(tt)))
        return zs, ths
    ztr, thtr = load(split["train"][: a.n_train])
    zte, thte = load(split["test"][: a.n_test])
    zf = np.concatenate([x[0] for x in thtr]); tf = np.concatenate([x[1] for x in thtr])
    r = float(np.corrcoef(zf, tf)[0, 1])
    s_ols = float(np.cov(zf, tf)[0, 1] / zf.var(ddof=1))
    s_inv = float(tf.var(ddof=1) / np.cov(zf, tf)[0, 1])
    s_geo = float(np.sign(r) * tf.std(ddof=1) / zf.std(ddof=1))
    b_geo = float(tf.mean() - s_geo * zf.mean())
    print(f"[phys] gauge map theta' = s z0 + b: r {r:+.3f}; s geometric {s_geo:+.3f} K (OLS {s_ols:+.3f}, inverse OLS {s_inv:+.3f}), "
          f"b {b_geo:+.3f} K; theta' mean {tf.mean():+.2f} sd {tf.std():.2f} K, z0 mean {zf.mean():+.2f} sd {zf.std():.2f}", flush=True)

    R_tr = [rows_of(z, f0, min(f1, z.shape[0] - 2), halo) for z in ztr]
    R_te = [rows_of(z, f0, min(f1, z.shape[0] - 2), halo) for z in zte]
    cat = lambda Rs: tuple(np.concatenate([r_[i] for r_ in Rs]) for i in range(4))
    res = {"run": a.run, "map": {"r": r, "s_geometric": s_geo, "s_ols": s_ols, "s_inverse_ols": s_inv, "b": b_geo,
                                 "theta_mean": float(tf.mean()), "theta_sd": float(tf.std())}}
    for tag, Rs in (("train", R_tr), ("test", R_te)):
        y, zc, d1, d2 = cat(Rs)
        coef = solve(y, zc, d1, d2)
        # R^2 of the one-step fit
        X = np.stack([np.ones_like(y), d1, zc * d1, d2], 1)
        r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2))
        res[tag] = {"coef_sites": coef.tolist(), "R2": r2, **convert(coef, s_geo, b_geo, zc, d1)}
        # sensitivity to the map slope: bounds
        for nm, s in (("ols", s_ols), ("inverse_ols", s_inv)):
            res[tag][f"c1_with_{nm}_slope"] = float(-coef[2] / s * SITE_KM)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.n_boot):
        idx = rng.integers(0, len(R_tr), len(R_tr))
        y, zc, d1, d2 = cat([R_tr[i] for i in idx])
        coef = solve(y, zc, d1, d2)
        boots.append(convert(coef, s_geo, b_geo, zc, d1))
    res["bootstrap_train_storms"] = {k: [float(np.percentile([b[k] for b in boots if b[k] is not None], q)) for q in (2.5, 50, 97.5)]
                                    for k in ("c0_km_min", "c1_km_min_per_K", "nu_m2_s", "c_eff_km_min")}
    json.dump(res, open(out / "phys.json", "w"), indent=1)
    for tag in ("train", "test"):
        v = res[tag]
        print(f"[phys] {tag}: R2 {v['R2']:.3f}; c(theta') = {v['c0_km_min']:+.3f} {v['c1_km_min_per_K']:+.4f} theta' km/min "
              f"(theta' in K); c_eff {v['c_eff_km_min']:+.3f} km/min at weighted theta' {v['theta_weighted_K']:+.2f} K; "
              f"density-current c1 {v['density_current_c1_km_min_per_K']}; nu {v['nu_m2_s']:.0f} m2/s; "
              f"source {v['source_K_per_min']:+.4f} K/min", flush=True)
    b = res["bootstrap_train_storms"]
    for k, v in b.items():
        print(f"[phys] bootstrap over storms {k}: median {v[1]:+.4g}, 95% CI ({v[0]:+.4g}, {v[2]:+.4g})")
    print(f"[phys] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
