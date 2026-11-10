"""FPe control: the same one-step library fitted directly to site-mean surface theta' (no autoencoder).

Compares with law_physical_units.py (channel-0 law converted to theta' units). Same storms, frames, interior sites,
centred-difference target and bootstrap over storms.  Library [1, dx th, th dx th, d2x th]  (sites 2 km, minutes).
  python scripts/cm1/law_theta_control.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/fpe_thctl
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402

SITE_KM = 2.0
NU_UNIT = (SITE_KM * 1000.0) ** 2 / 60.0


def fit(y, th, d1, d2):
    X = np.stack([np.ones_like(y), d1, th * d1, d2], 1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = float(1 - np.sum((y - X @ c) ** 2) / np.sum((y - y.mean()) ** 2))
    return c, r2


def phys(c, th, d1):
    k0, a1, a2, nu = c
    w = d1 ** 2
    return {"c0_km_min": float(-a1 * SITE_KM), "c1_km_min_per_K": float(-a2 * SITE_KM),
            "nu_m2_s": float(nu * NU_UNIT), "source_K_per_min": float(k0),
            "c_eff_km_min": float(np.sum(w * -(a1 + a2 * th)) / np.sum(w) * SITE_KM)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330"); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); t0 = time.time()
    torch.set_num_threads(2)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]; p = ae.cfg.patch
    files = member_files(a.source); f0, f1 = map(int, a.frames.split(":"))
    sc_all = np.asarray(norm.scale, float)

    def rows(names):
        R = []
        for n in names:
            d = ColumnData([load_member(files[n])], norm, dev)
            sc = np.repeat(sc_all, d.Z)[:, None]
            hi = min(f1, d.T[0] - 2)
            th = np.stack([(d.full(0, t)[0].cpu().numpy() * sc)[VARS.index("th") * d.Z].reshape(-1, p).mean(1)
                           for t in range(f0 - 1, hi + 1)])[:, :, None]
            y = (0.5 * (th[2:] - th[:-2]))[..., 0]; tc = th[1:-1, :, 0]
            d1, d2 = (x[..., 0] for x in site_derivs(th[1:-1], 2))
            sl = slice(halo, -halo)
            R.append(tuple(v[:, sl].ravel() for v in (y, tc, d1, d2)))
        return R
    Rtr, Rte = rows(split["train"][: a.n_train]), rows(split["test"][: a.n_test])
    cat = lambda Rs: tuple(np.concatenate([r[i] for r in Rs]) for i in range(4))
    res = {"run": a.run}
    ytr, ttr, d1tr, d2tr = cat(Rtr)
    c, r2tr = fit(ytr, ttr, d1tr, d2tr)
    yte, tte, d1te, d2te = cat(Rte)
    X = np.stack([np.ones_like(yte), d1te, tte * d1te, d2te], 1)
    r2te = float(1 - np.sum((yte - X @ c) ** 2) / np.sum((yte - yte.mean()) ** 2))
    res["train"] = {"R2": r2tr, **phys(c, ttr, d1tr)}; res["test_R2"] = r2te
    cte, r2te_own = fit(yte, tte, d1te, d2te); res["test_refit"] = {"R2": r2te_own, **phys(cte, tte, d1te)}
    rng = np.random.default_rng(0); B = []
    for _ in range(a.n_boot):
        i = rng.integers(0, len(Rtr), len(Rtr)); y, t, d1, d2 = cat([Rtr[j] for j in i])
        cb, _ = fit(y, t, d1, d2); B.append(phys(cb, t, d1))
    res["bootstrap"] = {k: [float(np.percentile([b[k] for b in B], q)) for q in (2.5, 50, 97.5)]
                        for k in ("c0_km_min", "c1_km_min_per_K", "nu_m2_s", "c_eff_km_min")}
    json.dump(res, open(out / "thctl.json", "w"), indent=1)
    print("[thctl] theta' stats: mean %.2f sd %.2f K" % (ttr.mean(), ttr.std()))
    print(f"[thctl] train R2 {r2tr:.3f} test R2 {r2te:.3f} (test refit {r2te_own:.3f})")
    print("[thctl]", {k: round(v, 4) for k, v in res["train"].items()})
    for k, v in res["bootstrap"].items():
        print(f"[thctl] bootstrap {k}: median {v[1]:+.4g}, 95% CI ({v[0]:+.4g}, {v[2]:+.4g})")
    print(f"[thctl] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
