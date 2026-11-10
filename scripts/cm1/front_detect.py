"""Gust-front detectors for the squall line, calibrated against the measured cold-pool speed (-0.29 km/min, FPe-S).

All work on the site-mean (8-cell = 2 km) surface theta' profile. The storm moves toward -x, so the leading edge is on the
-x side of the cold pool.  Candidates:
  grad    : argmax |d theta'/dx| inside the halo (the first definition, 5j; failed)
  edge1   : from the deepest cold-pool site walk toward -x to the first site with theta' > -1 K
  edge05  : same with -0.5 K;   edge2: same with -2 K
Calibration (before any paired run): front speed from a linear fit of the front position over t0..t0+30 min on every
storm in the cache and t0 in {120, 180, 240}; report median, IQR and the fraction within 30% of -0.29 km/min.
  python scripts/cm1/front_detect.py --source cache:DIR[+DIR2...] --out ~/cm1_da/front_calib
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ks_latent.cm1.data import VARS, load_member  # noqa: E402

P_CELLS, SITE_KM = 8, 2.0


def site_theta(m, t, lev=0):
    th = m.X[t, :, VARS.index("th"), lev].astype(np.float64)
    return th.reshape(-1, P_CELLS).mean(1)


def detect(th, kind, halo=6):
    P = th.shape[0]
    lo, hi = halo, P - halo
    if kind == "min":                        # cold-pool core: the deepest site-mean theta' (calibration fallback, 5k)
        return int(np.argmin(np.where((np.arange(P) >= lo) & (np.arange(P) < hi), th, np.inf)))
    if kind == "grad":
        g = np.abs(np.gradient(th)); g[:lo] = 0; g[hi:] = 0
        return int(np.argmax(g))
    s = int(np.argmin(np.where((np.arange(P) >= lo) & (np.arange(P) < hi), th, np.inf)))
    thr = {"edge1": -1.0, "edge05": -0.5, "edge2": -2.0}[kind]
    j = s
    while j > lo and th[j] <= thr:
        j -= 1
    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in a.source[len("cache:"):].replace("+", ",").split(","):
        files += sorted(Path(d).expanduser().glob("*.npz"))
    if a.n: files = files[: a.n]
    kinds = ("grad", "edge05", "edge1", "edge2")
    res = {k: [] for k in kinds}
    for f in files:
        m = load_member(f)
        dt_min = m.dt / 60.0
        for t0 in [int(x) for x in a.t0.split(",")]:
            nf = int(round(30 / dt_min))
            if t0 + nf >= m.X.shape[0]: continue
            ths = [site_theta(m, t0 + i) for i in range(nf + 1)]
            tt = np.arange(nf + 1) * dt_min
            for k in kinds:
                x = np.array([detect(th, k) for th in ths], float)
                res[k].append({"member": m.name, "t0": t0, "speed": float(np.polyfit(tt, x, 1)[0] * SITE_KM)})
        print(m.name, flush=True)
    summ = {}
    for k in kinds:
        v = np.array([r["speed"] for r in res[k]])
        summ[k] = {"n": len(v), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                   "frac_within30pct_of_-0.29": float(np.mean(np.abs(v + 0.29) <= 0.3 * 0.29))}
        print(k, summ[k])
    json.dump({"summary": summ, "cases": res}, open(out / "front_calib.json", "w"), indent=1)


if __name__ == "__main__":
    main()
