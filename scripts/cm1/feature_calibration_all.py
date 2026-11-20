"""9e: threshold calibration for presence features of every CM1 variable (generalizes feature_calibration.py, which did w only).

Anomaly variables (u', theta', qv', p' as stored: perturbations from the base state) get two classes each (positive and negative
anomalies); condensate (liquid, ice; stored as sqrt(q)) gets one class (presence above a mixing ratio). For each variable this prints
quantiles, exceedance fractions at candidate thresholds (domain and storm region), and the fraction by height.

  PYTHONPATH=. python scripts/cm1/feature_calibration_all.py --cache ~/cm1_cache_B ~/cm1_cache_C ~/cm1_cache_D --n 30 --out cal_all.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import VARS, load_member

CAND = {"u": [2, 5, 8, 10, 15], "w": [1, 1.5, 2, 3, 5], "th": [0.5, 1, 2, 3, 5], "qv": [0.5, 1, 2, 3], "prs": [0.25, 0.5, 1, 2],
        "qliq": [0.01, 0.1, 0.5, 1, 2], "qice": [0.01, 0.05, 0.1, 0.5, 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--t0", type=int, default=120); ap.add_argument("--stride", type=int, default=5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for c in a.cache for f in Path(c).expanduser().glob("*.npz"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in sorted(rng.choice(len(files), min(a.n, len(files)), replace=False))]
    F = {v: [] for v in VARS}
    for f in files:
        X = np.asarray(load_member(f, mmap=True).X[a.t0::a.stride], dtype=np.float32)        # (T, NX, V, Z)
        for i, v in enumerate(VARS):
            x = X[:, :, i, :].transpose(0, 2, 1)
            F[v].append(x ** 2 if v in ("qliq", "qice") else x)                              # condensate stored as sqrt(q)
    F = {v: np.concatenate(x) for v, x in F.items()}
    storm = ((F["qliq"] + F["qice"]).max(axis=1, keepdims=True) > 0.01).repeat(F["w"].shape[1], axis=1)
    rep = {"members": [f.name for f in files], "storm_fraction": float(storm.mean())}
    qs = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    for v, x in F.items():
        r = {"quantiles": dict(zip(map(str, qs), np.quantile(x, qs).round(4).tolist())), "std": float(x.std())}
        for t in CAND[v]:
            r[f">{t}"] = {"frac": float((x > t).mean()), "frac_storm": float((x > t)[storm].mean()),
                          "by_level": (x > t).mean(axis=(0, 2))[::4].round(3).tolist()}
            if v not in ("qliq", "qice"):
                r[f"<-{t}"] = {"frac": float((x < -t).mean()), "frac_storm": float((x < -t)[storm].mean()),
                               "by_level": (x < -t).mean(axis=(0, 2))[::4].round(3).tolist()}
        rep[v] = r
    Path(a.out).expanduser().write_text(json.dumps(rep, indent=1))
    for v in VARS:
        print(v, "q:", rep[v]["quantiles"], "std", round(rep[v]["std"], 3))
        print("   ", {k: (round(x["frac"], 4), round(x["frac_storm"], 4)) for k, x in rep[v].items() if k[0] in "<>"})


if __name__ == "__main__":
    main()
