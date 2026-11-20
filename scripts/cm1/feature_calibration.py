"""9d: calibrate the updraft / downdraft feature definitions for the cross-entropy (feature-mask) loss.

For a sample of training storms (physical units) this reports
  - quantiles of w over the whole domain and over the storm region (columns with condensate),
  - the fraction of points above candidate updraft thresholds and below candidate downdraft thresholds,
  - the same after a minimum-width rule (LeMone and Zipser 1980: a convective core is a contiguous run of w beyond the
    threshold at least 500 m wide = 2 columns at 250 m),
  - the fraction by height,
and, for a trained run, the reconstruction error of w near features (lag 0), which sets the softness of the sigmoid
that turns the decoded w into a probability (a decoded value within one reconstruction error of the threshold should
not be treated as certain).

  PYTHONPATH=. python scripts/cm1/feature_calibration.py --cache ~/cm1_cache_B --n 30 --out feature_calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import VARS, load_member


def min_width(mask: np.ndarray, k: int) -> np.ndarray:
    """Keep only runs of True along the last axis that are at least k long (binary opening with a length-k segment)."""
    if k <= 1:
        return mask
    m = mask.astype(np.int8)
    c = np.cumsum(np.pad(m, [(0, 0)] * (m.ndim - 1) + [(1, 0)]), axis=-1)
    full = (c[..., k:] - c[..., :-k]) == k                       # windows of k entirely inside a run
    out = np.zeros_like(mask)
    for s in range(k):                                          # dilate back by k-1
        out[..., s:s + full.shape[-1]] |= full
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30, help="members to sample (spread over the caches)")
    ap.add_argument("--t0", type=int, default=120, help="first frame used (skip the spin-up)")
    ap.add_argument("--stride", type=int, default=5, help="frame stride")
    ap.add_argument("--out", default="feature_calibration.json")
    a = ap.parse_args()
    files = sorted(f for c in a.cache for f in Path(c).expanduser().glob("*.npz"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in sorted(rng.choice(len(files), min(a.n, len(files)), replace=False))]
    iw, il, ii = VARS.index("w"), VARS.index("qliq"), VARS.index("qice")
    W, Q = [], []
    for f in files:
        m = load_member(f, mmap=True)
        X = np.asarray(m.X[a.t0::a.stride], dtype=np.float32)       # (T, NX, V, Z)
        W.append(X[:, :, iw, :].transpose(0, 2, 1))                # (T, Z, NX)
        Q.append((X[:, :, il, :] ** 2 + X[:, :, ii, :] ** 2).transpose(0, 2, 1))   # stored as sqrt(q): q in g/kg
    w, q = np.concatenate(W), np.concatenate(Q)
    storm = (q.max(axis=1, keepdims=True) > 0.01).repeat(w.shape[1], axis=1)   # columns with any condensate
    rep = {"members": [f.name for f in files], "points": int(w.size), "storm_fraction": float(storm.mean())}
    qs = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    rep["w_quantiles_domain"] = dict(zip(map(str, qs), np.quantile(w, qs).round(3).tolist()))
    rep["w_quantiles_storm"] = dict(zip(map(str, qs), np.quantile(w[storm], qs).round(3).tolist()))
    up, down = {}, {}
    for t in (0.5, 1.0, 2.0, 3.0, 5.0):
        mu, md = w > t, w < -t
        up[str(t)] = {"frac": float(mu.mean()), "frac_storm": float(mu[storm].mean()), "frac_minwidth2": float(min_width(mu, 2).mean())}
        down[str(t)] = {"frac": float(md.mean()), "frac_storm": float(md[storm].mean()), "frac_minwidth2": float(min_width(md, 2).mean())}
    rep["updraft"], rep["downdraft"] = up, down
    rep["frac_by_level_up1"] = (w > 1.0).mean(axis=(0, 2)).round(4).tolist()
    rep["frac_by_level_down1"] = (w < -1.0).mean(axis=(0, 2)).round(4).tolist()
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("frac_by") and k != "members"}, indent=1))


if __name__ == "__main__":
    main()
