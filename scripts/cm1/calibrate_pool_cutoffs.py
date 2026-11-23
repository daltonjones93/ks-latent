"""10b: per-feature, per-lead vote cutoffs for a pooled ensemble, chosen on the VALIDATION storms (user, 2026-11-23: "calibrate for each one
of these videos, how many votes lead to the best performance... then rerun the visualizations").

Input: ensemble_design.py output on --split val (vote histograms k_n / k_h per pool size, lead and feature). For the pool of n models
(all S = n x members members) K = argmax_K CSI(present if >= K of S). Optionally scores the chosen K on a test-split design file.
Output: {"members": S, "n_models": n, "cutoffs": {feature: {lead: K}}, "val_csi": ..., "val_bias": ...} (make_vote_anim reads it).

  python scripts/cm1/calibrate_pool_cutoffs.py --val design/pool_val.json [--test design/now_test.json] --out vote_cutoffs_pool4.json
"""
import argparse
import json

import numpy as np


def curve(v):
    n, h = np.array(v["k_n"]), np.array(v["k_h"]); Y = h.sum()
    hit = np.cumsum(h[::-1])[::-1]; pa = np.cumsum(n[::-1])[::-1]          # forecasts / hits with count >= K, K = 0..S
    return hit / np.maximum(pa + Y - hit, 1e-9), pa / max(Y, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--test", default=None); ap.add_argument("--n-models", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.val)); n = a.n_models or d["n_models"]; S = n * d["members_per_model"]
    lags = d["config"]["lags"].split(","); feats = d["features"]
    t = json.load(open(a.test)) if a.test else None
    out = {"members": S, "n_models": n, "models": d["config"]["models"], "split": "val", "cutoffs": {}, "val_csi": {}, "val_bias": {}, "test": {}}
    print(f"pool of {n} models, S = {S} members; K chosen on {d['n_cases']} validation cases")
    print("feature " + " ".join(f"{'+' + L:>13s}" for L in lags))
    for f in feats:
        out["cutoffs"][f] = {}; out["val_csi"][f] = {}; out["val_bias"][f] = {}; row = []
        for L in lags:
            csi, bias = curve(d["votes"][f"pool|{n}|{L}|{f}"])
            K = int(np.argmax(csi[1:]) + 1)
            out["cutoffs"][f][L] = K; out["val_csi"][f][L] = float(csi[K]); out["val_bias"][f][L] = float(bias[K])
            row.append(f"K={K:2d} {csi[K]:.2f}/{bias[K]:.2f}")
            if t is not None and f"pool|{n}|{L}|{f}" in t["votes"]:
                tc, tb = curve(t["votes"][f"pool|{n}|{L}|{f}"])
                out["test"].setdefault(f, {})[L] = {"csi": float(tc[K]), "bias": float(tb[K]), "best_csi": float(tc[1:].max()),
                                                   "best_K": int(np.argmax(tc[1:]) + 1)}
        print(f"{f:7s} " + " ".join(f"{r:>13s}" for r in row))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
