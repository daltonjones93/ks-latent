"""10b: the animation's voting rule (docs/fpe_report/data/vote_cutoffs.json: K of 8 per feature and lead, chosen on the validation storms for F6's
8 members) applied to pooled / larger ensembles from ensemble_design.py: the cutoff is kept as a FRACTION of the members, K = ceil(K8 / 8 x S).
Prints CSI and area ratio per feature and lead for pool n = 1..N against the single model with the same number of members.

  python scripts/cm1/vote_pool_report.py --design design/test_T1.0_I0.0.json --cutoffs docs/fpe_report/data/vote_cutoffs.json
"""
import argparse
import json
import math

import numpy as np

EK = {"w+": "w+", "w-": "w-"}


def csi_bias(v, K):
    n, h = np.array(v["k_n"]), np.array(v["k_h"]); Y = h.sum()
    hit = h[K:].sum(); pa = n[K:].sum()
    return hit / (pa + Y - hit), pa / max(Y, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True); ap.add_argument("--cutoffs", required=True)
    a = ap.parse_args()
    d = json.load(open(a.design)); tab = json.load(open(a.cutoffs))["cutoffs"]
    M, N = d["members_per_model"], d["n_models"]
    feats = d["features"]; lags = d["config"]["lags"].split(",")
    print(f"{d['n_cases']} cases; members per model {M}; models {N}; noise T={d['config']['temp']} IC={d['config']['ic_noise']}")
    rows = {}
    for kd in ("pool", "single"):
        for nn in range(1, N + 1):
            S = nn * M
            for L in lags:
                for f in feats:
                    key = f"{kd}|{nn}|{L}|{f}"
                    if key not in d["votes"]:
                        continue
                    t = tab[f]; k8 = t[L] if L in t else t[min(t, key=lambda q: abs(int(q) - int(L)))]
                    K = max(1, min(S, math.ceil(k8 / 8.0 * S)))
                    rows[(kd, nn, L, f)] = csi_bias(d["votes"][key], K) + (K,)
    for L in lags:
        print(f"\n+{L} min: CSI (area ratio) of the animation's rule; pool of n models vs ONE model with the same members")
        print(f"{'feature':8s}" + "".join(f" | n={nn} pool        single      " for nn in range(1, N + 1)))
        for f in feats:
            print(f"{f:8s}" + "".join(f" | {rows[('pool', nn, L, f)][0]:.3f} ({rows[('pool', nn, L, f)][1]:.2f})  {rows[('single', nn, L, f)][0]:.3f} ({rows[('single', nn, L, f)][1]:.2f})"
                                       for nn in range(1, N + 1) if ("pool", nn, L, f) in rows))
        mean = lambda kd, nn: np.mean([rows[(kd, nn, L, f)][0] for f in feats])
        print(f"{'MEAN':8s}" + "".join(f" | {mean('pool', nn):.3f}              {mean('single', nn):.3f}" for nn in range(1, N + 1)))


if __name__ == "__main__":
    main()
