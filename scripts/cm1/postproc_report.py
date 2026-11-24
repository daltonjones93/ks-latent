"""10b: merge postproc_eval.py parts; table + figure of accuracy against realism for each post-processed single field.

  python scripts/cm1/postproc_report.py --parts DIR --out docs/fpe_report/data/postproc_test.json --fig docs/fpe_report/figs_pool/fig_postproc.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NICE = {"mean": "average", "member": "one member", "medoid": "closest member (4)", "align": "aligned average (1)",
        "align_local": "aligned average, local (1b)", "pm": "probability matched (2)", "pm_align": "prob. matched aligned avg (2+1)",
        "blend10": "avg >10 km + member <10 km (3)", "blend20": "avg >20 km + member <20 km (3)", "blend10_align": "avg >10 km + aligned <10 km (3+1)"}
COL = {"mean": "#2A78C2", "member": "#9AA0A6", "medoid": "#6B6B6B", "align": "#E07B39", "align_local": "#B8651E", "pm": "#B5446E",
       "pm_align": "#7A2E4A", "blend10": "#3A9A5B", "blend20": "#7BC08E", "blend10_align": "#1F6B3A"}


def merge(parts):
    tot = None
    for p in sorted(Path(parts).glob("part_*.json")):
        d = json.load(open(p))
        if tot is None:
            tot = d; continue
        for key, A in d["acc"].items():
            T = tot["acc"][key]
            for k in ("se", "cor", "pw", "pt"):
                T[k] = (np.array(T[k]) + np.array(A[k])).tolist()
            T["n"] += A["n"]
            for k in A["fss"]:
                T["fss"][k] = [x + y for x, y in zip(T["fss"][k], A["fss"][k])]
            for k in A["cont"]:
                T["cont"][k] = [x + y for x, y in zip(T["cont"][k], A["cont"][k])]
        tot["storms"] += d["storms"]
    return tot


def summarize(tot):
    S = {}
    for key, A in tot["acc"].items():
        n = A["n"]; pw, pt = np.array(A["pw"]), np.array(A["pt"])
        cont = {f: c[0] / max(sum(c), 1) for f, c in A["cont"].items()}
        bias = {f: (c[0] + c[2]) / max(c[0] + c[1], 1) for f, c in A["cont"].items()}
        S[key] = {"rmse": float(np.sqrt(np.array(A["se"]) / n).mean()), "rmse_var": np.sqrt(np.array(A["se"]) / n).tolist(),
                  "corr": float(np.mean(A["cor"]) / n), "power": (pw / pt).mean(0).tolist(),
                  "fss": {k: 1 - v[0] / max(v[1], 1e-12) for k, v in A["fss"].items()}, "csi": cont, "bias": bias,
                  "csi_mean": float(np.mean(list(cont.values()))), "cases": n}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    tot = merge(a.parts); S = summarize(tot)
    kinds, lags = tot["kinds"], tot["lags"]
    json.dump({"storms": tot["storms"], "kinds": kinds, "lags": lags, "bands_km": tot["bands_km"], "summary": S}, open(a.out, "w"), indent=1)
    for L in lags:
        m = S[f"mean|{L}"]["rmse"]
        print(f"\n+{L} min ({S[f'mean|{L}']['cases']} cases)  RMSE (vs average)   power 2-5 / 5-10 / 10-20 km   FSS w>3 @10km   liquid@10   mean CSI")
        for k in kinds:
            s = S[f"{k}|{L}"]
            print(f"  {NICE[k]:34s} {s['rmse']:.3f} ({100 * (s['rmse'] / m - 1):+5.1f}%)   {s['power'][1]:.2f} / {s['power'][2]:.2f} / {s['power'][3]:.2f}"
                  f"       {s['fss']['w3@10']:.3f}         {s['fss']['liq05@10']:.3f}     {s['csi_mean']:.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.28, left=0.05, right=0.8, top=0.86, bottom=0.13))
    show = [L for L in (15, 30, 60) if L in lags]
    for j, L in enumerate(show):
        for k in kinds:
            s = S[f"{k}|{L}"]; p = np.mean(s["power"][1:3])
            ax[j].scatter(p, s["rmse"], s=70 if k in ("mean", "member") else 45, color=COL[k], edgecolor="k", lw=0.5, zorder=3, label=NICE[k])
        ax[j].axvline(1, color="k", lw=0.8, ls=":")
        ax[j].set_xlabel("power / truth, 2-10 km (1 = realistic)"); ax[j].set_title(f"+{L} min")
        ax[j].set_xscale("log"); ax[j].set_xticks([0.1, 0.2, 0.5, 1]); ax[j].set_xticklabels(["0.1", "0.2", "0.5", "1"]); ax[j].minorticks_off()
    ax[0].set_ylabel("RMSE (normalized units, lower is better)")
    ax[-1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Accuracy against realism of single-field products from the 4-model pool (66 test cases). Best: low and near the dotted line",
                 fontsize=10)
    fig.savefig(a.fig, dpi=150); plt.close(fig)
    print("figure", a.fig)


if __name__ == "__main__":
    main()
