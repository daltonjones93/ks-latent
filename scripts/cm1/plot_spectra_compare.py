"""8a: horizontal power spectra of the forecasts by scale band, relative to truth (from the twoscale_eval JSON files: 30 test
cases, 8 samples). Lines = models, one panel per variable and lead; 1 = the truth's power in that band.
  python scripts/cm1/plot_spectra_compare.py --out OUT.png --json F1 F2 ... --runs file_index:run_name=label ..."""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--json", nargs="+", required=True)
ap.add_argument("--runs", nargs="+", required=True, help="i:run=label (i = index into --json)")
ap.add_argument("--vars", nargs="+", default=["w", "th", "qliq"])
ap.add_argument("--lags", nargs="+", default=["30", "60"])
a = ap.parse_args()
J = [json.load(open(f)) for f in a.json]
bands = J[0]["bands_km"]; V = J[0]["vars"]
lab = [f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}" for lo, hi in bands]
LAB = {"w": "w", "th": "θ′", "qliq": "liquid", "u": "u′", "qice": "ice"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(len(a.lags), len(a.vars), figsize=(3.2 * len(a.vars), 2.9 * len(a.lags)), sharex=True, sharey=True, constrained_layout=True)
axs = np.atleast_2d(axs)
for r, L in enumerate(a.lags):
    for c, v in enumerate(a.vars):
        ax = axs[r, c]; iv = V.index(v)
        for spec in a.runs:
            i, rest = spec.split(":", 1); rn, lb = rest.split("=", 1)
            cs = [x["lags"][L] for x in J[int(i)]["runs"][rn]["cases"]]
            pr = np.mean([np.array(x["power"])[iv] for x in cs], 0) / np.mean([np.array(x["power_truth"])[iv] for x in cs], 0)
            ax.plot(range(len(lab)), pr, "o-", ms=3, lw=1.3, label=lb)
        ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=30)
        ax.set_title(f"{LAB.get(v, v)}, +{L} min")
        if c == 0: ax.set_ylabel("power / truth")
        if r == len(a.lags) - 1: ax.set_xlabel("wavelength band (km)")
axs[0, -1].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.suptitle("Forecast power spectra relative to truth (1 = perfect); small scales on the left")
fig.savefig(a.out, dpi=150); print("wrote", a.out)
