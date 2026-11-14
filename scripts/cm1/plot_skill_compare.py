"""8a/8b: skill by lead and by variable for the best models, from the twoscale_eval JSON files (30 test cases, 8 samples).
  python scripts/cm1/plot_skill_compare.py --out F.png --json F1 F2 --runs i:run=label ... --ref i:run"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--json", nargs="+", required=True)
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--ref", required=True)
a = ap.parse_args()
J = [json.load(open(f)) for f in a.json]; V = J[0]["vars"]; lags = [str(L) for L in (5, 15, 30, 60)]
def cases(spec):
    i, rn = spec.split(":", 1); return J[int(i)]["runs"][rn]["cases"]
def metric(cs, L, key, agg=np.mean):
    return agg([np.array(c["lags"][L][key]) for c in cs], 0)
ref = cases(a.ref)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 4, figsize=(14, 3.3), constrained_layout=True)
for spec in a.runs:
    lhs, lab = spec.split("=", 1); cs = cases(lhs)
    cr = [metric(cs, L, "crps").mean() / metric(ref, L, "crps").mean() for L in lags]
    rm = [metric(cs, L, "rmse").mean() for L in lags]
    sp = [metric(cs, L, "spread").mean() / metric(cs, L, "rmse").mean() for L in lags]
    axs[0].plot([int(L) for L in lags], (np.array(cr) - 1) * 100, "o-", ms=3, label=lab)
    axs[1].plot([int(L) for L in lags], rm, "o-", ms=3, label=lab)
    axs[2].plot([int(L) for L in lags], sp, "o-", ms=3, label=lab)
    pv = metric(cs, "30", "crps") / metric(ref, "30", "crps")
    axs[3].plot(range(len(V)), (pv - 1) * 100, "o-", ms=3, label=lab)
axs[0].axhline(0, color="k", lw=0.8, ls="--"); axs[0].set_title("CRPS vs the original model (%, lower is better)"); axs[0].set_xlabel("lead (min)")
axs[1].set_title("RMSE of the sample mean (normalized)"); axs[1].set_xlabel("lead (min)")
axs[2].axhline(1, color="k", lw=0.8, ls="--"); axs[2].set_title("spread / RMSE (1 = calibrated)"); axs[2].set_xlabel("lead (min)")
axs[3].axhline(0, color="k", lw=0.8, ls="--"); axs[3].set_xticks(range(len(V))); axs[3].set_xticklabels(V); axs[3].set_title("CRPS by variable at +30 min vs original (%)")
axs[0].legend(fontsize=6.5, frameon=False)
fig.savefig(a.out, dpi=150); print("wrote", a.out)
