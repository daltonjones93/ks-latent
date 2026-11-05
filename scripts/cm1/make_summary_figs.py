"""Figures for docs/cm1_summary/cm1_summary.tex, from the saved CM1 analysis and run results.

New figures: learning curve (members), sampling (1- vs 5-min output), forecast comparison of the
trained models. Existing figures are copied in. Inputs: artifacts/cm1/*.json, artifacts/cm1/runs/*.
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts" / "cm1"
OUT = ROOT / "docs" / "cm1_summary" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
BLUE, ORANGE, TEAL, PURPLE, GRAY = "#2a6fdb", "#d9822b", "#1b9e77", "#8e6bd1", "#888888"

# existing figures
for src, dst in [("fig1_sections.png", "sections.png"), ("fig2_hovmoller.png", "hovmoller.png"),
                 ("fig4_locality.png", "locality.png"),
                 ("runs/cashew_ens_A_cpu_k45/fig_sections.png", "forecast_sections_k45.png")]:
    shutil.copy(A / src, OUT / dst)

# learning curve
lc = json.loads((A / "learning_curve.json").read_text())["rows"]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for model, col in (("ridge", BLUE), ("mlp", ORANGE)):
    for mir, ls, mk in ((False, "-", "o"), (True, "--", "s")):
        ks = sorted({r["k"] for r in lc})
        m = [np.mean([r[model]["active"] for r in lc if r["k"] == k and r["mirror"] == mir]) for k in ks]
        s = [np.std([r[model]["active"] for r in lc if r["k"] == k and r["mirror"] == mir]) for k in ks]
        lab = {"ridge": "linear", "mlp": "neural net"}[model] + (" + mirrored members" if mir else "")
        ax.errorbar(ks, m, yerr=s, fmt=ls + mk, color=col, ms=4, lw=1.8 if not mir else 1.2,
                    mfc=col if not mir else "none", capsize=2, label=lab)
ax.set_xlabel("training members"); ax.set_ylabel("one-step skill vs persistence\n(storm columns, held-out members)")
ax.set_xticks([1, 2, 3, 5, 7]); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7)
fig.savefig(OUT / "learning_curve.pdf"); plt.close(fig)

# sampling: 1-min vs 5-min output
st = json.loads((A / "sampling_test.json").read_text())
dts = sorted(int(k) for k in st["by_dt_min"])
D = np.array([st["by_dt_min"][str(d)]["D"] for d in dts])
rho = np.array([st["by_dt_min"][str(d)]["rho_successive"] for d in dts])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
ax[0].plot(dts, D / D[dts.index(5)], "-o", color=BLUE, lw=2, ms=4, label="CM1 storm columns")
dd = np.linspace(1, 10, 50)
ax[0].plot(dd, (dd / 5) ** 2, ":", color=GRAY, label=r"smooth: $\propto\Delta^2$")
ax[0].set_xlabel("interval Δ (min)"); ax[0].set_ylabel(r"$E|x(t+\Delta)-x(t)|^2$, relative to Δ = 5 min")
ax[0].set_ylim(0, 1.3); ax[0].legend(frameon=False, fontsize=7); ax[0].grid(alpha=0.25)
ax[1].axhline(0, color=GRAY, lw=0.8)
ax[1].axhline(-0.5, color=GRAY, lw=0.8, ls=":")
ax[1].text(10, -0.585, "−0.5: decorrelates within one interval", ha="right", fontsize=7, color="#555")
ax[1].set_ylim(-0.62, 0.42)
ax[1].plot(dts, rho, "-o", color=BLUE, lw=2, ms=4)
ax[1].plot([5], [st["original_5min_rho"]["30-90min"]], "D", color=ORANGE, ms=6, label="original 5-min files")
ax[1].set_xlabel("output interval Δ (min)"); ax[1].set_ylabel("correlation of successive changes")
ax[1].legend(frameon=False, fontsize=7, loc="upper right"); ax[1].grid(alpha=0.25)
fig.savefig(OUT / "sampling.pdf"); plt.close(fig)

# forecast comparison of trained models
runs = [("local_v1", "5-min, 7 members (Mac)", PURPLE), ("cashew_ens_A_cpu_v1", "1-min, 28 members, k ≤ 20", BLUE),
        ("cashew_ens_A_cpu_k45", "1-min, 28 members, k ≤ 45", TEAL)]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
for i, (name, lab, col) in enumerate(runs):
    fc = json.loads((A / "runs" / name / "results.json").read_text())["forecast"]
    lead = np.array(fc["lead_s"]) / 60
    for a_, k in zip(ax, ("nmse", "th_sfc_rmse", "front_err_km")):
        a_.plot(lead, fc[k], "-", color=col, lw=2, label=lab, marker="o" if len(lead) < 20 else None, ms=3)
    if name == "cashew_ens_A_cpu_k45":
        ax[0].plot(lead, fc["nmse_pers"], "--", color=GRAY, lw=1.3, label="persistence (1-min test set)")
        ax[0].plot(lead, fc["nmse_spread"], ":", color=ORANGE, lw=2, label="truth-member spread")
        ax[1].plot(lead, fc["th_sfc_rmse_pers"], "--", color=GRAY, lw=1.3)
        ax[2].plot(lead, fc["front_err_pers_km"], "--", color=GRAY, lw=1.3)
for a_, yl in zip(ax, ("normalized MSE, all variables", "surface θ′ RMSE (K)", "gust-front position error (km)")):
    a_.set_xlabel("lead time (min)"); a_.set_ylabel(yl); a_.grid(alpha=0.25)
ax[0].set_ylim(0, 0.8); ax[2].set_ylim(0, 8)
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.savefig(OUT / "forecast_comparison.pdf"); plt.close(fig)
print("figures in", OUT, sorted(p.name for p in OUT.iterdir()))
