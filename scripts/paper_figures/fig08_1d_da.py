"""Figure 8 (task F1): 1D KS data assimilation at the trained size and after transfer.

Skill = free-run RMSE / analysis RMSE (above 1: assimilation helps) against ensemble size.
(a) Trained size L = 100 (earlier Phase C/D sweeps, 40 cycles, whole-run skill, observation
    spacing 3.125): global latent without localization and with SEC; local field without
    localization and with Gaspari-Cohn; particle flow filter.
(b) L = 200 and (c) L = 400: the L = 100 local model transferred without retraining, fixed
    observation spacing 3.125 (task D1), 100 cycles, steady-state skill (cycles 51-100). No
    localization vs Gaspari-Cohn for the particle flow filter (solid) and the stochastic EnKF
    (dashed). In (b) also SEC (task D3): PSD-projected with the PFF, and the serial SEC EnKF;
    crosses mark ensemble sizes at which SEC in matrix form failed (indefinite covariance).
Dotted vertical line: n_+ + 1, with n_+ the truth's number of positive exponents (Table III).
Sources: artifacts/da_sweep_phaseC2_global.csv, artifacts/da_sweep_phaseD3_local224.csv,
artifacts/da_pff_ltransfer_d1_*.json, artifacts/e3_sec_vs_gc_ltransfer_d3_*.json,
artifacts/experiments/e1_model/summary_1d_transfer.json (n_+).
"""

import csv
import json

import numpy as np

from _style import AQUA, ART, BLUE, COL2, EXP, INK2, MUTED, ORANGE, YELLOW, panel_label, plt, save

NS = [8, 16, 32, 64, 128, 256]
inputs = ["artifacts/da_sweep_phaseC2_global.csv", "artifacts/da_sweep_phaseD3_local224.csv"]


def csv_series(path, localizer):
    with open(path) as f:
        rs = [r for r in csv.DictReader(f) if r["localizer"] == localizer and r.get("status", "OK") == "OK"]
    pts = sorted((int(r["n_ensemble"]), float(r["skill_free_over_da"])) for r in rs)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def json_series(pattern):
    xs, ys = [], []
    for N in NS:
        p = ART / pattern.format(N=N)
        if p.exists():
            inputs.append(str(p.relative_to(ART.parent)))
            xs.append(N); ys.append(json.loads(p.read_text())["skill_steady"])
    return np.array(xs), np.array(ys)


S = json.loads((EXP / "e1_model/summary_1d_transfer.json").read_text())
inputs.append("artifacts/experiments/e1_model/summary_1d_transfer.json")


def nplus(L):
    """Range of the number of positive exponents over the truth runs at L (from their files)."""
    vals = []
    for rz in S["truth"][str(L)]["realizations"]:
        for f in rz["files"]:
            p = ART.parent / f
            if p.exists():
                d = json.loads(p.read_text())
                if "n_positive" in d:
                    vals.append(int(d["n_positive"]))
    return (min(vals), max(vals)) if vals else None


fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.75), sharey=True, gridspec_kw={"wspace": 0.08})
a, b, c = axes

a.plot(*csv_series(ART / "da_sweep_phaseC2_global.csv", "none"), "^-", color=MUTED, mfc="white", label="global, none")
a.plot(*csv_series(ART / "da_sweep_phaseC2_global.csv", "sec"), "v--", color=AQUA, label="global + SEC")  # M10: shape+fill+dash, not hue alone
a.plot(*csv_series(ART / "da_sweep_phaseD3_local224.csv", "none"), "v-", color=ORANGE, label="local, none")
a.plot(*csv_series(ART / "da_sweep_phaseD3_local224.csv", "gaspari_cohn"), "o-", color=BLUE, label="local + GC")
a.set_title(r"$L=100$ (trained)", fontsize=8)

for ax, L in ((b, 200), (c, 400)):
    ax.plot(*json_series(f"da_pff_ltransfer_d1_pff_L{L}_none_nens{{N}}.json"), "v-", color=ORANGE, label="none, GN-PFF")
    ax.plot(*json_series(f"da_pff_ltransfer_d1_pff_L{L}_gaspari_cohn_nens{{N}}.json"), "o-", color=BLUE,
            label="GC, GN-PFF")
    ax.plot(*json_series(f"da_pff_ltransfer_d1_letkf_L{L}_none_nens{{N}}.json"), "^--", color=ORANGE, mfc="white",
            lw=1.0, label="none, EnKF")
    ax.plot(*json_series(f"da_pff_ltransfer_d1_letkf_L{L}_gaspari_cohn_nens{{N}}.json"), "s--", color=BLUE,
            mfc="white", lw=1.0, label="GC, EnKF")
    lo, hi = nplus(L)
    ax.axvspan(lo + 1, hi + 1, color="#d9d8d3", zorder=0) if hi > lo else None
    ax.axvline(lo + 1, color=INK2, lw=0.7, ls=":")
    ax.text((hi + 1) * 1.07, 3.6, rf"$n_+{{+}}1$" + "\n" + rf"$={lo + 1}$–${hi + 1}$", fontsize=7, color=INK2,
            va="top")
    ax.set_title(rf"$L={L}$", fontsize=8)

b.plot(*json_series("e3_sec_vs_gc_ltransfer_d3_L200_pff_sec_psd_N{N}.json"), "D-", color=AQUA, ms=4,
       label="SEC (PSD), GN-PFF")
b.plot(*json_series("e3_sec_vs_gc_ltransfer_d3_L200_serial_sec_enkf_none_N{N}.json"), "D--", color=YELLOW, ms=4,
       mfc="white", label="SEC, serial EnKF")
failed = [N for N in NS if not (ART / f"e3_sec_vs_gc_ltransfer_d3_L200_pff_sec_N{N}.json").exists()]
b.plot(failed, np.full(len(failed), 0.42), "x", color=AQUA, ms=5, mew=1.3, label="SEC (matrix) failed")
h, lab = b.get_legend_handles_labels()
ha_, la_ = a.get_legend_handles_labels()
keep = [i for i, l in enumerate(la_) if l.startswith("global")]  # local curves in (a) share the PFF styles
h, lab = [ha_[i] for i in keep] + h, [la_[i] for i in keep] + lab
fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.6, 0.0), handlelength=1.8,
           columnspacing=1.0)

for ax, letter in zip(axes, "abc"):
    ax.axhline(1.0, color=INK2, lw=0.7, ls=":", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel(r"ensemble size $N_{\mathrm{ens}}$")
    panel_label(ax, letter)
a.set_ylabel("skill (free-run / analysis RMSE)")
a.set_ylim(0.3, 3.7)
c.text(8.5, 1.05, "above: DA helps", fontsize=7, color=INK2, va="bottom", ha="left")

save(fig, "fig08_1d_da", inputs)
