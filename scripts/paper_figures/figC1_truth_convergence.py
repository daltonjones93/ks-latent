"""Appendix C figure (task C8): running Kaplan-Yorke dimension of every ground-truth run that
recorded it. Kolmogorov 120-unit runs (crop/truth_*_T120.json): how far a 40-unit estimate
(dotted line) sits from the 120-unit value. 2D KS 40-unit runs (truth_ks2d/*.json): drift over
the last 20 time units. One panel per (system, size); runs that land later (task A3) appear
automatically. Colors distinguish runs within a panel only (IC seed), and every run is also
labelled in the legend.
"""

import glob
import json
from pathlib import Path

from _style import AQUA, BLUE, COL2, EXP, INK2, ORANGE, ROOT, panel_label, plt, save

SEED_COL = {0: BLUE, 1: ORANGE, 2: AQUA}
groups, inputs = {}, []
for f in sorted(glob.glob(str(EXP / "crop/truth_*_T120.json"))):
    d = json.loads(Path(f).read_text())
    if d.get("running_D_KY_by_time"):
        groups.setdefault((f"Kolm. $\\alpha={d['alpha']:g}$, ${d['scale']}\\times$", 40.0), []).append(d)
        inputs.append(str(Path(f).relative_to(ROOT)))
for f in sorted(glob.glob(str(EXP / "truth_ks2d/truth_ks2d_*.json"))):
    d = json.loads(Path(f).read_text())
    groups.setdefault((f"2D KS, ${d['L']:g}^2$", None), []).append(d)
    inputs.append(str(Path(f).relative_to(ROOT)))

n = len(groups)
ncol = 3
nrow = -(-n // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(COL2, 1.95 * nrow), squeeze=False,
                         gridspec_kw={"hspace": 0.62, "wspace": 0.32})
for ax, ((title, mark), runs), letter in zip(axes.ravel(), groups.items(), "abcdefghij"):
    for d in sorted(runs, key=lambda d: d["seed"]):
        r = sorted((float(t), v) for t, v in d["running_D_KY_by_time"].items())
        t, v = zip(*r)
        ax.plot(t, v, "-" + {0: "o", 1: "s", 2: "^"}.get(d["seed"], "o"), ms=2.8, lw=1.1, ls=("-", "--", ":")[d["seed"] % 3],
                color=SEED_COL.get(d["seed"], INK2), label=f"IC seed {d['seed']}")  # S6.1: marker and line style per seed
    if mark:
        ax.axvline(mark, color=INK2, lw=0.6, ls=":")
    ax.set_title(title, fontsize=7.5, pad=8)
    ax.set_xlabel("averaging time", labelpad=1)
    panel_label(ax, letter)
for ax in axes[:, 0]:
    ax.set_ylabel(r"running $D_{\mathrm{KY}}$")
for ax in axes.ravel()[n:]:
    ax.set_visible(False)

from matplotlib.lines import Line2D  # noqa: E402
fig.legend([Line2D([], [], color=c, marker="o", ms=2.6, lw=1.1) for c in SEED_COL.values()],
           [f"IC seed {k}" for k in SEED_COL], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
save(fig, "figC1_truth_convergence", inputs)
