"""Figure 3: zero-shot transfer of the 1D KS model trained at L=100.

(a) D_KY vs L, ground truth vs the transferred model, with the extensive line 0.226 L.
(b) D_KY / L vs L for L >= 50, with the two additional training seeds.

Model: three initial conditions per size, standardized with the training statistics
(task A1, scripts/eval_1d_transfer.py); points are means, bars span the ICs.
Truth: every independent finite-time realization on disk (task A1/A5; distinct values
come from round-off-level trajectory differences); points are means, bars span them.
Source: artifacts/experiments/e1_model/summary_1d_transfer.json
(scripts/summarize_1d_transfer.py).
"""

import numpy as np

from _style import BLUE, COL2, EXP, INK, INK2, MUTED, load, panel_label, plt, save

S = load(EXP / "e1_model" / "summary_1d_transfer.json")
L = np.array([25, 50, 100, 200, 400])
key = lambda l: str(l)  # noqa: E731  (JSON keys are strings)
T = np.array([S["truth"][key(l)]["mean"] for l in L])
Terr = np.array([S["truth"][key(l)]["half_range"] for l in L])
M = np.array([S["model"][key(l)]["mean"] for l in L])
Mlo = np.array([S["model"][key(l)]["min"] for l in L])
Mhi = np.array([S["model"][key(l)]["max"] for l in L])
rho = 0.226

fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.45), gridspec_kw={"wspace": 0.32})

Lline = np.linspace(0, 420, 50)
a.plot(Lline, rho * Lline, ls="--", lw=0.8, color=MUTED, zorder=1)
a.text(330, rho * 330 - 13, r"$0.226\,L$", color=MUTED, fontsize=7.5, ha="center")
a.errorbar(L, T, yerr=Terr, fmt="o", mfc="white", mec=INK, ecolor=INK, mew=1.1, ms=6, capsize=0,
           elinewidth=0.8, label="ground truth", zorder=3)
a.errorbar(L, M, yerr=[M - Mlo, Mhi - M], fmt="o", color=BLUE, ms=4, capsize=0, elinewidth=0.9,
           label="transferred model", zorder=4)
a.axvline(100, color=INK2, lw=0.6, ls=":")
a.text(106, 60, "trained\nsize", fontsize=7, color=INK2, va="top")
a.annotate("$L=25$: outside validity\n(4 sites < 7-site kernel)", xy=(25, 1.5), xytext=(150, 9), fontsize=7,
           color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=4))
a.set_xlabel("domain length $L$")
a.set_ylabel(r"$D_{\mathrm{KY}}$")
a.set_xlim(0, 420)
a.set_ylim(-3, 100)
a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
panel_label(a, "a")

Lb = L[L >= 50]
idx = L >= 50
b.errorbar(Lb, T[idx] / Lb, yerr=Terr[idx] / Lb, fmt="o-", mfc="white", mec=INK, color=INK, ecolor=INK,
           mew=1.1, ms=6, lw=0.8, capsize=0, elinewidth=0.8, label="ground truth")
b.errorbar(Lb, M[idx] / Lb, yerr=[(M - Mlo)[idx] / Lb, (Mhi - M)[idx] / Lb], fmt="o-", color=BLUE, ms=4,
           lw=1.2, capsize=0, elinewidth=0.9, label="model, seed 0 (3 ICs)")
first = True
for l, vals in S["model_seeds"].items():
    l = int(l)
    v = [x / l for x in vals.values()]
    b.plot([l * 1.04] * len(v), v, "D", color=BLUE, mfc="white", ms=4, mew=0.9,
           label="model, training seeds 1–2" if first else None)
    first = False
b.set_xscale("log", base=2)
b.set_xticks(Lb)
b.set_xticklabels([str(l) for l in Lb])
b.minorticks_off()
b.set_ylim(0.212, 0.232)
b.set_xlabel("domain length $L$")
b.set_ylabel(r"$D_{\mathrm{KY}}/L$")
h, lab = b.get_legend_handles_labels()
order = [lab.index("ground truth"), lab.index("model, seed 0 (3 ICs)"), lab.index("model, training seeds 1–2")]
b.legend([h[i] for i in order], [lab[i] for i in order], loc="lower right")
panel_label(b, "b")

save(fig, "fig03_1d_transfer", ["artifacts/experiments/e1_model/summary_1d_transfer.json"])
