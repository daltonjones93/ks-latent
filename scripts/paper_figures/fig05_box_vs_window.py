"""Figure 5: the training distribution decides transfer.

Relative D_KY error (linear scale, broken between 42% and 88%) of models trained on a small periodic box,
box-trained models with added cross-patch coupling, models trained on windows
of a large domain (same architecture as the box models), and the canonical
recipe (per-patch c=6 autoencoder + long free rollouts). Bars: mean over
initial-condition seeds; dots: individual seeds.

Truth references: paper Table II, read from the run JSONs via _truths.py (means over
runs; Kolmogorov alpha=0.05 at 1x scored per IC against the matched-IC runs). The
alpha=0.1 canonical model is the one chosen by the truth-free rule of task A10
(_style.MAIN_A01). All model evaluations use normalized initial conditions
except the alpha=0.1 box-trained 3x value (raw IC; the only one available).
"""

import re

import numpy as np

from _style import AQUA, BLUE, COL2, EXP, INK2, ORANGE, YELLOW, load, panel_label, plt, save, MAIN_A01

inputs: list[str] = []


def dky(path):
    inputs.append(str(path.relative_to(EXP.parents[1])))
    return load(path)["model_D_KY"]


def crop(tag, scale, seeds):
    return [dky(EXP / f"crop/{tag}_x{scale}_seed{s}.json") for s in seeds]


import numpy as np  # noqa: E402

from _truths import runs as truth_runs  # noqa: E402


def tmean(system, size):
    rs = truth_runs(system, size)
    inputs.extend(r["file"] for r in rs)
    return float(np.mean([r["D_KY"] for r in rs]))


TRUTH_KOLM = {0.1: {s: tmean("kolm0.1", s) for s in (1, 2, 3)}, 0.05: {s: tmean("kolm0.05", s) for s in (2, 3)}}
MATCHED_005_X1 = [r["D_KY"] for r in truth_runs("kolm0.05", 1)]  # IC seeds 0, 1, 2 (matched ICs)
T_KS = {1.0: tmean("ks2d", 1.0), 1.5: tmean("ks2d", 1.5)}

CATS = [("canonical recipe", BLUE), ("window-trained", ORANGE),
        ("box-trained", AQUA), ("box-trained + coupling", YELLOW)]

panels = []

# 2D KS
ks = {"44$^2$": {}, "66$^2$": {}}
ks["44$^2$"]["box-trained"] = ([dky(EXP / f"icnorm/ks2d_x1_normalized_seed{s}.json") for s in (0, 1, 2)], T_KS[1.0])
ks["66$^2$"]["box-trained"] = ([dky(EXP / "icnorm/ks2d_x1.5_normalized.json"),
                                dky(EXP / "icnorm/ks2d_x1.5_normalized_seed1.json"),
                                dky(EXP / "icnorm/ks2d_x1.5_normalized_seed2.json")], T_KS[1.5])
ks["44$^2$"]["window-trained"] = ([dky(EXP / "crop/p7_ks2d_x1.0_seed0.json")], T_KS[1.0])
ks["66$^2$"]["window-trained"] = ([dky(EXP / "crop/p7_ks2d_x1.5_seed0.json")], T_KS[1.5])
ks["44$^2$"]["canonical recipe"] = ([dky(EXP / f"crop/p7b_ks2d_x1.0_seed{s}.json") for s in (0, 1)], T_KS[1.0])
ks["66$^2$"]["canonical recipe"] = ([dky(EXP / f"crop/p7b_ks2d_x1.5_seed{s}.json") for s in (0, 1)], T_KS[1.5])
panels.append(("2D KS", ks))

# Kolmogorov alpha = 0.1
k1 = {f"{s}$\\times$": {} for s in (1, 2, 3)}
T = TRUTH_KOLM[0.1]
k1["1$\\times$"]["box-trained"] = ([dky(EXP / "c4/s233_a01_x1_m0.json")], T[1])
k1["2$\\times$"]["box-trained"] = ([dky(EXP / "icnorm/s233_x2_normalized.json")], T[2])
k1["3$\\times$"]["box-trained"] = ([dky(EXP / "g3_kolmogorov_alpha_transfer_section233_3x.json")], T[3])
for sc in (1, 2, 3):
    k1[f"{sc}$\\times$"]["window-trained"] = (crop("p1_a01", sc, (0, 1, 2)), T[sc])
    k1[f"{sc}$\\times$"]["canonical recipe"] = (crop(MAIN_A01, sc, (0, 1, 2)), T[sc])
panels.append((r"Kolmogorov, $\alpha=0.1$", k1))

# Kolmogorov alpha = 0.05 (1x scored per IC against matched truths)
k5 = {f"{s}$\\times$": {} for s in (1, 2, 3)}
T = TRUTH_KOLM[0.05]
k5["1$\\times$"]["box-trained"] = ([dky(EXP / "c4/s231_a005_x1_m0.json")], [MATCHED_005_X1[0]])
k5["2$\\times$"]["box-trained"] = ([dky(EXP / "icnorm/s231_x2_normalized.json")], T[2])
coupling2 = ([dky(EXP / f"icnorm/d2fix_x2_normalized_seed{s}.json") for s in (0, 1, 2)]
             + [dky(EXP / "icnorm/d3_x2_normalized.json")]
             + [dky(EXP / f"icnorm/c9ii_x2_seed{s}.json") for s in (0, 1, 2)])
k5["2$\\times$"]["box-trained + coupling"] = (coupling2, T[2])
k5["3$\\times$"]["box-trained + coupling"] = ([dky(EXP / f"icnorm/c9ii_x3_seed{s}.json") for s in (0, 1, 2)], T[3])
k5["1$\\times$"]["window-trained"] = (crop("p2_a005", 1, (0, 1, 2)), MATCHED_005_X1)
k5["1$\\times$"]["canonical recipe"] = (crop("c19_e0d0_c6", 1, (0, 1, 2)), MATCHED_005_X1)
for sc in (2, 3):
    k5[f"{sc}$\\times$"]["window-trained"] = (crop("p2_a005", sc, (0, 1, 2)), T[sc])
    k5[f"{sc}$\\times$"]["canonical recipe"] = (crop("c19_e0d0_c6", sc, (0, 1, 2)), T[sc])
panels.append((r"Kolmogorov, $\alpha=0.05$", k5))


def errors(vals, truth):
    t = np.broadcast_to(np.asarray(truth, float), (len(vals),))
    return 100 * np.abs(np.asarray(vals) - t) / t


# Linear error axis with a break: the lower axes show 0-42%, the upper strip 88-102%, where
# only the box-trained + coupling models land. No floor is needed on a linear axis.
fig, axes = plt.subplots(2, 3, figsize=(COL2, 2.75), sharex="col", sharey="row",
                         gridspec_kw={"width_ratios": [2, 3, 3], "height_ratios": [1, 3.2],
                                      "wspace": 0.08, "hspace": 0.06})
width = 0.2
for col, ((title, groups), letter) in enumerate(zip(panels, "abc")):
    top, ax = axes[0, col], axes[1, col]
    for gi, (size, entries) in enumerate(groups.items()):
        present = [c for c in CATS if c[0] in entries]
        offsets = (np.arange(len(present)) - (len(present) - 1) / 2) * width
        for (cat, color), off in zip(present, offsets):
            vals, truth = entries[cat]
            e = errors(vals, truth)
            for axx in (top, ax):
                axx.bar(gi + off, e.mean(), width * 0.92, color=color, zorder=2)
                axx.plot(np.full(len(e), gi + off), e, "o", ms=2.6, mfc="white", mec=INK2, mew=0.6, zorder=3,
                         clip_on=True)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups))
    top.set_title(title)
    ax.set_ylim(0, 42)
    top.set_ylim(88, 102)
    for axx in (top, ax):
        axx.grid(axis="x", visible=False)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", length=0)
    # break marks
    kw = dict(color=INK2, clip_on=False, lw=0.6)
    for axx, y in ((top, 0), (ax, 1)):
        axx.plot([-0.012, 0.012], [y - 0.025 * (3.2 if y == 0 else 1), y + 0.025 * (3.2 if y == 0 else 1)],
                 transform=axx.transAxes, **kw)
    panel_label(top, letter)
axes[1, 0].set_ylabel(r"relative error in $D_{\mathrm{KY}}$ (%)")
axes[1, 0].yaxis.set_label_coords(-0.3, 0.62)
axes[0, 0].set_yticks([90, 100])
axes[1, 0].set_yticks([0, 10, 20, 30, 40])
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CATS]
fig.legend(handles, [n for n, _ in CATS], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))

save(fig, "fig05_box_vs_window", inputs)
