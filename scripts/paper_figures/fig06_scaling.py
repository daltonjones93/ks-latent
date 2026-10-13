"""Figure 6: D_KY scaling with domain size, ground truth vs transferred models.

Rows: D_KY; D_KY per base box; number of positive exponents n_+ per base box. Left column
2D KS (base box 44^2; sizes 44^2, 66^2, 88^2 grow both axes). Right column Kolmogorov flow
at three drags (base box 2pi x 2pi; the channel is widened in x only).

Marker semantics as in Fig. 3: open = ground truth (neutral ink), filled = model (system hue;
Kolmogorov drags by marker shape and lightness). Truth: mean over the runs in `_truths.py`
(paper Table II), bars spanning the runs where several exist; where a size rests on a single
run, a shaded +-15% band marks the run-to-run spread measured elsewhere (App. C). Models
(canonical recipe): mean over initial conditions, bars spanning them. 2D KS 88^2 has no truth.
"""

import numpy as np
from matplotlib.lines import Line2D

from _style import (BLUE, COL2, DRAG_COLOR, DRAG_MARKER, EXP, HUE, INK, MUTED, MAIN_A01, load, panel_label, plt,
                    save)
from _truths import runs as truth_runs

inputs: list[str] = []
BAND = 0.15
BAND_FC = "#e9e8e4"


def crop(tag, scale, seeds):
    out = []
    for t in ([tag] if isinstance(tag, str) else tag):
        for s in seeds:
            p = EXP / f"crop/{t}_x{scale}_seed{s}.json"
            if p.exists():
                inputs.append(str(p.relative_to(EXP.parents[1])))
                d = load(p)
                out.append((d["model_D_KY"], d["model_n_positive"]))
    return np.array(out)


# Paper I rewrite (2026-10-13): small-box models (route a) next to the window-trained ones (route b).
C2 = [f"r13_C2_ks2d_s44big_k48_seed{s}" for s in (0, 1, 2)]
SMALL_KOLM = {0.1: "r12_c6_a01_s1x_r3", 0.05: "r4k_c6_a005_s1x_r3"}


def truths(system, size):
    rs = truth_runs(system, size)
    inputs.extend(r["file"] for r in rs)
    return np.array([(r["D_KY"], r["n_positive"]) for r in rs])


def mark(ax, x, v, per_box, open_, color, marker, ms=5.5, label=None):
    v = np.asarray(v, float) / per_box
    m = v.mean()
    yerr = [[m - v.min()], [v.max() - m]] if len(v) > 1 else None
    kw = dict(mfc="white", mec=color, mew=1.1) if open_ else dict(mfc=color, mec=color, mew=0.6)
    ax.errorbar(x, m, yerr=yerr, fmt=marker, ms=ms, ecolor=color, elinewidth=0.9, capsize=1.8, zorder=4,
                label=label, **kw)
    if open_ and len(v) == 1:
        ax.add_patch(plt.Rectangle((x - 0.05, m * (1 - BAND)), 0.1, 2 * BAND * m, fc=BAND_FC, ec="none",
                                   zorder=1))


fig, axes = plt.subplots(3, 2, figsize=(COL2, 4.7), sharex="col",
                         gridspec_kw={"height_ratios": [1.3, 1, 1], "hspace": 0.2, "wspace": 0.28})
(a, b), (c, d), (e, f) = axes

# ---------------- 2D KS ----------------
KS = [(1.0, "1.0"), (2.25, "1.5"), (4.0, "2.0")]
col = HUE["ks2d"]
for x, scale in KS:
    m = crop("p7b_ks2d", scale, (0, 1, 2))
    sb = crop(C2, scale, (0, 1, 2))
    t = truths("ks2d", float(scale))
    if len(t):
        for ax, j, pb in ((a, 0, 1), (c, 0, x), (e, 1, x)):
            mark(ax, x - 0.1, t[:, j], pb, True, INK, "o")
    for ax, j, pb in ((a, 0, 1), (c, 0, x), (e, 1, x)):
        mark(ax, x + 0.0, sb[:, j], pb, False, BLUE, "D", ms=4.0)
        mark(ax, x + 0.1, m[:, j], pb, False, col, "o", ms=4.5)
base = float(truths("ks2d", 1.0)[:, 0].mean())
xs = np.linspace(0.8, 4.2, 10)
a.plot(xs, base * xs, ls="--", lw=0.8, color=MUTED, zorder=0)
a.text(3.3, base * 3.3 - 70, "extensive", color=MUTED, fontsize=7, ha="left")
c.axhline(base, ls="--", lw=0.8, color=MUTED, zorder=0)
a.set_title("2D Kuramoto–Sivashinsky")
a.set_ylabel(r"$D_{\mathrm{KY}}$")
c.set_ylabel(r"$D_{\mathrm{KY}}$ per base box")
e.set_ylabel(r"$n_+$ per base box")
e.set_xlabel(r"domain area / $44^2$")
e.set_xticks([1, 2.25, 4])
e.set_xticklabels(["1\n($44^2$)", "2.25\n($66^2$)", "4\n($88^2$)"])
c.set_ylim(70, 95)
e.set_ylim(30, 52)
a.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.1, ms=5.5),
          Line2D([], [], marker="D", ls="", mfc=BLUE, mec=BLUE, ms=4.0),
          Line2D([], [], marker="o", ls="", mfc=col, mec=col, ms=4.5)],
         ["ground truth", "small box ($44^2$)", "windows of $264^2$"], loc="upper left")

# ---------------- Kolmogorov ----------------
MODELS = {0.1: MAIN_A01, 0.05: "c19_e0d0_c6", 0.025: "k_xs0025"}
DX = {0.1: -0.16, 0.05: 0.0, 0.025: 0.16}
for alpha, tag in MODELS.items():
    for sc in (1, 2, 3):
        t = truths(f"kolm{alpha}", sc)
        m = crop(tag, sc, (0, 1, 2))
        sm = crop(SMALL_KOLM[alpha], sc, (0, 1, 2)) if alpha in SMALL_KOLM else np.empty((0, 2))
        for ax, j in ((b, 0), (d, 0), (f, 1)):
            pb = 1 if ax is b else sc
            mark(ax, sc + DX[alpha] - 0.05, t[:, j], pb, True, INK, DRAG_MARKER[alpha], ms=5)
            mark(ax, sc + DX[alpha] + 0.03, m[:, j], pb, False, DRAG_COLOR[alpha], DRAG_MARKER[alpha], ms=4.3)
            if len(sm):
                mark(ax, sc + DX[alpha] + 0.09, sm[:, j], pb, False, BLUE, DRAG_MARKER[alpha], ms=3.6)
b.set_title("Kolmogorov flow")
f.set_xlabel(r"channel width / $2\pi$")
f.set_xticks([1, 2, 3])
d.set_ylim(8, 33)
f.set_ylim(5, 17)
leg_alpha = [Line2D([], [], marker=DRAG_MARKER[a_], ls="", mfc=DRAG_COLOR[a_], mec=DRAG_COLOR[a_], ms=4.3)
             for a_ in MODELS]
fig.legend(leg_alpha + [Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=3.6),
                         Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.1, ms=5),
                         plt.Rectangle((0, 0), 1, 1, fc=BAND_FC, ec="none")],
           [rf"windows, $\alpha={a_}$" for a_ in MODELS] + ["small box (same shape)", "truth (open, same shape)",
                                                           r"$\pm15\%$: single-run truth"],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
for ax, letter in zip((a, b, c, d, e, f), "abcdef"):
    panel_label(ax, letter)

save(fig, "fig06_scaling", inputs)
