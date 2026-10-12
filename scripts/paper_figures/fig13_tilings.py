"""Fig. 13 (Block K3b, 2026-10-10): shifted tilings, the Part II DA mechanism.

(a) Schematic: the patch lattice of tiling 0 (solid) and the three tilings shifted by half a patch in x,
    y and both (dashed); every tiling has its own encoder frame and its own independent filter.
(b) The blend weights along one axis: Hann weights of the unshifted and the half-patch-shifted tiling,
    each vanishing on its own patch edges, summing to one everywhere (in 2D the weights are products).
(c) Analysis RMSE against position within a patch (steady state, 44^2, seed 0), 1 tiling and the 4-tiling
    blend, noisy-data model (Part II) and clean-data model: the single-tiling error peaks at patch edges.
(d) Error variance by wavenumber band (44^2): tiling 0 alone, the Hann blend, the uniform average of the
    4 tilings and its prediction 1'C1/K^2 from the inter-tiling error covariance C.
(e) Mean correlation between the tilings' analysis errors, by band, at 44^2 and 220^2: weakly correlated
    errors are what averaging removes.
(f) Seam ratio (mean gradient across patch edges / inside patches) of the truth, one tiling and the
    blend, for both models.
Inputs: h9_D10_L44_diag_T4.json, h9_D10_L220_diag_T4.json, the four *_fields.npz below, seams_I9.json.
"""
import json

import numpy as np

from _style import AQUA, COL2, EXP, INK, INK2, MUTED, ORANGE, OUT, panel_label, plt, save

D = EXP / "da2d"
inputs = []


def J(name):
    inputs.append(f"artifacts/experiments/da2d/{name}")
    return json.loads((D / name).read_text())


# Paper II shows the tilings and the blend weights in Fig. 1(c,d), so its version drops (a)-(b) and
# relabels (c)-(f) as (a)-(d) on a 2 x 2 grid.
P2 = "paperII" in str(OUT)
if P2:
    fig = plt.figure(figsize=(COL2 * 0.78, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.45)
    c, d_ = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    e, f = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    a = b = None
    LET = dict(zip("cdef", "abcd"))
else:
    fig = plt.figure(figsize=(COL2, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.7)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    d_, e, f = (fig.add_subplot(gs[1, i]) for i in range(3))
    LET = {k: k for k in "abcdef"}

# (a) schematic, (b) Hann weights
if a is not None:
    p = 1.0
    a.set_aspect("equal"); a.grid(False)
    for k in range(4):
        a.axvline(k * p, color=AQUA, lw=1.2); a.axhline(k * p, color=AQUA, lw=1.2)
    for k in range(4):
        a.axvline(k * p + p / 2, color=INK2, lw=0.8, ls="--"); a.axhline(k * p + p / 2, color=INK2, lw=0.8, ls="--")
    for x, y in ((1.5, 1.5),):
        a.add_patch(plt.Rectangle((1, 1), 1, 1, fill=True, color=AQUA, alpha=0.18, lw=0))
        a.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, ec=INK2, lw=1.0, ls="--"))
    a.set_xlim(0.2, 3.3); a.set_ylim(0.2, 3.3); a.set_xticks([]); a.set_yticks([])
    a.text(1.06, 1.06, "tiling 0", fontsize=5.8, color=INK, va="bottom")
    a.text(2.45, 2.45, "shifted", fontsize=5.8, color=INK2, ha="right", va="top")
    a.set_xlabel("tiling 0: solid; shifted: dashed", fontsize=6.6)
    panel_label(a, LET["a"])

    # (b) Hann weights along one axis
    P = 16
    x = np.arange(33)
    w0 = np.sin(np.pi * ((x % P) + 0.5) / P) ** 2
    w1 = np.sin(np.pi * (((x - P // 2) % P) + 0.5) / P) ** 2
    b.plot(x, w0, color=AQUA, label="tiling 0"); b.plot(x, w1, color=INK2, ls="--", label="shifted by $p/2$")
    b.plot(x, w0 + w1, color=INK, lw=0.8, ls=":", label="sum")
    for k in range(3):
        b.axvline(k * P - 0.5, color=AQUA, lw=0.4, alpha=0.6)
    b.set_xlabel("grid point ($p = 16$)"); b.set_ylabel("blend weight"); b.set_ylim(0, 1.75); b.set_yticks([0, 0.5, 1])
    b.legend(fontsize=5.6, loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.6, bbox_to_anchor=(0.5, 1.03))
    panel_label(b, LET["b"])

# (c) error profile within a patch
for fn, lab, col, ls in (("h9_D10_L44_fields_T1_fields.npz", "noisy, 1 tiling", AQUA, "--"),
                         ("h9_D10_L44_diag_T4_fields.npz", "noisy, 4 tilings", AQUA, "-"),
                         ("h9_J18c_clean_T1_fields.npz", "clean, 1 tiling", INK2, "--"),
                         ("h9_J18c_clean_T4_fields.npz", "clean, 4 tilings", INK2, "-")):
    inputs.append(f"artifacts/experiments/da2d/{fn}")
    z = np.load(D / fn)
    E = ((z["blend"] - z["truth"]) ** 2).mean(0)
    pos = np.arange(E.shape[0]) % int(z["patch"])
    prof = [np.sqrt(0.5 * (E[pos == i].mean() + E[:, pos == i].mean())) for i in range(int(z["patch"]))]
    c.plot(np.arange(len(prof)) + 0.5, prof, ls=ls, color=col, label=lab)
c.set_xlabel("position in patch"); c.set_ylabel("analysis RMSE"); c.set_ylim(0.25, 0.95); c.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
c.legend(fontsize=5.4, loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.6, bbox_to_anchor=(0.5, 1.03))
panel_label(c, LET["c"])

# (d) error variance by band
dg = J("h9_D10_L44_diag_T4.json")["steady"]["diag"]
nb = len(dg["bands"]); xb = np.arange(nb); wbar = 0.2
for i, (k, lab, col) in enumerate((("tiling0_var", "1 tiling", INK2), ("blend_var", "Hann blend", AQUA),
                                   ("uniform_var", "uniform average", ORANGE))):
    d_.bar(xb + (i - 1) * wbar, dg[k], width=wbar, color=col, label=lab,  # hatch: tells the blend from the
           hatch="////" if k == "uniform_var" else None, edgecolor="white", lw=0)  # uniform average in grayscale
d_.plot(xb + wbar, dg["uniform_var_predicted"], "_", color=INK, ms=7, mew=1.2, label=r"$\mathbf{1}^\top C\mathbf{1}/K^2$")
d_.set_xticks(xb); d_.set_xticklabels(["<0.6", "0.6–\n1.1", "1.1–\n1.6", ">1.6"], fontsize=6)
d_.set_ylim(0, 0.125); d_.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
d_.set_xlabel(r"wavenumber band $|k|$"); d_.set_ylabel("error variance"); d_.grid(axis="x", visible=False)
d_.legend(fontsize=5.4, loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.5, bbox_to_anchor=(0.5, 1.04))
panel_label(d_, LET["d"])

# (e) inter-tiling error correlation by band
for fn, lab, col, mk in (("h9_D10_L44_diag_T4.json", "$44^2$", AQUA, "o"), ("h9_D10_L220_diag_T4.json", "$220^2$", INK2, "s")):
    g = J(fn)["steady"]["diag"]
    e.plot(xb, g["mean_offdiag_corr"], marker=mk, color=col, ms=3.5, label=lab)
e.axhline(1, color=MUTED, lw=0.6, ls=":"); e.text(nb - 1, 0.93, "identical errors", fontsize=6, color=MUTED, ha="right")
e.set_xticks(xb); e.set_xticklabels(["<0.6", "0.6–\n1.1", "1.1–\n1.6", ">1.6"], fontsize=6)
e.set_xlabel(r"wavenumber band $|k|$"); e.set_ylabel("inter-tiling error corr."); e.set_ylim(0, 1.05)
e.legend(fontsize=6, loc="center left"); panel_label(e, LET["e"])

# (f) seam ratio
sm = J("seams_I9.json")
vals = [("truth", sm["h9_D10_L44_diag_T4_fields"]["truth"], MUTED),
        ("clean\n1 tiling", sm["h9_J18c_clean_T1_fields"]["tiling0"], INK2),
        ("noisy\n1 tiling", sm["h9_D10_L44_fields_T1_fields"]["tiling0"], AQUA),
        ("clean\n4 tilings", sm["h9_J18c_clean_T4_fields"]["blend"], INK2),
        ("noisy\n4 tilings", sm["h9_D10_L44_diag_T4_fields"]["blend"], AQUA)]
for i, (lab, v, col) in enumerate(vals):
    f.bar(i, v, color=col, width=0.65, alpha=0.9)
    f.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6, color=INK)
f.axhline(1, color=INK, lw=0.6, ls=":")
f.set_xticks(range(len(vals))); f.set_xticklabels([v[0].replace("\n", ", ") for v in vals], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
f.tick_params(axis="x", length=0)
f.set_ylabel("seam ratio (edge / interior)"); f.set_ylim(0, 2.3); f.grid(axis="x", visible=False)
panel_label(f, LET["f"])

save(fig, "fig13_tilings", inputs)
