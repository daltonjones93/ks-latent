"""Paper II Fig. 1 (docs/split_paper_tasks.md S4.17): exactly local data assimilation in a latent field.

(a) Schematic: a 1D slice of the field, its patches, the c channels of each site, the per-patch
    encoder and decoder (each patch to and from its own site), and the propagator's finite reach.
(b) |J| of the latent observation operator at 44^2, observations sorted by owning site (12 of 64
    sites), model p7c12d10_ks2d: exactly zero outside the 12 columns of each site (as Fig. 9a of paper1).
(c) Shifted tilings: the patch lattice of tiling 0 (solid) and the half-patch-shifted lattice (dashed).
(d) Blend weights along one axis, Eq. (blend): each tiling's Hann weight vanishes on its own patch
    edges; the weights sum to one (as Fig. 13a-b of paper1).
Input: artifacts/experiments/da2d/jacobian_sparsity_L44_p7c12d10_ks2d.npz.
Writes fig_concept_da -> content name concept_da.
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import AQUA, BLUE, COL2, EXP, INK, INK2, MUTED, ORANGE, panel_label, plt, save

LIGHT = "#d9e6f6"
D = EXP / "da2d"
inputs = ["artifacts/experiments/da2d/jacobian_sparsity_L44_p7c12d10_ks2d.npz"]

fig = plt.figure(figsize=(COL2, 2.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 1.0, 0.95, 1.05], wspace=0.75)
a, b, c, d = (fig.add_subplot(gs[0, i]) for i in range(4))


def arrow(ax, p, q, color=INK2, lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


# (a) schematic
a.set_axis_off(); a.set_xlim(0, 10); a.set_ylim(-0.3, 7.4)
xs = np.linspace(0, 1, 400)
f = 0.55 * np.sin(2 * np.pi * 5 * xs + 0.4) + 0.35 * np.sin(2 * np.pi * 9 * xs + 1.3) + 0.2 * np.sin(2 * np.pi * 14 * xs + 2.1)
a.plot(0.3 + 9.4 * xs, 6.2 + 0.45 * f, color=INK, lw=0.9)
a.text(0.3, 7.15, r"field $u$ and its patches", fontsize=6.5, color=INK2)
P, w = 6, 9.4 / 6
for j in range(P + 1):
    a.add_line(plt.Line2D([0.3 + j * w] * 2, [5.5, 6.9], color=MUTED, lw=0.5, ls=":"))
for j in range(P):
    xc = 0.3 + (j + 0.5) * w
    for ch in range(3):
        a.add_patch(Rectangle((xc - 0.22, 2.2 + 0.45 * ch), 0.44, 0.4, fc=LIGHT, ec=BLUE, lw=0.6))
    arrow(a, (xc - 0.25, 5.4), (xc - 0.25, 3.7))
    arrow(a, (xc + 0.25, 3.7), (xc + 0.25, 5.4), color=AQUA)
a.text(0.3, 4.45, r"$\mathcal{E}$", fontsize=7, color=INK2, va="center")
a.text(9.7, 4.45, r"$\mathcal{D}$", fontsize=7, color=AQUA, va="center", ha="right")
# propagator reach bracket over three sites
x0, x1 = 0.3 + 1.5 * w - 0.3, 0.3 + 4.5 * w + 0.3
a.plot([x0, x0, x1, x1], [1.95, 1.65, 1.65, 1.95], color=ORANGE, lw=1.0)
a.text((x0 + x1) / 2, 1.2, r"$\mathcal{F}$: finite reach", fontsize=6.5, color=INK, ha="center", va="center")
a.text(0.3, 0.2, r"$z_j\in\mathbb{R}^c$; patch $j\leftrightarrow$ site $j$ only", fontsize=6.5, color=INK)
panel_label(a, "a")

# (b) Jacobian
Jz = np.load(D / "jacobian_sparsity_L44_p7c12d10_ks2d.npz")
ch, nsite = int(Jz["channels"]), 12
J = np.abs(Jz["J"][Jz["site"] < nsite][:, : nsite * ch])
b.imshow(np.log10(J + 1e-12), cmap="Greys", vmin=-4, vmax=0.5, aspect="auto", interpolation="nearest")
b.grid(False)
for s in range(nsite + 1):
    b.axvline(s * ch - 0.5, color=AQUA, lw=0.3)
b.set_xlabel(f"latent coord. ({ch}/site)", fontsize=6.5); b.set_ylabel("obs. (by site)", fontsize=6.5)
b.set_xticks([0, nsite * ch - 1]); b.set_xticklabels(["0", str(nsite * ch)]); b.set_yticks([])
b.set_title(r"$|J|$ at $44^2$", fontsize=7, pad=4)
panel_label(b, "b")

# (c) lattices
c.set_aspect("equal"); c.grid(False)
for k in range(4):
    c.axvline(k, color=AQUA, lw=1.1); c.axhline(k, color=AQUA, lw=1.1)
    c.axvline(k + 0.5, color=INK2, lw=0.7, ls="--"); c.axhline(k + 0.5, color=INK2, lw=0.7, ls="--")
c.add_patch(plt.Rectangle((1, 1), 1, 1, color=AQUA, alpha=0.18, lw=0))
c.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, ec=INK2, lw=0.9, ls="--"))
c.set_xlim(0.2, 3.3); c.set_ylim(0.2, 3.3); c.set_xticks([]); c.set_yticks([])
c.set_xlabel("solid: tiling 0\ndashed: shifted", fontsize=6.3)
c.set_anchor("N")
panel_label(c, "c")

# (d) blend weights
p = 16
x = np.arange(33)
w0 = np.sin(np.pi * ((x % p) + 0.5) / p) ** 2
w1 = np.sin(np.pi * (((x - p // 2) % p) + 0.5) / p) ** 2
d.plot(x, w0, color=AQUA, label="tiling 0"); d.plot(x, w1, color=INK2, ls="--", label="shifted")
d.plot(x, w0 + w1, color=INK, lw=0.8, ls=":", label="sum")
for k in range(3):
    d.axvline(k * p - 0.5, color=AQUA, lw=0.4, alpha=0.6)
d.set_xlabel("grid point ($p=16$)", fontsize=6.5); d.set_ylabel("weight", fontsize=6.5, labelpad=1)
d.set_ylim(0, 1.12); d.set_yticks([0, 0.5, 1])
h, l = d.get_legend_handles_labels()
c.legend(h, l, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.55), frameon=False, handlelength=1.6, title="weights in (d)", title_fontsize=5.8)
panel_label(d, "d")

save(fig, "fig_concept_da", inputs)
