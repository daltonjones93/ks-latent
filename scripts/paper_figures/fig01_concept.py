"""Figure 1 (task C1): concept schematic. Hand-drawn geometry, no data.

(a) a global latent vector vs a local latent field;
(b) transfer: the same weight-shared operators applied on a larger lattice;
(c) window (crop) training: a window of a large simulation, its forced halo and free interior;
(d) per-patch decoder -> block-diagonal observation Jacobian -> per-site analysis.
Writes fig01_concept (a-d, paper1) and fig01_concept_transfer (a-c, Paper I).
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import AQUA, BLUE, COL2, INK, INK2, MUTED, ORANGE, panel_label, plt, save

LIGHT = "#d9e6f6"  # light blue fill for latent sites
HALO = "#fbd9c9"   # light orange fill for the forced halo
FREE = "#d3efe3"   # light aqua fill for the free interior

rng = np.random.default_rng(3)
xs = np.linspace(0, 1, 400)
field = (0.55 * np.sin(2 * np.pi * 5 * xs + 0.4) + 0.35 * np.sin(2 * np.pi * 9 * xs + 1.3)
         + 0.2 * np.sin(2 * np.pi * 14 * xs + 2.1))


def arrow(ax, p, q, color=INK2, lw=0.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def wave(ax, x0, x1, y0, amp, color=INK, lw=0.9, n=None):
    t = xs if n is None else np.linspace(0, n, 400) % 1.0
    f = field if n is None else np.interp(np.linspace(0, n, 400) % 1.0, xs, field)
    ax.plot(x0 + (x1 - x0) * np.linspace(0, 1, 400), y0 + amp * f, color=color, lw=lw, solid_capstyle="round")


def draw(a, b, c, d=None):
    """Draw panels (a)-(c), and (d) if an axis is given, into the given axes."""
    for ax in (a, b, c) + (() if d is None else (d,)):
        blank(ax)
    # ---------------- (a) global vector vs local field ----------------
    a.set_xlim(0, 10); a.set_ylim(0, 7.2)
    P = 8
    # top: global
    wave(a, 0.3, 9.7, 6.35, 0.38)
    a.text(0.3, 6.95, r"field $u(x)$", fontsize=7, color=INK2)
    for k in range(5):
        a.add_patch(Rectangle((4.35 + 0.27 * k, 4.6), 0.25, 0.25, fc=LIGHT, ec=BLUE, lw=0.7))
    for xx in (1.0, 3.0, 5.0, 7.0, 9.0):
        arrow(a, (xx, 5.85), (5.0 + 0.12 * (xx - 5.0), 4.95), color=MUTED, lw=0.5)
    a.text(5.82, 4.72, r"$z\in\mathbb{R}^{d}$, $d$ fixed", fontsize=7, color=INK, va="center")
    a.text(0.3, 4.72, "global", fontsize=7.5, color=INK, fontweight="bold", va="center")
    # bottom: local field
    wave(a, 0.3, 9.7, 3.3, 0.38)
    w = 9.4 / P
    for j in range(P):
        x0 = 0.3 + j * w
        a.add_line(plt.Line2D([x0, x0], [2.75, 3.85], color=MUTED, lw=0.5, ls=":"))
        for ch in range(3):
            a.add_patch(Rectangle((x0 + w / 2 - 0.2, 0.7 + 0.42 * ch), 0.4, 0.38, fc=LIGHT, ec=BLUE, lw=0.7))
        arrow(a, (x0 + w / 2, 2.75), (x0 + w / 2, 2.0), color=MUTED, lw=0.5)
    a.add_line(plt.Line2D([9.7, 9.7], [2.75, 3.85], color=MUTED, lw=0.5, ls=":"))
    a.text(0.3, 0.25, r"$z_j\in\mathbb{R}^{c}$ at each site $j$", fontsize=7, color=INK)
    a.text(0.3, 2.35, "local", fontsize=7.5, color=INK, fontweight="bold")
    panel_label(a, "a")

    # ---------------- (b) transfer: same weights, more sites ----------------
    b.set_xlim(0, 10); b.set_ylim(0, 7.2)


    def lattice(ax, x0, y, n, size=0.42, gap=0.08, hi=None):
        for j in range(n):
            fc = LIGHT
            ec = BLUE
            if hi is not None and abs(j - hi) <= 1:
                fc, ec = "#bcd4f0", BLUE
            ax.add_patch(Rectangle((x0 + j * (size + gap), y), size, size, fc=fc, ec=ec, lw=0.7))
        return x0 + n * (size + gap)


    def stencil(ax, xc, y, size=0.42, gap=0.08):
        """Bracket over three sites: the shared local operator."""
        x0 = xc - 1.5 * size - gap
        ax.plot([x0, x0, x0 + 3 * size + 2 * gap, x0 + 3 * size + 2 * gap], [y, y + 0.22, y + 0.22, y],
                color=ORANGE, lw=1.1)
        ax.text(xc, y + 0.32, r"$f_\theta$", color=INK, fontsize=7, ha="center", va="bottom")


    xe = lattice(b, 0.3, 4.9, 8, hi=4)
    stencil(b, 0.3 + 4 * 0.5 + 0.21, 5.4)
    b.text(0.3, 6.55, r"trained: $P$ sites", fontsize=7, color=INK)
    xe2 = lattice(b, 0.3, 1.7, 18, hi=4)
    lattice(b, 0.3, 1.7, 18, hi=13)
    stencil(b, 0.3 + 4 * 0.5 + 0.21, 2.2)
    stencil(b, 0.3 + 13 * 0.5 + 0.21, 2.2)
    b.text(0.3, 3.35, r"deployed: more sites, same $\theta$", fontsize=7, color=INK)
    arrow(b, (4.6, 4.75), (4.6, 3.75), color=INK2, lw=0.8)
    b.text(4.8, 4.25, "add sites", fontsize=7, color=INK2, va="center")
    b.text(0.3, 0.6, r"$\theta$ does not depend on $P$",
           fontsize=7, color=INK)
    panel_label(b, "b")

    # ---------------- (c) crop training ----------------
    c.set_xlim(0, 10); c.set_ylim(-0.4, 6.8)
    # big domain (grid of sites)
    N = 12; s = 0.42; X0, Y0 = 0.3, 1.2
    for i in range(N + 1):
        c.add_line(plt.Line2D([X0 + i * s, X0 + i * s], [Y0, Y0 + N * s], color=GRID_C if (GRID_C := "#dedcd6") else MUTED, lw=0.4))
        c.add_line(plt.Line2D([X0, X0 + N * s], [Y0 + i * s, Y0 + i * s], color=GRID_C, lw=0.4))
    c.add_patch(Rectangle((X0, Y0), N * s, N * s, fc="none", ec=INK2, lw=0.8))
    cx, cy, cw = 5, 4, 5  # crop at sites [cx, cx+cw)
    c.add_patch(Rectangle((X0 + cx * s, Y0 + cy * s), cw * s, cw * s, fc=HALO, ec=INK, lw=1.2, zorder=3))
    c.add_patch(Rectangle((X0 + (cx + 1.4) * s, Y0 + (cy + 1.4) * s), (cw - 2.8) * s, (cw - 2.8) * s, fc=FREE, ec="none",
                          zorder=3))
    c.text(X0, Y0 + N * s + 0.25, "one large simulation", fontsize=7, color=INK)
    # zoom
    Z0x, Z0y, Zs = 6.2, 1.2, 0.6
    n = 7
    for i in range(n):
        for j in range(n):
            ring = min(i, j, n - 1 - i, n - 1 - j)
            fc = HALO if ring < 2 else FREE
            c.add_patch(Rectangle((Z0x + i * Zs * 0.8, Z0y + j * Zs * 0.8), Zs * 0.8, Zs * 0.8, fc=fc, ec="white", lw=0.8))
    c.add_patch(Rectangle((Z0x, Z0y), n * Zs * 0.8, n * Zs * 0.8, fc="none", ec=INK, lw=1.2))
    arrow(c, (X0 + (cx + cw) * s + 0.05, Y0 + (cy + cw) * s), (Z0x - 0.1, Z0y + n * Zs * 0.8 - 0.1), color=MUTED, lw=0.6)
    c.text(Z0x + n * Zs * 0.4, Z0y + n * Zs * 0.4, "free\ninterior", fontsize=7, ha="center", va="center", color=INK)
    c.text(Z0x + n * Zs * 0.4, Z0y + n * Zs * 0.8 + 0.2, "window", fontsize=7, ha="center", color=INK)
    for k, (fc, txt) in enumerate([(HALO, "halo: true encoding forced each step"),
                                   (FREE, "interior: runs free, scored")]):
        c.add_patch(Rectangle((0.3, 0.45 - 0.55 * k), 0.32, 0.32, fc=fc, ec="none"))
        c.text(0.8, 0.61 - 0.55 * k, txt, fontsize=7, va="center", color=INK)

    panel_label(c, "c")

    GRID_C = "#dedcd6"
    if d is None:
        return
    # ---------------- (d) per-patch decoder -> block-diagonal Jacobian -> per-site analysis -------------
    d.set_xlim(0, 10); d.set_ylim(-0.4, 6.8)
    nb, bs = 5, 0.62
    M0x, M0y = 0.6, 1.4
    for i in range(nb):
        for j in range(nb):
            on = i == j
            d.add_patch(Rectangle((M0x + j * bs, M0y + (nb - 1 - i) * bs), bs, bs, fc=BLUE if on else "white",
                                  ec=GRID_C, lw=0.5, alpha=0.85 if on else 1))
    d.add_patch(Rectangle((M0x, M0y), nb * bs, nb * bs, fc="none", ec=INK2, lw=0.8))
    d.text(M0x + nb * bs / 2, M0y + nb * bs + 0.25, r"$J=\partial\,\mathcal{H}(\mathcal{D}(z))/\partial z$",
           fontsize=7, ha="center", color=INK)
    d.text(M0x - 0.2, M0y + nb * bs / 2, "obs.\nby patch", fontsize=7, ha="right", va="center", color=INK2,
           rotation=90)
    d.text(M0x + nb * bs / 2, M0y - 0.15, r"latent by site", fontsize=7, ha="center", va="top",
           color=INK2)
    d.text(M0x + nb * bs / 2, 0.1, "block-diagonal;\n$c$ JVPs give all of $J$", fontsize=7,
           ha="center", va="center", color=INK)
    arrow(d, (4.1, M0y + nb * bs / 2), (5.0, M0y + nb * bs / 2), color=INK2, lw=0.8)
    # per-site analysis: grid of sites, one analysed with a local neighbourhood
    G0x, G0y, gs, gn = 5.4, 1.4, 0.62, 7
    for i in range(gn):
        for j in range(gn):
            dd = max(abs(i - 3), abs(j - 3))
            fc = BLUE if dd == 0 else ("#bcd4f0" if dd <= 1 else "white")
            d.add_patch(Rectangle((G0x + i * gs, G0y + j * gs), gs, gs, fc=fc, ec=GRID_C, lw=0.5))
    for (i, j) in [(1, 5), (2, 2), (4, 3), (5, 5), (3, 4), (2, 3), (4, 1), (6, 2), (0, 0), (5, 2)]:
        d.plot(G0x + (i + 0.5) * gs, G0y + (j + 0.5) * gs, "x", color=INK, ms=3.5, mew=0.9)
    d.add_patch(Rectangle((G0x, G0y), gn * gs, gn * gs, fc="none", ec=INK2, lw=0.8))
    d.text(G0x + gn * gs / 2, G0y + gn * gs + 0.25, "per-site analysis", fontsize=7, ha="center", color=INK)
    d.text(G0x + gn * gs / 2, 0.1, "site uses only nearby\nobservations ($\\times$)", fontsize=7,
           ha="center", va="center", color=INK)
    panel_label(d, "d")



# paper1: all four panels, 2 x 2.
fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.0), gridspec_kw={"hspace": 0.18, "wspace": 0.08})
draw(*axes.ravel())
save(fig, "fig01_concept", [])

# Paper I (docs/split_paper_tasks.md S3.8): panels (a)-(c) only, (c) centred under (a) and (b), each
# panel at the same size as in the 2 x 2 version. Panel (d) belongs to the data-assimilation paper.
fig = plt.figure(figsize=(COL2, 4.0))
gs = fig.add_gridspec(2, 4, hspace=0.18, wspace=0.08)
draw(fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3]))
save(fig, "fig01_concept_transfer", [])

