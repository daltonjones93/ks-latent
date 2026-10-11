"""Figure 2 (task C2): the three systems and the training windows.

(a) 1D KS, L = 100: space-time plot of one training trajectory (250 time units).
(b) 2D KS on the 264^2 training domain (one snapshot), with one 40x40-site training window
    outlined: forced halo (6 sites) hatched, free interior clear.
(c) Kolmogorov flow, alpha = 0.05, vorticity on the 16 x 2pi training strip (one snapshot),
    with one 48-site training window (full height) and its 6-site x-halo.
All fields standardized by their dataset's std; one shared diverging colormap with symmetric
limits. Sources: artifacts/datasets/{stage1_trajectories_dtsnap1, ks2d_wide6,
kolmogorov_wide16_alpha0.05}.h5 (training split, trajectory 0, last snapshot for 2D).
"""

import h5py
import numpy as np
from matplotlib.patches import Rectangle

from _style import ART, COL1, COL2, INK, panel_label, plt, save

CMAP, VMAX = "RdBu_r", 2.5
D1 = ART / "datasets/stage1_trajectories_dtsnap1.h5"
D2 = ART / "datasets/ks2d_wide6.h5"
D3 = ART / "datasets/kolmogorov_wide16_alpha0.05.h5"


def read(path, flat2d=True):
    with h5py.File(path, "r") as h:
        md = dict(h["metadata"].attrs)
        if flat2d:  # 2D datasets are stored already standardized
            return np.asarray(h["trajectories"][0, -1], dtype=np.float64).reshape(int(md["NX"]), int(md["NY"])), md
        u = np.asarray(h["trajectories_raw"][0], dtype=np.float64)
    return (u - float(md["normalization_mean"])) / float(md["normalization_std"]), md


u1, m1 = read(D1, flat2d=False)
u2, m2 = read(D2)
u3, m3 = read(D3)

fig = plt.figure(figsize=(COL2, 4.15))
gs = fig.add_gridspec(2, 3, height_ratios=[2.35, 1.0], width_ratios=[1.25, 1.0, 0.05], hspace=0.42, wspace=0.28)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :2])
cax = fig.add_subplot(gs[0, 2])
for ax in (a, b, c):
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)

# (a) 1D KS space-time, explicit edges
L, T = float(m1["L"]), u1.shape[0]
xe = np.linspace(0, L, u1.shape[1] + 1)
te = np.arange(T + 1) - 0.5
im = a.pcolormesh(xe, te * float(m1["dt_snap"]), u1, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat",
                  rasterized=True)
a.set_xlabel("$x$")
a.set_ylabel("$t$")
a.set_title(r"1D KS, $L=100$", loc="left")
a.set_xticks([0, 50, 100])
panel_label(a, "a")


def window(ax, x0, y0, w, h, halo_x, halo_y):
    """Outline a training window; hatch its forced halo."""
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=INK, lw=1.1))
    ax.add_patch(Rectangle((x0 + halo_x, y0 + halo_y), w - 2 * halo_x, h - 2 * halo_y, fill=False, ec=INK,
                           lw=0.6, ls="--"))
    kw = dict(fill=False, ec="none", hatch="////", lw=0)
    ax.add_patch(Rectangle((x0, y0), halo_x, h, **kw))
    ax.add_patch(Rectangle((x0 + w - halo_x, y0), halo_x, h, **kw))
    if halo_y:
        ax.add_patch(Rectangle((x0 + halo_x, y0), w - 2 * halo_x, halo_y, **kw))
        ax.add_patch(Rectangle((x0 + halo_x, y0 + h - halo_y), w - 2 * halo_x, halo_y, **kw))


plt.rcParams["hatch.color"] = INK
plt.rcParams["hatch.linewidth"] = 0.35

# (b) 2D KS 264^2
Lx2 = float(m2["Lx"]); nx2 = int(m2["NX"]); h2 = Lx2 / 48  # 48 sites of 16 grid points
e2 = np.linspace(0, Lx2, nx2 + 1)
b.pcolormesh(e2, e2, u2.T, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat", rasterized=True)
b.set_aspect("equal")
window(b, 3 * h2, 4 * h2, 40 * h2, 40 * h2, 6 * h2, 6 * h2)
b.set_xlabel("$x$"); b.set_ylabel("$y$", labelpad=1)
b.set_xticks([0, 132, 264]); b.set_yticks([0, 132, 264])
b.set_title(r"2D KS, $264^2$ training domain", loc="left")
panel_label(b, "b")

# (c) Kolmogorov strip
Lx3, Ly3 = float(m3["Lx"]), float(m3["Ly"]); nx3, ny3 = int(m3["NX"]), int(m3["NY"])
h3 = Lx3 / 128  # 128 sites of 16 grid points
c.pcolormesh(np.linspace(0, Lx3, nx3 + 1), np.linspace(0, Ly3, ny3 + 1), u3.T, cmap=CMAP, vmin=-VMAX,
             vmax=VMAX, shading="flat", rasterized=True)
c.set_aspect("equal")
window(c, 40 * h3, 0, 48 * h3, Ly3, 6 * h3, 0)
c.set_xlabel("$x$"); c.set_ylabel("$y$", labelpad=1)
c.set_yticks([0, 2 * np.pi]); c.set_yticklabels(["0", r"$2\pi$"])
c.set_xticks([2 * np.pi * k for k in range(0, 17, 4)])
c.set_xticklabels(["0"] + [rf"${k}\cdot2\pi$" for k in range(4, 17, 4)])
c.set_title(r"Kolmogorov flow, $\alpha=0.05$: vorticity on the $16\times2\pi$ training strip", loc="left")
panel_label(c, "c")

cb = fig.colorbar(im, cax=cax)
cb.set_label("field / dataset std")
cb.outline.set_linewidth(0.4)
cax.set_box_aspect(None)

save(fig, "fig02_systems", [str(p.relative_to(ART.parent)) for p in (D1, D2, D3)])

# Paper II (S4.6): the 2D KS panel alone, one column wide.
fig = plt.figure(figsize=(COL1, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.08)
b, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
b.grid(False)
for sp in ("top", "right"):
    b.spines[sp].set_visible(True)
im = b.pcolormesh(e2, e2, u2.T, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat", rasterized=True)
b.set_aspect("equal")
window(b, 3 * h2, 4 * h2, 40 * h2, 40 * h2, 6 * h2, 6 * h2)
b.set_xlabel("$x$"); b.set_ylabel("$y$", labelpad=1)
b.set_xticks([0, 132, 264]); b.set_yticks([0, 132, 264])
cb = fig.colorbar(im, cax=cax)
cb.set_label("field / dataset std")
cb.outline.set_linewidth(0.4)
save(fig, "fig02_systems_ks2d", [str(D2.relative_to(ART.parent))])
