"""Figure 11 (task K3f): one steady-state analysis of the Part II latent filter at 440^2
(p7c12d10_ks2d, d_lat = 76 800 per tiling, 64 members, observation stride 8; cycle 99 of 100).

Top row, full domain: truth, the analysis of one tiling, the blend of four shifted tilings, and the
two analysis errors. Bottom row: the same five fields in a 55 x 55 window (10 x 10 sites), with the
observation sites on the truth panel and tiling 0's patch edges on the error panels, so the seams of
the single tiling and their removal by the blend are visible.
Fields (stored in field units) divided by the training standard deviation for display; RMSE in field units,
as in the tables. Separate symmetric colour limits for fields and errors.
Source: artifacts/experiments/da2d/h9_K10e_L440_T4_fields_fields.npz
(exp_h9_tiled_da.py --L 440 --n-tilings 4 --save-fields 3, scripts/run_k10.sh).
"""
import numpy as np
from matplotlib.patches import Rectangle

from _style import COL2, EXP, INK, ROOT, plt, save

SRC = EXP / "da2d/h9_K10e_L440_T4_fields_fields.npz"
D = np.load(SRC)
CYCLE = 99
i = int(np.nonzero(D["cycle"] == CYCLE)[0][0])
L, NX, ns, s = float(D["L"]), int(D["NX"]), int(D["n_sites"]), 8
dx, h = L / NX, L / ns
STD = 6.172126667325263  # training normalization std of p7c12d10_ks2d (checkpoint)
tr = D["truth"][i] / STD
F = {"truth": tr, "one tiling": D["tiling0"][i] / STD, "four tilings": D["blend"][i] / STD}
F["error, one tiling"] = F["one tiling"] - tr
F["error, four tilings"] = F["four tilings"] - tr
rms = {"one tiling": STD * float(np.sqrt(np.mean(F["error, one tiling"] ** 2))),
       "four tilings": STD * float(np.sqrt(np.mean(F["error, four tilings"] ** 2)))}
VMAX, EMAX = 3.0, 0.25
x0, w = 40 * h, 10 * h  # zoom window: sites 40-49 in x and y
e = np.linspace(0, L, NX + 1)

fig, axes = plt.subplots(2, 5, figsize=(COL2, 2.75), gridspec_kw={"hspace": 0.3, "wspace": 0.06})
for j, (name, fld) in enumerate(F.items()):
    err = name.startswith("error")
    for r, ax in enumerate(axes[:, j]):
        im = ax.pcolormesh(e, e, fld.T, cmap="cividis", vmin=-(EMAX if err else VMAX), vmax=EMAX if err else VMAX,
                           shading="flat", rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(name, fontsize=7, pad=3)
            if err:
                ax.text(0.97, 0.03, f"RMSE {rms[name.replace('error, ', '')]:.2f}", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
            ax.add_patch(Rectangle((x0, x0), w, w, fill=False, ec=INK, lw=0.7))
        else:
            ax.set_xlim(x0, x0 + w); ax.set_ylim(x0, x0 + w)
            if err:
                for k in range(41, 50):
                    ax.axvline(k * h, color=INK, lw=0.25, alpha=0.5); ax.axhline(k * h, color=INK, lw=0.25, alpha=0.5)
    if err:
        im_e = im
    elif j == 0:
        im_f = im
ob = np.arange(0, NX, s) * dx
X, Y = np.meshgrid(ob[(ob >= x0) & (ob < x0 + w)], ob[(ob >= x0) & (ob < x0 + w)])
axes[1, 0].plot(X.ravel(), Y.ravel(), ".", color=INK, ms=0.5)
for ax, letter in zip(axes.ravel(), "abcdefghij"):  # inside the panel: five columns leave no room outside
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold",
            color=INK, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
axes[0, 0].set_ylabel(r"$440^2$", fontsize=7)
axes[1, 0].set_ylabel(r"$55\times55$ window", fontsize=7)
for im_, y0, lab in ((im_f, 0.53, r"$u/\sigma$"), (im_e, 0.12, r"error$/\sigma$")):
    cax = fig.add_axes([0.92, y0, 0.01, 0.33])
    cb = fig.colorbar(im_, cax=cax); cb.ax.set_title(lab, fontsize=7, pad=3); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7)
save(fig, "fig11_da_snapshot", [str(SRC.relative_to(ROOT))])
print(rms, float(tr.std()))
