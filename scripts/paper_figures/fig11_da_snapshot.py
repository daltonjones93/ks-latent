"""Figure 11 (task F3): one steady-state analysis of the per-site latent filter at 330^2
(d_lat = 21 600, 64 members; cycle 78 of 100).

Top row, full domain: truth, free run (same model, no observations), analysis (decoded ensemble
mean), analysis error. Bottom row: the same four fields in a 55 x 55 window (10 x 10 sites), with
the observation sites on the truth panel and patch edges on the error panel.
All fields divided by the training standard deviation; one shared diverging colormap with
symmetric limits. Source: artifacts/experiments/da2d/da2d_D6_L330_local_N64_s0_fields.npz
(run_da_pff_2d_ltransfer.py --phys-rmse --save-fields 10).
"""
import numpy as np
from matplotlib.patches import Rectangle

from _style import COL2, EXP, INK, ROOT, panel_label, plt, save

SRC = EXP / "da2d/da2d_D6_L330_local_N64_s0_fields.npz"
D = np.load(SRC)
STD = 6.172126667325263  # training normalization std of the 2D KS model (checkpoint)
CYCLE = 78
i = int(np.nonzero(D["cycles"] == CYCLE)[0][0])
L, NX, ns, s = float(D["L"]), int(D["NX"]), int(D["n_sites"]), int(D["obs_stride"])
dx, h = L / NX, L / ns
F = {"truth": D["truth"][i] / STD, "free run": D["free"][i] / STD, "analysis": D["analysis"][i] / STD}
F["analysis error"] = F["analysis"] - F["truth"]
rms = {k: float(np.sqrt(np.mean((v * STD if k == "analysis error" else (v - F["truth"]) * STD) ** 2)))
       for k, v in F.items() if k != "truth"}
VMAX = 3.0
x0, w = 22 * h, 10 * h  # zoom window: sites 22-31 in x and y
e = np.linspace(0, L, NX + 1)

fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.25), gridspec_kw={"hspace": 0.35, "wspace": 0.08})
for j, (name, fld) in enumerate(F.items()):
    for r, ax in enumerate(axes[:, j]):
        im = ax.pcolormesh(e, e, fld.T, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, shading="flat", rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            title = name if name == "truth" else f"{name}\n(RMSE {rms['free run' if name == 'free run' else 'analysis']:.2f})"
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.add_patch(Rectangle((x0, x0), w, w, fill=False, ec=INK, lw=0.7))
        else:
            ax.set_xlim(x0, x0 + w); ax.set_ylim(x0, x0 + w)
    if j == 0:
        ob = np.arange(0, NX, s) * dx
        X, Y = np.meshgrid(ob[(ob >= x0) & (ob < x0 + w)], ob[(ob >= x0) & (ob < x0 + w)])
        axes[1, 0].plot(X.ravel(), Y.ravel(), ".", color=INK, ms=0.8)
    if name == "analysis error":
        for k in range(23, 32):
            axes[1, j].axvline(k * h, color=INK, lw=0.25, alpha=0.5)
            axes[1, j].axhline(k * h, color=INK, lw=0.25, alpha=0.5)
for ax, letter in zip(axes.ravel(), "abcdefgh"):
    panel_label(ax, letter)
axes[0, 0].set_ylabel(r"$330^2$", fontsize=7.5)
axes[1, 0].set_ylabel(r"$55\times55$ window", fontsize=7.5)
cax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
cb = fig.colorbar(im, cax=cax); cb.ax.set_title(r"$u/\sigma$", fontsize=7.5, pad=4); cb.outline.set_linewidth(0.4)
save(fig, "fig11_da_snapshot", [str(SRC.relative_to(ROOT))])
print(rms)
