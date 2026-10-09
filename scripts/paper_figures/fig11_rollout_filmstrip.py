"""Figure 11 (task C9): free-rollout filmstrips for the two 2D systems at transfer size.

Fig. 7 already shows one 1D space-time and one 2D KS snapshot; this figure adds what it lacks:
how a 2D rollout departs from the PDE in time, and the Kolmogorov system at all.
Rows 1-2: 2D KS at 88^2 (model trained on 264^2 windows), PDE then model, at four times from a
common encoded initial condition. Rows 3-4: Kolmogorov flow at 3x (Lx = 6*pi), alpha = 0.1,
PDE then model, at the first and last stored times. Bottom: pattern (Pearson) correlation of
model against PDE vs time. It leaves 1 after about 1/lambda_1 (dotted line, 2D KS) and, for 2D KS, falls to ~0.1: the
rollout is statistically, not pointwise, faithful. Kolmogorov flow plateaus near 0.5, since
the forced large-scale pattern keeps model and PDE partly aligned. The correlation of two
independent PDE states is ~0 (2D KS), so 0 is the floor, not a failure.
Fields standardized by the training std; one shared diverging colormap with symmetric limits.

Sources: artifacts/experiments/spectra/fields/{ks2d_x2,kolm_x3}.npz and the matching
spectra/{ks2d_x2,kolm_x3}.json (std). The Kolmogorov fields are not produced by default;
generate them with (CPU, ~minutes; run when the machine is free)
  python scripts/eval_spectra_recon.py --system kolm --scale 3 --n-steps 200 --field-stride 10 \
      --fields-out artifacts/experiments/spectra/fields/kolm_x3.npz
and the std comes from spectra/kolm_x3.json (A6). lambda_1 for 2D KS: mean over
artifacts/experiments/truth_ks2d/truth_ks2d_x1_seed*.json.
"""

import json

import numpy as np

from _style import COL2, EXP, HUE, INK, INK2, MUTED, ROOT, panel_label, plt, save

CMAP, VMAX = "RdBu_r", 2.5
inputs = []


def need(path, hint=""):
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}. {hint}")
    inputs.append(str(path.relative_to(ROOT)))
    return path


def fields(name, hint=""):
    return np.load(need(EXP / f"spectra/fields/{name}.npz", hint))


def std_of(name):
    return json.loads(need(EXP / f"spectra/{name}.json").read_text())["normalization_std"]


def pattern_corr(a, b):
    """Pearson correlation of two fields over all grid points, per frame."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


F2 = fields("ks2d_x2")
FK = fields("kolm_x3", "Generate it with the command in this file's docstring.")
std2, stdk = std_of("ks2d_x2"), std_of("kolm_x3")
dt2, dtk = float(F2["dt_snap"]), float(FK["dt_snap"])
geo2, geok = json.loads(str(F2["geom"])), json.loads(str(FK["geom"]))

lam = [json.loads(need(p).read_text())["lambda1"] for p in sorted((EXP / "truth_ks2d").glob("truth_ks2d_x1_seed*.json"))]
lam1 = float(np.mean(lam))

n2, nk = F2["truth"].shape[0], FK["truth"].shape[0]
near = lambda t, dt, n: min(int(round(t / dt)), n - 1)  # noqa: E731
cols2 = [near(t, dt2, n2) for t in (0, 10, 20, 40)]
colsk = [near(t, dtk, nk) for t in (0, 40, 80)]

fig = plt.figure(figsize=(COL2, 7.3))
gs = fig.add_gridspec(5, 13, height_ratios=[3, 3, 2, 2, 2.6], width_ratios=[1] * 12 + [0.2],
                      hspace=0.34, wspace=0.18)
CMAPS = []


def draw(ax, field, extent, std):
    im = ax.pcolormesh(extent[0], extent[1], field / std, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat",
                       rasterized=True)
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    return im


# ---------------- 2D KS ----------------
e2 = np.linspace(0, geo2["Lx"], F2["truth"].shape[1] + 1)
for r, (key, lab) in enumerate((("truth", "PDE"), ("model", "model"))):
    for c, i in enumerate(cols2):
        ax = fig.add_subplot(gs[r, 3 * c:3 * c + 3])
        im = draw(ax, F2[key][i].T, (e2, e2), std2)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(rf"$t={i * dt2:g}$", loc="left")
        if c == 0:
            ax.set_ylabel(f"2D KS, $88^2$\n{lab}")
        if r == 0 and c == 0:
            panel_label(ax, "a")

# ---------------- Kolmogorov ----------------
ex = np.linspace(0, geok["Lx"], FK["truth"].shape[1] + 1)
ey = np.linspace(0, geok["Ly"], FK["truth"].shape[2] + 1)
for r, (key, lab) in enumerate((("truth", "PDE"), ("model", "model"))):
    for c, i in enumerate(colsk):
        ax = fig.add_subplot(gs[2 + r, 4 * c:4 * c + 4])
        draw(ax, FK[key][i].T, (ex, ey), stdk)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(rf"$t={i * dtk:g}$", loc="left")
        if c == 0:
            ax.set_ylabel(f"Kolm.\n{lab}")
        if r == 0 and c == 0:
            panel_label(ax, "b")

cax = fig.add_subplot(gs[0:4, 12])
cb = fig.colorbar(im, cax=cax)
cb.ax.set_title(r"$u/\sigma$", fontsize=7.5, pad=4)
cb.outline.set_linewidth(0.4)

# ---------------- correlation vs time ----------------
ax = fig.add_subplot(gs[4, 0:12])
ax.plot(np.arange(n2) * dt2, pattern_corr(F2["truth"], F2["model"]), "-", color=HUE["ks2d"],
        label="2D KS, $88^2$")
ax.plot(np.arange(nk) * dtk, pattern_corr(FK["truth"], FK["model"]), "-", color=HUE["kolm"],
        label=rf"Kolmogorov, $\alpha={geok['alpha']:g}$, $3\times$")
ax.axhline(0, color=MUTED, lw=0.6)
ax.axvline(1 / lam1, color=INK2, lw=0.7, ls=":")
ax.text(1 / lam1 + 0.8, 0.5, r"$1/\lambda_1$ (2D KS)", fontsize=7, color=INK2, va="center")
ax.set_xlabel("time since the common initial condition")
ax.set_ylabel("PDE–model correlation")
ax.set_ylim(-0.3, 1.02)
ax.legend(loc="upper right")
panel_label(ax, "c")

save(fig, "fig11_rollout_filmstrip", inputs)
