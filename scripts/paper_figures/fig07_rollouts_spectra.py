"""Figure 7 (tasks C4 + C9): free rollouts, energy spectra and pattern correlation after transfer.

Rows 1-3, one system each, columns PDE / transferred model / time-averaged spectra:
  1D KS at L = 400 (trained at L = 100): space-time, first 150 time units; spectra at
      L = 100, 200, 400.
  2D KS at 88^2 (trained on windows of 264^2): snapshots 35 time units after a common encoded
      initial condition; shell-summed spectra at 44^2, 66^2, 88^2.
  Kolmogorov flow, alpha = 0.1, at 3x (Lx = 6pi; main-text model of task A10): vorticity
      40 time units after a common initial condition; kinetic-energy spectra at 1x, 2x, 3x.
Row 4: pattern correlation (spatial Pearson) between decoded model and PDE against time in
  Lyapunov times t * lambda_1, lambda_1 of the truth (1D: L = 400 run; 2D KS: mean of the 66^2
  runs, the largest size with a truth; Kolmogorov: mean of the 3x runs).
Fields standardized by the training std; one shared diverging colormap with symmetric limits.
Sources: spectra/*.json (task A6; steps 100-999 of a 1000-step rollout) and
spectra/fields/{ks1d_x4,ks2d_x2,kolm_x3}.npz (eval_spectra_recon.py --fields-out).
"""

import json

import numpy as np

from _style import COL2, EXP, HUE, INK, ROOT, panel_label, plt, save, shade
from _truths import runs

CMAP, VMAX = "RdBu_r", 2.5
inputs = []


def spec(name):
    p = EXP / f"spectra/{name}.json"
    inputs.append(str(p.relative_to(ROOT)))
    d = json.loads(p.read_text())
    return d, np.array(d["k"]), np.array(d["spectrum_truth"]), np.array(d["spectrum_model"])


def fields(name):
    p = EXP / f"spectra/fields/{name}.npz"
    inputs.append(str(p.relative_to(ROOT)))
    return np.load(p)


def corr(F, profile=False):
    """Spatial pattern correlation per snapshot. profile=True (Kolmogorov): first remove the PDE's
    time- and x-mean vorticity profile in y, the part locked to the forcing (anomaly correlation)."""
    T, M = F["truth"].astype(np.float64), F["model"].astype(np.float64)
    if profile:
        clim = T.mean(axis=(0, 1), keepdims=True)  # (1, 1, NY)
        T, M = T - clim, M - clim
    t, m = T.reshape(len(T), -1), M.reshape(len(M), -1)
    t = t - t.mean(1, keepdims=True); m = m - m.mean(1, keepdims=True)
    return (t * m).sum(1) / np.sqrt((t * t).sum(1) * (m * m).sum(1))


def frame(ax):
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


def spectra_panel(ax, items, col, title, ylabel, xmax, ylim, legloc="lower left"):
    for name, t, lab in items:
        _, k, et, em = spec(name)
        m = (k > 0) & (k <= xmax)
        ax.plot(k[m], et[m], "-", color=shade(INK, t), lw=1.4, label=f"PDE, {lab}")
        ax.plot(k[m], em[m], "--", color=shade(col, t), lw=1.1, label=f"model, {lab}")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax); ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    if legloc:
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], color=INK, lw=1.4), Line2D([], [], color=col, lw=1.1, ls="--")],
                  ["PDE", "model"], fontsize=7, loc=legloc, handlelength=1.6, borderaxespad=0.2)


fig = plt.figure(figsize=(COL2, 5.25))
gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 0.05, 0.12, 1.25], height_ratios=[1.2, 1.2, 0.72, 0.85],
                      hspace=0.62, wspace=0.12)
a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
c = fig.add_subplot(gs[0, 4])
d_, e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
f = fig.add_subplot(gs[1, 4])
g, h_ = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
i_ = fig.add_subplot(gs[2, 4])
cax = fig.add_subplot(gs[:3, 2])
j = fig.add_subplot(gs[3, :])
for ax in (a, b, d_, e, g, h_):
    frame(ax)

# ---------------- 1D KS, L = 400 ----------------
j4, *_ = spec("ks1d_x4")
F1 = fields("ks1d_x4")
std1, L1 = j4["normalization_std"], j4["L"]
T1 = 150
xe = np.linspace(0, L1, F1["truth"].shape[1] + 1)
te = (np.arange(T1 + 1) - 0.5) * float(F1["dt_snap"])
for ax, key, title in ((a, "truth", "PDE"), (b, "model", "model")):
    im = ax.pcolormesh(xe, te, F1[key][:T1] / std1, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat",
                       rasterized=True)
    ax.set_title(title, loc="left")
a.set_xticks([0, 200, 400]); b.set_xticks([200, 400])
a.set_ylabel("1D KS, $L=400$\n$t$")
b.set_yticklabels([])
spectra_panel(c, [("ks1d_x1", 0.55, "$L$=100"), ("ks1d_x2", 0.3, "$L$=200"), ("ks1d_x4", 0.0, "$L$=400")],
              HUE["ks1d"], "1D KS spectra", r"$\langle|\hat u_k|^2\rangle$", 3.0, (1e-9, 1))


# ---------------- 2D KS, 88^2 ----------------
j2, *_ = spec("ks2d_x2")
F2 = fields("ks2d_x2")
std2, L2 = j2["normalization_std"], j2["Lx"]
t2 = 35.0
idx2 = int(round(t2 / float(F2["dt_snap"])))
e2 = np.linspace(0, L2, F2["truth"].shape[1] + 1)
for ax, key, title in ((d_, "truth", "PDE"), (e, "model", "model")):
    ax.pcolormesh(e2, e2, F2[key][idx2].T / std2, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat",
                  rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
d_.set_xticks([0, 44, 88]); d_.set_yticks([0, 44, 88]); e.set_xticks([44, 88]); e.set_yticks([0, 44, 88])
d_.set_ylabel("2D KS, $88^2$\n$y$")
e.set_yticklabels([])
spectra_panel(f, [("ks2d_x1", 0.55, "$44^2$"), ("ks2d_x1.5", 0.3, "$66^2$"), ("ks2d_x2", 0.0, "$88^2$")],
              HUE["ks2d"], "2D KS spectra", r"$\langle|\hat u_{\mathbf{k}}|^2\rangle$, shell sum", 3.0,
              (1e-7, 10), legloc=None)
f.axvline(2 * np.pi / 5.5, color=INK, lw=0.6, ls=":")
f.text(2 * np.pi / 5.5 + 0.05, 1.5, r"$2\pi/h$", fontsize=7, color=INK)


# ---------------- Kolmogorov, alpha = 0.1, 3x ----------------
j3, *_ = spec("kolm_x3")
F3 = fields("kolm_x3")
std3 = j3["normalization_std"]
Lx3, Ly3 = j3["Lx"], j3["Ly"]
t3 = 40.0
idx3 = int(round(t3 / float(F3["dt_snap"])))
for ax, key, title in ((g, "truth", "PDE"), (h_, "model", "model")):
    ax.pcolormesh(np.linspace(0, Lx3, F3["truth"].shape[1] + 1), np.linspace(0, Ly3, F3["truth"].shape[2] + 1),
                  F3[key][idx3].T / std3, cmap=CMAP, vmin=-VMAX, vmax=VMAX, shading="flat", rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.set_xticks([0, 3 * np.pi, 6 * np.pi]); ax.set_xticklabels(["0", r"$3\pi$", r"$6\pi$"])
g.set_yticks([0, 2 * np.pi]); g.set_yticklabels(["0", r"$2\pi$"]); g.set_ylabel("Kolm., $3\\times$\n$y$")
h_.set_yticks([0, 2 * np.pi]); h_.set_yticklabels([])
h_.set_xticks([3 * np.pi, 6 * np.pi]); h_.set_xticklabels([r"$3\pi$", r"$6\pi$"])
spectra_panel(i_, [("kolm_x1", 0.55, r"$1\times$"), ("kolm_x2", 0.3, r"$2\times$"), ("kolm_x3", 0.0, r"$3\times$")],
              HUE["kolm"], "Kolmogorov spectra", r"$\langle|\hat\omega_{\mathbf{k}}|^2/|\mathbf{k}|^2\rangle$",
              20.0, (1e-9, 1), legloc=None)
i_.set_xlabel(r"wavenumber $|\mathbf{k}|$", labelpad=1)

cb = fig.colorbar(im, cax=cax)
cb.ax.set_title(r"$u/\sigma$", fontsize=7.5, pad=4)
cb.outline.set_linewidth(0.4)

# ---------------- pattern correlation vs Lyapunov time ----------------
lam_1d = json.loads((EXP / "e1_model/spectra_overlay_1d.json").read_text())["400"]["truth_exponents"][0]
inputs.append("artifacts/experiments/e1_model/spectra_overlay_1d.json")
lam_2d = float(np.mean([r["lambda1"] for r in runs("ks2d", 1.5)]))
lam_k = float(np.mean([r["lambda1"] for r in runs("kolm0.1", 3)]))
corr_summary = {}
for F, lam, col, lab, key, prof in ((F1, lam_1d, HUE["ks1d"], r"1D KS, $L=400$", "ks1d_L400", False),
                                    (F2, lam_2d, HUE["ks2d"], r"2D KS, $88^2$", "ks2d_88", False),
                                    (F3, lam_k, HUE["kolm"], r"Kolmogorov, $3\times$ (anomaly)", "kolm_a0.1_x3",
                                     True)):
    r = corr(F, prof)
    tt = np.arange(len(r)) * float(F["dt_snap"]) * lam
    j.plot(tt, r, "-", color=col, lw=1.3, label=lab)
    below = np.nonzero(r < 0.5)[0]
    corr_summary[key] = {"lambda1": lam, "dt_snap": float(F["dt_snap"]),
                         "lyap_times_to_corr_0.5": float(tt[below[0]]) if len(below) else None,
                         "corr_at_0": float(r[0]), "mean_corr_last_half": float(r[len(r) // 2:].mean()),
                         "anomaly_wrt_y_profile": prof}
j.axhline(0, color=INK, lw=0.6)
j.set_xlim(0, 8)
j.set_ylim(-0.3, 1.55)
j.set_yticks([0, 0.5, 1])
j.set_xlabel(r"time in Lyapunov times, $t\,\lambda_1$")
j.set_ylabel("pattern correlation")
j.set_title("model vs PDE from a common initial condition", loc="left")
j.legend(loc="upper right", ncol=3, fontsize=7, borderaxespad=0.1)

for ax, letter in zip((a, b, c, d_, e, f, g, h_, i_, j), "abcdefghij"):
    panel_label(ax, letter)

save(fig, "fig07_rollouts_spectra", inputs)
(EXP / "spectra/fields/pattern_correlation_summary.json").write_text(json.dumps(corr_summary, indent=2))
print(json.dumps(corr_summary, indent=1))
