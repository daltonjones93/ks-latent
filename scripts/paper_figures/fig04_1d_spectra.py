"""Figure 4 (task A5): the whole Lyapunov spectrum of the transferred 1D model, not just
D_KY, collapses onto the true spectral density.

(a) lambda_i against i/L for L = 100, 200, 400: truth (neutral ink) and the transferred
    model (blue); domain size encoded by lightness within each hue.
(b) cumulative sum (1/L) sum_{j<=i} lambda_j against i/L. It crosses zero where the
    Kaplan-Yorke construction places D_KY / L, so panel (b) shows the dimension density
    directly.

Model: section224, standardized ICs, exponents averaged index-wise over the three ICs
(artifacts/experiments/e1_model/section224_L{L}_ic{0,1,2}_normalized.json).
Truth: the realization with the most directions at each L
(e1_groundtruth_lyapunov_L{L}_fullspec.json where available).
Also writes artifacts/experiments/e1_model/spectra_overlay_1d.json with the plotted data
and the divergence diagnostics quoted in the text.
"""

import glob
import json

import numpy as np

from _style import BLUE, COL2, EXP, INK, MUTED, load, panel_label, plt, save

Ls = [100, 200, 400]
shade = {100: 0.35, 200: 0.65, 400: 1.0}  # lightness: lighter = smaller domain


def mix(hex_color, t):
    """Blend `hex_color` toward white; t=1 is the full color."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(1 - t * (1 - c))


inputs, data = [], {}
for L in Ls:
    mf = sorted(glob.glob(str(EXP / "e1_model" / f"section224_L{L}_ic*_normalized.json")))
    ms = [load(__import__("pathlib").Path(f)) for f in mf]
    ms = [m for m in ms if m["exponents"] is not None]
    model = np.mean([m["exponents"] for m in ms], axis=0)
    tfiles = glob.glob(str(EXP / f"e1_groundtruth_lyapunov_L{L}*.json"))
    tr = max((load(__import__("pathlib").Path(f)) | {"_f": f} for f in tfiles), key=lambda d: d["n_directions"])
    truth = np.array(tr["exponents"])
    inputs += mf + [tr["_f"]]
    data[L] = {"model": model, "truth": truth, "truth_file": tr["_f"], "n_model_ics": len(ms)}

fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"wspace": 0.42})
summary = {}
for L in Ls:
    m, t = data[L]["model"], data[L]["truth"]
    xm, xt = (np.arange(len(m)) + 0.5) / L, (np.arange(len(t)) + 0.5) / L
    a.plot(xt, t, "-", color=mix("#0b0b0b", shade[L]), lw=1.6, label=f"truth, $L={L}$")
    a.plot(xm, m, "--", color=mix(BLUE, shade[L]), lw=1.2, label=f"model, $L={L}$")
    cm, ct = np.cumsum(m) / L, np.cumsum(t) / L
    b.plot(xt, ct, "-", color=mix("#0b0b0b", shade[L]), lw=1.6)
    b.plot(xm, cm, "--", color=mix(BLUE, shade[L]), lw=1.2)
    n = min(len(m), len(t))
    summary[L] = {
        "truth_file": data[L]["truth_file"], "n_model_ics": data[L]["n_model_ics"],
        "n_compared": n, "i_over_L_max": n / L,
        "mean_abs_diff_positive_part": float(np.mean(np.abs(m[:n] - t[:n])[t[:n] > 0])),
        "mean_abs_diff_all_compared": float(np.mean(np.abs(m[:n] - t[:n]))),
        "max_abs_diff_all_compared": float(np.max(np.abs(m[:n] - t[:n]))),
        "model_n_positive_over_L": float((m > 0).sum() / L), "truth_n_positive_over_L": float((t > 0).sum() / L),
        "model_exponents": m.tolist(), "truth_exponents": t.tolist(),
    }

a.axhline(0, color=MUTED, lw=0.6)
a.set_xlabel(r"index per unit length, $i/L$")
a.set_ylabel(r"Lyapunov exponent $\lambda_i$")
a.set_xlim(0, 0.48)
a.set_ylim(-0.75, 0.13)
h, lab = a.get_legend_handles_labels()
order = [0, 1, 2, 3, 4, 5]  # column-major in a 3-column legend: one column per L (truth over model)
fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=3, fontsize=7,
           handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
panel_label(a, "a")

b.axhline(0, color=MUTED, lw=0.6)
b.axvline(0.226, color=MUTED, lw=0.6, ls=":")
b.text(0.232, 0.0045, r"$0.226$", color=MUTED, fontsize=7)
b.set_xlabel(r"index per unit length, $i/L$")
b.set_ylabel(r"$\frac{1}{L}\sum_{j\leq i}\lambda_j$", labelpad=2)
b.set_xlim(0, 0.32)
b.set_ylim(-0.01, 0.012)
panel_label(b, "b")

save(fig, "fig04_1d_spectra", inputs)
(EXP / "e1_model" / "spectra_overlay_1d.json").write_text(json.dumps(summary, indent=2))
for L, s in summary.items():
    print(L, {k: v for k, v in s.items() if not k.endswith("exponents")})
