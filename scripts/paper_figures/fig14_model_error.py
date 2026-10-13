"""Fig. 14 (Block K3c, 2026-10-10): learned vs physics-based filters under model error (Sec. da_model_error).

(a) The crossover. x: the physics model's state-dependent one-cycle forecast error (std of the one-cycle
    residual after removing its mean, which bias correction removes), in units of the observation error;
    y: the physics-based filter's error over the latent filter's (3-seed means), tuned only (open) and with
    the best remedy (filled). Below 1 the physics-based filter is better. Hidden-field cases (J17B, diamonds)
    and missing-process cases (J17G, squares) use their own latent model and observation error.
(b) Tuning sensitivity: physics-based filter error against the RTPS weight for three wrong models (log y),
    with the latent filter's single default (RTPS 0.7) as a band.
Inputs: da2d/model_error_summary.json (tab07_p2_model_error.py), j16_model_error_*.json, i11_* RTPS sweeps.
"""
import json

import numpy as np

from _style import AQUA, COL2, EXP, INK, INK2, MUTED, ORANGE, YELLOW, panel_label, plt, save

D = EXP / "da2d"
inputs = ["artifacts/experiments/da2d/model_error_summary.json"]
SUM = json.loads((D / "model_error_summary.json").read_text())
SOBS = 0.1 * 6.172126667325263  # observation error, field units (0.1 x training std of p7c12d10_ks2d)
LAT = np.mean([0.324, 0.310, 0.302])

fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.7), gridspec_kw={"wspace": 0.38, "width_ratios": [1.35, 1]})
SHORT = {"$64^2$ grid": "64²", "$32^2$ grid": "32²", "forcing $0.2$": "forcing 0.2", "forcing $0.4$": "forcing 0.4",
         "$\\nabla^2$ coefficient $1.1$": "a = 1.1", "$\\nabla^2$ coefficient $1.25$": "a = 1.25",
         "damping order $5$": r"$\gamma = 5$", "damping order $6$": r"$\gamma = 6$", "damping order $8$": r"$\gamma = 8$",
         "advection $1$": "advection 1", "advection $2$": "advection 2"}
OFF = {"forcing $0.2$": (-4, -8), "forcing $0.4$": (-3, 5), "$\\nabla^2$ coefficient $1.1$": (4, -6),
       "$\\nabla^2$ coefficient $1.25$": (-5, 5), "damping order $8$": (-5, -3), "damping order $5$": (-4, -7),
       "damping order $6$": (5, -1)}
a.axhspan(0.9, 1.1, color=AQUA, alpha=0.15, lw=0)
a.axhline(1, color=AQUA, lw=1.0)
a.text(0.0035, 1.13, "latent filter (±10%)", color=AQUA, fontsize=6.3)
for r in SUM:
    if r["case"] in ("true equations",) or r.get("best") is None:
        continue
    if r["case"].startswith(("hidden", "missing")):
        continue
    x = max(r["one_cycle_state"], 0.002) / SOBS
    yt, yb = np.mean(r["tuned"]) / LAT, np.mean(r["best"]) / LAT
    col = INK2 if yb < 0.9 else (ORANGE if yb > 1.1 else MUTED)
    a.plot([x, x], [yt, yb], color=col, lw=0.7, alpha=0.7)
    a.plot(x, yt, "o", mfc="white", mec=col, ms=4)
    a.plot(x, yb, "o", color=col, ms=4)
    off = OFF.get(r["case"], (4, -3))
    a.annotate(SHORT.get(r["case"], r["case"]), (x, yb), textcoords="offset points", xytext=off, fontsize=5.8, color=INK2,
               ha="right" if off[0] < 0 else "left")
for r in SUM:  # hidden-field cases, if both filters are in
    if r["case"].startswith("hidden") and r.get("best") and r.get("latent"):
        eps = r["case"].split("=")[1]
        j = json.loads((D / f"j16_model_error_memB{eps}.json").read_text())
        sd = json.loads((D / f"i11_J17B_e{eps}_pde_c8.25_rtps0.7_s0.json").read_text())["args"]["norm_std"]
        x = j["sigma"] / (0.1 * sd); yb = np.mean(r["best"]) / np.mean(r["latent"])
        a.plot(x, yb, "D", color=YELLOW, ms=4.5)
        a.annotate(eps, (x, yb), textcoords="offset points", xytext=(0, -9), fontsize=5.5, color=INK2, ha="center")
for r in SUM:  # J17G missing-process cases: own latent model, observation error 0.1 x that dataset's std
    if r["case"].startswith("missing") and r.get("best") and r.get("latent"):
        d = r["case"].split("=")[1]
        x = r["one_cycle_state"] / (0.1 * r["norm_std"]); yb = np.mean(r["best"]) / np.mean(r["latent"])
        col = INK2 if yb < 0.9 else (ORANGE if yb > 1.1 else MUTED)
        a.plot(x, yb, "s", color=col, ms=4)
        off = {"0.1": (-1, 5), "0.2": (4, -8), "0.3": (5, -2)}[d]
        a.annotate(d, (x, yb), textcoords="offset points", xytext=off, fontsize=5.5, color=INK2,
                   ha="right" if off[0] < 0 else "left")
a.plot([], [], "o", mfc="white", mec=INK2, ms=4, ls="", label="tuned")
a.plot([], [], "o", color=INK2, ms=4, ls="", label="best remedy")
a.plot([], [], "s", color=INK2, ms=4, ls="", label="missing process, δ")
a.plot([], [], "D", color=YELLOW, ms=4, ls="", label="hidden field, ε")
a.axhline(0.076 / LAT, color=INK, lw=0.7, ls=":")
a.text(0.0035, 0.076 / LAT + 0.04, "true equations", fontsize=6, color=INK)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(0.003, 4); a.set_ylim(0.2, 3.5)
a.set_yticks([0.25, 0.5, 1, 2]); a.set_yticklabels(["0.25", "0.5", "1", "2"]); a.minorticks_off()
a.set_xlabel("state-dependent one-cycle model error / obs. error")
a.set_ylabel("physics-based / latent error")
a.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.07))
panel_label(a, "a")

# (b) RTPS sensitivity
SWEEPS = [(r"$\gamma = 6$", ORANGE, "o", {r: f"J17H_p6_c16_rtps{r}_s0" for r in ("0.5", "0.6", "0.7", "0.9", "1.0")}),
          ("a = 1.25", INK2, "s", {r: f"I28W_lap1.25_rtps{r}" for r in ("0", "0.5", "0.6", "0.7", "0.8", "0.9")}),
          ("advection 1", YELLOW, "^", {r: f"I28W_adv1_rtps{r}" for r in ("0", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")})]
for lab, col, mk, tags in SWEEPS:
    xs, ys = [], []
    for r, t in tags.items():
        p = D / f"i11_{t}.json"
        if not p.exists():
            continue
        inputs.append(str(p.relative_to(EXP.parents[1])))
        e = json.loads(p.read_text())["steady"].get("analysis")
        if e is None or not np.isfinite(e):
            e = 20.0
        xs.append(float(r)); ys.append(min(e, 20.0))
    b.plot(xs, ys, marker=mk, color=col, ms=3.5, lw=1.1, label=lab)
b.axhspan(0.302, 0.324, color=AQUA, alpha=0.3, lw=0)
b.plot([0.7], [LAT], "*", color=AQUA, ms=7, label="latent (one setting)")
b.set_yscale("log"); b.set_ylim(0.15, 25)
b.set_yticks([0.2, 0.5, 1, 2, 5, 20]); b.set_yticklabels(["0.2", "0.5", "1", "2", "5", "20"]); b.minorticks_off()
b.set_xlabel("RTPS weight"); b.set_ylabel("analysis RMSE")
b.legend(fontsize=6, loc="upper left")
panel_label(b, "b")
save(fig, "fig14_model_error", inputs)
