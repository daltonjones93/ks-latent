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
# Panel (a), revised 2026-10-13 (referee 2, M5): x is the state-dependent one-cycle error AFTER the remedy, i.e.
# of the model with the parameters the filter converged to (m5_post_summary.json), so every case is on one axis;
# cases without estimation (grids, forcing with bias correction, damping order 8) use their offline values.
# Outcome by the registered rule (winner >= 10% lower): physics = filled dark, tie = open, latent = filled orange.
POST = {r["case"]: r for r in json.loads((D / "m5_post_summary.json").read_text())}
inputs.append("artifacts/experiments/da2d/m5_post_summary.json")
KEY = {"$\\nabla^2$ coefficient $1.1$": "lap1.1", "$\\nabla^2$ coefficient $1.25$": "lap1.25", "advection $1$": "adv1",
       "advection $2$": "adv2", "damping order $5$": "hyp5", "damping order $6$": "hyp6"}
LAB = {"$64^2$ grid": "64²", "$32^2$ grid": "32²", "forcing $0.2$": "forc 0.2", "forcing $0.4$": "forc 0.4",
       "$\\nabla^2$ coefficient $1.1$": "a=1.1", "$\\nabla^2$ coefficient $1.25$": "a=1.25", "advection $1$": "adv 1",
       "advection $2$": "adv 2", "damping order $5$": r"$\gamma$=5", "damping order $6$": r"$\gamma$=6",
       "damping order $8$": r"$\gamma$=8"}
OFFS = {"adv 2": (5, 3), "adv 1": (5, -8), "a=1.25": (-4, -9), "a=1.1": (-4, 5), "64²": (-5, 5), "32²": (5, -8),
        "forc 0.2": (-5, -9), "forc 0.4": (6, 4)}
def outc(r):
    return "physics" if r <= 0.9 else ("latent" if r >= 1 / 0.9 else "tie")
def draw(x, y, shape, label=None, off=(5, -3)):
    o = outc(y)
    col = INK2 if o == "physics" else (ORANGE if o == "latent" else INK2)
    a.plot(x, y, shape, mfc=("white" if o == "tie" else col), mec=col, mew=1.0, ms=5)
    if label:
        a.annotate(label, (x, y), textcoords="offset points", xytext=off, fontsize=5.5, color=INK2,
                   ha="right" if off[0] < 0 else "left")
a.axhspan(0.9, 1 / 0.9, color=AQUA, alpha=0.15, lw=0)
a.axhline(1, color=AQUA, lw=1.0)
for r in SUM:
    if r["case"] == "true equations" or r.get("best") is None:
        continue
    c = r["case"]
    if c.startswith(("hidden", "missing")):
        continue
    yb = np.mean(r["best"]) / LAT
    x = (POST[KEY[c]]["sigma_after"] if c in KEY else r["one_cycle_state"]) / SOBS
    x = max(x, 0.004)
    lab = LAB.get(c, c)
    draw(x, yb, "o", lab if lab in ("adv 2", r"$\gamma$=5", r"$\gamma$=6", r"$\gamma$=8") else None,
         OFFS.get(lab, (5, -3)))
for r in SUM:  # missing process (squares) and hidden field (diamonds): own latent model and observation error
    c = r["case"]
    if c.startswith("missing") and r.get("best") and r.get("latent"):
        d = c.split("=")[1]; q = POST["gani" + d]
        draw(q["sigma_after"] / (0.1 * q["norm_std"]), np.mean(r["best"]) / np.mean(r["latent"]), "s")
    if c.startswith("hidden") and r.get("best") and r.get("latent"):
        e = c.split("=")[1]; q = POST["memB" + e]
        draw(q["sigma_after"] / (0.1 * q["norm_std"]), np.mean(r["best"]) / np.mean(r["latent"]), "D")
a.axhline(0.076 / LAT, color=INK, lw=0.7, ls=":")
a.text(0.0045, 0.076 / LAT + 0.04, "true equations", fontsize=6, color=INK)
a.axvline(0.19 / SOBS, color=AQUA, lw=0.8, ls="--")
a.text(0.19 / SOBS * 1.08, 3.2, "latent model's\ntotal one-cycle error", fontsize=5.6, color=AQUA, va="top")
for sh, lb in (("o", "2D KS, wrong model"), ("s", "missing process δ"), ("D", "hidden field ε")):
    a.plot([], [], sh, color=INK2, ms=4, ls="", label=lb)
a.plot([], [], "o", color=INK2, ms=4, ls="", label="physics wins (filled)")
a.plot([], [], "o", mfc="white", mec=INK2, ms=4, ls="", label="tie (open)")
a.plot([], [], "o", color=ORANGE, ms=4, ls="", label="latent wins (orange)")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(0.003, 2); a.set_ylim(0.2, 3.5)
a.set_yticks([0.25, 0.5, 1, 2]); a.set_yticklabels(["0.25", "0.5", "1", "2"]); a.minorticks_off()
a.set_xlabel("state-dependent one-cycle error after the remedy / obs. error")
a.set_ylabel("physics-based / latent error")
a.text(0.0042, 0.62, "a, adv 1, grids,\nforcing: x ≈ 0", fontsize=5.3, color=INK2, va="center")
a.legend(fontsize=5.3, loc="lower right", ncol=1, handletextpad=0.3, labelspacing=0.25)
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
