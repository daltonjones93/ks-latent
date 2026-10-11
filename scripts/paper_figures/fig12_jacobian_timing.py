"""Figure 12 (task F4): wall time of the latent observation Jacobian against latent dimension,
2D KS, observation spacing 2.75: dense reverse mode (jacrev, up to 132^2) vs the coloured
forward-mode construction the per-site filter uses (c = 12 JVPs, Part II model p7c12d10, up to 440^2). Least-squares
power-law slopes in the legend. Source: artifacts/experiments/da2d/jacobian_timing_F4_p7c12d10_ks2d.json
(scripts/exp_f4_jacobian_timing.py)."""
import json

import numpy as np

from _style import COL1, EXP, HUE, INK2, ROOT, plt, save

p = EXP / "da2d/jacobian_timing_F4_p7c12d10_ks2d.json"
R = json.loads(p.read_text())
fig, ax = plt.subplots(figsize=(COL1, 2.4))
for key, lab, col, mk in (("t_jacrev_s", "reverse mode (dense)", INK2, "s"),
                          ("t_coloured_s", "coloured, 12 forward passes", HUE["ks2d"], "o")):
    pts = np.array([(r["d"], r[key]) for r in R if r[key] is not None])
    slope = np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)[0]
    ax.plot(pts[:, 0], pts[:, 1] * 1e3, mk + "-", color=col, ms=4, lw=1.2, label=f"{lab}, slope {slope:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"latent dimension $d_{\mathrm{lat}}$")
ax.set_ylabel("time per Jacobian (ms)")
for r in R:
    if r["L"] in (44, 132, 440):
        ax.annotate(f"${r['L']}^2$", (r["d"], r["t_coloured_s"] * 1e3), textcoords="offset points", xytext=(0, -11),
                    fontsize=7, color=INK2, ha="center")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
ax.set_ylim(0.5, 1500)
save(fig, "fig12_jacobian_timing", [str(p.relative_to(ROOT))])
