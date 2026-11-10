"""Figures for the closure and mixed-derivative sections of docs/fpe_report: reads the small JSONs in docs/fpe_report/data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parents[2] / "docs" / "fpe_report"
OUT = D / "figs"
J = lambda n: json.load(open(D / "data" / n))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"base": "#6b6b6b", "w01": "#1f6f8b", "w1": "#d1495b"}

# ---- channels: what each leading channel carries ----
h = J("headP_w01.json")["channel_correlations"]
keys = list(h)
lab = {"theta_sfc": "$\\theta'_{sfc}$", "w_max": "$w_{max}$", "w_min": "$w_{min}$", "u_sfc": "$u'_{sfc}$",
       "u_top": "$u'_{top}$", "condensate": "condensate", "qv_sfc": "$q_{v,sfc}$"}
M = np.array([[h[k][i] for k in keys] for i in range(4)])
fig, ax = plt.subplots(figsize=(5.6, 2.4))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(keys)), [lab[k] for k in keys]); ax.set_yticks(range(4), [f"$z_{i}$" for i in range(4)])
for i in range(4):
    for j in range(len(keys)):
        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M[i, j]) > 0.55 else "black")
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, label="correlation", fraction=0.04, pad=0.02)
fig.savefig(OUT / "fig_channels.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- closure vs fidelity ----
runs = [("base", "no loss", J("head_base.json")), ("w01", "$w=0.1$", J("head_w01.json")), ("w1", "$w=1$", J("head_w1.json"))]
fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.7), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
x = np.arange(4); wd = 0.26
for a, key, title in ((ax[0], "closure_R2_test", "closure: $R^2$ of $F_i$ on channels $0..3$ only"),
                      (ax[1], "F_vs_data_R2_test", "fidelity: $R^2$ of $F_i$ against the data")):
    for s, (tag, name, r) in enumerate(runs):
        v = [r["channels"][str(i)][key] for i in range(4)]
        a.bar(x + (s - 1) * wd, v, wd, color=C[tag], label=name)
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x, [f"ch {i}" for i in range(4)]); a.set_title(title, fontsize=8.5)
    a.set_ylim(-0.7, 1.05)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
# forecast cost (test storms, results.json of each Stage C run)
disp = {"base": (0.59, 1.12), "w01": (0.65, 1.54), "w1": (0.83, 1.77)}
clo = {t: np.mean([r["channels"][str(i)]["closure_R2_test"] for i in range(4)]) for t, _, r in runs}
for tag, name, _ in runs:
    ax[2].scatter(clo[tag], disp[tag][1], color=C[tag], s=40, zorder=3)
    ax[2].annotate(name, (clo[tag], disp[tag][1]), textcoords="offset points", xytext=(-4, 7), fontsize=8, ha="right" if tag == "w1" else "left")
ax[2].axhline(1.12 * 1.25, color="k", ls=":", lw=0.8); ax[2].text(0.45, 1.12 * 1.25 + 0.03, "pre-registered limit (+25%)", fontsize=7)
ax[2].set_xlabel("mean closure $R^2$"); ax[2].set_ylabel("displacement error at 60 min (km)"); ax[2].set_title("cost of closure", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig_closure.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- mixed derivatives: refit l2 and rollout skill ----
fits = J("fpe_mixed.json")["fits"]
get = lambda f: f["skill"].get("30") or f["skill"].get(30)
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.8))
for thr, mk in ((0.03, "o"), (0.1, "s")):
    s, r = [], []
    for f in fits:
        d = dict(f["z0_terms_refit"] or []).get("d2tz0")
        if f["threshold"] == thr and d is not None and get(f)["law_rt"] < 1:
            s.append(f["l2"]); r.append(d)
    ax[0].scatter(s, r, marker=mk, color="#1f6f8b" if thr == 0.03 else "#d1495b", alpha=0.8, label=f"threshold {thr}", s=28)
ax[0].plot([0.07, 4], [0.07, 4], "k:", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("$\\ell^2$ fixed in the fit (sites$^2$)"); ax[0].set_ylabel("$\\ell^2_0$ after the rollout refit")
ax[0].set_title("the refit does not stay near its start", fontsize=8.5); ax[0].legend(frameon=False, fontsize=7.5)
rng = np.random.default_rng(0)
for f in fits:
    v = get(f)["law_rt"]
    k = f["k"] * (1 + 0.06 * rng.standard_normal())
    if v > 1:
        ax[1].scatter(k, 0.32, marker="x", color="#d1495b", s=30)
    else:
        ax[1].scatter(k, v, color="#1f6f8b", s=22, alpha=0.8)
ax[1].axhline(get(fits[0])["pers"], color="k", ls="--", lw=0.8); ax[1].text(1.0, get(fits[0])["pers"] + 0.008, "persistence", fontsize=7.5)
ax[1].set_xscale("log", base=2); ax[1].set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"]); ax[1].set_ylim(0, 0.35)
ax[1].set_xlabel("latent channels $k$ in the law"); ax[1].set_ylabel("$z_0$ MSE at 30 min (refit law)")
ax[1].set_title("closed rollouts after the refit (x: blew up)", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig_mixed.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written")
