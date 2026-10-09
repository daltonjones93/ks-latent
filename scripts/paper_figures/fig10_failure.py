"""Figure 10 (task C7): failure modes.

(a) The six alpha = 0.1 training variants (Table VIII): mean |relative error| in D_KY (%) at
    1x/2x/3x as a heat map, with the truth-free validation forecast error (k = 8) annotated.
(b) Longer squared-error rollouts lower the dimension (Lemma 3): signed relative error in
    D_KY of three pairs of models that differ in k_max (short -> long), at 1x/2x/3x.
    alpha = 0.05 per-patch c=6: c19_e0d0_c6 (k_max 12) -> k_xs005_noreg (32);
    alpha = 0.05 2D-patch crops: p9_a005 (12) -> p9b_a005 (32, end noise 0.05);
    alpha = 0.1: k_xs01_v2 (16) -> k_xs01 (32, slightly stronger barrier).
    Kolmogorov alpha = 0.05 at 1x is scored per IC against the matched-IC truth (paper
    convention); elsewhere against the mean of the truth runs.
(c) Spurious regimes from short rollouts: share of kinetic energy at |k| < 2 during free
    rollouts at 2x, model p9_a005 (k_max 12) vs the PDE from the same initial conditions,
    and the k_max = 32 retrain p9b_a005 (IC seed 0).
Sources: crop/*.json evaluations, a10_selection/decision.json, rollout_stats/*.json, _truths.py.
"""

import json

import numpy as np

from _style import COL2, DRAG_COLOR, EXP, INK, INK2, MAIN_A01, MUTED, ROOT, ORANGE, panel_label, plt, save, shade
from _truths import runs

inputs = []


def jl(p):
    inputs.append(str(p.relative_to(ROOT)))
    return json.loads(p.read_text())


def evals(tag, s):
    out = []
    for k in range(3):
        p = EXP / f"crop/{tag}_x{s}_seed{k}.json"
        if p.exists():
            out.append((k, jl(p)["model_D_KY"]))
    return out


def signed_err(tag, alpha, s, absolute=False):
    rs = runs(f"kolm{alpha}", s)
    inputs.extend(r["file"] for r in rs)
    mean_t = np.mean([r["D_KY"] for r in rs])
    by_seed = {r["seed"]: r["D_KY"] for r in rs}
    e = []
    for ic, v in evals(tag, s):
        t = by_seed[ic] if (alpha == 0.05 and s == 1) else mean_t
        e.append(100 * (v - t) / t)
    e = np.array(e)
    return np.abs(e).mean() if absolute else e.mean()


fig = plt.figure(figsize=(COL2, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.5)
a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

# ---------------- (a) alpha = 0.1 variants ----------------
TAGS = ["k_xs01", "k_xs01_v2", "k_xs01_v3", "k_xs01_v4", "k_xs01_v5", "k_xs01_v7"]
dec = jl(EXP / "a10_selection/decision.json")
cand = {r["tag"]: r for r in dec["candidates"]}
KMAX = {"k_xs01": 32, "k_xs01_v2": 16, "k_xs01_v3": 20, "k_xs01_v4": 20, "k_xs01_v5": 16, "k_xs01_v7": 16}
E = np.array([[signed_err(t, 0.1, s, absolute=True) for s in (1, 2, 3)] for t in TAGS])
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#ffffff", ORANGE, shade(ORANGE, -0.55)])
a.imshow(E, cmap=cmap, vmin=0, vmax=16, aspect="auto", extent=(-0.5, 2.5, len(TAGS) - 0.5, -0.5))
for i in range(E.shape[0]):
    for j in range(3):
        a.text(j, i, f"{E[i, j]:.1f}", ha="center", va="center", fontsize=7,
               color="white" if E[i, j] > 10 else INK)
    a.text(3.3, i, f"{cand[TAGS[i]]['val_k8']:.4f}", ha="center", va="center", fontsize=7, color=INK2)
a.text(3.3, len(TAGS) - 0.38, "val.", ha="center", va="top", fontsize=7, color=INK2)
a.set_xlim(-0.5, 3.95)
a.set_xticks(range(3)); a.set_xticklabels([r"$1\times$", r"$2\times$", r"$3\times$"])
labels = [f"v{i + 1}" + (r"$^\star$" if t == MAIN_A01 else "") + f" ({KMAX[t]})" for i, t in enumerate(TAGS)]
a.set_yticks(range(len(TAGS))); a.set_yticklabels(labels)
a.tick_params(length=0)
a.grid(False)
for sp in a.spines.values():
    sp.set_visible(False)
a.set_title(r"$|$error$|$ (%), $\alpha=0.1$", fontsize=7.5, pad=9)
panel_label(a, "a")

# ---------------- (b) k_max pairs ----------------
PAIRS = [("c19_e0d0_c6", "k_xs005_noreg", 0.05, r"$\alpha=0.05$, per-patch: 12$\to$32", "s"),
         ("p9_a005", "p9b_a005", 0.05, r"$\alpha=0.05$, 2D crops: 12$\to$32", "D"),
         (MAIN_A01, "k_xs01", 0.1, r"$\alpha=0.1$, v2$\to$v1: 16$\to$32", "o")]
off = {0: -0.18, 1: 0.0, 2: 0.18}
for pi, (short, long_, alpha, lab, mk) in enumerate(PAIRS):
    col = DRAG_COLOR[alpha]
    for s in (1, 2, 3):
        e0, e1 = signed_err(short, alpha, s), signed_err(long_, alpha, s)
        x = s + off[pi]
        b.annotate("", xy=(x, e1), xytext=(x, e0),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7, shrinkA=2.5, shrinkB=2.5))
        b.plot(x, e0, mk, ms=4, mfc="white", mec=col, mew=1.0, zorder=3)
        b.plot(x, e1, mk, ms=4, mfc=col, mec=col, zorder=3, label=lab if s == 1 else None)
b.axhline(0, color=MUTED, lw=0.7)
b.set_xticks([1, 2, 3]); b.set_xticklabels([r"$1\times$", r"$2\times$", r"$3\times$"])
b.set_xlim(0.6, 3.4)
b.set_ylabel(r"signed error in $D_{\mathrm{KY}}$ (%)")
b.set_title(r"raising $k_{\max}$", fontsize=7.5, pad=9)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7, handlelength=1.0, borderaxespad=0.2)
b.grid(axis="x", visible=False)
panel_label(b, "b")

# ---------------- (c) spurious regimes ----------------
dt = 0.2
for seed, ls in ((0, "-"),):
    d = jl(EXP / f"rollout_stats/p9_a005_seed{seed}_n2000.json")
    sc = d["scales"]["2"]
    tm = np.linspace(0, d["n_steps"] * dt, len(sc["model"]["ts_share_k_lt_2"]))
    c.plot(tm, sc["truth"]["ts_share_k_lt_2"], ls, color=INK, lw=1.0, label="PDE")
    c.plot(tm, sc["model"]["ts_share_k_lt_2"], ls, color=DRAG_COLOR[0.05], lw=1.2,
           label=r"$k_{\max}=12$")
d = jl(EXP / "rollout_stats/p9b_a005_seed0_n1000_y1.json")
sc = d["scales"]["2x1"]
tm = np.linspace(0, d["n_steps"] * dt, len(sc["model"]["ts_share_k_lt_2"]))
c.plot(tm, sc["model"]["ts_share_k_lt_2"], "-", color=shade(DRAG_COLOR[0.05], 0.55), lw=1.2,
       label=r"$k_{\max}=32$")
c.set_xlabel("time", labelpad=1)
c.set_ylabel(r"energy share at $|\mathbf{k}|<2$")
c.set_ylim(0, 0.85)
c.set_xlim(0, 400)
c.set_title(r"regime at $2\times$", fontsize=7.5, pad=9)
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7, ncol=3, handlelength=1.4, columnspacing=0.8)
panel_label(c, "c")

save(fig, "fig10_failure", inputs)
