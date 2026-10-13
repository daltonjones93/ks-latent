"""Paper I Fig. 5 (rewrite of 2026-10-13, Block Z11): two training routes, and where the small box fails.

(a-c) Signed relative error in D_KY against domain size for three training recipes:
  small box   -- a small periodic box with a forced halo and propagator reach 3 sites (route a);
  windows     -- windows of one large simulation (route b; the canonical window-trained models);
  full-reach box -- the earlier box-trained models (full reach on the whole 8x8-site box).
Bars: signed error of the mean over evaluations (training seeds x initial conditions); dots: single
evaluations. Bars from a single evaluation are drawn hollow (referee: n=1 must be distinguishable).
(d-f) Share of kinetic energy at |k|<2 in the late half of 1000-step free rollouts (one initial
condition, the same as the PDE run), against channel width: PDE vs small-box and window-trained
models. Reference for errors: means over the truth runs in `_truths.py` (Kolmogorov alpha=0.05 at
1x: each initial condition against its own run). Paper I only; refuses to write into docs/paper1.
"""

import json

import numpy as np
from matplotlib.lines import Line2D

from _style import AQUA, BLUE, COL2, EXP, INK, INK2, MAIN_A01, ORANGE, OUT, load, panel_label, plt, save
from _truths import runs as truth_runs

if "docs/paper1/" in str(OUT) + "/":
    raise SystemExit("fig05_routes.py (Paper I) writes only into docs/paperI; set PAPER_OUT")

inputs: list[str] = []
CATS = [("small box", BLUE, ""), ("windows", ORANGE, "///"), ("full-reach box", AQUA, "\\\\\\")]


def crop(tags, scale):
    out = []
    for tag in tags:
        for ic in (0, 1, 2):
            p = EXP / f"crop/{tag}_x{scale}_seed{ic}.json"
            if p.exists():
                inputs.append(str(p.relative_to(EXP.parents[1])))
                out.append((ic, load(p)["model_D_KY"]))
    return out


def plain(files):
    out = []
    for k, f in enumerate(files):
        inputs.append(f"artifacts/experiments/{f}")
        out.append((k, load(EXP / f)["model_D_KY"]))
    return out


def signed_errors(ev, tsys, size):
    rs = truth_runs(tsys, size)
    if not rs:
        return None
    inputs.extend(r["file"] for r in rs)
    if tsys == "kolm0.05" and size == 1:
        ref = {r["seed"]: r["D_KY"] for r in rs}
        return np.array([100 * (v - ref[ic]) / ref[ic] for ic, v in ev])
    t = float(np.mean([r["D_KY"] for r in rs]))
    return np.array([100 * (v - t) / t for _, v in ev])


C2 = [f"r13_C2_ks2d_s44big_k48_seed{s}" for s in (0, 1, 2)]
PANELS = [
    ("2D KS", "ks2d", [("44$^2$", "1.0", 1.0), ("66$^2$", "1.5", 1.5), ("88$^2$", "2.0", 2.0)], {
        "small box": lambda sf: crop(C2, sf),
        "windows": lambda sf: crop(["p7b_ks2d"], sf),
        "full-reach box": lambda sf: plain({"1.0": [f"icnorm/ks2d_x1_normalized_seed{s}.json" for s in (0, 1, 2)],
                                            "1.5": ["icnorm/ks2d_x1.5_normalized.json",
                                                    "icnorm/ks2d_x1.5_normalized_seed1.json",
                                                    "icnorm/ks2d_x1.5_normalized_seed2.json"]}.get(sf, [])),
    }),
    (r"Kolm., $\alpha=0.1$", "kolm0.1", [("1$\\times$", "1", 1), ("2$\\times$", "2", 2), ("3$\\times$", "3", 3)], {
        # referee 2 (A4): every small-box model at this drag (recipe C seeds 0-2, C2, C3), as at alpha = 0.05
        "small box": lambda sf: crop(["r12_c6_a01_s1x_r3", "r12_c6_a01_s1x_r3_seed1", "r12_c6_a01_s1x_r3_seed2",
                                      "r14_KC2_a0.1_seed0", "r14_KC3_a0.1_seed0"], sf),
        "windows": lambda sf: crop([MAIN_A01], sf),
        "full-reach box": lambda sf: plain({"1": ["c4/s233_a01_x1_m0.json"], "2": ["icnorm/s233_x2_normalized.json"],
                                            "3": ["g3_kolmogorov_alpha_transfer_section233_3x.json"]}[sf]),
    }),
    (r"Kolm., $\alpha=0.05$", "kolm0.05", [("1$\\times$", "1", 1), ("2$\\times$", "2", 2), ("3$\\times$", "3", 3)], {
        "small box": lambda sf: crop(["r4k_c6_a005_s1x_r3", "r14_KC2_a0.05_seed0", "r14_KC3_a0.05_seed0"], sf),
        "windows": lambda sf: crop(["c19_e0d0_c6"], sf),
        "full-reach box": lambda sf: plain({"1": ["c4/s231_a005_x1_m0.json"], "2": ["icnorm/s231_x2_normalized.json"],
                                            "3": []}[sf]),
    }),
]

fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.1), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.45,
                                                                  "wspace": 0.3})
width = 0.24
for col, ((title, tsys, sizes, getters), letter) in enumerate(zip(PANELS, "abc")):
    ax = axes[0, col]
    sizes = [s for s in sizes if truth_runs(tsys, s[2])]  # 2D KS 88^2 enters once its truth exists
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axhspan(-20, 20, color="#f1f0ec", zorder=0, lw=0)
    for gi, (lab, sf, size) in enumerate(sizes):
        present = []
        for cat, color, hatch in CATS:
            ev = getters[cat](sf)
            e = signed_errors(ev, tsys, size) if ev else None
            if e is not None and len(e):
                present.append((cat, color, hatch, e))
        offs = (np.arange(len(present)) - (len(present) - 1) / 2) * width
        for (cat, color, hatch, e), off in zip(present, offs):
            m = e.mean()
            if len(e) > 1:
                ax.bar(gi + off, m, width * 0.92, color=color, hatch=hatch, edgecolor="white", lw=0, zorder=2)
            else:  # a single evaluation: hollow bar
                ax.bar(gi + off, m, width * 0.92, facecolor="white", edgecolor=color, lw=1.0, zorder=2)
            ax.plot(np.full(len(e), gi + off), e, "o", ms=2.4, mfc="white", mec=INK2, mew=0.6, zorder=3)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([s[0] for s in sizes])
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-68, 22)
    panel_label(ax, letter)
axes[0, 0].set_ylabel(r"signed error in $D_{\mathrm{KY}}$ (%)")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, edgecolor="white", lw=0) for _, c, h in CATS]
handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, lw=1.0))
axes[0, 0].legend(handles, [c for c, _, _ in CATS] + ["one evaluation"], loc="lower left", ncol=1,
                  handlelength=1.2, fontsize=6.8)

# ---------------- (d-f) large-scale energy share in free rollouts ----------------
RS = EXP / "rollout_stats"
SERIES = {  # alpha -> [(label, file, color, marker, linestyle)]
    0.1: [("small box", "r12_c6_a01_s1x_r3_seed0_n1000_y1.json", BLUE, "o", "-")],
    0.05: [("small box", "r4k_c6_a005_s1x_r3_seed0_n1000_y1.json", BLUE, "o", "-"),
           ("small box, $4\\times$ data", "r14_KC2_a0.05_seed0_seed0_n1000_y1.json", BLUE, "s", "--"),
           ("windows", "c19_e0d0_c6_seed0_n1000_y1.json", ORANGE, "D", "-")],  # main-text model (B5)
    0.025: [("windows", "k_xs0025_seed0_n1000_y1.json", ORANGE, "D", "-")],
}
for col, ((alpha, series), letter) in enumerate(zip(SERIES.items(), "def")):
    ax = axes[1, col]
    truth = {}
    for lab, fn, color, mk, ls in series:
        d = json.loads((RS / fn).read_text()); inputs.append(f"artifacts/experiments/rollout_stats/{fn}")
        xs, ys = [], []
        for key, e in sorted(d["scales"].items()):
            w = int(key.split("x")[0]); xs.append(w); ys.append(e["model"]["share_k_lt_2_late"])
            if "truth" in e:
                truth.setdefault(w, []).append(e["truth"]["share_k_lt_2_late"])
        ax.plot(xs, ys, ls=ls, marker=mk, color=color, mfc=color, ms=4, lw=1.1, label=lab, zorder=3)
    tw = sorted(truth)
    ax.plot(tw, [np.mean(truth[w]) for w in tw], ls="-", marker="o", color=INK, mfc="white", mec=INK, ms=4.5,
            lw=1.0, label="PDE", zorder=4)
    ax.set_title(rf"$\alpha={alpha}$")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.7, 3.3)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel(r"channel width / $2\pi$")
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6)
    panel_label(ax, letter)
axes[1, 0].set_ylabel(r"energy share at $|\mathbf{k}|<2$")

save(fig, "fig05_routes", inputs)
