"""Appendix figure (task C8): three supporting panels.

(a) Per-IC scatter at the native Kolmogorov box, alpha = 0.05: model D_KY from each initial
    condition against the truth run from the same initial condition, for every model with
    three native-size evaluations (canonical model in the Kolmogorov hue, others muted).
(b) Reach/channel ablation at alpha = 0.05 (window training, same data): mean |relative error|
    in D_KY at 1x/2x/3x, dots per IC. Same scoring as Table IV / App. D.
(c) Interaction range of the 1D latent dynamics (task A7): validation R^2 of a single shared
    stencil (linear and quadratic) predicting each site's one-step increment from the sites
    within half-width w (scripts/fit_local_stencil_pde.py).
"""

import glob
import json
import re

import numpy as np

from _style import AQUA, BLUE, COL2, DRAG_COLOR, EXP, INK, INK2, MUTED, ROOT, ORANGE, panel_label, plt, save
from _truths import runs

inputs = []


def jl(p):
    inputs.append(str(p.relative_to(ROOT)))
    return json.loads(p.read_text())


T05 = {s: runs("kolm0.05", s) for s in (1, 2, 3)}
for s in T05:
    inputs.extend(r["file"] for r in T05[s])
MATCH = {r["seed"]: r["D_KY"] for r in T05[1]}
MEAN = {s: float(np.mean([r["D_KY"] for r in T05[s]])) for s in (2, 3)}


def errs(tag, s):
    out = []
    for f in sorted(glob.glob(str(EXP / f"crop/{tag}_x{s}_seed*.json"))):
        ic = int(re.search(r"seed(\d+)", f)[1])
        v = jl(__import__("pathlib").Path(f))["model_D_KY"]
        t = MATCH[ic] if s == 1 else MEAN[s]
        out.append(100 * abs(v - t) / t)
    return np.array(out)


fig, (a, b, c) = plt.subplots(1, 3, figsize=(COL2, 2.35), gridspec_kw={"wspace": 0.42,
                                                                       "width_ratios": [1, 1.25, 1]})

# ---------------- (a) per-IC scatter ----------------
MAIN = "c19_e0d0_c6"
tags = set()
for f in glob.glob(str(EXP / "crop/*_x1_seed*.json")):
    name = f.split("/")[-1]
    if name.startswith("truth"):
        continue
    d = json.loads(open(f).read())
    if d.get("system") != "ks2d" and abs(float(d.get("alpha", -1)) - 0.05) < 1e-9:
        tags.add(re.match(r"(.+)_x1_seed\d+\.json", name)[1])
mk = {0: "o", 1: "s", 2: "^"}
for tag in sorted(tags):
    pts = []
    for ic in (0, 1, 2):
        p = EXP / f"crop/{tag}_x1_seed{ic}.json"
        if p.exists():
            pts.append((ic, MATCH[ic], jl(p)["model_D_KY"]))
    if len(pts) < 3:
        continue
    main = tag == MAIN
    for ic, t, m in pts:
        a.plot(t, m, mk[ic], ms=4.5 if main else 3.2, mfc=DRAG_COLOR[0.05] if main else "white",
               mec=DRAG_COLOR[0.05] if main else MUTED, mew=0.8, zorder=3 if main else 2)
lo, hi = 14, 30
a.plot([lo, hi], [lo, hi], color=INK, lw=0.7, ls="--", zorder=1)
a.set_xlim(19.5, 23.5); a.set_ylim(lo, hi)
a.set_xlabel(r"truth $D_{\mathrm{KY}}$, same IC")
a.set_ylabel(r"model $D_{\mathrm{KY}}$")
a.set_title(r"native box, $\alpha=0.05$", fontsize=7.5, pad=8)
from matplotlib.lines import Line2D  # noqa: E402
a.legend([Line2D([], [], marker=mk[i], ls="", mfc="white", mec=INK2, ms=3.5) for i in (0, 1, 2)]
         + [Line2D([], [], marker="o", ls="", mfc=DRAG_COLOR[0.05], mec=DRAG_COLOR[0.05], ms=4.5)],
         ["IC 0", "IC 1", "IC 2", "canonical"], fontsize=7, loc="upper left", handlelength=0.8, ncol=2,
         columnspacing=0.6)
panel_label(a, "a")

# ---------------- (b) reach / channel ablation ----------------
ABL = [("p2_a005", "reach 6/6, $c=3$", MUTED), ("c19_e6d0", "reach 6/0, $c=3$", AQUA),
       ("c19_e6d6_c6", "reach 6/6, $c=6$", BLUE), ("c19_e0d0_c6", "reach 0/0, $c=6$", ORANGE)]
HATCH = ["", "///", "\\\\\\", "xxx"]  # S6.1: second cue for grayscale (AQUA and ORANGE have equal luminance)
w = 0.19
for i, (tag, lab, col) in enumerate(ABL):
    for s in (1, 2, 3):
        e = errs(tag, s)
        if len(e) == 0:
            continue
        x = s + (i - 1.5) * w
        b.bar(x, e.mean(), w * 0.9, color=col, hatch=HATCH[i], edgecolor="white", lw=0, zorder=2, label=lab if s == 1 else None)
        b.plot(np.full(len(e), x), e, "o", ms=2.3, mfc="white", mec=INK2, mew=0.5, zorder=3)
b.set_xticks([1, 2, 3]); b.set_xticklabels([r"$1\times$", r"$2\times$", r"$3\times$"])
b.set_ylabel(r"$|$error$|$ in $D_{\mathrm{KY}}$ (%)")
b.set_title("reach and channels", fontsize=7.5, pad=8)
b.grid(axis="x", visible=False)
b.legend(fontsize=7, loc="upper right", handlelength=1.0, ncol=1, borderaxespad=0.2)
b.set_ylim(0, 40)
panel_label(b, "b")

# ---------------- (c) stencil R^2 ----------------
for deg, ls, lab in ((2, "-o", "quadratic"), (1, "--s", "linear")):
    d = jl(EXP / f"stencil_r2_1d_deg{deg}.json")
    r = d["results"] if isinstance(d["results"], list) else eval(d["results"], {"np": np})  # noqa: S307
    wv = [x["half_width_sites"] for x in r]
    c.plot(wv, [x["val_R2"] for x in r], ls, color=BLUE, ms=3, lw=1.1, label=lab, mfc="white" if deg == 1 else BLUE)
c.set_xlabel("half-width $w$ (sites)")
c.set_ylabel(r"validation $R^2$")
c.set_ylim(0, 1)
c.set_xticks(range(0, 9, 2))
c.axvspan(3, 4, color="#e9e8e4", zorder=0)
c.text(3.5, 0.3, "interaction\nrange", ha="center", fontsize=7, color=INK2)
c.set_title("1D latent stencil", fontsize=7.5, pad=8)
c.legend(fontsize=7, loc="center right")
panel_label(c, "c")

save(fig, "figC2_appendix_panels", inputs)
