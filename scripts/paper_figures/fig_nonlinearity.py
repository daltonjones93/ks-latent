"""Paper II figure (Z20 W3, 2026-10-20): where the nonlinearity of the latent coordinates starts to matter,
2D KS 44^2, seed 0, the E5 grid (observation error sigma_o x cycle length).

(a) Nonlinearity index eta_o of the forecast ensemble (Eq. nl_index) against the prior spread at the observations
    in units of sigma_o, one marker per (sigma_o, cycle length) cell, the standard filter at its best RTPS.
(b) Analysis error over sigma_o (both in field units / standardized units as stored), against sigma_o, one-cycle
    length 5 snapshots: standard latent filter, relinearized (3 iterations), oracle forecast (incremental),
    perfect-model physical-space LETKF (best RTPS of 0 / 0.3).
(c) Rank histograms of the observations among the blended prior members at the paper's configuration (E1),
    seeds 0-2, in 13 bins, normalized by the flat expectation.
Missing cells are skipped. Inputs: h9_z20_e5_*.json, i11_z20_e5_phys_*.json, h9_z20_e1_s*.json.
"""
import glob
import json

import numpy as np

from _style import AQUA, BLUE, COL2, EXP, INK2, MUTED, ORANGE, panel_label, plt, save

D = EXP / "da2d"
inputs = []


def J(path):
    inputs.append(str(path).split("latent_DA/")[-1])
    return json.loads(open(path).read())


SO = [0.05, 0.1, 0.2, 0.4]
PS = [5, 10, 20]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(COL2, 2.0), gridspec_kw={"wspace": 0.5})

# (a)
mk = {5: "o", 10: "s", 20: "^"}
for p in PS:
    xs, ys = [], []
    for so in SO:
        best = None
        for r in (0.5, 0.7, 0.9):
            f = D / f"h9_z20_e5_std_so{so}_p{p}_r{r}.json"
            if f.exists():
                s = J(f)["steady"]
                if s.get("diverged_at_cycle") is None and (best is None or s["blend"] < best["blend"]):
                    best = s
        if best is not None and "nl_eta_o" in best:
            xs.append(best["nl_spread_obs"]); ys.append(best["nl_eta_o"])
    if xs:
        a.loglog(xs, ys, mk[p], color=AQUA, ms=4, label=f"{p} snapshots", mfc="none" if p != 5 else AQUA)
a.set_xlabel(r"prior spread at obs. / $\sigma_o$"); a.set_ylabel(r"$\eta_o$")
a.legend(frameon=False, fontsize=7)
panel_label(a, "a")


# (b)
def err(stem, key="blend", phys=False):
    f = D / f"{stem}.json"
    if not f.exists():
        return None
    s = J(f)["steady"]
    if s.get("diverged_at_cycle") is not None:
        return None
    return s["analysis"] if phys else s[key]


series = {"latent": [], "relinearized": [], "oracle": [], "physics": []}
for so in SO:
    std = [err(f"h9_z20_e5_std_so{so}_p5_r{r}") for r in (0.5, 0.7, 0.9)]
    std = [x for x in std if x is not None]
    series["latent"].append(min(std) if std else None)
    series["relinearized"].append(err(f"h9_z20_e5_it3_so{so}_p5"))
    series["oracle"].append(err(f"h9_z20_e5_orc_so{so}_p5"))
    ph = [err(f"i11_z20_e5_phys_so{so}_p5_r{r}", phys=True) for r in (0, 0.3)]
    ph = [x for x in ph if x is not None]
    series["physics"].append(min(ph) if ph else None)
sty = {"latent": (AQUA, "o-"), "relinearized": (BLUE, "s--"), "oracle": (ORANGE, "^-"), "physics": (INK2, "d:")}
for k, v in series.items():
    pts = [(so, y) for so, y in zip(SO, v) if y is not None]
    if pts:
        b.plot(*zip(*pts), sty[k][1], color=sty[k][0], ms=3.5, lw=1, label=k)
b.set_xscale("log"); b.set_xlabel(r"$\sigma_o$ (standardized)"); b.set_ylabel("analysis error")
b.legend(frameon=False, fontsize=7)
panel_label(b, "b")

# (c)
for s, col in zip(range(3), (AQUA, BLUE, ORANGE)):
    f = D / f"h9_z20_e1_s{s}.json"
    if f.exists():
        h = np.array(J(f)["steady"]["rank_hist_blend"], float)
        nb = 13  # 65 ranks (64 members) in 13 bins of 5
        hb = h[: nb * (len(h) // nb)].reshape(nb, -1).sum(1)
        c.step((np.arange(nb) + 0.5) / nb, hb / hb.mean(), where="mid", color=col, lw=1, label=f"seed {s}")
c.axhline(1, color=MUTED, lw=0.6)
c.set_xlabel("rank of observation / ($N$+1)"); c.set_ylabel("frequency / flat")
c.legend(frameon=False, fontsize=7)
panel_label(c, "c")
save(fig, "fig_nonlinearity", inputs)
