"""Paper II figure (Z20 W3, 2026-10-21): where the nonlinearity of the latent coordinates starts to matter,
2D KS 44^2, seed 0, the E5 grid (RTPS 0.3 added at 20 snapshots, E5d) (observation error sigma_o x cycle length), standard filter at its best RTPS.

(a) Nonlinearity index eta_o of the forecast ensemble (Eq. nl_index) against the cycle length (snapshots between
    analyses), one line per sigma_o; the dotted line is 0.3, the threshold of the pre-registered rule.
(b) Relative change of the analysis error from relinearizing the update (3 Gauss-Newton iterations), same axes;
    negative = relinearizing helps.
(c) Analysis error against sigma_o at 5 snapshots per cycle: standard latent filter, relinearized, oracle forecast
    (incremental, RTPS 0.3), perfect-model physical-space LETKF (best of RTPS 0 / 0.3, 64 members).
Missing cells are skipped. Inputs: h9_z20_e5_*.json, i11_z20_e5_phys_*.json.
"""
import json

from _style import AQUA, BLUE, COL2, EXP, INK2, MUTED, ORANGE, panel_label, plt, save, shade

D = EXP / "da2d"
inputs = []


def J(name):
    p = D / name
    if not p.exists():
        return None
    inputs.append(f"artifacts/experiments/da2d/{name}")
    s = json.loads(p.read_text())["steady"]
    return None if s.get("diverged_at_cycle") is not None else s


SO = [0.05, 0.1, 0.2, 0.4]
PS = [5, 10, 20]


def best_std(so, p):
    c = [J(f"h9_z20_e5_std_so{so}_p{p}_r{r}.json") for r in (0.3, 0.5, 0.7, 0.9)]
    c = [x for x in c if x]
    return min(c, key=lambda s: s["blend"]) if c else None


cols = {0.05: shade(AQUA, -0.35), 0.1: AQUA, 0.2: BLUE, 0.4: shade(BLUE, -0.4)}
fig, (a, b, c) = plt.subplots(1, 3, figsize=(COL2, 2.0), gridspec_kw={"wspace": 0.55})
for so in SO:
    xs, eta, gain = [], [], []
    for p in PS:
        s = best_std(so, p)
        # 20 snapshots: the relinearized run at RTPS 0.3 (E5d), the best weight there; else the E5 run
        it = (J(f"h9_z20_e5d_it3_so{so}_p{p}_s0.json") if p == 20 else None) or J(f"h9_z20_e5_it3_so{so}_p{p}.json")
        if s is None:
            continue
        xs.append(p); eta.append(s["nl_eta_o"])
        gain.append(100 * (it["blend"] / s["blend"] - 1) if it else float("nan"))
    if xs:
        a.plot(xs, eta, "o-", color=cols[so], ms=3.5, lw=1, label=rf"$\sigma_o={so}$")
        b.plot(xs, gain, "o-", color=cols[so], ms=3.5, lw=1)
a.axhline(0.3, color=MUTED, lw=0.7, ls=":")
for ax in (a, b):
    ax.set_xticks(PS); ax.set_xlabel("snapshots per cycle")
a.set_ylabel(r"$\eta_o$")
fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=7, handlelength=1.2, ncol=4, loc="upper left", bbox_to_anchor=(0.06, 1.1))
b.axhline(0, color=MUTED, lw=0.6); b.set_ylabel("relinearized error change (%)")
panel_label(a, "a"); panel_label(b, "b")


def phys(so):
    v = [J(f"i11_z20_e5_phys_so{so}_p5_r{r}.json") for r in (0, 0.3)]
    v = [x["analysis"] for x in v if x]
    return min(v) if v else None


series = {
    "latent": [best_std(so, 5)["blend"] if best_std(so, 5) else None for so in SO],
    "relinearized": [(J(f"h9_z20_e5_it3_so{so}_p5.json") or {}).get("blend") for so in SO],
    "oracle": [(J(f"h9_z20_e5_orc_so{so}_p5.json") or {}).get("blend") for so in SO],
    "physics": [phys(so) for so in SO],
}
sty = {"latent": (AQUA, "o-"), "relinearized": (BLUE, "s--"), "oracle": (ORANGE, "^-"), "physics": (INK2, "d:")}
for k, v in series.items():
    pts = [(so, y) for so, y in zip(SO, v) if y is not None]
    if pts:
        c.plot(*zip(*pts), sty[k][1], color=sty[k][0], ms=3.5, lw=1, label=k)
c.set_xscale("log"); c.set_xticks(SO); c.set_xticklabels(["0.05", "0.1", "0.2", "0.4"]); c.minorticks_off()
c.set_xlabel(r"$\sigma_o$ (standardized)"); c.set_ylabel("analysis error")
c.legend(frameon=False, fontsize=7, handlelength=1.6)
panel_label(c, "c")
save(fig, "fig_nonlinearity", inputs)
