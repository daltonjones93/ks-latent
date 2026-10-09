"""Figure 9 (task F2): 2D KS data assimilation in the local latent field.

(a) Observation Jacobian at 44^2 (256 observations x 384 latent coordinates; rows sorted by
    owning site), computed densely by reverse mode: exactly zero outside the 6-column site
    blocks (exp_f2_jacobian_sparsity.py). First 12 sites shown.
(b) Steady-state skill vs domain area at N = 64: per-site filter, dense filter + GC, and the
    unlocalized dense filter (N = 64 and 128); the truth's n_+ annotated where known.
(c) Per-site filter skill vs ensemble size at 44^2, 88^2, 220^2; dense + GC at 44^2.
(d) Per-cycle analysis error, ensemble spread and free-run error at 330^2 (local, N = 64).
(e) Filter-only peak memory vs latent dimension, per-site and dense + GC, with linear and
    quadratic guides through the 88^2 points.
(f) Skill vs observation spacing in units of the site length, 2D KS (44^2, per-site, N = 64)
    and 1D KS (L = 200, PFF + GC, N = 64).
Sources: artifacts/experiments/da2d/*.json, jacobian_sparsity_L44.npz, cost_summary.json
(tab_da_cost.py), artifacts/da_pff_ltransfer_{d1,d2}_*.json, _truths.py.
"""
import json
import statistics as st

import numpy as np

from _style import ART, BLUE, COL2, EXP, HUE, INK, INK2, MUTED, ORANGE, YELLOW, panel_label, plt, save, shade
from _truths import runs

D = EXP / "da2d"
inputs = []


def jl(p):
    inputs.append(str(p.relative_to(ART.parent)))
    return json.loads(p.read_text())


def steady(tag):
    p = D / f"da2d_{tag}.json"
    if not p.exists():
        return None
    d = jl(p)
    fr, da = d["rmse_free_per_cycle"], d["rmse_da_per_cycle"]
    late = slice(len(da) // 2, None)
    return st.mean(f / a for f, a in zip(fr[late], da[late]))


AQ = HUE["ks2d"]
fig = plt.figure(figsize=(COL2, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.55)
a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
d_, e, f = (fig.add_subplot(gs[1, i]) for i in range(3))

# (a) Jacobian sparsity
Jz = np.load(D / "jacobian_sparsity_L44.npz")
inputs.append("artifacts/experiments/da2d/jacobian_sparsity_L44.npz")
ch = int(Jz["channels"]); nsite = 12
rows = Jz["site"] < nsite
J = np.abs(Jz["J"][rows][:, : nsite * ch])
a.imshow(np.log10(J + 1e-12), cmap="Greys", vmin=-4, vmax=0.5, aspect="auto", interpolation="nearest")
a.grid(False)
for s in range(nsite + 1):
    a.axvline(s * ch - 0.5, color=AQ, lw=0.3)
a.set_xlabel(f"latent coordinate ({ch} per site)")
a.set_ylabel("observation (by site)")
a.set_xticks([0, 6 * ch, 12 * ch - 1]); a.set_xticklabels(["0", "36", "72"])
a.set_yticks([])
a.set_title(r"$|J|$ at $44^2$", fontsize=7.5, pad=8)
panel_label(a, "a")

# (b) skill vs area
L_all = [44, 66, 88, 220, 330, 440]
area = lambda L: (L / 44) ** 2  # noqa: E731
pre = lambda L: "A" if L <= 88 else "C"  # noqa: E731
for tagf, lab, col, mk, ls in ((lambda L: f"{pre(L)}_L{L}_local_N64_s0", "per-site", AQ, "o", "-"),
                               (lambda L: f"{pre(L)}_L{L}_denseGC_N64_s0", "dense GC", INK2, "s", "-"),
                               (lambda L: f"A_L{L}_denseNone_N64_s0", "none", ORANGE, "^", "-"),
                               (lambda L: f"A_L{L}_denseNone_N128_s0", "none, 128", ORANGE, "v", "--")):
    pts = [(area(L), steady(tagf(L))) for L in L_all if steady(tagf(L)) is not None]
    x, y = zip(*pts)
    b.plot(x, y, mk + ls, color=col, mfc="white" if ls == "--" else col, ms=4, lw=1.1, label=lab)
    lx, ly = {"per-site": (100, 4.05), "dense GC": (25, 3.05), "none": (4.6, 0.62), "none, 128": (4.6, 1.25)}[lab]
    b.text(lx, ly, lab, fontsize=7, color=col, ha="right" if lab in ("per-site", "dense GC") else "left",
           va="bottom" if lab in ("per-site", "dense GC") else "center")
b.axhline(1, color=INK2, lw=0.6, ls=":")
b.set_xscale("log")
b.set_xticks([1, 4, 25, 100]); b.set_xticklabels(["1", "4", "25", "100"])
b.minorticks_off()
b.set_xlabel(r"domain area / $44^2$")
b.set_ylabel("steady-state skill")
b.set_ylim(0.5, 4.6)
panel_label(b, "b")

# (c) skill vs ensemble size
NS = [8, 16, 32, 64, 128]
for L, t in ((44, 0.55), (88, 0.25), (220, 0.0)):
    ys = []
    for N in NS:
        tag = f"A_L{L}_local_N{N}_s0" if L <= 88 else (f"C_L{L}_local_N64_s0" if N == 64 else f"D8_L{L}_local_N{N}_s0")
        ys.append(steady(tag))
    c.plot(NS, ys, "o-", color=shade(AQ, -0.45 * (1 - t) + 0.45 * t), ms=3.5, lw=1.1, label=f"${L}^2$")
gc = [(N, steady(f"A_L44_denseGC_N{N}_s0")) for N in NS if steady(f"A_L44_denseGC_N{N}_s0") is not None]
c.plot(*zip(*gc), "s--", color=INK2, mfc="white", ms=3.5, lw=1.0, label=r"dense GC")
c.set_xscale("log", base=2); c.set_xticks(NS); c.set_xticklabels([str(n) for n in NS]); c.minorticks_off()
c.set_xlabel(r"ensemble size $N_{\mathrm{ens}}$")
c.set_ylabel("steady-state skill")
c.set_ylim(0.8, 4.6)
c.legend(fontsize=7, loc="lower right", handlelength=1.0, borderaxespad=0.2, labelspacing=0.3)
panel_label(c, "c")

# (d) per-cycle at 330^2
r = jl(D / "da2d_C_L330_local_N64_s0.json")
cyc = np.arange(1, len(r["rmse_da_per_cycle"]) + 1)
d_.plot(cyc, r["rmse_free_per_cycle"], "-", color=MUTED, lw=1.0, label="free run")
d_.plot(cyc, r["rmse_da_per_cycle"], "-", color=AQ, lw=1.2, label="analysis error")
d_.plot(cyc, r["spread_per_cycle"], "--", color=AQ, lw=1.0, label="spread")
d_.axvspan(51, 100, color="#ecebe7", zorder=0)
d_.text(75, 0.02, "steady state", fontsize=7, color=INK2, ha="center")
d_.set_xlabel("cycle"); d_.set_ylabel("latent RMSE")
d_.set_title(r"$330^2$", fontsize=7.5, pad=8)
d_.set_ylim(0, None); d_.set_xlim(1, 100)
d_.legend(fontsize=7, loc="center right")
panel_label(d_, "d")

# (e) memory vs d
cost = jl(D / "cost_summary.json")
for key, lab, col, mk in (("local_filter_rss_GB", "per-site", AQ, "o"), ("dense_filter_rss_GB", "dense + GC", INK2, "s")):
    pts = [(v["d"], v[key]) for v in cost.values() if v.get(key) is not None]
    e.plot(*zip(*sorted(pts)), mk + "-", color=col, ms=3.5, lw=1.1, label=lab)
dd = np.array([300, 4e4])
r88 = cost["88"]
e.plot(dd, r88["local_filter_rss_GB"] * dd / r88["d"], ":", color=AQ, lw=0.8)
e.plot(dd, r88["dense_filter_rss_GB"] * (dd / r88["d"]) ** 2, ":", color=INK2, lw=0.8)
e.text(3.3e4, r88["local_filter_rss_GB"] * 3.3e4 / r88["d"] * 1.25, r"$\propto d$", fontsize=7, color=AQ, ha="right",
       va="bottom")
e.text(3.3e3, 30, r"$\propto d^2$", fontsize=7, color=INK2, ha="right")
e.set_xscale("log"); e.set_yscale("log")
e.set_xlim(300, 4e4); e.set_ylim(0.05, 60)
e.set_xlabel(r"latent dimension $d_{\mathrm{lat}}$"); e.set_ylabel("filter memory (GB)")
e.legend(fontsize=7, loc="lower right", handlelength=1.2, borderaxespad=0.2)
panel_label(e, "e")

# (f) observation spacing
sp2 = [(st_ * 128 / 128 * 44 / 128 / 5.5, steady(t)) for st_, t in
       ((4, "D2_L44_local_N64_st4_s0"), (8, "A_L44_local_N64_s0"), (16, "D2_L44_local_N64_st16_s0"),
        (32, "D2_L44_local_N64_st32_s0"))]
f.plot(*zip(*sp2), "o-", color=AQ, ms=3.5, lw=1.1, label="2D KS")
sp1 = []
for sp, path in ((1.5625, "da_pff_ltransfer_d2_pff_L200_gc_N64_sp1.5625.json"),
                 (3.125, "da_pff_ltransfer_d1_pff_L200_gaspari_cohn_nens64.json"),
                 (6.25, "da_pff_ltransfer_d2_pff_L200_gc_N64_sp6.25.json"),
                 (12.5, "da_pff_ltransfer_d2_pff_L200_gc_N64_sp12.5.json")):
    sp1.append((sp / 6.25, jl(ART / path)["skill_steady"]))
f.plot(*zip(*sp1), "D-", color=HUE["ks1d"], ms=3.5, lw=1.1, label="1D KS")
f.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.72), borderaxespad=0.2, handlelength=1.2)
f.axvline(1, color=INK2, lw=0.6, ls="--")
f.text(1.06, 7.8, "site\nlength", fontsize=7, color=INK2, va="top")
f.axhline(1, color=INK2, lw=0.6, ls=":")
f.set_xscale("log", base=2)
f.set_xticks([0.25, 0.5, 1, 2]); f.set_xticklabels(["1/4", "1/2", "1", "2"])
f.minorticks_off()
f.set_xlabel("obs. spacing / site length"); f.set_ylabel("steady-state skill")
panel_label(f, "f")

save(fig, "fig09_2d_da", inputs)
