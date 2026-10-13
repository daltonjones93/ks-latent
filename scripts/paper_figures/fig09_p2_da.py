"""Fig. 9 (Part II headline, Block K3a, 2026-10-10): latent DA with shifted tilings, model p7c12d10_ks2d.

(a) |J| of the latent observation operator at 44^2 (12 sites shown): exactly block-diagonal, 12 columns
    per site (Prop. blockdiag).
(b) Analysis RMSE (field units, steady state) against domain area: 4 shifted tilings (all seeds) and 1
    tiling; interpolation of the observations; decoder floor with 1 and 4 tilings; perfect-model PDE
    filter at 44^2; unlocalized filters (same model, taper = 1) where run.
(c) RMSE against members per tiling (4 tilings) at 44^2 and 220^2.
(d) Per-cycle analysis error and ensemble spread at 440^2 (d = 76 800), 4 tilings; 1-tiling error.
(e) Equal resources: a fixed total of 64 members split over 1, 2, 4 tilings, and 4 x 64 members on 4
    shifted tilings vs on one tiling (control), 44^2, mean and range over seeds 0-2.
(f) Number of tilings K (64 members each, 44^2): RMSE, with spread/error printed at each point.
Inputs: artifacts/experiments/da2d/h9_*.json, i11_L44_st8_c24_rtps0.json, jacobian_sparsity_L44_p7c12d10_ks2d.npz.
"""
import json

import numpy as np

from _style import AQUA, COL2, EXP, INK, INK2, MUTED, ORANGE, OUT, panel_label, plt, save

D = EXP / "da2d"
inputs = []


def S(tag):
    p = D / f"h9_{tag}.json"
    if not p.exists():
        return None
    inputs.append(str(p.relative_to(EXP.parents[1])))
    d = json.loads(p.read_text())
    return d["steady"] | {"_pc": d["per_cycle"]} if d["steady"].get("diverged_at_cycle") is None else None


# Paper II shows the Jacobian in Fig. 1(b), so its version drops panel (a) and relabels (b)-(f) as (a)-(e).
P2 = "paperII" in str(OUT)
fig = plt.figure(figsize=(COL2, 5.0))
if P2:
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.25)
    a = None
    b, c, d_ = (fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3))
    e, f = fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])
    LET = dict(zip("bcdef", "abcde"))
else:
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.6)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    d_, e, f = (fig.add_subplot(gs[1, i]) for i in range(3))
    LET = {k: k for k in "abcdef"}

# (a) Jacobian
if a is not None:
    Jz = np.load(D / "jacobian_sparsity_L44_p7c12d10_ks2d.npz")
    inputs.append("artifacts/experiments/da2d/jacobian_sparsity_L44_p7c12d10_ks2d.npz")
    ch, nsite = int(Jz["channels"]), 12
    J = np.abs(Jz["J"][Jz["site"] < nsite][:, : nsite * ch])
    a.imshow(np.log10(J + 1e-12), cmap="Greys", vmin=-4, vmax=0.5, aspect="auto", interpolation="nearest")
    a.grid(False)
    for s in range(nsite + 1):
        a.axvline(s * ch - 0.5, color=AQUA, lw=0.3)
    a.set_xlabel(f"latent coordinate ({ch} per site)"); a.set_ylabel("observation (by site)")
    a.set_xticks([0, 6 * ch, nsite * ch - 1]); a.set_xticklabels(["0", str(6 * ch), str(nsite * ch)]); a.set_yticks([])
    a.set_title(r"$|J|$ at $44^2$", fontsize=7.5, pad=8)
    panel_label(a, LET["a"])
    offblock = float(J[np.arange(J.shape[0])[:, None], :].max())  # for the caption check below

# (b) RMSE vs area
SIZES = {44: (["I33_d10_T4_s0", "I33_d10_T4_s1", "I33_d10_T4_s2"], "I33_d10_T1_s0"),
         66: (["K1l_L66_T4_N64"], "K1l_L66_T1_N64"),
         88: (["D10_L88_I1_T4"], "D10_L88_I1_T1"),
         220: (["D10_L220_I1_T4", "D10_L220_I1_T4_s1"], "D10_L220_I1_T1"),
         330: (["D10_L330_I1_T4", "D10_L330_I1_T4_s1"], "D10_L330_I1_T1"),
         440: (["D10_L440_I1_T4"], "D10_L440_I1_T1")}
ar = lambda L: (L / 44) ** 2  # noqa: E731
k4, k1, interp, fl1, fl4 = [], [], [], [], []
for L, (t4, t1) in SIZES.items():
    s4 = [S(t) for t in t4]; s1 = S(t1)
    for i, s in enumerate(s4):
        if s:
            b.plot(ar(L), s["blend"], "o", color=AQUA, ms=3.5 if i == 0 else 2.5, alpha=1 if i == 0 else 0.6)
    if s4[0]:
        k4.append((ar(L), s4[0]["blend"])); interp.append((ar(L), s4[0]["interp"])); fl4.append((ar(L), s4[0]["floorK"]))
    if s1:
        k1.append((ar(L), s1["blend"])); fl1.append((ar(L), s1["floor1"]))
for pts, lab, col, ls, mk in ((k4, "4 tilings", AQUA, "-", None), (k1, "1 tiling", INK2, "-", "s"),
                              (interp, "interpolation", MUTED, "--", None), (fl1, "decoder, 1 tiling", INK2, ":", None),
                              (fl4, "decoder, 4 tilings", AQUA, "-.", None)):
    x, y = zip(*pts)
    b.plot(x, y, ls=ls, marker=mk, color=col, lw=1.0 if ls != "-" else 1.3, ms=3, label=lab)
for N, mk, lab in ((64, "^", "unlocalized, 4 tilings, 64"), (128, "v", "unlocalized, 4 tilings, 128")):
    if N == 128 and not P2:
        continue
    unl = [(ar(L), S(f"K1u_L{L}_T4_N{N}")) for L in (44, 66, 88)]
    unl = [(x, s["blend"]) for x, s in unl if s]
    if unl:
        x, y = zip(*unl)
        b.plot(x, y, mk + ("--" if N == 64 else ":"), color=ORANGE, ms=3, lw=1.0,
               label=lab if P2 else "unlocalized, 4 tilings")
pm = json.loads((D / "i11_L44_st8_c24_rtps0.json").read_text())["steady"]["analysis"]
inputs.append("artifacts/experiments/da2d/i11_L44_st8_c24_rtps0.json")
b.plot([1], [pm], "*", color=INK, ms=6, label="PDE filter, true model")
if P2:  # R8 (2026-10-12): log axis so the failing unlocalized filters (2.26-3.40) stay in the frame
    b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(0.05, 5)
    b.set_yticks([0.1, 0.2, 0.5, 1, 2]); b.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"]); b.minorticks_off()
else:
    b.set_xscale("log"); b.set_ylim(0, 1.32); b.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
b.set_xlabel(r"area / $44^2$"); b.set_ylabel("analysis RMSE")
if P2:  # panel (a)'s legend fills the left of the bottom row
    lax = fig.add_subplot(gs[1, 0:2]); lax.axis("off")
    lax.legend(*b.get_legend_handles_labels(), fontsize=5.8, loc="center", handlelength=1.6,
               bbox_to_anchor=(0.45, 0.5), frameon=False)
else:
    b.legend(fontsize=5.6, loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.6, borderaxespad=0.1,
             bbox_to_anchor=(0.5, 1.03))
panel_label(b, LET["b"])

# (c) members per tiling
for L, tags, col, mk in ((44, {8: "D10_L44_I3_N8", 16: "D10_L44_I3_N16", 32: "D10_L44_I3_N32", 64: "I33_d10_T4_s0",
                                 128: "D10_L44_I3_N128"}, AQUA, "o"),
                         (220, {16: "D10_L220_I3_N16", 32: "D10_L220_I3_N32", 64: "D10_L220_I1_T4",
                                128: "D10_L220_I3_N128"}, INK2, "s")):
    pts = [(N, S(t)) for N, t in tags.items()]
    pts = [(N, s["blend"]) for N, s in pts if s]
    x, y = zip(*pts)
    c.plot(x, y, marker=mk, color=col, ms=3, label=f"${L}^2$")
c.set_xscale("log", base=2); c.set_xticks([8, 16, 32, 64, 128]); c.set_xticklabels(["8", "16", "32", "64", "128"])
c.set_xlabel("members per tiling"); c.set_ylabel("analysis RMSE"); c.set_ylim(0.25, 0.5)
c.legend(fontsize=6.5); panel_label(c, LET["c"])

# (d) per-cycle at 440^2
s4, s1 = S("D10_L440_I1_T4"), S("D10_L440_I1_T1")
cyc = np.arange(1, len(s4["_pc"]["blend"]) + 1)
d_.axvspan(51, cyc[-1], color="#f3f2ee", lw=0)
d_.plot(cyc, s4["_pc"]["blend"], color=AQUA, lw=1.0, label="error, 4 tilings")
d_.plot(cyc, s4["_pc"]["spread_blend"], color=AQUA, lw=0.9, ls="--", label="spread, 4 tilings")
d_.plot(cyc, s1["_pc"]["blend"], color=INK2, lw=0.9, label="error, 1 tiling")
d_.set_xlabel("cycle"); d_.set_ylabel("RMSE / spread"); d_.set_ylim(0, 0.8)
d_.set_title(r"$440^2$, $d_{\mathrm{lat}} = 76\,800$", fontsize=7.5, pad=8)
d_.legend(fontsize=6, loc="upper right"); panel_label(d_, LET["d"])

# (e) equal resources
groups = [("1×64", "D10_L44_I2_T1_N64_s{}"), ("2×32", "D10_L44_I2_T2_N32_s{}"), ("4×16", "D10_L44_I2_T4_N16_s{}"),
          ("one\ntiling", "K2b_ctrl_T4_s{}"), ("shifted", "I33_d10_T4_s{}")]
for i, (lab, pat) in enumerate(groups):
    v = [S(pat.format(k)) for k in range(3)]
    v = [s["blend"] for s in v if s]
    col = AQUA if "shifted" in lab or lab in ("2×32", "4×16") else INK2
    e.bar(i, np.mean(v), color=col, width=0.65, alpha=0.85)
    e.errorbar(i, np.mean(v), yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], color=INK, lw=0.8, capsize=2)
e.axvline(2.5, color=MUTED, lw=0.6, ls=":")
e.set_xticks(range(len(groups))); e.set_xticklabels([g[0].replace("\n", " ") for g in groups], fontsize=5.8, rotation=35, ha="right", rotation_mode="anchor")
e.tick_params(axis="x", length=0, pad=2)
e.set_ylabel("analysis RMSE"); e.set_ylim(0, 0.55); e.grid(axis="x", visible=False)
e.text(1.0, 0.505, "64 in total", ha="center", fontsize=6.2, color=INK2)
e.text(3.5, 0.505, "4×64", ha="center", fontsize=6.2, color=INK2)
panel_label(e, LET["e"])

# (f) number of tilings
pts = [(1, "I33_d10_T1_s0"), (2, "D10_L44_I4_T2"), (4, "I33_d10_T4_s0"), (16, "D10_L44_I4_T16")]
x, y, cal = [], [], []
for K, t in pts:
    s = S(t); x.append(K); y.append(s["blend"]); cal.append(s["calib_blend"])
f.plot(x, y, "o-", color=AQUA, ms=3.5, label="Hann weights")
su = S("D10_L44_I4_T4u")
f.plot([4], [su["blend"]], "D", color=ORANGE, ms=3.5, label="uniform, $K=4$")
for xi, yi, ci in zip(x, y, cal):
    f.annotate(f"{ci:.2f}", (xi, yi), textcoords="offset points", xytext=(3, 4), fontsize=6.2, color=INK2)
f.annotate(f"uniform: {su['calib_blend']:.2f}", (4, su["blend"]), textcoords="offset points", xytext=(-6, -4),
           ha="right", fontsize=6.2, color=ORANGE)
f.set_xscale("log", base=2); f.set_xticks([1, 2, 4, 16]); f.set_xticklabels(["1", "2", "4", "16"])
f.text(0.97, 0.97, "labels: spread/error", transform=f.transAxes, ha="right", va="top", fontsize=6, color=INK2)
f.set_xlabel("tilings $K$ (Hann weights)"); f.set_ylabel("analysis RMSE"); f.set_ylim(0.25, 0.5)
panel_label(f, LET["f"])

save(fig, "fig09_p2_da", inputs)
