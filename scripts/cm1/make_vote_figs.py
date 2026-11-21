"""9g: how the ensemble vote works: F6's 8 members, a feature called present where at least K of them have it. Run on Saidie.

  fig_vote_walkthrough_<feat>_<storm>.png   one storm, one lead: the truth and the 8 members (field, with the true outline), each member's
                                            0/1 mask, the vote count 0-8, and the K-of-8 decision against the truth as hits / misses /
                                            false alarms; per-storm bias and CSI for every K
  fig_vote_cut_<feat>_<storm>.png           a horizontal cut: the 8 members' fields against the truth and the threshold, the vote count,
                                            and the decision against the true mask
  fig_vote_leads_<storm>.png                updraft vote count and decision at +15 / +30 / +60 min
  fig_vote_features_<storm>.png             every feature at +30 min: truth mask, vote count, decision
  fig_vote_cutoff.png                       from evaluation JSONs: CSI and bias against K for each feature and lead, validation storms (where K
                                            is chosen) and test storms, against the members' own masks and the deterministic F7

  python scripts/cm1/make_vote_figs.py --run ~/cm1_runs/F6_..._stagec --storms mem029 --k 3 --val-eval V.json --test-eval T.json --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.cell_losses import FEATURES  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from make_status_figs import STYLE, phys, section  # noqa: E402

NICE = {"w+": "updraft w > 2 m/s", "w-": "downdraft w < −1.5 m/s", "u+": "u′ > 10 m/s", "u-": "u′ < −10 m/s", "th-": "cold θ′ < −2 K",
        "th+": "warm θ′ > 5 K", "qv+": "moist qv′ > 1 g/kg", "qv-": "dry qv′ < −3 g/kg", "prs+": "mesohigh p′ > 3 hPa",
        "prs-": "mesolow p′ < −1 hPa", "qliq+": "liquid > 0.5 g/kg", "qice+": "ice > 1 g/kg"}
EVAL_KEY = {"w+": "up", "w-": "down"}
# contingency categories: correct negative, hit, miss, false alarm (fixed order and colours)
CONT_COLORS = ["#FFFFFF", "#2A78C2", "#E07B39", "#B5446E"]
CONT = ListedColormap(CONT_COLORS)
CONT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
VOTE = ListedColormap(["#FFFFFF"] + [plt.cm.Greys(x) for x in np.linspace(0.2, 1.0, 8)])
VOTE_NORM = BoundaryNorm(np.arange(-0.5, 9.5), 9)
TRUTH, MUTED = "#3B3F45", "#8A8F96"
MEMBER = "#2A78C2"


def K_of(a, key, L):
    """the cutoff for this feature and lead: the per-feature table (--cutoffs, chosen on the validation storms) or the single --k"""
    if not getattr(a, "table", None):
        return a.k
    t = a.table[key]
    return t[str(L)] if str(L) in t else t[min(t, key=lambda q: abs(int(q) - L))]


def mask(f, key):
    _, sign, thr = FEATURES[key]
    return sign * f > thr


def contingency(dec, y):
    out = np.zeros(y.shape, dtype=int)
    out[dec & y] = 1; out[~dec & y] = 2; out[dec & ~y] = 3
    return out


def scores(dec, y):
    h, m, f = (dec & y).sum(), (~dec & y).sum(), (dec & ~y).sum()
    return (h + f) / max(y.sum(), 1), h / max(h + m + f, 1)


def legend_cont(fig_or_ax, **kw):
    hs = [Patch(facecolor=c, edgecolor="0.5", label=l) for c, l in zip(CONT_COLORS[1:], ("hit", "miss", "false alarm"))]
    fig_or_ax.legend(handles=hs, fontsize=7, **kw)


def forecasts(a, storm, ae, prop, norm, cfg, files, vars_, lags):
    d = ColumnData([load_member(files[storm])], norm, torch.device("cpu")); Z = d.Z
    t0 = int(round(a.start_min / (d.dt / 60)))
    torch.manual_seed(a.seed)
    with torch.no_grad():
        z = encode_members(ae, d)[0]
        lat = rollout_from(prop, z, t0, max(lags), cfg["stage_b"]["halo_sites"], n_samples=a.members, boundary="persistence")
        dec = {L: torch.cat([ae.decode(lat[L - 1, s:s + 1]) for s in range(a.members)]) for L in lags}      # (S, C, NX)
    fc = {L: {v: np.stack([phys(dec[L][s], norm, Z, v) for s in range(a.members)]) for v in vars_} for L in lags}
    tr = {L: {v: phys(d.full(0, t0 + L)[0], norm, Z, v) for v in vars_} for L in lags}
    return fc, tr, d.NX


def walkthrough(a, storm, key, fc, tr, x, zk, out, L=30):
    v = FEATURES[key][0]
    y = mask(tr[L][v], key)
    ms = np.stack([mask(fc[L][v][s], key) for s in range(a.members)])
    votes = ms.sum(0)
    fig = plt.figure(figsize=(20, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[0.8, 0.8, 1.2])
    lim = lambda ax: (ax.set_xlim(*a.xlim), ax.set_ylim(0, 14))
    # row 1: truth field and members 1-5 (fields); row 2: members 6-8 fields? -> keep: truth + 8 member fields over two rows of 9 small panels
    sub = gs[0:2, :].subgridspec(2, 9)
    axs = [fig.add_subplot(sub[0, i]) for i in range(9)] + [fig.add_subplot(sub[1, i]) for i in range(9)]    # fields above, masks below
    panels = [("TRUTH", tr[L][v], y)] + [(f"member {s + 1}", fc[L][v][s], ms[s]) for s in range(a.members)]
    for i, (ttl, f, mk) in enumerate(panels):           # fields (first 9 panels) then masks (next 9)
        ax = axs[i]
        mf = section(ax, f, x, zk, v, title=f"{ttl}: {v}")
        ax.contour(x, zk, y.astype(float), levels=[0.5], colors="k", linewidths=0.5); lim(ax)
        ax2 = axs[9 + i]
        ax2.pcolormesh(x, zk, mk.astype(float), cmap=ListedColormap(["#FFFFFF", TRUTH if i == 0 else MEMBER]), vmin=0, vmax=1, shading="auto", rasterized=True)
        ax2.contour(x, zk, y.astype(float), levels=[0.5], colors="k", linewidths=0.5); lim(ax2)
        ax2.set_title(f"{ttl}: 1[{NICE[key]}]" if i == 0 else f"{ttl}: 0/1 mask", fontsize=7.5)
    for ax in axs:
        ax.tick_params(labelsize=6)
    fig.colorbar(mf, ax=axs[:9], shrink=0.6, label=STYLE[v][0])
    bot = gs[2, :].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.9])
    av = fig.add_subplot(bot[0]); mv = av.pcolormesh(x, zk, votes, cmap=VOTE, norm=VOTE_NORM, shading="auto", rasterized=True)
    av.contour(x, zk, y.astype(float), levels=[0.5], colors="#C0392B", linewidths=0.7); lim(av)
    av.set_title("vote count: members with the feature (0-8)\nred: TRUE outline", fontsize=8.5, loc="left"); av.set_ylabel("z (km)")
    fig.colorbar(mv, ax=av, ticks=range(9), shrink=0.9, label="members")
    Kc = K_of(a, key, L)
    for j, (K, ttl) in enumerate(((Kc, f"DECISION: present if ≥ {Kc} of {a.members}"), (1, "for comparison: ≥ 1 (any member)"))):
        ac = fig.add_subplot(bot[1 + j])
        dec = votes >= K
        ac.pcolormesh(x, zk, contingency(dec, y), cmap=CONT, norm=CONT_NORM, shading="auto", rasterized=True); lim(ac)
        b, c = scores(dec, y)
        ac.set_title(f"{ttl}\nthis case: area/true area {b:.2f}, CSI {c:.2f}", fontsize=8.5, loc="left")
        if j == 0:
            legend_cont(ac, loc="upper right")
    at = fig.add_subplot(bot[3])
    Ks = np.arange(1, a.members + 1)
    bc = np.array([scores(votes >= K, y) for K in Ks])
    sm = np.array([scores(ms[s], y) for s in range(a.members)])
    at.plot(Ks, bc[:, 1], "-o", color=MEMBER, lw=2, ms=4, label="CSI of the vote")
    at.axhline(sm[:, 1].mean(), color=MUTED, lw=1.2, ls="--", label=f"mean CSI of one member ({sm[:, 1].mean():.2f})")
    at.axvline(Kc, color="0.75", lw=6, zorder=0)
    at.set_xlabel(f"K (present if ≥ K of {a.members})"); at.set_ylabel("CSI"); at.set_xticks(Ks); at.grid(alpha=0.25); at.legend(fontsize=7, loc="lower center")
    for K, (bb, cc) in zip(Ks, bc):
        at.annotate(f"{bb:.1f}", (K, cc), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5, color=TRUTH)
    at.set_title("this case: CSI against K\n(numbers: area / true area)", fontsize=8.5, loc="left")
    fig.suptitle(f"The {Kc}-of-{a.members} vote, step by step: {NICE[key]}, {storm} (unseen), +{L} min from t = {a.start_min} min, F6 ensemble "
                 f"(black outline on every panel: the TRUE feature)", fontsize=10)
    fig.savefig(out / f"fig_vote_walkthrough_{key.replace('+', 'p').replace('-', 'm')}_{storm}.png", dpi=125); plt.close(fig)


def cut(a, storm, key, fc, tr, x, zk, out, L=30, z_km=5.0):
    v = FEATURES[key][0]; _, sign, thr = FEATURES[key]
    iz = int(np.argmin(np.abs(zk - z_km)))
    y = mask(tr[L][v], key)[iz]
    ms = np.stack([mask(fc[L][v][s], key)[iz] for s in range(a.members)]); votes = ms.sum(0)
    Kc = K_of(a, key, L)
    fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1.4, 0.8, 0.6]})
    for s in range(a.members):
        axs[0].plot(x, fc[L][v][s][iz], color=MEMBER, lw=0.8, alpha=0.55, label="the 8 members" if s == 0 else None)
    axs[0].plot(x, tr[L][v][iz], color=TRUTH, lw=2, label="truth")
    axs[0].axhline(sign * thr, color="#C0392B", lw=1, ls="--", label=f"threshold ({NICE[key]})")
    axs[0].set_ylabel(STYLE[v][0]); axs[0].legend(fontsize=7.5, ncol=3, loc="upper right"); axs[0].grid(alpha=0.25)
    axs[0].set_title(f"(a) z = {zk[iz]:.1f} km: the members' fields against the truth", fontsize=9, loc="left")
    axs[1].bar(x, votes, width=0.25, color=MEMBER)
    axs[1].axhline(Kc - 0.5, color="#C0392B", lw=1, ls="--", label=f"cutoff: ≥ {Kc} of {a.members}")
    axs[1].set_ylim(0, a.members + 0.3); axs[1].set_yticks(range(0, a.members + 1, 2)); axs[1].set_ylabel("members")
    axs[1].legend(fontsize=7.5, loc="upper right"); axs[1].grid(alpha=0.25)
    axs[1].set_title("(b) the vote: how many members have the feature at each point", fontsize=9, loc="left")
    dec = votes >= Kc
    cat = contingency(dec[None], y[None])[0]
    xe = np.concatenate([x - 0.125, [x[-1] + 0.125]])
    axs[2].pcolormesh(xe, [0, 1], cat[None], cmap=CONT, norm=CONT_NORM, shading="flat")
    axs[2].step(x, np.where(y, 1.6, np.nan), where="mid", color=TRUTH, lw=3)
    axs[2].text(x[0], 1.75, "truth", fontsize=7, color=TRUTH)
    axs[2].set_ylim(0, 2.1); axs[2].set_yticks([]); axs[2].set_xlabel("x (km)")
    legend_cont(axs[2], loc="upper right", ncol=3)
    b, c = scores(dec, y)
    axs[2].set_title(f"(c) the decision (≥ {Kc}) against the truth along this line: area/true area {b:.2f}, CSI {c:.2f}", fontsize=9, loc="left")
    axs[0].set_xlim(*a.xlim)
    fig.suptitle(f"{NICE[key]}: {storm} (unseen), +{L} min, F6's {a.members} members", fontsize=10)
    fig.savefig(out / f"fig_vote_cut_{key.replace('+', 'p').replace('-', 'm')}_{storm}.png", dpi=125, bbox_inches="tight"); plt.close(fig)


def leads(a, storm, fc, tr, x, zk, out, key="w+"):
    """one feature at every lead (the cutoff may change with lead)"""
    v = FEATURES[key][0]
    Ls = sorted(fc)
    fig, axs = plt.subplots(3, len(Ls), figsize=(4.3 * len(Ls), 8.6), sharex=True, sharey=True, constrained_layout=True)
    for j, L in enumerate(Ls):
        y = mask(tr[L][v], key); votes = sum(mask(fc[L][v][s], key).astype(int) for s in range(a.members))
        section(axs[0, j], tr[L][v], x, zk, v, title=f"TRUTH {v}, +{L} min")
        axs[0, j].contour(x, zk, y.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        mv = axs[1, j].pcolormesh(x, zk, votes, cmap=VOTE, norm=VOTE_NORM, shading="auto", rasterized=True)
        axs[1, j].contour(x, zk, y.astype(float), levels=[0.5], colors="#C0392B", linewidths=0.6); axs[1, j].set_title(f"vote count, +{L} min")
        Kc = K_of(a, key, L); dec = votes >= Kc; b, c = scores(dec, y)
        axs[2, j].pcolormesh(x, zk, contingency(dec, y), cmap=CONT, norm=CONT_NORM, shading="auto", rasterized=True)
        axs[2, j].set_title(f"≥ {Kc} of {a.members}, +{L} min: area ratio {b:.2f}, CSI {c:.2f}")
    for ax in axs.flat:
        ax.set_xlim(*a.xlim); ax.set_ylim(0, 14)
    for ax in axs[:, 0]:
        ax.set_ylabel("z (km)")
    for ax in axs[-1]:
        ax.set_xlabel("x (km)")
    fig.colorbar(mv, ax=axs[1], ticks=range(9), shrink=0.8, label="members")
    legend_cont(axs[2, -1], loc="upper right")
    fig.suptitle(f"{NICE[key]} as the lead grows: the members disagree more, the vote count spreads out ({storm}, unseen; red outline: truth)", fontsize=10)
    fig.savefig(out / f"fig_vote_leads_{key.replace('+', 'p').replace('-', 'm')}_{storm}.png", dpi=125); plt.close(fig)


def all_features(a, storm, fc, tr, x, zk, out, L=30):
    keys = list(FEATURES)
    fig, axs = plt.subplots(len(keys), 3, figsize=(11, 1.55 * len(keys) + 0.8), sharex=True, sharey=True, constrained_layout=True)
    for i, key in enumerate(keys):
        v = FEATURES[key][0]
        y = mask(tr[L][v], key); votes = sum(mask(fc[L][v][s], key).astype(int) for s in range(a.members))
        axs[i, 0].pcolormesh(x, zk, y.astype(float), cmap=ListedColormap(["#FFFFFF", TRUTH]), vmin=0, vmax=1, shading="auto", rasterized=True)
        axs[i, 0].set_title(f"TRUTH: {NICE[key]}", fontsize=7.5, loc="left")
        mv = axs[i, 1].pcolormesh(x, zk, votes, cmap=VOTE, norm=VOTE_NORM, shading="auto", rasterized=True)
        axs[i, 1].contour(x, zk, y.astype(float), levels=[0.5], colors="#C0392B", linewidths=0.5)
        axs[i, 1].set_title("vote count", fontsize=7.5, loc="left")
        Kc = K_of(a, key, L); dec = votes >= Kc; b, c = scores(dec, y)
        axs[i, 2].pcolormesh(x, zk, contingency(dec, y), cmap=CONT, norm=CONT_NORM, shading="auto", rasterized=True)
        axs[i, 2].set_title(f"≥ {Kc} of {a.members}: area ratio {b:.2f}, CSI {c:.2f}", fontsize=7.5, loc="left")
    for ax in axs.flat:
        ax.set_xlim(*a.xlim); ax.set_ylim(0, 14); ax.tick_params(labelsize=6)
    for ax in axs[-1]:
        ax.set_xlabel("x (km)")
    fig.colorbar(mv, ax=axs[:, 1], ticks=range(9), shrink=0.3, label="members")
    legend_cont(axs[0, 2], loc="upper right")
    rule = "per-feature cutoff (chosen on validation storms)" if getattr(a, "table", None) else f"the ≥ {a.k} decision"
    fig.suptitle(f"Every feature, +{L} min, {storm} (unseen): truth, vote count of F6's {a.members} members, and {rule}", fontsize=10)
    fig.savefig(out / f"fig_vote_features_{storm}.png", dpi=120); plt.close(fig)


def cutoff_fig(a, out):
    rn = "F6_conv416_mixer512_gelu_pred_all_stagec"; f7 = "F7_conv416_mixer512_det_stagec"
    ev = {"validation storms (K chosen here)": json.load(open(Path(a.val_eval).expanduser())),
          "test storms": json.load(open(Path(a.test_eval).expanduser()))}
    keys = list(FEATURES)
    fig, axs = plt.subplots(3, 4, figsize=(15, 9.5), constrained_layout=True)
    lead_col = {"15": "#9DC3E6", "30": "#2A78C2", "60": "#0B3C6E"}
    for i, key in enumerate(keys):
        ax = axs.flat[i]; ek = EVAL_KEY.get(key, key)
        for tag, r in ev.items():
            for L in ("15", "30", "60"):
                cs = [c["lags"][L]["feat"][ek] for c in r["runs"][rn]["cases"]]
                n = np.sum([c["rel_n"] for c in cs], 0); h = np.sum([c["rel_hits"] for c in cs], 0); Y = h.sum()
                csi = [h[k:].sum() / (n[k:].sum() + Y - h[k:].sum()) for k in range(1, a.members + 1)]
                ax.plot(range(1, a.members + 1), csi, "-o" if tag.startswith("test") else "--", color=lead_col[L], lw=1.8 if tag.startswith("test") else 1.0,
                        ms=3, label=f"+{L} min, {tag}" if i == 0 else None)
                if tag.startswith("test"):
                    ax.axhline(np.mean([c["csi"] for c in cs]), color=lead_col[L], lw=0.8, ls=":")      # the members' own masks
                    if f7 in r["runs"]:
                        c7 = np.mean([c["lags"][L]["feat"][ek]["csi"] for c in r["runs"][f7]["cases"]])
                        ax.plot([a.members + 0.35], [c7], marker="D", color=lead_col[L], ms=4)
        if getattr(a, "table", None):
            for L in ("15", "30", "60"):
                cs = [c["lags"][L]["feat"][ek] for c in ev["test storms"]["runs"][rn]["cases"]]
                n = np.sum([c["rel_n"] for c in cs], 0); h = np.sum([c["rel_hits"] for c in cs], 0); Y = h.sum(); K = K_of(a, key, int(L))
                ax.plot([K], [h[K:].sum() / (n[K:].sum() + Y - h[K:].sum())], marker="*", ms=13, color=lead_col[L], mec="k", mew=0.6, zorder=5)
        else:
            ax.axvline(a.k, color="0.85", lw=8, zorder=0)
        ax.set_title(NICE[key], fontsize=8.5); ax.set_xticks(range(1, a.members + 1)); ax.grid(alpha=0.25)
        ax.set_xlim(0.6, a.members + 0.7)
        if i % 4 == 0:
            ax.set_ylabel("CSI")
        if i >= 8:
            ax.set_xlabel(f"K: present if ≥ K of {a.members}")
    fig.legend(loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Choosing K: CSI of the K-of-{a.members} vote. Solid: test storms; dashed: validation storms (where K is chosen); dotted: CSI of "
                 f"a single member's mask; diamonds: deterministic F7 (2 m/s etc.). "
                 + ("Stars: the cutoff chosen per feature and lead on the validation storms" if getattr(a, "table", None) else f"Grey band: K = {a.k}"), fontsize=9.5)
    fig.savefig(out / ("fig_vote_cutoff_perfeature.png" if getattr(a, "table", None) else "fig_vote_cutoff.png"), dpi=125, bbox_inches="tight"); plt.close(fig)


def summary_fig(a, out):
    """test storms: CSI and area ratio of the per-feature cutoff against K = 3, each feature and lead"""
    rn = "F6_conv416_mixer512_gelu_pred_all_stagec"
    r = json.load(open(Path(a.test_eval).expanduser()))
    keys = list(FEATURES); Ls = ("15", "30", "60")
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    xb = np.arange(len(keys)); wd = 0.13
    for j, L in enumerate(Ls):
        c3, cK, b3, bK, Ks = [], [], [], [], []
        for key in keys:
            cs = [c["lags"][L]["feat"][EVAL_KEY.get(key, key)] for c in r["runs"][rn]["cases"]]
            n = np.sum([c["rel_n"] for c in cs], 0); h = np.sum([c["rel_hits"] for c in cs], 0); Y = h.sum()
            f = lambda K: (h[K:].sum() / (n[K:].sum() + Y - h[K:].sum()), n[K:].sum() / Y)
            K = K_of(a, key, int(L)); Ks.append(K)
            (c_, b_), (cc, bb) = f(3), f(K); c3.append(c_); b3.append(b_); cK.append(cc); bK.append(bb)
        col = {"15": "#9DC3E6", "30": "#2A78C2", "60": "#0B3C6E"}[L]
        off = (j - 1) * 2 * wd
        axs[0].bar(xb + off - wd / 2, c3, wd, color=col, alpha=0.35, label=f"+{L} min, K = 3" if True else None)
        axs[0].bar(xb + off + wd / 2, cK, wd, color=col, label=f"+{L} min, per-feature K")
        axs[1].bar(xb + off - wd / 2, b3, wd, color=col, alpha=0.35)
        axs[1].bar(xb + off + wd / 2, bK, wd, color=col)
        for xi, (K, cc) in enumerate(zip(Ks, cK)):
            axs[0].text(xi + off + wd / 2, cc + 0.01, str(K), ha="center", fontsize=6.5)
    axs[1].axhline(1.0, color="0.4", lw=0.8, ls="--")
    for ax, ttl in zip(axs, ("CSI (number above each bar: the cutoff K)", "forecast area / true area (1 is unbiased)")):
        ax.set_xticks(xb); ax.set_xticklabels([NICE[k] for k in keys], rotation=35, ha="right", fontsize=7.5); ax.set_title(ttl, fontsize=9, loc="left")
        ax.grid(alpha=0.25, axis="y")
    axs[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle("Test storms: per-feature cutoff (solid) against the single cutoff K = 3 (pale), F6's 8 members", fontsize=10)
    fig.savefig(out / "fig_vote_perfeature_summary.png", dpi=125); plt.close(fig)


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--storms", nargs="*", default=[])
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--start-min", type=int, default=60); ap.add_argument("--members", type=int, default=8); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--walk-feats", default="w+,w-,th-,qliq+"); ap.add_argument("--val-eval", default=None); ap.add_argument("--test-eval", default=None)
    ap.add_argument("--cutoffs", default=None, help="9g: JSON with a per-feature, per-lead cutoff table (vote_cutoffs.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.table = json.load(open(Path(a.cutoffs).expanduser()))["cutoffs"] if a.cutoffs else None
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    if a.storms:
        ae, prop, norm, cfg, _ = load_run(Path(a.run).expanduser(), torch.device("cpu"))
        files = member_files(a.source); zk = BaseState.from_sounding().z / 1000.0
        for storm in a.storms:
            fc, tr, NX = forecasts(a, storm, ae, prop, norm, cfg, files, sorted({v for v, _, _ in FEATURES.values()}), [15, 30, 60])
            x = (np.arange(NX) + 0.5) * 0.25
            for key in a.walk_feats.split(","):
                walkthrough(a, storm, key, fc, tr, x, zk, out)
                cut(a, storm, key, fc, tr, x, zk, out)
            for key in a.walk_feats.split(","):
                leads(a, storm, fc, tr, x, zk, out, key)
            all_features(a, storm, fc, tr, x, zk, out)
            print(f"[vote figs] {storm} done", flush=True)
    if a.val_eval and a.test_eval:
        cutoff_fig(a, out)
        if a.table:
            summary_fig(a, out)


if __name__ == "__main__":
    main()
