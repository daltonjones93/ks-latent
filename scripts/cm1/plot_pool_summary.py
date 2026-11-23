"""10b summary figures: the pool of independently trained F6 models against single models (fields and vote).

Inputs (docs/fpe_report/data/): pool_now_{a,b,c}.json (twoscale_eval: F6, seeds 2, 3, 5 with 24 members each; pool3, pool4 with 24
members), design_now_test.json (ensemble_design: votes of pools of n = 1..4 models x 8 members and of ONE model with n x 8 members),
vote_cutoffs.json (the animation's K of 8). Output: docs/fpe_report/figs_pool/fig_pool_{skill,pervar,vote_n,vote_feat}.png.

  python scripts/cm1/plot_pool_summary.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/fpe_report/data"; OUT = ROOT / "docs/fpe_report/figs_pool"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25})
C_SINGLE, C_P3, C_P4, C_BIG = "#9AA0A6", "#E07B39", "#2A78C2", "#B5446E"
NICE = {"u+": "u′ > 10", "u-": "u′ < −10", "w+": "updraft", "w-": "downdraft", "th-": "cold pool", "th+": "warm θ′",
        "qv+": "moist", "qv-": "dry", "prs+": "high p′", "prs-": "mesolow", "qliq+": "liquid", "qice+": "ice"}
VNICE = {"u": "u", "w": "w", "th": "θ′", "qv": "qv", "prs": "p′", "qliq": "liquid", "qice": "ice"}


def load_fields():
    R, vars_ = {}, None
    for x in "abc":
        d = json.load(open(D / f"pool_now_{x}.json")); R.update(d["runs"]); vars_ = d["vars"]
    lags = sorted({int(L) for r in R.values() for c in r["cases"] for L in c["lags"]})
    out = {}
    for n, r in R.items():
        out[n] = {L: {k: np.mean([c["lags"][str(L)][k] for c in r["cases"]], axis=0) for k in ("rmse", "crps", "spread")} for L in lags}
    return out, vars_, lags


def fig_skill(F, lags):
    singles = [n for n in F if not n.startswith("pool")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw=dict(wspace=0.32, left=0.05, right=0.86, top=0.85, bottom=0.15))
    L = [x for x in lags if x > 0]
    for k, a, t in (("crps", ax[0], "fair CRPS (normalized units)"), ("rmse", ax[1], "ensemble-mean RMSE")):
        for i, n in enumerate(singles):
            a.plot(L, [F[n][x][k].mean() for x in L], "-", color=C_SINGLE, lw=1, label="single models (4 seeds)" if i == 0 else None)
        a.plot(L, [F["pool3"][x][k].mean() for x in L], "-o", color=C_P3, ms=3, lw=1.6, label="pool of 3 models")
        a.plot(L, [F["pool4"][x][k].mean() for x in L], "-o", color=C_P4, ms=3, lw=2, label="pool of 4 models")
        a.set_title(t); a.set_xlabel("lead (min)")
    ax[0].legend(fontsize=7, frameon=False)
    a = ax[2]
    for i, n in enumerate(singles):
        a.plot(L, [F[n][x]["spread"].mean() / F[n][x]["rmse"].mean() for x in L], "-", color=C_SINGLE, lw=1)
    for n, c in (("pool3", C_P3), ("pool4", C_P4)):
        a.plot(L, [F[n][x]["spread"].mean() / F[n][x]["rmse"].mean() for x in L], "-o", color=c, ms=3, lw=1.8)
    a.axhline(1, color="k", lw=0.8, ls=":"); a.set_title("spread / RMSE (1 = calibrated)"); a.set_xlabel("lead (min)")
    a = ax[3]
    for k, ls, lab in (("crps", "-", "CRPS"), ("rmse", "--", "ens.-mean RMSE")):
        base = lambda x: np.mean([F[n][x][k].mean() for n in singles])
        lo = lambda x: min(F[n][x][k].mean() for n in singles); hi = lambda x: max(F[n][x][k].mean() for n in singles)
        a.plot(L, [100 * (F["pool4"][x][k].mean() / base(x) - 1) for x in L], ls, marker="o", color=C_P4, ms=3, label=f"pool of 4: {lab}")
        a.plot(L, [100 * (F["pool3"][x][k].mean() / base(x) - 1) for x in L], ls, marker="o", color=C_P3, ms=3, label=f"pool of 3: {lab}")
        if k == "crps":
            a.fill_between(L, [100 * (lo(x) / base(x) - 1) for x in L], [100 * (hi(x) / base(x) - 1) for x in L], color=C_SINGLE, alpha=0.35,
                           lw=0, label="seed-to-seed range (CRPS)")
    a.axhline(0, color="k", lw=0.8); a.set_title("change vs the mean single model (%)"); a.set_xlabel("lead (min)")
    a.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Pooling independently trained F6 models (24 members in every ensemble; 22 unseen storms x 3 starts = 66 cases)", fontsize=10)
    fig.savefig(OUT / "fig_pool_skill.png"); plt.close(fig)


def fig_pervar(F, vars_):
    singles = [n for n in F if not n.startswith("pool")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw=dict(wspace=0.2, left=0.07, right=0.99, top=0.85, bottom=0.12))
    x = np.arange(len(vars_)); w = 0.2
    for a, k, t in ((ax[0], "crps", "fair CRPS"), (ax[1], "rmse", "ensemble-mean RMSE")):
        for j, (L, c) in enumerate(((15, "#9CC3E6"), (30, "#5B9BD5"), (60, "#2A78C2"), (90, "#1A4A7A"))):
            base = np.mean([F[n][L][k] for n in singles], axis=0)
            a.bar(x + (j - 1.5) * w, 100 * (F["pool4"][L][k] / base - 1), w, color=c, label=f"+{L} min")
        a.set_xticks(x, [VNICE[v] for v in vars_]); a.axhline(0, color="k", lw=0.8)
        a.set_title(f"{t}: pool of 4 vs mean single model (%)")
    ax[0].legend(fontsize=7, frameon=False, ncol=4)
    fig.savefig(OUT / "fig_pool_pervar.png"); plt.close(fig)


def votes():
    d = json.load(open(D / "design_now_test.json")); tab = json.load(open(D / "vote_cutoffs.json"))["cutoffs"]
    M, N, feats, lags = d["members_per_model"], d["n_models"], d["features"], d["config"]["lags"].split(",")
    R = {}
    for kd in ("pool", "single"):
        for nn in range(1, N + 1):
            S = nn * M
            for L in lags:
                for f in feats:
                    v = d["votes"].get(f"{kd}|{nn}|{L}|{f}")
                    if v is None:
                        continue
                    t = tab[f]; k8 = t[L] if L in t else t[min(t, key=lambda q: abs(int(q) - int(L)))]
                    K = max(1, min(S, math.ceil(k8 / 8.0 * S)))
                    n_, h = np.array(v["k_n"]), np.array(v["k_h"]); Y = h.sum(); hit = h[K:].sum(); pa = n_[K:].sum()
                    R[(kd, nn, L, f)] = (hit / (pa + Y - hit), pa / max(Y, 1))
    return R, N, M, feats, lags


def fig_vote_n(R, N, M, feats, lags):
    fig, ax = plt.subplots(1, len(lags), figsize=(11, 3.0), sharey=False, gridspec_kw=dict(wspace=0.25, left=0.06, right=0.99, top=0.83, bottom=0.15))
    n = np.arange(1, N + 1)
    for a, L in zip(ax, lags):
        a.plot(n * M, [np.mean([R[("pool", k, L, f)][0] for f in feats]) for k in n], "-o", color=C_P4, lw=2, ms=4, label="pool of n models x 8")
        a.plot(n * M, [np.mean([R[("single", k, L, f)][0] for f in feats]) for k in n], "-o", color=C_BIG, lw=1.5, ms=4,
               label="ONE model, same members")
        a.set_xticks(n * M); a.set_xlabel("members (n models x 8)"); a.set_title(f"+{L} min")
    ax[0].set_ylabel("mean CSI over 12 features"); ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Vote CSI (the animation's per-feature cutoff, kept as a fraction of the members): model diversity vs member count", fontsize=10)
    fig.savefig(OUT / "fig_pool_vote_n.png"); plt.close(fig)


def fig_vote_feat(R, N, M, feats, lags):
    fig, ax = plt.subplots(2, len(lags), figsize=(13, 5.4), sharex=True, gridspec_kw=dict(hspace=0.25, wspace=0.18, left=0.05, right=0.99, top=0.9, bottom=0.12))
    x = np.arange(len(feats)); w = 0.27
    arms = ((("pool", 1), f"one model, {M} members", C_SINGLE), (("single", N), f"one model, {N * M} members", C_BIG),
            (("pool", N), f"pool of {N} models x {M}", C_P4))
    for j, L in enumerate(lags):
        for i, ((kd, nn), lab, c) in enumerate(arms):
            ax[0, j].bar(x + (i - 1) * w, [R[(kd, nn, L, f)][0] for f in feats], w, color=c, label=lab)
            ax[1, j].bar(x + (i - 1) * w, [R[(kd, nn, L, f)][1] for f in feats], w, color=c)
        ax[0, j].set_title(f"+{L} min: CSI"); ax[1, j].set_title(f"+{L} min: forecast area / true area")
        ax[1, j].axhline(1, color="k", lw=0.8); ax[1, j].set_xticks(x, [NICE[f] for f in feats], rotation=60, ha="right")
        ax[0, j].set_ylim(0, 0.9)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper right", ncol=3)
    fig.suptitle("Vote per feature, 66 cases on 22 unseen storms", fontsize=10, x=0.2)
    fig.savefig(OUT / "fig_pool_vote_feat.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F, vars_, lags = load_fields()
    fig_skill(F, lags); fig_pervar(F, vars_)
    R, N, M, feats, lags_v = votes()
    fig_vote_n(R, N, M, feats, lags_v); fig_vote_feat(R, N, M, feats, lags_v)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
