"""10d: figures of the nested-dropout latent (from latent_physics.py output).

  python scripts/cm1/plot_latent_physics.py --data DIR --out docs/fpe_report/figs_pool
writes fig_nested_order.png, fig_nested_meaning.png, fig_nested_footprint.png, fig_nested_hov.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDC = ["#1F8A4B", "#2A78C2", "#E07B39", "#8E5BB5"]
VLAB = {"u": "u (m/s)", "w": "w (m/s)", "th": "θ′ (K)", "qv": "q$_v$′ (g/kg)", "prs": "p′ (hPa)", "qliq": "√q$_{liq}$", "qice": "√q$_{ice}$"}
VCOL = {"u": "#2A78C2", "w": "#C2402A", "th": "#E07B39", "qv": "#1F8A4B", "prs": "#8E5BB5", "qliq": "#3BA7B5", "qice": "#9AA0A6"}
COVLAB = {"mean u": "column-mean u", "mean w": "column-mean w", "mean th": "column-mean θ′", "mean qv": "column-mean q$_v$′",
          "mean p": "column-mean p′", "mean qliq": "column-mean √q$_{liq}$", "mean qice": "column-mean √q$_{ice}$",
          "surface th": "surface θ′", "max w": "column-max w", "min w": "column-min w", "condensate path": "condensate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data).expanduser(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    r = json.load(open(D / "latent_physics.json"))
    PM, ms, V = r["per_model"], r["ms"], r["vars"]
    nmod = len(PM); lab = [f"model {i + 1}" for i in range(nmod)]
    c = len(PM[0]["std"])

    # ---------- figure 1: ordering, scale, dynamics
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw=dict(wspace=0.32, left=0.045, right=0.99, top=0.84, bottom=0.17))
    for i, R in enumerate(PM):
        ax[0].plot(ms, [R["prefix_first"][str(m)]["all"] for m in ms], "-o", color=SEEDC[i], ms=4, lw=1.8, label=f"{lab[i]}: first m")
        ax[0].plot(ms, [R["prefix_last"][str(m)]["all"] for m in ms], ":", color=SEEDC[i], lw=1.4, label=f"{lab[i]}: last m")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ms); ax[0].set_xticklabels(ms); ax[0].set_ylim(-0.1, 1.02)
    ax[0].set_xlabel("channels kept (m)"); ax[0].set_ylabel("variance explained (all variables)")
    ax[0].set_title("Reconstruction from the first vs the last m channels", fontsize=10)
    ax[0].legend(fontsize=6.5, ncol=2, frameon=False, loc="center right")
    R0 = PM[0]
    for vi, v in enumerate(V):
        ax[1].plot(ms, [R0["prefix_first"][str(m)]["by_var"][vi] for m in ms], "-o", color=VCOL[v], ms=3.5, lw=1.6, label=VLAB[v].split(" ")[0])
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(ms); ax[1].set_xticklabels(ms); ax[1].set_ylim(-0.1, 1.02)
    ax[1].set_xlabel("first m channels kept"); ax[1].set_ylabel("variance explained")
    ax[1].set_title("By variable (model 1)", fontsize=10); ax[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    for i, R in enumerate(PM):
        ax[2].plot(range(c), R["wavelength_km"], "-o", color=SEEDC[i], ms=3, lw=1.4, label=lab[i])
    ax[2].set_yscale("log"); ax[2].set_xlabel("latent channel"); ax[2].set_ylabel("power-weighted wavelength (km)")
    ax[2].set_title("Spatial scale of each channel", fontsize=10); ax[2].legend(fontsize=7.5, frameon=False)
    ks = [f"c{k}" for k in range(8)] + ["g0-3", "g4-7", "g8-15", "g16-31"]
    kl = [str(k) for k in range(8)] + ["0-3", "4-7", "8-15", "16-31"]
    x = np.arange(len(ks))
    fr = np.array([[R["dyn"]["frozen_ratio"][k] for k in ks] for R in PM]); fo = np.array([[R["dyn"]["forced_ratio"][k] for k in ks] for R in PM])
    ax[3].bar(x - 0.2, fr.mean(0), 0.38, color="#C2402A", alpha=0.8, label="frozen at t0 (error ratio)")
    ax[3].bar(x + 0.2, fo.mean(0), 0.38, color="#2A78C2", alpha=0.8, label="forced to truth (error ratio)")
    for i in range(nmod):
        ax[3].plot(x - 0.2, fr[i], "k.", ms=3); ax[3].plot(x + 0.2, fo[i], "k.", ms=3)
    ax[3].axhline(1, color="k", lw=0.8); ax[3].set_xticks(x); ax[3].set_xticklabels(kl, fontsize=8)
    ax[3].set_yscale("log"); ax[3].set_xlabel("latent channel(s)"); ax[3].set_ylabel("30-min error / free forecast")
    ax[3].set_title("Which channels drive the forecast", fontsize=10); ax[3].set_ylim(0.22, 3.2); ax[3].set_yticks([0.25, 0.5, 1, 2, 3]); ax[3].set_yticklabels(["0.25", "0.5", "1", "2", "3"]); ax[3].minorticks_off(); ax[3].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
    for x_ in ax:
        x_.grid(alpha=0.25)
    fig.suptitle(f"Nested dropout orders the latent channels: {nmod} independently trained models, {len(r['names'])} test storms", fontsize=11)
    fig.savefig(O / "fig_nested_order.png", dpi=150); plt.close(fig)

    # ---------- figure 2: meaning (correlation with physical quantities) and consistency across models
    nshow = 12
    fig = plt.figure(figsize=(16, 5.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.6, 0.32, 1, 1, 1], height_ratios=[1, 1], wspace=0.35, hspace=0.45,
                          left=0.06, right=0.99, top=0.88, bottom=0.24)
    axc = fig.add_subplot(gs[:, 0])
    C = np.array(R0["corr"])[:nshow]                     # (nshow, K)
    im = axc.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axc.set_yticks(range(nshow)); axc.set_yticklabels([f"z{k}" for k in range(nshow)])
    axc.set_xticks(range(len(r["cov"]))); axc.set_xticklabels([COVLAB[k] for k in r["cov"]], rotation=40, ha="right", fontsize=8)
    for i in range(nshow):
        for j in range(C.shape[1]):
            if abs(C[i, j]) >= 0.5:
                axc.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if abs(C[i, j]) > 0.75 else "k")
    axc.set_title("Correlation of each latent channel with physical quantities at the same site (model 1)", fontsize=9.5)
    cb = fig.colorbar(im, ax=axc, fraction=0.03, pad=0.01); cb.ax.tick_params(labelsize=7)
    axr = fig.add_subplot(gs[:, 1])
    rr = np.array([R["ridge_r2"][:nshow] for R in PM])
    axr.barh(range(nshow), rr.mean(0), color="#9AA0A6"); axr.plot(rr.T, np.arange(nshow)[:, None].repeat(nmod, 1), "k.", ms=3)
    axr.set_ylim(nshow - 0.5, -0.5); axr.set_yticks([]); axr.set_xlim(0, 1); axr.set_xlabel("R$^2$", fontsize=8)
    axr.set_title("all quantities\n(held out)", fontsize=8.5); axr.tick_params(labelsize=7)
    for ri, M in enumerate(r["consistency"][:3]):
        axm = fig.add_subplot(gs[:, 2 + ri])
        Mm = np.abs(np.array(M))
        axm.imshow(Mm, cmap="Greys", vmin=0, vmax=1)
        for i in range(Mm.shape[0]):
            for j in range(Mm.shape[1]):
                axm.text(j, i, f"{Mm[i, j]:.1f}".lstrip("0"), ha="center", va="center", fontsize=6, color="w" if Mm[i, j] > 0.6 else "k")
        axm.set_xticks(range(8)); axm.set_yticks(range(8)); axm.tick_params(labelsize=7)
        axm.set_xlabel(f"model {ri + 2} channel", fontsize=8); axm.set_ylabel("model 1 channel", fontsize=8)
        axm.set_title(f"|corr| model 1 vs model {ri + 2}", fontsize=9)
    fig.suptitle("What the leading channels represent, and whether independently trained models agree", fontsize=11)
    fig.savefig(O / "fig_nested_meaning.png", dpi=150); plt.close(fig)

    # ---------- figure 3: decoded footprint of a one-site kick, x-z sections
    foot = np.load(D / "foot_0.npy")                     # (nch, V, Z, X)
    zk = np.array(r["z_km"]); iz = zk <= 16
    show = ["u", "w", "th", "qv", "prs", "qliq"]
    nch = 4
    X = foot.shape[-1]; xk = (np.arange(X) - X / 2 + 0.5) * 0.25
    fig, ax = plt.subplots(nch, len(show), figsize=(16, 2.25 * nch + 0.8), sharex=True, sharey=True,
                           gridspec_kw=dict(wspace=0.08, hspace=0.18, left=0.05, right=0.99, top=0.9, bottom=0.07))
    for ci in range(nch):
        for vj, v in enumerate(show):
            f = foot[ci, V.index(v)][iz]
            lim = np.abs(foot[:nch, V.index(v)][:, iz]).max() + 1e-12
            ax[ci, vj].pcolormesh(xk, zk[iz], f, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
            ax[ci, vj].axvline(-1, color="k", lw=0.4, ls=":"); ax[ci, vj].axvline(1, color="k", lw=0.4, ls=":")
            if ci == 0:
                ax[ci, vj].set_title(f"{VLAB[v]}  (±{lim:.2g})", fontsize=9)
            if vj == 0:
                ax[ci, vj].set_ylabel(f"z{ci}\nheight (km)", fontsize=9)
            ax[ci, vj].set_xlim(-12, 12)
            if ci == nch - 1:
                ax[ci, vj].set_xlabel("x from the kicked site (km)", fontsize=8)
    fig.suptitle(f"Physical change from raising one latent channel by one standard deviation at one 2 km site (dotted), model 1, "
                 f"average of {PM[0]['foot_n_cases']} cases at the strongest updraft; red = increase", fontsize=10)
    fig.savefig(O / "fig_nested_footprint.png", dpi=150); plt.close(fig)

    # ---------- figure 4: space-time diagrams
    ph = np.load(D / "hov_phys.npz"); hz = np.load(D / "hov_z_0.npy")    # (T, P, 4)
    T, NX = ph["max w"].shape; P = hz.shape[1]; dt = r.get("dt_min", 1.0)
    t = np.arange(T) * dt / 60; xc = (np.arange(NX) + 0.5) * 0.25; xs = (np.arange(P) + 0.5) * NX / P * 0.25
    panels = [("surface θ′ (K)", xc, ph["surface th"], "RdBu_r", True), ("column-max w (m/s)", xc, ph["max w"], "magma_r", False),
              ("condensate (column mean q)", xc, ph["condensate"] * 1e3, "Blues", False)]
    panels += [(f"latent z{k}", xs, hz[:, :, k], "PuOr_r", True) for k in range(4)]
    fig, ax = plt.subplots(1, len(panels), figsize=(17, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.04, right=0.99, top=0.85, bottom=0.13))
    for axi, (ttl, xx, f, cm, sym) in zip(ax, panels):
        if sym:
            lim = np.percentile(np.abs(f - (np.median(f) if ttl.startswith("latent") else 0)), 99)
            cen = np.median(f) if ttl.startswith("latent") else 0
            axi.pcolormesh(xx, t, f, cmap=cm, vmin=cen - lim, vmax=cen + lim, shading="nearest", rasterized=True)
        else:
            axi.pcolormesh(xx, t, f, cmap=cm, vmin=0, vmax=np.percentile(f, 99.5), shading="nearest", rasterized=True)
        axi.set_title(ttl, fontsize=9.5); axi.set_xlabel("x (km)", fontsize=8.5); axi.tick_params(labelsize=7.5)
    ax[0].set_ylabel("time (h)")
    fig.suptitle(f"One test storm ({r['hov_storm']}): physical fields (left) and the first four latent channels of model 1 (right)", fontsize=11)
    fig.savefig(O / "fig_nested_hov.png", dpi=130); plt.close(fig)
    print("figures in", O)


if __name__ == "__main__":
    main()
