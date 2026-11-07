"""Claim 3 (G2): latent DA against the three equation-free baselines, per test storm (seed-averaged), paired.
Prints the paired table and draws fig_claim3.png: for each baseline, its nMSE against the latent DA's on the
same storms (cycled forecast for the EnOI baselines, analysis for observation-only, which has no forecast).
  python scripts/cm1/claim3_fig.py --da ~/cm1_da --out ~/cm1_figs/fig_claim3.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINES = (("claim3_enoi_surrogate", "enoi_surrogate", "prior", "Surrogate EnOI\n(static covariance)"),
             ("claim3_enoi_phys", "enoi_phys", "prior", "Physical-archive EnOI\n(no model)"),
             ("claim3_obs_only", "obs_only", "analysis", "Observation-only analysis\n(no model, no memory)"))


def per(rows, mode, stage):
    return {t: float(np.mean([s[stage]["nmse_mean_dec"] for s in r[mode]])) for t, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da", default="~/cm1_da")
    ap.add_argument("--headline", default="noise_det_0.1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.da).expanduser()
    lat = json.loads((B / a.headline / "rows.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, (name, mode, stage, title) in zip(axs, BASELINES):
        rows = json.loads((B / name / "rows.json").read_text())
        x_, y_ = per(rows, mode, stage), per(lat, "latent", stage)
        ts = sorted(set(x_) & set(y_))
        x, y = np.array([x_[t] for t in ts]), np.array([y_[t] for t in ts])
        d = y - x
        h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        print(f"{mode:16s} {stage:8s} latent {y.mean():.3f}  baseline {x.mean():.3f}  latent - baseline "
              f"{d.mean():+.3f} +- {h:.3f} ({100 * (1 - y.mean() / x.mean()):.0f}% lower)  latent lower on "
              f"{(d < 0).sum()}/{d.size}")
        for pre, lab, c in (("ens_B", "ens_B", "C2"), ("ens_A2", "ens_A2", "C1"), ("mem", "ens_A", "C0")):
            sel = np.array([t.startswith(pre) for t in ts])
            ax.scatter(x[sel], y[sel], s=22, color=c, edgecolor="white", linewidth=0.6, label=lab, zorder=3)
        hi = 1.05 * max(x.max(), y.max())
        ax.plot([0, hi], [0, hi], color="0.6", lw=1, zorder=1)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        what = "analysis" if stage == "analysis" else "cycled forecast"
        ax.set_xlabel(f"baseline nMSE ({what})"); ax.set_ylabel(f"latent DA nMSE ({what})")
        ax.text(0.04, 0.96, f"{100 * (1 - y.mean() / x.mean()):.0f}% lower\nbetter on {(d < 0).sum()}/{d.size}",
                transform=ax.transAxes, va="top", fontsize=8)
    axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
