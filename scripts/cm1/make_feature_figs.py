"""9d: pictures of the updraft / downdraft presence forecasts (the cross-entropy experiment). Run on Saidie (models, caches).

  fig_feat_sections_<storm>.png  rows: truth and each model; columns: w at +30 and +60 min (sample 0) with the TRUE updraft (black) and
                                 downdraft (green) outlines, then P(updraft) and P(downdraft) at +30 min (ensemble fraction; 0/1 for a
                                 deterministic model) with the same outlines
  fig_feat_scores.png            from an evaluation JSON (--eval): CSI, frequency bias and FSS at 10 km for updrafts and downdrafts at
                                 15 / 30 / 60 min, and the reliability of P(updraft) at +30 min for the stochastic models

  python scripts/cm1/make_feature_figs.py --models a=RUN b=RUN --labels a="..." --storms mem029 --eval ts_eval.json --out DIR
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from make_status_figs import phys, section  # noqa: E402

UP, DOWN = 2.0, 1.5
# categorical order (fixed, never cycled): one colour per model in the score figure
CAT = ["#5B5F66", "#2A78C2", "#E07B39", "#B5446E", "#3A9D78", "#8C6BC8", "#C9A227"]


def outlines(ax, x, zk, w_true):
    ax.contour(x, zk, (w_true > UP).astype(float), levels=[0.5], colors="k", linewidths=0.6)
    ax.contour(x, zk, (w_true < -DOWN).astype(float), levels=[0.5], colors="#1B8A4B", linewidths=0.6)


def sections(a, out):
    names = [m.split("=", 1)[0] for m in a.models]; paths = {m.split("=", 1)[0]: Path(m.split("=", 1)[1]).expanduser() for m in a.models}
    text = {k: k for k in names}; text.update({m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.labels})
    torch.set_num_threads(16); dev = torch.device("cpu")
    files = member_files(a.source); zk = BaseState.from_sounding().z / 1000.0
    loaded = {nm: load_run(paths[nm], dev) for nm in names}
    norm0 = loaded[names[0]][2]
    for storm in a.storms:
        fc, truth = {}, {}
        for nm in names:
            ae, prop, norm, cfg, _ = loaded[nm]
            halo = cfg["stage_b"]["halo_sites"]
            d = ColumnData([load_member(files[storm])], norm, dev); Z, NX = d.Z, d.NX; dtm = d.dt / 60
            z = encode_members(ae, d)[0]
            t0 = int(round(a.start_min / dtm))
            S = a.samples if getattr(prop.cfg, "noise_channels", 0) > 0 else 1
            torch.manual_seed(0)
            with torch.no_grad():
                lat = rollout_from(prop, z, t0, 60, halo, n_samples=S, boundary="persistence")
                w = {lag: np.stack([phys(ae.decode(lat[lag - 1, s:s + 1])[0], norm, Z, "w") for s in range(S)]) for lag in (30, 60)}
            fc[nm] = w
            if nm == names[0]:
                truth = {lag: phys(d.full(0, t0 + lag)[0], norm0, Z, "w") for lag in (30, 60)}
        x = (np.arange(NX) + 0.5) * 0.25
        rows = [("truth", None)] + [(text[nm], nm) for nm in names]
        fig, axs = plt.subplots(len(rows), 4, figsize=(11.5, 1.75 * len(rows) + 0.6), sharex=True, sharey=True, constrained_layout=True)
        for r, (lab, nm) in enumerate(rows):
            for c, lag in enumerate((30, 60)):
                f = truth[lag] if nm is None else fc[nm][lag][0]
                mw = section(axs[r, c], f, x, zk, "w", title=f"{lab}: w, +{lag} min" if c == 0 else f"w, +{lag} min")
                outlines(axs[r, c], x, zk, truth[lag])
            for c, (ttl, sgn, thr) in enumerate((("P(updraft > 2 m/s), +30 min", 1, UP), ("P(downdraft < −1.5 m/s), +30 min", -1, DOWN))):
                p = (sgn * truth[30] > thr).astype(float) if nm is None else (sgn * fc[nm][30] > thr).mean(0)
                mp = axs[r, 2 + c].pcolormesh(x, zk, p, cmap="Greys", vmin=0, vmax=1, shading="auto", rasterized=True)
                outlines(axs[r, 2 + c], x, zk, truth[30])
                axs[r, 2 + c].set_title(ttl)
            axs[r, 0].set_ylabel("z (km)")
        for ax in axs.flat:
            ax.set_xlim(a.xlim[0], a.xlim[1]); ax.set_ylim(0, 14)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.colorbar(mw, ax=axs[:, :2], shrink=0.5, label="w (m/s)")
        fig.colorbar(mp, ax=axs[:, 2:], shrink=0.5, label="probability (ensemble fraction; 0 or 1 if deterministic)")
        fig.suptitle(f"{storm} (unseen), from t = {a.start_min} min, held boundaries. Outlines: TRUE updrafts above 2 m/s (black), "
                     f"downdrafts below −1.5 m/s (green)", fontsize=9)
        fig.savefig(out / f"fig_feat_sections_{storm}.png", dpi=130); plt.close(fig)
        print(f"[feat figs] {storm} done", flush=True)


def scores(a, out):
    r = json.load(open(Path(a.eval).expanduser()))
    text = {m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.labels}
    runs = [rn for rn in a.order if rn in r["runs"]] if a.order else list(r["runs"])
    lags = [L for L in ("15", "30", "60") if L in r["runs"][runs[0]]["cases"][0]["lags"]]
    mets = [("csi", "CSI (higher is better)"), ("bias", "frequency bias (1 is unbiased)"), ("fss@10", "FSS at 10 km (higher is better)")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6), constrained_layout=True)
    for i, feat in enumerate(("up", "down")):
        for j, (mk, ttl) in enumerate(mets):
            ax = axs[i, j]
            for k, rn in enumerate(runs):
                v = [np.mean([c["lags"][L]["feat"][feat][mk] for c in r["runs"][rn]["cases"]]) for L in lags]
                ax.plot([int(L) for L in lags], v, "-o", color=CAT[k % len(CAT)], lw=2, ms=4, label=text.get(rn, rn))
            if mk == "bias":
                ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
            ax.set_title(f"{'updrafts (w > 2 m/s)' if feat == 'up' else 'downdrafts (w < −1.5 m/s)'}: {ttl}", fontsize=8)
            ax.set_xticks([15, 30, 60]); ax.set_xlabel("lead (min)"); ax.grid(alpha=0.25)
        ax = axs[i, 3]                                           # reliability at +30 min, stochastic models
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        for k, rn in enumerate(runs):
            cs = [c["lags"]["30"]["feat"][feat] for c in r["runs"][rn]["cases"]]
            n = np.sum([c["rel_n"] for c in cs], 0); h = np.sum([c["rel_hits"] for c in cs], 0)
            if len(n) <= 2:
                continue
            p = np.arange(len(n)) / (len(n) - 1); ok = n > 50
            ax.plot(p[ok], h[ok] / n[ok], "-o", color=CAT[k % len(CAT)], lw=2, ms=4, label=text.get(rn, rn))
        ax.set_title(f"reliability of P({'updraft' if feat == 'up' else 'downdraft'}), +30 min", fontsize=8)
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Updraft / downdraft presence on the 22 unseen storms (deterministic models: P is 0 or 1; the reliability panel shows the "
                 "8-member ensembles only)", fontsize=9)
    fig.savefig(out / "fig_feat_scores.png", dpi=130); plt.close(fig)


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--order", nargs="*", default=[], help="run names (eval JSON keys) in plotting order")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--storms", nargs="*", default=[]); ap.add_argument("--start-min", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--eval", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.models and a.storms:
        sections(a, out)
    if a.eval:
        scores(a, out)


if __name__ == "__main__":
    main()
