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
from ks_latent.cm1.cell_losses import FEATURE_GROUPS, FEATURES  # noqa: E402
from make_status_figs import STYLE, phys, section  # noqa: E402

UNIT = {"u": "m/s", "w": "m/s", "th": "K", "qv": "g/kg", "prs": "hPa", "qliq": "g/kg", "qice": "g/kg"}
NICE = {"u+": "u′ > 10 m/s", "u-": "u′ < −10 m/s", "w+": "updraft w > 2 m/s", "w-": "downdraft w < −1.5 m/s", "th-": "cold θ′ < −2 K",
        "th+": "warm θ′ > 5 K", "qv+": "moist qv′ > 1 g/kg", "qv-": "dry qv′ < −3 g/kg", "prs+": "mesohigh p′ > 3 hPa", "prs-": "mesolow p′ < −1 hPa",
        "qliq+": "liquid > 0.5 g/kg", "qice+": "ice > 1 g/kg", "up": "updraft w > 2 m/s", "down": "downdraft w < −1.5 m/s"}
OUTC = ["k", "#1B8A4B"]


def feat_mask(f, key):
    """physical field (g/kg for condensate, as `phys` returns) -> mask of FEATURES[key]"""
    _, sign, thr = FEATURES[key]
    return sign * f > thr
# categorical order (fixed, never cycled): one colour per model in the score figure
CAT = ["#5B5F66", "#2A78C2", "#E07B39", "#B5446E", "#3A9D78", "#8C6BC8", "#C9A227"]


def outlines(ax, x, zk, f_true, keys):
    for c, k in enumerate(keys):
        m = feat_mask(f_true, k).astype(float)
        if m.any():
            ax.contour(x, zk, m, levels=[0.5], colors=OUTC[c], linewidths=0.6)


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
                w = {lag: np.stack([phys(ae.decode(lat[lag - 1, s:s + 1])[0], norm, Z, a.var) for s in range(S)]) for lag in (30, 60)}
            fc[nm] = w
            if nm == names[0]:
                truth = {lag: phys(d.full(0, t0 + lag)[0], norm0, Z, a.var) for lag in (30, 60)}
        x = (np.arange(NX) + 0.5) * 0.25
        rows = [("truth", None)] + [(text[nm], nm) for nm in names]
        keys = FEATURE_GROUPS[a.var].split(",")
        fig, axs = plt.subplots(len(rows), 2 + len(keys), figsize=(2.9 * (2 + len(keys)), 1.75 * len(rows) + 0.6), sharex=True, sharey=True, constrained_layout=True)
        for r, (lab, nm) in enumerate(rows):
            for c, lag in enumerate((30, 60)):
                f = truth[lag] if nm is None else fc[nm][lag][0]
                mw = section(axs[r, c], f, x, zk, a.var, title=f"{lab}: {a.var}, +{lag} min" if c == 0 else f"{a.var}, +{lag} min")
                outlines(axs[r, c], x, zk, truth[lag], keys)
            for c, key in enumerate(keys):
                p = feat_mask(truth[30], key).astype(float) if nm is None else feat_mask(fc[nm][30], key).mean(0)
                mp = axs[r, 2 + c].pcolormesh(x, zk, p, cmap="Greys", vmin=0, vmax=1, shading="auto", rasterized=True)
                outlines(axs[r, 2 + c], x, zk, truth[30], keys)
                axs[r, 2 + c].set_title(f"P({NICE[key]}), +30 min")
            axs[r, 0].set_ylabel("z (km)")
        for ax in axs.flat:
            ax.set_xlim(a.xlim[0], a.xlim[1]); ax.set_ylim(0, 14)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.colorbar(mw, ax=axs[:, :2], shrink=0.5, label=STYLE[a.var][0])
        fig.colorbar(mp, ax=axs[:, 2:], shrink=0.5, label="probability (ensemble fraction; 0 or 1 if deterministic)")
        leg = ", ".join(f"{NICE[k]} ({'black' if i == 0 else 'green'})" for i, k in enumerate(keys))
        fig.suptitle(f"{storm} (unseen), from t = {a.start_min} min, held boundaries. TRUE feature outlines: {leg}", fontsize=9)
        fig.savefig(out / f"fig_feat_sections_{a.var}_{storm}.png", dpi=130); plt.close(fig)
        print(f"[feat figs] {storm} done", flush=True)


def scores(a, out):
    r = json.load(open(Path(a.eval).expanduser()))
    text = {m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.labels}
    runs = [rn for rn in a.order if rn in r["runs"]] if a.order else list(r["runs"])
    lags = [L for L in ("15", "30", "60") if L in r["runs"][runs[0]]["cases"][0]["lags"]]
    mets = [("csi", "CSI (higher is better)"), ("bias", "frequency bias (1 is unbiased)"), ("fss@10", "FSS at 10 km (higher is better)")]
    feats = a.feats.split(",")
    fig, axs = plt.subplots(len(feats), 4, figsize=(13, 2.8 * len(feats)), constrained_layout=True, squeeze=False)
    for i, feat in enumerate(feats):
        for j, (mk, ttl) in enumerate(mets):
            ax = axs[i, j]
            for k, rn in enumerate(runs):
                v = [np.mean([c["lags"][L]["feat"][feat][mk] for c in r["runs"][rn]["cases"]]) for L in lags]
                ax.plot([int(L) for L in lags], v, "-o", color=CAT[k % len(CAT)], lw=2, ms=4, label=text.get(rn, rn))
            if mk == "bias":
                ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
            ax.set_title(f"{NICE.get(feat, feat)}: {ttl}", fontsize=8)
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
        ax.set_title(f"reliability of P({NICE.get(feat, feat)}), +30 min", fontsize=8)
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    axs[0, 0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Feature presence on the 22 unseen storms (deterministic models: P is 0 or 1; the reliability panel shows the "
                 "8-member ensembles only)", fontsize=9)
    fig.savefig(out / f"fig_feat_scores_{a.tag}.png", dpi=130); plt.close(fig)


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--order", nargs="*", default=[], help="run names (eval JSON keys) in plotting order")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--storms", nargs="*", default=[]); ap.add_argument("--start-min", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--eval", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--var", default="w", help="variable of the sections figure (its FEATURE_GROUPS features)")
    ap.add_argument("--feats", default="up,down", help="eval feature keys for the score figure"); ap.add_argument("--tag", default="w")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.models and a.storms:
        sections(a, out)
    if a.eval:
        scores(a, out)


if __name__ == "__main__":
    main()
