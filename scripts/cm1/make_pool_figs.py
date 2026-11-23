"""10b figures: the vote of a pool of independently trained models against the vote of ONE model, on one storm (user, 2026-11-23:
"make some more plots for this so we can compare"). Run on Saidie (CPU is enough).

For each lead, one figure: rows are features, columns are
  truth (field + true outline) | one model, 8 members | one model, 8*n members | pool of n models x 8 members (member fraction)
  | the three decisions as hits / misses / false alarms.
The cutoff is the animation's rule (vote_cutoffs.json, K of 8 per feature and lead) kept as a fraction: K = ceil(K8 / 8 * S).
Each model encodes the truth with its own autoencoder and is decoded with its own normalizer, so all fields are in physical units.

  python scripts/cm1/make_pool_figs.py --runs R0,R1,R2,R3 --storm mem029 --cutoffs vote_cutoffs.json --out DIR
"""
import argparse
import json
import math
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
from make_status_figs import STYLE, phys  # noqa: E402
from make_vote_figs import CONT, CONT_COLORS, CONT_NORM, NICE, contingency, mask, scores  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

VAR = {"w+": "w", "w-": "w", "th-": "th", "th+": "th", "qliq+": "qliq", "qice+": "qice", "prs-": "prs", "prs+": "prs",
       "u+": "u", "u-": "u", "qv+": "qv", "qv-": "qv"}


def k8(table, f, L):
    t = table[f]
    return t[str(L)] if str(L) in t else t[min(t, key=lambda q: abs(int(q) - L))]


def run_model(run, storm, source, t0_min, leads, n, seed):
    ae, prop, norm, cfg, _ = load_run(Path(run).expanduser(), torch.device("cpu"))
    d = ColumnData([load_member(member_files(source)[storm])], norm, torch.device("cpu"))
    t0 = int(round(t0_min / (d.dt / 60)))
    torch.manual_seed(seed)
    lat = rollout_from(prop, encode_members(ae, d)[0], t0, max(leads), cfg["stage_b"]["halo_sites"], n_samples=n, boundary="persistence")
    out = {}
    for L in leads:
        dec = torch.cat([ae.decode(lat[L - 1, s:s + 1]) for s in range(n)])
        out[L] = {v: np.stack([phys(dec[s], norm, d.Z, v) for s in range(n)]).astype(np.float32) for v in STYLE}
    tru = {L: {v: phys(d.full(0, t0 + L)[0], norm, d.Z, v).astype(np.float32) for v in STYLE} for L in leads}
    return out, tru, d.NX


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run dirs; the first is the single model")
    ap.add_argument("--storm", default="mem029")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--start-min", type=int, default=60); ap.add_argument("--leads", default="15,30,60")
    ap.add_argument("--members", type=int, default=8); ap.add_argument("--cutoffs", required=True)
    ap.add_argument("--features", default="w+,w-,th-,prs-,qliq+")
    ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    table = json.load(open(Path(a.cutoffs).expanduser()))["cutoffs"]
    torch.set_num_threads(16); torch.set_grad_enabled(False)
    runs = a.runs.split(","); n = len(runs); M = a.members; leads = [int(x) for x in a.leads.split(",")]
    feats = a.features.split(",")
    pool, tru, NX = [], None, None
    for i, r in enumerate(runs):
        fc, t, NX = run_model(r, a.storm, a.source, a.start_min, leads, M, seed=i)
        pool.append(fc); tru = tru or t
        print(f"[pool] {r}", flush=True)
    big, _, _ = run_model(runs[0], a.storm, a.source, a.start_min, leads, M * n, seed=100)   # ONE model, the same number of members
    x = (np.arange(NX) + 0.5) * 0.25; zk = BaseState.from_sounding().z / 1000.0
    summary = {}
    for L in leads:
        nf = len(feats)
        fig, ax = plt.subplots(nf, 7, figsize=(19, 2.35 * nf + 0.9), sharex=True, sharey=True,
                               gridspec_kw=dict(hspace=0.45, wspace=0.08, left=0.04, right=0.99, top=0.9, bottom=0.07))
        for r, f in enumerate(feats):
            v = VAR[f]; y = mask(tru[L][v], f)
            ens = {"one model, 8": pool[0][L][v], f"one model, {M * n}": big[L][v],
                   f"{n} models x 8": np.concatenate([p[L][v] for p in pool])}
            lab, cmap, (lo, hi), _ = STYLE[v]
            ax[r, 0].pcolormesh(x, zk, tru[L][v], cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
            ax[r, 0].contour(x, zk, y.astype(float), levels=[0.5], colors="k", linewidths=0.7)
            ax[r, 0].set_title(f"truth {lab}: {NICE[f]}", fontsize=8)
            for j, (name, e) in enumerate(ens.items()):
                S = e.shape[0]; K = max(1, min(S, math.ceil(k8(table, f, L) / 8.0 * S)))
                ms = mask(e, f); frac = ms.mean(0); dec = ms.sum(0) >= K
                b, c = scores(dec, y)
                summary[f"{L}|{f}|{name}"] = {"csi": float(c), "bias": float(b), "K": K, "S": S}
                axf = ax[r, 1 + j]
                axf.pcolormesh(x, zk, frac, cmap="Greys", vmin=0, vmax=1, shading="auto", rasterized=True)
                axf.contour(x, zk, y.astype(float), levels=[0.5], colors="#C0392B", linewidths=0.6)
                axf.set_title(f"{name}: member fraction", fontsize=8)
                axd = ax[r, 4 + j]
                axd.pcolormesh(x, zk, contingency(dec, y), cmap=CONT, norm=CONT_NORM, shading="auto", rasterized=True)
                axd.set_title(f"{name}, >= {K}/{S}: CSI {c:.2f}, area {b:.2f}", fontsize=8)
            ax[r, 0].set_ylabel("z (km)")
        for a_ in ax.flat:
            a_.set_xlim(*a.xlim); a_.set_ylim(0, 14); a_.tick_params(labelsize=7)
        for a_ in ax[-1]:
            a_.set_xlabel("x (km)")
        fig.legend(handles=[Patch(facecolor=c, edgecolor="0.5", label=l) for c, l in zip(CONT_COLORS[1:], ("hit", "miss", "false alarm"))],
                   loc="upper right", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"{a.storm} (unseen), forecast +{L} min from t = {a.start_min} min. Grey: fraction of members with the feature "
                     f"(red: true outline). Cutoff: the animation's K of 8, kept as a fraction of the members", fontsize=10, x=0.42)
        p = out / f"fig_pool_vote_{a.storm}_L{L}.png"; fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[fig] {p}", flush=True)
    json.dump(summary, open(out / f"pool_vote_{a.storm}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
