"""Forecast pictures for the two-scale / baseline models (6b): x-z cross-sections of one variable at several lags (rows: truth and each model's
first sample; one extra row per model for a second sample if stochastic), plus the surface curve of every sample against truth.
  python scripts/cm1/twoscale_plot.py --runs A B C --source cache:... --member NAME --t0 180 --out DIR"""
from __future__ import annotations

import argparse
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
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from twoscale_eval import is_stochastic, rollout  # noqa: E402

LABEL = {"th": "theta' (K)", "w": "w (m/s)", "qliq": "q_liq (g/kg)", "u": "u (m/s)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--member", default=None); ap.add_argument("--t0", type=int, default=180)
    ap.add_argument("--lags", default="5,15,30,60"); ap.add_argument("--vars", default="w,th,qliq"); ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    models = {Path(r).name: load_run(Path(r).expanduser(), torch.device("cpu")) for r in a.runs}
    files = member_files(a.source)
    names = next(iter(models.values()))[4]["test"]; nm = a.member or names[0]
    trajs, dec = {}, {}
    for rn, (ae, prop, norm, cfg, split) in models.items():
        halo = cfg["stage_b"]["halo_sites"]; mar = (halo + ae.cfg.reach_sites) * ae.cfg.patch
        d = ColumnData([load_member(files[nm])], norm, torch.device("cpu"))
        z = encode_members(ae, d)[0]
        S = a.samples if is_stochastic(prop) else 1
        tr = rollout(prop, z, a.t0, n, halo, S, seed=a.t0)
        with torch.no_grad():
            dec[rn] = {L: ae.decode(tr[L])[..., mar:-mar] * 1.0 for L in lags}
        truth = {L: d.full(0, a.t0 + L)[0][..., mar:-mar] for L in lags}
        scale, Zl = norm.scale, d.Z
    W = min([truth[lags[0]].shape[-1]] + [dec[r][lags[0]].shape[-1] for r in dec])
    def cc(t):
        e = (t.shape[-1] - W) // 2
        return t[..., e:e + W]
    truth = {L: cc(t) for L, t in truth.items()}; dec = {r: {L: cc(t) for L, t in dd.items()} for r, dd in dec.items()}
    x = (np.arange(W) + 0.5) * 0.25
    for v in a.vars.split(","):
        iv = VARS.index(v); sl = slice(iv * Zl, (iv + 1) * Zl); sc = float(scale[iv])
        rows = [("CM1 truth", truth, None)]
        for rn in models:
            for s in range(dec[rn][lags[0]].shape[0]):
                rows.append((f"{rn}" + (f" sample {s}" if dec[rn][lags[0]].shape[0] > 1 else ""), dec[rn], s))
        vmax = max(float(np.abs(truth[L][sl].numpy() * sc).max()) for L in lags) * 0.8
        fig, ax = plt.subplots(len(rows), len(lags), figsize=(3.6 * len(lags), 1.35 * len(rows)), sharex=True, sharey=True, squeeze=False)
        for i, (lab, src, s) in enumerate(rows):
            for j, L in enumerate(lags):
                f = (src[L][sl] if s is None else src[L][s, sl]).numpy() * sc
                im = ax[i, j].imshow(f, origin="lower", aspect="auto", cmap="RdBu_r" if v != "qliq" else "Blues", vmin=-vmax if v != "qliq" else 0,
                                     vmax=vmax, extent=[x[0], x[-1], 0, f.shape[0]])
                if i == 0: ax[i, j].set_title(f"+{L} min")
                if j == 0: ax[i, j].set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
                ax[i, j].set_yticks([])
        fig.suptitle(f"{LABEL.get(v, v)} x-z sections, storm {nm}, start frame {a.t0}", fontsize=9)
        fig.supxlabel("x (km, interior)"); fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
        fig.savefig(out / f"fc_{v}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # surface (lowest level) line plots: every sample against truth
    for v in a.vars.split(","):
        iv = VARS.index(v); sc = float(scale[iv]); k0 = iv * Zl
        fig, ax = plt.subplots(len(models), len(lags), figsize=(3.6 * len(lags), 1.7 * len(models)), sharex=True, sharey=True, squeeze=False)
        for i, rn in enumerate(models):
            for j, L in enumerate(lags):
                ax[i, j].plot(x, truth[L][k0].numpy() * sc, "k", lw=1.2)
                for s in range(dec[rn][L].shape[0]):
                    ax[i, j].plot(x, dec[rn][L][s, k0].numpy() * sc, lw=0.6, alpha=0.8)
                if i == 0: ax[i, j].set_title(f"+{L} min")
                if j == 0: ax[i, j].set_ylabel(rn, fontsize=6, rotation=0, ha="right", va="center")
        fig.suptitle(f"{LABEL.get(v, v)} lowest level: truth (black) vs samples, storm {nm}", fontsize=9)
        fig.savefig(out / f"sfc_{v}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("done", nm)


if __name__ == "__main__":
    main()
