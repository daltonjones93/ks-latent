"""Forecast pictures comparing models on one unseen storm (8a): the original conv model against the ViT + dense-MLP model, in the
style of docs/cm1_status/figs/fig_forecast_sections.png (held boundaries: nothing after the start is known). Run on Saidie.

  fig_sections_<label>.png   the status-figure layout for each model: truth +30 min / forecast +30 / forecast +60 / truth +60,
                             5 variables (sample 0 of the stochastic models)
  fig_compare_sections.png   rows w, theta', liquid, ice; columns truth / A / B at +30 min and at +60 min
  fig_compare_zoom.png       zoom on the storm at +30 min: truth and two samples of each model (the fine structure)
  python scripts/cm1/make_model_compare_figs.py --models orig=$HOME/cm1_runs/g_local48_stagec vitmlp=$HOME/cm1_runs/u_win_mlp_stagec \\
      --labels orig="original (conv 2.6 M + stencil)" vitmlp="ViT 198 M + dense MLP" --out ~/cm1_figs/compare
"""
import argparse
import sys
import textwrap
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
from make_status_figs import STYLE, phys, section  # noqa: E402


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=run_dir (first = reference model A, second = B)")
    ap.add_argument("--labels", nargs="*", default=[], help="label=text for the titles")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--storm", default="ens_A2_mem001")
    ap.add_argument("--start-min", type=int, default=60)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", nargs=2, type=float, default=[135.0, 215.0])
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    names = [m.split("=", 1)[0] for m in a.models]
    paths = {m.split("=", 1)[0]: Path(m.split("=", 1)[1]).expanduser() for m in a.models}
    text = {k: k for k in names}; text.update({m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.labels})
    wrap = lambda t, w=24: textwrap.fill(t, w)
    dev = torch.device("cpu")
    torch.set_num_threads(16)
    files = member_files(a.source)
    zk = BaseState.from_sounding().z / 1000.0
    fc, truth = {}, {}
    norm0 = None
    for nm in names:
        ae, prop, norm, cfg, split = load_run(paths[nm], dev)
        if norm0 is None:
            norm0 = norm
        if not np.allclose(norm.scale, norm0.scale):
            raise ValueError(f"{nm}: different normalizer; fields would not be comparable")
        halo = cfg["stage_b"]["halo_sites"]
        d = ColumnData([load_member(files[a.storm])], norm, dev)
        Z, NX = d.Z, d.NX
        dt_min = d.dt / 60
        z = encode_members(ae, d)[0]
        t0 = int(round(a.start_min / dt_min)); n = int(round(60 / dt_min))
        S = a.samples if getattr(prop.cfg, "noise_channels", 0) > 0 else 1
        torch.manual_seed(0)
        with torch.no_grad():
            lat = rollout_from(prop, z, t0, n, halo, n_samples=S, boundary="persistence")       # (n, S, P, c)
            L30, L60 = int(round(30 / dt_min)) - 1, int(round(60 / dt_min)) - 1
            fc[nm] = {L: torch.stack([ae.decode(lat[L, s:s + 1])[0] for s in range(S)]) for L in (L30, L60)}
        truth[30] = d.full(0, t0 + L30 + 1)[0]; truth[60] = d.full(0, t0 + L60 + 1)[0]
        print(f"[fig] {nm}: {S} sample(s), halo {halo}", flush=True)
    x = (np.arange(NX) + 0.5) * 0.25
    LK = {30: L30, 60: L60}
    P = lambda f, v: phys(f, norm0, Z, v)

    # 1) the status-figure layout for each model
    vs = ["w", "th", "u", "qliq", "qice"]
    for nm in names:
        fig, axs = plt.subplots(len(vs), 4, figsize=(7.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
        for r, v in enumerate(vs):
            section(axs[r, 0], P(truth[30], v), x, zk, v, title="truth, +30 min" if r == 0 else "")
            section(axs[r, 1], P(fc[nm][LK[30]][0], v), x, zk, v, title="forecast, 30 min" if r == 0 else "")
            mm = section(axs[r, 2], P(fc[nm][LK[60]][0], v), x, zk, v, title="forecast, 60 min" if r == 0 else "")
            section(axs[r, 3], P(truth[60], v), x, zk, v, title="truth, +60 min" if r == 0 else "")
            fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0]); axs[r, 0].set_ylabel("z (km)")
            for ax in axs[r]:
                ax.set_xlim(100, 300)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.suptitle(f"{text[nm]}: forecasts from t = {a.start_min} min ({a.storm}, unseen), boundaries held at the start")
        fig.savefig(out / f"fig_sections_{nm}.png"); plt.close(fig)

    # 2) side by side
    A, B = names[0], names[1]
    vs = ["w", "th", "qliq", "qice"]
    fig, axs = plt.subplots(len(vs), 6, figsize=(12.5, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for r, v in enumerate(vs):
        for c0, lag in ((0, 30), (3, 60)):
            section(axs[r, c0], P(truth[lag], v), x, zk, v, title=f"truth, +{lag} min" if r == 0 else "")
            section(axs[r, c0 + 1], P(fc[A][LK[lag]][0], v), x, zk, v, title=wrap(text[A]) if r == 0 else "")
            mm = section(axs[r, c0 + 2], P(fc[B][LK[lag]][0], v), x, zk, v, title=wrap(text[B]) if r == 0 else "")
        fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0]); axs[r, 0].set_ylabel("z (km)")
        for ax in axs[r]:
            ax.set_xlim(100, 300)
    for ax in axs[-1]:
        ax.set_xlabel("x (km)")
    fig.suptitle(f"+30 min (left three) and +60 min (right three) from t = {a.start_min} min, {a.storm}, held boundaries")
    fig.savefig(out / "fig_compare_sections.png"); plt.close(fig)

    # 3) zoom, +30 min, truth and two samples of each model
    vs = ["w", "qliq"]
    cols = [("truth", None, None)] + [(wrap(text[m], 26) + f"\nsample {s}", m, s) for m in (A, B) for s in (0, 1)]
    fig, axs = plt.subplots(len(vs), len(cols), figsize=(2.6 * len(cols), 5.2), sharex=True, sharey=True, constrained_layout=True)
    for r, v in enumerate(vs):
        for c, (lab, m, s) in enumerate(cols):
            f = P(truth[30], v) if m is None else P(fc[m][LK[30]][min(s, fc[m][LK[30]].shape[0] - 1)], v)
            mm = section(axs[r, c], f, x, zk, v, zmax=13.0, title=lab if r == 0 else "")
            axs[r, c].set_xlim(*a.zoom)
        fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0]); axs[r, 0].set_ylabel("z (km)")
    for ax in axs[-1]:
        ax.set_xlabel("x (km)")
    fig.suptitle(f"Zoom, +30 min: fine structure and sample spread ({a.storm}, held boundaries)")
    fig.savefig(out / "fig_compare_zoom.png"); plt.close(fig)
    print("[fig] done")


if __name__ == "__main__":
    main()
