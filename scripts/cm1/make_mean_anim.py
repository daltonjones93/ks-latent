"""10b: animations of the AVERAGE of the independently trained models, one per variable (user, 2026-11-23: "how well would averaging the
output from the independently trained models work? ... create more visualizations like these [anim_vote]. obviously there will be no
thresholding"). Run on Saidie.

Each frame is one lead (1-min steps), same storm / start / boundary handling as make_vote_anim.py:
  row 1  truth | the pool mean (n models x M members)
  row 2  the mean of ONE model's M members | one member alone
  row 3  error of the pool mean (mean - truth) | spread of the pool (standard deviation over its members)
  bottom RMSE against lead (physical units, plotted window) for the pool mean, one model's mean and one member, drawn up to the current lead.

  python scripts/cm1/make_mean_anim.py --run R0,R1,R2,R3 --storm mem029 --out DIR
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from ks_latent.cm1.physics import BaseState  # noqa: E402
from make_status_figs import STYLE, phys  # noqa: E402
from make_vote_anim import roll  # noqa: E402

LINES = [("pool mean", "#2A78C2", "-", 2.2), ("one model, mean", "#B5446E", "--", 1.5), ("one member", "#9AA0A6", ":", 1.6)]


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated runs: the pool (the first is also the single model)")
    ap.add_argument("--storm", default="mem029")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--start-min", type=int, default=60); ap.add_argument("--lead-max", type=int, default=90)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--vars", default="u,w,th,qv,prs,qliq,qice"); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--fps", type=int, default=6); ap.add_argument("--dpi", type=int, default=100); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16); torch.set_grad_enabled(False)
    leads = list(range(1, a.lead_max + 1)); runs = a.run.split(","); nm = len(runs); M = a.members; S = nm * M
    models = []
    for i, r in enumerate(runs):
        torch.manual_seed(i)
        models.append(roll(r, a, leads)); print(f"[roll] {r}", flush=True)
    _, tru, norm, Z, NX = models[0]
    x = (np.arange(NX) + 0.5) * 0.25; zk = BaseState.from_sounding().z / 1000.0
    win = (x >= a.xlim[0]) & (x <= a.xlim[1])
    for v in a.vars.split(","):
        fc = {L: np.concatenate([np.stack([phys(m[0][L][s], m[2], Z, v) for s in range(M)]) for m in models]).astype(np.float32) for L in leads}
        tr = {L: phys(tru[L], norm, Z, v).astype(np.float32) for L in leads}
        pm = {L: fc[L].mean(0) for L in leads}; m1 = {L: fc[L][:M].mean(0) for L in leads}; one = {L: fc[L][0] for L in leads}
        sd = {L: fc[L].std(0) for L in leads}
        rm = lambda f, L: float(np.sqrt(((f[..., win] - tr[L][..., win]) ** 2).mean()))
        curves = {"pool mean": [rm(pm[L], L) for L in leads], "one model, mean": [rm(m1[L], L) for L in leads],
                  "one member": [rm(one[L], L) for L in leads]}
        lab, cmap, (lo, hi), _ = STYLE[v]
        span = max(abs(lo), abs(hi))
        fig = plt.figure(figsize=(12.5, 12.0))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.55], hspace=0.38, wspace=0.12, left=0.06, right=0.9, top=0.93, bottom=0.08)
        ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
        ax_c = fig.add_subplot(gs[3, :])
        panels = [(ax[0][0], tr, cmap, lo, hi), (ax[0][1], pm, cmap, lo, hi), (ax[1][0], m1, cmap, lo, hi), (ax[1][1], one, cmap, lo, hi),
                  (ax[2][0], {L: pm[L] - tr[L] for L in leads}, "RdBu_r", -span, span), (ax[2][1], sd, "magma_r", 0, span / 2)]
        meshes = []
        for axx, f, cm, vlo, vhi in panels:
            meshes.append(axx.pcolormesh(x, zk, f[1], cmap=cm, vmin=vlo, vmax=vhi, shading="auto", rasterized=True))
            axx.set_xlim(*a.xlim); axx.set_ylim(0, 14); axx.tick_params(labelsize=7)
        for r in range(3):
            ax[r][0].set_ylabel("z (km)", fontsize=8)
        for c in range(2):
            ax[2][c].set_xlabel("x (km)", fontsize=8)
        fig.colorbar(meshes[0], cax=fig.add_axes([0.915, 0.55, 0.012, 0.37]), label=lab)
        fig.colorbar(meshes[4], cax=fig.add_axes([0.915, 0.36, 0.012, 0.15]), label="error")
        fig.colorbar(meshes[5], cax=fig.add_axes([0.915, 0.27, 0.012, 0.07]), label="spread")
        titles = ["TRUTH", f"MEAN of {nm} models x {M} members", f"mean of ONE model's {M} members", "one member alone",
                  "pool mean − truth", f"spread of the pool ({S} members)"]
        unit = lab[lab.find("(") + 1:lab.find(")")] if "(" in lab else ""
        ax_c.set_xlim(0, a.lead_max); ax_c.set_ylim(0, 1.08 * max(max(c) for c in curves.values()))
        ax_c.set_xlabel("lead (min)", fontsize=8); ax_c.set_ylabel(f"RMSE ({unit}), plotted window", fontsize=8)
        ax_c.grid(alpha=0.25); ax_c.tick_params(labelsize=7)
        lines = {k: ax_c.plot([], [], ls, color=c, lw=w, label=k)[0] for k, c, ls, w in LINES}
        ax_c.legend(fontsize=7, ncol=3, loc="upper left")
        sup = fig.suptitle("", fontsize=11)

        def frame(i):
            L = leads[i]
            for (axx, f, *_), mesh, t in zip(panels, meshes, titles):
                mesh.set_array(f[L].ravel()); axx.set_title(f"{t}, +{L} min", fontsize=9)
            ax[0][1].set_title(f"{titles[1]}, +{L} min: RMSE {curves['pool mean'][i]:.2f}", fontsize=9)
            ax[1][0].set_title(f"{titles[2]}, +{L} min: RMSE {curves['one model, mean'][i]:.2f}", fontsize=9)
            ax[1][1].set_title(f"{titles[3]}, +{L} min: RMSE {curves['one member'][i]:.2f}", fontsize=9)
            for k, ln in lines.items():
                ln.set_data(leads[: i + 1], curves[k][: i + 1])
            sup.set_text(f"{a.storm} (unseen), average of {nm} independently trained models x {M} members, +{L} min from t = {a.start_min} min")
            return []

        ani = manim.FuncAnimation(fig, frame, frames=len(leads), blit=False)
        f = out / f"mean_pool{nm}_{v}_{a.storm}.mp4"
        ani.save(f, writer=manim.FFMpegWriter(fps=a.fps, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=a.dpi)
        plt.close(fig)
        print(f"[anim] {f}", flush=True)


if __name__ == "__main__":
    main()
