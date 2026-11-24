"""10b: animations of the scale blend (idea 3 of the post-processing test; user, 2026-11-24: "make more mp4 visualizations for (3)").

blend = large scales of the pool average + small scales of one member (Gaussian split along x; sigma 1.9 km: half power at ~10 km
wavelength; sigma 3.7 km: ~20 km). Condensate is blended as sqrt(q) (the model's units) and squared back.
Same storm / start / boundary handling as make_vote_anim.py and make_mean_anim.py. Each frame (one lead, 1-min steps):
  row 1  truth | blend at 10 km
  row 2  the pool average | the member whose small scales are used
  row 3  blend at 20 km | error of the 10 km blend (blend - truth)
  bottom RMSE against lead (physical units, plotted window): average, both blends, the member.

  python scripts/cm1/make_blend_anim.py --run R0,R1,R2,R3 --storm mem029 --out DIR
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

DX = 0.25
LINES = [("average", "#2A78C2", "-", 1.8), ("blend 10 km", "#1F8A4B", "-", 2.4), ("blend 20 km", "#7BC08E", "--", 1.6),
         ("one member", "#9AA0A6", ":", 1.6)]


def lowpass(f, sigma_km):
    """Gaussian low-pass along the last axis (numpy in, numpy out; edges replicated)"""
    t = torch.as_tensor(f, dtype=torch.float32)
    s = sigma_km / DX; r = int(np.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / s) ** 2); k = k / k.sum()
    sh = t.shape
    y = torch.nn.functional.pad(t.reshape(-1, 1, sh[-1]), (r, r), mode="replicate")
    return torch.nn.functional.conv1d(y, k.view(1, 1, -1)).reshape(sh).numpy()


def blend(mean, member, sigma_km, cond):
    if cond:   # blend in sqrt(q), the model's units
        m, b = np.sqrt(np.clip(mean, 0, None)), np.sqrt(np.clip(member, 0, None))
        return np.clip(lowpass(m, sigma_km) + b - lowpass(b, sigma_km), 0, None) ** 2
    return lowpass(mean, sigma_km) + member - lowpass(member, sigma_km)


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--storm", default="mem029")
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
    leads = list(range(1, a.lead_max + 1)); runs = a.run.split(","); nm = len(runs); M = a.members
    models = []
    for i, r in enumerate(runs):
        torch.manual_seed(i)
        models.append(roll(r, a, leads)); print(f"[roll] {r}", flush=True)
    _, tru, norm, Z, NX = models[0]
    x = (np.arange(NX) + 0.5) * 0.25; zk = BaseState.from_sounding().z / 1000.0
    win = (x >= a.xlim[0]) & (x <= a.xlim[1])
    for v in a.vars.split(","):
        cond = v in ("qliq", "qice")
        fc = {L: np.concatenate([np.stack([phys(m[0][L][s], m[2], Z, v) for s in range(M)]) for m in models]).astype(np.float32) for L in leads}
        tr = {L: phys(tru[L], norm, Z, v).astype(np.float32) for L in leads}
        avg = {L: fc[L].mean(0) for L in leads}; one = {L: fc[L][0] for L in leads}
        b10 = {L: blend(avg[L], one[L], 1.9, cond) for L in leads}; b20 = {L: blend(avg[L], one[L], 3.7, cond) for L in leads}
        rm = lambda f, L: float(np.sqrt(((f[..., win] - tr[L][..., win]) ** 2).mean()))
        curves = {"average": [rm(avg[L], L) for L in leads], "blend 10 km": [rm(b10[L], L) for L in leads],
                  "blend 20 km": [rm(b20[L], L) for L in leads], "one member": [rm(one[L], L) for L in leads]}
        lab, cmap, (lo, hi), _ = STYLE[v]
        span = max(abs(lo), abs(hi))
        fig = plt.figure(figsize=(12.5, 12.0))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.55], hspace=0.38, wspace=0.12, left=0.06, right=0.9, top=0.93, bottom=0.08)
        ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
        ax_c = fig.add_subplot(gs[3, :])
        panels = [(ax[0][0], tr, cmap, lo, hi, "TRUTH", None), (ax[0][1], b10, cmap, lo, hi, "BLEND: average > 10 km + member < 10 km", "blend 10 km"),
                  (ax[1][0], avg, cmap, lo, hi, f"average of {nm} models x {M}", "average"),
                  (ax[1][1], one, cmap, lo, hi, "the member (its small scales are used)", "one member"),
                  (ax[2][0], b20, cmap, lo, hi, "blend: average > 20 km + member < 20 km", "blend 20 km"),
                  (ax[2][1], {L: b10[L] - tr[L] for L in leads}, "RdBu_r", -span, span, "10 km blend − truth", None)]
        meshes = []
        for axx, f, cm, vlo, vhi, *_ in panels:
            meshes.append(axx.pcolormesh(x, zk, f[1], cmap=cm, vmin=vlo, vmax=vhi, shading="auto", rasterized=True))
            axx.set_xlim(*a.xlim); axx.set_ylim(0, 14); axx.tick_params(labelsize=7)
        for r in range(3):
            ax[r][0].set_ylabel("z (km)", fontsize=8)
        for c in range(2):
            ax[2][c].set_xlabel("x (km)", fontsize=8)
        fig.colorbar(meshes[0], cax=fig.add_axes([0.915, 0.45, 0.012, 0.47]), label=lab)
        fig.colorbar(meshes[5], cax=fig.add_axes([0.915, 0.27, 0.012, 0.14]), label="error")
        unit = lab[lab.find("(") + 1:lab.find(")")] if "(" in lab else ""
        ax_c.set_xlim(0, a.lead_max); ax_c.set_ylim(0, 1.08 * max(max(c) for c in curves.values()))
        ax_c.set_xlabel("lead (min)", fontsize=8); ax_c.set_ylabel(f"RMSE ({unit}), plotted window", fontsize=8)
        ax_c.grid(alpha=0.25); ax_c.tick_params(labelsize=7)
        lines = {k: ax_c.plot([], [], ls, color=c, lw=w, label=k)[0] for k, c, ls, w in LINES}
        ax_c.legend(fontsize=7, ncol=4, loc="upper left")
        sup = fig.suptitle("", fontsize=11)

        def frame(i):
            L = leads[i]
            for (axx, f, _, _, _, t, key), mesh in zip(panels, meshes):
                mesh.set_array(f[L].ravel())
                axx.set_title(f"{t}, +{L} min" + (f": RMSE {curves[key][i]:.2f}" if key else ""), fontsize=9)
            for k, ln in lines.items():
                ln.set_data(leads[: i + 1], curves[k][: i + 1])
            sup.set_text(f"{a.storm} (unseen), scale blend from {nm} models x {M} members, +{L} min from t = {a.start_min} min")
            return []

        ani = manim.FuncAnimation(fig, frame, frames=len(leads), blit=False)
        f = out / f"blend_pool{nm}_{v}_{a.storm}.mp4"
        ani.save(f, writer=manim.FFMpegWriter(fps=a.fps, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=a.dpi)
        plt.close(fig)
        print(f"[anim] {f}", flush=True)


if __name__ == "__main__":
    main()
