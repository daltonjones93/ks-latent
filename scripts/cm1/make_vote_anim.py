"""9g: animations of the ensemble vote over the forecast, one per variable (user, 2026-11-21: "an animation showing the evolution of the
member voting over time like in here but for all variables. one animation per variable"). Run on Saidie.

Each frame is one lead (1-min steps): row 1 the true field and member 1's field with the TRUE feature outlines; one column per feature of
the variable (cell_losses.FEATURE_GROUPS) for the vote count (0-8 members, red: the true outline) and the decision (present if >= K of 8,
K from the per-feature cutoff table chosen on the validation storms, nearest lead) as hits / misses / false alarms; bottom: CSI against
lead for the vote (solid) and a single member (dashed), drawn up to the current lead.

  python scripts/cm1/make_vote_anim.py --run ~/cm1_runs/F6_..._stagec --storm mem029 --cutoffs vote_cutoffs.json --out DIR

Pooled model (10b): --run R0,R1,R2,R3 pools independently trained models (each encodes the truth with its own autoencoder, rolls out
--members members and is decoded with its own normalizer); the vote panels show the count over all S = n x members, the cutoff is the
K of 8 kept as a fraction (K = ceil(K8 / 8 x S)), and the CSI panel adds the 8-member vote of the first model alone for comparison.
"""
import argparse
import json
import math
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
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.cell_losses import FEATURE_GROUPS, FEATURES  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from make_status_figs import STYLE, phys  # noqa: E402
from make_vote_figs import CONT, CONT_COLORS, CONT_NORM, NICE, contingency, mask, scores  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FEAT_COLORS = ["#2A78C2", "#B5446E"]


def k_of(table, key, L, default=3):
    if not table:
        return default
    t = table[key]
    return t[str(L)] if str(L) in t else t[min(t, key=lambda q: abs(int(q) - L))]


def k_frac(table, key, L, S):
    return max(1, min(S, math.ceil(k_of(table, key, L) / 8.0 * S)))


def roll(run, a, leads):
    ae, prop, norm, cfg, _ = load_run(Path(run).expanduser(), torch.device("cpu"))
    d = ColumnData([load_member(member_files(a.source)[a.storm])], norm, torch.device("cpu"))
    t0 = int(round(a.start_min / (d.dt / 60)))
    lat = rollout_from(prop, encode_members(ae, d)[0], t0, a.lead_max, cfg["stage_b"]["halo_sites"], n_samples=a.members, boundary="persistence")
    dec = {L: torch.cat([ae.decode(lat[L - 1, s:s + 1]) for s in range(a.members)]) for L in leads}
    tru = {L: d.full(0, t0 + L)[0] for L in leads}
    return dec, tru, norm, d.Z, d.NX


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--storm", default="mem029")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--start-min", type=int, default=60); ap.add_argument("--lead-max", type=int, default=90)
    ap.add_argument("--members", type=int, default=8); ap.add_argument("--cutoffs", default=None)
    ap.add_argument("--cutoffs8", default=None, help="K-of-8 table for the one-model 8-member comparison line (default: --cutoffs if it is K of 8)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--vars", default="u,w,th,qv,prs,qliq,qice"); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--fps", type=int, default=6); ap.add_argument("--dpi", type=int, default=100); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    tj = json.load(open(Path(a.cutoffs).expanduser())) if a.cutoffs else {}
    table = tj.get("cutoffs")
    table_S = tj.get("members", 8)                 # cutoffs calibrated for this many members (calibrate_pool_cutoffs.py) or K of 8
    if a.cutoffs8:
        a.cutoffs8 = json.load(open(Path(a.cutoffs8).expanduser()))["cutoffs"]
    elif table_S == 8:
        a.cutoffs8 = table
    torch.set_num_threads(16); torch.set_grad_enabled(False)
    leads = list(range(1, a.lead_max + 1)); runs = a.run.split(","); nm = len(runs); M = a.members; S = nm * M
    models = []
    for i, r in enumerate(runs):
        torch.manual_seed(i)
        models.append(roll(r, a, leads)); print(f"[roll] {r}", flush=True)
    _, tru, norm, Z, NX = models[0]
    vote_cmap = ListedColormap(["#FFFFFF"] + [plt.cm.Greys(q) for q in np.linspace(0.2, 1.0, S)])
    vote_norm = BoundaryNorm(np.arange(-0.5, S + 1.5), S + 1)
    x = (np.arange(NX) + 0.5) * 0.25; zk = BaseState.from_sounding().z / 1000.0
    for v in a.vars.split(","):
        keys = FEATURE_GROUPS[v].split(",")
        fc = {L: np.concatenate([np.stack([phys(m[0][L][s], m[2], Z, v) for s in range(M)]) for m in models]).astype(np.float32) for L in leads}
        tr = {L: phys(tru[L], norm, Z, v).astype(np.float32) for L in leads}
        stats = {}
        for L in leads:
            for k in keys:
                y = mask(tr[L], k); ms = mask(fc[L], k)
                K = (k_of(table, k, L) if table_S == S else k_frac(table, k, L, S)) if table else 3
                b, c = scores(ms.sum(0) >= K, y); _, c1 = scores(ms[0], y)
                c8 = scores(ms[:M].sum(0) >= (k_of(a.cutoffs8, k, L) if a.cutoffs8 else 3), y)[1] if nm > 1 else None
                stats[(L, k)] = (K, b, c, ms.sum(0), y, contingency(ms.sum(0) >= K, y), c1, c8)
        nc = 2
        fig = plt.figure(figsize=(12.5, 12.0))
        gs = fig.add_gridspec(4, nc, height_ratios=[1, 1, 1, 0.55], hspace=0.38, wspace=0.12, left=0.06, right=0.9, top=0.93, bottom=0.05 if nm == 1 else 0.1)
        ax_t, ax_m = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax_v = [fig.add_subplot(gs[1, j]) for j in range(nc)]; ax_d = [fig.add_subplot(gs[2, j]) for j in range(nc)]
        ax_c = fig.add_subplot(gs[3, :])
        lab, cmap, (lo, hi), _ = STYLE[v]
        for ax in [ax_t, ax_m] + ax_v + ax_d:
            ax.set_xlim(*a.xlim); ax.set_ylim(0, 14); ax.tick_params(labelsize=7)
        for ax in (ax_t, ax_v[0], ax_d[0]):
            ax.set_ylabel("z (km)", fontsize=8)
        for ax in ax_d:
            ax.set_xlabel("x (km)", fontsize=8)
        mt = ax_t.pcolormesh(x, zk, tr[1], cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
        mm = ax_m.pcolormesh(x, zk, fc[1][0], cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
        cax = fig.add_axes([0.915, 0.735, 0.012, 0.19]); fig.colorbar(mt, cax=cax, label=lab)
        vm, dm = [], []
        for j in range(nc):
            if j < len(keys):
                vm.append(ax_v[j].pcolormesh(x, zk, stats[(1, keys[j])][3], cmap=vote_cmap, norm=vote_norm, shading="auto", rasterized=True))
                dm.append(ax_d[j].pcolormesh(x, zk, stats[(1, keys[j])][5], cmap=CONT, norm=CONT_NORM, shading="auto", rasterized=True))
            else:   # one-feature variables: the second column shows a single member's mask against the truth, for comparison
                vm.append(None)
                dm.append(ax_d[j].pcolormesh(x, zk, contingency(mask(fc[1][0], keys[0]), stats[(1, keys[0])][4]), cmap=CONT, norm=CONT_NORM,
                                             shading="auto", rasterized=True))
                ax_v[j].axis("off")
                ax_v[j].text(0.02, 0.5, f"{NICE[keys[0]]}\n\nleft: the vote of {S} members\n"
                             "right, below: ONE member alone,\nfor comparison", transform=ax_v[j].transAxes, fontsize=9, va="center")
        cax2 = fig.add_axes([0.915, 0.42, 0.012, 0.19]); fig.colorbar(vm[0], cax=cax2, ticks=range(0, S + 1, max(1, S // 8)), label="members with the feature")
        fig.legend(handles=[Patch(facecolor=c, edgecolor="0.5", label=l) for c, l in zip(CONT_COLORS[1:], ("hit", "miss", "false alarm"))],
                   loc="center left", bbox_to_anchor=(0.905, 0.2), fontsize=8, frameon=False)
        contours = []
        ax_c.set_xlim(0, a.lead_max); ax_c.set_ylim(0, 1); ax_c.set_xlabel("lead (min)", fontsize=8); ax_c.set_ylabel("CSI", fontsize=8)
        ax_c.grid(alpha=0.25); ax_c.tick_params(labelsize=7)
        lines = []
        for j, k in enumerate(keys):
            l1, = ax_c.plot([], [], "-", color=FEAT_COLORS[j], lw=2, label=f"{NICE[k]}: vote" + (f" ({nm} models x {M})" if nm > 1 else ""))
            l2, = ax_c.plot([], [], "--", color=FEAT_COLORS[j], lw=1.2, label=f"{NICE[k]}: one member")
            l3 = ax_c.plot([], [], ":", color=FEAT_COLORS[j], lw=1.6, label=f"{NICE[k]}: one model, {M}-member vote")[0] if nm > 1 else None
            lines.append((k, l1, l2, l3))
        if nm > 1:
            ax_c.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
        else:
            ax_c.legend(fontsize=7, ncol=2, loc="upper right")
        sup = fig.suptitle("", fontsize=11)

        def frame(i):
            L = leads[i]
            for c in contours:
                c.remove()
            contours.clear()
            mt.set_array(tr[L].ravel()); mm.set_array(fc[L][0].ravel())
            ax_t.set_title(f"TRUTH {lab}, +{L} min", fontsize=9); ax_m.set_title(f"member 1 of {S}, +{L} min", fontsize=9)
            for j, k in enumerate(keys):
                K, b, c, votes, y, cat, c1, _ = stats[(L, k)]
                vm[j].set_array(votes.ravel()); dm[j].set_array(cat.ravel())
                ax_v[j].set_title(f"vote count: {NICE[k]}", fontsize=9)
                ax_d[j].set_title(f"present if >= {K} of {S}: area ratio {b:.2f}, CSI {c:.2f}", fontsize=9)
                contours.append(ax_v[j].contour(x, zk, y.astype(float), levels=[0.5], colors="#C0392B", linewidths=0.6))
                for ax in (ax_t, ax_m):
                    contours.append(ax.contour(x, zk, y.astype(float), levels=[0.5], colors="k" if j == 0 else "#1B8A4B", linewidths=0.6))
            if len(keys) == 1:
                k = keys[0]; y = stats[(L, k)][4]
                dm[1].set_array(contingency(mask(fc[L][0], k), y).ravel())
                b1, c1 = scores(mask(fc[L][0], k), y)
                ax_d[1].set_title(f"ONE member alone: area ratio {b1:.2f}, CSI {c1:.2f}", fontsize=9)
            for k, l1, l2, l3 in lines:
                l1.set_data(leads[: i + 1], [stats[(q, k)][2] for q in leads[: i + 1]])
                l2.set_data(leads[: i + 1], [stats[(q, k)][6] for q in leads[: i + 1]])
                if l3 is not None:
                    l3.set_data(leads[: i + 1], [stats[(q, k)][7] for q in leads[: i + 1]])
            outl = " / ".join(f"{NICE[k]} ({'black' if j == 0 else 'green'})" for j, k in enumerate(keys))
            who = f"pool of {nm} models x {M} members" if nm > 1 else f"F6 {M}-member vote"
            sup.set_text(f"{a.storm} (unseen), {who}, +{L} min from t = {a.start_min} min" + ("\n" if nm > 1 else ". ") + f"True outlines: {outl}")
            return []

        ani = manim.FuncAnimation(fig, frame, frames=len(leads), blit=False)
        f = out / (f"vote_pool{nm}{a.tag}_{v}_{a.storm}.mp4" if nm > 1 else f"vote_{v}{a.tag}_{a.storm}.mp4")
        ani.save(f, writer=manim.FFMpegWriter(fps=a.fps, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=a.dpi)
        plt.close(fig)
        print(f"[anim] {f}", flush=True)


if __name__ == "__main__":
    main()
