"""9d/9e: how the "updraft present" 0/1 target is made from w, and how a forecast is scored against it (one held-out storm). Run on Saidie.

  fig_feat_prep_<storm>.png
    row 1 (the target, from the TRUE field): (a) w; (b) distribution of w over the storm with the calibrated thresholds; (c) the updraft mask
          1[w > 2 m/s]; (d) the downdraft mask 1[w < -1.5 m/s] (a separate class)
    row 2 (the forecast side): (e) a forecast's w (+30 min); (f) the transfer w -> p = sigmoid((w - 2) / 0.5) against the hard step the target
          uses; (g) the forecast's P(updraft) with the true updraft outline; (h) the per-point cross-entropy -[y log p + (1 - y) log(1 - p)]
    row 3 (a horizontal cut at --z-km): w of truth and forecast against x with the threshold; the target y and the forecast's p against x

  python scripts/cm1/make_feature_prep_fig.py --run ~/cm1_runs/F7_conv416_mixer512_det_stagec --storm mem029 --out DIR
"""
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
from ks_latent.cm1.cell_losses import FEATURE_SOFT, FEATURES  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from make_status_figs import phys, section  # noqa: E402

TRUTH, FCST, MUTED = "#3B3F45", "#2A78C2", "#8A8F96"


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--label", default="F7 deterministic forecast")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--storm", default="mem029"); ap.add_argument("--start-min", type=int, default=60); ap.add_argument("--lag", type=int, default=30)
    ap.add_argument("--z-km", type=float, default=5.0); ap.add_argument("--xlim", type=float, nargs=2, default=(140, 235))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    tau_u, tau_d, s = FEATURES["w+"][2], FEATURES["w-"][2], FEATURE_SOFT["w"]
    torch.set_num_threads(16); dev = torch.device("cpu")
    ae, prop, norm, cfg, _ = load_run(Path(a.run).expanduser(), dev)
    d = ColumnData([load_member(member_files(a.source)[a.storm])], norm, dev); Z, NX = d.Z, d.NX; dtm = d.dt / 60
    t0 = int(round(a.start_min / dtm)); L = int(round(a.lag / dtm))
    with torch.no_grad():
        z = encode_members(ae, d)[0]
        lat = rollout_from(prop, z, t0, L, cfg["stage_b"]["halo_sites"], n_samples=1, boundary="persistence")
        wf = phys(ae.decode(lat[L - 1, 0:1])[0], norm, Z, "w")
    wt = phys(d.full(0, t0 + L)[0], norm, Z, "w")
    w_all = np.concatenate([phys(d.full(0, t)[0], norm, Z, "w").ravel() for t in range(t0, d.full(0, slice(0, None)).shape[0], 5)])
    x = (np.arange(NX) + 0.5) * 0.25; zk = BaseState.from_sounding().z / 1000.0
    yu, yd = (wt > tau_u).astype(float), (wt < -tau_d).astype(float)
    p = 1 / (1 + np.exp(-(wf - tau_u) / s))
    ce = -(yu * np.log(np.clip(p, 1e-6, 1)) + (1 - yu) * np.log(np.clip(1 - p, 1e-6, 1)))
    iz = int(np.argmin(np.abs(zk - a.z_km)))

    fig = plt.figure(figsize=(14, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.62, 0.48])
    ax = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    def fin(axx, title):
        axx.set_xlim(*a.xlim); axx.set_ylim(0, 14); axx.set_title(title, fontsize=8.5, loc="left"); axx.set_ylabel("z (km)")
        axx.axhline(zk[iz], color=MUTED, lw=0.6, ls=":")

    m = section(ax[0][0], wt, x, zk, "w"); fin(ax[0][0], f"(a) TRUE w at t = {a.start_min + a.lag} min (m/s)")
    fig.colorbar(m, ax=ax[0][0], shrink=0.85)
    hb = ax[0][1]
    bins = np.linspace(-12, 20, 161)
    hb.hist(w_all, bins=bins, color=TRUTH, log=True)
    hb.axvspan(tau_u, 20, color="#C0392B", alpha=0.12); hb.axvspan(-12, -tau_d, color="#1B8A4B", alpha=0.12)
    hb.axvline(tau_u, color="#C0392B", lw=1.2); hb.axvline(-tau_d, color="#1B8A4B", lw=1.2)
    fu, fd = (w_all > tau_u).mean(), (w_all < -tau_d).mean()
    hb.text(tau_u + 0.4, hb.get_ylim()[1] * 0.25, f"updraft\nw > {tau_u:g} m/s\n{100 * fu:.1f}% of points", fontsize=7.5, color="#C0392B")
    hb.text(-11.5, hb.get_ylim()[1] * 0.25, f"downdraft\nw < −{tau_d:g} m/s\n{100 * fd:.1f}% of points", fontsize=7.5, color="#1B8A4B")
    hb.set_xlabel("w (m/s)"); hb.set_ylabel("points (log)"); hb.set_title(f"(b) w over this storm, every 5th frame from {a.start_min} min:\n"
                                                                            "the calibrated thresholds", fontsize=8.5, loc="left")
    for c, (yy, col, ttl) in enumerate(((yu, "Reds", f"(c) TARGET: updraft present = 1[w > {tau_u:g} m/s]"),
                                        (yd, "Greens", f"(d) TARGET: downdraft present = 1[w < −{tau_d:g} m/s]\n(separate class)"))):
        mm = ax[0][2 + c].pcolormesh(x, zk, yy, cmap=col, vmin=0, vmax=1.3, shading="auto", rasterized=True); fin(ax[0][2 + c], ttl)
        cb = fig.colorbar(mm, ax=ax[0][2 + c], shrink=0.85, ticks=[0, 1]); cb.set_label("0 / 1")

    m = section(ax[1][0], wf, x, zk, "w"); fin(ax[1][0], f"(e) {a.label}: w at +{a.lag} min (m/s)")
    ax[1][0].contour(x, zk, yu, levels=[0.5], colors="k", linewidths=0.6)
    fig.colorbar(m, ax=ax[1][0], shrink=0.85)
    tf = ax[1][1]
    ww = np.linspace(-2, 6, 400)
    tf.plot(ww, (ww > tau_u).astype(float), color=TRUTH, lw=2, label="target: hard step 1[w > 2]")
    tf.plot(ww, 1 / (1 + np.exp(-(ww - tau_u) / s)), color=FCST, lw=2, label=f"forecast: p = σ((w − 2) / {s:g})")
    tf.axvspan(tau_u - s, tau_u + s, color=FCST, alpha=0.1)
    tf.set_xlabel("w (m/s)"); tf.set_ylabel("value"); tf.grid(alpha=0.25); tf.legend(fontsize=7, loc="upper left")
    tf.set_title(f"(f) the transformation: truth = hard step;\nforecast = sigmoid, width {s:g} m/s (≈2× recon. error)",
                 fontsize=8.5, loc="left")
    mm = ax[1][2].pcolormesh(x, zk, p, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
    ax[1][2].contour(x, zk, yu, levels=[0.5], colors="k", linewidths=0.6); fin(ax[1][2], "(g) forecast P(updraft) = σ((w − 2)/0.5);\nblack: TRUE updraft outline")
    fig.colorbar(mm, ax=ax[1][2], shrink=0.85, label="probability")
    mm = ax[1][3].pcolormesh(x, zk, ce, cmap="magma_r", vmin=0, vmax=4, shading="auto", rasterized=True)
    ax[1][3].contour(x, zk, yu, levels=[0.5], colors="#2A78C2", linewidths=0.6)
    fin(ax[1][3], f"(h) per-point cross-entropy (nats); mean {ce.mean():.3f}\nhigh = missed cells and false alarms")
    fig.colorbar(mm, ax=ax[1][3], shrink=0.85, label="nats")

    c1 = fig.add_subplot(gs[2, :]); c2 = fig.add_subplot(gs[3, :], sharex=c1)
    c1.plot(x, wt[iz], color=TRUTH, lw=1.6, label="truth"); c1.plot(x, wf[iz], color=FCST, lw=1.6, label=a.label)
    c1.axhline(tau_u, color="#C0392B", lw=1, ls="--", label=f"updraft threshold {tau_u:g} m/s")
    c1.set_ylabel("w (m/s)"); c1.legend(fontsize=7, ncol=3, loc="upper right"); c1.grid(alpha=0.25)
    c1.set_title(f"(i) cut at z = {zk[iz]:.1f} km: BEFORE the transformation (the continuous field)", fontsize=8.5, loc="left")
    c2.step(x, yu[iz], where="mid", color=TRUTH, lw=1.6, label="target y = 1[w > 2]")
    c2.plot(x, p[iz], color=FCST, lw=1.6, label="forecast p = σ((w − 2)/0.5)")
    c2.set_ylabel("0 – 1"); c2.set_ylim(-0.05, 1.1); c2.set_xlabel("x (km)"); c2.grid(alpha=0.25); c2.legend(fontsize=7, ncol=2, loc="upper right")
    c2.set_title("(j) the same cut AFTER the transformation: the 0/1 target and the forecast probability that the cross-entropy compares",
                 fontsize=8.5, loc="left")
    c1.set_xlim(*a.xlim)
    fig.suptitle(f"How the 'updraft present' target is built and scored ({a.storm}, unseen storm; forecast from t = {a.start_min} min)", fontsize=10)
    fig.savefig(out / f"fig_feat_prep_{a.storm}.png", dpi=130); plt.close(fig)
    print("[prep fig] done", flush=True)


if __name__ == "__main__":
    main()
