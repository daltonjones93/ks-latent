"""Figures for the FPe report (docs/fpe_report): storm-centred sampling, nested-dropout ordering, what channel 0 is,
how much of its tendency a law explains, and the channel-0 law in physical units. Writes PNGs and a small JSON of
the numbers behind them.
  python scripts/cm1/fpe_report_figs.py --nested ~/cm1_runs/fpc_nested_stagec --control ~/cm1_runs/fpc_law_reuse_stagec \
      --source cache:A+A2+B --phys ~/cm1_da/fpe_phys/phys.json --speed ~/cm1_da/fpe_speed_s0/speed.json \
      --fpe-dirs ~/cm1_da/fpe_fpc_nested_stagec ~/cm1_da/fpe_fpc_law_reuse_stagec ~/cm1_da/fpe_refit_o1 \
      --out ~/cm1_da/fpe_report_figs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0
BLUE, ORANGE, GREEN, GRAY, INK = "#2563c9", "#d9711c", "#2f8f57", "#8c8c8c", "#222222"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.labelcolor": INK, "xtick.color": "#555555", "ytick.color": "#555555", "lines.linewidth": 1.6,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"[figs] {name}", flush=True)


def fig_sampling(member, out, rng, width_sites=48, n_frames=47, patch=8, n=6):
    """Hovmoller of composite reflectivity with Stage B windows: storm-centred vs uniform."""
    cref = member.cref
    T, NX = cref.shape
    xkm = np.arange(NX) * 0.25
    act = np.argwhere(cref > 10.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, mode in zip(axes, ("storm-centred", "uniform")):
        ax.pcolormesh(xkm, np.arange(T) / 60.0, np.clip(cref, 0, 60), cmap="Greys", vmin=0, vmax=60, shading="auto",
                      rasterized=True)
        ax.grid(False)
        for _ in range(n):
            if mode == "storm-centred":
                t, xc = act[rng.integers(len(act))]
                t = min(t, T - n_frames)
                x0 = min(max(xc // patch - width_sites // 2, 0), NX // patch - width_sites)
            else:
                t = rng.integers(0, T - n_frames)
                x0 = rng.integers(0, NX // patch - width_sites)
            ax.add_patch(plt.Rectangle((x0 * SITE_KM, t / 60.0), width_sites * SITE_KM, n_frames / 60.0, fill=False,
                                       ec=ORANGE if mode == "storm-centred" else BLUE, lw=1.4))
        ax.set_title(f"{mode} windows", fontsize=9)
        ax.set_xlabel("x (km)")
    axes[0].set_ylabel("time (h)")
    fig.suptitle("Composite reflectivity (dBZ, grey) and Stage B training windows (96 km x 47 min)", fontsize=9, y=1.03)
    save(fig, out, "fig_sampling")


def nmse_vs_k(ae, members, frames):
    errs = None
    for d in members:
        for t in frames:
            u = d.full(0, t)
            with torch.no_grad():
                z = ae.encode(u, None)
                c = z.shape[-1]
                if errs is None:
                    errs = np.zeros((c, 2))
                var = float(((u - u.mean()) ** 2).sum())
                for k in range(1, c + 1):
                    zk = z.clone(); zk[..., k:] = 0
                    errs[k - 1, 0] += float(((ae.decode(zk) - u) ** 2).sum())
                    errs[k - 1, 1] += var
    return errs[:, 0] / errs[:, 1]


def fig_ordering(curves, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    style = {"nested (ordered)": (ORANGE, "-"), "nested, seed 1": (ORANGE, "--"), "standard (unordered)": (BLUE, "-")}
    for name, y in curves.items():
        col, ls = style.get(name, (GRAY, "-"))
        ax.plot(np.arange(1, len(y) + 1), y, color=col, ls=ls, label=name, marker="o", ms=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("channels kept (first k, others zeroed)")
    ax.set_ylabel("reconstruction nMSE")
    ax.legend(frameon=False)
    save(fig, out, "fig_ordering")


def fig_channel0(ae, norm, Zl, out, s, b):
    """Decoded column for a field with only channel 0 set (others zero): physical profiles per value of z0."""
    P, c = 48, ae.cfg.latent_channels
    vals = [-2.0, -1.0, 0.0, 1.0]
    sc = np.repeat(np.asarray(norm.scale, float), Zl)
    cols = {}
    with torch.no_grad():
        for v in vals:
            z = torch.zeros(1, P, c); z[..., 0] = v
            u = ae.decode(z)[0].numpy()[:, u_mid(P, ae.cfg.patch)] * sc
            cols[v] = u
    names = {"th": "theta' (K)", "w": "w (m/s)", "u": "u' (m/s)", "qliq": "sqrt q_liq", "qice": "sqrt q_ice"}
    fig, axes = plt.subplots(1, len(names), figsize=(8.6, 3.0), sharey=True)
    cmap = plt.get_cmap("RdBu_r")
    for ax, (vn, lab) in zip(axes, names.items()):
        i = VARS.index(vn)
        for j, v in enumerate(vals):
            th_equiv = s * v + b
            ax.plot(cols[v][i * Zl:(i + 1) * Zl], np.arange(Zl), color=cmap(0.1 + 0.8 * j / (len(vals) - 1)),
                    label=f"z0 = {v:+.0f}  (theta'_sfc ~ {th_equiv:+.1f} K)")
        ax.set_xlabel(lab)
        ax.axvline(0, color="#bbbbbb", lw=0.8)
    axes[0].set_ylabel("model level (stretched; 0 = surface)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("What channel 0 encodes: decoded column with only channel 0 set", fontsize=9)
    save(fig, out, "fig_channel0_profiles")


def u_mid(P, patch):
    return (P // 2) * patch + patch // 2


def fig_hovmoller(z0, th, coef, s, b, out, halo):
    T = z0.shape[0]
    f0, f1 = 30, min(330, T - 2)
    zz = z0[f0 - 1:f1 + 1][:, :, None]
    dz = (0.5 * (zz[2:] - zz[:-2]))[..., 0]
    zc = zz[1:-1, :, 0]
    d1, d2 = (d[..., 0] for d in site_derivs(zz[1:-1], 2))
    k0, a1, a2, nu = coef
    law = k0 + a1 * d1 + a2 * zc * d1 + nu * d2
    x = np.arange(z0.shape[1]) * SITE_KM
    tt = np.arange(f0, f1) / 60.0
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.4), sharey=True)
    lim = 8.0
    panels = [(th[f0:f1], "truth: surface theta' (K)", lim),
              (s * zc + b, "channel 0 in K (s z0 + b)", lim),
              (s * dz, "truth: d/dt of s z0 (K/min)", None),
              (s * law, "4-term law (K/min)", None)]
    tlim = np.percentile(np.abs(s * dz), 99)
    for ax, (f, title, L) in zip(axes, panels):
        L = L or tlim
        im = ax.pcolormesh(x, tt, f, cmap="RdBu_r", vmin=-L, vmax=L, shading="auto", rasterized=True)
        ax.grid(False)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x (km)")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.05)
    axes[0].set_ylabel("time (h)")
    save(fig, out, "fig_hovmoller_law")


def rows(z0, halo, f0=30, f1=330):
    f1 = min(f1, z0.shape[0] - 2)
    z = z0[f0 - 1:f1 + 1][:, :, None]
    y = (0.5 * (z[2:] - z[:-2]))[..., 0]
    zc = z[1:-1, :, 0]
    d1, d2 = (d[..., 0] for d in site_derivs(z[1:-1], 2))
    sl = slice(halo, -halo)
    return tuple(a[:, sl].ravel() for a in (y, zc, d1, d2))


def binned_speed(R, s, b, edges, rng, n_boot=200):
    """Local advection speed per theta' bin: within a bin, OLS of d theta'/dt on (1, d theta'/dx, d2x theta')."""
    def speeds(Rs):
        y, zc, d1, d2 = (np.concatenate([r[i] for r in Rs]) for i in range(4))
        th = s * zc + b
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (th >= lo) & (th < hi) & (np.abs(d1) > 1e-3)
            if m.sum() < 500:
                out.append(np.nan); continue
            X = np.stack([np.ones(m.sum()), d1[m], d2[m]], 1)
            co, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            out.append(-co[1] * SITE_KM)
        return np.array(out)
    base = speeds(R)
    boots = np.array([speeds([R[i] for i in rng.integers(0, len(R), len(R))]) for _ in range(n_boot)])
    return base, np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)


def fig_law(phys, speed, binned, edges, out, phys1=None):
    tr = phys["train"]
    c0, c1 = tr["c0_km_min"], tr["c1_km_min_per_K"]
    thw, ceff = tr["theta_weighted_K"], tr["c_eff_km_min"]
    th = np.linspace(-16, 3, 200)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    mid = 0.5 * (edges[:-1] + edges[1:])
    base, lo, hi = binned
    ax.fill_between(mid, lo, hi, color=GRAY, alpha=0.25, lw=0, label="binned local speed, 95% CI")
    ax.plot(mid, base, color=GRAY, marker="o", ms=3, lw=1, label="binned local speed")
    ax.plot(th, c0 + c1 * th, color=ORANGE, label=f"law: c = {c0:+.2f} {c1:+.3f} theta'")
    if phys1:
        t1 = phys1["train"]
        ax.plot(th, t1["c0_km_min"] + t1["c1_km_min_per_K"] * th, color=ORANGE, ls="--", lw=1.2,
                label=f"law, autoencoder seed 1: c = {t1['c0_km_min']:+.2f} {t1['c1_km_min_per_K']:+.3f} theta'")
    thn = th[th < 0]
    ax.plot(thn, -abs(ceff) * np.sqrt(thn / thw), color=BLUE, ls="--",
            label="density-current scaling, through the front point")
    ax.plot([thw], [ceff], "o", color=INK, ms=5)
    ax.annotate(f"front: theta' {thw:.1f} K, c {ceff:+.2f} km/min", (thw, ceff), xytext=(-9.0, -0.62), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("surface theta' (K)")
    ax.set_ylabel("advection speed c (km/min; + = toward +x)")
    ax.set_title("(a) channel-0 law in physical units", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]
    st = speed["storms"]
    lawv = np.array([r["c_eff_km_min"] for r in st]); cold = np.array([r["v_cold_km_min"] for r in st])
    ax.plot(cold, lawv, "o", color=ORANGE, ms=4, mec="white", mew=0.5)
    lim = [min(cold.min(), lawv.min()) - 0.03, max(cold.max(), lawv.max()) + 0.03]
    ax.plot(lim, lim, color="#999999", lw=0.8, ls=":", label="1:1")
    b = speed["v_cold_km_min"]
    ax.set_xlabel("measured cold-pool centroid speed (km/min)")
    ax.set_ylabel("law's effective speed (km/min)")
    ax.set_title(f"(b) per storm (n = {len(st)}): r = {b['pearson']:+.2f}, slope {b['slope_measured_on_law']:.2f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "fig_law_physical")


def fig_r2(fpe_dirs, out):
    """Held-out R^2 of channel 0's tendency vs k: ordered vs unordered, local ODE vs PDE (threshold 0.03)."""
    series = {}
    for d in fpe_dirs:
        p = Path(d).expanduser() / "fpe.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        ordered = "nested" in r["run"] or "refit" in str(d)
        for f in r["fits"]:
            if f.get("time_order", 1) != 1 or abs(f["threshold"] - 0.03) > 1e-9:
                continue
            r2 = f["r2_test"][0] if isinstance(f["r2_test"], list) else f["r2_test"]
            key = (("ordered" if ordered else "unordered"), f["kind"])
            series.setdefault(key, {})[f["k"]] = max(r2, series.get(key, {}).get(f["k"], -1))
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    sty = {("ordered", "ode"): (ORANGE, ":", "ordered, local ODE"), ("ordered", "pde"): (ORANGE, "-", "ordered, PDE (d1, d2)"),
           ("ordered", "pde4"): (GREEN, "-", "ordered, PDE (d1..d4)"),
           ("unordered", "ode"): (BLUE, ":", "unordered, local ODE"), ("unordered", "pde"): (BLUE, "-", "unordered, PDE (d1, d2)")}
    for key, (col, ls, lab) in sty.items():
        if key in series:
            ks = sorted(series[key])
            ax.plot(ks, [series[key][k] for k in ks], color=col, ls=ls, marker="o", ms=3, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("channels in the law (first k)")
    ax.set_ylabel("held-out R^2, channel 0 tendency")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "fig_r2_vs_k")
    return {f"{a}|{b}": v for (a, b), v in series.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nested", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seed1", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--phys", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--phys-seed1", default=None)
    ap.add_argument("--fpe-dirs", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")
    files = member_files(a.source)
    phys = json.load(open(Path(a.phys).expanduser()))
    speed = json.load(open(Path(a.speed).expanduser()))
    s, b = phys["map"]["s_geometric"], phys["map"]["b"]
    summary = {"map": phys["map"]}

    ae, prop, norm, cfg, split = load_run(Path(a.nested).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]
    fig_sampling(load_member(files[split["train"][0]]), out, rng)
    test = [ColumnData([load_member(files[n])], norm, dev) for n in split["test"][: a.n_test]]
    Zl = test[0].Z
    frames = list(range(30, 331, 30))
    curves = {"nested (ordered)": nmse_vs_k(ae, test, frames)}
    fig_channel0(ae, norm, Zl, out, s, b)
    # Hovmoller of the first test storm, with the law
    with torch.no_grad():
        z = encode_members(ae, test[0])[0].numpy().astype(np.float64)
    sc = np.repeat(np.asarray(norm.scale, float), Zl)[:, None]
    T = z.shape[0]
    th = np.stack([(test[0].full(0, t)[0].numpy() * sc)[VARS.index("th") * Zl].reshape(-1, ae.cfg.patch).mean(1)
                   for t in range(T)])
    fig_hovmoller(z[..., 0], th, phys["train"]["coef_sites"], s, b, out, halo)
    # binned speeds on the training storms
    R = []
    for n in split["train"][: a.n_train]:
        d = ColumnData([load_member(files[n])], norm, dev)
        with torch.no_grad():
            R.append(rows(encode_members(ae, d)[0].numpy().astype(np.float64)[..., 0], halo))
    edges = np.arange(-16, 4.01, 2.0)
    binned = binned_speed(R, s, b, edges, rng)
    summary["binned_speed"] = {"edges": edges.tolist(), "speed": binned[0].tolist(), "lo": binned[1].tolist(),
                               "hi": binned[2].tolist()}
    fig_law(phys, speed, binned, edges, out, json.load(open(Path(a.phys_seed1).expanduser())) if a.phys_seed1 else None)
    del ae
    for tag, run in (("standard (unordered)", a.control), ("nested, seed 1", a.seed1)):
        if not run:
            continue
        ae2, _, norm2, _, split2 = load_run(Path(run).expanduser(), dev)
        test2 = [ColumnData([load_member(files[n])], norm2, dev) for n in split2["test"][: a.n_test]]
        curves[tag] = nmse_vs_k(ae2, test2, frames)
    summary["nmse_vs_k"] = {k: v.tolist() for k, v in curves.items()}
    fig_ordering(curves, out)
    summary["r2_vs_k"] = fig_r2(a.fpe_dirs, out)
    json.dump(summary, open(out / "figs.json", "w"), indent=1, default=str)
    print(f"[figs] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
