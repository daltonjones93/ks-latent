"""Overview figures of the CM1 2D squall-line runs (analysis 1).

fig 1  x-z sections of theta', w, condensate and reflectivity at 1, 3, 6 h (run_e1)
fig 2  time-distance (Hovmoller) plots: surface theta', composite reflectivity,
       ensemble spread of surface theta', and growth of the ensemble spread
fig 3  12 h control: composite reflectivity Hovmoller
Writes PNGs and a JSON of numbers to artifacts/cm1/.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import cm1_io as c

OUT = c.CACHE
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
XL = (40, 300)
ZL = (0, 16)


def sections(d):
    t, x, z, X, dbz = d["t"], d["x"], d["z"], d["X"], d["dbz"]
    times = [1, 3, 6]
    rows = [
        ("θ′ (K)", X[:, 2], "RdBu_r", (-8, 8)),
        ("w (m s⁻¹)", X[:, 1], "RdBu_r", (-10, 10)),
        ("√(q_liq+q_ice) (√(g kg⁻¹))", np.sqrt(X[:, 5] ** 2 + X[:, 6] ** 2), "Blues", (0, 2.0)),
        ("reflectivity (dBZ)", dbz, "viridis", (0, 60)),
    ]
    fig, ax = plt.subplots(len(rows), len(times), figsize=(13, 8.5), sharex=True, sharey=True,
                           constrained_layout=True)
    for r, (lab, F, cm, (lo, hi)) in enumerate(rows):
        for j, h in enumerate(times):
            ti = int(np.argmin(np.abs(t - h * 3600)))
            m = ax[r, j].pcolormesh(x, z, F[ti], cmap=cm, vmin=lo, vmax=hi, shading="auto", rasterized=True)
            ax[r, j].set_xlim(*XL); ax[r, j].set_ylim(*ZL)
            if r == 0:
                ax[r, j].set_title(f"t = {h} h")
            if j == 0:
                ax[r, j].set_ylabel("z (km)")
            if r == len(rows) - 1:
                ax[r, j].set_xlabel("x in moving grid (km)")
        fig.colorbar(m, ax=ax[r, :], shrink=0.9, label=lab)
    fig.suptitle("CM1 2D squall line, member 1 (grid moves east at 12.5 m s⁻¹)")
    fig.savefig(OUT / "fig1_sections.png", dpi=130)
    plt.close(fig)


def hovmoller(runs):
    d = runs[0]
    t, x = d["t"] / 3600, d["x"]
    sfc = np.array([r["X"][:, 2, 0, :] for r in runs])   # (M, T, NX) surface theta'
    allth = np.array([r["X"][:, 2] for r in runs])        # (M, T, Z, NX)
    spread_sfc = sfc.std(0)
    spread_all = allth.std(0)                             # (T, Z, NX)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6), constrained_layout=True)
    m = ax[0].pcolormesh(x, t, sfc[0], cmap="RdBu_r", vmin=-8, vmax=8, shading="auto", rasterized=True)
    fig.colorbar(m, ax=ax[0], label="surface θ′ (K)"); ax[0].set_title("member 1: surface θ′ (cold pool)")
    m = ax[1].pcolormesh(x, t, d["cref"], cmap="viridis", vmin=0, vmax=60, shading="auto", rasterized=True)
    fig.colorbar(m, ax=ax[1], label="dBZ"); ax[1].set_title("member 1: composite reflectivity")
    m = ax[2].pcolormesh(x, t, spread_sfc, cmap="magma_r", vmin=0, vmax=3, shading="auto", rasterized=True)
    fig.colorbar(m, ax=ax[2], label="K"); ax[2].set_title("10-member spread of surface θ′")
    for a in ax[:3]:
        a.set_xlim(*XL); a.set_xlabel("x (km)"); a.set_ylabel("t (h)")
    # where member differences are: column-max theta spread, log colour
    sp_mean = spread_all.reshape(len(t), -1).mean(1)
    col = spread_all.max(1)                               # (T, NX)
    from matplotlib.colors import LogNorm
    m = ax[3].pcolormesh(x, t, np.maximum(col, 1e-3), cmap="magma_r", norm=LogNorm(1e-2, 3),
                         shading="auto", rasterized=True)
    fig.colorbar(m, ax=ax[3], label="K (log)")
    ax[3].set_title("column-max spread of θ′ (any height)")
    ax[3].set_xlim(0, 400); ax[3].set_xlabel("x (km)"); ax[3].set_ylabel("t (h)")
    # speed of the east and west edges of the 0.5 K spread region, grid-relative
    edges = []
    for k in range(len(t)):
        on = np.where(col[k] > 0.5)[0]
        edges.append((x[on].min(), x[on].max()) if on.size else (np.nan, np.nan))
    edges = np.array(edges)
    ok = np.where((edges[:, 0] > 5) & (edges[:, 1] < 395))[0]
    fit = lambda e: float(np.polyfit(t[ok] * 3600, e[ok] * 1000, 1)[0])
    ext = edges
    fig.savefig(OUT / "fig2_hovmoller.png", dpi=130)
    plt.close(fig)
    # quiet vs active area
    active = (np.array([r["cref"] for r in runs]) > 10)
    return dict(
        spread_mean_K={f"{h}h": float(sp_mean[int(np.argmin(np.abs(t - h)))]) for h in [0, 0.5, 1, 2, 3, 6]},
        spread_edge_speed_ms={"east": fit(edges[:, 1]), "west": fit(edges[:, 0]),
                              "fit_window_h": [float(t[ok][0]), float(t[ok][-1])]},
        spread_extent_km={f"{h}h": [float(v) for v in ext[int(np.argmin(np.abs(t - h)))]] for h in [0.5, 1, 2, 3, 6]},
        active_fraction_cref_gt_10dBZ={f"{h}h": float(active[:, int(np.argmin(np.abs(t - h)))].mean())
                                       for h in [1, 3, 6]},
    )


def control(d):
    t, x = d["t"] / 3600, d["x"]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    m = ax.pcolormesh(x, t, d["cref"], cmap="viridis", vmin=0, vmax=60, shading="auto", rasterized=True)
    fig.colorbar(m, ax=ax, label="dBZ")
    ax.set_xlabel("x (km)"); ax.set_ylabel("t (h)"); ax.set_title("12 h control run: composite reflectivity")
    fig.savefig(OUT / "fig3_control.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    runs = [c.load_run(r) for r in c.MEMBERS]
    sections(runs[0])
    nums = hovmoller(runs)
    control(c.load_run("run"))
    (OUT / "viz_overview.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))
