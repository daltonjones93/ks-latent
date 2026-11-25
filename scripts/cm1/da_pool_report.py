"""10c: merge da_pool.py parts; choose RTPS per system on validation; test table, JSON and figure.

  python scripts/cm1/da_pool_report.py --val DIR_VAL --test DIR_TEST --out docs/fpe_report/data/pool_da.json \
      --fig docs/fpe_report/figs_pool/fig_pool_da.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYS = {"pool_phys": "pool 4 x 8, physical LETKF", "single_phys": "one model x 32, physical LETKF",
       "single_latent": "one model x 32, latent LETKF"}
COL = {"free_pool": "#9AA0A6", "free_single": "#C9CCD1", "pool_phys": "#1F8A4B", "single_phys": "#2A78C2", "single_latent": "#E07B39"}


def load(d):
    res = {}
    for p in sorted(Path(d).expanduser().glob("part_*.json")):
        res.update(json.load(open(p))["results"])
    return res


def ci(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "ci95": float(1.96 * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan"), "n": int(x.size)}


def per_storm(res, key, f):
    return np.array([np.mean([f(r) for r in res[t][key]]) for t in sorted(res)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    val, test = load(a.val), load(a.test)
    nm = lambda r: r["prior"]["nmse_mean_dec"]
    keys = [k for k in next(iter(test.values())) if "|" in k]
    rt = sorted({k.split("|")[1] for k in keys})
    choice, vtab = {}, {}
    for s in SYS:
        cand = {r: float(per_storm(val, f"{s}|{r}", nm).mean()) for r in rt if f"{s}|{r}" in next(iter(val.values()))}
        if cand:
            vtab[s] = cand; choice[s] = min(cand, key=cand.get)
    print(f"validation ({len(val)} storms), prior nMSE by RTPS:", json.dumps(vtab), "\nchosen:", choice)
    storms = sorted(test)
    free = per_storm(test, "free_pool", nm)
    out = {"validation": vtab, "choice": choice, "n_test": len(storms), "storms": storms, "systems": {}}
    rows = ["free_pool", "free_single"] + [f"{s}|{choice[s]}" for s in SYS if s in choice]
    print(f"\ntest ({len(storms)} storms), second half of the cycles:")
    for k in rows:
        x = per_storm(test, k, nm)
        d = {"prior_nmse": ci(x), "minus_free_pool": ci(x - free), "frac_better_than_free_pool": float(np.mean(x < free))}
        if "analysis" in test[storms[0]][k][0]:
            d["analysis_nmse"] = ci(per_storm(test, k, lambda r: r["analysis"]["nmse_mean_dec"]))
        if "clip_frac_mean" in test[storms[0]][k][0]:
            d["clip_frac_mean"] = float(per_storm(test, k, lambda r: r["clip_frac_mean"]).mean())
            d["clip_frac_max"] = float(per_storm(test, k, lambda r: r["clip_frac_max"]).max())
            d["frac_storms_clipped"] = float(np.mean(per_storm(test, k, lambda r: r["cycles_clipped"]) > 0))
            d["cycles_clipped_mean"] = float(per_storm(test, k, lambda r: r["cycles_clipped"]).mean())
            print(f"  {k}: clamp fired in {d['cycles_clipped_mean']:.1f} of 36 cycles on average, on {100 * d['frac_storms_clipped']:.0f}% of storms; "
                  f"mean fraction of latent entries clipped {d['clip_frac_mean']:.2e}, max {d['clip_frac_max']:.2e}")
        d["spread_error"] = ci(per_storm(test, k, lambda r: np.sqrt(r["prior"]["spread"] / r["prior"]["nmse_mean_dec"])))
        for q in ("th_obs_rmse", "th_obs_spread", "refl_echo_rmse", "vr_obs_rmse", "front_abs_disp_km_median"):
            d[q] = ci(per_storm(test, k, lambda r, q=q: r["prior"][q]))
        d["rmse_phys"] = {v: ci(per_storm(test, k, lambda r, v=v: r["prior"]["rmse_phys"][v]))["mean"]
                          for v in test[storms[0]][k][0]["prior"]["rmse_phys"]}
        d["nmse_by_cycle"] = np.mean([np.mean([r["nmse_by_cycle"] for r in test[t][k]], 0) for t in storms], 0).tolist()
        fc = {}
        for q in ("nmse_mean", "nmse_blend10", "nmse_member", "pw_mean", "pw_blend10", "pw_member"):
            fc[q] = np.mean([np.mean([np.mean([v[q] for v in r["forecast"].values()], 0) for r in test[t][k]], 0) for t in storms], 0).tolist()
        d["forecast"] = fc
        out["systems"][k] = d
        print(f"  {k:22s} prior {d['prior_nmse']['mean']:.3f} +- {d['prior_nmse']['ci95']:.3f}  vs free pool "
              f"{d['minus_free_pool']['mean']:+.3f} +- {d['minus_free_pool']['ci95']:.3f} ({100 * d['frac_better_than_free_pool']:.0f}% storms)"
              + (f"  analysis {d['analysis_nmse']['mean']:.3f}" if "analysis_nmse" in d else "")
              + f"  spread/err {d['spread_error']['mean']:.2f}  th {d['th_obs_rmse']['mean']:.2f} K  front {d['front_abs_disp_km_median']['mean']:.2f} km"
              + f"  w {d['rmse_phys']['w']:.2f} m/s")
    pk = f"pool_phys|{choice['pool_phys']}"
    for k in [r for r in rows if r.startswith("single")]:
        x = per_storm(test, pk, nm) - per_storm(test, k, nm)
        out["systems"][pk][f"minus_{k}"] = ci(x); out["systems"][pk][f"frac_better_than_{k}"] = float(np.mean(x < 0))
        print(f"  pool minus {k}: {x.mean():+.4f} +- {1.96 * x.std(ddof=1) / np.sqrt(x.size):.4f}, pool better on {100 * np.mean(x < 0):.0f}%")
    leads = 5 * (np.arange(len(out["systems"][pk]["forecast"]["nmse_mean"])) + 1)
    out["forecast_leads_min"] = leads.tolist()
    print("\nforecasts from the analyses (mean over cycles 11/17/23/29), nMSE at 15 / 30 / 60 min:")
    for k in rows:
        f = out["systems"][k]["forecast"]
        g = lambda q: " / ".join(f"{f[q][i]:.3f}" for i in (2, 5, 11) if i < len(f[q]))
        print(f"  {k:22s} mean {g('nmse_mean')}   blend10 {g('nmse_blend10')}   member {g('nmse_member')}   "
              f"power 2-10 km at 60: mean {f['pw_mean'][-1]:.2f} blend {f['pw_blend10'][-1]:.2f} member {f['pw_member'][-1]:.2f}")
    json.dump(out, open(a.out, "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw=dict(wspace=0.27, left=0.05, right=0.99, top=0.85, bottom=0.25))
    lab = lambda k: {"free_pool": "pool, no DA", "free_single": "one model, no DA"}.get(k, SYS.get(k.split("|")[0], k) + f" (RTPS {k.split('|')[1]})" if "|" in k else k)
    for k in rows:
        c = COL[k.split("|")[0]]
        y = out["systems"][k]["nmse_by_cycle"]
        ax[0].plot(5 * (np.arange(len(y)) + 1), y, color=c, lw=2.2 if k.startswith("pool") else 1.6,
                   ls="--" if k.startswith("free") else "-", label=lab(k))
    ax[0].set_xlabel("minutes of cycling"); ax[0].set_ylabel("prior nMSE (all variables)"); ax[0].set_title("Error of the forecast before each analysis", fontsize=10)
    xs = per_storm(test, f"single_latent|{choice['single_latent']}", nm); xp = per_storm(test, pk, nm)
    ax[1].scatter(xs, xp, s=40, color=COL["pool_phys"], edgecolor="k", lw=0.5, zorder=3, label="vs one model, latent LETKF")
    xf = per_storm(test, "free_pool", nm)
    ax[1].scatter(xf, xp, s=40, color=COL["free_pool"], edgecolor="k", lw=0.5, zorder=3, marker="s", label="vs pool, no DA")
    lim = [0, 1.05 * max(xs.max(), xp.max(), xf.max())]
    ax[1].plot(lim, lim, "k:", lw=0.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("other system, prior nMSE"); ax[1].set_ylabel("pool DA, prior nMSE"); ax[1].set_title("Per test storm (below the line: pool DA better)", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left", frameon=False)
    f, ff = out["systems"][pk]["forecast"], out["systems"]["free_pool"]["forecast"]
    ax[2].plot(leads, f["nmse_mean"], "-o", color="#2A78C2", ms=4, lw=2, label="from analyses: average")
    ax[2].plot(leads, f["nmse_blend10"], "-s", color="#1F8A4B", ms=4, lw=2, label="from analyses: 10 km blend")
    ax[2].plot(leads, f["nmse_member"], ":D", color="#9AA0A6", ms=4, lw=1.6, label="from analyses: one member")
    ax[2].plot(leads, ff["nmse_mean"], "--", color="#2A78C2", lw=1.4, label="no DA: average")
    ax[2].plot(leads, ff["nmse_blend10"], "--", color="#1F8A4B", lw=1.4, label="no DA: 10 km blend")
    ax[2].set_xlabel("forecast lead (min)"); ax[2].set_ylabel("nMSE"); ax[2].set_title("60-min forecasts launched during cycling", fontsize=10)
    ax[2].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax[2].set_ylim(top=ax[2].get_ylim()[1] + 0.18 * (ax[2].get_ylim()[1] - ax[2].get_ylim()[0]))
    for x in ax:
        x.grid(alpha=0.25)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(f"Cycled DA twin, {len(storms)} test storms: theta' + reflectivity + radial velocity every 2 km, 5-min cycles, 3 h, held boundaries", fontsize=11)
    fig.savefig(a.fig, dpi=150); plt.close(fig)
    print("figure", a.fig)


if __name__ == "__main__":
    main()
