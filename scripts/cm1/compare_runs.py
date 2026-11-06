"""One table comparing trained CM1 runs on their test members (results.json of each run directory).

Columns: skill vs persistence (forecast nMSE / persistence nMSE; lower is better, and comparable across
runs with different normalizers) at 5 / 30 / 60 min, surface theta' RMSE at 60 min (K), cold-pool max-C
bias at 60 min (forecast / truth - 1), median |front displacement| at 30 / 60 min (km), low-level-rain
FSS (q > 1 g/kg, 1 km window) at 60 min, and reconstruction nMSE (storm columns).
  python scripts/cm1/compare_runs.py artifacts/cm1/runs/saidie_d34_c32_kuniform artifacts/cm1/runs/saidie_d12_stagec ...
A results_<tag>.json file (a train_cm1.py --results-tag re-score) can be given instead of a run directory.
"""
import json
import sys
from pathlib import Path


def row(d: Path) -> dict:
    """`d`: a run directory (its results.json) or a results_<tag>.json file inside one."""
    f = d if d.suffix == ".json" else d / "results.json"
    r = json.loads(f.read_text())
    if d.suffix == ".json":
        d = Path(f"{d.parent.name}:{d.stem.removeprefix('results_')}")
    fc, rec = r["forecast"], r["reconstruction"]
    lead = [round(s / 60) for s in fc["lead_s"]]
    j = {m: lead.index(m) for m in (5, 30, 60) if m in lead}
    g = lambda k, m: fc[k][j[m]] if k in fc and m in j else float("nan")
    ratio = lambda m: g("nmse", m) / g("nmse_pers", m)
    out = {"run": d.name, "skill5": ratio(5), "skill30": ratio(30), "skill60": ratio(60),
           "th60": g("th_sfc_rmse", 60), "front30": g("disp_abs_km_median", 30), "front60": g("disp_abs_km_median", 60),
           "recon": rec.get("nmse_storm", float("nan"))}
    out["cbias60"] = g("cold_pool_cmax_fc", 60) / g("cold_pool_cmax_truth", 60) - 1 if "cold_pool_cmax_fc" in fc else float("nan")
    cr = lambda m: g("crps", m) / g("crps_pers", m)        # CRPS skill vs persistence (MAE skill if deterministic)
    out.update({"crps5": cr(5), "crps30": cr(30), "crps60": cr(60), "sperr60": g("spread_err", 60)})
    out["fss60"] = fc["fss_lowliq"]["1"]["1km"][j[60]] if "fss_lowliq" in fc and 60 in j else float("nan")
    return out


def main():
    rows = [row(Path(p)) for p in sys.argv[1:]]
    hdr = (f"{'run':28s} {'skill 5':>8s} {'30':>6s} {'60':>6s} {'CRPS 5':>7s} {'30':>6s} {'60':>6s} {'sp/err60':>8s} "
           f"{'th60 K':>7s} {'Cbias60':>8s} {'front30':>8s} {'front60':>8s} {'FSS60':>6s} {'recon':>6s}")
    print(hdr)
    for r in rows:
        print(f"{r['run']:28s} {r['skill5']:8.3f} {r['skill30']:6.3f} {r['skill60']:6.3f} {r['crps5']:7.3f} {r['crps30']:6.3f} "
              f"{r['crps60']:6.3f} {r['sperr60']:8.2f} {r['th60']:7.2f} {r['cbias60']:+8.1%} {r['front30']:8.2f} "
              f"{r['front60']:8.2f} {r['fss60']:6.3f} {r['recon']:6.3f}")
    print("skill = nMSE / persistence; CRPS = CRPS / persistence MAE (a deterministic forecast's CRPS is its MAE); "
          "sp/err = ensemble spread / error of its mean (1 = calibrated)")


if __name__ == "__main__":
    main()
