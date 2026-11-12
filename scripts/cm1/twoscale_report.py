"""Summarize ts_eval.json files (6a): per model and lag, power ratio to truth per band (u, w, th, qliq), mean CRPS, spread/RMSE,
low-pass RMSE, median front error; and the paired-run divergence (T-chaos) if present.  python scripts/cm1/twoscale_report.py FILE..."""
import json
import sys

import numpy as np

SHOW = ("u", "w", "th", "qliq")
for f in sys.argv[1:]:
    r = json.load(open(f)); V = r["vars"]; bands = r["bands_km"]
    print(f"== {f}  bands {[f'{lo:g}-{hi:g}' for lo, hi in bands]}")
    for rn, x in r.get("runs", {}).items():
        print(f"-- {rn} (samples {x['samples']}, {len(x['cases'])} cases)")
        for L in map(str, r["lags"]):
            cs = [c["lags"][L] for c in x["cases"]]
            pw = np.mean([c["power"] for c in cs], 0); pt = np.mean([c["power_truth"] for c in cs], 0)
            rat = pw / pt
            rm = np.mean([c["rmse"] for c in cs], 0); cr = np.mean([c["crps"] for c in cs], 0); sp = np.mean([c["spread"] for c in cs], 0)
            lp = np.mean([c["rmse_lowpass"] for c in cs], 0)
            fr = np.nanmedian([c["front_abs_err_km"] for c in cs])
            dv = np.mean([c.get("divergence", 0.0) for c in cs])
            print(f"  lag {L:>2}: crps {cr.mean():.4f} rmse {rm.mean():.4f} spread/rmse {sp.mean() / rm.mean():.2f} lowpass-rmse {lp.mean():.4f} "
                  f"front {fr:.2f} km div {dv:.3f}")
            for v in SHOW:
                i = V.index(v); print(f"         {v:5s} power ratio " + " ".join(f"{q:5.2f}" for q in rat[i]))
    for rn, per in r.get("paired", {}).items():
        print(f"-- paired {rn}")
        for st, rec in per.items():
            cm = np.array(rec.get("cm1_tiny1e-3", [np.nan] * 61)); md = np.array(rec.get("model_divergence", [np.nan] * 61))
            for k0, c_ in rec.get("common_noise", {}).items():
                print(f"  {st} common-noise {k0}: log-growth/min 15-30: model {c_['slope_model_15_30']:+.3f} cm1 {c_['slope_cm1_15_30']:+.3f}; "
                      f"D_model(0)={c_['D_model'][0]:.4f} D_model(30)={c_['D_model'][30]:.4f}; D_model/D_cm1 at 30 = {c_['D_model_over_cm1_at_30']:.1f}")
            print(f"  {st}: D at 5/15/30/60 min  cm1 " + " ".join(f"{cm[k]:.3f}" for k in (5, 15, 30, 60))
                  + "  model " + " ".join(f"{md[k]:.3f}" for k in (5, 15, 30, 60)))
