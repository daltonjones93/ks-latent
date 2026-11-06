"""Paired comparison of DA studies (da_study.py outputs) that share their truths and seeds: e.g. the
learned model error of a stochastic propagator against hand-set latent noise on a deterministic one.

Per truth (seed-averaged), for the latent LETKF mode: cycled forecast (prior) nMSE, analysis nMSE,
spread/error at the theta' observations (prior; 1 = calibrated, < 1 overconfident), median |front
displacement|, and the free-forecast nMSE from the analyses at 5 / 30 / 60 min lead. Every row is
reported as mean +- 95% CI over truths, and as a paired difference against the FIRST study given.
  python scripts/cm1/compare_studies.py ~/cm1_da/noise_det_0.05 ~/cm1_da/noise_det_0.02 ~/cm1_da/noise_stoch_0
"""
import json
import sys
from pathlib import Path

import numpy as np

MODE = "latent"
LEADS = (5, 30, 60)   # minutes; the studies use 1-min frames


def ci(x) -> tuple:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size < 2:
        return float(np.mean(x)) if x.size else float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def per_truth(d: Path) -> tuple[dict, dict]:
    rows = json.loads((d / "rows.json").read_text())
    meta = json.loads((d / "study.json.meta.json").read_text())
    out = {}
    for t, r in rows.items():
        seeds = r[MODE]
        g = lambda st, k: float(np.mean([s[st][k] for s in seeds if k in s[st]]))
        rec = {"prior": g("prior", "nmse_mean_dec"), "analysis": g("analysis", "nmse_mean_dec"),
               "sperr": float(np.mean([s["prior"]["th_obs_spread"] / s["prior"]["th_obs_rmse"] for s in seeds])),
               "front": g("prior", "front_abs_disp_km_median"), "noDA": r["free"]["prior"]["nmse_mean_dec"]}
        curves = [c["nmse_mean_dec"] for c in r["fc_" + MODE] if c]
        for L in LEADS:
            rec[f"fc{L}"] = float(np.mean([c[L - 1] for c in curves])) if curves and len(curves[0]) >= L else float("nan")
        out[t] = rec
    return out, meta["extra"]["args"]


def main():
    dirs = [Path(p).expanduser() for p in sys.argv[1:]]
    if not dirs:
        raise SystemExit(__doc__)
    studies = [(d, *per_truth(d)) for d in dirs]
    truths = sorted(set.intersection(*[set(s[1]) for s in studies]))
    if not truths:
        raise SystemExit("no common truths")
    for d, r, _ in studies:
        missing = set(r) - set(truths)
        if missing:
            print(f"note: {d.name} has {len(missing)} truths not in every study (left out)")
    keys = ["prior", "analysis", "sperr", "front"] + [f"fc{L}" for L in LEADS]
    names = {"prior": "cycled nMSE", "analysis": "analysis", "sperr": "sprd/err", "front": "front km",
             **{f"fc{L}": f"fc {L} min" for L in LEADS}}
    ref = studies[0][1]
    print(f"{len(truths)} common truths; reference (paired differences): {studies[0][0].name}")
    nd = ci([ref[t]["noDA"] for t in truths])
    print(f"no DA (reference study's free run): {nd[0]:.3f} +- {nd[1]:.3f}\n")
    print(f"{'study':30s} {'noise':>6s} " + " ".join(f"{names[k]:>16s}" for k in keys))
    for d, r, args in studies:
        cells = []
        for k in keys:
            m, h = ci([r[t][k] for t in truths])
            cells.append(f"{m:7.3f}+-{h:6.3f}")
        print(f"{d.name:30s} {args.get('model_noise', float('nan')):6.3f} " + " ".join(f"{c:>16s}" for c in cells))
    print("\npaired difference against the reference (negative = lower than the reference)")
    for d, r, args in studies[1:]:
        cells = []
        for k in keys:
            m, h = ci([r[t][k] - ref[t][k] for t in truths])
            cells.append(f"{m:+7.3f}+-{h:6.3f}")
        print(f"{d.name:30s} {args.get('model_noise', float('nan')):6.3f} " + " ".join(f"{c:>16s}" for c in cells))
        wins = np.mean([r[t]["prior"] < ref[t]["prior"] for t in truths])
        print(f"{'':37s}lower cycled nMSE than the reference on {100 * wins:.0f}% of truths")


if __name__ == "__main__":
    main()
