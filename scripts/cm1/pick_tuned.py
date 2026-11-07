"""Pick the latent model-noise setting from validation-storm tuning studies (G0 protocol: chosen on the 6
validation storms only, before any test-storm run). Criterion: lowest mean cycled forecast (prior) nMSE.
Prints the chosen value (last line) and writes the choice and the table to --record.
  python scripts/cm1/pick_tuned.py --record ~/cm1_da/tune_attn256/choice.json ~/cm1_da/tune_attn256_noise_0.05 ...
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--record", required=True)
a = ap.parse_args()
rows = []
for d in a.dirs:
    d = Path(d).expanduser()
    s = json.loads((d / "study.json").read_text())
    meta = json.loads((d / "study.json.meta.json").read_text())
    val = s["modes"]["latent"]["prior"]["nmse_mean_dec"]["mean"]
    rows.append({"dir": str(d), "model_noise": meta["config"]["model_noise"], "val_prior_nmse": val,
                 "truths": s["truths"]})
truths = {tuple(r["truths"]) for r in rows}
if len(truths) != 1:
    raise ValueError(f"tuning studies used different truths: {truths}")
best = min(rows, key=lambda r: r["val_prior_nmse"])
for r in sorted(rows, key=lambda r: r["model_noise"]):
    print(f"  noise {r['model_noise']:<6g} validation prior nMSE {r['val_prior_nmse']:.4f}{'  <- chosen' if r is best else ''}")
Path(a.record).expanduser().parent.mkdir(parents=True, exist_ok=True)
Path(a.record).expanduser().write_text(json.dumps({"criterion": "lowest validation prior nMSE", "chosen": best,
                                                   "table": rows}, indent=1))
print(best["model_noise"])
