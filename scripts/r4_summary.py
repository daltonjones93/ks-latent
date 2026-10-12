"""One-line summaries for the R4 queue logs: `r4_summary.py dky <crop eval json>` or `r4_summary.py da <h9 json>`."""
import json
import sys

kind, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except (OSError, ValueError) as e:
    print(f"(no result: {e})")
    sys.exit(0)
if kind == "dky":
    print(f"D_KY {d['model_D_KY']:.1f} n+ {d.get('model_n_positive')} lam1 {d.get('model_lambda1', 0):.3f} "
          f"truth {d.get('true_D_KY')} err% {d.get('rel_err_pct')}")
else:
    s = d["steady"]
    print(" ".join(f"{k} {s[k]:.3f}" for k in ("blend", "tiling0", "interp", "floorK", "floor1", "free", "calib_blend")
                   if s.get(k) is not None))
