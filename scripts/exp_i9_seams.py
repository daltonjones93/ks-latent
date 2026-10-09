"""Task I9: seam diagnostic (D10's edge/interior gradient ratio) on the shifted-tiling filter's saved
fields (exp_h9_tiled_da.py --save-fields). Edges are measured on tiling 0's patch grid and, as a check
that the blend has not just moved the seams, on the half-patch-shifted grid.
Usage: python scripts/exp_i9_seams.py h9_<tag>_fields.npz [...]; writes seams_I9.json (merged)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_d10_seams import seam_ratio  # noqa: E402

out_p = Path("artifacts/experiments/da2d/seams_I9.json")
out = json.loads(out_p.read_text()) if out_p.exists() else {}
for f in sys.argv[1:]:
    d = np.load(f)
    patch = int(d["patch"]); hp = patch // 2
    r = {}
    for k in ("truth", "tiling0", "blend", "floor1", "floorK"):
        u = d[k].astype(np.float64)
        r[k] = seam_ratio(u, patch)
        r[k + "_halfshift_grid"] = seam_ratio(np.roll(u, (-hp, -hp), axis=(1, 2)), patch)
    r["n_fields"] = int(d["truth"].shape[0])
    out[Path(f).stem] = r
    print(Path(f).stem, {k: round(v, 3) for k, v in r.items()})
out_p.write_text(json.dumps(out, indent=2))
