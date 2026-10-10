"""J18c (2026-10-10): analysis error vs position within a patch, clean-data vs noisy-data model, single tiling
and K = 4 blend, from exp_h9_tiled_da.py --save-fields outputs (steady cycles). Usage:
fig_j18c_seams.py clean_T1.npz clean_T4.npz noisy_T1.npz noisy_T4.npz -> docs/figures/j18c_seams.png + json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["clean, 1 tiling", "clean, 4 tilings", "noisy, 1 tiling", "noisy, 4 tilings"]
out = {}
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for f, lab, ls in zip(sys.argv[1:5], labels, ["--", "-", "--", "-"]):
    d = np.load(f); p = int(d["patch"])
    key = "blend" if "blend" in d.files else "tiling0"
    E = (d[key] - d["truth"]) ** 2  # (T, NX, NX)
    E = E.mean(0)  # saved cycles are the steady ones
    pos = np.arange(E.shape[0]) % p
    prof = np.array([np.sqrt(0.5 * (E[pos == i].mean() + E[:, pos == i].mean())) for i in range(p)])
    out[lab] = {"profile": prof.tolist(), "edge_over_interior": float(prof[[0, p - 1]].mean() / prof[p // 2 - 1:p // 2 + 1].mean())}
    ax.plot(np.arange(p) + 0.5, prof, ls, color="C0" if "clean" in lab else "C1", label=lab)
ax.set_xlabel("position within patch (grid points)"); ax.set_ylabel("analysis RMSE (standardized)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
Path("docs/figures").mkdir(exist_ok=True); fig.savefig("docs/figures/j18c_seams.png", dpi=200)
Path("artifacts/experiments/da2d/j18c_seams.json").write_text(json.dumps(out, indent=1))
print({k: round(v["edge_over_interior"], 2) for k, v in out.items()})
