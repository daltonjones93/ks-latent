"""Plot skill vs window half-width from locality_test{,_pooled}.json."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import cm1_io as c

ex = json.loads((c.CACHE / "locality_test.json").read_text())["by_w"]
po = json.loads((c.CACHE / "locality_test_pooled.json").read_text())["by_w"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
col = {"ridge": "#2a6fdb", "mlp": "#d9822b"}
for a, reg in zip(ax, ["active", "quiet"]):
    a.axvspan(6, 7, color="#bbb", alpha=0.5, lw=0)
    a.text(6.5, 0.32, "spread front\nper frame", ha="center", fontsize=7, color="#555")
    for m in ["ridge", "mlp"]:
        w = [v["half_width_km"] for v in ex.values()]
        a.plot([max(x, 0.125) for x in w], [v[m][reg] for v in ex.values()], "-o", color=col[m], ms=4,
               lw=2, label=f"{m}, exact window")
        w = [v["half_width_km"] for v in po.values()]
        a.plot(w, [v[m][reg] for v in po.values()], "--s", color=col[m], ms=4, lw=1.5, mfc="none",
               label=f"{m}, far columns block-averaged")
    a.set_xscale("log"); a.set_xlabel("window half-width (km)  [w=0 plotted at 0.125]")
    a.set_title(f"{reg} columns" + (" (>10 dBZ)" if reg == "active" else ""))
    a.set_ylim(0.3, 0.9); a.grid(alpha=0.25)
ax[0].set_ylabel("skill vs persistence, one 5-min step\n(held-out members 9-10)")
ax[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.savefig(c.CACHE / "fig4_locality.png", dpi=140)
