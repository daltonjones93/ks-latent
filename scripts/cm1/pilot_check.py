"""B4: does every corner of the ens_B environment range give a storm that stays inside the domain?

For each member of a pilot ensemble (GrADS output): gust-front position (easternmost surface
theta' < -1 K), the x-extent of composite reflectivity > 20 dBZ, and peak surface rain proxy, every
10 min. A member fails if any storm echo or the front comes within `--margin-km` of either edge
(the lateral boundaries are open; the grid moves at a fixed 12.5 m/s, so environments with
different storm speeds drift). Writes a small JSON for plotting elsewhere.
Usage: python pilot_check.py ENS_DIR --out pilot.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import VARS
from ks_latent.cm1.io import load_grads_run

DX = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-km", type=float, default=30.0)
    a = ap.parse_args()
    ens = Path(a.ens)
    env = {r["member"]: r for r in csv.DictReader(open(ens / "manifest.csv"))}
    out = {"margin_km": a.margin_km, "members": {}}
    for mdir in sorted(ens.glob("members/mem*")):
        if not (mdir / "cm1.done").exists():
            print(f"{mdir.name}: not finished, skipped")
            continue
        d = load_grads_run(mdir / "cm1out_s.ctl")
        t = d["t"]
        sel = np.arange(0, len(t), 10)
        th_sfc = d["X"][sel, VARS.index("th"), 0]
        cref = d["cref"][sel]
        L = cref.shape[1] * DX
        front, west, east = [], [], []
        for k in range(len(sel)):
            cold = np.where(th_sfc[k] < -1.0)[0]
            front.append(float((cold.max() + 0.5) * DX) if cold.size else None)
            echo = np.where(cref[k] > 20)[0]
            west.append(float(echo.min() * DX) if echo.size else None)
            east.append(float((echo.max() + 1) * DX) if echo.size else None)
        lo = min([w for w in west if w is not None], default=None)
        hi = max([e for e in east if e is not None] + [f for f in front if f is not None], default=None)
        ok = lo is not None and lo >= a.margin_km and hi <= L - a.margin_km
        r = env.get(mdir.name, {})
        out["members"][mdir.name] = {"qv_offset_gkg": float(r.get("qv_offset_gkg", "nan")),
                                     "shear_scale": float(r.get("shear_scale", "nan")),
                                     "t_h": (t[sel] / 3600).tolist(), "front_km": front, "echo_west_km": west,
                                     "echo_east_km": east, "min_echo_west_km": lo, "max_east_km": hi,
                                     "inside_with_margin": bool(ok), "peak_cref_dbz": float(cref.max())}
        print(f"{mdir.name}: qv {r.get('qv_offset_gkg')} shear {r.get('shear_scale')}  echo spans {lo}-{hi} km "
              f"(domain 0-{L:.0f})  front at 6 h {front[-1]}  peak cref {cref.max():.0f} dBZ  -> "
              f"{'OK' if ok else 'TOO CLOSE TO AN EDGE'}")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
