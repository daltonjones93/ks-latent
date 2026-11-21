"""10c: checks on a pilot CM1 ensemble before the main one: does a squall line form, stay inside the (moving) domain, and finish?
For each cached member: peak w (m/s) per hour, the x of the leading gust front (outermost column with surface theta' < -2 K, either side of
the cold pool) every 30 min, the minimum distance of the cold-pool edges and of any |w| > 2 m/s to the domain edges, and the mean
front speed relative to the moving grid. Compared against members of a reference ensemble (e.g. ens_C) from the same caches.

  python scripts/cm1/pilot_check.py --cache ~/cm1_cache_E_pilot --manifest ~/cm1_bundle/ens_E_pilot/manifest.json --ref ~/cm1_cache_C --n-ref 6
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import VARS, load_member

DX = 0.25


def summarize(path):
    m = load_member(path, mmap=True)
    X = m.X                                                               # (T, NX, V, Z)
    T, NX = X.shape[0], X.shape[1]
    iw, ith = VARS.index("w"), VARS.index("th")
    out = {"name": m.name, "frames": T}
    wmax = np.array([float(np.abs(np.asarray(X[t, :, iw, :], np.float32)).max()) for t in range(T)])
    out["wmax_per_hour"] = [round(float(wmax[h * 60:(h + 1) * 60].max()), 1) for h in range((T + 59) // 60)]
    fronts, edge_w = [], []
    for t in range(0, T, 30):
        sth = np.asarray(X[t, :, ith, 0], np.float32)
        cold = np.where(sth < -2.0)[0]
        fronts.append((t, None, None) if cold.size == 0 else (t, round(cold.min() * DX, 1), round(cold.max() * DX, 1)))
        act = np.where(np.abs(np.asarray(X[t, :, iw, :], np.float32)).max(1) > 2.0)[0]
        if act.size:
            edge_w.append(min(act.min(), NX - 1 - act.max()) * DX)
    out["cold_pool_edges_km"] = fronts
    lefts = [f[1] for f in fronts if f[1] is not None]; rights = [f[2] for f in fronts if f[2] is not None]
    out["min_edge_dist_cold_km"] = round(min([min(l, NX * DX - r) for l, r in zip(lefts, rights)] or [np.nan]), 1)
    out["min_edge_dist_w_km"] = round(min(edge_w or [np.nan]), 1)
    # front speeds (km/h relative to the grid) over the second half, where the line is mature
    tt = [f[0] for f in fronts if f[1] is not None and f[0] >= T // 2]
    if len(tt) >= 2:
        L = [f[1] for f in fronts if f[1] is not None and f[0] >= T // 2]; R = [f[2] for f in fronts if f[2] is not None and f[0] >= T // 2]
        hrs = (tt[-1] - tt[0]) / 60.0
        out["front_speed_kmh_left_right"] = [round((L[-1] - L[0]) / hrs, 1), round((R[-1] - R[0]) / hrs, 1)]
    out["initiated"] = bool(wmax[:90].max() > 5.0)
    out["sustained"] = bool(wmax[-60:].max() > 5.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--manifest", default=None)
    ap.add_argument("--ref", default=None); ap.add_argument("--n-ref", type=int, default=6); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = {}
    if a.manifest:
        for m in json.load(open(Path(a.manifest).expanduser()))["members"]:
            env[m["member"]] = {k: m.get(k) for k in ("mlcape", "shear03", "shear06", "rh36", "upper_shear", "shear_depth_km")}
    rows = []
    for f in sorted(Path(a.cache).expanduser().glob("*.npz")):
        s = summarize(f); key = s["name"].split("_")[-1]; s["env"] = env.get(key, {}); rows.append(s)
    ref = [summarize(f) for f in sorted(Path(a.ref).expanduser().glob("*.npz"))[: a.n_ref]] if a.ref else []
    for s in rows + [dict(r, name="REF " + r["name"]) for r in ref]:
        e = s.get("env", {})
        print(f"{s['name'][-22:]:22s} CAPE {e.get('mlcape', '-')!s:>7} sh03 {e.get('shear03', '-')!s:>5} sh06 {e.get('shear06', '-')!s:>5} "
              f"rh36 {e.get('rh36', '-')!s:>5} | init {s['initiated']!s:5} sust {s['sustained']!s:5} wmax/h {s['wmax_per_hour']} "
              f"| edge dist cold {s['min_edge_dist_cold_km']} w {s['min_edge_dist_w_km']} km | front speed {s.get('front_speed_kmh_left_right')}")
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps({"pilot": rows, "ref": ref}, indent=1))


if __name__ == "__main__":
    main()
