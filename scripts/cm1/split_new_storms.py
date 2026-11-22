"""10c: held-out storms for the new, varied ensembles (ens_E1-E5, ens_E_pilot4): stratified by mixed-layer CAPE tercile x 0-6 km
bulk-shear tercile (3 x 3 cells), 4 test and 2 validation storms per cell (36 test, 18 validation), drawn with a fixed seed. Storms
excluded for domain-wide initiation (10c screening) are never chosen. Member names are the training code's: <ensemble>_<member>.

  python scripts/cm1/split_new_storms.py --manifests DIR --exclude ens_E1_mem010,... --out docs/cm1_split_E.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True); ap.add_argument("--exclude", default="")
    ap.add_argument("--n-test", type=int, default=4); ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    excl = set(x for x in a.exclude.split(",") if x)
    rows = []
    for f in sorted(Path(a.manifests).glob("*.json")):
        ens = f.stem
        for m in json.load(open(f))["members"]:
            name = f"{ens}_{m['member']}"
            if name not in excl:
                rows.append((name, m["mlcape"], abs(m["shear06"])))
    cape = np.array([r[1] for r in rows]); sh = np.array([r[2] for r in rows])
    qc, qs = np.quantile(cape, [1 / 3, 2 / 3]), np.quantile(sh, [1 / 3, 2 / 3])
    cell = lambda c, s: (int(np.searchsorted(qc, c)), int(np.searchsorted(qs, s)))
    cells = {}
    for r in rows:
        cells.setdefault(cell(r[1], r[2]), []).append(r[0])
    rng = random.Random(a.seed)
    test, val = [], []
    for k in sorted(cells):
        names = sorted(cells[k]); rng.shuffle(names)
        test += names[: a.n_test]; val += names[a.n_test: a.n_test + a.n_val]
    out = {"test": sorted(test), "val": sorted(val), "excluded": sorted(excl), "n_storms": len(rows),
           "cape_terciles": qc.round(1).tolist(), "shear06_terciles": qs.round(2).tolist(),
           "cells": {f"{k[0]}{k[1]}": len(v) for k, v in sorted(cells.items())}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} storms; test {len(test)}, val {len(val)}; CAPE terciles {qc.round(0)}, shear terciles {qs.round(1)}; cells {out['cells']}")


if __name__ == "__main__":
    main()
