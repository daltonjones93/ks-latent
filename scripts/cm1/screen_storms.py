"""10c: screen a cached CM1 ensemble for domain-wide initiation: a storm is flagged when |w| > 2 m/s appears within 10 km of a domain
edge before --minutes (default 120; reference single squall lines: first contact after ~190 min). Prints the flagged member names
(<ensemble>_<member>, comma-separated) on the last line; --out appends them to a file.

  python scripts/cm1/screen_storms.py ~/cm1_cache_E2 --out ~/flagged_E.txt
"""
import argparse
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import VARS, load_member


def first_contact(f, edge_km=10.0, thr=2.0, t_max=361):
    m = load_member(f, mmap=True); X = m.X; e = int(edge_km / 0.25); iw = VARS.index("w")
    for t in range(min(t_max, X.shape[0])):
        a = np.abs(np.asarray(X[t, :, iw, :], np.float32)).max(1)
        if a[:e].max() > thr or a[-e:].max() > thr:
            return t
    return t_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("--minutes", type=int, default=120); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    flagged = []
    for f in sorted(Path(a.cache).expanduser().glob("*.npz")):
        t = first_contact(f, t_max=a.minutes)
        if t < a.minutes:
            ens, mem = f.stem.split("__")
            flagged.append(f"{ens}_{mem}")
            print(f"{f.stem}: first edge contact {t} min", flush=True)
    if a.out:
        with open(Path(a.out).expanduser(), "a") as fh:
            fh.write("".join(x + "\n" for x in flagged))
    print(",".join(flagged))


if __name__ == "__main__":
    main()
