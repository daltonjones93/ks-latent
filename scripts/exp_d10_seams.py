"""Task D10 (docs/full_paper_tasks.md): seam diagnostic on saved DA fields (D6 runs).

For each saved field (truth, analysis = decode(mean z), free run, reconstruction D(E(truth))),
the mean |grad u| across patch edges divided by the mean |grad u| inside patches. Gradients are
one-sided differences between neighbouring grid points: an "edge" difference straddles a patch
boundary (between the last point of one patch and the first of the next), an "interior"
difference lies inside one patch. A smooth field gives a ratio near 1; decoder seams raise it.
Reads artifacts/experiments/da2d/da2d_D6_L*_local_N64_s0_fields.npz; writes
artifacts/experiments/da2d/seams_D10.json.
"""
import glob
import json
import re

import numpy as np


def seam_ratio(u, patch):
    """u: (T, NX, NY). Ratio of mean |du| across patch edges to mean |du| inside patches, x and y."""
    dx = np.abs(np.diff(u, axis=1, append=u[:, :1]))  # periodic, dx[:, i] = u[i+1] - u[i]
    dy = np.abs(np.diff(u, axis=2, append=u[:, :, :1]))
    edge = (np.arange(u.shape[1]) % patch) == patch - 1
    e = np.concatenate([dx[:, edge, :].ravel(), dy[:, :, edge].ravel()]).mean()
    i = np.concatenate([dx[:, ~edge, :].ravel(), dy[:, :, ~edge].ravel()]).mean()
    return float(e / i)


def main():
    out = {}
    for f in sorted(glob.glob("artifacts/experiments/da2d/da2d_D6_L*_local_N64_s0_fields.npz")):
        d = np.load(f)
        L = int(re.search(r"_L(\d+)_", f)[1])
        patch = int(d["NX"]) // int(d["n_sites"])
        out[L] = {k: seam_ratio(d[k].astype(np.float64), patch) for k in ("truth", "analysis", "free", "recon")}
        out[L]["n_fields"] = int(d["truth"].shape[0])
        print(L, {k: round(v, 3) for k, v in out[L].items()})
    with open("artifacts/experiments/da2d/seams_D10.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
