"""Cache the paired-run branches (restarted CM1 runs) with the SAME baseline as their base run.

A restarted run's GrADS file has nwrite records already counted: the first frames are empty (mtime 0) and the real ones start
at the first output after the restart. `transform` takes the far-field column of frame 0 as its baseline, which for a branch
would be an empty frame, so the base run's t = 0 frame is prepended for the baseline and dropped afterwards.
  python build_branch_cache.py --base ~/cm1_bundle/ens_P --branches ~/cm1_bundle/ens_Q --out ~/cm1_cache_Q
"""
import argparse
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import CM1Member, load_member, save_member
from ks_latent.cm1.io import read_grads, transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--branches", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, br, out = Path(a.base).expanduser(), Path(a.branches).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for short in sorted(p.name for p in (base / "members").glob("mem*")):
        d0 = read_grads(base / "members" / short / "cm1out_s.ctl")
        fr0 = {k: v[0] for k, v in d0.items()}
        del d0
        for bdir in sorted((br / "members").glob(f"{short}_*")):
            f = out / f"ens_Q__{bdir.name}.npz"
            if f.exists():
                continue
            d = read_grads(bdir / "cm1out_s.ctl")
            meta = read_grads(bdir / "cm1out_metadata.ctl")
            nt = min(d["th"].shape[0], meta["mtime"].shape[0])
            t = meta["mtime"].ravel()[:nt].astype(np.float64)
            idx = np.where(t > 0)[0]
            if len(idx) == 0:
                raise ValueError(f"{bdir.name}: no frames after the restart")
            g = lambda k: np.concatenate([fr0[k][None], d[k][idx]], 0).astype(np.float64)
            X = np.ascontiguousarray(transform(g)[1:].transpose(0, 3, 1, 2))   # (T, NX, V, Z) as member_from_cm1io
            cref = d["cref"][idx].astype(np.float32)
            m = CM1Member(name=f"ens_Q_{bdir.name}", X=X.astype(np.float16), cref=cref, t=t[idx])
            save_member(m, f)
            chk = load_member(f)
            print(f"{chk.name}: {chk.X.shape[0]} frames, t {chk.t[0]:.0f}..{chk.t[-1]:.0f}, dt {chk.dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
