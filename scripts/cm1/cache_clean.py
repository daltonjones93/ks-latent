"""8a: cache finished members of an ensemble as float16 .npz (as build_cache.py) and, only after the cache has been re-read from
disk and checked, delete the member's large raw output (cm1out_*.dat) to stay under the disk quota. Kept per member: namelist,
sounding, logs, cm1.done, the .ctl files; the member can be re-run from the manifest.
Checks before any deletion: the cache reloads, name matches, frames == --frames (361 for 6 h at 1-min output), X finite, X is
float16, cref and t shapes match, frame spacing uniform (CM1Member.dt).
  python scripts/cm1/cache_clean.py ENS_DIR CACHE_DIR [--shard K N] [--keep-raw] [--frames 361]"""
import argparse
from pathlib import Path

import numpy as np

from ks_latent.cm1.data import load_member, member_from_cm1io, save_member
from ks_latent.cm1.io import load_grads_run


def verify(path: Path, name: str, frames: int) -> str | None:
    """None if the cache file is sound, else the reason it is not."""
    try:
        m = load_member(path)
        if m.name != name:
            return f"name {m.name} != {name}"
        if m.X.shape[0] != frames:
            return f"{m.X.shape[0]} frames, expected {frames}"
        if m.X.dtype != np.float16:
            return f"dtype {m.X.dtype}"
        if m.cref.shape[0] != frames or m.t.shape[0] != frames:
            return "cref/t length"
        step = max(1, frames // 12)
        if not np.isfinite(m.X[::step].astype(np.float32)).all():
            return "non-finite values"
        m.dt  # raises on non-uniform spacing
    except Exception as e:  # report and keep the raw output
        return f"{type(e).__name__}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ens"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--frames", type=int, default=361)
    a = ap.parse_args()
    ens, out = Path(a.ens).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k, n_sh = a.shard
    n_new = n_ok = n_bad = n_del = 0
    freed = 0
    for i, mdir in enumerate(sorted(ens.glob("members/mem*"))):
        if i % n_sh != k or not (mdir / "cm1.done").exists():
            continue
        name = f"{ens.name}_{mdir.name}"
        f = out / f"{ens.name}__{mdir.name}.npz"
        raws = [p for p in mdir.glob("cm1out_*.dat")]
        if not f.exists():
            if not raws:
                print(f"{name}: no cache and no raw output", flush=True); n_bad += 1; continue
            m = member_from_cm1io(load_grads_run(mdir / "cm1out_s.ctl", False), name)
            tmp = out / f".{f.name}.tmp.npz"
            save_member(m, tmp)
            tmp.rename(f)
            n_new += 1
        why = verify(f, name, a.frames)
        if why:
            print(f"{name}: CACHE REJECTED ({why}); raw kept", flush=True); n_bad += 1; f.rename(f.with_suffix(".bad")); continue
        n_ok += 1
        if not a.keep_raw:
            for p in raws:
                freed += p.stat().st_size; p.unlink(); n_del += 1
    print(f"done: {n_new} new caches, {n_ok} verified, {n_bad} problems, {n_del} raw files deleted ({freed / 1e9:.0f} GB)")


if __name__ == "__main__":
    main()
