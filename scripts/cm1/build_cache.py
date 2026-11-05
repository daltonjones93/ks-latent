"""B6: float16 member caches for an ensemble made by cluster/cm1 (GrADS or NetCDF output).

One `<ENS>__<member>.npz` per finished member (cm1.done), named `<ENS>_<member>` inside so members
of different ensembles never clash (every ensemble numbers its members from mem001). Members already
cached are skipped; unfinished ones are reported and skipped.
Usage: python build_cache.py ENS_DIR OUT_DIR [--format grads|netcdf]
"""
import argparse
from pathlib import Path

from ks_latent.cm1.data import load_member, member_from_cm1io, save_member
from ks_latent.cm1.io import load_grads_run, load_netcdf_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ens")
    ap.add_argument("out")
    ap.add_argument("--format", choices=["grads", "netcdf"], default="grads")
    ap.add_argument("--split", action="store_true", help="condensate by species: qc, qr, qi, qs, qg (D15)")
    a = ap.parse_args()
    ens, out = Path(a.ens).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = ens.name
    n_new = n_old = n_skip = 0
    for mdir in sorted(ens.glob("members/mem*")):
        f = out / f"{tag}__{mdir.name}.npz"
        if f.exists():
            n_old += 1
            continue
        if not (mdir / "cm1.done").exists():
            print(f"{mdir.name}: not finished, skipped")
            n_skip += 1
            continue
        run = (load_grads_run(mdir / "cm1out_s.ctl", a.split) if a.format == "grads"
               else load_netcdf_run(mdir / "cm1out.nc", a.split))
        m = member_from_cm1io(run, f"{tag}_{mdir.name}")
        save_member(m, f)
        chk = load_member(f)
        assert chk.X.shape == m.X.shape and chk.name == m.name
        print(f"{chk.name}: {chk.X.shape[0]} frames, dt {chk.dt:.0f} s -> {f.name}")
        n_new += 1
    print(f"done: {n_new} new, {n_old} already cached, {n_skip} unfinished")


if __name__ == "__main__":
    main()
