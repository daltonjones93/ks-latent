"""Wide periodic 1D KS dataset for window training (Paper II, option B of 2026-10-11).

One periodic domain of length L = 1600 (16 x the L = 100 training box), grid spacing 100/256 as in
the L = 100 data, dt = 0.05, snapshot interval 1.0, spin-up 500, trajectories of 250 time units:
4 training + 1 validation trajectories, independent initial conditions (seeds 1000+i). The
coverage, length x time, is 4 x 1600 x 250 = 1.6e6, against 50 x 100 x 250 = 1.25e6 for the
L = 100 training set. The fields are standardized with the L = 100 dataset's mean and std (stored
in metadata, with the wide domain's own as wide_normalization_*), so the 1D evaluation and data
assimilation scripts, which standardize with the L = 100 statistics, see the same convention.

Usage: python scripts/gen_ks1d_wide.py [--L 1600] [--out artifacts/datasets/ks1d_wide16.h5]
       python scripts/gen_ks1d_wide.py --profile smoke   # L = 400, 2+1 trajectories of 20 units
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
from pathlib import Path

import h5py
import numpy as np

from ks_latent.solver.dataset import generate_trajectory_dataset
from ks_latent.solver.ks import KSConfig

REF = Path("artifacts/datasets/stage1_trajectories_dtsnap1.h5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["full", "smoke"], default="full")
    ap.add_argument("--L", type=float, default=1600.0)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--time", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/datasets/ks1d_wide16.h5")
    a = ap.parse_args()
    if a.profile == "smoke":
        a.L, a.n_train, a.n_val, a.time = 400.0, 2, 1, 20.0
        a.out = "artifacts/datasets/smoke_ks1d_wide.h5"
    NX = int(round(a.L * 256 / 100))
    cfg = KSConfig(L=a.L, NX=NX, dt=0.05, snapshot_every=20, spinup_time=500.0 if a.profile == "full" else 50.0,
                   seed=a.seed)
    out = Path(a.out)
    generate_trajectory_dataset(cfg, out, n_train=a.n_train, n_val=a.n_val, trajectory_time=a.time)
    with h5py.File(REF, "r") as f:
        mean, std = float(f["metadata"].attrs["normalization_mean"]), float(f["metadata"].attrs["normalization_std"])
    with h5py.File(out, "r+") as f:
        md = f["metadata"].attrs
        md["wide_normalization_mean"], md["wide_normalization_std"] = md["normalization_mean"], md["normalization_std"]
        md["normalization_mean"], md["normalization_std"] = mean, std
        md["normalization_source"] = str(REF)
        raw = f["trajectories_raw"][:]
        f["trajectories"][...] = (raw - mean) / std
        wide_std = float(md["wide_normalization_std"])
    print(f"wrote {out}: L={a.L} NX={NX}, {a.n_train}+{a.n_val} x {a.time}; standardized with {REF} "
          f"(mean {mean:.3g}, std {std:.4f}; wide std {wide_std:.4f})")


if __name__ == "__main__":
    main()
