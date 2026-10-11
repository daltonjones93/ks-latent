"""Copy a trajectory dataset with one fixed realization of Gaussian noise added to every field
(Paper II's noisy training, 1D; the 2D crop trainer does the same in memory, --data-noise).

The noise has standard deviation `--noise` in standardized units, is added to `trajectories`
(standardized) and, scaled by the dataset std, to `trajectories_raw`; train and validation splits
both carry it, as in 2D. The normalization statistics are left unchanged. Metadata gain
data_noise, data_noise_seed and noise_source.

Usage: python scripts/make_noisy_dataset.py --src artifacts/datasets/stage1_trajectories_dtsnap1.h5 \
         --noise 0.1 --out artifacts/datasets/stage1_trajectories_dtsnap1_noise0.1.h5
"""
from __future__ import annotations

import argparse
import shutil

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    shutil.copyfile(a.src, a.out)
    rng = np.random.default_rng(a.seed)
    with h5py.File(a.out, "r+") as f:
        std = float(f["metadata"].attrs["normalization_std"])
        eps = a.noise * rng.standard_normal(f["trajectories"].shape)
        f["trajectories"][...] = f["trajectories"][:] + eps
        f["trajectories_raw"][...] = f["trajectories_raw"][:] + std * eps
        f["metadata"].attrs["data_noise"] = a.noise
        f["metadata"].attrs["data_noise_seed"] = a.seed
        f["metadata"].attrs["noise_source"] = a.src
    print(f"wrote {a.out}: noise std {a.noise} (standardized units), seed {a.seed}")


if __name__ == "__main__":
    main()
