"""R14 (user 2026-10-13): a 4x larger 1x-box Kolmogorov training set (as the 2D KS C2 recipe: 4x the data), for
the small-box models at alpha = 0.05 and 0.1. Same physics, grid, trajectory length (40 time units, dt_snap 0.2) and
normalization discipline as `ks_latent.solver.dataset2d.generate_trajectory_dataset_kolmogorov` (stored float32),
trajectories generated in parallel with seeds `--seed-base + i` (disjoint from the default sets' 0-29).

Usage: python scripts/gen_kolmogorov_box_large.py --alpha 0.05 --n-train 96 --n-val 6 --workers 6 --out ...
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import dataclasses
from multiprocessing import Pool

import h5py
import numpy as np

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.dataset2d import _write_metadata_group_2d
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov

T = 40.0


def one(job):  # (seed, alpha): config rebuilt in the worker (macOS spawns fresh interpreters)
    seed, alpha = job
    cfg = KolmogorovConfig(alpha=alpha)
    u0 = spinup_kolmogorov(cfg, np.random.default_rng(seed))
    n_steps = int(round(T / cfg.dt))
    return integrate_kolmogorov(u0, cfg, n_steps=n_steps).reshape(-1, cfg.NX * cfg.NY).astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-val", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = KolmogorovConfig(alpha=a.alpha)
    n = a.n_train + a.n_val
    with Pool(a.workers) as pool:
        traj = np.stack(pool.map(one, [(a.seed_base + i, a.alpha) for i in range(n)]))
    mean, std = float(traj[:a.n_train].mean()), float(traj[:a.n_train].std())
    with h5py.File(a.out, "w") as f:
        f.create_dataset("trajectories", data=((traj - mean) / std).astype(np.float32))
        f.create_dataset("trajectories_raw", data=traj)
        _write_metadata_group_2d(f, cfg, {"n_train": a.n_train, "n_val": a.n_val, "trajectory_time": T,
                                          "normalization_mean": mean, "normalization_std": std,
                                          "seed_base": a.seed_base})
    print(f"wrote {a.out}: {traj.shape}, std {std:.4f}")
