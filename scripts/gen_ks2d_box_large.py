"""R4 (user 2026-10-12): a larger 44^2 training set for the small-source models -- as many independent box
trajectories as the 264^2 windows hold box-areas (4 train trajectories x 36 boxes = 144), so data volume is no longer
the difference between the small-source model D and the Paper II model. Same physics, layout and normalization
discipline as `ks_latent.solver.dataset2d.generate_trajectory_dataset2d` (stored float32), trajectories generated in
parallel with seeds `--seed-base + i` (disjoint from the default set's 0-29).

Usage: python scripts/gen_ks2d_box_large.py --n-train 144 --n-val 6 --workers 8
"""
import argparse
import dataclasses
from multiprocessing import Pool

import h5py
import numpy as np

from ks_latent.config import KSConfig2D
from ks_latent.solver.dataset2d import _write_metadata_group_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d

CFG = KSConfig2D()  # rebuilt for --L in __main__ (grid spacing kept at 44/128)
T = 25.0


def cfg_for(L):
    nx = int(round(L * 128 / 44))
    assert nx % 16 == 0, f"L={L} gives NX={nx}, not a whole number of 16-point sites"
    return dataclasses.replace(KSConfig2D(), Lx=L, Ly=L, NX=nx, NY=nx)


def one(job):  # (seed, L): the config is rebuilt in the worker (macOS spawns fresh interpreters)
    seed, L = job
    cfg = cfg_for(L)
    u0 = spinup2d(cfg, np.random.default_rng(seed))
    n_steps = int(round(T / cfg.dt))
    return integrate2d(u0, cfg, n_steps=n_steps).reshape(-1, cfg.NX * cfg.NY).astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=144)
    p.add_argument("--n-val", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="artifacts/datasets/ks2d_box44_large.h5")
    p.add_argument("--L", type=float, default=44.0, help="box side; NX = L / (44/128), must give whole 16-point sites")
    a = p.parse_args()
    CFG = cfg_for(a.L)
    n = a.n_train + a.n_val
    with Pool(a.workers) as pool:
        traj = np.stack(pool.map(one, [(a.seed_base + i, a.L) for i in range(n)]))
    mean, std = float(traj[:a.n_train].mean()), float(traj[:a.n_train].std())
    with h5py.File(a.out, "w") as f:
        f.create_dataset("trajectories", data=((traj - mean) / std).astype(np.float32))
        f.create_dataset("trajectories_raw", data=traj)
        _write_metadata_group_2d(f, CFG, {"n_train": a.n_train, "n_val": a.n_val, "trajectory_time": T,
                                          "normalization_mean": mean, "normalization_std": std,
                                          "seed_base": a.seed_base})
    print(f"wrote {a.out}: {traj.shape}, std {std:.4f}")
