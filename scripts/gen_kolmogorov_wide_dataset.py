"""P0 (docs/ks_local_field_ltransfer.tex Sec. crop_training): one wide
Kolmogorov-flow domain, Lx = x_scale * 2pi, Ly = 2pi, same nu / forcing /
dt / dt_snap as the periodic-box datasets, at the same grid spacing
(NX = 128 * x_scale, NY = 128).

Preemptable: each trajectory is saved to its own .npy as soon as it
finishes; a rerun skips finished ones and only assembles the HDF5 once all
exist. Stores the normalized float32 field (stats from the training split
only) and metadata; raw = normalized * std + mean.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import time
from pathlib import Path

import h5py
import numpy as np

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.dataset2d import _write_metadata_group_2d
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--x-scale", type=int, default=16)
    p.add_argument("--y-scale", type=int, default=1, help="P9: Ly = y_scale * 2pi (NY = 128 * y_scale)")
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-val", type=int, default=1)
    p.add_argument("--trajectory-time", type=float, default=40.0)
    p.add_argument("--seed-base", type=int, default=500)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    tmp = out.parent / (out.stem + "_parts")
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = KolmogorovConfig(alpha=args.alpha, Lx=2 * math.pi * args.x_scale, NX=128 * args.x_scale,
                           Ly=2 * math.pi * args.y_scale, NY=128 * args.y_scale, seed=args.seed_base)
    n_runs = args.n_train + args.n_val
    n_steps = int(round(args.trajectory_time / cfg.dt))

    for i in range(n_runs):
        part = tmp / f"traj{i}.npy"
        if part.exists():
            print(f"[p0] skip traj {i} (done)", flush=True)
            continue
        t0 = time.time()
        u0 = spinup_kolmogorov(cfg, np.random.default_rng(args.seed_base + i))
        traj = integrate_kolmogorov(u0, cfg, n_steps=n_steps).astype(np.float32)  # (T, NX, NY)
        np.save(part.with_suffix(".tmp.npy"), traj)
        os.replace(part.with_suffix(".tmp.npy"), part)
        print(f"[p0] traj {i}/{n_runs} done in {time.time() - t0:.0f}s, shape {traj.shape}", flush=True)

    trajs = np.stack([np.load(tmp / f"traj{i}.npy") for i in range(n_runs)])  # (N, T, NX, NY)
    train = trajs[: args.n_train].astype(np.float64)
    mean, std = float(train.mean()), float(train.std())
    normalized = ((trajs - mean) / std).astype(np.float32).reshape(n_runs, trajs.shape[1], -1)
    with h5py.File(out, "w") as f:
        f.create_dataset("trajectories", data=normalized, compression="gzip")
        _write_metadata_group_2d(f, cfg, {
            "n_train": args.n_train, "n_val": args.n_val, "trajectory_time": args.trajectory_time,
            "normalization_mean": mean, "normalization_std": std, "x_scale": args.x_scale, "y_scale": args.y_scale,
            "seed_base": args.seed_base,
        })
    print(f"[p0] wrote {out}  mean={mean:.4g} std={std:.4g}", flush=True)


if __name__ == "__main__":
    main()
