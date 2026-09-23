"""Lorenz-96 dataset generation -- exact schema mirror of
`ks_latent.solver.dataset` (KS's own dataset writer), so every downstream
consumer (`scripts/train_stage1_patched.py`'s `--dataset` loader,
`ks_latent.analysis.lyapunov`, the Gate 3/4 diagnostic scripts) reads a
Lorenz-96 HDF5 file with zero code changes: both write an `(n_traj, T, N)`
`trajectories` dataset (normalized) plus `trajectories_raw`, and a
`metadata` group carrying `n_train`, `dt_snap`, and the full producing
config. Added 2026-09-22, see `ks_latent.solver.lorenz96`'s module
docstring for the full motivation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np

from ks_latent.config import Lorenz96Config, config_hash
from ks_latent.solver.lorenz96 import integrate, spinup
from ks_latent.utils.io import get_git_sha


def _write_metadata_group(f: h5py.File, cfg: Lorenz96Config, extra: dict) -> None:
    grp = f.create_group("metadata")
    for k, v in dataclasses.asdict(cfg).items():
        grp.attrs[k] = v
    grp.attrs["dt_snap"] = cfg.dt_snap
    grp.attrs["config_hash"] = config_hash(cfg)
    grp.attrs["git_sha"] = get_git_sha()
    for k, v in extra.items():
        grp.attrs[k] = v


def generate_trajectory_dataset(
    cfg: Lorenz96Config,
    path: str | Path,
    *,
    n_train: int = 50,
    n_val: int = 10,
    trajectory_time: float,
) -> Path:
    """`n_train + n_val` independent trajectories, each spun up separately
    -- exact mirror of `ks_latent.solver.dataset.generate_trajectory_
    dataset`'s own contract and HDF5 layout. Normalization statistics
    (mean/std) computed once on the training split only, stored in
    metadata; downstream code must apply them, not recompute normalization
    elsewhere (same rule as KS's own dataset, brief §3.3)."""
    path = Path(path)
    n_runs = n_train + n_val
    n_steps = int(round(trajectory_time / cfg.dt))

    trajectories = np.empty((n_runs, n_steps // cfg.snapshot_every + 1, cfg.N))
    for i in range(n_runs):
        rng = np.random.default_rng(cfg.seed + i)
        x0 = spinup(cfg, rng)
        trajectories[i] = integrate(x0, cfg, n_steps=n_steps)

    train = trajectories[:n_train]
    mean = train.mean()
    std = train.std()
    normalized = (trajectories - mean) / std

    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=normalized, compression="gzip")
        f.create_dataset("trajectories_raw", data=trajectories, compression="gzip")
        _write_metadata_group(
            f,
            cfg,
            {
                "n_train": n_train,
                "n_val": n_val,
                "trajectory_time": trajectory_time,
                "normalization_mean": mean,
                "normalization_std": std,
            },
        )
    return path
