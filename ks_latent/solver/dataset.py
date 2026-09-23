"""KS dataset generation (brief §3.3).

Two modes, not interchangeable:

- `TrajectoryDataset`: long correlated runs for training propagators/AEs.
- `AttractorPointDataset`: one independent point per run, for dimension and
  topology estimation, where temporal correlation between consecutive
  samples corrupts the estimate and thinning does not fix it (see
  `tests/unit/test_dataset.py::test_thinning_does_not_fix_correlation`).

Both are written to HDF5 with a `metadata` group carrying the full
`KSConfig`, `dt_snap`, normalization stats (`TrajectoryDataset` only), and
the producing git SHA.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np

from ks_latent.config import KSConfig, config_hash
from ks_latent.solver.ks import integrate, spinup
from ks_latent.utils.io import get_git_sha


def _write_metadata_group(f: h5py.File, cfg: KSConfig, extra: dict) -> None:
    grp = f.create_group("metadata")
    for k, v in dataclasses.asdict(cfg).items():
        grp.attrs[k] = v
    grp.attrs["dt_snap"] = cfg.dt_snap
    grp.attrs["config_hash"] = config_hash(cfg)
    grp.attrs["git_sha"] = get_git_sha()
    for k, v in extra.items():
        grp.attrs[k] = v


def generate_trajectory_dataset(
    cfg: KSConfig,
    path: str | Path,
    *,
    n_train: int = 50,
    n_val: int = 10,
    trajectory_time: float,
) -> Path:
    """`n_train + n_val` independent trajectories, each spun up separately.

    Normalization statistics (mean/std) are computed once on the training
    split only and stored in metadata; downstream code must apply them, not
    recompute normalization elsewhere (brief §3.3).
    """
    path = Path(path)
    n_runs = n_train + n_val
    n_steps = int(round(trajectory_time / cfg.dt))

    trajectories = np.empty((n_runs, n_steps // cfg.snapshot_every + 1, cfg.NX))
    for i in range(n_runs):
        rng = np.random.default_rng(cfg.seed + i)
        u0 = spinup(cfg, rng)
        trajectories[i] = integrate(u0, cfg, n_steps=n_steps)

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


def generate_attractor_point_dataset(
    cfg: KSConfig,
    path: str | Path,
    *,
    n_runs: int = 10_000,
    spinup_discard_snapshots: int = 100,
    post_spinup_time: float = 50.0,
) -> Path:
    """One randomly chosen on-attractor snapshot per independent run.

    Each of the `n_runs` runs gets its own spin-up (`cfg.spinup_time`), then
    `spinup_discard_snapshots` further snapshots are discarded before one
    snapshot is drawn at random -- this is the only sampling scheme in this
    codebase that produces points independent enough for dimension/topology
    estimation (brief §3.3; see `test_thinning_does_not_fix_correlation`).
    """
    path = Path(path)
    n_post_steps = spinup_discard_snapshots * cfg.snapshot_every + int(
        round(post_spinup_time / cfg.dt)
    )
    points = np.empty((n_runs, cfg.NX))
    for i in range(n_runs):
        rng = np.random.default_rng(cfg.seed + i)
        u0 = spinup(cfg, rng)
        traj = integrate(u0, cfg, n_steps=n_post_steps)
        idx = rng.integers(spinup_discard_snapshots, traj.shape[0])
        points[i] = traj[idx]

    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=points, compression="gzip")
        _write_metadata_group(
            f,
            cfg,
            {
                "n_runs": n_runs,
                "spinup_discard_snapshots": spinup_discard_snapshots,
                "post_spinup_time": post_spinup_time,
            },
        )
    return path


def sample_thinned_single_trajectory(
    cfg: KSConfig, n_points: int, *, thin_stride: int = 1
) -> np.ndarray:
    """A single long trajectory, subsampled every `thin_stride` snapshots.

    Exists to demonstrate the failure mode `AttractorPointDataset` avoids:
    even heavily thinned, consecutive points from one trajectory are locally
    correlated and corrupt dimension estimates. Not a real dataset generator
    -- do not use this for anything but that regression test.
    """
    rng = np.random.default_rng(cfg.seed)
    u0 = spinup(cfg, rng)
    n_steps = n_points * thin_stride * cfg.snapshot_every
    traj = integrate(u0, cfg, n_steps=n_steps, snapshot_every=cfg.snapshot_every * thin_stride)
    return traj[:n_points]
