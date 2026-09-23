"""Rayleigh-Benard dataset generation. Schema mirrors `ks_latent.solver.
dataset`/`lorenz96_dataset` where it can (an HDF5 file with a normalized
`trajectories` dataset, a raw `trajectories_raw` dataset, and a
`metadata` group carrying `n_train`/`n_val`/`dt_snap`/normalization
stats/the full producing config/`git_sha`/`config_hash`), but differs in
two ways forced by this system's own shape:

1. **Two-channel 2D field, not a 1D scalar.** `trajectories` has shape
   `(n_runs, T, 2, Nz, Nx)` -- channel 0 is `theta` (temperature
   perturbation), channel 1 is `omega` (vorticity), the system's two
   independent dynamical fields (`u`, `w`, `psi` are all algebraically
   derivable from `omega` alone via the streamfunction relation, so
   storing them too would be redundant -- pass `include_velocity=True`
   to also store `u`/`w` as convenience/plotting data in a separate
   `velocity` dataset, never required for training).

   **Existing 1D-only training pipeline compatibility** (brief §2,
   `scripts/train_stage1_patched.py --nx`): `trajectories.reshape(
   n_runs, T, -1)` gives a flat `(n_runs, T, 2*Nz*Nx)` vector directly
   usable with `--nx 2*Nz*Nx` and zero further code changes -- a real,
   immediate integration point, though a crude one (it throws away the
   field's 2D spatial structure and periodic-x/bounded-z distinction
   entirely; a genuinely 2D-spatial encoder is future work, see
   `docs/RESULTS.md`'s "candidate next system" entry).

2. **True streaming I/O** (user-directed: "Do not store all trajectories
   in RAM. Stream and save ... chunk-by-chunk"). Three passes, each
   holding at most ONE trajectory's data in memory at a time:
   (a) integrate each trajectory, write its raw physical fields directly
   into `trajectories_raw` as it completes, accumulating Welford
   running mean/variance (per channel, TRAINING split only) as it goes
   -- never needs a second read of the raw data just to get statistics;
   (b) finalize mean/std from the accumulated running stats;
   (c) re-read each trajectory's raw chunk back from disk one at a time,
   normalize, write into `trajectories`. Peak Python-side memory is
   O(one trajectory), not O(n_runs), regardless of dataset size.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ks_latent.config import RayleighBenardConfig, config_hash
from ks_latent.solver.rayleigh_benard import (
    default_initial_condition,
    get_operators,
    integrate,
    to_physical_fields,
)
from ks_latent.utils.io import get_git_sha


def _write_metadata_group(f: h5py.File, cfg: RayleighBenardConfig, extra: dict) -> None:
    grp = f.create_group("metadata")
    for k, v in dataclasses.asdict(cfg).items():
        grp.attrs[k] = v
    grp.attrs["dt_snap"] = cfg.dt_snap
    grp.attrs["Lx"] = cfg.Lx
    grp.attrs["config_hash"] = config_hash(cfg)
    grp.attrs["git_sha"] = get_git_sha()
    for k, v in extra.items():
        grp.attrs[k] = v


class _WelfordAccumulator:
    """Streaming (single-pass) mean/variance per channel, Welford's
    algorithm -- never needs to hold more than one trajectory's data at
    a time. `update(x)` with `x` shape `(..., n_channels)`-broadcastable;
    here called once per trajectory with `x` shape `(T, 2, Nz, Nx)`."""

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.count = np.zeros(n_channels, dtype=np.int64)
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.m2 = np.zeros(n_channels, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        # x: (T, n_channels, Nz, Nx) -> flatten to (n_channels, n_samples)
        for c in range(self.n_channels):
            vals = x[:, c].reshape(-1).astype(np.float64)
            n_new = vals.shape[0]
            mean_new = vals.mean()
            m2_new = ((vals - mean_new) ** 2).sum()

            n_old = self.count[c]
            delta = mean_new - self.mean[c]
            n_total = n_old + n_new
            self.mean[c] += delta * n_new / n_total
            self.m2[c] += m2_new + delta**2 * n_old * n_new / n_total
            self.count[c] = n_total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        var = self.m2 / np.maximum(self.count, 1)
        std = np.sqrt(np.maximum(var, 1e-30))
        return self.mean, std


def generate_trajectory_dataset(
    cfg: RayleighBenardConfig,
    path: str | Path,
    *,
    n_train: int = 20,
    n_val: int = 5,
    trajectory_time: float,
    include_velocity: bool = False,
    show_progress: bool = True,
) -> Path:
    """`n_train + n_val` independent trajectories, each spun up
    separately for `cfg.spinup_time` (discarded) before recording
    `trajectory_time` physical time units of snapshots every
    `cfg.dt_snap`. Adjusting scale: `n_train`/`n_val` (trajectory count),
    `cfg.Nx`/`cfg.Nz` (resolution), `path` (output location) -- all
    plain function arguments/config fields, nothing hardcoded (ground
    rule 4). Raises whatever `ks_latent.solver.rayleigh_benard.integrate`
    raises (non-finite state, exceeded `max_abs_velocity`) with the
    trajectory index it failed on -- ground rule 2, no silent skipping of
    a diverged trajectory."""
    path = Path(path)
    ops = get_operators(cfg)
    n_runs = n_train + n_val
    n_snapshots = int(round(trajectory_time / cfg.dt_snap))

    with h5py.File(path, "w") as f:
        traj_raw = f.create_dataset(
            "trajectories_raw",
            shape=(n_runs, n_snapshots, 2, cfg.Nz, cfg.Nx),
            dtype="f8",
            chunks=(1, n_snapshots, 2, cfg.Nz, cfg.Nx),
            compression="gzip",
        )
        vel_ds = None
        if include_velocity:
            vel_ds = f.create_dataset(
                "velocity",
                shape=(n_runs, n_snapshots, 2, cfg.Nz, cfg.Nx),
                dtype="f8",
                chunks=(1, n_snapshots, 2, cfg.Nz, cfg.Nx),
                compression="gzip",
            )

        accum = _WelfordAccumulator(n_channels=2)

        iterator = range(n_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"generating {path.name}", unit="traj")

        for i in iterator:
            try:
                rng = np.random.default_rng(cfg.seed + i)
                omega_hat0, theta_hat0 = default_initial_condition(cfg, rng)
                spinup_result = integrate(omega_hat0, theta_hat0, cfg, cfg.spinup_time, n_snapshots=1)
                omega_hat_spun = spinup_result["omega_hat"][-1]
                theta_hat_spun = spinup_result["theta_hat"][-1]

                result = integrate(
                    omega_hat_spun, theta_hat_spun, cfg, trajectory_time, n_snapshots=n_snapshots
                )
            except RuntimeError as e:
                raise RuntimeError(f"trajectory {i}/{n_runs} failed during integration: {e}") from e

            # Decode every snapshot to physical space, one trajectory's
            # worth at a time (never accumulated across trajectories).
            theta_phys = np.empty((n_snapshots, cfg.Nz, cfg.Nx))
            omega_phys = np.empty((n_snapshots, cfg.Nz, cfg.Nx))
            u_phys = np.empty((n_snapshots, cfg.Nz, cfg.Nx)) if include_velocity else None
            w_phys = np.empty((n_snapshots, cfg.Nz, cfg.Nx)) if include_velocity else None
            for t_idx in range(n_snapshots):
                fields = to_physical_fields(result["omega_hat"][t_idx], result["theta_hat"][t_idx], ops)
                theta_phys[t_idx] = fields["theta"]
                omega_phys[t_idx] = fields["omega"]
                if include_velocity:
                    u_phys[t_idx] = fields["u"]
                    w_phys[t_idx] = fields["w"]

            traj_stack = np.stack([theta_phys, omega_phys], axis=1)  # (T, 2, Nz, Nx)
            traj_raw[i] = traj_stack
            if i < n_train:
                accum.update(traj_stack)
            if include_velocity:
                vel_ds[i] = np.stack([u_phys, w_phys], axis=1)

        mean, std = accum.finalize()

        traj_norm = f.create_dataset(
            "trajectories",
            shape=(n_runs, n_snapshots, 2, cfg.Nz, cfg.Nx),
            dtype="f8",
            chunks=(1, n_snapshots, 2, cfg.Nz, cfg.Nx),
            compression="gzip",
        )
        norm_iterator = range(n_runs)
        if show_progress:
            norm_iterator = tqdm(norm_iterator, desc="normalizing", unit="traj")
        for i in norm_iterator:
            raw_i = traj_raw[i]  # (T, 2, Nz, Nx), one trajectory at a time
            traj_norm[i] = (raw_i - mean[None, :, None, None]) / std[None, :, None, None]

        _write_metadata_group(
            f,
            cfg,
            {
                "n_train": n_train,
                "n_val": n_val,
                "trajectory_time": trajectory_time,
                "n_snapshots": n_snapshots,
                "channel_names": np.array([b"theta", b"omega"]),
                "normalization_mean": mean,
                "normalization_std": std,
                "include_velocity": include_velocity,
            },
        )

    return path
