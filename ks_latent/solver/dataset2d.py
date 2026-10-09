"""2D KS dataset generation (`docs/ks_local_field_ltransfer.tex` "Staged
next steps" Step 1), 2D analogue of `ks_latent.solver.dataset`'s
`generate_trajectory_dataset` -- same HDF5 layout/metadata convention,
same once-on-training-split normalization discipline (brief §3.3).

**Stored FLATTENED, `(n_runs, T, NX*NY)`, not `(n_runs, T, NX, NY)`** --
deliberate, matching `KSAutoencoderLocalField2D`'s own public
`encode`/`decode` interface (flat `(B, NX*NY)`, reshaped to `(NX,NY)`
only internally) and `ks_latent.training.loops.train_stage1`/
`train_stage2`'s own hard-coded flat-vector convention
(`n_runs, T, NX = trajectories.shape`) -- storing flattened means the
2D training driver can load this file with IDENTICAL code to the 1D/
multiscale case, no extra reshape-at-load-time logic needed anywhere.

**Generalized (2026-09-28, Step 2) to any 2D periodic-scalar-field
solver, not just 2D KS.** `generate_trajectory_dataset2d`'s `integrate_fn`/
`spinup_fn` parameters default to `ks2d.integrate2d`/`ks2d.spinup2d`
(preserving every existing caller's exact behavior unchanged), but accept
any `(u0, cfg, n_steps, ...) -> traj` / `(cfg, rng) -> u0` pair with the
same signature -- e.g. `ks_latent.solver.kolmogorov`'s
`integrate_kolmogorov`/`spinup_kolmogorov`, used by
`generate_trajectory_dataset_kolmogorov` below. Only `cfg.NX`/`NY`/`dt`/
`snapshot_every`/`seed` and the (solver-agnostic) HDF5 metadata writer are
required of `cfg`, not any KS-specific field.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from ks_latent.config import KSConfig2D, config_hash
from ks_latent.solver.ks2d import integrate2d, spinup2d
from ks_latent.utils.io import get_git_sha


def _write_metadata_group_2d(f: h5py.File, cfg, extra: dict) -> None:
    grp = f.create_group("metadata")
    for k, v in dataclasses.asdict(cfg).items():
        grp.attrs[k] = v
    grp.attrs["dt_snap"] = cfg.dt_snap
    grp.attrs["config_hash"] = config_hash(cfg)
    grp.attrs["git_sha"] = get_git_sha()
    for k, v in extra.items():
        grp.attrs[k] = v


def generate_trajectory_dataset2d(
    cfg: KSConfig2D,
    path: str | Path,
    *,
    n_train: int = 50,
    n_val: int = 10,
    trajectory_time: float,
    integrate_fn: Callable = integrate2d,
    spinup_fn: Callable = spinup2d,
) -> Path:
    """`n_train + n_val` independent trajectories, each spun up
    separately. Normalization statistics (mean/std) computed once on
    the training split only, stored in metadata -- same discipline as
    the 1D solver's own dataset generator. `integrate_fn`/`spinup_fn`
    default to 2D KS's own solver (see module docstring for reuse on a
    different 2D periodic-scalar-field system)."""
    path = Path(path)
    n_runs = n_train + n_val
    n_steps = int(round(trajectory_time / cfg.dt))
    n_snapshots = n_steps // cfg.snapshot_every + 1

    trajectories = np.empty((n_runs, n_snapshots, cfg.NX * cfg.NY))
    for i in range(n_runs):
        rng = np.random.default_rng(cfg.seed + i)
        u0 = spinup_fn(cfg, rng)
        traj = integrate_fn(u0, cfg, n_steps=n_steps)  # (n_snapshots, NX, NY)
        trajectories[i] = traj.reshape(n_snapshots, cfg.NX * cfg.NY)

    train = trajectories[:n_train]
    mean = train.mean()
    std = train.std()
    normalized = (trajectories - mean) / std

    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=normalized, compression="gzip")
        f.create_dataset("trajectories_raw", data=trajectories, compression="gzip")
        _write_metadata_group_2d(
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


def generate_trajectory_dataset_kolmogorov(
    cfg,
    path: str | Path,
    *,
    n_train: int = 50,
    n_val: int = 10,
    trajectory_time: float,
) -> Path:
    """Kolmogorov-flow dataset generator -- thin wrapper reusing
    `generate_trajectory_dataset2d` verbatim (same HDF5 layout,
    normalization discipline, and metadata writer) with `cfg:
    KolmogorovConfig` and `ks_latent.solver.kolmogorov`'s own
    `integrate_kolmogorov`/`spinup_kolmogorov` in place of 2D KS's."""
    from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov

    return generate_trajectory_dataset2d(
        cfg,
        path,
        n_train=n_train,
        n_val=n_val,
        trajectory_time=trajectory_time,
        integrate_fn=integrate_kolmogorov,
        spinup_fn=spinup_kolmogorov,
    )
