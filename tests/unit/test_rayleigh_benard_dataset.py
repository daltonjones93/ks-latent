"""Validation tests for the Rayleigh-Benard streaming dataset writer."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ks_latent.config import RayleighBenardConfig
from ks_latent.solver.rayleigh_benard_dataset import generate_trajectory_dataset


@pytest.fixture()
def small_dataset(tmp_path):
    cfg = RayleighBenardConfig(Nx=16, Nz=12, spinup_time=0.2, snapshot_dt=0.05)
    path = generate_trajectory_dataset(
        cfg, tmp_path / "rbc.h5", n_train=3, n_val=2, trajectory_time=0.15,
        include_velocity=True, show_progress=False,
    )
    return path, cfg


def test_shapes_and_finite(small_dataset):
    path, cfg = small_dataset
    with h5py.File(path, "r") as f:
        expected = (5, 3, 2, cfg.Nz, cfg.Nx)  # n_runs=3+2, n_snapshots=round(0.15/0.05)=3
        assert f["trajectories"].shape == expected
        assert f["trajectories_raw"].shape == expected
        assert f["velocity"].shape == expected
        assert np.isfinite(f["trajectories"][:]).all()
        assert np.isfinite(f["trajectories_raw"][:]).all()
        assert np.isfinite(f["velocity"][:]).all()


def test_normalization_matches_stored_stats(small_dataset):
    """trajectories == (trajectories_raw - mean) / std, using metadata's
    own stored per-channel mean/std -- the exact relationship downstream
    consumers rely on."""
    path, cfg = small_dataset
    with h5py.File(path, "r") as f:
        raw = f["trajectories_raw"][:]
        norm = f["trajectories"][:]
        mean = f["metadata"].attrs["normalization_mean"]
        std = f["metadata"].attrs["normalization_std"]
        reconstructed = (raw - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        assert np.allclose(norm, reconstructed, atol=1e-10)


def test_normalization_stats_computed_on_train_split_only(small_dataset):
    """mean/std must reflect ONLY the first n_train trajectories, not
    n_train+n_val -- same convention as KS/L96's own dataset writers."""
    path, cfg = small_dataset
    with h5py.File(path, "r") as f:
        raw = f["trajectories_raw"][:]
        n_train = f["metadata"].attrs["n_train"]
        mean_stored = f["metadata"].attrs["normalization_mean"]
        std_stored = f["metadata"].attrs["normalization_std"]

    train_raw = raw[:n_train]
    for c in range(2):
        expected_mean = train_raw[:, :, c].mean()
        expected_std = train_raw[:, :, c].std()
        assert mean_stored[c] == pytest.approx(expected_mean, abs=1e-8)
        assert std_stored[c] == pytest.approx(expected_std, abs=1e-8)


def test_metadata_provenance_fields(small_dataset):
    path, cfg = small_dataset
    with h5py.File(path, "r") as f:
        meta = f["metadata"].attrs
        assert meta["n_train"] == 3
        assert meta["n_val"] == 2
        assert meta["Ra"] == cfg.Ra
        assert meta["Pr"] == cfg.Pr
        assert meta["dt_snap"] == cfg.dt_snap
        assert "config_hash" in meta
        assert "git_sha" in meta
        assert list(meta["channel_names"]) == [b"theta", b"omega"]


def test_flatten_compatible_with_1d_training_pipeline(small_dataset):
    """trajectories.reshape(n_runs, T, -1) must give a clean flat vector
    of length 2*Nz*Nx per snapshot -- the documented compatibility path
    with train_stage1_patched.py's --nx override."""
    path, cfg = small_dataset
    with h5py.File(path, "r") as f:
        traj = f["trajectories"][:]
        flat = traj.reshape(traj.shape[0], traj.shape[1], -1)
        assert flat.shape[-1] == 2 * cfg.Nz * cfg.Nx
        # round-trip: reshape back must recover the original field exactly
        recovered = flat.reshape(traj.shape)
        assert np.array_equal(recovered, traj)


def test_raises_on_divergent_trajectory(tmp_path):
    """A config guaranteed to blow up (Ra far beyond what dt_max/cfl_target
    can stabilize at cfl_check_every left too coarse) must raise, not
    silently write garbage -- ground rule 2."""
    cfg = RayleighBenardConfig(
        Nx=8, Nz=8, Ra=3.0e4, Pr=0.7, spinup_time=0.05, snapshot_dt=0.02,
        cfl_check_every=1_000_000, dt0=5e-3, dt_max=5e-3,
    )
    with pytest.raises(RuntimeError, match="failed during integration"):
        generate_trajectory_dataset(
            cfg, tmp_path / "should_fail.h5", n_train=1, n_val=0,
            trajectory_time=0.3, show_progress=False,
        )
