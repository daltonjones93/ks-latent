"""Dataset generation tests (brief §3.3)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ks_latent.analysis.dimension import two_nn_dimension
from ks_latent.config import KSConfig
from ks_latent.solver.dataset import (
    generate_attractor_point_dataset,
    generate_trajectory_dataset,
    sample_thinned_single_trajectory,
)


def test_trajectory_dataset_shapes_and_metadata(tmp_path):
    cfg = KSConfig(L=22.0, NX=32, dt=0.05, snapshot_every=5, spinup_time=20.0, seed=0)
    path = generate_trajectory_dataset(
        cfg, tmp_path / "traj.h5", n_train=3, n_val=2, trajectory_time=2.0
    )
    with h5py.File(path, "r") as f:
        assert f["trajectories"].shape[0] == 5
        assert f["trajectories"].shape[2] == cfg.NX
        meta = f["metadata"].attrs
        assert meta["dt_snap"] == pytest.approx(cfg.dt_snap)
        assert meta["n_train"] == 3
        assert meta["n_val"] == 2
        assert "normalization_mean" in meta and "normalization_std" in meta
        train_norm = f["trajectories"][:3]
        assert abs(train_norm.mean()) < 1e-6
        assert train_norm.std() == pytest.approx(1.0, abs=1e-6)


def test_attractor_point_dataset_shapes_and_metadata(tmp_path):
    cfg = KSConfig(L=22.0, NX=32, dt=0.05, snapshot_every=5, spinup_time=20.0, seed=0)
    path = generate_attractor_point_dataset(
        cfg, tmp_path / "points.h5", n_runs=8, spinup_discard_snapshots=5, post_spinup_time=2.0
    )
    with h5py.File(path, "r") as f:
        assert f["points"].shape == (8, cfg.NX)
        assert f["metadata"].attrs["n_runs"] == 8


@pytest.mark.slow
def test_thinning_does_not_fix_correlation():
    """Two-NN dimension from a thinned single trajectory reads materially
    below the independent-runs estimate (brief §3.3). If this test ever
    passes trivially (dimensions agree), it went blind -- investigate rather
    than deleting it.

    Needs L=100 (true D_KY ~22) to show cleanly: at the smaller L=22 (true
    dim ~5) tried first, the effect was swamped by two-NN's own noise at
    n=800 points, and very short thinning strides (<~1 physical time unit)
    hit a *different* two-NN failure mode -- near-duplicate temporal
    neighbors give mu=r2/r1 close to 1, which is numerically unstable in the
    origin-forced regression and inflates the estimate rather than
    deflating it. Stride 10 (2.5 time units, well under the L=100 Lyapunov
    time of ~1/0.045~=22) lands past that instability and cleanly reproduces
    the brief's claimed direction: thinned reads materially *below*
    independent (measured ~7 vs ~18 here).
    """
    cfg = KSConfig(L=100.0, NX=256, dt=0.05, snapshot_every=5, spinup_time=200.0, seed=0)
    n_points = 800

    thinned = sample_thinned_single_trajectory(cfg, n_points, thin_stride=10)
    d_thinned = two_nn_dimension(thinned).dimension

    independent = np.empty((n_points, cfg.NX))
    for i in range(n_points):
        rng = np.random.default_rng(1000 + i)
        from ks_latent.solver.ks import integrate, spinup

        u0 = spinup(cfg, rng)
        traj = integrate(u0, cfg, n_steps=50)
        independent[i] = traj[rng.integers(10, traj.shape[0])]
    d_independent = two_nn_dimension(independent).dimension

    assert d_thinned < d_independent - 1.0, (
        f"thinning failure did not reproduce: thinned={d_thinned:.2f}, "
        f"independent={d_independent:.2f} (expected thinned materially lower)"
    )
