#!/usr/bin/env python
"""Visualize a Rayleigh-Benard dataset's fluid dynamics: 2D convection
patterns (temperature + vorticity), their evolution over time, and the
saturation history that confirms the flow reached a bounded, statistically
steady turbulent state (not a diverging or trivially-decayed one).

Produces, per trajectory index (`--traj-idx`, default 0):
  - `<tag>_snapshots.png`: a grid of (theta, omega) panel pairs at several
    times spanning the trajectory -- the classic "look at the convection
    cells" figure.
  - `<tag>_animation.gif`: the same two fields animated over the full
    recorded trajectory.
  - `<tag>_saturation_history.png`: max|theta|, max|omega| vs. time across
    ALL trajectories in the dataset -- confirms boundedness/saturation
    (see `RayleighBenardConfig`'s docstring: a healthy run's fields settle
    into a bounded oscillating range, not zero and not diverging).

    python scripts/visualize_rayleigh_benard.py --dataset artifacts/datasets/rayleigh_benard_ra3e4_pr0.7.h5 --tag rbc_ra3e4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("docs/figures")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_snapshots(theta: np.ndarray, omega: np.ndarray, t: np.ndarray, Lx: float, tag: str, n_panels: int = 6):
    """theta, omega: (T, Nz, Nx). Grid of n_panels evenly-spaced-in-time
    snapshots, theta on top row, omega on bottom row, shared color scale
    per field (so panels are directly comparable across time)."""
    T = theta.shape[0]
    idxs = np.linspace(0, T - 1, n_panels).astype(int)
    theta_vmax = np.abs(theta).max()
    omega_vmax = np.abs(omega).max()

    fig, axes = plt.subplots(2, n_panels, figsize=(2.6 * n_panels, 5.2), constrained_layout=True)
    for col, idx in enumerate(idxs):
        im0 = axes[0, col].imshow(
            theta[idx], origin="lower", extent=[0, Lx, 0, 1], aspect="auto",
            cmap="RdBu_r", vmin=-theta_vmax, vmax=theta_vmax,
        )
        axes[0, col].set_title(f"t={t[idx]:.2f}", fontsize=10)
        im1 = axes[1, col].imshow(
            omega[idx], origin="lower", extent=[0, Lx, 0, 1], aspect="auto",
            cmap="PuOr_r", vmin=-omega_vmax, vmax=omega_vmax,
        )
        if col == 0:
            axes[0, col].set_ylabel("theta\n(z)")
            axes[1, col].set_ylabel("omega\n(z)")
        axes[1, col].set_xlabel("x")
    fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label="theta (temperature perturbation)")
    fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label="omega (vorticity)")
    fig.suptitle(f"Rayleigh-Benard convection snapshots -- {tag}")
    return _save(fig, FIGURES_DIR / f"{tag}_snapshots.png")


def make_animation(theta: np.ndarray, omega: np.ndarray, t: np.ndarray, Lx: float, tag: str, fps: int = 10):
    theta_vmax = np.abs(theta).max()
    omega_vmax = np.abs(omega).max()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    im0 = ax0.imshow(theta[0], origin="lower", extent=[0, Lx, 0, 1], aspect="auto",
                      cmap="RdBu_r", vmin=-theta_vmax, vmax=theta_vmax)
    im1 = ax1.imshow(omega[0], origin="lower", extent=[0, Lx, 0, 1], aspect="auto",
                      cmap="PuOr_r", vmin=-omega_vmax, vmax=omega_vmax)
    ax0.set_title("theta")
    ax1.set_title("omega")
    ax0.set_xlabel("x")
    ax1.set_xlabel("x")
    ax0.set_ylabel("z")
    title = fig.suptitle(f"t={t[0]:.2f}")

    def update(frame):
        im0.set_data(theta[frame])
        im1.set_data(omega[frame])
        title.set_text(f"t={t[frame]:.2f}")
        return im0, im1, title

    anim = animation.FuncAnimation(fig, update, frames=theta.shape[0], blit=False)
    path = FIGURES_DIR / f"{tag}_animation.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_saturation_history(trajectories_raw: np.ndarray, t: np.ndarray, tag: str, n_traj_show: int = 8):
    """trajectories_raw: (n_runs, T, 2, Nz, Nx), channel 0=theta,
    1=omega. Plots max|theta|, max|omega| vs. time for up to
    n_traj_show trajectories, overlaid -- a healthy run shows each curve
    rising from the small initial perturbation and then settling into a
    bounded oscillating band (see RayleighBenardConfig's docstring)."""
    n_show = min(n_traj_show, trajectories_raw.shape[0])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    for i in range(n_show):
        max_theta = np.abs(trajectories_raw[i, :, 0]).max(axis=(-1, -2))
        max_omega = np.abs(trajectories_raw[i, :, 1]).max(axis=(-1, -2))
        ax0.plot(t, max_theta, alpha=0.7, lw=1)
        ax1.plot(t, max_omega, alpha=0.7, lw=1)
    ax0.set_xlabel("t")
    ax0.set_ylabel("max|theta|")
    ax0.set_title("Temperature perturbation amplitude")
    ax1.set_xlabel("t")
    ax1.set_ylabel("max|omega|")
    ax1.set_title("Vorticity amplitude")
    fig.suptitle(f"Saturation history ({n_show} trajectories) -- {tag}")
    fig.tight_layout()
    return _save(fig, FIGURES_DIR / f"{tag}_saturation_history.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--traj-idx", type=int, default=0, help="Which trajectory to plot snapshots/animation for.")
    parser.add_argument("--no-animation", action="store_true", help="Skip the (slower) GIF animation.")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        raw = f["trajectories_raw"][:]  # (n_runs, T, 2, Nz, Nx)
        Lx = float(f["metadata"].attrs["Lx"])
        dt_snap = float(f["metadata"].attrs["dt_snap"])
        spinup_time = float(f["metadata"].attrs["spinup_time"])

    n_runs, T = raw.shape[0], raw.shape[1]
    t = spinup_time + np.arange(1, T + 1) * dt_snap

    theta = raw[args.traj_idx, :, 0]
    omega = raw[args.traj_idx, :, 1]

    plot_snapshots(theta, omega, t, Lx, args.tag)
    if not args.no_animation:
        make_animation(theta, omega, t, Lx, args.tag)
    plot_saturation_history(raw, t, args.tag)


if __name__ == "__main__":
    main()
