"""Animate the ground-truth latent state z_true(t) (a length-d_latent
vector per snapshot) as a GIF, one frame per timestep -- a companion to
`visualize_rollout.py`'s static latent Hovmoller heatmap (same
z-scoring-by-ground-truth's-own-mean/std convention, same
dataset/run-index/start/rollout-steps defaults), added 2026-09-02,
user-directed ("make a gif out of the ground truth data from the latent
states ... see the evolution of the latent state over time").

Also plots the physical-space KS field u(x, t) in a panel above the
latent panel, same time index driving both (added 2026-09-03,
user-directed: "can we have the same gif for the actual KS dynamics above
the latent state gif? so we can compare the trajectory of both systems").
Pass `--no-physical` to omit it and reproduce the original latent-only GIF.

Only encodes the ground truth trajectory (`ae.encode`, no propagator
rollout) -- this is a view into the DATA, not a model evaluation.
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
import torch

from ks_latent.models import load_autoencoder_checkpoint

FIGURES_DIR = Path("docs/figures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--run-index", type=int, default=0, help="Which trajectory in --dataset to use.")
    parser.add_argument("--start", type=int, default=0, help="Starting snapshot index within that trajectory.")
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--dt-snap", type=float, default=1.0, help="Physical time between snapshots.")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--style", choices=["bar", "line"], default="bar")
    parser.add_argument("--L", type=float, default=100.0, help="KS domain length (physical panel only).")
    parser.add_argument(
        "--no-physical", action="store_true",
        help="Omit the physical-space u(x,t) panel, reproducing the original latent-only GIF.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint, device=device)
    ae.eval()

    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][args.run_index], dtype=torch.float32)
    u_true = traj[args.start : args.start + args.rollout_steps + 1]
    with torch.no_grad():
        z_true = ae.encode(u_true).numpy()  # (T, d_latent)

    mu = z_true.mean(axis=0, keepdims=True)
    sd = z_true.std(axis=0, keepdims=True) + 1e-12
    zt = (z_true - mu) / sd  # same z-scoring convention as plot_latent_hovmoller

    T, d = zt.shape
    ylim = float(np.abs(zt).max()) * 1.05
    x = np.arange(d)
    u_np = u_true.numpy()  # (T, NX), same T/frame indexing as zt
    NX = u_np.shape[1]
    u_ylim = float(np.abs(u_np).max()) * 1.05
    x_phys = np.arange(NX) * (args.L / NX)

    show_physical = not args.no_physical
    if show_physical:
        fig, (ax_phys, ax) = plt.subplots(2, 1, figsize=(7, 7))
        fig.suptitle("Ground truth: KS field u(x,t) and latent state z(t)")
        ax_phys.set_xlabel("x")
        ax_phys.set_ylabel("u(x)")
        ax_phys.set_xlim(0.0, args.L)
        ax_phys.set_ylim(-u_ylim, u_ylim)
        ax_phys.axhline(0.0, color="black", linewidth=0.5)
        (phys_artist,) = ax_phys.plot(x_phys, u_np[0], color="darkgreen")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle("Ground-truth latent state z(t) (z-scored by its own mean/std)")
    ax.set_xlabel("latent index")
    ax.set_ylabel("z-scored value")
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0.0, color="black", linewidth=0.5)

    if args.style == "bar":
        colors = np.where(zt[0] >= 0, "firebrick", "steelblue")
        artist = ax.bar(x, zt[0], color=colors)
    else:
        (artist,) = ax.plot(x, zt[0], marker="o", markersize=3, color="steelblue")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        row = zt[frame]
        if args.style == "bar":
            for rect, h in zip(artist, row):
                rect.set_height(h)
                rect.set_color("firebrick" if h >= 0 else "steelblue")
        else:
            artist.set_ydata(row)
        time_text.set_text(f"t = {frame * args.dt_snap:.1f}")
        artists = (*artist, time_text) if args.style == "bar" else (artist, time_text)
        if show_physical:
            phys_artist.set_ydata(u_np[frame])
            artists = (*artists, phys_artist)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = FIGURES_DIR / f"latent_state_evolution{suffix}.gif"
    anim.save(out_path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out_path} ({T} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
