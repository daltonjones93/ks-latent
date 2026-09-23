#!/usr/bin/env python
"""Visualizes a distilled/jointly-trained `spectral_pde_raw` pde_head's own
FREE-RUNNING rollout as a SMOOTH function of its internal ring coordinate,
by re-expanding each step's raw latent `z(t)` through the pde_head's own
truncated Fourier representation onto a much finer grid -- added 2026-09-10,
user-directed: "I would love to see a hovmoller plot and gif derived from
the evolution of the latent state under the pde. we want to plot this by
propagating the fourier basis forward using the pde, so we get a smooth
function evolution."

Mechanism: `backbone="spectral_pde_raw"` (`_SpectralPDERawDeltaBody`)
already treats z's own `d_latent` indices as a spatial coordinate on a
periodic ring (`ks_latent.models.spectral_field.encode_to_spectrum(z, K)`
-> a truncated rFFT spectrum -> evolved -> `decode_from_spectrum` back down
to `d_latent` points) -- see that class's own docstring. This script reuses
the SAME encode_to_spectrum/decode_from_spectrum pair the pde_head already
computes internally, but decodes each step's z(t) to a much FINER grid
(`--fine-resolution`, default 512, vs. the raw `d_latent`, e.g. 96) --
an exact spectral interpolation (zero-padding the same K truncated modes
before `irfft`), giving a smooth, continuous-looking curve at native
resolution instead of a blocky d_latent-wide step function -- the same
truncated Fourier series, just evaluated more densely.

Free-running (not compared against the propagator): rolls the pde_head's
own `.rollout` forward from a real initial condition and stops at
`--rollout-steps`, matching whatever stable window was found separately
(this pde_head's own standalone rollout diverges past a certain horizon --
check `visualize_rollout.py`'s own error-growth curve first to pick a safe
`--rollout-steps`, this script does not re-check divergence itself).
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

from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import decode_from_spectrum, encode_to_spectrum

FIGURES_DIR = Path("docs/figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--pdehead-checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=55)
    parser.add_argument("--fine-resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    pdehead, pdehead_cfg, _ = load_propagator_checkpoint(args.pdehead_checkpoint)
    ae.eval()
    pdehead.eval()
    K = pdehead_cfg.spectral_K
    d_latent = pdehead_cfg.d_latent
    L = pdehead_cfg.spectral_L
    print(f"pde_head: d_latent={d_latent}  spectral_K={K}  spectral_L={L}  "
          f"field_kind={pdehead_cfg.spectral_field_kind!r}")

    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][args.run_index], dtype=torch.float32)
    u0 = traj[args.start].unsqueeze(0)  # (1, NX)
    with torch.no_grad():
        z0 = ae.encode(u0)  # (1, d_latent)
        z_roll = pdehead.rollout(z0, z0, args.rollout_steps)  # (1, k, d_latent)
    z_all = torch.cat([z0.unsqueeze(1), z_roll], dim=1).squeeze(0)  # (k+1, d_latent)

    if not torch.isfinite(z_all).all():
        first_bad = int((~torch.isfinite(z_all).all(dim=-1)).float().argmax().item())
        print(f"WARNING: non-finite values starting at step {first_bad} -- "
              f"consider a smaller --rollout-steps (this pde_head is only a "
              f"short-horizon-accurate approximation, not a stable free-running system).")
        z_all = z_all[:first_bad]

    with torch.no_grad():
        z_hat = encode_to_spectrum(z_all, K)  # (T, 2K) -- pde_head's OWN truncated spectrum
        w_smooth = decode_from_spectrum(z_hat, K, args.fine_resolution)  # (T, fine_resolution)
    w_smooth_np = w_smooth.numpy()
    T = w_smooth_np.shape[0]
    print(f"rolled out {T} stable steps; smooth field shape {w_smooth_np.shape}")

    suffix = f"_{args.tag}" if args.tag else ""

    # --- Hovmoller (static PNG) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    x_fine = np.linspace(0, L, args.fine_resolution, endpoint=False)
    vmax = np.abs(w_smooth_np).max()
    im = ax.pcolormesh(
        x_fine, np.arange(T), w_smooth_np, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.set_xlabel("pde_head's own ring coordinate (d_latent-index units)")
    ax.set_ylabel("time (pde steps)")
    ax.set_title("pde_head free-running rollout, smoothed via its own truncated Fourier basis")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    png_path = FIGURES_DIR / f"pde_smooth_hovmoller{suffix}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")

    # --- GIF (line plot animated over time) ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    (line,) = ax2.plot(x_fine, w_smooth_np[0])
    ax2.set_xlim(0, L)
    ax2.set_ylim(-vmax * 1.05, vmax * 1.05)
    ax2.set_xlabel("pde_head's own ring coordinate (d_latent-index units)")
    ax2.set_ylabel("smoothed field value")
    title = ax2.set_title(f"pde_head free-running rollout -- step 0/{T - 1}")

    def _update(frame):
        line.set_ydata(w_smooth_np[frame])
        title.set_text(f"pde_head free-running rollout -- step {frame}/{T - 1}")
        return line, title

    anim = animation.FuncAnimation(fig2, _update, frames=T, blit=False)
    gif_path = FIGURES_DIR / f"pde_smooth_evolution{suffix}.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig2)
    print(f"wrote {gif_path} ({T} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
