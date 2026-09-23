#!/usr/bin/env python
"""Visualize a free-running rollout's evolution over time, in both latent
and physical space (user-directed 2026-08-31): "a way to visualize the
evolution of latent variables over time. something like a 2d heatmap, with
position on the x axis and time on the y axis for a given rollout ...
also ... rollout performance and the whole model's trajectory."

Adapted from the reference project's plotting code
(`/Users/daltonjones/Documents/experiments/ks_latent/plots.py`'s
`plot_latent_rollout_spacetime`/`plot_rollout_spacetime`/
`plot_error_vs_lead_time`), retargeted to this codebase's model interfaces
(`ae.encode`/`decode`, `LatentPropagator.rollout`/`rollout_history`,
`load_autoencoder_checkpoint`/`load_propagator_checkpoint`).

Produces three figures per run:
  - `<tag>_latent_hovmoller.png`: latent index (x) vs. time (y) heatmap
    triptych -- ground truth (encode(u_true) every step), the model's own
    free-running rollout (encode once, then propagate), and their
    difference. Per-coordinate normalized by the TRUTH's own mean/std (see
    `plot_latent_hovmoller`'s docstring for why).
  - `<tag>_physical_hovmoller.png`: the same triptych in physical space
    (position vs. time) -- the classic KS space-time plot, comparing the
    true trajectory against the decoded model rollout.
  - `<tag>_error_growth.png`: relative RMSE (physical space) vs. lead time.

    python scripts/visualize_rollout.py --ae-checkpoint ... --prop-checkpoint ... --tag mytag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation

FIGURES_DIR = Path("docs/figures")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_free_rollout(ae, prop, u_traj: torch.Tensor, n_steps: int) -> dict:
    """`u_traj`: `(T, NX)`, a single real trajectory, `T >= n_steps + 2`
    (2 seed steps + `n_steps` predicted). Returns a dict with `z_true`/
    `z_pred`/`u_true`/`u_pred` (all `(n_steps+2, ...)` numpy arrays -- the
    first 2 rows are the shared seed, identical between true/pred by
    construction) and `diverged_at` (first non-finite step index, or
    `None`)."""
    mode = getattr(prop.cfg, "mode", "markovian")
    n_hist = getattr(prop.cfg, "n_history", 2) if mode == "history" else 2
    T_needed = n_steps + n_hist
    if u_traj.shape[0] < T_needed:
        raise ValueError(f"trajectory has {u_traj.shape[0]} steps, need >= {T_needed}")
    u_true = u_traj[:T_needed]

    with torch.no_grad():
        z_true = ae.encode(u_true)  # (T_needed, d)

    with torch.no_grad():
        if mode == "history":
            z_hist = z_true[:n_hist].unsqueeze(0)  # (1, n_hist, d)
            z_roll = prop.rollout_history(z_hist, n_steps)  # (1, n_steps, d)
        else:
            z_prev, z_curr = z_true[0:1], z_true[1:2]
            z_roll = prop.rollout(z_prev, z_curr, n_steps)  # (1, n_steps, d)
        z_roll = z_roll.squeeze(0)  # (n_steps, d)

    z_seed = z_true[:n_hist]
    z_pred_full = torch.cat([z_seed, z_roll], dim=0)  # (T_needed, d)

    finite = torch.isfinite(z_pred_full).all(dim=1)
    diverged_at = None
    if not finite.all():
        diverged_at = int((~finite).nonzero()[0, 0].item())

    with torch.no_grad():
        z_pred_safe = torch.where(torch.isfinite(z_pred_full), z_pred_full, torch.zeros_like(z_pred_full))
        u_pred_full = ae.decode(z_pred_safe)
        if diverged_at is not None:
            u_pred_full[diverged_at:] = float("nan")

    return {
        "z_true": z_true.numpy(),
        "z_pred": z_pred_full.numpy(),
        "u_true": u_true.numpy(),
        "u_pred": u_pred_full.numpy(),
        "diverged_at": diverged_at,
        "n_seed": n_hist,
    }


def plot_latent_hovmoller(
    res: dict, dt: float, path: Path, title: str = "", permutation: np.ndarray | None = None,
    ae_cfg=None,
) -> Path:
    """`encoder_kind="spectral_field"` (`ae_cfg` carries `K`/`N_w`): x-axis
    is the intermediate field `w`'s own physical position, plotted via
    `decode_from_spectrum(z, K, N_w)` -- i.e. the irfft of `z` (added
    2026-09-08, user-directed: "when you plot the latent_hovmoller plots
    can you plot the irfft of the states z, instead of z itself"). Motivated
    directly by the Section 104 visualization finding: `z` itself is just
    `K` real + `K` imaginary rFFT coefficients, so a raw-`z` Hovmoller shows
    coefficient MAGNITUDES with no spatial meaning (a "hard-saturated
    high-index channel" there is just one Fourier coefficient blowing up,
    not visible spatial structure) -- decoding back to `w`-space turns this
    into a genuine space-time plot, directly comparable to
    `plot_physical_hovmoller`'s own `u`-space triptych, and lets spurious
    high-wavenumber content show up as actual fine-grained spatial
    streaking instead of an opaque per-coefficient number.

    Otherwise (no `K`/`N_w` on `ae_cfg`, e.g. non-spectral encoders): falls
    back to the original raw-`z` behavior, plotting latent index (x) vs.
    time (y).

    Ground truth (`encode(u_true)` every step, then decoded), the model's
    free-running rollout, and their difference. Each coordinate is z-scored
    by the TRUTH's own mean/std (not the rollout's) -- the physical field is
    statistically homogeneous so one shared colour scale works, but
    raw-`z`-fallback coordinates' scales routinely differ by an order of
    magnitude, and this keeps the difference panel readable as "error in
    units of that coordinate's own climatological spread" (same convention
    as the reference project's `plot_latent_rollout_spacetime`,
    `/Users/daltonjones/Documents/experiments/ks_latent/plots.py`). In the
    raw-`z` fallback, the latent index axis carries no inherent metric
    unless something (D3/D6/D7 structure, a `--latent-permutation`) put one
    there -- read the ROWS (a latent state at one time), not for
    travelling-wave shapes the way the physical plot invites, UNLESS
    `permutation` is given. `permutation` (added 2026-08-31, user-directed:
    "try the reordering by the D7 permutation") only applies in that
    raw-`z` fallback (a decoded `w`-space plot already has a physically
    meaningful position axis, nothing to reorder): a `(d,)` array of latent
    indices -- reindexes the COLUMNS (x-axis) of all three panels by it
    before plotting (a purely cosmetic display reorder, computed fresh from
    this same rollout's `z_true` via `--reorder-by d7`; does NOT change the
    model's actual computation the way `PermutedAutoencoder`/
    `--latent-permutation` does). X-axis tick labels show the ORIGINAL index
    at each reordered position so the reordering is still interpretable."""
    z_true, z_pred = res["z_true"], res["z_pred"]

    K = getattr(ae_cfg, "K", None)
    N_w = getattr(ae_cfg, "N_w", None)
    decoded_to_w = K is not None and N_w is not None
    if decoded_to_w:
        from ks_latent.models.spectral_field import decode_from_spectrum

        with np.errstate(invalid="ignore", over="ignore"):
            field_true = decode_from_spectrum(torch.from_numpy(z_true), K, N_w).numpy()
            z_pred_safe = np.where(np.isfinite(z_pred), z_pred, 0.0)
            field_pred = decode_from_spectrum(torch.from_numpy(z_pred_safe), K, N_w).numpy()
        field_pred = np.where(np.isfinite(z_pred).all(axis=-1, keepdims=True), field_pred, np.nan)
        permutation = None  # a decoded w-space axis already has a physical position meaning
    else:
        field_true, field_pred = z_true, z_pred

    mu = field_true.mean(axis=0, keepdims=True)
    sd = field_true.std(axis=0, keepdims=True) + 1e-12
    zt = (field_true - mu) / sd
    with np.errstate(invalid="ignore", over="ignore"):
        zp = (field_pred - mu) / sd
    zp = np.where(np.isfinite(zp), zp, np.nan)
    diff = zp - zt

    T, d = field_true.shape
    if permutation is not None:
        zt, zp, diff = zt[:, permutation], zp[:, permutation], diff[:, permutation]
    t = dt * np.arange(T)
    idx = np.arange(d)
    vmax = float(np.abs(zt).max())

    pred_title = "model rollout"
    if res["diverged_at"] is not None:
        pred_title += f" (diverged at step {res['diverged_at']})"

    truth_name = "ground truth  decode_from_spectrum(encode(u_true))" if decoded_to_w else "ground truth  encode(u_true)"
    x_label = "w-space position" if decoded_to_w else "latent index"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, field, name, cmap in (
        (axes[0], zt, truth_name, "RdBu_r"),
        (axes[1], zp, pred_title, "RdBu_r"),
        (axes[2], diff, "difference", "PuOr_r"),
    ):
        im = ax.pcolormesh(idx, t, field, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(x_label + (" (D7-reordered)" if permutation is not None else ""))
        if permutation is not None:
            n_ticks = min(d, 12)
            tick_pos = np.linspace(0, d - 1, n_ticks).round().astype(int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([str(permutation[i]) for i in tick_pos], fontsize=7)
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("time")
    default_title = (
        "Latent field w=irfft(z) rollout (z-scored by ground truth's own mean/std)"
        if decoded_to_w
        else "Latent-space rollout (z-scored by ground truth's own mean/std)"
    )
    fig.suptitle(title or default_title)
    return _save(fig, path)


def plot_physical_hovmoller(res: dict, L: float, dt: float, path: Path, title: str = "") -> Path:
    """The classic KS space-time (Hovmoller) triptych: ground truth,
    decoded model rollout, and their difference, position on x, time on y."""
    u_true, u_pred = res["u_true"], res["u_pred"]
    T, NX = u_true.shape
    x = np.arange(NX) * L / NX
    t = dt * np.arange(T)
    vmax = float(np.abs(u_true).max())
    diff = u_pred - u_true

    pred_title = "model rollout (decoded)"
    if res["diverged_at"] is not None:
        pred_title += f" (diverged at step {res['diverged_at']})"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, field, name, cmap in (
        (axes[0], u_true, "ground truth", "RdBu_r"),
        (axes[1], u_pred, pred_title, "RdBu_r"),
        (axes[2], diff, "difference", "PuOr_r"),
    ):
        im = ax.pcolormesh(x, t, field, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("time")
    fig.suptitle(title or "Physical-space rollout: decoded model prediction vs. true KS trajectory")
    return _save(fig, path)


def plot_error_growth(res: dict, dt: float, path: Path, title: str = "") -> Path:
    """Relative RMSE (physical space) vs. lead time -- the whole
    trajectory's rollout performance in one curve."""
    u_true, u_pred = res["u_true"], res["u_pred"]
    T = u_true.shape[0]
    t = dt * np.arange(T)
    with np.errstate(invalid="ignore"):
        num = np.linalg.norm(u_pred - u_true, axis=1)
    denom = np.linalg.norm(u_true, axis=1) + 1e-12
    rel_err = num / denom

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, rel_err, color="#1f77b4", lw=1.8)
    ax.axhline(np.sqrt(2), color="#999999", ls="--", lw=1)
    ax.annotate(
        "saturation ($\\sqrt{2}$)", xy=(0.02, np.sqrt(2)), xycoords=("axes fraction", "data"),
        va="bottom", fontsize=8, color="#999999",
    )
    if res["diverged_at"] is not None:
        ax.axvline(dt * res["diverged_at"], color="black", lw=1, ls=":")
    ax.set_xlabel("time")
    ax.set_ylabel(r"relative RMSE  $\|u_{pred}-u_{true}\| / \|u_{true}\|$")
    ax.set_title(title or "Free-running rollout error growth (physical space)")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument("--prop-checkpoint", default="artifacts/stage2_prop_patched_full.pt")
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--run-index", type=int, default=0, help="Which trajectory in --dataset to use.")
    parser.add_argument("--start", type=int, default=0, help="Starting snapshot index within that trajectory.")
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--dt-snap", type=float, default=1.0, help="Physical time between snapshots.")
    parser.add_argument("--L", type=float, default=100.0, help="KS domain length.")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--latent-permutation", type=str, default=None)
    parser.add_argument("--latent-permutation-key", type=str, default="d3_permutation")
    parser.add_argument(
        "--reorder-by", choices=["none", "d7"], default="none",
        help="Purely cosmetic reordering of the latent Hovmoller plot's x-axis (does NOT "
        "change the model's computation, unlike --latent-permutation). 'd7' computes the "
        "same_time_coupling_diagnostic Fiedler permutation fresh from this rollout's own "
        "ground-truth z_true and reorders the plot's columns by it -- user-directed "
        "2026-08-31: 'try the reordering by the D7 permutation.'",
    )
    parser.add_argument("--out-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    if args.latent_permutation:
        ae = PermutedAutoencoder(ae, load_latent_permutation(args.latent_permutation, args.latent_permutation_key))
    ae.eval()
    prop, prop_cfg, _ = load_propagator_checkpoint(args.prop_checkpoint, device="cpu")
    prop.eval()

    # n_seed matches run_free_rollout's own T_needed = n_steps + n_hist
    # (added 2026-09-03, found via a real crash on mode="history",
    # n_history=5: the previous hardcoded "+2" assumed every propagator
    # needs at most 2 seed states, which is only true for markovian/
    # two_step -- for n_history>2 the slice was short by a fixed amount
    # regardless of --rollout-steps, since both grew together.
    n_seed = getattr(prop.cfg, "n_history", 2) if getattr(prop.cfg, "mode", "markovian") == "history" else 2
    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][args.run_index], dtype=torch.float32)
    u_traj = traj[args.start : args.start + args.rollout_steps + n_seed]

    res = run_free_rollout(ae, prop, u_traj, args.rollout_steps)

    permutation = None
    if args.reorder_by == "d7":
        d7 = same_time_coupling_diagnostic(res["z_true"], n_null=500, seed=0)
        permutation = d7.permutation
        print(f"D7 reordering: bandedness={d7.bandedness_observed:.4f} p={d7.bandedness_p_value:.4f}")
        print(f"D7 permutation: {permutation.tolist()}")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.reorder_by != "none":
        suffix += f"_{args.reorder_by}reordered"
    out_dir = Path(args.out_dir)
    p1 = plot_latent_hovmoller(
        res, args.dt_snap, out_dir / f"latent_hovmoller{suffix}.png", permutation=permutation, ae_cfg=ae_cfg,
    )
    p2 = plot_physical_hovmoller(res, args.L, args.dt_snap, out_dir / f"physical_hovmoller{suffix}.png")
    p3 = plot_error_growth(res, args.dt_snap, out_dir / f"error_growth{suffix}.png")
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"wrote {p3}")
    if res["diverged_at"] is not None:
        print(f"NOTE: rollout diverged at step {res['diverged_at']} "
              f"(t={args.dt_snap * res['diverged_at']:.1f})")
    else:
        final_err = float(
            np.linalg.norm(res["u_pred"][-1] - res["u_true"][-1])
            / (np.linalg.norm(res["u_true"][-1]) + 1e-12)
        )
        print(f"final relative RMSE at t={args.dt_snap * (args.rollout_steps + res['n_seed'] - 1):.1f}: {final_err:.4f}")


if __name__ == "__main__":
    main()
