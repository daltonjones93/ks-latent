#!/usr/bin/env python
"""Phase 4 entry point: run the analysis suite (dimension, Lyapunov,
topology) on a trained Stage-1/Stage-2 checkpoint pair (brief §6).

    python scripts/run_analysis_suite.py                  # full run, needs both checkpoints
    python scripts/run_analysis_suite.py --profile smoke   # <60s, trains tiny models first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.analysis.dimension import correlation_dimension, diffusion_maps, two_nn_dimension
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.analysis.topology import dtm_persistence, lifetimes, max_lifetime, rips_persistence
from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    KSConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.solver.dataset import generate_attractor_point_dataset, generate_trajectory_dataset
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def _smoke_setup():
    ks_cfg = KSConfig(L=22.0, NX=64, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=0)
    ae_cfg = AutoencoderConfig(
        NX=64, patch_size=8, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    device = torch.device("cpu")

    traj_path = ARTIFACTS_DIR / "datasets" / "smoke_analysis_traj.h5"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    generate_trajectory_dataset(ks_cfg, traj_path, n_train=4, n_val=2, trajectory_time=15.0)
    with h5py.File(traj_path, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    train_stage1(ae, aux, trajectories[:n_train], trajectories[n_train:],
                 Stage1TrainingConfig(epochs=2, batch_size=8), device)
    ae.eval()

    seq = encode_dataset_with_shifts(ae, trajectories, [0, ae_cfg.NX // 2], device)
    T = seq.shape[1]
    split = int(T * 0.7)
    prop = LatentPropagator(PropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1))
    train_stage2(
        prop, seq[:, :split], seq[:, split:],
        Stage2TrainingConfig(epochs=2, batch_size=8, k_max=4, k_warmup_epochs=1), device,
    )

    points_path = ARTIFACTS_DIR / "datasets" / "smoke_analysis_points.h5"
    generate_attractor_point_dataset(ks_cfg, points_path, n_runs=200, spinup_discard_snapshots=2, post_spinup_time=1.0)
    with h5py.File(points_path, "r") as f:
        points_phys = torch.tensor(f["points"][:], dtype=torch.float32)

    return ae, prop, points_phys, seq


def _full_setup(args):
    ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint(args.ae_checkpoint)
    if args.latent_permutation:
        ae = PermutedAutoencoder(ae, load_latent_permutation(args.latent_permutation, args.latent_permutation_key))
    ae.eval()

    prop, prop_cfg, prop_ckpt = load_propagator_checkpoint(args.prop_checkpoint, device="cpu")
    prop.eval()

    points_path = Path(args.points_dataset)
    with h5py.File(points_path, "r") as f:
        points_phys = torch.tensor(f["points"][:], dtype=torch.float32)

    traj_path = Path(args.dataset)
    with h5py.File(traj_path, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    seq = encode_dataset_with_shifts(ae, trajectories, [0], torch.device("cpu"))

    return ae, prop, points_phys, seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ae-checkpoint", default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument("--prop-checkpoint", default="artifacts/stage2_prop_patched_full.pt")
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--points-dataset", default="artifacts/datasets/attractor_points.h5")
    parser.add_argument(
        "--dt-snap", type=float, default=1.0,
        help="Physical time between snapshots the propagator was trained on "
             "(brief §3.1: load-bearing for converting exponents to physical "
             "time units -- must match the checkpoint's actual training dt_snap).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix appended to the output JSON filename (added 2026-08-30, "
        "matching train_stage1/2_patched.py's --tag), e.g. 'ensemble5' -> "
        "analysis_suite_full_ensemble5.json. Without this, comparing runs "
        "on different checkpoints silently overwrites the previous run's "
        "output file (caught 2026-08-30 comparing docs/"
        "PHASE2_ARCHITECTURE_EXPERIMENTS.md's experiments back to back).",
    )
    parser.add_argument(
        "--latent-permutation", type=str, default=None,
        help="See run_diagnostics.py's --latent-permutation help text and "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 20.",
    )
    parser.add_argument(
        "--latent-permutation-key", type=str, default="d3_permutation",
        help="See run_diagnostics.py's --latent-permutation-key help text.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.profile == "smoke":
        ae, prop, points_phys, seq = _smoke_setup()
    else:
        for p in [args.ae_checkpoint, args.prop_checkpoint]:
            if not Path(p).exists():
                raise FileNotFoundError(f"{p} not found; run Phase 3 training first.")
        ae, prop, points_phys, seq = _full_setup(args)

    with torch.no_grad():
        Z = ae.encode(points_phys).numpy()

    report: dict = {}
    report["two_nn_dimension"] = two_nn_dimension(Z).dimension
    report["correlation_dimension"] = correlation_dimension(Z).dimension

    diff = diffusion_maps(Z, n_components=6)
    report["diffusion_map_eigenvalues"] = diff.eigenvalues.tolist()
    diffusion_embedding = diff.embedding[:, :6]

    for name, cloud in [("raw", Z), ("diffusion_embedding", diffusion_embedding)]:
        rips = rips_persistence(cloud, homology_dimensions=(0, 1, 2))
        report[f"rips_{name}_max_lifetime_H1"] = max_lifetime(rips.diagrams[1])
        report[f"rips_{name}_max_lifetime_H2"] = max_lifetime(rips.diagrams[2])
        for mass in [0.02, 0.05, 0.1]:
            dtm = dtm_persistence(cloud, mass=mass, homology_dimensions=(0, 1, 2))
            report[f"dtm_{name}_mass{mass}_max_lifetime_H1"] = max_lifetime(dtm.diagrams[1])
            report[f"dtm_{name}_mass{mass}_max_lifetime_H2"] = max_lifetime(dtm.diagrams[2])

    d = prop.cfg.d_latent
    if getattr(prop, "mode", None) == "history":
        # mode="history" (added 2026-08-29, user-directed): "single_state"/
        # "two_step" don't apply (there is no well-defined way to fit an
        # arbitrary-length history into either), so this is the only mode.
        n_hist = prop.cfg.n_history
        initial_hist = seq[0, 0:n_hist].reshape(-1).numpy()
        # Full n_hist*d directions (added 2026-08-30, user-directed follow-up
        # -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 12): history
        # mode has no single_state-style fallback that always uses the full
        # dimension, so capping this at 20 (as it used to be) can hide a real
        # result entirely rather than just needing a "two_step"-style cross-
        # check -- caught exactly this way on the mlp+history run, where
        # n_directions=20 found nothing but the full n_hist*d=132 resolved
        # cleanly to D_KY=22.14, the best result of the whole investigation.
        n_dirs = n_hist * d
        try:
            result = lyapunov_spectrum_latent_propagator(
                prop, initial_hist, mode="history", n_directions=n_dirs,
                n_steps=200, qr_every=1, dt_snap=args.dt_snap, warmup_steps=40, seed=args.seed,
            )
        except ValueError as e:
            # See the "two_step" comment below for why this bracketing search
            # can fail without indicating a script bug.
            report["lyapunov_history_error"] = str(e)
            result = None
        if result is not None:
            report["lyapunov_history_D_KY"] = result.kaplan_yorke_dimension
            report["lyapunov_history_n_positive"] = result.n_positive
            report["lyapunov_history_lambda1"] = float(result.exponents[0])
    else:
        initial_pair = seq[0, 0:2].reshape(-1).numpy()
        for mode in ["single_state", "two_step"]:
            n_dirs = d if mode == "single_state" else min(2 * d, 20)
            try:
                result = lyapunov_spectrum_latent_propagator(
                    prop, initial_pair, mode=mode, n_directions=n_dirs,
                    n_steps=200, qr_every=1, dt_snap=args.dt_snap, warmup_steps=40, seed=args.seed,
                )
            except ValueError as e:
                # Caught 2026-08-30 (FNO+ViT propagator): "two_step" embeds a
                # markovian map's true d-dim spectrum plus d numerically-near-
                # zero "trivial shift-register" directions (see
                # _make_two_step_step_fn's docstring) capped at n_directions=20
                # -- if enough of those near-zero directions land
                # noise-positive, D_KY's bracketing search can need more than
                # 20 of the 2d directions even though "single_state" (which
                # doesn't have this redundancy and always uses the full
                # n_directions=d) succeeds. Record the failure instead of
                # losing every other (expensive) result in this report to one
                # sub-mode's exception.
                report[f"lyapunov_{mode}_error"] = str(e)
                continue
            report[f"lyapunov_{mode}_D_KY"] = result.kaplan_yorke_dimension
            report[f"lyapunov_{mode}_n_positive"] = result.n_positive
            report[f"lyapunov_{mode}_lambda1"] = float(result.exponents[0])

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = ARTIFACTS_DIR / f"analysis_suite_{args.profile}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
