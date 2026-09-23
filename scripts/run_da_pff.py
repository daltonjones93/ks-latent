#!/usr/bin/env python
"""Phase 5 entry point: NAT-PFF data assimilation cycling experiment
(brief §7, §18 replication targets: free-run latent RMSE ~1.6/dim, DA
RMSE ~0.14/dim, spread ~0.11, calibration ~0.8, skill ~10x).

    python scripts/run_da_pff.py                  # full run, needs Stage-1/2 checkpoints
    python scripts/run_da_pff.py --profile smoke   # <60s, trains tiny models first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch

from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    KSConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.da.cycling import CycleConfig, run_da_experiment
from ks_latent.da.pff import PFFConfig
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.solver.dataset import generate_trajectory_dataset
from ks_latent.solver.ks import integrate, spinup
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def _sparse_obs_operator(stride: int):
    def op(u_phys: torch.Tensor) -> torch.Tensor:
        return u_phys[::stride]
    return op


def _smoke_setup():
    ks_cfg = KSConfig(L=22.0, NX=64, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=0)
    ae_cfg = AutoencoderConfig(
        NX=64, patch_size=8, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    device = torch.device("cpu")

    traj_path = ARTIFACTS_DIR / "datasets" / "smoke_da_traj.h5"
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
    train_stage2(prop, seq[:, :split], seq[:, split:],
                 Stage2TrainingConfig(epochs=2, batch_size=8, k_max=4, k_warmup_epochs=1), device)
    prop.eval()

    rng = torch.Generator(device="cpu")
    u0 = spinup(ks_cfg, __import__("numpy").random.default_rng(99))
    truth_np = integrate(u0, ks_cfg, n_steps=400)
    truth_traj = torch.tensor(truth_np, dtype=torch.float32)
    return ae, prop, truth_traj, ae_cfg.NX, ae_cfg.d_latent


def _full_setup(args):
    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    if args.latent_permutation:
        ae = PermutedAutoencoder(ae, load_latent_permutation(args.latent_permutation, args.latent_permutation_key))
    ae.eval()

    prop, prop_cfg, prop_ckpt = load_propagator_checkpoint(args.prop_checkpoint, device="cpu")
    prop.eval()

    snapshot_every = round(args.dt_snap / 0.05)
    ks_cfg = KSConfig(L=args.L, NX=ae_cfg.NX, dt=0.05, snapshot_every=snapshot_every, spinup_time=500.0, seed=args.seed)
    import numpy as np

    u0 = spinup(ks_cfg, np.random.default_rng(args.seed + 500))
    truth_np = integrate(u0, ks_cfg, n_steps=args.n_prop_steps * args.n_cycles * ks_cfg.snapshot_every)
    truth_traj = torch.tensor(truth_np, dtype=torch.float32)
    return ae, prop, truth_traj, ae_cfg.NX, ae_cfg.d_latent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ae-checkpoint", default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument("--prop-checkpoint", default="artifacts/stage2_prop_patched_full.pt")
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--n-prop-steps", type=int, default=1)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument("--obs-stride", type=int, default=8)
    parser.add_argument(
        "--dt-snap", type=float, default=1.0,
        help="Physical time between snapshots the propagator was trained on "
             "(must match the checkpoint's actual training dt_snap).",
    )
    parser.add_argument(
        "--L", type=float, default=100.0,
        help="Domain length for the ground-truth comparison trajectory this script "
        "generates itself via spinup()/integrate() (added 2026-09-07 -- this was "
        "hardcoded at 100.0 regardless of the checkpoint's own training L, silently "
        "wrong for any non-canonical-L experiment, e.g. docs/sine_transform_pde_plan.md's "
        "reduced-L spectral_pde runs). MUST match the checkpoint's actual training "
        "KSConfig.L or this comparison is meaningless.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix appended to the output JSON filename (added 2026-08-30, "
        "matching train_stage1/2_patched.py's --tag) so comparing runs on "
        "different checkpoints doesn't overwrite the previous run's output.",
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
        ae, prop, truth_traj, NX, d = _smoke_setup()
        n_prop_steps, n_cycles, n_ensemble, obs_stride = 2, 5, 16, 8
    else:
        ae, prop, truth_traj, NX, d = _full_setup(args)
        n_prop_steps, n_cycles, n_ensemble, obs_stride = args.n_prop_steps, args.n_cycles, args.n_ensemble, args.obs_stride

    encoder = lambda u: ae.encode(u)  # noqa: E731
    decoder = lambda z: ae.decode(z)  # noqa: E731
    obs_op = _sparse_obs_operator(obs_stride)

    with torch.no_grad():
        z0 = encoder(truth_traj[0:1]).squeeze(0)
    rng = torch.Generator().manual_seed(args.seed)
    z0_ensemble = z0.unsqueeze(0) + 0.1 * torch.randn(n_ensemble, d, generator=rng)
    z_minus1_ensemble = z0_ensemble.clone()

    cfg = CycleConfig(
        n_prop_steps=n_prop_steps, n_cycles=n_cycles, n_ensemble=n_ensemble,
        pff_config=PFFConfig(method="NAT", n_steps=100), seed=args.seed,
    )
    if getattr(prop, "mode", None) == "history":
        # mode="history" (added 2026-08-29, user-directed): bootstrap all
        # n_history slices the same way the 2-state case already bootstraps
        # z_minus1 (independently-jittered copies of z0) -- there is no real
        # "before truth_traj[0]" data to encode instead.
        n_hist = prop.cfg.n_history
        z_hist_ensemble = z0.unsqueeze(0).unsqueeze(0) + 0.1 * torch.randn(
            n_ensemble, n_hist, d, generator=rng
        )
        result = run_da_experiment(
            prop, decoder, encoder, obs_op, truth_traj, z0_ensemble, z_minus1_ensemble, cfg,
            z_hist_ensemble=z_hist_ensemble,
        )
    else:
        result = run_da_experiment(prop, decoder, encoder, obs_op, truth_traj, z0_ensemble, z_minus1_ensemble, cfg)

    rmse_da = float(result.rmse_da.mean())
    rmse_free = float(result.rmse_free.mean())
    spread = float(result.spread.mean())
    report = {
        "rmse_da": rmse_da,
        "rmse_free": rmse_free,
        "spread": spread,
        "calibration_spread_over_rmse": spread / rmse_da if rmse_da > 0 else float("nan"),
        "skill_free_over_da": rmse_free / rmse_da if rmse_da > 0 else float("nan"),
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = ARTIFACTS_DIR / f"da_pff_{args.profile}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
