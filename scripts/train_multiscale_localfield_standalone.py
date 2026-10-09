"""Standalone Stage-1+Stage-2 driver for `encoder_kind="multiscale_local_field"`
+ `backbone="multiscale_local_mlp"` (Section 227, added 2026-09-28,
user-directed: combine Section 224's local_field/local_mlp recipe with
the multiscale-latent idea in
`docs/spatially_structured_latent_DA_FMM_research_idea.md`).

Deliberately standalone rather than wired into
`scripts/train_stage1_patched.py`/`train_stage2_patched.py`'s own CLI
dispatch: those scripts back many already-validated, published results
this session, and their encoder/backbone dispatch is duplicated across
several code paths (profile smoke/full, warm-start, etc.) -- adding a
new encoder there safely would mean touching every one of those sites
correctly. This script instead calls `ks_latent.training.loops.
train_stage1`/`train_stage2` DIRECTLY as plain Python functions (already
duck-typed over the autoencoder/propagator classes they're given, and
their regularizer losses -- `spatial_coherence_loss` etc. -- operate on
the flat `z` vector alone, no shape assumptions), reusing the exact
same validated training loop/loss code with zero risk to the shared
CLI scripts. Checkpoints are written in the SAME `{"ae_state_dict",
"ae_config", "aux_config", "encoder_kind", "val_recon_final"}`/
`{"prop_state_dict", "prop_config"}` format `train_stage1_patched.py`/
`train_stage2_patched.py` use, so every downstream tool
(`load_autoencoder_checkpoint`, diagnostics, Lyapunov analysis, DA
scripts) works unchanged.

Recipe: Section 224's exact regularizer/schedule choices
(`w_decorr=0, w_var=0.02, w_spatial=0.01, spatial_signed=True,
w_var_floor=0, w_logdet=0.0035`, plain `k_pred=2`, Stage 1 200 epochs,
Stage 2 warm-started `k_max=12` over 300 epochs) -- the parts of 224
that are known to work, reused as literally as possible; the only new
things are the encoder/propagator architecture themselves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch

from ks_latent.config import (
    AuxPropagatorConfig,
    MultiscaleLocalFieldAutoencoderConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.autoencoder_multiscale_local_field import KSAutoencoderMultiscaleLocalField
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    device = get_device("mps")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "logs").mkdir(exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]

    # --- Stage 1 ---
    ae_cfg = MultiscaleLocalFieldAutoencoderConfig()
    ae = KSAutoencoderMultiscaleLocalField(ae_cfg).to(device)
    d_latent = ae_cfg.d_latent
    aux_cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="multiscale_local_mlp",
        multiscale_fine_n_sites=ae_cfg.fine_n_sites, multiscale_fine_channels=ae_cfg.fine_local_channels,
        multiscale_fine_window=3,
        multiscale_regional_n_sites=ae_cfg.regional_n_sites, multiscale_regional_channels=ae_cfg.regional_local_channels,
        multiscale_regional_window=1,
        multiscale_global_channels=ae_cfg.global_local_channels,
        multiscale_token_d_model=32, multiscale_n_layers=2,
        hidden=128, n_blocks=3,  # matches Section 224's own "full propagator" sizing convention
    )
    aux = AuxPropagator(aux_cfg).to(device)

    # `spatial_blocks`/reduced `w_spatial` (added 2026-09-28, user-directed:
    # "please only apply w_spatial per block, also reduce its influence"):
    # w_spatial was found, on inspection, to be applied to the FULL
    # concatenated [fine, regional, global] vector as one closed ring --
    # spurious at the two block-boundary indices (last fine site next to
    # first regional site is NOT a real physical adjacency). Fixed via
    # `blockwise_spatial_coherence_loss` (ks_latent/training/losses.py),
    # scoring each scale as its own ring; see that function's own tests
    # for a direct empirical demonstration of the boundary artifact this
    # avoids. Weight reduced from Section 224's own 0.01 to 0.003 (~1/3)
    # per the same instruction -- this is a NEW mechanism on a NEW
    # architecture (three separate rings instead of one), not yet
    # validated at the original weight.
    spatial_blocks = [
        ae_cfg.fine_n_sites * ae_cfg.fine_local_channels,
        ae_cfg.regional_n_sites * ae_cfg.regional_local_channels,
        ae_cfg.global_local_channels,
    ]
    stage1_cfg = Stage1TrainingConfig(
        w_decorr=0.0, w_var=0.02, w_spatial=0.003, spatial_signed=True, spatial_blocks=spatial_blocks,
        w_var_floor=0.0, w_logdet=0.0035,
        epochs=args.stage1_epochs, seed=args.seed,
    )

    print(f"=== [multiscale] Stage 1: encoder_kind=multiscale_local_field, "
          f"backbone=multiscale_local_mlp, d_latent={d_latent}, {args.stage1_epochs} epochs ===", flush=True)

    ae_ckpt_path = ARTIFACTS_DIR / f"stage1_ae_multiscale_full_{args.tag}.pt"
    aux_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_multiscale_full_{args.tag}.pt"

    def _save_periodic_checkpoint(epoch: int) -> None:
        torch.save(
            {"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
             "encoder_kind": "multiscale_local_field", "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_ae_multiscale_full_{args.tag}_checkpoint.pt",
        )
        torch.save(
            {"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_prop_multiscale_full_{args.tag}_checkpoint.pt",
        )
        print(f"  [stage1] wrote mid-training checkpoint at epoch {epoch}", flush=True)

    t0 = time.time()
    result1 = train_stage1(
        ae, aux, train_traj, val_traj, stage1_cfg, device, verbose=True, log_every=10,
        checkpoint_every=args.checkpoint_every,
        on_epoch_end=_save_periodic_checkpoint if args.checkpoint_every > 0 else None,
        amp=args.amp,
    )
    wall1 = time.time() - t0

    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
         "encoder_kind": "multiscale_local_field", "val_recon_final": result1.val_recon_final},
        ae_ckpt_path,
    )
    torch.save({"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg}, aux_ckpt_path)
    write_provenance(ae_ckpt_path, config=ae_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    write_provenance(aux_ckpt_path, config=aux_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    (ARTIFACTS_DIR / f"stage1_history_multiscale_full_{args.tag}.json").write_text(
        json.dumps(result1.train_history, indent=2)
    )
    print(f"val_recon_final = {result1.val_recon_final:.6f}")
    print(f"wrote {ae_ckpt_path}, {aux_ckpt_path}")

    # --- Stage 2 ---
    ae2, ae_cfg2, _ = load_autoencoder_checkpoint(str(ae_ckpt_path), device=device)
    ae2.eval()
    for p in ae2.parameters():
        p.requires_grad_(False)

    shifts = [0, ae_cfg2.NX // 4, ae_cfg2.NX // 2, 3 * ae_cfg2.NX // 4]
    train_seq = encode_dataset_with_shifts(ae2, train_traj, shifts, device)
    val_seq = encode_dataset_with_shifts(ae2, val_traj, shifts, device)

    prop, prop_cfg, _ = load_propagator_checkpoint(str(aux_ckpt_path), device=device)

    stage2_cfg = Stage2TrainingConfig(
        epochs=args.stage2_epochs, k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
        seed=args.seed,
    )

    print(f"=== [multiscale] Stage 2: warm-started, k_max=12, {args.stage2_epochs} epochs ===", flush=True)

    stage2_tag = f"{args.tag}_warmstart_k12_{args.stage2_epochs}ep"
    stage2_ckpt_path = ARTIFACTS_DIR / f"stage2_prop_multiscale_full_{stage2_tag}.pt"

    def _save_periodic_checkpoint2(epoch: int) -> None:
        torch.save(
            {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage2_prop_multiscale_full_{stage2_tag}_checkpoint.pt",
        )
        print(f"  [stage2] wrote mid-training checkpoint at epoch {epoch}", flush=True)

    t0 = time.time()
    result2 = train_stage2(
        prop, train_seq, val_seq, stage2_cfg, device, verbose=True, log_every=10,
        checkpoint_every=args.checkpoint_every,
        on_epoch_end=_save_periodic_checkpoint2 if args.checkpoint_every > 0 else None,
        amp=args.amp,
    )
    wall2 = time.time() - t0

    torch.save(
        {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg,
         "best_val_kmax_mse": result2.best_val_kmax_mse},
        stage2_ckpt_path,
    )
    write_provenance(stage2_ckpt_path, config=prop_cfg, seed=args.seed, device=device.type, wall_time_s=wall2)
    (ARTIFACTS_DIR / f"stage2_history_multiscale_full_{stage2_tag}.json").write_text(
        json.dumps(result2.train_history, indent=2)
    )
    print(f"best_val_kmax_mse = {result2.best_val_kmax_mse:.6f}")
    print(f"wrote {stage2_ckpt_path}")
    print("=== Section 227 (multiscale) complete ===")


if __name__ == "__main__":
    main()
