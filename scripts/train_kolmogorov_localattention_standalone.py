"""Standalone Stage-1+Stage-2 driver for `encoder_kind="local_attention_2d"`
on Kolmogorov flow (added 2026-09-29, user-directed: "please implement
the local attention variant you described above ... leave the propagator
the same"). Almost entirely a copy of `train_kolmogorov_localfield_
standalone.py` (Section 231's own isotropic conv-based driver) --
deliberately, so this is a single-variable-changed comparison against
that recipe: same dataset, same n_sites_x/n_sites_y/local_channels/
attn_window/training recipe, with ONLY the autoencoder swapped for the
windowed-local-attention variant. The PROPAGATOR stays `local_mlp_2d`
(the conv-based mixer that worked, per the user's explicit instruction
not to touch it again after the anisotropic propagator's collapse).
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
    LocalAttentionAutoencoderConfig2D,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.autoencoder_local_attention_2d import KSAutoencoderLocalAttention2D
from ks_latent.models.propagator import AuxPropagator
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
    parser.add_argument(
        "--attn-window", type=int, default=3,
        help="PROPAGATOR mixing radius, in SITES -- matches Section 231's own value. "
        "The propagator backbone/config is unchanged (local_mlp_2d); only the "
        "autoencoder is the new local-attention variant.",
    )
    parser.add_argument("--n-sites-x", type=int, default=8)
    parser.add_argument("--n-sites-y", type=int, default=8)
    parser.add_argument("--local-channels", type=int, default=3)
    parser.add_argument(
        "--encoder-window", type=int, default=2,
        help="ENCODER windowed-attention radius (in sites), matching Section 231's "
        "own site_mix_radius=2 for a like-for-like comparison.",
    )
    parser.add_argument("--n-site-mix-layers", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)

    device = get_device("mps")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "logs").mkdir(exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        NX = int(f["metadata"].attrs["NX"])
        NY = int(f["metadata"].attrs["NY"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]

    # --- Stage 1 ---
    ae_cfg = LocalAttentionAutoencoderConfig2D(
        NX=NX, NY=NY, n_sites_x=args.n_sites_x, n_sites_y=args.n_sites_y,
        local_channels=args.local_channels, window=args.encoder_window,
        n_site_mix_layers=args.n_site_mix_layers, d_model=args.d_model, n_heads=args.n_heads,
    )
    ae = KSAutoencoderLocalAttention2D(ae_cfg).to(device)
    d_latent = ae_cfg.d_latent
    aux_cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="local_mlp_2d", attn_window=args.attn_window,
        local_mlp2d_n_sites_x=ae_cfg.n_sites_x, local_mlp2d_n_sites_y=ae_cfg.n_sites_y,
        local_mlp2d_local_channels=ae_cfg.local_channels,
        local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
    )
    aux = AuxPropagator(aux_cfg).to(device)

    stage1_cfg = Stage1TrainingConfig(
        w_decorr=0.0, w_var=0.02, w_spatial=0.01, spatial_signed=True,
        w_var_floor=0.0, w_logdet=0.0035,
        epochs=args.stage1_epochs, seed=args.seed,
    )

    print(f"=== [Kolmogorov local-attention] Stage 1: encoder_kind=local_attention_2d, "
          f"backbone=local_mlp_2d (unchanged), encoder_window={args.encoder_window}, "
          f"propagator attn_window={args.attn_window}, d_latent={d_latent}, "
          f"{args.stage1_epochs} epochs ===", flush=True)

    ae_ckpt_path = ARTIFACTS_DIR / f"stage1_ae_kolmogorov_full_{args.tag}.pt"
    aux_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_kolmogorov_full_{args.tag}.pt"

    def _save_periodic_checkpoint(epoch: int) -> None:
        torch.save(
            {"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
             "encoder_kind": "local_attention_2d", "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_ae_kolmogorov_full_{args.tag}_checkpoint.pt",
        )
        torch.save(
            {"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_prop_kolmogorov_full_{args.tag}_checkpoint.pt",
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
         "encoder_kind": "local_attention_2d", "val_recon_final": result1.val_recon_final},
        ae_ckpt_path,
    )
    torch.save({"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg}, aux_ckpt_path)
    write_provenance(ae_ckpt_path, config=ae_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    write_provenance(aux_ckpt_path, config=aux_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    (ARTIFACTS_DIR / f"stage1_history_kolmogorov_full_{args.tag}.json").write_text(
        json.dumps(result1.train_history, indent=2)
    )
    print(f"val_recon_final = {result1.val_recon_final:.6f}")
    print(f"wrote {ae_ckpt_path}, {aux_ckpt_path}")

    # --- Stage 2 ---
    ae2, ae_cfg2, _ = load_autoencoder_checkpoint(str(ae_ckpt_path), device=device)
    ae2.eval()
    for p in ae2.parameters():
        p.requires_grad_(False)

    shifts = [0]
    train_seq = encode_dataset_with_shifts(ae2, train_traj, shifts, device)
    val_seq = encode_dataset_with_shifts(ae2, val_traj, shifts, device)

    prop, prop_cfg, _ = load_propagator_checkpoint(str(aux_ckpt_path), device=device)

    stage2_cfg = Stage2TrainingConfig(
        epochs=args.stage2_epochs, k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
        seed=args.seed,
    )

    print(f"=== [Kolmogorov local-attention] Stage 2: warm-started, k_max=12, "
          f"{args.stage2_epochs} epochs ===", flush=True)

    stage2_tag = f"{args.tag}_warmstart_k12_{args.stage2_epochs}ep"
    stage2_ckpt_path = ARTIFACTS_DIR / f"stage2_prop_kolmogorov_full_{stage2_tag}.pt"

    def _save_periodic_checkpoint2(epoch: int) -> None:
        torch.save(
            {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage2_prop_kolmogorov_full_{stage2_tag}_checkpoint.pt",
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
    (ARTIFACTS_DIR / f"stage2_history_kolmogorov_full_{stage2_tag}.json").write_text(
        json.dumps(result2.train_history, indent=2)
    )
    print(f"best_val_kmax_mse = {result2.best_val_kmax_mse:.6f}")
    print(f"wrote {stage2_ckpt_path}")
    print("=== Kolmogorov local-attention training complete ===")


if __name__ == "__main__":
    main()
