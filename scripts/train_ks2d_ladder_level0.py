"""R2 (`docs/ks_local_field_ltransfer.tex`, "A frozen, scale-incremental
hierarchy" -- Revision: build the ladder from scratch): train Level 0 of
the 2D KS domain-size ladder from scratch at the smallest chaotic rung
($L_0=16$, $NX=64$, $n_{\mathrm{sites}}=8$, patch length 2.0 physical
units -- fixed across the whole ladder so every level stays
zero-retraining-resizable to the next).

Deliberately almost identical to `scripts/train_2dks_localfield_standalone.py`
(same regularizer recipe, same `local_mlp_2d` backbone) -- the ladder's
own Level-0 rung is meant to be a direct, like-for-like miniature of the
existing Section 230 recipe, not a new architecture search. The only
real differences: NX/n_sites are read from the dataset (any ladder rung,
not just $L_0$) rather than hardcoded, and `--n-sites` is derived from
`NX // patch_size` (patch_size=8 grid points, fixed) instead of a
hardcoded 8.
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
    LocalFieldAutoencoderConfig2D,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")
PATCH_SIZE = 8  # grid points -- fixed across the whole ladder (patch length = 2.0 physical units at dx=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--attn-window", type=int, default=2,
                         help="Reduced from Section 230's own 3 (2026-09-29, user-directed): at L0's own tiny "
                         "domain, window=3 covered ~87%% of it (near-global, not genuinely local); a v1 attempt "
                         "at that setting under-captured the true attractor's own chaos (model D_KY=2.21 vs "
                         "true ~10.3) -- diagnosed directly via rollout+Lyapunov before this rerun, not assumed.")
    parser.add_argument("--n-sites", type=int, default=None,
                         help="Default: NX // PATCH_SIZE (8-grid-point patches). Explicit override lets "
                         "n_sites/local_channels be chosen jointly to hit a specific d_latent target.")
    parser.add_argument("--local-channels", type=int, default=2,
                         help="Reduced from Section 230's own 3 (2026-09-29, user-directed, alongside "
                         "d_latent=32=4*4*2, down from 192=8*8*3) -- v1's d_latent=192 was plausibly "
                         "oversized for a domain whose true attractor is only ~10-dimensional.")
    parser.add_argument("--n-layers", type=int, default=2,
                         help="local_mlp_2d's own stacked-mixer-layer count -- effective receptive field is "
                         "n_layers*attn_window sites, not attn_window alone (2026-09-29, user-directed: v1/v2 "
                         "both used the hardcoded default of 2, meaning their real RF was 2x wider than the "
                         "attn_window number alone suggested).")
    parser.add_argument("--amp", action="store_true")
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
    if args.n_sites is not None:
        n_sites = args.n_sites
        assert NX % n_sites == 0 and NY % n_sites == 0, f"NX/NY={NX}/{NY} not divisible by n_sites={n_sites}"
    else:
        assert NX % PATCH_SIZE == 0 and NY % PATCH_SIZE == 0, f"NX/NY={NX}/{NY} not divisible by patch_size={PATCH_SIZE}"
        n_sites = NX // PATCH_SIZE

    # --- Stage 1 ---
    ae_cfg = LocalFieldAutoencoderConfig2D(
        NX=NX, NY=NY, n_sites_x=n_sites, n_sites_y=n_sites, local_channels=args.local_channels,
        site_mix_radius=2, n_site_mix_layers=3, hidden=32,
    )
    ae = KSAutoencoderLocalField2D(ae_cfg).to(device)
    d_latent = ae_cfg.d_latent
    aux_cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="local_mlp_2d", attn_window=args.attn_window,
        local_mlp2d_n_sites_x=ae_cfg.n_sites_x, local_mlp2d_n_sites_y=ae_cfg.n_sites_y,
        local_mlp2d_local_channels=ae_cfg.local_channels,
        local_mlp2d_token_d_model=32, local_mlp2d_n_layers=args.n_layers,
    )
    aux = AuxPropagator(aux_cfg).to(device)

    stage1_cfg = Stage1TrainingConfig(
        w_decorr=0.0, w_var=0.02, w_spatial=0.01, spatial_signed=True,
        w_var_floor=0.0, w_logdet=0.0035,
        epochs=args.stage1_epochs, seed=args.seed,
    )

    print(f"=== [2D KS ladder] Stage 1: NX=NY={NX}, n_sites={n_sites}, attn_window={args.attn_window}, "
          f"n_layers={args.n_layers} (effective RF={args.n_layers * args.attn_window} sites), "
          f"d_latent={d_latent}, {args.stage1_epochs} epochs ===", flush=True)

    ae_ckpt_path = ARTIFACTS_DIR / f"stage1_ae_2dks_full_{args.tag}.pt"
    aux_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_2dks_full_{args.tag}.pt"

    def _save_periodic_checkpoint(epoch: int) -> None:
        torch.save(
            {"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
             "encoder_kind": "local_field_2d", "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_ae_2dks_full_{args.tag}_checkpoint.pt",
        )
        torch.save(
            {"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_prop_2dks_full_{args.tag}_checkpoint.pt",
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
         "encoder_kind": "local_field_2d", "val_recon_final": result1.val_recon_final},
        ae_ckpt_path,
    )
    torch.save({"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg}, aux_ckpt_path)
    write_provenance(ae_ckpt_path, config=ae_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    write_provenance(aux_ckpt_path, config=aux_cfg, seed=args.seed, device=device.type, wall_time_s=wall1)
    (ARTIFACTS_DIR / f"stage1_history_2dks_full_{args.tag}.json").write_text(
        json.dumps(result1.train_history, indent=2)
    )
    print(f"val_recon_final = {result1.val_recon_final:.6f}")
    print(f"wrote {ae_ckpt_path}, {aux_ckpt_path}")

    # --- Stage 2 ---
    ae2, ae_cfg2, _ = load_autoencoder_checkpoint(str(ae_ckpt_path), device=device)
    ae2.eval()
    for p in ae2.parameters():
        p.requires_grad_(False)

    shifts = [0]  # 2D flat-vector shift augmentation isn't meaningful, same reasoning as Section 230's own driver
    train_seq = encode_dataset_with_shifts(ae2, train_traj, shifts, device)
    val_seq = encode_dataset_with_shifts(ae2, val_traj, shifts, device)

    prop, prop_cfg, _ = load_propagator_checkpoint(str(aux_ckpt_path), device=device)

    stage2_cfg = Stage2TrainingConfig(
        epochs=args.stage2_epochs, k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
        seed=args.seed,
    )

    print(f"=== [2D KS ladder] Stage 2: warm-started, k_max=12, {args.stage2_epochs} epochs ===", flush=True)

    stage2_tag = f"{args.tag}_warmstart_k12_{args.stage2_epochs}ep"
    stage2_ckpt_path = ARTIFACTS_DIR / f"stage2_prop_2dks_full_{stage2_tag}.pt"

    def _save_periodic_checkpoint2(epoch: int) -> None:
        torch.save(
            {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage2_prop_2dks_full_{stage2_tag}_checkpoint.pt",
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
    (ARTIFACTS_DIR / f"stage2_history_2dks_full_{stage2_tag}.json").write_text(
        json.dumps(result2.train_history, indent=2)
    )
    print(f"best_val_kmax_mse = {result2.best_val_kmax_mse:.6f}")
    print(f"wrote {stage2_ckpt_path}")
    print("=== 2D KS ladder Level-0 training complete ===")


if __name__ == "__main__":
    main()
