"""Standalone Stage-1+Stage-2 driver for `encoder_kind="local_field_2d"`
+ `backbone="site_conv_2d"` on 2D KS (`docs/ks_local_field_ltransfer.tex`
"Staged next steps" Step 1, added 2026-09-28). Same standalone-driver
choice as `scripts/train_multiscale_localfield_standalone.py` (calls
`ks_latent.training.loops.train_stage1`/`train_stage2` directly, not
routed through `train_stage1_patched.py`'s own CLI dispatch -- see that
script's own docstring for why) and the same Section 224 recipe reused
as literally as possible: `w_decorr=0, w_var=0.02, w_spatial=0.01,
spatial_signed=True, w_var_floor=0, w_logdet=0.0035`, plain `k_pred=2`,
Stage 1 200 epochs, Stage 2 warm-started `k_max=12` over 300 epochs.

Architecture sizing (deliberately conservative for this first attempt,
NOT matching `LocalFieldAutoencoderConfig2D`'s own default site density,
which mirrors Section 224's 1D site-density ratio directly and would
give d_latent=768, a much larger first experiment than warranted before
anything 2D has been validated at all): `n_sites_x=n_sites_y=8,
local_channels=3` -> `d_latent=192`, `site_conv_2d` radius (`attn_window`)
`=2` matching the encoder's own `site_mix_radius` default.
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
        "--backbone", choices=["site_conv_2d", "local_mlp_2d"], default="site_conv_2d",
        help="Propagator backbone (added 2026-09-28 for Section 229's site_conv_2d-vs-"
        "local_mlp_2d comparison). Default 'site_conv_2d' preserves Section 228's exact behavior.",
    )
    parser.add_argument(
        "--attn-window", type=int, default=2,
        help="Propagator mixing radius, in SITES (added 2026-09-28 for Section 230: "
        "attn_window=2 gave local_mlp_2d a 4-site/~22-physical-unit receptive field, "
        "provisional (reused from the encoder's own site_mix_radius default, not derived "
        "from a measured 2D light-cone bound); widening to 3 tests receptive field width "
        "as a hypothesis for Section 229's own collapse, per user direction).",
    )
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
    ae_cfg = LocalFieldAutoencoderConfig2D(
        NX=128, NY=128, n_sites_x=8, n_sites_y=8, local_channels=3,
        site_mix_radius=2, n_site_mix_layers=3, hidden=32,
    )
    ae = KSAutoencoderLocalField2D(ae_cfg).to(device)
    d_latent = ae_cfg.d_latent
    if args.backbone == "site_conv_2d":
        aux_cfg = AuxPropagatorConfig(
            d_latent=d_latent, mode="markovian", backbone="site_conv_2d", attn_window=args.attn_window,
            site_conv2d_n_sites_x=ae_cfg.n_sites_x, site_conv2d_n_sites_y=ae_cfg.n_sites_y,
            site_conv2d_local_channels=ae_cfg.local_channels,
            site_conv2d_hidden=32, site_conv2d_n_layers=3,
        )
    else:  # local_mlp_2d -- the actual 2D analogue of Section 224's own propagator
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

    print(f"=== [2D KS] Stage 1: encoder_kind=local_field_2d, backbone={args.backbone}, "
          f"attn_window={args.attn_window}, d_latent={d_latent}, {args.stage1_epochs} epochs ===", flush=True)

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

    NXY = ae_cfg2.NX * ae_cfg2.NY
    # Shift augmentation: in units of the FLAT NX*NY index this codebase's
    # training loop always shifts in (`torch.roll` on the flat physical
    # vector) -- for a 2D field this is emphatically NOT a physically
    # meaningful spatial shift (rolling a flattened 2D array by an
    # arbitrary flat offset scrambles rows), so shift augmentation is
    # DISABLED here (`shifts=[0]` only) rather than silently applying a
    # meaningless transform. A real 2D-aware shift augmentation (roll in
    # x and/or y before flattening) is a real follow-up, not implemented
    # in this first pass.
    shifts = [0]
    train_seq = encode_dataset_with_shifts(ae2, train_traj, shifts, device)
    val_seq = encode_dataset_with_shifts(ae2, val_traj, shifts, device)

    prop, prop_cfg, _ = load_propagator_checkpoint(str(aux_ckpt_path), device=device)

    stage2_cfg = Stage2TrainingConfig(
        epochs=args.stage2_epochs, k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
        seed=args.seed,
    )

    print(f"=== [2D KS] Stage 2: warm-started, k_max=12, {args.stage2_epochs} epochs ===", flush=True)

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
    print("=== 2D KS training complete ===")


if __name__ == "__main__":
    main()
