"""Standalone Stage-1+Stage-2 driver for `encoder_kind="local_field_2d"`
on Kolmogorov flow (`docs/ks_local_field_ltransfer.tex` "Staged next
steps" Step 2, gated on Step 1 -- 2D KS -- succeeding first, which it
did: Section 230, D_KY=71.9).

**Almost entirely a copy of `scripts/train_2dks_localfield_standalone.py`**
-- the whole point of Step 2's architecture reuse (2D KS and Kolmogorov
flow share the exact same periodic-2D-scalar-field state representation,
`KSAutoencoderLocalField2D`/`site_conv_2d`/`local_mlp_2d` care only about
`NX`/`NY`/`d_latent`, never about which PDE produced the field). The only
real differences: (1) `NX`/`NY` are read from the dataset's own metadata
rather than hardcoded, since Kolmogorov flow's natural domain (`Lx=Ly=
2*pi`, matched to the forcing wavenumber) has no reason to share 2D KS's
own grid convention a priori; (2) the tag/docstring naming.
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
        "--backbone", choices=["site_conv_2d", "local_mlp_2d", "local_mlp_2d_biotsavart",
                               "multiscale_local_mlp_2d"],
        default="local_mlp_2d",
        help="Propagator backbone. Default 'local_mlp_2d': 2D KS's own comparison "
        "(Section 228 vs. 230) found site_conv_2d collapsed to D_KY=0 while "
        "local_mlp_2d recovered genuine chaos, so local_mlp_2d is the well-precedented "
        "starting point here rather than repeating that same negative result. "
        "'local_mlp_2d_biotsavart' (D2 of docs/oct_1_exp.md) adds the exact, "
        "non-learned Biot-Savart/streamfunction velocity-recovery term as an extra "
        "propagator input -- typically used with --reuse-ae-ckpt, since it only "
        "changes the propagator's input, not anything about the encoder/decoder.",
    )
    parser.add_argument(
        "--bs-vel-scale", default="1.0",
        help="local_mlp_2d_biotsavart only: velocity normalization. 'auto' = std of the exact "
        "Biot-Savart velocity of the training set's anchor field (C9); 1.0 reproduces D2.",
    )
    parser.add_argument("--bs-zero-init-embed", action="store_true",
                        help="local_mlp_2d_biotsavart only: zero-initialize the velocity embedding (C9).")
    parser.add_argument(
        "--reuse-ae-ckpt", default=None,
        help="Path to an existing Stage-1 AE checkpoint to reuse as-is (frozen, not "
        "retrained) instead of training a new one. Skips Stage 1 training entirely; "
        "only the propagator (Stage 2) is trained, with a FRESH (not warm-started) "
        "propagator of whatever --backbone is given. The reused checkpoint's own "
        "n_sites_x/n_sites_y/local_channels must match --n-sites-x/--n-sites-y/"
        "--local-channels (checked, raises otherwise).",
    )
    parser.add_argument(
        "--attn-window", type=int, default=3,
        help="Propagator mixing radius, in SITES. Default 3, matching 2D KS's own "
        "Section 230 (the value that worked there) -- provisional here too, reused "
        "rather than re-derived from a Kolmogorov-flow-specific light-cone measurement.",
    )
    parser.add_argument("--n-sites-x", type=int, default=8)
    parser.add_argument("--n-sites-y", type=int, default=8)
    parser.add_argument("--local-channels", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip Stage 1 if its final checkpoint already exists (preemptable T1 runs).",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    device = get_device("mps")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "logs").mkdir(exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        NX = int(f["metadata"].attrs["NX"])
        NY = int(f["metadata"].attrs["NY"])
        Lx = float(f["metadata"].attrs["Lx"])
        Ly = float(f["metadata"].attrs["Ly"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]

    ae_ckpt_path = ARTIFACTS_DIR / f"stage1_ae_kolmogorov_full_{args.tag}.pt"
    aux_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_kolmogorov_full_{args.tag}.pt"

    bs_vel_scale = 1.0
    if args.bs_vel_scale == "auto":
        from ks_latent.models.propagator import _biot_savart_velocity
        from ks_latent.training.crop import site_anchor
        anc = site_anchor(train_traj.reshape(train_traj.shape[0], train_traj.shape[1], NX, NY)[:, ::5],
                          NX // args.n_sites_x, NY // args.n_sites_y)
        bs_vel_scale = float(_biot_savart_velocity(anc.reshape(-1, *anc.shape[-2:]).double(), Lx, Ly).std())
        print(f"[bs] auto velocity scale = {bs_vel_scale:.4f}", flush=True)
    elif args.bs_vel_scale != "1.0":
        bs_vel_scale = float(args.bs_vel_scale)

    def _build_aux_cfg(ae_cfg) -> AuxPropagatorConfig:
        d_latent = ae_cfg.d_latent
        if args.backbone == "site_conv_2d":
            return AuxPropagatorConfig(
                d_latent=d_latent, mode="markovian", backbone="site_conv_2d", attn_window=args.attn_window,
                site_conv2d_n_sites_x=ae_cfg.n_sites_x, site_conv2d_n_sites_y=ae_cfg.n_sites_y,
                site_conv2d_local_channels=ae_cfg.local_channels,
                site_conv2d_hidden=32, site_conv2d_n_layers=3,
            )
        elif args.backbone == "local_mlp_2d_biotsavart":
            return AuxPropagatorConfig(
                d_latent=d_latent, mode="markovian", backbone="local_mlp_2d_biotsavart",
                attn_window=args.attn_window,
                local_mlp2d_n_sites_x=ae_cfg.n_sites_x, local_mlp2d_n_sites_y=ae_cfg.n_sites_y,
                local_mlp2d_local_channels=ae_cfg.local_channels,
                local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
                biot_savart_Lx=Lx, biot_savart_Ly=Ly,
                biot_savart_vel_scale=bs_vel_scale, biot_savart_zero_init_embed=args.bs_zero_init_embed,
            )
        elif args.backbone == "multiscale_local_mlp_2d":
            return AuxPropagatorConfig(
                d_latent=d_latent, mode="markovian", backbone="multiscale_local_mlp_2d",
                attn_window=args.attn_window,
                local_mlp2d_n_sites_x=ae_cfg.n_sites_x, local_mlp2d_n_sites_y=ae_cfg.n_sites_y,
                local_mlp2d_local_channels=ae_cfg.local_channels,
                local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
                multiscale_pool_x=2, multiscale_pool_y=2,
                multiscale_regional_token_d_model=32, multiscale_regional_n_layers=2,
                multiscale_regional_window=2,
            )
        else:  # local_mlp_2d
            return AuxPropagatorConfig(
                d_latent=d_latent, mode="markovian", backbone="local_mlp_2d", attn_window=args.attn_window,
                local_mlp2d_n_sites_x=ae_cfg.n_sites_x, local_mlp2d_n_sites_y=ae_cfg.n_sites_y,
                local_mlp2d_local_channels=ae_cfg.local_channels,
                local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
            )

    if args.reuse_ae_ckpt:
        # D2 (docs/oct_1_exp.md): the encoder/decoder are untouched by this
        # backbone change (only the propagator's INPUT changes), so reuse an
        # existing, already-trained Stage-1 AE verbatim -- Stage 1 (which
        # exists to train ae+aux JOINTLY) is skipped entirely, not merely
        # shortened, since there is nothing left for it to do.
        print(f"=== [Kolmogorov] Stage 1 SKIPPED -- reusing AE checkpoint {args.reuse_ae_ckpt} ===", flush=True)
        ae, ae_cfg, ae_ckpt_raw = load_autoencoder_checkpoint(args.reuse_ae_ckpt, device=device)
        if (ae_cfg.n_sites_x, ae_cfg.n_sites_y, ae_cfg.local_channels) != (
            args.n_sites_x, args.n_sites_y, args.local_channels
        ):
            raise ValueError(
                f"--reuse-ae-ckpt's own sizing (n_sites_x={ae_cfg.n_sites_x}, "
                f"n_sites_y={ae_cfg.n_sites_y}, local_channels={ae_cfg.local_channels}) "
                f"does not match --n-sites-x/--n-sites-y/--local-channels "
                f"({args.n_sites_x}, {args.n_sites_y}, {args.local_channels})."
            )
        ae = ae.to(device)
        for p in ae.parameters():
            p.requires_grad_(False)
        aux_cfg = _build_aux_cfg(ae_cfg)
        aux = AuxPropagator(aux_cfg).to(device)
        # No Stage-1 training happened, so write the (untrained, freshly
        # initialized) aux straight to aux_ckpt_path so Stage 2's own loading
        # code below works completely unchanged -- "Stage 2" here is a cold
        # start for this backbone, not a warm start (there is no Stage-1
        # biotsavart checkpoint to warm-start FROM).
        torch.save({"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg}, aux_ckpt_path)
        ae_ckpt_path = Path(args.reuse_ae_ckpt)
    else:
        # --- Stage 1 ---
        ae_cfg = LocalFieldAutoencoderConfig2D(
            NX=NX, NY=NY, n_sites_x=args.n_sites_x, n_sites_y=args.n_sites_y,
            local_channels=args.local_channels, site_mix_radius=2, n_site_mix_layers=3, hidden=32,
        )
        ae = KSAutoencoderLocalField2D(ae_cfg).to(device)
        aux_cfg = _build_aux_cfg(ae_cfg)
        aux = AuxPropagator(aux_cfg).to(device)

        stage1_cfg = Stage1TrainingConfig(
            w_decorr=0.0, w_var=0.02, w_spatial=0.01, spatial_signed=True,
            w_var_floor=0.0, w_logdet=0.0035,
            epochs=args.stage1_epochs, seed=args.seed,
        )

        print(f"=== [Kolmogorov] Stage 1: encoder_kind=local_field_2d, backbone={args.backbone}, "
              f"attn_window={args.attn_window}, d_latent={ae_cfg.d_latent}, {args.stage1_epochs} epochs ===", flush=True)

    def _save_periodic_checkpoint(epoch: int) -> None:
        torch.save(
            {"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
             "encoder_kind": "local_field_2d", "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_ae_kolmogorov_full_{args.tag}_checkpoint.pt",
        )
        torch.save(
            {"prop_state_dict": aux.state_dict(), "prop_config": aux_cfg, "epoch": epoch},
            ARTIFACTS_DIR / f"stage1_prop_kolmogorov_full_{args.tag}_checkpoint.pt",
        )
        print(f"  [stage1] wrote mid-training checkpoint at epoch {epoch}", flush=True)

    if args.reuse_ae_ckpt:
        print(f"[reuse-ae-ckpt] Stage 1 skipped entirely; aux_ckpt_path written fresh above.", flush=True)
    elif args.resume and ae_ckpt_path.exists() and aux_ckpt_path.exists():
        print(f"[resume] Stage 1 already complete ({ae_ckpt_path}); skipping.", flush=True)
    else:
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

    # Shift augmentation disabled -- same reasoning as the 2D KS driver:
    # this codebase's shift augmentation rolls the FLAT NX*NY vector,
    # which scrambles rows for a genuinely 2D field. Not meaningful here
    # either.
    shifts = [0]
    train_seq = encode_dataset_with_shifts(ae2, train_traj, shifts, device)
    val_seq = encode_dataset_with_shifts(ae2, val_traj, shifts, device)

    prop, prop_cfg, _ = load_propagator_checkpoint(str(aux_ckpt_path), device=device)

    stage2_cfg = Stage2TrainingConfig(
        epochs=args.stage2_epochs, k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
        seed=args.seed,
    )

    print(f"=== [Kolmogorov] Stage 2: warm-started, k_max=12, {args.stage2_epochs} epochs ===", flush=True)

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
    print("=== Kolmogorov training complete ===")


if __name__ == "__main__":
    main()
