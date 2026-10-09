"""Train a local-field autoencoder + local_mlp_2d propagator on x-crops of a
wide Kolmogorov domain (P1/P2, docs/ks_local_field_ltransfer.tex
Sec. crop_training, variant B). Architecture identical to Sections 231/233
(patch 16x16, c=3, encoder radius 2 x 3 layers, propagator window 3 x 2
layers); only the training data and loss masking differ. See
ks_latent/training/crop.py for the masking/forced-halo details.

Stage A and Stage B are each skipped if their checkpoint already exists
(preemptable at the stage level).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch

from ks_latent.config import AuxPropagatorConfig, LocalFieldAutoencoderConfig2D
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
from ks_latent.models.propagator import AuxPropagator
from ks_latent.models.propagator import _biot_savart_velocity
from ks_latent.training.crop import (
    BSContext, CropAEConfig, CropPropConfig, encode_crop_bank, encode_crop_bank_aligned, site_anchor,
    train_crop_ae, train_crop_prop,
)
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

A = Path("artifacts")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop-sites", type=int, default=48)
    p.add_argument("--crop-sites-y", type=int, default=None,
                   help="P7: also crop in y (2D crops for isotropic systems, e.g. 2D KS). Default: full periodic y.")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--ae-steps", type=int, default=8000)
    p.add_argument("--prop-steps", type=int, default=6000)
    p.add_argument("--k-max", type=int, default=12, help="P9b: max rollout length in Stage B")
    p.add_argument("--k-mid", type=int, default=8)
    p.add_argument("--noise-end", type=float, default=0.02, help="final input-noise std in Stage B")
    # I30(d) fine-tuning (2026-10-09): continue Stage B from an existing propagator with a different
    # input-noise level. Defaults reproduce the original recipe exactly.
    p.add_argument("--init-prop", default=None, help="Stage B: start from this propagator checkpoint (fine-tune)")
    p.add_argument("--noise-start", type=float, default=None, help="initial input-noise std (default: config's 0.10)")
    p.add_argument("--prop-lr", type=float, default=None, help="Stage B peak learning rate (default: config's)")
    p.add_argument("--k-full", action="store_true", help="Stage B: rollout length k_max from the first step")
    p.add_argument("--n-offsets", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--local-channels", type=int, default=3)
    p.add_argument("--backbone", choices=["local_mlp_2d", "local_mlp_2d_consensus", "local_mlp_2d_biotsavart"],
                   default="local_mlp_2d")
    p.add_argument("--consensus-n-iter", type=int, default=4)
    p.add_argument("--consensus-radius", type=int, default=1)
    p.add_argument("--w-rank", type=float, default=0.0)
    p.add_argument("--site-mix-radius", type=int, default=2)
    p.add_argument("--n-site-mix-layers", type=int, default=3)
    p.add_argument("--w-logdet", type=float, default=None,
                    help="Override CropAEConfig.w_logdet (default 0.0035, Stage A anti-collapse "
                         "regularizer). Not previously CLI-exposed.")
    p.add_argument("--w-var", type=float, default=None,
                    help="Override CropAEConfig.w_var (default 0.02, marginal per-channel variance "
                         "floor). User-directed 2026-10-07: redundant with w_logdet for collapse "
                         "prevention (w_logdet's full-covariance log-det catches correlation-driven "
                         "collapse that a marginal variance floor provably misses -- see "
                         "logdet_barrier_loss's docstring); pass 0 to test removing it. Not "
                         "previously CLI-exposed.")
    p.add_argument("--logdet-eps", type=float, default=None,
                    help="Override CropAEConfig.logdet_eps (default 1e-3). User-directed 2026-10-07: "
                         "pass 0 for the bare (unstabilized) logdet barrier, or negative for a genuine "
                         "hard floor at eigenvalue |eps| (infeasible at init below about -0.001 to "
                         "-0.004 for this architecture, empirically measured). Not previously "
                         "CLI-exposed.")
    p.add_argument("--y-align", type=int, default=1,
                   help="P9: 2D-crop y-offsets restricted to multiples of this many grid points")
    p.add_argument("--dec-site-mix-radius", type=int, default=None, help="C19: decoder reach (None = encoder's)")
    p.add_argument("--dec-n-site-mix-layers", type=int, default=None)
    p.add_argument("--periodic-full", action="store_true",
                   help="P6 upper bound: the crop IS the whole periodic training domain (crop_sites must equal "
                        "its site count), so circular padding is exact: no loss mask, no dropped encoder sites, "
                        "no forced halo.")
    p.add_argument("--w-spectral", type=float, default=0.0,
                   help="Propagator-lessons item 3 (docs/ks_local_field_ltransfer.tex sec:propagator_lessons): "
                        "free-rollout spectral-band-share regularizer. Needs decoder reach 0 "
                        "(--dec-site-mix-radius 0 or --dec-n-site-mix-layers 0).")
    p.add_argument("--spectral-bands", default="0,2,3,5",
                   help="Flat list of band edges lo1,hi1,lo2,hi2,... in |k| units, e.g. '0,2,3,5' for two bands "
                        "[0,2) and [3,5) (Kolmogorov: large-scale / forcing-scale).")
    p.add_argument("--spectral-every", type=int, default=1)
    p.add_argument("--w-grad", type=float, default=0.0,
                   help="Stage A gradient-matching (H1) reconstruction weight; penalizes patch-edge seams.")
    p.add_argument("--stage-a-only", action="store_true", help="Train/load the autoencoder and stop.")
    args = p.parse_args()
    set_seed(args.seed)
    dev = get_device(args.device)

    with h5py.File(args.dataset, "r") as f:
        md = dict(f["metadata"].attrs)
        n_train, NX, NY = int(md["n_train"]), int(md["NX"]), int(md["NY"])
        traj = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    traj = traj.reshape(traj.shape[0], traj.shape[1], NX, NY)
    u_train, u_val = traj[:n_train].to(dev), traj[n_train:].to(dev)
    norm = {"normalization_mean": float(md["normalization_mean"]),
            "normalization_std": float(md["normalization_std"]), "train_dataset": args.dataset}
    print(f"[crop] {args.dataset}: train {tuple(u_train.shape)} val {tuple(u_val.shape)} alpha={md.get('alpha', 'n/a')}", flush=True)

    NYc = args.crop_sites_y * 16 if args.crop_sites_y else NY
    ae_cfg = LocalFieldAutoencoderConfig2D(NX=args.crop_sites * 16, NY=NYc, n_sites_x=args.crop_sites, n_sites_y=NYc // 16,
                                           local_channels=args.local_channels, site_mix_radius=args.site_mix_radius,
                                           n_site_mix_layers=args.n_site_mix_layers, hidden=32,
                                           dec_site_mix_radius=args.dec_site_mix_radius,
                                           dec_n_site_mix_layers=args.dec_n_site_mix_layers)
    enc_sites = ae_cfg.site_mix_radius * ae_cfg.n_site_mix_layers  # 6 by default
    dr = ae_cfg.site_mix_radius if args.dec_site_mix_radius is None else args.dec_site_mix_radius
    dl = ae_cfg.n_site_mix_layers if args.dec_n_site_mix_layers is None else args.dec_n_site_mix_layers
    dec_sites = dr * dl
    if args.periodic_full:
        assert args.crop_sites * 16 == NX, f"--periodic-full needs crop_sites*16 == NX ({NX})"
        enc_sites = dec_sites = 0
    ae_kwargs = {}
    if args.w_logdet is not None:
        ae_kwargs["w_logdet"] = args.w_logdet
    if args.w_var is not None:
        ae_kwargs["w_var"] = args.w_var
    if args.logdet_eps is not None:
        ae_kwargs["logdet_eps"] = args.logdet_eps
    acfg = CropAEConfig(crop_sites=args.crop_sites, crop_sites_y=args.crop_sites_y, y_align=args.y_align,
                        mask_sites=enc_sites + dec_sites,
                        enc_sites=enc_sites, steps=args.ae_steps, seed=args.seed, w_grad=args.w_grad, **ae_kwargs)
    ae_path = A / f"stage1_ae_kolmogorov_crop_{args.tag}.pt"
    ae = KSAutoencoderLocalField2D(ae_cfg).to(dev)
    if ae_path.exists():
        ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=False)["ae_state_dict"])
        print(f"[crop] Stage A done already ({ae_path}); skipping", flush=True)
    else:
        print(f"[crop] Stage A: crop {args.crop_sites} sites ({args.crop_sites * 16} grid), "
              f"mask {acfg.mask_sites} sites/side, {acfg.steps} steps", flush=True)
        t0 = time.time()
        resA = train_crop_ae(ae, u_train, u_val, acfg, log=lambda s: print(s, flush=True))
        torch.save({"ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "encoder_kind": "local_field_2d",
                    "val_recon_interior": resA["best_val_recon_interior"], "crop_ae_config": acfg.__dict__, **norm},
                   ae_path)
        write_provenance(ae_path, config=ae_cfg, seed=args.seed, device=dev.type, wall_time_s=time.time() - t0)
        (A / f"crop_stageA_history_{args.tag}.json").write_text(json.dumps(resA, indent=2))

    if args.stage_a_only:
        print("[crop] --stage-a-only: stopping after Stage A", flush=True)
        return

    # Halo must cover the propagator's per-step receptive field (exactness test in
    # tests/unit/test_crop_training.py): n_layers x window, plus n_iter x radius
    # per layer for the consensus relaxation.
    per_layer = args.window + (args.consensus_n_iter * args.consensus_radius
                               if args.backbone == "local_mlp_2d_consensus" else 0)
    halo = 0 if args.periodic_full else 2 * per_layer
    bands_flat = [float(x) for x in args.spectral_bands.split(",")]
    spectral_bands = tuple(zip(bands_flat[0::2], bands_flat[1::2]))
    pcfg = CropPropConfig(halo_sites=halo, halo_sites_y=halo if args.crop_sites_y else 0, seed=args.seed,
                          steps=args.prop_steps, w_rank=args.w_rank, k_max=args.k_max, k_mid=args.k_mid,
                          noise_in_end=args.noise_end, w_spectral=args.w_spectral,
                          spectral_bands=spectral_bands, spectral_every=args.spectral_every)
    if args.noise_start is not None:
        pcfg.noise_in_start = args.noise_start
    if args.prop_lr is not None:
        pcfg.lr = args.prop_lr
    if args.k_full:
        pcfg.k_mid_frac = pcfg.k_warmup_frac = 0.0
    prop_path = A / f"stage2_prop_kolmogorov_crop_{args.tag}.pt"
    if prop_path.exists():
        print(f"[crop] Stage B done already ({prop_path})", flush=True)
        return
    for q in ae.parameters():
        q.requires_grad_(False)
    bs_train = bs_val = None
    extra_cfg = {}
    px, py = ae_cfg.patch_x, ae_cfg.patch_y
    site_len = float(md["Lx"]) / (NX // px)  # physical length/site (square patches assumed)
    if args.backbone == "local_mlp_2d_biotsavart":
        Lx_w, Ly_w = float(md["Lx"]), float(md["Ly"])
        Pw = NX // px
        # C9: patch-aligned crops + truth anchors of the WIDE domain, so the
        # propagator's velocity input is the exact global Biot-Savart velocity
        # (truth outside the free interior, prediction inside) -- the same
        # operator it applies to its own anchor field at deployment.
        bank_tr, ctr, cx0_tr = encode_crop_bank_aligned(ae, u_train, args.n_offsets, args.crop_sites, enc_sites,
                                                        seed=args.seed + 1)
        bank_va, cva, cx0_va = encode_crop_bank_aligned(ae, u_val, args.n_offsets, args.crop_sites, enc_sites,
                                                        seed=args.seed + 2)
        anc_tr, anc_va = site_anchor(u_train, px, py), site_anchor(u_val, px, py)
        with torch.no_grad():
            v = _biot_savart_velocity(anc_tr[:, ::5].reshape(-1, Pw, anc_tr.shape[-1]).cpu().double(), Lx_w, Ly_w)
            vel_scale = float(v.std())
        bs_train = (ctr, cx0_tr, BSContext(anc_tr, Lx_w, Ly_w))
        bs_val = (cva, cx0_va, BSContext(anc_va, Lx_w, Ly_w))
        extra_cfg = dict(biot_savart_Lx=bank_tr.shape[2] * site_len, biot_savart_Ly=Ly_w,
                         biot_savart_vel_scale=vel_scale, biot_savart_zero_init_embed=True)
        print(f"[crop] C9 global Biot-Savart input: wide Lx={Lx_w:.3f}, site {site_len:.4f}, "
              f"vel_scale={vel_scale:.4f}", flush=True)
    else:
        bank_tr = encode_crop_bank(ae, u_train, args.n_offsets, args.crop_sites, enc_sites, seed=args.seed + 1,
                                   crop_sites_y=args.crop_sites_y, y_align=args.y_align)
        bank_va = encode_crop_bank(ae, u_val, args.n_offsets, args.crop_sites, enc_sites, seed=args.seed + 2,
                                   crop_sites_y=args.crop_sites_y, y_align=args.y_align)
    Pu = bank_tr.shape[2]
    print(f"[crop] Stage B: bank {tuple(bank_tr.shape)}, usable strip {Pu} sites, halo {halo}/side "
          f"(y: {pcfg.halo_sites_y}), free interior {Pu - 2 * halo} x {bank_tr.shape[3] - 2 * pcfg.halo_sites_y}",
          flush=True)
    c = ae_cfg.local_channels
    prop_cfg = AuxPropagatorConfig(
        d_latent=Pu * bank_tr.shape[3] * c, mode="markovian", backbone=args.backbone, attn_window=args.window,
        local_mlp2d_n_sites_x=Pu, local_mlp2d_n_sites_y=bank_tr.shape[3], local_mlp2d_local_channels=c,
        local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
        consensus_n_iter=args.consensus_n_iter, consensus_radius=args.consensus_radius, **extra_cfg,
    )
    prop = AuxPropagator(prop_cfg).to(dev)
    if args.init_prop is not None:
        init = torch.load(args.init_prop, map_location="cpu", weights_only=False)
        if init["prop_config"] != prop_cfg:
            raise ValueError(f"--init-prop config differs from this run's:\n{init['prop_config']}\nvs\n{prop_cfg}")
        prop.load_state_dict(init["prop_state_dict"])
        print(f"[crop] Stage B: fine-tuning from {args.init_prop} (noise {pcfg.noise_in_start} -> "
              f"{pcfg.noise_in_end}, lr {pcfg.lr}, k_full={args.k_full})", flush=True)
    t0 = time.time()
    resB = train_crop_prop(prop, bank_tr, bank_va, pcfg, log=lambda s: print(s, flush=True),
                           bs_train=bs_train, bs_val=bs_val, ae=ae, site_len=site_len)
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg,
                "best_val_kmax_mse": resB["best_val_kmax_mse"], "crop_prop_config": pcfg.__dict__, **norm},
               prop_path)
    write_provenance(prop_path, config=prop_cfg, seed=args.seed, device=dev.type, wall_time_s=time.time() - t0)
    (A / f"crop_stageB_history_{args.tag}.json").write_text(json.dumps(resB, indent=2))
    print("[crop] done", flush=True)


if __name__ == "__main__":
    main()
