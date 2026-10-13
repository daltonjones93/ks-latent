"""Window-trained, per-site local latent field for Lorenz-96 (2026-10-13; adapted from train_ks1d_crop.py).

Design (see docs/split_paper_tasks.md Z16): one L96 site per patch (patch 1, so the encoder and decoder
are pointwise, reach 0, and the latent observation operator is block-diagonal); c channels with channel 0
the site value itself (the anchor); a `local_mlp` propagator of radius 3 and 2 blocks (reach 6 sites,
covering the formal RK4 radius of the dt_snap = 0.1 map to dt^3 order); trained on forced-halo
windows of one wide ring (N = 256, F = 8, dt_snap = 0.1, scalar-standardized).

Original docstring of the KS script:


The 1D counterpart of the 2D recipe (`scripts/train_kolmogorov_crop.py`, the c19 / Paper II models):
a 1D `KSAutoencoderLocalField` with encoder and decoder reach 0 by default (per-patch, so the
latent observation operator is exactly block-diagonal, Paper II Prop. 1), `c` channels (channel 0
the patch-average anchor), and a `local_mlp` propagator (radius 3, 2 blocks: reach 6 sites),
trained on random windows of one wide periodic domain (`scripts/gen_ks1d_wide.py`):
  Stage A  autoencoder on `--crop-sites` windows (reconstruction + w_var + log-det barrier);
  Stage B  propagator on forced-halo rollouts (halo 6 sites >= reach), k ramped 2 -> k_mid -> k_max,
           input noise annealed 0.10 -> 0.02 (`ks_latent.training.crop.train_crop_prop`).
`--data-noise` adds one fixed realization of Gaussian noise (standardized units) to every training
and validation field, as for the 2D Paper II model.

Checkpoints are written in the L = 100 format (16 sites, NX = 256) that `scripts/eval_1d_transfer.py`
and the 1D DA scripts load and resize from:
  artifacts/stage1_ae_patched_full_<tag>.pt, artifacts/stage2_prop_patched_full_<tag>.pt
(+ .meta.json provenance, training histories in artifacts/crop1d_stage{A,B}_history_<tag>.json).

Usage: python scripts/train_ks1d_crop.py --tag pp1d_crop_c6d10 --local-channels 6 --data-noise 0.1
       python scripts/train_ks1d_crop.py --profile smoke   # tiny dataset and steps, < 1 min
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch

from ks_latent.config import LocalFieldAutoencoderConfig, PropagatorConfig
from ks_latent.models import build_autoencoder, build_propagator_from_config
from ks_latent.training.crop import CropPropConfig, train_crop_prop
from ks_latent.training.crop1d import (CropAE1DConfig, encode_crop_bank_1d, resized_ae_config,
                                       resized_prop_config, train_crop_ae_1d)
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance

A = Path("artifacts")
REF_SITES = 16  # checkpoints are saved at 16 sites (NX = 16); sizes are independent of N


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", choices=["full", "smoke"], default="full")
    p.add_argument("--dataset", default="artifacts/datasets/lorenz96_trajectories_n256.h5")
    p.add_argument("--tag", default="l96_crop_c3")
    p.add_argument("--patch", type=int, default=1, help="L96 sites per latent site")
    p.add_argument("--local-channels", type=int, default=3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--site-mix-radius", type=int, default=0)
    p.add_argument("--n-site-mix-layers", type=int, default=0)
    p.add_argument("--crop-sites", type=int, default=48)
    p.add_argument("--halo", type=int, default=6)
    p.add_argument("--attn-window", type=int, default=3)
    p.add_argument("--prop-layers", type=int, default=2)
    p.add_argument("--token-d-model", type=int, default=64)
    p.add_argument("--ae-steps", type=int, default=4000)
    p.add_argument("--prop-steps", type=int, default=9000)
    p.add_argument("--k-mid", type=int, default=8)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--noise-start", type=float, default=0.10)
    p.add_argument("--noise-end", type=float, default=0.02)
    p.add_argument("--n-offsets", type=int, default=6)
    p.add_argument("--data-noise", type=float, default=0.0)
    p.add_argument("--data-noise-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    if args.profile == "smoke":
        args.dataset, args.tag = "artifacts/datasets/lorenz96_trajectories_n64.h5", "smoke_l96_crop"
        args.ae_steps, args.prop_steps, args.n_offsets, args.crop_sites = 20, 20, 2, 24
        args.k_mid, args.k_max, args.device = 3, 4, "cpu"

    PATCH = args.patch
    torch.manual_seed(args.seed)
    dev = get_device(args.device)
    with h5py.File(args.dataset, "r") as f:
        md = dict(f["metadata"].attrs)
        n_train = int(md["n_train"])
        u = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    if args.data_noise > 0:
        g = torch.Generator().manual_seed(args.data_noise_seed)
        u = u + args.data_noise * torch.randn(u.shape, generator=g)
        print(f"[crop1d] added fixed Gaussian data noise, std {args.data_noise} (standardized units)", flush=True)
    u_train, u_val = u[:n_train].to(dev), u[n_train:].to(dev)
    print(f"[crop1d] {args.dataset}: train {tuple(u_train.shape)} val {tuple(u_val.shape)} N={md['N']} F={md['F']} dt_snap={md['dt_snap']}", flush=True)
    assert u.shape[-1] == int(md["N"]) and float(md["dt_snap"]) == 0.1 and float(md["F"]) == 8.0
    norm = {"normalization_mean": float(md["normalization_mean"]), "normalization_std": float(md["normalization_std"]),
            "l96_F": float(md["F"]), "l96_dt_snap": float(md["dt_snap"]),
            "train_dataset": args.dataset, "data_noise": args.data_noise, "data_noise_seed": args.data_noise_seed}

    c, reach = args.local_channels, args.site_mix_radius * args.n_site_mix_layers
    ae_cfg = LocalFieldAutoencoderConfig(NX=args.crop_sites * PATCH, n_sites=args.crop_sites, local_channels=c,
                                         site_mix_radius=args.site_mix_radius,
                                         n_site_mix_layers=args.n_site_mix_layers, hidden=args.hidden)
    ae = build_autoencoder("local_field", ae_cfg).to(dev)
    acfg = CropAE1DConfig(crop_sites=args.crop_sites, mask_sites=2 * reach, steps=args.ae_steps, seed=args.seed)
    ae_path = A / f"stage1_ae_patched_full_{args.tag}.pt"
    if ae_path.exists():
        ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=False)["ae_state_dict"])
        print(f"[crop1d] Stage A done already ({ae_path}); skipping", flush=True)
    else:
        print(f"[crop1d] Stage A: windows of {args.crop_sites} sites, reach {reach}, mask {acfg.mask_sites} "
              f"sites/side, {acfg.steps} steps", flush=True)
        t0 = time.time()
        resA = train_crop_ae_1d(ae, u_train, u_val, acfg, log=lambda s: print(s, flush=True))
        ref = build_autoencoder("local_field", resized_ae_config(ae_cfg, REF_SITES))
        ref.load_state_dict(ae.state_dict())  # same tensors; only the number of sites differs
        torch.save({"ae_state_dict": ref.state_dict(), "ae_config": ref.cfg, "encoder_kind": "local_field",
                    "val_recon_final": resA["best_val_recon_interior"], "crop_ae_config": acfg.__dict__, **norm},
                   ae_path)
        write_provenance(ae_path, config=ref.cfg, seed=args.seed, device=dev.type, wall_time_s=time.time() - t0)
        (A / f"crop1d_stageA_history_{args.tag}.json").write_text(json.dumps(resA, indent=2))

    Pu = args.crop_sites - 2 * reach  # the encoder's outer `reach` sites saw padding
    prop_cfg = PropagatorConfig(d_latent=Pu * c, mode="markovian", backbone="local_mlp", n_tokens=Pu,
                                token_d_model=args.token_d_model, token_n_layers=args.prop_layers,
                                token_mlp_ratio=4, attn_window=args.attn_window, zero_init=True)
    assert args.halo >= args.attn_window * args.prop_layers, "halo must cover the propagator's reach (Prop. 2)"
    prop = build_propagator_from_config(prop_cfg).to(dev)
    bank_tr = encode_crop_bank_1d(ae, u_train, args.n_offsets, args.crop_sites, reach, seed=args.seed + 1)
    bank_va = encode_crop_bank_1d(ae, u_val, max(1, args.n_offsets // 4), args.crop_sites, reach, seed=args.seed + 2)
    print(f"[crop1d] Stage B: bank {tuple(bank_tr.shape)}, halo {args.halo}, k 2->{args.k_mid}->{args.k_max}, "
          f"noise {args.noise_start}->{args.noise_end}, {args.prop_steps} steps", flush=True)
    pcfg = CropPropConfig(halo_sites=args.halo, halo_sites_y=0, steps=args.prop_steps, k_mid=args.k_mid,
                          k_max=args.k_max, noise_in_start=args.noise_start, noise_in_end=args.noise_end,
                          seed=args.seed)
    t0 = time.time()
    resB = train_crop_prop(prop, bank_tr, bank_va, pcfg, log=lambda s: print(s, flush=True))
    ref_cfg = resized_prop_config(prop_cfg, REF_SITES, c)
    ref = build_propagator_from_config(ref_cfg)
    ref.load_state_dict(prop.state_dict())
    prop_path = A / f"stage2_prop_patched_full_{args.tag}.pt"
    torch.save({"prop_state_dict": ref.state_dict(), "prop_config": ref_cfg,
                "best_val_kmax_mse": resB["best_val_kmax_mse"], "crop_prop_config": pcfg.__dict__, **norm}, prop_path)
    write_provenance(prop_path, config=ref_cfg, seed=args.seed, device=dev.type, wall_time_s=time.time() - t0)
    (A / f"crop1d_stageB_history_{args.tag}.json").write_text(json.dumps(resB, indent=2))
    print("[crop1d] done", flush=True)


if __name__ == "__main__":
    main()
