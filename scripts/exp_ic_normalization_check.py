"""Does the transfer-evaluation initial condition need normalizing?

Every training dataset stores (u - mean)/std with std computed on the
training split, but the transfer-evaluation scripts (exp_g1/g2/g3,
run_ltransfer_test) encode raw solver output. The mis-scale factor is the
dataset's `normalization_std`: ~1.32 (1D KS), ~1.39/1.49 (Kolmogorov
alpha=0.05/0.1), ~3.8 (2D KS). Lyapunov exponents are measured after a
warmup, so the IC mostly selects which basin the model's latent dynamics
settle into; this script measures whether that changes D_KY.

Identical settings to exp_g3 (n_directions=d_latent, 2000 steps, qr every
10, warmup 200); only `--ic-mode` differs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig, KSConfig2D
from ks_latent.models import (
    load_autoencoder_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d_biotsavart,
    load_propagator_checkpoint_resized_2d_multiscale,
)
from ks_latent.solver.kolmogorov import spinup_kolmogorov
from ks_latent.solver.ks2d import spinup2d

OUT = Path("artifacts/experiments/icnorm")


def load_prop(path, nsx, nsy):
    backbone = getattr(torch.load(path, map_location="cpu", weights_only=False)["prop_config"], "backbone", None)
    loader = {
        "local_mlp_2d_biotsavart": load_propagator_checkpoint_resized_2d_biotsavart,
        "multiscale_local_mlp_2d": load_propagator_checkpoint_resized_2d_multiscale,
    }.get(backbone, load_propagator_checkpoint_resized_2d)
    return loader(path, new_n_sites_x=nsx, new_n_sites_y=nsy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["kolmogorov", "ks2d"], required=True)
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--prop-ckpt", required=True)
    p.add_argument("--dataset", required=True, help="training dataset (source of normalization stats and native grid)")
    p.add_argument("--x-scale", type=float, default=2.0)
    p.add_argument("--y-scale", type=float, default=1.0)
    p.add_argument("--ic-mode", choices=["raw", "normalized"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    with h5py.File(args.dataset, "r") as f:
        md = dict(f["metadata"].attrs)
    mean, std = float(md["normalization_mean"]), float(md["normalization_std"])
    Lx0, Ly0, NX0, NY0 = float(md["Lx"]), float(md["Ly"]), int(md["NX"]), int(md["NY"])

    ae_cfg0 = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)["ae_config"]
    nsx = int(round(ae_cfg0.n_sites_x * args.x_scale))
    nsy = int(round(ae_cfg0.n_sites_y * args.y_scale))
    NX, NY = int(round(NX0 * args.x_scale)), int(round(NY0 * args.y_scale))
    Lx, Ly = Lx0 * args.x_scale, Ly0 * args.y_scale

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        args.ae_ckpt, new_n_sites_x=nsx, new_n_sites_y=nsy, new_NX=NX, new_NY=NY,
    )
    prop, prop_cfg, _ = load_prop(args.prop_ckpt, nsx, nsy)
    ae.eval(); prop.eval()
    d = ae_cfg.d_latent

    rng = np.random.default_rng(args.seed)
    if args.system == "kolmogorov":
        cfg = KolmogorovConfig(Lx=Lx, Ly=Ly, NX=NX, NY=NY, nu=float(md["nu"]), alpha=float(md["alpha"]),
                               forcing_n=int(md["forcing_n"]), dt=float(md["dt"]), spinup_time=100.0,
                               seed=args.seed)
        u0 = spinup_kolmogorov(cfg, rng)
    else:
        cfg = KSConfig2D(Lx=Lx, Ly=Ly, NX=NX, NY=NY, spinup_time=200.0, seed=args.seed)
        u0 = spinup2d(cfg, rng)
    dt_snap = float(md["dt_snap"])

    u_in = (u0 - mean) / std if args.ic_mode == "normalized" else u0
    t0 = time.time()
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(u_in.reshape(1, -1), dtype=torch.float32))[0].numpy().astype(np.float64)
    res = lyapunov_spectrum_latent_propagator(
        prop, np.stack([z0, z0]), mode="single_state", n_directions=d, n_steps=2000, qr_every=10,
        dt_snap=dt_snap, warmup_steps=200, seed=args.seed, max_abs_state=1e3,
    )
    out = {
        "system": args.system, "ic_mode": args.ic_mode, "x_scale": args.x_scale, "y_scale": args.y_scale,
        "seed": args.seed, "d_latent": d, "normalization_std": std, "raw_ic_std": float(u0.std()),
        "model_D_KY": res.kaplan_yorke_dimension, "model_lambda1": float(res.exponents[0]),
        "model_n_positive": int(res.n_positive), "wall_s": time.time() - t0,
        "ae_ckpt": args.ae_ckpt, "prop_ckpt": args.prop_ckpt, "dataset": args.dataset,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
