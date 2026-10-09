"""Zero-shot evaluation of a crop-trained model on periodic boxes of
Lx = scale * 2pi (Ly = 2pi): resize both checkpoints to 8*scale x 8 sites,
spin up the true PDE, normalize the IC with the WIDE training set's stats
(stored in the checkpoint), encode, and measure the model's Lyapunov
spectrum. Settings identical to exp_g3 (n_directions=d_latent, 2000 steps,
QR every 10, warmup 200)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig
from ks_latent.models import (
    load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d_biotsavart,
)
from ks_latent.solver.kolmogorov import spinup_kolmogorov

OUT = Path("artifacts/experiments/crop")
TRUTH = {0.1: {1: 15.88, 2: 30.34, 3: 56.35}, 0.05: {1: 23.26, 2: 56.84}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--prop-ckpt", required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--truth-json", default=None, help="optional truth file overriding the built-in table")
    args = p.parse_args()

    ck = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    mean, std = ck["normalization_mean"], ck["normalization_std"]
    import h5py
    with h5py.File(ck["train_dataset"], "r") as f:
        md = dict(f["metadata"].attrs)
    alpha = float(md["alpha"])
    nsx, nsy, NX, NY = 8 * args.scale, 8, 128 * args.scale, 128
    ae = load_autoencoder_checkpoint_resized_2d(args.ae_ckpt, new_n_sites_x=nsx, new_n_sites_y=nsy,
                                                new_NX=NX, new_NY=NY)[0].eval()
    backbone = getattr(torch.load(args.prop_ckpt, map_location="cpu", weights_only=False)["prop_config"], "backbone", None)
    loader = (load_propagator_checkpoint_resized_2d_biotsavart if backbone == "local_mlp_2d_biotsavart"
              else load_propagator_checkpoint_resized_2d)
    prop, prop_cfg, _ = loader(args.prop_ckpt, new_n_sites_x=nsx, new_n_sites_y=nsy)
    prop.eval()
    cfg = KolmogorovConfig(Lx=2 * math.pi * args.scale, Ly=2 * math.pi, NX=NX, NY=NY, nu=float(md["nu"]),
                           alpha=alpha, forcing_n=int(md["forcing_n"]), dt=float(md["dt"]), spinup_time=100.0,
                           seed=args.seed)
    u0 = (spinup_kolmogorov(cfg, np.random.default_rng(args.seed)) - mean) / std
    t0 = time.time()
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(u0.reshape(1, -1), dtype=torch.float32))[0].numpy().astype(np.float64)
    res = lyapunov_spectrum_latent_propagator(
        prop, np.stack([z0, z0]), mode="single_state", n_directions=prop_cfg.d_latent, n_steps=2000,
        qr_every=10, dt_snap=float(md["dt_snap"]), warmup_steps=200, seed=args.seed, max_abs_state=1e3,
    )
    truth = TRUTH.get(alpha, {}).get(args.scale)
    if args.truth_json and Path(args.truth_json).exists():
        truth = json.loads(Path(args.truth_json).read_text()).get("D_KY", truth)
    out = {"alpha": alpha, "scale": args.scale, "seed": args.seed, "d_latent": prop_cfg.d_latent,
           "model_D_KY": res.kaplan_yorke_dimension, "model_lambda1": float(res.exponents[0]),
           "model_n_positive": int(res.n_positive), "true_D_KY": truth,
           "rel_err_pct": abs(res.kaplan_yorke_dimension - truth) / truth * 100.0 if truth else None,
           "ae_ckpt": args.ae_ckpt, "prop_ckpt": args.prop_ckpt, "ic": "normalized (wide-set stats)",
           "wall_s": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
