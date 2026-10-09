"""G3 (Kolmogorov flow, alpha=0.1 follow-up to G1/G2/G2b): zero-retraining
D_KY at a given x_scale for EITHER isotropic architecture trained on the
alpha=0.1 dataset -- Section 233 (`local_field_2d`/conv) or Section 234
(`local_attention_2d`/windowed attention). Dispatches on the checkpoint's
own `encoder_kind` so one script covers both.

`--x-scale 1.0` (no resize at all -- the checkpoint's own trained size)
is the collapse sanity check this project's own discipline requires
before trusting any transfer number (`docs/ks_local_field_ltransfer.tex`'s
G2b episode: a phase-conditioned propagator passed Stage-1/Stage-2 loss
checks fine but had collapsed to a near-fixed point, D_KY~1, only caught
by measuring the actual Lyapunov spectrum). `--x-scale 2.0` is the real
transfer test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig
from ks_latent.models import (
    load_autoencoder_checkpoint_resized_2d,
    load_autoencoder_checkpoint_resized_2d_attn,
    load_propagator_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d_biotsavart,
    load_propagator_checkpoint_resized_2d_multiscale,
)
from ks_latent.solver.kolmogorov import spinup_kolmogorov

ARTIFACTS_DIR = Path("artifacts/experiments")

TRAINED_LX = TRAINED_LY = 2.0 * np.pi
TRAINED_NX = TRAINED_NY = 128
TRAINED_N_SITES_X = TRAINED_N_SITES_Y = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--prop-ckpt", required=True)
    parser.add_argument("--x-scale", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--forcing-n", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--spinup-time", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--true-dky-json", default=None, help="optional path to a true-D_KY json at this x_scale for comparison")
    args = parser.parse_args()

    new_Lx = TRAINED_LX * args.x_scale
    new_Ly = TRAINED_LY
    new_NX = int(round(TRAINED_NX * args.x_scale))
    new_NY = TRAINED_NY
    new_n_sites_x = int(round(TRAINED_N_SITES_X * args.x_scale))
    new_n_sites_y = TRAINED_N_SITES_Y
    assert new_NX % new_n_sites_x == 0, f"new_NX={new_NX} not divisible by new_n_sites_x={new_n_sites_x}"

    ckpt_probe = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    encoder_kind = ckpt_probe.get("encoder_kind", "transformer")
    print(f"[g3] ae checkpoint encoder_kind={encoder_kind!r}")

    if encoder_kind == "local_field_2d":
        ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
            args.ae_ckpt, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y, new_NX=new_NX, new_NY=new_NY,
        )
    elif encoder_kind == "local_attention_2d":
        ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d_attn(
            args.ae_ckpt, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y, new_NX=new_NX, new_NY=new_NY,
        )
    else:
        raise ValueError(f"unsupported encoder_kind={encoder_kind!r} for this script")

    prop_ckpt_probe = torch.load(args.prop_ckpt, map_location="cpu", weights_only=False)
    prop_backbone = getattr(prop_ckpt_probe["prop_config"], "backbone", None)
    if prop_backbone == "local_mlp_2d_biotsavart":
        prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d_biotsavart(
            args.prop_ckpt, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y,
        )
    elif prop_backbone == "multiscale_local_mlp_2d":
        prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d_multiscale(
            args.prop_ckpt, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y,
        )
    else:
        prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d(
            args.prop_ckpt, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y,
        )
    ae.eval(); prop.eval()
    d_latent = ae_cfg.d_latent

    print(f"[g3] x_scale={args.x_scale}  Lx={new_Lx:.4f} Ly={new_Ly:.4f}  NX={new_NX} NY={new_NY}  "
          f"n_sites=({new_n_sites_x},{new_n_sites_y})  d_latent={d_latent}  alpha={args.alpha}", flush=True)

    t0 = time.time()
    new_cfg = KolmogorovConfig(
        Lx=new_Lx, Ly=new_Ly, NX=new_NX, NY=new_NY, nu=args.nu, alpha=args.alpha,
        forcing_n=args.forcing_n, dt=args.dt, spinup_time=args.spinup_time, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        u0_phys = spinup_kolmogorov(new_cfg, rng)
        u0_flat = torch.tensor(u0_phys.reshape(-1), dtype=torch.float32).unsqueeze(0)
        z0 = ae.encode(u0_flat)[0].numpy().astype(np.float64)

    model_result = lyapunov_spectrum_latent_propagator(
        prop, np.stack([z0, z0]), mode="single_state",
        n_directions=d_latent, n_steps=2000, qr_every=10,
        dt_snap=new_cfg.dt_snap, warmup_steps=200, seed=args.seed, max_abs_state=1e3,
    )
    model_wall = time.time() - t0
    print(f"[g3] MODEL ({encoder_kind}, x_scale={args.x_scale}): lambda1={model_result.exponents[0]:.4f} "
          f"n_positive={model_result.n_positive}/{d_latent} D_KY={model_result.kaplan_yorke_dimension:.2f} "
          f"({model_wall:.1f}s)", flush=True)

    true_dky = None
    if args.true_dky_json and Path(args.true_dky_json).exists():
        true_data = json.loads(Path(args.true_dky_json).read_text())
        true_dky = true_data.get("D_KY") or true_data.get("true_D_KY")
        rel_err = abs(model_result.kaplan_yorke_dimension - true_dky) / true_dky * 100.0
        print(f"[g3] TRUE D_KY (from {args.true_dky_json}): {true_dky:.2f}  relative error: {rel_err:.2f}%")

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_sha = None

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "encoder_kind": encoder_kind, "x_scale": args.x_scale, "Lx": new_Lx, "Ly": new_Ly,
        "NX": new_NX, "NY": new_NY, "n_sites_x": new_n_sites_x, "n_sites_y": new_n_sites_y,
        "d_latent": d_latent, "alpha": args.alpha, "nu": args.nu, "forcing_n": args.forcing_n,
        "dt": args.dt, "spinup_time": args.spinup_time, "seed": args.seed,
        "model_lambda1": float(model_result.exponents[0]), "model_n_positive": model_result.n_positive,
        "model_D_KY": model_result.kaplan_yorke_dimension,
        "true_D_KY": true_dky, "ae_ckpt": args.ae_ckpt, "prop_ckpt": args.prop_ckpt,
        "git_sha": git_sha, "model_wall_s": model_wall,
    }
    out_path = ARTIFACTS_DIR / f"g3_kolmogorov_alpha_transfer{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[g3] wrote {out_path}")


if __name__ == "__main__":
    main()
