"""G2 (`docs/ks_local_field_ltransfer.tex`, "L-transfer for the 2D
systems" plan): ground-truth Lyapunov spectrum of the TRUE Kolmogorov
flow PDE at the L-transfer size, computed via `lyapunov_spectrum_kolmogorov`
-- not the learned model -- compared against Section 231's checkpoint
transferred to the SAME size with ZERO retraining.

The single most scientifically important experiment in the G-plan (see
the tex plan's own G2 entry): tests whether Kolmogorov flow's own
extensivity -- genuinely uncertain a priori, given 2D turbulence's
inverse-cascade condensation problem, unlike either KS variant's fixed
critical wavenumber -- holds well enough for the local-field architecture
to transfer at all.

Transfer is in x ONLY (Ly=2*pi fixed at the forcing period; Kolmogorov
flow is only exactly translation-equivariant in x -- see
`ks_latent/solver/kolmogorov.py`'s own module docstring), Lx doubled.
Same drag alpha=0.05 carried through unchanged, testing whether that
choice ALSO transfers, not just whether it exists at the trained size.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_kolmogorov, lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.kolmogorov import spinup_kolmogorov

ARTIFACTS_DIR = Path("artifacts/experiments")

AE_CKPT = "artifacts/stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt"
PROP_CKPT = "artifacts/stage2_prop_kolmogorov_full_section231_kolmogorov_localmlp2d_window3_warmstart_k12_300ep.pt"
TRAINED_LX = TRAINED_LY = 2.0 * np.pi
TRAINED_NX = TRAINED_NY = 128
TRAINED_N_SITES_X = TRAINED_N_SITES_Y = 8
D_KY_OVER_AREA_TRAINED = 23.26 / (TRAINED_LX * TRAINED_LY)  # Section 231's own ground-truth value at the trained size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-scale", type=float, default=2.0, help="Lx transfer factor (Ly fixed).")
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--forcing-n", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-directions", type=int, default=None, help="Default: ceil(1.6 * predicted D_KY).")
    parser.add_argument("--total-time", type=float, default=40.0)
    parser.add_argument("--warmup-time", type=float, default=20.0)
    parser.add_argument("--qr-interval", type=float, default=0.5)
    parser.add_argument("--spinup-time", type=float, default=100.0)
    parser.add_argument("--max-abs-state", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    new_Lx = TRAINED_LX * args.x_scale
    new_Ly = TRAINED_LY
    new_NX = int(round(TRAINED_NX * args.x_scale))
    new_NY = TRAINED_NY
    new_n_sites_x = int(round(TRAINED_N_SITES_X * args.x_scale))
    new_n_sites_y = TRAINED_N_SITES_Y
    assert new_NX % new_n_sites_x == 0, f"new_NX={new_NX} not divisible by new_n_sites_x={new_n_sites_x}"

    predicted_d_ky = D_KY_OVER_AREA_TRAINED * new_Lx * new_Ly
    n_directions = args.n_directions if args.n_directions is not None else math.ceil(1.6 * predicted_d_ky)

    print(f"[g2] x_scale={args.x_scale}  Lx={new_Lx:.4f} (Ly={new_Ly:.4f} fixed)  NX={new_NX} NY={new_NY}  "
          f"n_sites=({new_n_sites_x},{new_n_sites_y})  dx={new_Lx/new_NX:.6f} (trained dx={TRAINED_LX/TRAINED_NX:.6f})")
    print(f"[g2] predicted D_KY (area-scaled from trained-size ground truth): {predicted_d_ky:.1f}")
    print(f"[g2] n_directions={n_directions}  total_time={args.total_time}  warmup_time={args.warmup_time}  "
          f"qr_interval={args.qr_interval}  spinup_time={args.spinup_time}  nu={args.nu}  alpha={args.alpha}  seed={args.seed}",
          flush=True)

    # --- (a) model's own zero-retraining transfer prediction (cheap) ---
    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        AE_CKPT, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y, new_NX=new_NX, new_NY=new_NY,
    )
    prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d(
        PROP_CKPT, new_n_sites_x=new_n_sites_x, new_n_sites_y=new_n_sites_y,
    )
    ae.eval(); prop.eval()
    d_latent = ae_cfg.d_latent

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
    print(f"[g2] MODEL (zero-retraining transfer): lambda1={model_result.exponents[0]:.4f} "
          f"n_positive={model_result.n_positive}/{d_latent} D_KY={model_result.kaplan_yorke_dimension:.2f} "
          f"({model_wall:.1f}s)", flush=True)

    # --- (b) true PDE ground truth (expensive) ---
    t0 = time.time()
    true_result = lyapunov_spectrum_kolmogorov(
        new_cfg,
        n_directions=n_directions,
        total_time=args.total_time,
        qr_interval=args.qr_interval,
        warmup_time=args.warmup_time,
        seed=args.seed,
        max_abs_state=args.max_abs_state,
    )
    true_wall = time.time() - t0
    print(f"[g2] TRUE PDE: lambda1={true_result.exponents[0]:.4f} "
          f"n_positive={true_result.n_positive}/{n_directions} D_KY={true_result.kaplan_yorke_dimension:.2f} "
          f"({true_wall:.1f}s)", flush=True)

    rel_err = abs(model_result.kaplan_yorke_dimension - true_result.kaplan_yorke_dimension) / true_result.kaplan_yorke_dimension * 100.0
    print(f"[g2] relative error, model vs TRUE PDE D_KY: {rel_err:.2f}%")

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_sha = None

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "x_scale": args.x_scale, "Lx": new_Lx, "Ly": new_Ly, "NX": new_NX, "NY": new_NY,
        "n_sites_x": new_n_sites_x, "n_sites_y": new_n_sites_y, "d_latent": d_latent,
        "nu": args.nu, "alpha": args.alpha, "forcing_n": args.forcing_n, "dt": args.dt,
        "n_directions": n_directions, "total_time": args.total_time, "warmup_time": args.warmup_time,
        "qr_interval": args.qr_interval, "spinup_time": args.spinup_time, "seed": args.seed,
        "model_lambda1": float(model_result.exponents[0]), "model_n_positive": model_result.n_positive,
        "model_D_KY": model_result.kaplan_yorke_dimension,
        "true_lambda1": float(true_result.exponents[0]), "true_n_positive": true_result.n_positive,
        "true_D_KY": true_result.kaplan_yorke_dimension,
        "true_exponents": [float(x) for x in true_result.exponents],
        "relative_error_pct": rel_err, "git_sha": git_sha,
        "model_wall_s": model_wall, "true_wall_s": true_wall,
    }
    out_path = ARTIFACTS_DIR / f"g2_kolmogorov_transfer{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[g2] wrote {out_path}")


if __name__ == "__main__":
    main()
