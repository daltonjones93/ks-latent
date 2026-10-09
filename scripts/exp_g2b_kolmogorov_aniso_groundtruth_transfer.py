"""G2b (`docs/ks_local_field_ltransfer.tex` "orientation hypothesis"
follow-up, Kolmogorov-flow-specific correction): does the anisotropic,
phase-conditioned architecture (`local_field_2d_aniso`/`local_mlp_2d_aniso`,
user-directed "sure, build it") get a different zero-retraining x-transfer
result than Section 231's own isotropic checkpoint did in G2?

Reuses G2's own already-computed TRUE ground-truth Lyapunov spectrum at
the x_scale=2 transfer size (`artifacts/experiments/
g2_kolmogorov_transfer_full.json`, true_D_KY=56.84 -- an expensive,
~2-hour computation of the actual PDE, unaffected by which learned model
is being compared against it) rather than recomputing it. Only the
model's own cheap zero-shot prediction is computed here, using
`load_autoencoder_checkpoint_resized_2d_aniso_x`/`load_propagator_
checkpoint_resized_2d_aniso_x` (X-ONLY resize -- Ly/NY/n_sites_y stay
fixed at the trained values, since the y-phase buffers are tied to the
trained forcing_n/n_sites_y; see those functions' own docstrings).

G2's own isotropic result for reference: model_D_KY=28.44 vs.
true_D_KY=56.84 (49.96% relative error). This script answers the user's
own stated goal ("let's see if we can get a different result") directly.
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
    load_autoencoder_checkpoint_resized_2d_aniso_x,
    load_propagator_checkpoint_resized_2d_aniso_x,
)
from ks_latent.solver.kolmogorov import spinup_kolmogorov

ARTIFACTS_DIR = Path("artifacts/experiments")

G2_ISO_RESULT_PATH = ARTIFACTS_DIR / "g2_kolmogorov_transfer_full.json"

TRAINED_LX = TRAINED_LY = 2.0 * np.pi
TRAINED_NX = TRAINED_NY = 128
TRAINED_N_SITES_X = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--prop-ckpt", required=True)
    parser.add_argument("--x-scale", type=float, default=2.0, help="Lx transfer factor (Ly fixed).")
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--forcing-n", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--spinup-time", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    new_Lx = TRAINED_LX * args.x_scale
    new_NX = int(round(TRAINED_NX * args.x_scale))
    new_n_sites_x = int(round(TRAINED_N_SITES_X * args.x_scale))
    assert new_NX % new_n_sites_x == 0, f"new_NX={new_NX} not divisible by new_n_sites_x={new_n_sites_x}"

    print(f"[g2b] x_scale={args.x_scale}  Lx={new_Lx:.4f} (Ly={TRAINED_LY:.4f} fixed)  NX={new_NX} NY={TRAINED_NY}  "
          f"n_sites_x={new_n_sites_x} (n_sites_y fixed at trained value)")

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d_aniso_x(
        args.ae_ckpt, new_n_sites_x=new_n_sites_x, new_NX=new_NX,
    )
    prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d_aniso_x(
        args.prop_ckpt, new_n_sites_x=new_n_sites_x,
    )
    ae.eval(); prop.eval()
    d_latent = ae_cfg.d_latent
    print(f"[g2b] resized d_latent={d_latent} (n_sites_x={ae_cfg.n_sites_x}, n_sites_y={ae_cfg.n_sites_y})")

    t0 = time.time()
    new_cfg = KolmogorovConfig(
        Lx=new_Lx, Ly=TRAINED_LY, NX=new_NX, NY=TRAINED_NY, nu=args.nu, alpha=args.alpha,
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
    print(f"[g2b] ANISO MODEL (zero-retraining x-transfer): lambda1={model_result.exponents[0]:.4f} "
          f"n_positive={model_result.n_positive}/{d_latent} D_KY={model_result.kaplan_yorke_dimension:.2f} "
          f"({model_wall:.1f}s)", flush=True)

    iso_result = None
    if G2_ISO_RESULT_PATH.exists():
        iso_result = json.loads(G2_ISO_RESULT_PATH.read_text())
        true_D_KY = iso_result["true_D_KY"]
        iso_D_KY = iso_result["model_D_KY"]
        rel_err_true = abs(model_result.kaplan_yorke_dimension - true_D_KY) / true_D_KY * 100.0
        print(f"[g2b] TRUE PDE D_KY (reused from G2, same x_scale): {true_D_KY:.2f}")
        print(f"[g2b] G2's ISOTROPIC model D_KY (for reference): {iso_D_KY:.2f}")
        print(f"[g2b] ANISO model relative error vs TRUE PDE: {rel_err_true:.2f}%  "
              f"(isotropic model's own error was {iso_result['relative_error_pct']:.2f}%)")
    else:
        print(f"[g2b] WARNING: {G2_ISO_RESULT_PATH} not found -- cannot compare against true PDE or isotropic model.")

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_sha = None

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "x_scale": args.x_scale, "Lx": new_Lx, "Ly": TRAINED_LY, "NX": new_NX, "NY": TRAINED_NY,
        "n_sites_x": new_n_sites_x, "n_sites_y": ae_cfg.n_sites_y, "d_latent": d_latent,
        "nu": args.nu, "alpha": args.alpha, "forcing_n": args.forcing_n, "dt": args.dt,
        "spinup_time": args.spinup_time, "seed": args.seed,
        "model_lambda1": float(model_result.exponents[0]), "model_n_positive": model_result.n_positive,
        "model_D_KY": model_result.kaplan_yorke_dimension,
        "reused_true_D_KY": iso_result["true_D_KY"] if iso_result else None,
        "reused_isotropic_model_D_KY": iso_result["model_D_KY"] if iso_result else None,
        "relative_error_pct_vs_true": (
            abs(model_result.kaplan_yorke_dimension - iso_result["true_D_KY"]) / iso_result["true_D_KY"] * 100.0
            if iso_result else None
        ),
        "ae_ckpt": args.ae_ckpt, "prop_ckpt": args.prop_ckpt,
        "git_sha": git_sha, "model_wall_s": model_wall,
    }
    out_path = ARTIFACTS_DIR / f"g2b_kolmogorov_aniso_transfer{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[g2b] wrote {out_path}")


if __name__ == "__main__":
    main()
