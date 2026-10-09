"""G1 (`docs/ks_local_field_ltransfer.tex`, "L-transfer for the 2D
systems" plan): ground-truth Lyapunov spectrum of the TRUE 2D KS PDE at
the L-transfer size, computed directly by this project's own
already-validated `lyapunov_spectrum_ks2d` (Benettin QR, float64 CPU,
ETDRK4 variational integrator) -- not the learned model -- compared
against Section 230's checkpoint transferred to the SAME size with ZERO
retraining, via `load_autoencoder_checkpoint_resized_2d`/
`load_propagator_checkpoint_resized_2d`.

2D analogue of `exp_e1_groundtruth_lyapunov_ltransfer.py`; see that
script's own docstring for the general rationale. Differences specific
to 2D (see the tex plan's own G1 entry for why): transfer factor is
1.5x per axis (not 4x -- ground-truth cost scales with grid AREA in 2D,
a genuinely different compute regime from 1D's own dx-fixed length
scaling), and `n_directions` defaults from Section 230's own measured
D_KY/Area ratio (0.0444, at the Lx=Ly=32 test domain) rather than a
literature constant (no equivalent asymptotic-law literature constant
exists yet for 2D KS the way Edson et al. 2019 gives 1D).
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

from ks_latent.analysis.lyapunov import lyapunov_spectrum_ks2d, lyapunov_spectrum_latent_propagator
from ks_latent.config import KSConfig2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import spinup2d

ARTIFACTS_DIR = Path("artifacts/experiments")

AE_CKPT = "artifacts/stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt"
PROP_CKPT = "artifacts/stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt"
TRAINED_LX = TRAINED_LY = 44.0
TRAINED_NX = TRAINED_NY = 128
TRAINED_N_SITES = 8
D_KY_OVER_AREA_SMALL_DOMAIN = 71.9 / (44.0 * 44.0)  # Section 230's own measured value, at the TRAINED size
# (the Lx=Ly=32 test-domain ratio 0.0444 predates dealiasing/production tuning; the
# trained-size model ratio is the more directly comparable anchor for sizing n_directions here)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.5, help="Per-axis transfer factor.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n-directions", type=int, default=None, help="Default: ceil(1.6 * predicted D_KY).")
    parser.add_argument("--total-time", type=float, default=40.0)
    parser.add_argument("--warmup-time", type=float, default=20.0)
    parser.add_argument("--qr-interval", type=float, default=1.0)
    parser.add_argument("--spinup-time", type=float, default=200.0)
    parser.add_argument("--max-abs-state", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    new_Lx = TRAINED_LX * args.scale
    new_Ly = TRAINED_LY * args.scale
    new_NX = int(round(TRAINED_NX * args.scale))
    new_NY = int(round(TRAINED_NY * args.scale))
    new_n_sites = int(round(TRAINED_N_SITES * args.scale))
    assert new_NX % new_n_sites == 0, f"new_NX={new_NX} not divisible by new_n_sites={new_n_sites}"

    predicted_d_ky = D_KY_OVER_AREA_SMALL_DOMAIN * new_Lx * new_Ly
    n_directions = args.n_directions if args.n_directions is not None else math.ceil(1.6 * predicted_d_ky)

    print(f"[g1] scale={args.scale}  Lx=Ly={new_Lx}  NX=NY={new_NX}  n_sites={new_n_sites}  "
          f"dx={new_Lx/new_NX:.6f} (trained dx={TRAINED_LX/TRAINED_NX:.6f})")
    print(f"[g1] predicted D_KY (area-scaled from trained-size model ratio): {predicted_d_ky:.1f}")
    print(f"[g1] n_directions={n_directions}  total_time={args.total_time}  warmup_time={args.warmup_time}  "
          f"qr_interval={args.qr_interval}  spinup_time={args.spinup_time}  seed={args.seed}", flush=True)

    # --- (a) model's own zero-retraining transfer prediction (cheap) ---
    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        AE_CKPT, new_n_sites_x=new_n_sites, new_n_sites_y=new_n_sites, new_NX=new_NX, new_NY=new_NY,
    )
    prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d(
        PROP_CKPT, new_n_sites_x=new_n_sites, new_n_sites_y=new_n_sites,
    )
    ae.eval(); prop.eval()
    d_latent = ae_cfg.d_latent

    t0 = time.time()
    new_cfg_2d = KSConfig2D(Lx=new_Lx, Ly=new_Ly, NX=new_NX, NY=new_NY, dt=args.dt, spinup_time=args.spinup_time, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        u0_phys = spinup2d(new_cfg_2d, rng)
        u0_flat = torch.tensor(u0_phys.reshape(-1), dtype=torch.float32).unsqueeze(0)
        z0 = ae.encode(u0_flat)[0].numpy().astype(np.float64)

    model_result = lyapunov_spectrum_latent_propagator(
        prop, np.stack([z0, z0]), mode="single_state",
        n_directions=d_latent, n_steps=2000, qr_every=10,
        dt_snap=new_cfg_2d.dt_snap, warmup_steps=200, seed=args.seed, max_abs_state=1e3,
    )
    model_wall = time.time() - t0
    print(f"[g1] MODEL (zero-retraining transfer): lambda1={model_result.exponents[0]:.4f} "
          f"n_positive={model_result.n_positive}/{d_latent} D_KY={model_result.kaplan_yorke_dimension:.2f} "
          f"({model_wall:.1f}s)", flush=True)

    # --- (b) true PDE ground truth (expensive) ---
    t0 = time.time()
    true_result = lyapunov_spectrum_ks2d(
        new_cfg_2d,
        n_directions=n_directions,
        total_time=args.total_time,
        qr_interval=args.qr_interval,
        warmup_time=args.warmup_time,
        seed=args.seed,
        max_abs_state=args.max_abs_state,
    )
    true_wall = time.time() - t0
    print(f"[g1] TRUE PDE: lambda1={true_result.exponents[0]:.4f} "
          f"n_positive={true_result.n_positive}/{n_directions} D_KY={true_result.kaplan_yorke_dimension:.2f} "
          f"({true_wall:.1f}s)", flush=True)

    rel_err = abs(model_result.kaplan_yorke_dimension - true_result.kaplan_yorke_dimension) / true_result.kaplan_yorke_dimension * 100.0
    print(f"[g1] relative error, model vs TRUE PDE D_KY: {rel_err:.2f}%")

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_sha = None

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "scale": args.scale, "Lx": new_Lx, "Ly": new_Ly, "NX": new_NX, "NY": new_NY,
        "n_sites": new_n_sites, "d_latent": d_latent, "dt": args.dt,
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
    out_path = ARTIFACTS_DIR / f"g1_2dks_transfer{('_' + args.tag) if args.tag else ''}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[g1] wrote {out_path}")


if __name__ == "__main__":
    main()
