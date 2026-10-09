"""E1 (docs/ks_local_field_ltransfer.tex, Section "Experiments to complete
this into a submittable result"): ground-truth Lyapunov spectrum of the
TRUE Kuramoto-Sivashinsky PDE at the L-transfer sizes (L=200, L=400),
computed directly by this project's own already-validated Gate-1 machinery
(`ks_latent.analysis.lyapunov.lyapunov_spectrum_ks`, Benettin QR, float64
CPU, ETDRK4 variational integrator) -- not the learned model.

Why this matters: docs/RESULTS.md's Phase F2 L-transfer result
(D_KY/L = 0.2214 / 0.2231 / 0.2228 at L=100/200/400) is currently a
self-consistency check -- the transferred model's own D_KY/L agrees with
its own trained-size D_KY/L and with the literature's ASYMPTOTIC constant
(~0.226, Edson et al. 2019). It has never been checked against the TRUE
system's D_KY at L=200 or L=400 specifically, because the solver's own
Lyapunov gate (tests/replication/test_gate1_kaplan_yorke.py) was only ever
run at L=100 and L=22. This script runs that same gate-quality computation
at the transfer sizes, so the model's transferred D_KY can be compared
against genuine ground truth rather than only against itself.

Resolution: NX is chosen by default to hold dx FIXED at Gate 1's own
validated value (dx = 100/1024 = 0.09765625) -- the resolution the
Lyapunov gate was actually shown trustworthy at, not the coarser NX=256
the autoencoder training data (and the transferred model) uses. This is
deliberately independent of the model's own resolution: the point of this
script is an outside check, not another self-consistency comparison.

n_directions defaults to ceil(1.6 * 0.226 * L) -- the same
n_directions/D_KY ratio Gate 1 used at L=100 (35 directions for an
expected D_KY~22, ratio ~1.6), applied to the LITERATURE's extensivity
constant (not the model's own transferred D_KY estimate, to avoid biasing
the ground-truth measurement toward the number it's meant to check).

Cost note: this is expensive by construction (Gate-1-grade resolution and
a QR-reorthonormalized tangent-space integration, at 2x/4x the grid and
2x/4x-ish the tangent directions of the original L=100 gate). Defaults
below are deliberately lighter than Gate 1's own L=100 numbers
(total_time=2000, warmup=300) to fit a first pass in a tractable amount
of wall time; all are exposed on the CLI so a longer, more rigorous rerun
is a one-flag change once a first estimate is in hand.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from ks_latent.analysis.lyapunov import lyapunov_spectrum_ks
from ks_latent.config import KSConfig

ARTIFACTS_DIR = Path("artifacts/experiments")

# Model's own transferred D_KY at each L-transfer size (Phase F2,
# docs/RESULTS.md), reported alongside the ground truth for comparison --
# NOT used to choose this script's own n_directions (see module docstring).
MODEL_TRANSFERRED_D_KY = {100.0: 22.14, 200.0: 44.61, 400.0: 89.13}
LITERATURE_ALPHA = 0.226  # Edson et al. 2019 extensivity constant, D_KY ~ alpha*L - c
GATE1_DX = 100.0 / 1024.0  # the dx Gate 1 validated the Lyapunov pipeline at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=float, required=True, help="Domain length, e.g. 200.0 or 400.0.")
    parser.add_argument(
        "--nx", type=int, default=None,
        help="Grid points. Default: round(L/GATE1_DX) to hold dx fixed at Gate 1's own validated resolution.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-directions", type=int, default=None, help="Default: ceil(1.6*alpha*L), alpha=0.226.")
    parser.add_argument("--total-time", type=float, default=1200.0, help="Physical time units (lighter than Gate 1's 2000 at L=100).")
    parser.add_argument("--warmup-time", type=float, default=300.0)
    parser.add_argument("--qr-interval", type=float, default=1.0)
    parser.add_argument("--max-abs-state", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    nx = args.nx if args.nx is not None else int(round(args.L / GATE1_DX))
    n_directions = args.n_directions if args.n_directions is not None else math.ceil(1.6 * LITERATURE_ALPHA * args.L)
    dx = args.L / nx

    print(f"[e1] L={args.L}  NX={nx}  dx={dx:.6f} (Gate 1's own dx={GATE1_DX:.6f})")
    print(f"[e1] n_directions={n_directions}  total_time={args.total_time}  warmup_time={args.warmup_time}  "
          f"qr_interval={args.qr_interval}  seed={args.seed}")

    cfg = KSConfig(L=args.L, NX=nx, dt=args.dt, snapshot_every=5, spinup_time=500.0, seed=args.seed)

    t0 = time.time()
    result = lyapunov_spectrum_ks(
        cfg,
        n_directions=n_directions,
        total_time=args.total_time,
        qr_interval=args.qr_interval,
        warmup_time=args.warmup_time,
        seed=args.seed,
        max_abs_state=args.max_abs_state,
    )
    wall_time = time.time() - t0

    d_ky = result.kaplan_yorke_dimension
    d_ky_over_l = d_ky / args.L
    lambda1 = float(result.exponents[0])
    model_d_ky = MODEL_TRANSFERRED_D_KY.get(args.L)

    print(f"[e1] wall time: {wall_time:.1f}s")
    print(f"[e1] TRUE PDE:   lambda1={lambda1:.4f}  n_positive={result.n_positive}/{n_directions}  D_KY={d_ky:.4f}  D_KY/L={d_ky_over_l:.4f}")
    print(f"[e1] literature alpha (Edson et al. 2019): {LITERATURE_ALPHA}")
    if model_d_ky is not None:
        model_ratio = model_d_ky / args.L
        rel_err = abs(d_ky - model_d_ky) / d_ky * 100.0
        print(f"[e1] TRANSFERRED MODEL (docs/RESULTS.md Phase F2): D_KY={model_d_ky:.4f}  D_KY/L={model_ratio:.4f}")
        print(f"[e1] relative error, model vs. TRUE PDE: {rel_err:.2f}%")

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_sha = None

    report = {
        "L": args.L, "NX": nx, "dx": dx, "dt": args.dt,
        "n_directions": n_directions, "total_time": args.total_time,
        "warmup_time": args.warmup_time, "qr_interval": args.qr_interval,
        "seed": args.seed, "max_abs_state": args.max_abs_state,
        "lambda1": lambda1, "n_positive": result.n_positive,
        "exponents": [float(x) for x in result.exponents],
        "D_KY": d_ky, "D_KY_over_L": d_ky_over_l,
        "literature_alpha": LITERATURE_ALPHA,
        "model_transferred_D_KY": model_d_ky,
        "model_transferred_D_KY_over_L": (model_d_ky / args.L) if model_d_ky is not None else None,
        "wall_time_seconds": wall_time,
        "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = ARTIFACTS_DIR / f"e1_groundtruth_lyapunov_L{int(args.L)}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[e1] wrote {out_path}")


if __name__ == "__main__":
    main()
