#!/usr/bin/env python
"""Phase 2 entry point: measure the KS information-spreading velocity v_star
and the derived light-cone/stencil/localization bounds (brief §4, Gate 2).

    python scripts/run_spreading.py                  # full run (~50s on CPU)
    python scripts/run_spreading.py --profile smoke   # <60s at tiny dimensions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ks_latent.config import KSConfig
from ks_latent.analysis.spreading import (
    light_cone_half_width,
    measure_ks_spreading,
    required_stencil_half_width_sites,
)
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.profile == "smoke":
        cfg = KSConfig(L=22.0, NX=64, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=args.seed)
        n_trials, total_time, v_grid = 3, 2.0, np.linspace(-1.0, 1.0, 5)
    else:
        cfg = KSConfig(L=100.0, NX=1024, dt=0.05, snapshot_every=5, spinup_time=200.0, seed=args.seed)
        n_trials, total_time, v_grid = 150, 20.0, np.linspace(-2.5, 2.5, 26)

    t0 = time.time()
    result = measure_ks_spreading(
        cfg, n_trials=n_trials, total_time=total_time, v_grid=v_grid, seed=args.seed
    )
    wall_time = time.time() - t0

    strides = [1, 2, 5, 10, 20]
    lattice_h = {"P=16": 6.25, "P=32": 3.13, "P=64": 1.56}
    bounds = {
        "light_cone_half_width": {
            s: light_cone_half_width(result.v_star_mean, cfg.dt, s) for s in strides
        },
        "required_stencil_half_width_sites": {
            f"stride={s}_{name}": required_stencil_half_width_sites(
                result.v_star_mean, cfg.dt, s, h
            )
            for s in strides
            for name, h in lattice_h.items()
        },
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / f"spreading_{args.profile}.json"
    payload = {
        "v_star_mean": result.v_star_mean,
        "v_star_std": result.v_star_std,
        "front_velocity_mean": result.front_velocity_mean,
        "front_velocity_std": result.front_velocity_std,
        "v_grid": result.v_grid.tolist(),
        "lambda_mean": result.lambda_mean.tolist(),
        "lambda_std": result.lambda_std.tolist(),
        "n_trials": result.n_trials,
        "total_time": result.total_time,
        "bounds": bounds,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    write_provenance(out_path, config=cfg, seed=args.seed, device="cpu", wall_time_s=wall_time)

    print(f"v_star = {result.v_star_mean:.4f} +/- {result.v_star_std / np.sqrt(n_trials):.4f} (SEM)")
    print(f"front velocities: {result.front_velocity_mean}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
