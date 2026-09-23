#!/usr/bin/env python
"""Generate a 2D Rayleigh-Benard convection trajectory dataset.

User-directed 2026-09-23, following `docs/RESULTS.md`'s "Weather-
relevance context and candidate next system" recommendation: a genuinely
2D, non-periodic-in-one-axis chaotic testbed, bridging L96's 1D periodic
chain and a real convection-resolving model (CM1-class).

Quick start (small/fast, to check the pipeline works on your machine):

    python scripts/generate_rayleigh_benard_dataset.py --profile smoke

Full dataset at this project's default physical regime (Ra=3e4, Pr=0.7,
64x64, ~46x supercritical -- see `RayleighBenardConfig`'s docstring):

    python scripts/generate_rayleigh_benard_dataset.py \\
        --n-train 20 --n-val 5 --trajectory-time 5.0 \\
        --output artifacts/datasets/rayleigh_benard_ra3e4_pr0.7.h5

**Adjusting scale** (all plain CLI flags, nothing hardcoded -- ground
rule 4): `--n-train`/`--n-val` change trajectory COUNT; `--nx`/`--nz`
change resolution; `--trajectory-time` changes how much physical time
each trajectory covers (more snapshots = more RAM-light streaming writes,
not more peak memory -- see `ks_latent.solver.rayleigh_benard_dataset`'s
module docstring); `--output` changes where the `.h5` file is written.

**Performance note, measured directly on this machine**: at the default
Ra=3e4/Pr=0.7/64x64 regime, the CFL-limited timestep settles to
~2.7e-5 once the flow saturates into turbulence (a genuine physical
consequence of the diffusive-time nondimensionalization at this
Rayleigh number, not a solver inefficiency -- the free-fall velocity
scale in these units is `sqrt(Ra*Pr) ~ 145`, matching the measured
saturated `max|w| ~ 130` closely). With `--fft-workers 4` (this
project's default), integration runs at roughly 1,300+ steps/second on
an 18-core Apple Silicon machine -- budget accordingly for
`--trajectory-time`/`--n-train`/`--n-val`; a single trajectory of
`trajectory_time=5.0` plus the default `spinup_time=5.0` is on the order
of several minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ks_latent.config import RayleighBenardConfig
from ks_latent.solver.rayleigh_benard_dataset import generate_trajectory_dataset
from ks_latent.utils.io import write_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full",
                         help="'smoke' = full code path in under ~60s on CPU, tiny dims (ground rule 5).")
    parser.add_argument("--output", type=str, default=None,
                         help="Output .h5 path. Default: artifacts/datasets/rayleigh_benard_ra<Ra>_pr<Pr>.h5"
                              " ('smoke' profile writes to artifacts/datasets/rayleigh_benard_smoke.h5).")
    parser.add_argument("--n-train", type=int, default=20, help="Training trajectory count.")
    parser.add_argument("--n-val", type=int, default=5, help="Validation trajectory count.")
    parser.add_argument("--trajectory-time", type=float, default=5.0,
                         help="Physical time units of snapshots recorded per trajectory (after spinup).")
    parser.add_argument("--ra", type=float, default=None, help="Override RayleighBenardConfig.Ra (default 3e4).")
    parser.add_argument("--aspect-ratio", type=float, default=None,
                         help="Override RayleighBenardConfig.aspect_ratio (default 2*sqrt(2) ~= 2.83, chosen so "
                              "the box's fundamental wavenumber exactly matches the critical one -- see "
                              "RayleighBenardConfig's docstring). Found 2026-09-23: at the default aspect ratio, "
                              "the box only fits a single pair of convection rolls and Ra=3e4 settles into an "
                              "essentially EXACT steady state (no chaos/oscillation) -- there's no lateral room "
                              "for the pattern competition that drives 2D RBC spatiotemporal chaos at moderate "
                              "Ra. A wider box (more rolls) is the standard lever; try e.g. 4x-8x the default "
                              "for genuine time-dependent dynamics.")
    parser.add_argument("--pr", type=float, default=None, help="Override RayleighBenardConfig.Pr (default 0.7).")
    parser.add_argument("--nx", type=int, default=None, help="Override horizontal resolution (default 64).")
    parser.add_argument("--nz", type=int, default=None, help="Override vertical resolution (default 64).")
    parser.add_argument("--snapshot-dt", type=float, default=None,
                         help="Override physical time between saved snapshots (default 0.05).")
    parser.add_argument("--spinup-time", type=float, default=None,
                         help="Override physical time discarded before recording each trajectory (default 5.0).")
    parser.add_argument("--fft-workers", type=int, default=None,
                         help="Override scipy.fft(workers=) -- multi-core CPU use (default 4). Raise for more "
                              "speed on a machine with many idle cores; lower to leave more headroom for other "
                              "work running at the same time.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-velocity", action="store_true",
                         help="Also store u/w (derivable from omega, so not required for training) in a "
                              "separate 'velocity' dataset -- convenient for visualization/diagnostics, "
                              "roughly doubles file size.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args()

    default_aspect_ratio = RayleighBenardConfig().aspect_ratio

    if args.profile == "smoke":
        cfg = RayleighBenardConfig(
            Nx=args.nx or 8, Nz=args.nz or 8, Ra=args.ra or 3.0e4, Pr=args.pr or 0.7,
            aspect_ratio=args.aspect_ratio or default_aspect_ratio,
            spinup_time=args.spinup_time if args.spinup_time is not None else 0.1,
            snapshot_dt=args.snapshot_dt or 0.02,
            fft_workers=args.fft_workers or 2, seed=args.seed,
        )
        n_train, n_val, trajectory_time = 2, 1, 0.1
        default_output = "artifacts/datasets/rayleigh_benard_smoke.h5"
    else:
        cfg = RayleighBenardConfig(
            Nx=args.nx or 64, Nz=args.nz or 64, Ra=args.ra or 3.0e4, Pr=args.pr or 0.7,
            aspect_ratio=args.aspect_ratio or default_aspect_ratio,
            spinup_time=args.spinup_time if args.spinup_time is not None else 5.0,
            snapshot_dt=args.snapshot_dt or 0.05,
            fft_workers=args.fft_workers or 4, seed=args.seed,
        )
        n_train, n_val, trajectory_time = args.n_train, args.n_val, args.trajectory_time
        default_output = f"artifacts/datasets/rayleigh_benard_ra{cfg.Ra:.0g}_pr{cfg.Pr}.h5"

    output_path = Path(args.output) if args.output else Path(default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[generate_rayleigh_benard_dataset] Ra={cfg.Ra:.4g} Pr={cfg.Pr} Nx={cfg.Nx} Nz={cfg.Nz} "
        f"Lx={cfg.Lx:.4f} n_train={n_train} n_val={n_val} trajectory_time={trajectory_time} "
        f"spinup_time={cfg.spinup_time} snapshot_dt={cfg.snapshot_dt} fft_workers={cfg.fft_workers} "
        f"-> {output_path}",
        flush=True,
    )

    t0 = time.time()
    try:
        generate_trajectory_dataset(
            cfg,
            output_path,
            n_train=n_train,
            n_val=n_val,
            trajectory_time=trajectory_time,
            include_velocity=args.include_velocity,
            show_progress=not args.no_progress,
        )
    except RuntimeError as e:
        print(f"[generate_rayleigh_benard_dataset] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    wall_time = time.time() - t0

    write_provenance(output_path, config=cfg, seed=args.seed, device="cpu", wall_time_s=wall_time)
    print(f"[generate_rayleigh_benard_dataset] wrote {output_path} in {wall_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
