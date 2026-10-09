"""Generate a Kolmogorov-flow training dataset
(`docs/ks_local_field_ltransfer.tex` "Staged next steps" Step 2).

`--alpha` added 2026-09-29 (user-directed: "let's return to the model
from 231 and increase drag through alpha. maybe we'll see more of a
generalization"), so a stronger-drag dataset can be generated without
duplicating this script -- everything else (NX/NY/nu/forcing_n/
trajectory sizing) stays Section 231's own default unless overridden."""

import argparse

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.dataset2d import generate_trajectory_dataset_kolmogorov

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="artifacts/datasets/kolmogorov_trajectories_default.h5")
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--trajectory-time", type=float, default=40.0)
    args = parser.parse_args()

    # Sizing note, same discipline as 2D KS's own dataset (a first attempt
    # there produced a ~15.8GB file by reusing 1D's snapshot-count
    # convention unmodified -- caught and corrected, see
    # docs/ks_local_field_ltransfer.tex Step 1's own progress note):
    # NX=NY=128 gives the same per-snapshot element count as 2D KS.
    # trajectory_time=40.0 at this config's own dt_snap=0.2 gives 201
    # snapshots/trajectory (vs. 2D KS's 251 at dt_snap=0.1) -- ~3.6
    # Lyapunov times at the measured lambda1=0.0905 (a bit short of 2D
    # KS's own ~6, a deliberate tradeoff to keep the file near the same
    # ~1.5-2GB ballpark; this system's own lambda1 is smaller than KS's,
    # so matching Lyapunov-time coverage exactly would cost much more
    # file size for the same snapshot cadence).
    cfg = KolmogorovConfig(alpha=args.alpha)  # Lx=Ly=2*pi, NX=NY=128, nu=0.01 (Re=100), forcing_n=4
    path = generate_trajectory_dataset_kolmogorov(
        cfg, args.out,
        n_train=args.n_train, n_val=args.n_val, trajectory_time=args.trajectory_time,
    )
    print(f"wrote {path}")
