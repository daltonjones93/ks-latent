"""R1 (`docs/ks_local_field_ltransfer.tex`, "Revision: build the ladder
from scratch, progressively" -- Section "A frozen, scale-incremental
hierarchy"): generate a 2D KS trajectory dataset for one rung of the
from-scratch domain-size ladder ($L_0{=}16, L_1{=}32, L_2{=}64,
L_3{=}128, \ldots$, patch length fixed at 2.0 physical units throughout,
so every rung's architecture stays zero-retraining-resizable to the
next).

Usage: `python scripts/gen_ks2d_ladder_dataset.py --level 0` (L=16),
`--level 1` (L=32), etc. -- `Lx=Ly=16*2**level`, `NX=NY=64*2**level`
(holds dx=0.25 and patch length=2.0 fixed across every rung).
"""

from __future__ import annotations

import argparse

from ks_latent.config import KSConfig2D
from ks_latent.solver.dataset2d import generate_trajectory_dataset2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, help="Ladder rung k: L=16*2**k, NX=64*2**k.")
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--trajectory-time", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    L = 16.0 * (2 ** args.level)
    NX = 64 * (2 ** args.level)
    cfg = KSConfig2D(Lx=L, Ly=L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=args.seed)
    print(f"[r1] level={args.level}  L={L}  NX=NY={NX}  dx={L/NX}  n_train={args.n_train}  "
          f"n_val={args.n_val}  trajectory_time={args.trajectory_time}", flush=True)

    out_path = f"artifacts/datasets/ks2d_ladder_L{int(L)}.h5"
    path = generate_trajectory_dataset2d(
        cfg, out_path,
        n_train=args.n_train, n_val=args.n_val, trajectory_time=args.trajectory_time,
    )
    print(f"[r1] wrote {path}")
