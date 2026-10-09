"""Generate the default 2D KS training dataset (`docs/ks_local_field_ltransfer.tex`
"Staged next steps" Step 1)."""

from ks_latent.config import KSConfig2D
from ks_latent.solver.dataset2d import generate_trajectory_dataset2d

if __name__ == "__main__":
    # Sizing note (corrected after a first attempt produced a ~15.8GB
    # file and was killed): a 2D snapshot at NX=NY=128 is 64x the raw
    # element count of 1D KS's own NX=256 field, so reusing 1D's own
    # trajectory_time=100/n_train=50 convention directly (which gave
    # 1001 snapshots/trajectory) would have produced ~15.8GB across the
    # raw+normalized arrays -- unnecessarily large to generate, load, or
    # train on for a first attempt. trajectory_time=25.0 (n_snapshots=
    # 126, ~6 Lyapunov times at lambda1~0.24) x n_train=24/n_val=6 (30
    # independent trajectories, still real diversity) keeps the total
    # dataset under ~1GB.
    cfg = KSConfig2D()  # Lx=Ly=44, NX=NY=128, dt=0.02, spinup_time=200
    path = generate_trajectory_dataset2d(
        cfg, "artifacts/datasets/ks2d_trajectories_default.h5",
        n_train=24, n_val=6, trajectory_time=25.0,
    )
    print(f"wrote {path}")
