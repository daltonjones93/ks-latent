"""True-PDE Kaplan-Yorke dimension of 2D KS on an L x L periodic box, one IC seed
(task A4, docs/full_paper_tasks.md). Same protocol as the existing truths in
`exp_g1_2dks_groundtruth_transfer.py` (dt 0.02, dx = 44/128, spin-up 200, warm-up 20,
averaging window 40, QR every 1.0, max_abs_state 200, IC from spinup2d(cfg, rng(seed)),
tangent rng seed+1), plus a preemptable checkpoint and running D_KY at `record_times`
(physical time after warm-up), like `run_truth_kolmogorov.py`.

Usage: python scripts/run_truth_ks2d.py SCALE SEED N_DIRECTIONS [RECORD_TIMES]
Writes artifacts/experiments/truth_ks2d/truth_ks2d_x{scale}_seed{seed}.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys
import time
from pathlib import Path

import numpy as np

from ks_latent.analysis.lyapunov import (
    _ks2d_step_with_tangent_physical_space,
    benettin,
    kaplan_yorke_dimension,
)
from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import spinup2d

scale, seed, ndir = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
record_times = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else [10.0, 20.0, 30.0, 40.0]
dt, total_time, warmup_time, qr_interval, spinup, max_abs = 0.02, 40.0, 20.0, 1.0, 200.0, 200.0

L = 44.0 * scale
N = int(round(128 * scale))
assert abs(L / N - 44.0 / 128) < 1e-12, "grid spacing must equal the training/truth spacing 44/128"
cfg = KSConfig2D(Lx=L, Ly=L, NX=N, NY=N, dt=dt, spinup_time=spinup, seed=seed)

out = Path("artifacts/experiments/truth_ks2d")
out.mkdir(parents=True, exist_ok=True)
tag = f"truth_ks2d_x{scale:g}_seed{seed}".replace(".", "p")

t0 = time.time()
u0 = spinup2d(cfg, np.random.default_rng(seed), M=32)
qr_every = max(1, int(round(qr_interval / dt)))
n_steps = (int(round(total_time / dt)) // qr_every) * qr_every
warmup_steps = (int(round(warmup_time / dt)) // qr_every) * qr_every
record_steps = [(int(round(t / dt)) // qr_every) * qr_every for t in record_times]
res = benettin(
    _ks2d_step_with_tangent_physical_space(cfg, 32), u0.reshape(-1),
    n_directions=ndir, n_steps=n_steps, qr_every=qr_every, dt_per_step=dt,
    warmup_steps=warmup_steps, rng=np.random.default_rng(seed + 1), max_abs_state=max_abs,
    checkpoint_path=str(out / f"{tag}.ckpt.npz"), checkpoint_every=4, record_steps=record_steps,
)
ex = res.exponents
dky = kaplan_yorke_dimension(ex)
running = None
if res.partial_exponents is not None:
    running = {str(t): kaplan_yorke_dimension(e) for t, e in sorted(res.partial_exponents.items())}
(out / f"{tag}.json").write_text(json.dumps({
    "system": "ks2d", "L": L, "NX": N, "scale": scale, "seed": seed, "n_directions": ndir,
    "dt": dt, "spinup_time": spinup, "warmup_time": warmup_time, "total_time": total_time,
    "qr_interval": qr_interval, "D_KY": dky, "D_KY_over_area": dky / L**2,
    "lambda1": float(ex[0]), "n_positive": int((ex > 0).sum()),
    "exponents": [float(x) for x in ex], "running_D_KY_by_time": running,
    "wall_s_this_session": time.time() - t0,
}, indent=2))
print("done", tag, dky, "n_pos", int((ex > 0).sum()), "of", ndir, flush=True)
if int((ex > 0).sum()) > 0.8 * ndir:
    print("WARNING: n_positive close to n_directions", flush=True)
