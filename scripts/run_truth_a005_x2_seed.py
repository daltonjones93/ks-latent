"""alpha=0.05, Lx=2x2pi true-PDE D_KY for one seed (2026-10-03): the single-seed 2x
truth (56.84, G2) is the basis of P2's '21% deficit'; the new 3x truth (75.2) suggests
2x is an outlier. Preemptable via Benettin checkpoint. Usage: python ... SEED"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, sys, time
from pathlib import Path
from ks_latent.analysis.lyapunov import kaplan_yorke_dimension, lyapunov_spectrum_kolmogorov
from ks_latent.config import KolmogorovConfig
seed = int(sys.argv[1])
out = Path("artifacts/experiments/crop"); out.mkdir(parents=True, exist_ok=True)
cfg = KolmogorovConfig(Lx=4 * math.pi, Ly=2 * math.pi, NX=256, NY=128, nu=0.01, alpha=0.05, forcing_n=4,
                       dt=0.01, spinup_time=100.0, seed=seed)
t0 = time.time()
res = lyapunov_spectrum_kolmogorov(cfg, n_directions=96, total_time=40.0, qr_interval=0.5, warmup_time=20.0,
                                   seed=seed, max_abs_state=500.0,
                                   checkpoint_path=str(out / f"truth_a005_x2_seed{seed}.ckpt.npz"),
                                   checkpoint_every_time=4.0)
ex = res.exponents
(out / f"truth_a005_x2_seed{seed}.json").write_text(json.dumps({
    "alpha": 0.05, "scale": 2, "seed": seed, "n_directions": 96, "D_KY": kaplan_yorke_dimension(ex),
    "lambda1": float(ex[0]), "n_positive": int((ex > 0).sum()), "exponents": [float(x) for x in ex],
    "wall_s_this_session": time.time() - t0}, indent=2))
print("done", seed, kaplan_yorke_dimension(ex))
